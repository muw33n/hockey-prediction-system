#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Elo Rating Model (REFACTORED)
=========================================================
Implementuje dynamický Elo rating systém pro NHL týmy.
Upraveno pro franchise-based databázové schéma s trénink/backtesting rozdělením.

REFACTORING: SRP-based refactoring
- DatabaseConnectionManager moved to src/database/connection.py
- Pure Elo functions delegated to src/models/elo_calculations.py

Umístění: src/models/elo_rating_model.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import text
from datetime import datetime, date
from typing import Dict, Tuple, List, Optional

# === Enhanced infrastructure imports ===
from config.paths import PATHS
from config.settings import settings
from config.logging_config import (
    setup_logging, get_component_logger, PerformanceLogger,
    LoggingConfig
)
from src.utils.file_handlers import (
    save_model_safe, load_model_safe, write_json, read_json,
    save_processed_data
)

# === REFACTORING: Import shared DatabaseConnectionManager ===
from src.database.connection import DatabaseConnectionManager

# === REFACTORING: Import pure Elo calculation functions ===
from src.models.elo_calculations import (
    expected_score as _expected_score,
    game_result_to_score as _game_result_to_score,
    calculate_rating_update,
    apply_season_regression
)

# === Component-specific logger pro models ===
logger = get_component_logger(__name__, 'models')


class EloRatingSystem:
    """
    Enhanced Elo Rating System pro NHL tým predikce
    MIGRACE: Aktualizováno s enhanced infrastructure
    """
    
    def __init__(self, 
                 initial_rating: Optional[float] = None,
                 k_factor: Optional[float] = None,
                 home_advantage: Optional[float] = None,
                 season_regression: Optional[float] = None):
        """
        Inicializace Enhanced Elo Rating Systému
        
        Args:
            initial_rating: Výchozí Elo rating pro všechny týmy (z settings pokud None)
            k_factor: Learning rate - vyšší = volatilnější (z settings pokud None)
            home_advantage: Bonus pro domácí tým (z settings pokud None)
            season_regression: Regrese ratingů mezi sezónami 0-1 (z settings pokud None)
        """
        # Načti parametry ze settings nebo použij zadané
        self.initial_rating = initial_rating or settings.model.elo_initial_rating
        self.k_factor = k_factor or settings.model.elo_k_factor
        self.home_advantage = home_advantage or settings.model.elo_home_advantage
        self.season_regression = season_regression or settings.model.elo_season_regression
        
        # Úložiště team ratingů
        self.team_ratings = {}  # {team_id: current_rating}
        self.rating_history = []  # Historické ratings pro analýzu
        
        # === MIGRACE: Enhanced database connection manager ===
        self.db_manager = DatabaseConnectionManager(
            settings.database.connection_string,
            max_retries=3,
            retry_delay=1.0
        )
        
        # === MIGRACE: Performance monitoring ===
        self.perf_logger = PerformanceLogger(logger)
        
        # Tracking výkonu
        self.predictions = []
        self.results = []
        
        logger.info("Enhanced Elo Rating System inicializován:")
        logger.info(f"  Initial rating: {self.initial_rating}")
        logger.info(f"  K-factor: {self.k_factor}")
        logger.info(f"  Home advantage: {self.home_advantage}")
        logger.info(f"  Season regression: {self.season_regression}")
    
    def expected_score(self, rating_a: float, rating_b: float, home_advantage: float = 0) -> float:
        """
        Vypočítá očekávané skóre pro tým A proti týmu B.
        Delegates to pure function in elo_calculations.py.
        """
        return _expected_score(rating_a, rating_b, home_advantage)
    
    def update_ratings(self, team_a_id: int, team_b_id: int, actual_score: float,
                      home_advantage: float = 0, k_multiplier: float = 1.0) -> Tuple[float, float]:
        """
        Aktualizuje Elo ratings po zápase.
        Delegates calculation to pure function in elo_calculations.py.
        """
        # Získej aktuální ratings
        rating_a = self.team_ratings.get(team_a_id, self.initial_rating)
        rating_b = self.team_ratings.get(team_b_id, self.initial_rating)

        # Použij pure funkci pro výpočet
        new_rating_a, new_rating_b = calculate_rating_update(
            rating_a, rating_b, actual_score, self.k_factor,
            home_advantage, k_multiplier
        )

        # Aktualizuj ratings
        self.team_ratings[team_a_id] = new_rating_a
        self.team_ratings[team_b_id] = new_rating_b

        return new_rating_a, new_rating_b
    
    def game_result_to_score(self, home_score: int, away_score: int,
                           overtime_shootout: str = '') -> Tuple[float, str]:
        """
        Převede výsledek zápasu na Elo skóre formát.
        Delegates to pure function in elo_calculations.py.
        """
        return _game_result_to_score(home_score, away_score, overtime_shootout)
    
    def load_historical_games(self, season_start: str = '2022', season_end: str = '2024') -> pd.DataFrame:
        """
        Načte historické zápasy z databáze pro trénování
        KRITICKÉ: Načítá pouze data do sezóny 2023/24 (včetně)
        Data z 2024/25 jsou rezervována pro backtesting!
        
        Args:
            season_start: První sezóna k zahrnutí (např. '2022')
            season_end: Poslední sezóna pro trénování (default '2024' = sezóna 2023/24)
            
        Returns:
            DataFrame s výsledky zápasů pro trénování
        """
        query = text("""
        SELECT
            g.id,
            g.date,
            g.datetime_et,
            g.season,
            g.home_team_id,
            g.away_team_id,
            g.home_score,
            g.away_score,
            g.overtime_shootout,

            -- Domácí tým s franchise info
            ht.name as home_team_name,
            hf.franchise_name as home_franchise_name,

            -- Hostující tým s franchise info
            at.name as away_team_name,
            af.franchise_name as away_franchise_name

        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN franchises hf ON ht.franchise_id = hf.id
        JOIN teams at ON g.away_team_id = at.id
        JOIN franchises af ON at.franchise_id = af.id

        WHERE g.status = 'completed'
            AND g.season >= :season_start
            AND g.season <= :season_end
            AND g.home_score IS NOT NULL
            AND g.away_score IS NOT NULL
        ORDER BY g.date, g.id
        """)

        try:
            self.perf_logger.start_timer('load_historical_games')

            df = self.db_manager.execute_query_safe(
                query,
                f"Loading historical games ({season_start}-{season_end})",
                params={"season_start": season_start, "season_end": season_end}
            )
            
            self.perf_logger.end_timer('load_historical_games')
            
            logger.info(f"Načteno {len(df)} dokončených zápasů pro TRÉNOVÁNÍ z {season_start} do {season_end}")
            logger.info("DŮLEŽITÉ: Sezóna 2024/25 vyloučena - rezervována pro backtesting!")
            
            return df
            
        except Exception as e:
            logger.error("Failed to load historical games data")
            LoggingConfig.log_exception(logger, e, "load_historical_games")
            raise
    
    def train_on_historical_data(self, games_df: pd.DataFrame, 
                            evaluate_predictions: bool = True) -> Dict:
        """
        Trénuje Elo ratings na historických datech zápasů
        KRITICKÉ: Používá pouze data do sezóny 2023/24!
        Data z 2024/25 jsou rezervována pro backtesting.
        """
        # === MIGRACE: Enhanced validation s better error messages ===
        logger.info("Trénování Elo ratings na historických datech...")
        if not games_df.empty:
            max_season = games_df['season'].max()
            
            # Convert to int for comparison (handle both string and int season formats)
            try:
                max_season_int = int(max_season)
                if max_season_int > 2024:
                    raise ValueError(f"KRITICKÉ: Trénovací data obsahují sezónu {max_season}!")
            except (ValueError, TypeError):
                if str(max_season) > '2024':
                    raise ValueError(f"KRITICKÉ: Trénovací data obsahují sezónu {max_season}!")
        
        logger.info("Trénování Elo ratings na historických datech s franchise podporou...")
        logger.info(f"Trénovací data: sezóny {games_df['season'].min()} do {games_df['season'].max()}")
        logger.info("POTVRZENO: Sezóna 2024/25 vyloučena z tréninku")
        
        # === MIGRACE: Performance monitoring pro training ===
        self.perf_logger.start_timer('elo_training')
        
        try:
            # Debug sample team IDs from data
            sample_teams = set(list(games_df['home_team_id'].unique())[:5])
            logger.debug(f"Ukázkové team ID z dat zápasů: {sample_teams}")

            # Inicializuj všechny týmy se základním ratingem - ZJEDNODUŠENO
            unique_teams = set(games_df['home_team_id'].unique()) | set(games_df['away_team_id'].unique())
            for team_id in unique_teams:
                self.team_ratings[team_id] = self.initial_rating
            
            logger.info(f"Inicializováno {len(unique_teams)} týmů s ratingem {self.initial_rating}")
            
            # Track for evaluation
            predictions = []
            actuals = []
            current_season = None
            games_processed = 0
            
            # === MIGRACE: Progress monitoring ===
            total_games = len(games_df)
            progress_interval = max(100, total_games // 20)  # Show progress every 5%
            
            # Zpracuj zápasy chronologicky
            for idx, game in games_df.iterrows():
                home_team_id = game['home_team_id']
                away_team_id = game['away_team_id']
                
                # Aplikuj sezónní regresi pokud nová sezóna
                if current_season != game['season']:
                    if current_season is not None:
                        self._apply_season_regression()
                        logger.info(f"Aplikována sezónní regrese pro sezónu {game['season']}")
                    current_season = game['season']
                
                # Udělej predikci před aktualizací ratingů
                if evaluate_predictions:
                    home_rating = self.team_ratings.get(home_team_id, self.initial_rating)
                    away_rating = self.team_ratings.get(away_team_id, self.initial_rating)
                    predicted_prob = self.expected_score(home_rating, away_rating, self.home_advantage)
                    predictions.append(predicted_prob)
                
                # Získej skutečný výsledek
                home_score, result_type = self.game_result_to_score(
                    game['home_score'], 
                    game['away_score'],
                    game['overtime_shootout']
                )
                
                if evaluate_predictions:
                    actuals.append(home_score if home_score in [0.0, 1.0] else int(home_score > 0.5))
                
                # Aktualizuj ratings
                old_home_rating = self.team_ratings.get(home_team_id, self.initial_rating)
                old_away_rating = self.team_ratings.get(away_team_id, self.initial_rating)
                
                new_home_rating, new_away_rating = self.update_ratings(
                    home_team_id, away_team_id, home_score, self.home_advantage
                )
                
                # Ulož rating history
                self.rating_history.append({
                    'game_id': game['id'],
                    'date': game['date'],
                    'season': game['season'],
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'home_rating_before': old_home_rating,
                    'away_rating_before': old_away_rating,
                    'home_rating_after': new_home_rating,
                    'away_rating_after': new_away_rating,
                    'actual_result': home_score,
                    'result_type': result_type
                })
                
                games_processed += 1
                
                # === MIGRACE: Progress logging ===
                if games_processed % progress_interval == 0:
                    progress = (games_processed / total_games) * 100
                    logger.info(f"Training progress: {games_processed}/{total_games} games ({progress:.1f}%)")
            
            self.perf_logger.end_timer('elo_training')
            
            # Vypočítej evaluation metrics
            metrics = {}
            if evaluate_predictions and predictions:
                self.perf_logger.start_timer('metrics_calculation')
                metrics = self._calculate_metrics(predictions, actuals)
                self.perf_logger.end_timer('metrics_calculation')
                logger.info(f"Trénování dokončeno. Přesnost: {metrics.get('accuracy', 0):.3f}")
            
            # Získej finální team ratings
            team_ratings_df = self.get_current_ratings()
            
            return {
                'metrics': metrics,
                'team_ratings': team_ratings_df,
                'games_processed': games_processed,
                'rating_history': self.rating_history[-10:]  # Posledních 10 pro kontrolu
            }
            
        except Exception as e:
            self.perf_logger.end_timer('elo_training')  # Ensure timer is stopped
            logger.error("Training failed")
            LoggingConfig.log_exception(logger, e, "train_on_historical_data")
            raise
    
    def _apply_season_regression(self):
        """
        Aplikuje regresi k průměru mezi sezónami.
        Delegates to pure function in elo_calculations.py.
        """
        if not self.team_ratings:
            logger.warning("No team ratings to apply regression to")
            return

        # Použij pure funkci pro výpočet regrese
        self.team_ratings = apply_season_regression(self.team_ratings, self.season_regression)
        logger.debug(f"Applied season regression to {len(self.team_ratings)} teams")
    
    def predict_game(self, home_team_id: int, away_team_id: int) -> Dict:
        """
        Předpovídá výsledek jednoho zápasu
        
        Args:
            home_team_id: ID domácího týmu
            away_team_id: ID hostujícího týmu
            
        Returns:
            Dictionary s detaily predikce
        """
        try:
            home_rating = self.team_ratings.get(home_team_id, self.initial_rating)
            away_rating = self.team_ratings.get(away_team_id, self.initial_rating)
            
            home_win_prob = self.expected_score(home_rating, away_rating, self.home_advantage)
            away_win_prob = 1 - home_win_prob
            
            # Získej jména týmů
            team_names = self._get_team_names([home_team_id, away_team_id])
            
            prediction = {
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_team_name': team_names.get(home_team_id, f'Team_{home_team_id}'),
                'away_team_name': team_names.get(away_team_id, f'Team_{away_team_id}'),
                'home_rating': home_rating,
                'away_rating': away_rating, 
                'home_win_probability': home_win_prob,
                'away_win_probability': away_win_prob,
                'predicted_winner': 'HOME' if home_win_prob > 0.5 else 'AWAY',
                'confidence': abs(home_win_prob - 0.5) * 2,  # 0-1 škála
                'rating_difference': home_rating - away_rating + self.home_advantage
            }
            
            logger.debug(f"Game prediction: {prediction['home_team_name']} vs {prediction['away_team_name']} -> {prediction['predicted_winner']} ({prediction['confidence']:.2f})")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Failed to predict game {home_team_id} vs {away_team_id}")
            LoggingConfig.log_exception(logger, e, "predict_game")
            raise
    
    def predict_upcoming_games(self, days_ahead: int = 7) -> List[Dict]:
        """
        Předpovídá výsledky nadcházejících zápasů
        Aktualizováno pro franchise-based schéma
        
        Args:
            days_ahead: Počet dní dopředu pro predikce
            
        Returns:
            Seznam prediction dictionary
        """
        query = text("""
        SELECT
            g.id,
            g.date,
            g.datetime_et,
            g.home_team_id,
            g.away_team_id,

            -- Aktuální jména týmů (is_current = TRUE)
            ht.name as home_team_name,
            at.name as away_team_name,

            -- Franchise info pro kontext
            hf.franchise_name as home_franchise,
            af.franchise_name as away_franchise

        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id AND ht.is_current = TRUE
        JOIN franchises hf ON ht.franchise_id = hf.id
        JOIN teams at ON g.away_team_id = at.id AND at.is_current = TRUE
        JOIN franchises af ON at.franchise_id = af.id

        WHERE g.status = 'scheduled'
            AND g.date BETWEEN CURRENT_DATE AND CURRENT_DATE + CAST(:days_ahead || ' days' AS INTERVAL)
        ORDER BY g.date, g.datetime_et
        """)

        try:
            self.perf_logger.start_timer('predict_upcoming_games')

            upcoming_games = self.db_manager.execute_query_safe(
                query,
                f"Loading upcoming games ({days_ahead} days ahead)",
                params={"days_ahead": str(int(days_ahead))}
            )
            
            predictions = []
            for _, game in upcoming_games.iterrows():
                try:
                    prediction = self.predict_game(game['home_team_id'], game['away_team_id'])
                    prediction['game_id'] = game['id']
                    prediction['game_date'] = game['date']
                    prediction['game_datetime'] = game['datetime_et']
                    prediction['home_franchise'] = game['home_franchise']
                    prediction['away_franchise'] = game['away_franchise']
                    predictions.append(prediction)
                except Exception as e:
                    logger.warning(f"Failed to predict game {game['id']}: {e}")
                    continue
            
            self.perf_logger.end_timer('predict_upcoming_games')
            
            logger.info(f"Vygenerovány predikce pro {len(predictions)} nadcházejících zápasů")
            return predictions
            
        except Exception as e:
            logger.error("Failed to predict upcoming games")
            LoggingConfig.log_exception(logger, e, "predict_upcoming_games")
            raise
    
    def get_current_ratings(self) -> pd.DataFrame:
        """
        Získá aktuální team ratings jako DataFrame
        """
        if not self.team_ratings:
            logger.warning("No team ratings available")
            return pd.DataFrame()
        
        logger.debug("Získávání aktuálních ratingů...")
        
        try:
            # Získej pouze validní team IDs
            team_ids = [tid for tid in self.team_ratings.keys() 
                if isinstance(tid, (int, np.integer)) and not isinstance(tid, str)]
            logger.debug(f"Získávání jmen pro {len(team_ids)} týmů...")
            
            team_names = self._get_team_names(team_ids)
            logger.debug(f"Získáno {len(team_names)} jmen týmů")
            
            ratings_data = []
            for team_id, rating in self.team_ratings.items():
                # Přeskoč historické ratings
                if isinstance(team_id, str) and 'historical' in str(team_id):
                    continue
                    
                team_name = team_names.get(team_id, f'Team_{team_id}')
                ratings_data.append({
                    'team_id': team_id,
                    'team_name': team_name,
                    'elo_rating': rating,
                    'rating_rank': 0
                })
            
            df = pd.DataFrame(ratings_data)
            if not df.empty:
                df = df.sort_values('elo_rating', ascending=False).reset_index(drop=True)
                df['rating_rank'] = range(1, len(df) + 1)
            
            logger.debug(f"Successfully created ratings DataFrame with {len(df)} teams")
            return df
            
        except Exception as e:
            logger.error("Failed to get current ratings")
            LoggingConfig.log_exception(logger, e, "get_current_ratings")
            return pd.DataFrame()
    
    def _get_team_names(self, team_ids: List[int]) -> Dict[int, str]:
        """
        Získá jména týmů pro dané team ID s enhanced error handling
        """
        logger.debug(f"_get_team_names voláno s {len(team_ids)} ID")
        
        if not team_ids:
            return {}
        
        try:
            # Validní team IDs
            valid_team_ids = [int(tid) for tid in team_ids 
                      if isinstance(tid, (int, np.integer))]
            if not valid_team_ids:
                logger.warning("Nenalezena žádná validní integer team ID")
                return {}
            
            logger.debug(f"Validní team ID: {valid_team_ids[:5]}...")

            # Parametrizovaný dotaz s dynamickými placeholdery pro IN clause
            placeholders = ', '.join(f':id_{i}' for i in range(len(valid_team_ids)))
            params = {f'id_{i}': tid for i, tid in enumerate(valid_team_ids)}

            query = text(f"""
                SELECT t.id, t.name, f.franchise_name
                FROM teams t
                JOIN franchises f ON t.franchise_id = f.id
                WHERE t.id IN ({placeholders}) AND t.is_current = TRUE
            """)

            logger.debug("Provádění SQL dotazu...")
            df = self.db_manager.execute_query_safe(
                query,
                f"Getting team names for {len(valid_team_ids)} teams",
                params=params
            )
            logger.debug(f"Dotaz vrátil {len(df)} řádků")

            if df.empty:
                logger.warning("Dotaz nevrátil žádné výsledky - týmy možná nejsou aktuální")
                # Zkus bez is_current filtru
                query_fallback = text(f"""
                    SELECT t.id, t.name, f.franchise_name
                    FROM teams t
                    JOIN franchises f ON t.franchise_id = f.id
                    WHERE t.id IN ({placeholders})
                """)
                df = self.db_manager.execute_query_safe(
                    query_fallback,
                    "Getting team names (fallback without is_current filter)",
                    params=params
                )
                logger.debug(f"Fallback dotaz vrátil {len(df)} řádků")
            
            result = dict(zip(df['id'], df['name']))
            logger.debug(f"Úspěšně zmapováno {len(result)} jmen týmů")
            return result
            
        except Exception as e:
            logger.error("Failed to get team names")
            LoggingConfig.log_exception(logger, e, "_get_team_names")
            # Vrať fallback
            return {int(tid): f'Team_{tid}' for tid in team_ids if isinstance(tid, (int, np.integer))}
    
    def _calculate_metrics(self, predictions: List[float], actuals: List[int]) -> Dict:
        """Vypočítá metriky predikce s enhanced error handling"""
        try:
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            if len(predictions) != len(actuals):
                logger.warning(f"Mismatch in predictions ({len(predictions)}) vs actuals ({len(actuals)})")
                return {}
            
            # Převeď pravděpodobnosti na binární predikce
            binary_predictions = (predictions > 0.5).astype(int)
            
            # Přesnost
            accuracy = np.mean(binary_predictions == actuals)
            
            # Brier Score (nižší je lepší)
            brier_score = np.mean((predictions - actuals) ** 2)
            
            # Log Loss 
            epsilon = 1e-15  # Zabraň log(0)
            predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
            log_loss = -np.mean(actuals * np.log(predictions_clipped) + 
                               (1 - actuals) * np.log(1 - predictions_clipped))
            
            metrics = {
                'accuracy': accuracy,
                'brier_score': brier_score,
                'log_loss': log_loss,
                'total_predictions': len(predictions)
            }
            
            logger.debug(f"Calculated metrics: accuracy={accuracy:.3f}, brier_score={brier_score:.3f}")
            return metrics
            
        except Exception as e:
            logger.error("Failed to calculate metrics")
            LoggingConfig.log_exception(logger, e, "_calculate_metrics")
            return {}

    def load_backtesting_games(self, season: str = '2025') -> pd.DataFrame:
        """
        Načte zápasy pro backtesting (sezóna 2024/25)
        KRITICKÉ: Tato data NESMÍ být použita pro trénning!
        
        Args:
            season: Sezóna pro backtesting (default '2025' = sezóna 2024/25)
            
        Returns:
            DataFrame se zápasy pro backtesting (včetně naplánovaných zápasů)
        """
        query = text("""
        SELECT
            g.id,
            g.date,
            g.datetime_et,
            g.season,
            g.home_team_id,
            g.away_team_id,
            g.home_score,
            g.away_score,
            g.overtime_shootout,
            g.status,

            -- Aktuální jména týmů (z franchise perspektivy)
            ht.name as home_team_name,
            hf.franchise_name as home_franchise_name,

            -- Hostující tým s franchise info
            at.name as away_team_name,
            af.franchise_name as away_franchise_name

        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN franchises hf ON ht.franchise_id = hf.id
        JOIN teams at ON g.away_team_id = at.id
        JOIN franchises af ON at.franchise_id = af.id

        WHERE g.season = :season
            -- Zahrnout dokončené I naplánované zápasy pro backtesting
        ORDER BY g.date, g.datetime_et, g.id
        """)

        try:
            self.perf_logger.start_timer('load_backtesting_games')

            df = self.db_manager.execute_query_safe(
                query,
                f"Loading backtesting games (season {season})",
                params={"season": season}
            )
            
            self.perf_logger.end_timer('load_backtesting_games')
            
            completed_games = len(df[df['status'] == 'completed'])
            scheduled_games = len(df[df['status'] == 'scheduled'])
            
            logger.info(f"Načteno {len(df)} zápasů pro BACKTESTING ze sezóny {season}")
            logger.info(f"  - Dokončené: {completed_games} zápasů")
            logger.info(f"  - Naplánované: {scheduled_games} zápasů")
            logger.warning("PŘIPOMÍNKA: Tyto zápasy nebyly použity pro trénování modelu!")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load backtesting games for season {season}")
            LoggingConfig.log_exception(logger, e, "load_backtesting_games")
            raise

    def get_data_split_summary(self) -> Dict:
        """
        Poskytne přehled rozdělení dat pro trénování vs backtesting
        """
        try:
            self.perf_logger.start_timer('data_split_summary')
            
            training_query = """
            SELECT season, COUNT(*) as game_count, 'TRAINING' as dataset_type
            FROM games 
            WHERE status = 'completed' AND season <= '2024'
            GROUP BY season
            """
            
            backtesting_query = """
            SELECT season, COUNT(*) as game_count, 'BACKTESTING' as dataset_type
            FROM games 
            WHERE season = '2025'
            GROUP BY season
            """
            
            training_df = self.db_manager.execute_query_safe(
                training_query, "Getting training data summary"
            )
            backtesting_df = self.db_manager.execute_query_safe(
                backtesting_query, "Getting backtesting data summary"
            )
            
            combined_df = pd.concat([training_df, backtesting_df], ignore_index=True)
            
            summary = {
                'training_seasons': training_df['season'].tolist(),
                'training_games': int(training_df['game_count'].sum()),
                'backtesting_season': '2025',
                'backtesting_games': int(backtesting_df['game_count'].sum()) if not backtesting_df.empty else 0,
                'data_split': combined_df.to_dict('records')
            }
            
            self.perf_logger.end_timer('data_split_summary')
            
            return summary
            
        except Exception as e:
            logger.error("Failed to get data split summary")
            LoggingConfig.log_exception(logger, e, "get_data_split_summary")
            return {
                'training_seasons': [],
                'training_games': 0,
                'backtesting_season': '2025',
                'backtesting_games': 0,
                'data_split': []
            }
    
    def save_model(self, filepath: Optional[str] = None):
        """Uloží natrénovaný model s verzí schématu - ENHANCED"""
        try:
            if filepath is None:
                # === MIGRACE: Použij PATHS pro default filepath ===
                filepath = PATHS.get_model_file('elo_model_trained_2024', 'pkl')
            
            # Vytvoř adresář pokud neexistuje
            PATHS.trained_models.mkdir(parents=True, exist_ok=True)
            
            self.perf_logger.start_timer('save_model')
            
            model_data = {
                'team_ratings': self.team_ratings,
                'parameters': {
                    'initial_rating': self.initial_rating,
                    'k_factor': self.k_factor,
                    'home_advantage': self.home_advantage,
                    'season_regression': self.season_regression
                },
                'rating_history': self.rating_history,
                'trained_date': datetime.now().isoformat(),
                'schema_version': '2.1',  # ENHANCED: označuje enhanced infrastructure
                'franchise_support': True,
                'enhanced_features': {  # NOVÉ: enhanced infrastructure metadata
                    'per_component_logging': True,
                    'performance_monitoring': True,
                    'safe_file_handling': True,
                    'database_resilience': True
                }
            }
            
            # === MIGRACE: Safe model saving ===
            save_model_safe(model_data, 'elo_model_trained_2024', 'pkl')
            
            self.perf_logger.end_timer('save_model')
            
            logger.info(f"Enhanced model uložen do {filepath} (schéma v2.1)")
            
        except Exception as e:
            logger.error("Failed to save model")
            LoggingConfig.log_exception(logger, e, "save_model")
            raise
        
    def load_model(self, filepath: Optional[str] = None):
        """Načte natrénovaný model s kontrolou kompatibility schématu - ENHANCED"""
        try:
            if filepath is None:
                filepath = PATHS.get_model_file('elo_model_trained_2024', 'pkl')
            
            self.perf_logger.start_timer('load_model')
            
            # === MIGRACE: Safe model loading ===
            model_data = load_model_safe('elo_model_trained_2024', 'pkl')
            
            # Kontrola verze schématu
            schema_version = model_data.get('schema_version', '1.0')
            if schema_version == '1.0':
                logger.warning("Načítání starého schématu modelu - zvažte migraci")
            elif schema_version.startswith('2.1'):
                logger.info("Načítání enhanced modelu s plnou infrastructure podporou")
            
            self.team_ratings = model_data['team_ratings']
            self.rating_history = model_data['rating_history']
            
            # Načti parametry
            params = model_data['parameters']
            self.initial_rating = params['initial_rating']
            self.k_factor = params['k_factor'] 
            self.home_advantage = params['home_advantage']
            self.season_regression = params['season_regression']
            
            self.perf_logger.end_timer('load_model')
            
            logger.info(f"Enhanced model načten z {filepath} (schéma v{schema_version})")
            
        except Exception as e:
            logger.error("Failed to load model")
            LoggingConfig.log_exception(logger, e, "load_model")
            raise

    def debug_team_mapping(self):
        """Debug metoda pro sledování mapování týmů - ENHANCED"""
        logger.info("🔍 DEBUGGING TEAM MAPPING (ENHANCED):")
        
        try:
            # 1. Zkontroluj jaké team IDs máme v self.team_ratings
            team_ids_sample = list(self.team_ratings.keys())[:10]
            logger.info(f"Team IDs in self.team_ratings: {team_ids_sample}...")
            
            # 2. Zkontroluj co je v databázi
            query = """
            SELECT t.id, t.name, t.is_current, f.franchise_name
            FROM teams t
            JOIN franchises f ON t.franchise_id = f.id
            WHERE t.id IN (1,2,3,4,5,9,13,19,24)
            ORDER BY t.id
            """
            
            df = self.db_manager.execute_query_safe(
                query, "Debug team mapping query"
            )
            
            logger.info("UKÁZKOVÉ TÝMY Z DATABÁZE:")
            for _, row in df.iterrows():
                current_flag = "✓" if row['is_current'] else "✗"
                logger.info(f"  ID {row['id']:2d}: {row['name']} ({row['franchise_name']}) {current_flag}")
        
            # 3. Test _get_team_names() metodu přímo
            test_ids = [1, 9, 24]  # Top teams from log
            logger.info(f"Testování _get_team_names() s ID: {test_ids}")
            result = self._get_team_names(test_ids)
            logger.info(f"Výsledek: {result}")
            
        except Exception as e:
            logger.error("Debug team mapping failed")
            LoggingConfig.log_exception(logger, e, "debug_team_mapping")


def main():
    """
    Hlavní funkce pro trénování Enhanced Elo modelu
    KRITICKÉ: Trénuje pouze na datech do 2023/24, 2024/25 je pro backtesting
    """
    
    # === MIGRACE: Setup enhanced logging na začátku ===
    setup_logging(
        log_level='INFO',
        log_to_file=True,
        log_to_console=True,
        component_files=True  # Per-component log files
    )
    
    # Vytvoř potřebné adresáře
    PATHS.ensure_directories()
    
    logger.info("🏒 Spuštění Enhanced Elo Rating System trénování...")
    
    # === MIGRACE: Performance monitoring pro celý main ===
    main_perf = PerformanceLogger(logger)
    main_perf.start_timer('complete_training_process')
    
    try:
        # Inicializuj Enhanced Elo systém s parametry ze settings
        logger.info("Inicializace Enhanced Elo Rating System...")
        elo = EloRatingSystem()  # Parametry se načtou automaticky ze settings
        
        # Získej souhrn rozdělení dat nejprve
        data_summary = elo.get_data_split_summary()
        logger.info("📊 SOUHRN ROZDĚLENÍ DAT:")
        logger.info(f"  Trénovací sezóny: {data_summary['training_seasons']}")
        logger.info(f"  Trénovací zápasy: {data_summary['training_games']}")
        logger.info(f"  Backtesting sezóna: {data_summary['backtesting_season']}")
        logger.info(f"  Backtesting zápasy: {data_summary['backtesting_games']}")
        
        # Načti TRÉNOVACÍ data (sezóny 2022-2024, vyjímaje 2024/25)
        logger.info("📚 Načítání TRÉNOVACÍCH dat (do 2023/24)...")
        games_df = elo.load_historical_games(season_start='2022', season_end='2024')
        
        if games_df.empty:
            logger.error("Nenalezeny žádné trénovací zápasy. Ujistěte se, že data jsou importována.")
            return
        
        # Validuj rozdělení dat
        max_season = games_df['season'].max()
        try:
            max_season_int = int(max_season)
            if max_season_int > 2024:
                logger.error(f"KRITICKÁ CHYBA: Trénovací data obsahují sezónu {max_season}!")
                logger.error("Sezóna 2024/25 musí být rezervována pro backtesting!")
                return
        except (ValueError, TypeError):
            # Handle string season format
            if str(max_season) > '2024':
                logger.error(f"KRITICKÁ CHYBA: Trénovací data obsahují sezónu {max_season}!")
                logger.error("Sezóna 2024/25 musí být rezervována pro backtesting!")
                return
        
        logger.info(f"✅ Trénovací data validována: {len(games_df)} zápasů ze sezón {games_df['season'].min()}-{max_season}")
        
        # Trénuj model
        logger.info("🎯 Trénování Enhanced Elo ratingů...")
        results = elo.train_on_historical_data(games_df, evaluate_predictions=True)
        
        # Zobraz výsledky
        logger.info("🎯 VÝSLEDKY TRÉNINKU:")
        metrics = results['metrics']
        logger.info(f"  Přesnost: {metrics.get('accuracy', 0):.3f}")
        logger.info(f"  Brier Score: {metrics.get('brier_score', 0):.3f}")
        logger.info(f"  Log Loss: {metrics.get('log_loss', 0):.3f}")
        logger.info(f"  Zpracované zápasy: {results['games_processed']}")
        
        # Debug mapování týmů
        logger.info("🔍 DEBUGGING JMEN TÝMŮ:")
        elo.debug_team_mapping()

        # Ukaž aktuální žebříček týmů
        logger.info("🏆 TOP 10 ENHANCED TEAM RATINGŮ:")
        ratings_df = results['team_ratings']
        for _, team in ratings_df.head(10).iterrows():
            logger.info(f"  {team['rating_rank']:2d}. {team['team_name']:<25} {team['elo_rating']:7.1f}")
        
        # Arizona → Utah přechod kontrola
        utah_teams = ratings_df[ratings_df['team_name'].str.contains('Utah', na=False)]
        if not utah_teams.empty:
            logger.info("🦣 UTAH PŘECHOD OVĚŘEN:")
            for _, team in utah_teams.iterrows():
                logger.info(f"  {team['team_name']}: {team['elo_rating']:.1f} (pořadí {team['rating_rank']})")
        
        # === MIGRACE: Enhanced model saving ===
        logger.info("💾 Ukládání Enhanced modelu...")
        model_filepath = PATHS.trained_models / 'elo_model_enhanced_trained_2024.pkl'
        elo.save_model(model_filepath)
        
        # Ulož také team ratings do JSON pro další použití
        ratings_json_path = PATHS.trained_models / 'team_ratings_enhanced_current.json'
        ratings_dict = ratings_df.set_index('team_id')['elo_rating'].to_dict()
        write_json(ratings_dict, ratings_json_path)
        logger.info(f"Enhanced team ratings uloženy do {ratings_json_path}")
        
        # === MIGRACE: Enhanced training summary export ===
        training_summary = {
            'model_version': '2.1_enhanced',
            'trained_date': datetime.now().isoformat(),
            'training_data': {
                'seasons': f"{games_df['season'].min()}-{games_df['season'].max()}",
                'games_count': len(games_df),
                'teams_count': len(set(games_df['home_team_id']) | set(games_df['away_team_id']))
            },
            'model_parameters': {
                'initial_rating': elo.initial_rating,
                'k_factor': elo.k_factor,
                'home_advantage': elo.home_advantage,
                'season_regression': elo.season_regression
            },
            'performance_metrics': metrics,
            'enhanced_features': [
                'per_component_logging',
                'performance_monitoring', 
                'database_resilience',
                'safe_file_handling',
                'enhanced_error_handling'
            ]
        }
        
        summary_path = save_processed_data(
            pd.DataFrame([training_summary]), 
            'elo_training_summary_enhanced',
            index=False
        )
        logger.info(f"Enhanced training summary uloženo do {summary_path}")
        
        # Ukaž dostupnost backtesting dat
        logger.info("📊 DOSTUPNÁ BACKTESTING DATA:")
        backtesting_df = elo.load_backtesting_games('2025')
        completed_backtest = len(backtesting_df[backtesting_df['status'] == 'completed'])
        logger.info(f"  Celkem zápasů: {len(backtesting_df)}")
        logger.info(f"  Dokončené (připravené pro backtest): {completed_backtest}")
        logger.info(f"  Naplánované (budoucí predikce): {len(backtesting_df) - completed_backtest}")
        
        main_perf.end_timer('complete_training_process')
        
        logger.info("🎉 Enhanced trénování modelu úspěšně dokončeno!")
        logger.info("📋 DALŠÍ KROKY:")
        logger.info("  1. Spusť backtesting na datech sezóny 2024/25")
        logger.info("  2. Validuj výkon modelu na out-of-sample datech")
        logger.info("  3. Pokud úspěšné, pokračuj k implementaci live tradingu")
        logger.info("  4. Monitor performance v per-component log files:")
        logger.info(f"     - Models: {PATHS.logs}/models.log")
        logger.info(f"     - Database ops: {PATHS.logs}/database.log")
        logger.info(f"     - Main system: {PATHS.logs}/hockey_system.log")
        
    except Exception as e:
        main_perf.end_timer('complete_training_process')
        logger.error("❌ Enhanced trénování Elo modelu selhalo")
        LoggingConfig.log_exception(logger, e, "main")
        raise


if __name__ == "__main__":
    main()
