#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Elo Rating Model
==========================================
Implementuje dynamický Elo rating systém pro NHL týmy.
Upraveno pro franchise-based databázové schéma s trénink/backtesting rozdělením.

Umístění: src/models/elo_rating_model.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, date
from typing import Dict, Tuple, List, Optional
import json

# Import centrálních komponent
from config.paths import PATHS
from config.settings import settings
from config.logging_config import get_logger
from src.utils.file_handlers import save_model, load_model, write_json

# Logger pro tento modul
logger = get_logger(__name__)


class EloRatingSystem:
    """
    Elo Rating System pro NHL tým predikce
    Aktualizováno pro franchise-based databázové schéma
    """
    
    def __init__(self, 
                 initial_rating: Optional[float] = None,
                 k_factor: Optional[float] = None,
                 home_advantage: Optional[float] = None,
                 season_regression: Optional[float] = None):
        """
        Inicializace Elo Rating Systému
        
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
        
        # Databázové připojení z centrálních settings
        self.engine = create_engine(settings.database.connection_string)
        
        # Tracking výkonu
        self.predictions = []
        self.results = []
        
        logger.info(f"Elo Rating System inicializován:")
        logger.info(f"  Initial rating: {self.initial_rating}")
        logger.info(f"  K-factor: {self.k_factor}")
        logger.info(f"  Home advantage: {self.home_advantage}")
        logger.info(f"  Season regression: {self.season_regression}")
    
    def expected_score(self, rating_a: float, rating_b: float, home_advantage: float = 0) -> float:
        """
        Vypočítá očekávané skóre pro tým A proti týmu B
        
        Args:
            rating_a: Elo rating týmu A
            rating_b: Elo rating týmu B
            home_advantage: Dodatečný rating pro domácí tým
            
        Returns:
            Očekávaná pravděpodobnost výhry týmu A (0-1)
        """
        adjusted_rating_a = rating_a + home_advantage
        rating_diff = adjusted_rating_a - rating_b
        
        # Standardní Elo formule
        expected = 1 / (1 + 10 ** (-rating_diff / 400))
        return expected
    
    def update_ratings(self, team_a_id: int, team_b_id: int, actual_score: float, 
                      home_advantage: float = 0, k_multiplier: float = 1.0) -> Tuple[float, float]:
        """
        Aktualizuje Elo ratings po zápase
        
        Args:
            team_a_id: ID domácího týmu
            team_b_id: ID hostujícího týmu
            actual_score: 1 pokud tým A vyhrál, 0 pokud tým B vyhrál, 0.5 pro OT/SO porážku
            home_advantage: Bonus pro domácí výhodu
            k_multiplier: Násobitel pro K-factor (pro playoff zápasy, atd.)
            
        Returns:
            Tuple (new_rating_a, new_rating_b)
        """
        # Získej aktuální ratings
        rating_a = self.team_ratings.get(team_a_id, self.initial_rating)
        rating_b = self.team_ratings.get(team_b_id, self.initial_rating)
        
        # Vypočítaj očekávané skóre
        expected_a = self.expected_score(rating_a, rating_b, home_advantage)
        expected_b = 1 - expected_a
        
        # Vypočítaj změny ratingu
        k_factor = self.k_factor * k_multiplier
        change_a = k_factor * (actual_score - expected_a)
        change_b = k_factor * ((1 - actual_score) - expected_b)
        
        # Aktualizuj ratings
        new_rating_a = rating_a + change_a
        new_rating_b = rating_b + change_b
        
        self.team_ratings[team_a_id] = new_rating_a
        self.team_ratings[team_b_id] = new_rating_b
        
        return new_rating_a, new_rating_b
    
    def game_result_to_score(self, home_score: int, away_score: int, 
                           overtime_shootout: str = '') -> Tuple[float, str]:
        """
        Převede výsledek zápasu na Elo skóre formát
        
        Args:
            home_score: Finální skóre domácího týmu
            away_score: Finální skóre hostujícího týmu
            overtime_shootout: 'OT', 'SO', nebo prázdný string
            
        Returns:
            Tuple (home_team_score, result_type)
            home_team_score: 1.0 výhra, 0.0 porážka, 0.6 OT/SO výhra, 0.4 OT/SO porážka
        """
        if home_score > away_score:
            if overtime_shootout in ['OT', 'SO']:
                return 0.6, f'HOME_WIN_{overtime_shootout}'  # OT/SO výhra méně cenná
            else:
                return 1.0, 'HOME_WIN_REG'  # Regulérní výhra
        elif away_score > home_score:
            if overtime_shootout in ['OT', 'SO']:
                return 0.4, f'AWAY_WIN_{overtime_shootout}'  # OT/SO porážka dostane body
            else:
                return 0.0, 'AWAY_WIN_REG'  # Regulérní porážka
        else:
            return 0.5, 'TIE'  # Nemělo by se stát v moderní NHL
    
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
        query = f"""
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
            AND g.season >= '{season_start}'
            AND g.season <= '{season_end}'  -- KRITICKÉ: Excluded 2024/25 from training!
            AND g.home_score IS NOT NULL 
            AND g.away_score IS NOT NULL
        ORDER BY g.date, g.id
        """
        
        df = pd.read_sql(query, self.engine)
        
        logger.info(f"Načteno {len(df)} dokončených zápasů pro TRÉNOVÁNÍ z {season_start} do {season_end}")
        logger.info(f"DŮLEŽITÉ: Sezóna 2024/25 vyloučena - rezervována pro backtesting!")
        return df
    
    def train_on_historical_data(self, games_df: pd.DataFrame, 
                            evaluate_predictions: bool = True) -> Dict:
        """
        Trénuje Elo ratings na historických datech zápasů
        KRITICKÉ: Používá pouze data do sezóny 2023/24!
        Data z 2024/25 jsou rezervována pro backtesting.
        """
        # VALIDATION: Kontrola, že žádná data z 2024/25 neunikla do tréninku
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
        logger.info(f"POTVRZENO: Sezóna 2024/25 vyloučena z tréninku")
        
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
        
        # Vypočítej evaluation metrics
        metrics = {}
        if evaluate_predictions and predictions:
            metrics = self._calculate_metrics(predictions, actuals)
            logger.info(f"Trénování dokončeno. Přesnost: {metrics.get('accuracy', 0):.3f}")
        
        # Získej finální team ratings
        team_ratings_df = self.get_current_ratings()
        
        return {
            'metrics': metrics,
            'team_ratings': team_ratings_df,
            'games_processed': len(games_df),
            'rating_history': self.rating_history[-10:]  # Posledních 10 pro kontrolu
        }
    
    def _apply_season_regression(self):
        """Aplikuje regresi k průměru mezi sezónami"""
        mean_rating = np.mean(list(self.team_ratings.values()))
        
        for team_id in self.team_ratings:
            current_rating = self.team_ratings[team_id]
            regressed_rating = current_rating + self.season_regression * (mean_rating - current_rating)
            self.team_ratings[team_id] = regressed_rating
    
    def predict_game(self, home_team_id: int, away_team_id: int) -> Dict:
        """
        Předpovídá výsledek jednoho zápasu
        
        Args:
            home_team_id: ID domácího týmu
            away_team_id: ID hostujícího týmu
            
        Returns:
            Dictionary s detaily predikce
        """
        home_rating = self.team_ratings.get(home_team_id, self.initial_rating)
        away_rating = self.team_ratings.get(away_team_id, self.initial_rating)
        
        home_win_prob = self.expected_score(home_rating, away_rating, self.home_advantage)
        away_win_prob = 1 - home_win_prob
        
        # Získej jména týmů
        team_names = self._get_team_names([home_team_id, away_team_id])
        
        return {
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
    
    def predict_upcoming_games(self, days_ahead: int = 7) -> List[Dict]:
        """
        Předpovídá výsledky nadcházejících zápasů
        Aktualizováno pro franchise-based schéma
        
        Args:
            days_ahead: Počet dní dopředu pro predikce
            
        Returns:
            Seznam prediction dictionary
        """
        query = f"""
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
            AND g.date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '{days_ahead} days'
        ORDER BY g.date, g.datetime_et
        """
        
        upcoming_games = pd.read_sql(query, self.engine)
        
        predictions = []
        for _, game in upcoming_games.iterrows():
            prediction = self.predict_game(game['home_team_id'], game['away_team_id'])
            prediction['game_id'] = game['id']
            prediction['game_date'] = game['date']
            prediction['game_datetime'] = game['datetime_et']
            prediction['home_franchise'] = game['home_franchise']
            prediction['away_franchise'] = game['away_franchise']
            predictions.append(prediction)
        
        logger.info(f"Vygenerovány predikce pro {len(predictions)} nadcházejících zápasů")
        return predictions
    
    def get_current_ratings(self) -> pd.DataFrame:
        """
        Získá aktuální team ratings jako DataFrame
        """
        if not self.team_ratings:
            return pd.DataFrame()
        
        logger.debug("Získávání aktuálních ratingů...")
        
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
        
        return df
    
    def _get_team_names(self, team_ids: List[int]) -> Dict[int, str]:
        """
        Získá jména týmů pro dané team ID
        """
        logger.debug(f"_get_team_names voláno s {len(team_ids)} ID")
        
        if not team_ids:
            return {}
        
        # Validní team IDs
        valid_team_ids = [int(tid) for tid in team_ids 
                  if isinstance(tid, (int, np.integer))]
        if not valid_team_ids:
            logger.warning("Nenalezena žádná validní integer team ID")
            return {}
        
        logger.debug(f"Validní team ID: {valid_team_ids[:5]}...")
        
        # Jednoduchý dotaz - získej aktuální týmy
        if len(valid_team_ids) == 1:
            query = f"""
            SELECT t.id, t.name, f.franchise_name
            FROM teams t
            JOIN franchises f ON t.franchise_id = f.id
            WHERE t.id = {valid_team_ids[0]} AND t.is_current = TRUE
            """
        else:
            team_ids_str = ','.join(map(str, valid_team_ids))
            query = f"""
            SELECT t.id, t.name, f.franchise_name
            FROM teams t
            JOIN franchises f ON t.franchise_id = f.id
            WHERE t.id IN ({team_ids_str}) AND t.is_current = TRUE
            """
        
        try:
            logger.debug(f"Provádění SQL dotazu...")
            df = pd.read_sql(query, self.engine)
            logger.debug(f"Dotaz vrátil {len(df)} řádků")
            
            if df.empty:
                logger.warning("Dotaz nevrátil žádné výsledky - týmy možná nejsou aktuální")
                # Zkus bez is_current filtru
                query_fallback = query.replace(" AND t.is_current = TRUE", "")
                df = pd.read_sql(query_fallback, self.engine)
                logger.debug(f"Fallback dotaz vrátil {len(df)} řádků")
            
            result = dict(zip(df['id'], df['name']))
            logger.debug(f"Úspěšně zmapováno {len(result)} jmen týmů")
            return result
            
        except Exception as e:
            logger.error(f"SQL chyba v _get_team_names: {e}")
            logger.error(f"Dotaz byl: {query}")
            # Vrať fallback
            return {int(tid): f'Team_{tid}' for tid in valid_team_ids}
    
    def _calculate_metrics(self, predictions: List[float], actuals: List[int]) -> Dict:
        """Vypočítá metriky predikce"""
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
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
        
        return {
            'accuracy': accuracy,
            'brier_score': brier_score,
            'log_loss': log_loss,
            'total_predictions': len(predictions)
        }

    def load_backtesting_games(self, season: str = '2025') -> pd.DataFrame:
        """
        Načte zápasy pro backtesting (sezóna 2024/25)
        KRITICKÉ: Tato data NESMÍ být použita pro tréninng!
        
        Args:
            season: Sezóna pro backtesting (default '2025' = sezóna 2024/25)
            
        Returns:
            DataFrame se zápasy pro backtesting (včetně naplánovaných zápasů)
        """
        query = f"""
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
        
        WHERE g.season = '{season}'
            -- Zahrnout dokončené I naplánované zápasy pro backtesting
        ORDER BY g.date, g.datetime_et, g.id
        """
        
        df = pd.read_sql(query, self.engine)
        
        completed_games = len(df[df['status'] == 'completed'])
        scheduled_games = len(df[df['status'] == 'scheduled'])
        
        logger.info(f"Načteno {len(df)} zápasů pro BACKTESTING ze sezóny {season}")
        logger.info(f"  - Dokončené: {completed_games} zápasů")
        logger.info(f"  - Naplánované: {scheduled_games} zápasů")
        logger.warning(f"PŘIPOMÍNKA: Tyto zápasy nebyly použity pro trénování modelu!")
        
        return df

    def get_data_split_summary(self) -> Dict:
        """
        Poskytne přehled rozdělení dat pro trénování vs backtesting
        """
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
        
        training_df = pd.read_sql(training_query, self.engine)
        backtesting_df = pd.read_sql(backtesting_query, self.engine)
        
        combined_df = pd.concat([training_df, backtesting_df], ignore_index=True)
        
        summary = {
            'training_seasons': training_df['season'].tolist(),
            'training_games': int(training_df['game_count'].sum()),
            'backtesting_season': '2025',
            'backtesting_games': int(backtesting_df['game_count'].sum()) if not backtesting_df.empty else 0,
            'data_split': combined_df.to_dict('records')
        }
        
        return summary
    
    def save_model(self, filepath: Optional[str] = None):
        """Uloží natrénovaný model s verzí schématu"""
        if filepath is None:
            filepath = PATHS.trained_models / 'elo_model.pkl'
        
        # Vytvoř adresář pokud neexistuje
        PATHS.trained_models.mkdir(parents=True, exist_ok=True)
        
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
            'schema_version': '2.0',  # NOVÉ: označuje franchise-based schéma
            'franchise_support': True  # NOVÉ: podporuje franchise tracking
        }
        
        save_model(model_data, filepath)
        logger.info(f"Model uložen do {filepath} (schéma v2.0)")
        
    def load_model(self, filepath: Optional[str] = None):
        """Načte natrénovaný model s kontrolou kompatibility schématu"""
        if filepath is None:
            filepath = PATHS.trained_models / 'elo_model.pkl'
        
        model_data = load_model(filepath)
        
        # Kontrola verze schématu
        schema_version = model_data.get('schema_version', '1.0')
        if schema_version == '1.0':
            logger.warning("Načítání starého schématu modelu - zvažte migraci")
            # Zde by byla migrace pokud potřeba
        
        self.team_ratings = model_data['team_ratings']
        self.rating_history = model_data['rating_history']
        
        # Načti parametry
        params = model_data['parameters']
        self.initial_rating = params['initial_rating']
        self.k_factor = params['k_factor'] 
        self.home_advantage = params['home_advantage']
        self.season_regression = params['season_regression']
        
        logger.info(f"Model načten z {filepath} (schéma v{schema_version})")

    def debug_team_mapping(self):
        """Debug metoda pro sledování mapování týmů"""
        logger.info("🔍 DEBUGGING TEAM MAPPING:")
        
        # 1. Zkontroluj jaké team IDs máme v self.team_ratings
        logger.info(f"Team IDs in self.team_ratings: {list(self.team_ratings.keys())[:10]}...")
        
        # 2. Zkontroluj co je v databázi
        try:
            query = """
            SELECT t.id, t.name, t.is_current, f.franchise_name
            FROM teams t
            JOIN franchises f ON t.franchise_id = f.id
            WHERE t.id IN (1,2,3,4,5,9,13,19,24)
            ORDER BY t.id
            """
            df = pd.read_sql(query, self.engine)
            logger.info("UKÁZKOVÉ TÝMY Z DATABÁZE:")
            for _, row in df.iterrows():
                current_flag = "✓" if row['is_current'] else "✗"
                logger.info(f"  ID {row['id']:2d}: {row['name']} ({row['franchise_name']}) {current_flag}")
        except Exception as e:
            logger.error(f"Chyba dotazování týmů: {e}")
        
        # 3. Test _get_team_names() metodu přímo
        test_ids = [1, 9, 24]  # Top teams from log
        logger.info(f"Testování _get_team_names() s ID: {test_ids}")
        try:
            result = self._get_team_names(test_ids)
            logger.info(f"Výsledek: {result}")
        except Exception as e:
            logger.error(f"_get_team_names() selhala: {e}")


def main():
    """
    Hlavní funkce pro trénování Elo modelu
    KRITICKÉ: Trénuje pouze na datech do 2023/24, 2024/25 je pro backtesting
    """
    
    # Vytvoř potřebné adresáře
    PATHS.ensure_directories()
    
    logger.info("🏒 Spuštění trénování Elo Rating System...")
    
    try:
        # Inicializuj Elo systém s parametry ze settings
        elo = EloRatingSystem()  # Parametry se načtou automaticky ze settings
        
        # Získej souhrn rozdělení dat nejprve
        data_summary = elo.get_data_split_summary()
        logger.info("\n📊 SOUHRN ROZDĚLENÍ DAT:")
        logger.info(f"  Trénovací sezóny: {data_summary['training_seasons']}")
        logger.info(f"  Trénovací zápasy: {data_summary['training_games']}")
        logger.info(f"  Backtesting sezóna: {data_summary['backtesting_season']}")
        logger.info(f"  Backtesting zápasy: {data_summary['backtesting_games']}")
        
        # Načti TRÉNOVACÍ data (sezóny 2022-2024, vyjímaje 2024/25)
        logger.info("\n📚 Načítání TRÉNOVACÍCH dat (do 2023/24)...")
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
        logger.info("\n🎯 Trénování Elo ratingů...")
        results = elo.train_on_historical_data(games_df, evaluate_predictions=True)
        
        # Zobraz výsledky
        logger.info("\n🎯 VÝSLEDKY TRÉNINKU:")
        metrics = results['metrics']
        logger.info(f"  Přesnost: {metrics.get('accuracy', 0):.3f}")
        logger.info(f"  Brier Score: {metrics.get('brier_score', 0):.3f}")
        logger.info(f"  Log Loss: {metrics.get('log_loss', 0):.3f}")
        logger.info(f"  Zpracované zápasy: {results['games_processed']}")
        
        # Debug mapování týmů
        logger.info("\n🔍 DEBUGGING JMEN TÝMŮ:")
        elo.debug_team_mapping()

        # Ukaž aktuální žebříček týmů
        logger.info("\n🏆 TOP 10 TEAM RATINGŮ:")
        ratings_df = results['team_ratings']
        for _, team in ratings_df.head(10).iterrows():
            logger.info(f"  {team['rating_rank']:2d}. {team['team_name']:<25} {team['elo_rating']:7.1f}")
        
        # Arizona → Utah přechod kontrola
        utah_teams = ratings_df[ratings_df['team_name'].str.contains('Utah', na=False)]
        if not utah_teams.empty:
            logger.info(f"\n🦣 UTAH PŘECHOD OVĚŘEN:")
            for _, team in utah_teams.iterrows():
                logger.info(f"  {team['team_name']}: {team['elo_rating']:.1f} (pořadí {team['rating_rank']})")
        
        # Ulož model
        model_filepath = PATHS.trained_models / 'elo_model_trained_2024.pkl'
        elo.save_model(model_filepath)
        
        # Ulož také team ratings do JSON pro další použití
        ratings_json_path = PATHS.trained_models / 'team_ratings_current.json'
        ratings_dict = ratings_df.set_index('team_id')['elo_rating'].to_dict()
        write_json(ratings_dict, ratings_json_path)
        logger.info(f"Team ratings uloženy do {ratings_json_path}")
        
        # Ukaž dostupnost backtesting dat
        logger.info("\n📊 DOSTUPNÁ BACKTESTING DATA:")
        backtesting_df = elo.load_backtesting_games('2025')
        completed_backtest = len(backtesting_df[backtesting_df['status'] == 'completed'])
        logger.info(f"  Celkem zápasů: {len(backtesting_df)}")
        logger.info(f"  Dokončené (připravené pro backtest): {completed_backtest}")
        logger.info(f"  Naplánované (budoucí predikce): {len(backtesting_df) - completed_backtest}")
        
        logger.info("\n🎉 Trénování modelu úspěšně dokončeno!")
        logger.info("📋 DALŠÍ KROKY:")
        logger.info("  1. Spusť backtesting na datech sezóny 2024/25")
        logger.info("  2. Validuj výkon modelu na out-of-sample datech")
        logger.info("  3. Pokud úspěšné, pokračuj k implementaci live tradingu")
        
    except Exception as e:
        logger.error(f"❌ Trénování Elo modelu selhalo: {e}")
        raise


if __name__ == "__main__":
    main()
