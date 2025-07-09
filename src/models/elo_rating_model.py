#!/usr/bin/env python3
"""
Elo Rating System for NHL Hockey Predictions
Implements dynamic team ratings that update after each game
Updated for franchise-based database schema with training/backtesting split 
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import logging
from datetime import datetime, date
from typing import Dict, Tuple, List, Optional
import json
import pickle

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/elo_model.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EloRatingSystem:
    """
    Elo Rating System for NHL team predictions
        Updated for franchise-based database schema
    """
    
    def __init__(self, 
                 initial_rating: float = 1500.0,
                 k_factor: float = 32.0,
                 home_advantage: float = 100.0,
                 season_regression: float = 0.25):
        """
        Initialize Elo Rating System
        
        Args:
            initial_rating: Starting Elo rating for all teams
            k_factor: Learning rate (higher = more volatile)
            home_advantage: Home team rating bonus
            season_regression: How much ratings regress to mean between seasons (0-1)
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.season_regression = season_regression
        
        # Team ratings storage
        self.team_ratings = {}  # {team_id: current_rating}
        self.rating_history = []  # Historical ratings for analysis
        
        # Database connection
        self.engine = create_engine(os.getenv('DATABASE_URL'))
        
        # Performance tracking
        self.predictions = []
        self.results = []
        
    def expected_score(self, rating_a: float, rating_b: float, home_advantage: float = 0) -> float:
        """
        Calculate expected score for team A against team B
        
        Args:
            rating_a: Team A's Elo rating
            rating_b: Team B's Elo rating  
            home_advantage: Additional rating for home team
            
        Returns:
            Expected probability that team A wins (0-1)
        """
        adjusted_rating_a = rating_a + home_advantage
        rating_diff = adjusted_rating_a - rating_b
        
        # Standard Elo formula
        expected = 1 / (1 + 10 ** (-rating_diff / 400))
        return expected
    
    def update_ratings(self, team_a_id: int, team_b_id: int, actual_score: float, 
                      home_advantage: float = 0, k_multiplier: float = 1.0) -> Tuple[float, float]:
        """
        Update Elo ratings after a game
        
        Args:
            team_a_id: Home team ID 
            team_b_id: Away team ID
            actual_score: 1 if team A won, 0 if team B won, 0.5 for OT/SO loss
            home_advantage: Home advantage rating bonus
            k_multiplier: Multiplier for K-factor (for playoff games, etc.)
            
        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        # Get current ratings
        rating_a = self.team_ratings.get(team_a_id, self.initial_rating)
        rating_b = self.team_ratings.get(team_b_id, self.initial_rating)
        
        # Calculate expected scores
        expected_a = self.expected_score(rating_a, rating_b, home_advantage)
        expected_b = 1 - expected_a
        
        # Calculate rating changes
        k_factor = self.k_factor * k_multiplier
        change_a = k_factor * (actual_score - expected_a)
        change_b = k_factor * ((1 - actual_score) - expected_b) 
        
        # Update ratings
        new_rating_a = rating_a + change_a
        new_rating_b = rating_b + change_b
        
        self.team_ratings[team_a_id] = new_rating_a
        self.team_ratings[team_b_id] = new_rating_b
        
        return new_rating_a, new_rating_b
    
    def game_result_to_score(self, home_score: int, away_score: int, 
                           overtime_shootout: str = '') -> Tuple[float, str]:
        """
        Convert game result to Elo score format
        
        Args:
            home_score: Home team final score
            away_score: Away team final score  
            overtime_shootout: 'OT', 'SO', or empty string
            
        Returns:
            Tuple of (home_team_score, result_type)
            home_team_score: 1.0 for win, 0.0 for loss, 0.6 for OT/SO win, 0.4 for OT/SO loss
        """
        if home_score > away_score:
            if overtime_shootout in ['OT', 'SO']:
                return 0.6, f'HOME_WIN_{overtime_shootout}'  # OT/SO win worth less
            else:
                return 1.0, 'HOME_WIN_REG'  # Regulation win
        elif away_score > home_score:
            if overtime_shootout in ['OT', 'SO']:
                return 0.4, f'AWAY_WIN_{overtime_shootout}'  # OT/SO loss gets some points
            else:
                return 0.0, 'AWAY_WIN_REG'  # Regulation loss
        else:
            return 0.5, 'TIE'  # Shouldn't happen in modern NHL
    
    # === NAHRADIT EXISTUJÍCÍ METODU ===
    def load_historical_games(self, season_start: str = '2022', season_end: str = '2024') -> pd.DataFrame:
        """
        Load historical games from database for training
        KRITICKÉ: Načítá pouze data do sezóny 2023/24 (včetně)
        Data z 2024/25 jsou rezervována pro backtesting!
        
        Args:
            season_start: First season to include (e.g., '2022')
            season_end: Last season to include for training (default '2024' = season 2023/24)
            
        Returns:
            DataFrame with game results for training
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
            
            -- Home team with franchise info
            ht.name as home_team_name,
            hf.franchise_name as home_franchise_name,
            
            -- Away team with franchise info  
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
        
        logger.info(f"Loaded {len(df)} completed games for TRAINING from {season_start} to {season_end}")
        logger.info(f"IMPORTANT: Season 2024/25 excluded - reserved for backtesting!")
        return df
    
    # === UPRAVIT EXISTUJÍCÍ METODU ===
    # V metodě train_on_historical_data(), PŘIDAT na začátek metody:
    # OPRAVA: Odstraň zbytečnou team validation nebo ji zjednodušte (mapování týmů)
    def train_on_historical_data(self, games_df: pd.DataFrame, 
                            evaluate_predictions: bool = True) -> Dict:
        """
        Train Elo ratings on historical game data
        KRITICKÉ: Používá pouze data do sezóny 2023/24!
        Data z 2024/25 jsou rezervována pro backtesting.
        """
        # VALIDATION: Check that no 2024/25 data leaked into training
        logger.info("Training Elo ratings on historical data...")
        if not games_df.empty:
            max_season = games_df['season'].max()
            
            # Convert to int for comparison (handle both string and int season formats)
            try:
                max_season_int = int(max_season)
                if max_season_int > 2024:
                    raise ValueError(f"CRITICAL: Training data contains season {max_season}!")
            except (ValueError, TypeError):
                if str(max_season) > '2024':
                    raise ValueError(f"CRITICAL: Training data contains season {max_season}!")
        
        logger.info("Training Elo ratings on historical data with franchise support...")
        logger.info(f"Training data: seasons {games_df['season'].min()} to {games_df['season'].max()}")
        logger.info(f"CONFIRMED: Season 2024/25 excluded from training")
        
        # Initialize all teams with base rating - SIMPLIFIED (odstraň validation)
        unique_teams = set(games_df['home_team_id'].unique()) | set(games_df['away_team_id'].unique())
        for team_id in unique_teams:
            self.team_ratings[team_id] = self.initial_rating
        
        logger.info(f"Initialized {len(unique_teams)} teams with rating {self.initial_rating}")
        
        # === ZBYTEK METODY ZŮSTÁVÁ BEZE ZMĚN ===
        # (pouze přidat validation na začátek)
        
        # Track for evaluation
        predictions = []
        actuals = []
        current_season = None
        
        # Process games chronologically
        for idx, game in games_df.iterrows():
            home_team_id = game['home_team_id']
            away_team_id = game['away_team_id']
            
            # Apply season regression if new season
            if current_season != game['season']:
                if current_season is not None:
                    self._apply_season_regression()
                    logger.info(f"Applied season regression for season {game['season']}")
                current_season = game['season']
            
            # Make prediction before updating ratings
            if evaluate_predictions:
                home_rating = self.team_ratings.get(home_team_id, self.initial_rating)
                away_rating = self.team_ratings.get(away_team_id, self.initial_rating)
                predicted_prob = self.expected_score(home_rating, away_rating, self.home_advantage)
                predictions.append(predicted_prob)
            
            # Get actual result
            home_score, result_type = self.game_result_to_score(
                game['home_score'], 
                game['away_score'],
                game['overtime_shootout']
            )
            
            if evaluate_predictions:
                actuals.append(home_score if home_score in [0.0, 1.0] else int(home_score > 0.5))
            
            # Update ratings
            old_home_rating = self.team_ratings.get(home_team_id, self.initial_rating)
            old_away_rating = self.team_ratings.get(away_team_id, self.initial_rating)
            
            new_home_rating, new_away_rating = self.update_ratings(
                home_team_id, away_team_id, home_score, self.home_advantage
            )
            
            # Store rating history
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
        
        # Calculate evaluation metrics
        metrics = {}
        if evaluate_predictions and predictions:
            metrics = self._calculate_metrics(predictions, actuals)
            logger.info(f"Training completed. Accuracy: {metrics.get('accuracy', 0):.3f}")
        
        # Get final team ratings
        team_ratings_df = self.get_current_ratings()
        
        return {
            'metrics': metrics,
            'team_ratings': team_ratings_df,
            'games_processed': len(games_df),
            'rating_history': self.rating_history[-10:]  # Last 10 for inspection
        }
    
    # PŘIDAT helper metodu (Zjednoduš _team_exists() pokud ji chceš zachovat - (správné mapování týmů):
    def _team_exists(self, team_id: int) -> bool:
        """Check if team exists in current schema - SIMPLIFIED"""
        try:
            # Pokud team_id je v games tabulce, existuje
            query = "SELECT 1 FROM games WHERE home_team_id = %s OR away_team_id = %s LIMIT 1"
            result = pd.read_sql(query, self.engine, params=[team_id, team_id])
            return not result.empty
        except:
            return True  # Fallback - assume exists if can't check
    
    def _apply_season_regression(self):
        """Apply regression towards mean between seasons"""
        mean_rating = np.mean(list(self.team_ratings.values()))
        
        for team_id in self.team_ratings:
            current_rating = self.team_ratings[team_id]
            regressed_rating = current_rating + self.season_regression * (mean_rating - current_rating)
            self.team_ratings[team_id] = regressed_rating
    
    def predict_game(self, home_team_id: int, away_team_id: int) -> Dict:
        """
        Predict outcome of a single game
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            
        Returns:
            Dictionary with prediction details
        """
        home_rating = self.team_ratings.get(home_team_id, self.initial_rating)
        away_rating = self.team_ratings.get(away_team_id, self.initial_rating)
        
        home_win_prob = self.expected_score(home_rating, away_rating, self.home_advantage)
        away_win_prob = 1 - home_win_prob
        
        # Get team names
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
            'confidence': abs(home_win_prob - 0.5) * 2,  # 0-1 scale
            'rating_difference': home_rating - away_rating + self.home_advantage
        }
    
    # === NAHRADIT EXISTUJÍCÍ METODU ===
    def predict_upcoming_games(self, days_ahead: int = 7) -> List[Dict]:
        """
        Predict outcomes for upcoming games
        Updated for franchise-based schema
        
        Args:
            days_ahead: Number of days ahead to predict
            
        Returns:
            List of prediction dictionaries
        """
        query = f"""
        SELECT 
            g.id,
            g.date,
            g.datetime_et,
            g.home_team_id,
            g.away_team_id,
            
            -- Current team names (is_current = TRUE)
            ht.name as home_team_name,
            at.name as away_team_name,
            
            -- Franchise info for context
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
        
        logger.info(f"Generated predictions for {len(predictions)} upcoming games")
        return predictions
    
    # === NAHRADIT EXISTUJÍCÍ METODU ===
    def get_current_ratings(self) -> pd.DataFrame:
        """
        Get current team ratings as DataFrame
        Updated for franchise-based schema
        """
        if not self.team_ratings:
            return pd.DataFrame()
        
        team_names = self._get_team_names(list(self.team_ratings.keys()))
        
        ratings_data = []
        for team_id, rating in self.team_ratings.items():
            # Skip historical ratings
            if isinstance(team_id, str) and 'historical' in str(team_id):
                continue
                
            ratings_data.append({
                'team_id': team_id,
                'team_name': team_names.get(team_id, f'Team_{team_id}'),
                'elo_rating': rating,
                'rating_rank': 0  # Will be filled after sorting
            })
        
        df = pd.DataFrame(ratings_data)
        if not df.empty:
            df = df.sort_values('elo_rating', ascending=False).reset_index(drop=True)
            df['rating_rank'] = range(1, len(df) + 1)
        
        return df
    
    # === NAHRADIT EXISTUJÍCÍ METODU ===
    # OPRAVA: _get_team_names() metody - SQL chyba s COALESCE (správné mapování týmů)
    def _get_team_names(self, team_ids: List[int]) -> Dict[int, str]:
        """
        Get team names for given team IDs
        Updated for franchise-based schema - FIXED SQL
        """
        if not team_ids:
            return {}
        
        # Filter out non-integer team IDs (historical ratings)
        valid_team_ids = [tid for tid in team_ids if isinstance(tid, int)]
        if not valid_team_ids:
            return {}
        
        # Handle single item case for SQL IN clause
        if len(valid_team_ids) == 1:
            query = f"""
            SELECT t.id, 
                CASE 
                    WHEN t.is_current = TRUE THEN t.name
                    ELSE CONCAT(t.name, ' (', 
                                EXTRACT(YEAR FROM t.effective_from)::text, 
                                '-', 
                                CASE 
                                    WHEN t.effective_to IS NULL THEN 'present'
                                    ELSE EXTRACT(YEAR FROM t.effective_to)::text 
                                END, 
                                ')')
                END as display_name,
                f.franchise_name
            FROM teams t
            JOIN franchises f ON t.franchise_id = f.id
            WHERE t.id = {valid_team_ids[0]}
            """
        else:
            team_ids_str = ','.join(map(str, valid_team_ids))
            query = f"""
            SELECT t.id, 
                CASE 
                    WHEN t.is_current = TRUE THEN t.name
                    ELSE CONCAT(t.name, ' (', 
                                EXTRACT(YEAR FROM t.effective_from)::text, 
                                '-', 
                                CASE 
                                    WHEN t.effective_to IS NULL THEN 'present'
                                    ELSE EXTRACT(YEAR FROM t.effective_to)::text 
                                END, 
                                ')')
                END as display_name,
                f.franchise_name
            FROM teams t
            JOIN franchises f ON t.franchise_id = f.id
            WHERE t.id IN ({team_ids_str})
            """
        
        try:
            df = pd.read_sql(query, self.engine)
            return dict(zip(df['id'], df['display_name']))
        except Exception as e:
            logger.error(f"Error getting team names: {e}")
            return {tid: f'Team_{tid}' for tid in valid_team_ids}
    
    def _calculate_metrics(self, predictions: List[float], actuals: List[int]) -> Dict:
        """Calculate prediction metrics"""
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Convert probabilities to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Accuracy
        accuracy = np.mean(binary_predictions == actuals)
        
        # Brier Score (lower is better)
        brier_score = np.mean((predictions - actuals) ** 2)
        
        # Log Loss 
        epsilon = 1e-15  # Prevent log(0)
        predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
        log_loss = -np.mean(actuals * np.log(predictions_clipped) + 
                           (1 - actuals) * np.log(1 - predictions_clipped))
        
        return {
            'accuracy': accuracy,
            'brier_score': brier_score,
            'log_loss': log_loss,
            'total_predictions': len(predictions)
        }

    # === PŘIDAT NOVÉ METODY ===
    def get_team_id_for_date(self, team_name: str, game_date: str) -> Optional[int]:
        """
        Získá správné team_id pro daný název a datum
        Řeší historické změny (Arizona → Utah, Winnipeg Jets disambiguation)
        
        Args:
            team_name: Název týmu (může být historický)
            game_date: Datum zápasu (YYYY-MM-DD)
            
        Returns:
            team_id nebo None pokud tým nebyl nalezen
        """
        # Normalize team name first
        normalized_name = self._normalize_team_name(team_name, game_date)
        
        query = """
        SELECT t.id, t.name, f.franchise_name 
        FROM teams t
        JOIN franchises f ON t.franchise_id = f.id
        WHERE (t.name = %s OR f.franchise_name LIKE %s)
        AND (%s >= t.effective_from)
        AND (%s <= t.effective_to OR t.effective_to IS NULL)
        ORDER BY t.effective_from DESC
        LIMIT 1
        """
        
        try:
            result = pd.read_sql(query, self.engine, params=[
                normalized_name, f'%{normalized_name}%', game_date, game_date
            ])
            
            if not result.empty:
                return int(result['id'].iloc[0])
            else:
                logger.warning(f"Team not found: {team_name} for date {game_date}")
                return None
                
        except Exception as e:
            logger.error(f"Error resolving team {team_name} for date {game_date}: {e}")
            return None

    def _normalize_team_name(self, team_name: str, game_date: str) -> str:
        """
        Normalizuje názvy týmů podle data
        Handles Arizona → Utah transition and Winnipeg Jets disambiguation
        """
        game_date = str(game_date)  # Ensure string format
        
        # Arizona/Utah mapping podle data
        if 'Arizona Coyotes' in team_name:
            if game_date >= '2024-04-18':
                return 'Utah Mammoth'
            return 'Arizona Coyotes'
        
        if any(x in team_name for x in ['Utah Hockey Club', 'Utah Mammoth', 'Utah HC']):
            return 'Utah Mammoth'
            
        # Winnipeg Jets disambiguation
        if 'Winnipeg Jets' in team_name:
            if game_date <= '2011-05-31':
                return 'Utah Mammoth'  # Historical Jets → Utah lineage
            return 'Winnipeg Jets'      # Current Jets (from Atlanta)
        
        # Phoenix Coyotes historical mapping
        if 'Phoenix Coyotes' in team_name:
            return 'Utah Mammoth'
        
        return team_name

    def get_franchise_id_for_team(self, team_id: int) -> Optional[int]:
        """Získá franchise_id pro daný team_id"""
        query = "SELECT franchise_id FROM teams WHERE id = %s"
        try:
            result = pd.read_sql(query, self.engine, params=[team_id])
            return int(result['franchise_id'].iloc[0]) if not result.empty else None
        except Exception as e:
            logger.error(f"Error getting franchise for team {team_id}: {e}")
            return None

    def handle_franchise_transition(self, old_team_id: int, new_team_id: int, transition_date: str):
        """
        Přenese rating ze starého týmu na nový při změně franchise
        """
        if old_team_id in self.team_ratings:
            old_rating = self.team_ratings[old_team_id]
            self.team_ratings[new_team_id] = old_rating
            
            logger.info(f"Transferred rating {old_rating:.1f} from team {old_team_id} to {new_team_id} on {transition_date}")
            
            # Keep old rating for historical analysis but mark it
            self.team_ratings[f"{old_team_id}_historical"] = old_rating

    def load_backtesting_games(self, season: str = '2025') -> pd.DataFrame:
        """
        Load games for backtesting (season 2024/25)
        KRITICKÉ: Tyto data NESMÍ být použita pro trénování!
        
        Args:
            season: Season for backtesting (default '2025' = season 2024/25)
            
        Returns:
            DataFrame with games for backtesting (includes scheduled games)
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
            
            -- Current team names (from franchise perspective)
            ht.name as home_team_name,
            hf.franchise_name as home_franchise_name,
            
            -- Away team with franchise info  
            at.name as away_team_name,
            af.franchise_name as away_franchise_name
            
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN franchises hf ON ht.franchise_id = hf.id
        JOIN teams at ON g.away_team_id = at.id  
        JOIN franchises af ON at.franchise_id = af.id
        
        WHERE g.season = '{season}'
            -- Include both completed AND scheduled games for backtesting
        ORDER BY g.date, g.datetime_et, g.id
        """
        
        df = pd.read_sql(query, self.engine)
        
        completed_games = len(df[df['status'] == 'completed'])
        scheduled_games = len(df[df['status'] == 'scheduled'])
        
        logger.info(f"Loaded {len(df)} games for BACKTESTING from season {season}")
        logger.info(f"  - Completed: {completed_games} games")
        logger.info(f"  - Scheduled: {scheduled_games} games")
        logger.warning(f"REMINDER: These games were NOT used for model training!")
        
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
    
    # 4. DEBUG: Přidej dotaz na zjištění skutečných team IDs v databázi
    def debug_team_ids(self):
        """Debug method to see actual team IDs in database"""
        try:
            query = """
            SELECT t.id, t.name, t.is_current, f.franchise_name
            FROM teams t
            JOIN franchises f ON t.franchise_id = f.id
            ORDER BY t.id
            """
            df = pd.read_sql(query, self.engine)
            logger.info("ACTUAL TEAM IDs IN DATABASE:")
            for _, row in df.iterrows():
                current_flag = "✓" if row['is_current'] else "✗"
                logger.info(f"  ID {row['id']:2d}: {row['name']} ({row['franchise_name']}) {current_flag}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error checking team IDs: {e}")
            return pd.DataFrame()
    # === PŘIDAT NOVÉ METODY ===
    
    # === UPRAVIT save_model() metodu ===
    def save_model(self, filepath: str = 'models/elo_model.pkl'):
        """Save the trained model with schema version"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
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
            'schema_version': '2.0',  # NOVÉ: označuje franchise-based schema
            'franchise_support': True  # NOVÉ: podporuje franchise tracking
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath} (schema v2.0)")
        
    # === UPRAVIT load_model() metodu ===
    def load_model(self, filepath: str = 'models/elo_model.pkl'):
        """Load a trained model with schema compatibility check"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check schema version
        schema_version = model_data.get('schema_version', '1.0')
        if schema_version == '1.0':
            logger.warning("Loading old schema model - consider migrating")
            # Zde by byla migrace pokud potřeba
        
        self.team_ratings = model_data['team_ratings']
        self.rating_history = model_data['rating_history']
        
        # Load parameters
        params = model_data['parameters']
        self.initial_rating = params['initial_rating']
        self.k_factor = params['k_factor'] 
        self.home_advantage = params['home_advantage']
        self.season_regression = params['season_regression']
        
        logger.info(f"Model loaded from {filepath} (schema v{schema_version})")

# === NAHRADIT EXISTUJÍCÍ main() FUNKCI ===
def main():
    """
    Main function to train Elo model
    KRITICKÉ: Trénuje pouze na datech do 2023/24, 2024/25 je pro backtesting
    """
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logger.info("🏒 Starting Elo Rating System training...")
    
    try:
        # Initialize Elo system
        elo = EloRatingSystem(
            initial_rating=1500.0,
            k_factor=32.0,
            home_advantage=100.0,
            season_regression=0.25
        )
        
        # Get data split summary first
        data_summary = elo.get_data_split_summary()
        logger.info("\n📊 DATA SPLIT SUMMARY:")
        logger.info(f"  Training seasons: {data_summary['training_seasons']}")
        logger.info(f"  Training games: {data_summary['training_games']}")
        logger.info(f"  Backtesting season: {data_summary['backtesting_season']}")
        logger.info(f"  Backtesting games: {data_summary['backtesting_games']}")
        
        # Load TRAINING data (seasons 2022-2024, excluding 2024/25)
        logger.info("\n📚 Loading TRAINING data (up to 2023/24)...")
        games_df = elo.load_historical_games(season_start='2022', season_end='2024')
        
        if games_df.empty:
            logger.error("No training games found. Please ensure data is imported.")
            return
        
        # Validate data split
        max_season = games_df['season'].max()
        try:
            max_season_int = int(max_season)
            if max_season_int > 2024:
                logger.error(f"CRITICAL ERROR: Training data contains season {max_season}!")
                logger.error("Season 2024/25 must be reserved for backtesting!")
                return
        except (ValueError, TypeError):
            # Handle string season format
            if str(max_season) > '2024':
                logger.error(f"CRITICAL ERROR: Training data contains season {max_season}!")
                logger.error("Season 2024/25 must be reserved for backtesting!")
            return
        
        logger.info(f"✅ Training data validated: {len(games_df)} games from seasons {games_df['season'].min()}-{max_season}")
        
        # Train the model
        logger.info("\n🎯 Training Elo ratings...")
        results = elo.train_on_historical_data(games_df, evaluate_predictions=True)
        
        # Display results
        logger.info("\n🎯 TRAINING RESULTS:")
        metrics = results['metrics']
        logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
        logger.info(f"  Brier Score: {metrics.get('brier_score', 0):.3f}")
        logger.info(f"  Log Loss: {metrics.get('log_loss', 0):.3f}")
        logger.info(f"  Games Processed: {results['games_processed']}")
        
        # Show current team rankings
        logger.info("\n🏆 TOP 10 TEAM RATINGS:")
        ratings_df = results['team_ratings']
        for _, team in ratings_df.head(10).iterrows():
            logger.info(f"  {team['rating_rank']:2d}. {team['team_name']:<25} {team['elo_rating']:7.1f}")
        
        # Arizona → Utah transition check
        utah_teams = ratings_df[ratings_df['team_name'].str.contains('Utah', na=False)]
        if not utah_teams.empty:
            logger.info(f"\n🦣 UTAH TRANSITION VERIFIED:")
            for _, team in utah_teams.iterrows():
                logger.info(f"  {team['team_name']}: {team['elo_rating']:.1f} (rank {team['rating_rank']})")
        
        # Save the model
        elo.save_model('models/elo_model_trained_2024.pkl')
        
        # Show backtesting data availability
        logger.info("\n📊 BACKTESTING DATA AVAILABLE:")
        backtesting_df = elo.load_backtesting_games('2025')
        completed_backtest = len(backtesting_df[backtesting_df['status'] == 'completed'])
        logger.info(f"  Total games: {len(backtesting_df)}")
        logger.info(f"  Completed (ready for backtest): {completed_backtest}")
        logger.info(f"  Scheduled (future predictions): {len(backtesting_df) - completed_backtest}")
        
        logger.info("\n🎉 Model training completed successfully!")
        logger.info("📋 NEXT STEPS:")
        logger.info("  1. Run backtesting on 2024/25 season data")
        logger.info("  2. Validate model performance on out-of-sample data")
        logger.info("  3. If successful, proceed to live trading implementation")
        
    except Exception as e:
        logger.error(f"❌ Elo model training failed: {e}")
        raise

if __name__ == "__main__":
    main()