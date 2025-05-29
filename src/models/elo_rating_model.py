#!/usr/bin/env python3
"""
Elo Rating System for NHL Hockey Predictions
Implements dynamic team ratings that update after each game
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
    
    def load_historical_games(self, season_start: str = '2022') -> pd.DataFrame:
        """
        Load historical games from database for training
        
        Args:
            season_start: First season to include (e.g., '2022')
            
        Returns:
            DataFrame with game results
        """
        query = f"""
        SELECT 
            g.id,
            g.date,
            g.season,
            g.home_team_id,
            g.away_team_id,
            g.home_score,
            g.away_score,
            g.overtime_shootout,
            ht.name as home_team_name,
            at.name as away_team_name
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE g.status = 'completed'
            AND g.season >= '{season_start}'
            AND g.home_score IS NOT NULL 
            AND g.away_score IS NOT NULL
        ORDER BY g.date, g.id
        """
        
        df = pd.read_sql(query, self.engine)
        
        logger.info(f"Loaded {len(df)} completed games from season {season_start} onwards")
        return df
    
    def train_on_historical_data(self, games_df: pd.DataFrame, 
                               evaluate_predictions: bool = True) -> Dict:
        """
        Train Elo ratings on historical game data
        
        Args:
            games_df: DataFrame with historical games
            evaluate_predictions: Whether to track predictions for evaluation
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("Training Elo ratings on historical data...")
        
        # Initialize all teams with base rating
        unique_teams = set(games_df['home_team_id'].unique()) | set(games_df['away_team_id'].unique())
        for team_id in unique_teams:
            self.team_ratings[team_id] = self.initial_rating
        
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
    
    def predict_upcoming_games(self, days_ahead: int = 7) -> List[Dict]:
        """
        Predict outcomes for upcoming games
        
        Args:
            days_ahead: Number of days ahead to predict
            
        Returns:
            List of prediction dictionaries
        """
        query = f"""
        SELECT 
            g.id,
            g.date,
            g.home_team_id,
            g.away_team_id,
            ht.name as home_team_name,
            at.name as away_team_name
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE g.status = 'scheduled'
            AND g.date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '{days_ahead} days'
        ORDER BY g.date
        """
        
        upcoming_games = pd.read_sql(query, self.engine)
        
        predictions = []
        for _, game in upcoming_games.iterrows():
            prediction = self.predict_game(game['home_team_id'], game['away_team_id'])
            prediction['game_id'] = game['id']
            prediction['game_date'] = game['date']
            predictions.append(prediction)
        
        logger.info(f"Generated predictions for {len(predictions)} upcoming games")
        return predictions
    
    def get_current_ratings(self) -> pd.DataFrame:
        """Get current team ratings as DataFrame"""
        if not self.team_ratings:
            return pd.DataFrame()
        
        team_names = self._get_team_names(list(self.team_ratings.keys()))
        
        ratings_data = []
        for team_id, rating in self.team_ratings.items():
            ratings_data.append({
                'team_id': team_id,
                'team_name': team_names.get(team_id, f'Team_{team_id}'),
                'elo_rating': rating,
                'rating_rank': 0  # Will be filled after sorting
            })
        
        df = pd.DataFrame(ratings_data)
        df = df.sort_values('elo_rating', ascending=False).reset_index(drop=True)
        df['rating_rank'] = range(1, len(df) + 1)
        
        return df
    
    def _get_team_names(self, team_ids: List[int]) -> Dict[int, str]:
        """Get team names for given team IDs"""
        if not team_ids:
            return {}
        
        # Handle single item case for SQL IN clause
        if len(team_ids) == 1:
            query = f"SELECT id, name FROM teams WHERE id = {team_ids[0]}"
        else:
            team_ids_str = ','.join(map(str, team_ids))
            query = f"SELECT id, name FROM teams WHERE id IN ({team_ids_str})"
        
        df = pd.read_sql(query, self.engine)
        return dict(zip(df['id'], df['name']))
    
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
    
    def save_model(self, filepath: str = 'models/elo_model.pkl'):
        """Save the trained model"""
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
            'trained_date': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'models/elo_model.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.team_ratings = model_data['team_ratings']
        self.rating_history = model_data['rating_history']
        
        # Load parameters
        params = model_data['parameters']
        self.initial_rating = params['initial_rating']
        self.k_factor = params['k_factor'] 
        self.home_advantage = params['home_advantage']
        self.season_regression = params['season_regression']
        
        logger.info(f"Model loaded from {filepath}")

def main():
    """Main function to train and evaluate Elo model"""
    
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
        
        # Load historical data
        logger.info("Loading historical games...")
        games_df = elo.load_historical_games(season_start='2022')
        
        if games_df.empty:
            logger.error("No historical games found. Please ensure data is imported.")
            return
        
        # Train the model
        logger.info("Training Elo ratings...")
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
        
        # Make predictions for upcoming games
        logger.info("\n🔮 UPCOMING GAME PREDICTIONS:")
        upcoming_predictions = elo.predict_upcoming_games(days_ahead=7)
        
        for prediction in upcoming_predictions[:5]:  # Show first 5
            home_team = prediction['home_team_name']
            away_team = prediction['away_team_name']
            home_prob = prediction['home_win_probability']
            confidence = prediction['confidence']
            
            logger.info(f"  {away_team} @ {home_team}")
            logger.info(f"    Home Win: {home_prob:.1%} | Confidence: {confidence:.1%}")
        
        # Save the model
        elo.save_model('models/elo_model.pkl')
        
        logger.info("🎉 Elo model training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Elo model training failed: {e}")
        raise

if __name__ == "__main__":
    main()