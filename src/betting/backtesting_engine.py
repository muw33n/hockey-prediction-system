#!/usr/bin/env python3
"""
Hockey Backtesting Engine
Dynamic Elo-based value betting simulation with comprehensive analysis

Features:
- Dynamic Elo updates per gaming day
- Multiple EV calculation variants (basic, Kelly-enhanced, confidence-weighted)
- Best odds selection from multiple bookmakers
- Chronological processing with no future information leakage
- Comprehensive performance tracking and analysis

Location: src/betting/backtesting_engine.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import logging
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
from collections import defaultdict
import warnings

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtesting.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BacktestingEngine:
    """
    Hockey Backtesting Engine with Dynamic Elo Evolution
    
    Simulates value betting on NHL games using trained Elo model
    with dynamic rating updates per gaming day
    """
    
    def __init__(self, 
                 elo_model_path: str = 'models/elo_model_trained_2024.pkl',
                 initial_bankroll: float = 10000.0,
                 database_url: Optional[str] = None):
        """
        Initialize Backtesting Engine
        
        Args:
            elo_model_path: Path to trained Elo model
            initial_bankroll: Starting bankroll amount
            database_url: Database connection string (optional)
        """
        self.elo_model_path = elo_model_path
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        
        # Database connection
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.engine = create_engine(self.database_url)
        
        # Load Elo model
        self.elo_model = None
        self._load_elo_model()
        
        # Backtesting data
        self.games_df = None
        self.odds_df = None
        self.gaming_days = None
        
        # Performance tracking
        self.bet_history = []
        self.daily_bankroll = []
        self.daily_performance = []
        self.elo_evolution = []
        
        # Statistics
        self.stats = {
            'total_bets': 0,
            'winning_bets': 0,
            'total_staked': 0.0,
            'total_returns': 0.0,
            'max_bankroll': initial_bankroll,
            'min_bankroll': initial_bankroll,
            'max_drawdown': 0.0,
            'games_processed': 0,
            'gaming_days_processed': 0
        }
        
        logger.info(f"🏒 BacktestingEngine initialized with bankroll: €{initial_bankroll:,.0f}")
    
    def _load_elo_model(self):
        """Load trained Elo model from pickle file"""
        try:
            if not os.path.exists(self.elo_model_path):
                raise FileNotFoundError(f"Elo model not found: {self.elo_model_path}")
            
            with open(self.elo_model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Import EloRatingSystem class
            import sys
            sys.path.append('src/models')
            from elo_rating_model import EloRatingSystem
            
            # Recreate Elo model
            self.elo_model = EloRatingSystem(
                initial_rating=model_data['parameters']['initial_rating'],
                k_factor=model_data['parameters']['k_factor'],
                home_advantage=model_data['parameters']['home_advantage'],
                season_regression=model_data['parameters']['season_regression']
            )
            
            # Load trained ratings
            self.elo_model.team_ratings = model_data['team_ratings'].copy()
            self.elo_model.rating_history = model_data['rating_history'].copy()
            
            schema_version = model_data.get('schema_version', '1.0')
            trained_date = model_data.get('trained_date', 'Unknown')
            
            logger.info(f"✅ Elo model loaded successfully")
            logger.info(f"   Schema version: {schema_version}")
            logger.info(f"   Trained date: {trained_date}")
            logger.info(f"   Teams with ratings: {len(self.elo_model.team_ratings)}")
            
            # Validate model has ratings
            if not self.elo_model.team_ratings:
                raise ValueError("Loaded Elo model has no team ratings")
                
        except Exception as e:
            logger.error(f"❌ Failed to load Elo model: {e}")
            raise
    
    def load_backtesting_data(self, season: str = '2025') -> Dict[str, int]:
        """
        Load games and odds data for backtesting
        
        Args:
            season: Season to backtest (default '2025' = 2024/25)
            
        Returns:
            Dictionary with data loading summary
        """
        logger.info(f"📊 Loading backtesting data for season {season}...")
        
        # Load games data
        games_query = f"""
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
            
            -- Current team names
            ht.name as home_team_name,
            at.name as away_team_name
            
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id AND ht.is_current = TRUE
        JOIN teams at ON g.away_team_id = at.id AND at.is_current = TRUE
        
        WHERE g.season = '{season}'
            AND g.status = 'completed'
            AND g.home_score IS NOT NULL 
            AND g.away_score IS NOT NULL
        ORDER BY g.date, g.datetime_et, g.id
        """
        
        self.games_df = pd.read_sql(games_query, self.engine)
        
        if self.games_df.empty:
            raise ValueError(f"No completed games found for season {season}")
        
        # Load odds data
        odds_query = f"""
        SELECT 
            o.game_id,
            o.bookmaker,
            o.market_type,
            o.home_odd,
            o.away_odd,
            o.home_opening_odd,
            o.away_opening_odd,
            g.date as game_date
            
        FROM odds o
        JOIN games g ON o.game_id = g.id
        
        WHERE g.season = '{season}'
            AND o.market_type = 'moneyline_2way'
            AND o.home_odd IS NOT NULL 
            AND o.away_odd IS NOT NULL
        ORDER BY g.date, o.game_id, o.bookmaker
        """
        
        self.odds_df = pd.read_sql(odds_query, self.engine)
        
        # Group games by gaming days
        self.games_df['game_date'] = pd.to_datetime(self.games_df['date']).dt.date
        self.gaming_days = self.games_df.groupby('game_date')
        
        # Summary
        summary = {
            'games_loaded': len(self.games_df),
            'gaming_days': len(self.gaming_days),
            'odds_records': len(self.odds_df),
            'unique_bookmakers': self.odds_df['bookmaker'].nunique() if not self.odds_df.empty else 0,
            'date_range': f"{self.games_df['date'].min()} to {self.games_df['date'].max()}"
        }
        
        logger.info(f"✅ Backtesting data loaded:")
        logger.info(f"   Games: {summary['games_loaded']:,}")
        logger.info(f"   Gaming days: {summary['gaming_days']:,}")
        logger.info(f"   Odds records: {summary['odds_records']:,}")
        logger.info(f"   Bookmakers: {summary['unique_bookmakers']}")
        logger.info(f"   Date range: {summary['date_range']}")
        
        if self.odds_df.empty:
            logger.warning("⚠️ No odds data found - backtesting will only validate predictions")
        
        return summary
    
    def calculate_ev_variants(self, 
                            model_prob: float, 
                            bookmaker_odds: float,
                            confidence: Optional[float] = None,
                            max_kelly: float = 0.25) -> Dict[str, float]:
        """
        Calculate multiple Expected Value variants
        
        Args:
            model_prob: Model's predicted probability
            bookmaker_odds: Bookmaker's decimal odds
            confidence: Model confidence (0-1), calculated if None
            max_kelly: Maximum Kelly fraction cap
            
        Returns:
            Dictionary with EV calculations
        """
        if confidence is None:
            confidence = abs(model_prob - 0.5) * 2  # 0-1 scale
        
        # Basic Expected Value
        basic_ev = (model_prob * bookmaker_odds) - 1
        
        # Kelly-enhanced EV
        if bookmaker_odds > 1.0:
            kelly_fraction = (bookmaker_odds * model_prob - 1) / (bookmaker_odds - 1)
            kelly_fraction = max(0, min(kelly_fraction, max_kelly))
            kelly_enhanced_ev = basic_ev * kelly_fraction if kelly_fraction > 0 else 0
        else:
            kelly_enhanced_ev = 0
        
        # Confidence-weighted EV
        confidence_weighted_ev = basic_ev * (0.5 + 0.5 * confidence)
        
        return {
            'basic_ev': basic_ev,
            'kelly_enhanced_ev': kelly_enhanced_ev,
            'confidence_weighted_ev': confidence_weighted_ev,
            'kelly_fraction': kelly_fraction if 'kelly_fraction' in locals() else 0,
            'confidence': confidence
        }
    
    def get_best_odds(self, game_id: int) -> Dict[str, Any]:
        """
        Get best available odds for a game from all bookmakers
        
        Args:
            game_id: Game ID to get odds for
            
        Returns:
            Dictionary with best odds information
        """
        game_odds = self.odds_df[self.odds_df['game_id'] == game_id]
        
        if game_odds.empty:
            return {
                'has_odds': False,
                'home_odd': None,
                'away_odd': None,
                'home_bookmaker': None,
                'away_bookmaker': None
            }
        
        # Find best (highest) odds for each selection
        best_home_idx = game_odds['home_odd'].idxmax()
        best_away_idx = game_odds['away_odd'].idxmax()
        
        best_home_odd = game_odds.loc[best_home_idx, 'home_odd']
        best_away_odd = game_odds.loc[best_away_idx, 'away_odd']
        best_home_bookmaker = game_odds.loc[best_home_idx, 'bookmaker']
        best_away_bookmaker = game_odds.loc[best_away_idx, 'bookmaker']
        
        return {
            'has_odds': True,
            'home_odd': best_home_odd,
            'away_odd': best_away_odd,
            'home_bookmaker': best_home_bookmaker,
            'away_bookmaker': best_away_bookmaker,
            'bookmakers_count': len(game_odds),
            'avg_home_odd': game_odds['home_odd'].mean(),
            'avg_away_odd': game_odds['away_odd'].mean()
        }
    
    def calculate_stake_size(self, 
                           ev_value: float,
                           odds: float,
                           model_prob: float,
                           stake_method: str = 'fixed',
                           stake_size: float = 0.02,
                           max_stake_pct: float = 0.10) -> float:
        """
        Calculate stake size based on specified method
        
        Args:
            ev_value: Expected value of the bet
            odds: Bookmaker odds
            model_prob: Model's predicted probability
            stake_method: 'fixed', 'kelly', or 'hybrid'
            stake_size: Fixed percentage or Kelly multiplier
            max_stake_pct: Maximum stake as percentage of bankroll
            
        Returns:
            Stake amount in currency units
        """
        if ev_value <= 0:
            return 0.0
        
        if stake_method == 'fixed':
            stake_amount = self.current_bankroll * stake_size
            
        elif stake_method == 'kelly':
            if odds > 1.0:
                kelly_fraction = (odds * model_prob - 1) / (odds - 1)
                kelly_fraction = max(0, kelly_fraction * stake_size)  # stake_size as Kelly multiplier
                stake_amount = self.current_bankroll * kelly_fraction
            else:
                stake_amount = 0.0
                
        elif stake_method == 'hybrid':
            # Combine fixed base with Kelly adjustment
            base_stake = self.current_bankroll * (stake_size * 0.5)
            kelly_fraction = (odds * model_prob - 1) / (odds - 1) if odds > 1.0 else 0
            kelly_adjustment = self.current_bankroll * kelly_fraction * (stake_size * 0.5)
            stake_amount = base_stake + max(0, kelly_adjustment)
        else:
            raise ValueError(f"Unknown stake method: {stake_method}")
        
        # Apply maximum stake limit
        max_stake = self.current_bankroll * max_stake_pct
        stake_amount = min(stake_amount, max_stake)
        
        # Minimum stake (avoid micro bets)
        min_stake = 1.0  # €1 minimum
        stake_amount = max(stake_amount, min_stake) if stake_amount > 0 else 0.0
        
        return round(stake_amount, 2)
    
    def process_gaming_day(self, 
                          day_date: date, 
                          day_games: pd.DataFrame,
                          edge_threshold: float = 0.05,
                          min_odds: float = 1.20,
                          stake_method: str = 'fixed',
                          stake_size: float = 0.02,
                          ev_method: str = 'basic') -> Dict[str, Any]:
        """
        Process all games for a single gaming day
        
        Args:
            day_date: Date of the gaming day
            day_games: DataFrame with games for this day
            edge_threshold: Minimum edge required for value bet
            min_odds: Minimum odds threshold
            stake_method: Staking method ('fixed', 'kelly', 'hybrid')
            stake_size: Stake size parameter
            ev_method: EV calculation method ('basic', 'kelly_enhanced', 'confidence_weighted')
            
        Returns:
            Dictionary with day processing results
        """
        day_results = {
            'date': day_date,
            'games_count': len(day_games),
            'bets_placed': 0,
            'total_staked': 0.0,
            'predictions': [],
            'value_bets': [],
            'elo_updates': []
        }
        
        starting_bankroll = self.current_bankroll
        
        # 1. Make predictions for all games (before any Elo updates)
        for _, game in day_games.iterrows():
            prediction = self.elo_model.predict_game(
                game['home_team_id'], 
                game['away_team_id']
            )
            
            prediction['game_id'] = game['id']
            prediction['actual_home_score'] = game['home_score']
            prediction['actual_away_score'] = game['away_score']
            prediction['actual_winner'] = 'HOME' if game['home_score'] > game['away_score'] else 'AWAY'
            prediction['correct_prediction'] = prediction['predicted_winner'] == prediction['actual_winner']
            
            day_results['predictions'].append(prediction)
        
        # 2. Evaluate value bets if odds available
        if not self.odds_df.empty:
            for prediction in day_results['predictions']:
                game_id = prediction['game_id']
                odds_info = self.get_best_odds(game_id)
                
                if not odds_info['has_odds']:
                    continue
                
                # Check both home and away opportunities
                opportunities = [
                    {
                        'selection': 'home',
                        'model_prob': prediction['home_win_probability'],
                        'odds': odds_info['home_odd'],
                        'bookmaker': odds_info['home_bookmaker']
                    },
                    {
                        'selection': 'away', 
                        'model_prob': prediction['away_win_probability'],
                        'odds': odds_info['away_odd'],
                        'bookmaker': odds_info['away_bookmaker']
                    }
                ]
                
                for opp in opportunities:
                    if opp['odds'] < min_odds:
                        continue
                    
                    ev_calcs = self.calculate_ev_variants(
                        opp['model_prob'], 
                        opp['odds'],
                        prediction['confidence']
                    )
                    
                    selected_ev = ev_calcs[f'{ev_method}_ev']
                    
                    if selected_ev > edge_threshold:
                        # Calculate stake
                        stake = self.calculate_stake_size(
                            selected_ev, opp['odds'], opp['model_prob'],
                            stake_method, stake_size
                        )
                        
                        if stake > 0 and stake <= self.current_bankroll:
                            # Place bet
                            actual_winner = prediction['actual_winner']
                            bet_won = (opp['selection'].upper() == actual_winner)
                            
                            payout = stake * opp['odds'] if bet_won else 0.0
                            net_result = payout - stake
                            
                            bet_record = {
                                'game_id': game_id,
                                'date': day_date,
                                'selection': opp['selection'],
                                'model_prob': opp['model_prob'],
                                'odds': opp['odds'],
                                'bookmaker': opp['bookmaker'],
                                'stake': stake,
                                'ev_method': ev_method,
                                'selected_ev': selected_ev,
                                'all_ev_calcs': ev_calcs,
                                'bet_won': bet_won,
                                'payout': payout,
                                'net_result': net_result,
                                'bankroll_before': self.current_bankroll,
                                'predicted_winner': prediction['predicted_winner'],
                                'actual_winner': actual_winner,
                                'prediction_correct': prediction['correct_prediction']
                            }
                            
                            # Update bankroll
                            self.current_bankroll += net_result
                            bet_record['bankroll_after'] = self.current_bankroll
                            
                            # Track statistics
                            self.bet_history.append(bet_record)
                            day_results['value_bets'].append(bet_record)
                            day_results['bets_placed'] += 1
                            day_results['total_staked'] += stake
                            
                            self.stats['total_bets'] += 1
                            self.stats['total_staked'] += stake
                            self.stats['total_returns'] += payout
                            
                            if bet_won:
                                self.stats['winning_bets'] += 1
        
        # 3. Update Elo ratings after all games of the day
        for _, game in day_games.iterrows():
            home_score, result_type = self.elo_model.game_result_to_score(
                game['home_score'], 
                game['away_score'],
                game.get('overtime_shootout', '')
            )
            
            old_home_rating = self.elo_model.team_ratings.get(game['home_team_id'], 1500.0)
            old_away_rating = self.elo_model.team_ratings.get(game['away_team_id'], 1500.0)
            
            new_home_rating, new_away_rating = self.elo_model.update_ratings(
                game['home_team_id'], 
                game['away_team_id'], 
                home_score,
                self.elo_model.home_advantage
            )
            
            elo_update = {
                'game_id': game['id'],
                'home_team_id': game['home_team_id'],
                'away_team_id': game['away_team_id'],
                'old_home_rating': old_home_rating,
                'old_away_rating': old_away_rating,
                'new_home_rating': new_home_rating,
                'new_away_rating': new_away_rating,
                'rating_change_home': new_home_rating - old_home_rating,
                'rating_change_away': new_away_rating - old_away_rating,
                'result_type': result_type
            }
            
            day_results['elo_updates'].append(elo_update)
            self.elo_evolution.append(elo_update)
        
        # 4. Track daily performance
        ending_bankroll = self.current_bankroll
        daily_return = ending_bankroll - starting_bankroll
        daily_roi = (daily_return / starting_bankroll) if starting_bankroll > 0 else 0.0
        
        daily_performance = {
            'date': day_date,
            'starting_bankroll': starting_bankroll,
            'ending_bankroll': ending_bankroll,
            'daily_return': daily_return,
            'daily_roi': daily_roi,
            'bets_placed': day_results['bets_placed'],
            'total_staked': day_results['total_staked'],
            'games_processed': len(day_games)
        }
        
        self.daily_performance.append(daily_performance)
        self.daily_bankroll.append({
            'date': day_date,
            'bankroll': ending_bankroll
        })
        
        # Update statistics
        self.stats['max_bankroll'] = max(self.stats['max_bankroll'], ending_bankroll)
        self.stats['min_bankroll'] = min(self.stats['min_bankroll'], ending_bankroll)
        self.stats['games_processed'] += len(day_games)
        self.stats['gaming_days_processed'] += 1
        
        return day_results
    
    def run_backtest(self,
                    edge_threshold: float = 0.05,
                    min_odds: float = 1.20,
                    stake_method: str = 'fixed',
                    stake_size: float = 0.02,
                    ev_method: str = 'basic',
                    max_stake_pct: float = 0.10) -> Dict[str, Any]:
        """
        Run complete backtesting simulation
        
        Args:
            edge_threshold: Minimum edge required for value bet
            min_odds: Minimum odds threshold  
            stake_method: Staking method ('fixed', 'kelly', 'hybrid')
            stake_size: Stake size parameter
            ev_method: EV calculation method ('basic', 'kelly_enhanced', 'confidence_weighted')
            max_stake_pct: Maximum stake as percentage of bankroll
            
        Returns:
            Complete backtesting results
        """
        logger.info("🚀 Starting comprehensive backtesting simulation...")
        logger.info(f"Parameters:")
        logger.info(f"  Edge threshold: {edge_threshold:.1%}")
        logger.info(f"  Min odds: {min_odds:.2f}")
        logger.info(f"  Stake method: {stake_method}")
        logger.info(f"  Stake size: {stake_size}")
        logger.info(f"  EV method: {ev_method}")
        logger.info(f"  Max stake: {max_stake_pct:.1%}")
        
        if self.games_df is None:
            raise ValueError("No backtesting data loaded. Call load_backtesting_data() first.")
        
        # Reset simulation state
        self.current_bankroll = self.initial_bankroll
        self.bet_history = []
        self.daily_bankroll = []
        self.daily_performance = []
        self.elo_evolution = []
        self.stats = {
            'total_bets': 0,
            'winning_bets': 0,
            'total_staked': 0.0,
            'total_returns': 0.0,
            'max_bankroll': self.initial_bankroll,
            'min_bankroll': self.initial_bankroll,
            'max_drawdown': 0.0,
            'games_processed': 0,
            'gaming_days_processed': 0
        }
        
        # Process each gaming day chronologically
        gaming_day_results = []
        
        for game_date, day_games in self.gaming_days:
            day_result = self.process_gaming_day(
                game_date, day_games,
                edge_threshold, min_odds, 
                stake_method, stake_size, ev_method
            )
            
            gaming_day_results.append(day_result)
            
            # Progress logging
            if self.stats['gaming_days_processed'] % 30 == 0:
                current_roi = ((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll)
                logger.info(f"📊 Progress: {self.stats['gaming_days_processed']} days, "
                          f"ROI: {current_roi:.2%}, "
                          f"Bets: {self.stats['total_bets']}")
        
        # Calculate final performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Create comprehensive results
        results = {
            'parameters': {
                'edge_threshold': edge_threshold,
                'min_odds': min_odds,
                'stake_method': stake_method,
                'stake_size': stake_size,
                'ev_method': ev_method,
                'max_stake_pct': max_stake_pct
            },
            'performance': performance_metrics,
            'statistics': self.stats,
            'gaming_day_results': gaming_day_results,
            'bet_history': self.bet_history,
            'daily_performance': self.daily_performance,
            'elo_evolution': self.elo_evolution[-10:],  # Last 10 for inspection
            'summary': self._generate_summary()
        }
        
        logger.info("🎉 Backtesting simulation completed!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   ROI: {performance_metrics['roi']:.2%}")
        logger.info(f"   Total Bets: {self.stats['total_bets']:,}")
        logger.info(f"   Win Rate: {performance_metrics['win_rate']:.1%}")
        logger.info(f"   Final Bankroll: €{self.current_bankroll:,.0f}")
        
        return results
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        if not self.bet_history:
            return {
                'roi': 0.0,
                'win_rate': 0.0,
                'avg_odds': 0.0,
                'profit': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'prediction_accuracy': 0.0
            }
        
        bet_df = pd.DataFrame(self.bet_history)
        daily_df = pd.DataFrame(self.daily_performance)
        
        # Basic metrics
        total_profit = self.current_bankroll - self.initial_bankroll
        roi = total_profit / self.initial_bankroll
        win_rate = bet_df['bet_won'].mean()
        avg_odds = bet_df['odds'].mean()
        
        # Drawdown calculation
        bankroll_series = pd.Series([day['bankroll'] for day in self.daily_bankroll])
        running_max = bankroll_series.cummax()
        drawdown_series = (bankroll_series - running_max) / running_max
        max_drawdown = abs(drawdown_series.min())
        
        # Sharpe ratio (daily returns)
        if len(daily_df) > 1 and daily_df['daily_roi'].std() > 0:
            sharpe_ratio = daily_df['daily_roi'].mean() / daily_df['daily_roi'].std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Prediction accuracy (separate from betting performance)
        all_predictions = []
        for day_result in getattr(self, 'gaming_day_results', []):
            all_predictions.extend(day_result.get('predictions', []))
        
        if all_predictions:
            prediction_accuracy = sum(p['correct_prediction'] for p in all_predictions) / len(all_predictions)
        else:
            prediction_accuracy = 0.0
        
        self.stats['max_drawdown'] = max_drawdown
        
        return {
            'roi': roi,
            'win_rate': win_rate,
            'avg_odds': avg_odds,
            'profit': total_profit,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'prediction_accuracy': prediction_accuracy,
            'total_return_pct': (self.current_bankroll / self.initial_bankroll - 1),
            'avg_bet_size': bet_df['stake'].mean(),
            'largest_win': bet_df['net_result'].max(),
            'largest_loss': bet_df['net_result'].min()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate executive summary of backtesting results"""
        
        if not self.bet_history:
            return {'message': 'No bets placed during simulation'}
        
        bet_df = pd.DataFrame(self.bet_history)
        
        # Date range
        date_range = f"{self.games_df['date'].min()} to {self.games_df['date'].max()}"
        
        # Betting activity
        betting_days = bet_df['date'].nunique()
        avg_bets_per_day = len(bet_df) / betting_days if betting_days > 0 else 0
        
        # EV method performance
        ev_methods = bet_df['ev_method'].value_counts().to_dict()
        
        # Bookmaker distribution
        bookmaker_dist = bet_df['bookmaker'].value_counts().to_dict()
        
        # Monthly performance if available
        monthly_performance = None
        if len(self.daily_performance) > 30:
            daily_df = pd.DataFrame(self.daily_performance)
            daily_df['month'] = pd.to_datetime(daily_df['date']).dt.to_period('M')
            monthly_performance = daily_df.groupby('month').agg({
                'daily_return': 'sum',
                'bets_placed': 'sum'
            }).to_dict('index')
        
        return {
            'simulation_period': date_range,
            'gaming_days_total': len(self.gaming_days),
            'gaming_days_with_bets': betting_days,
            'avg_bets_per_active_day': avg_bets_per_day,
            'ev_method_usage': ev_methods,
            'bookmaker_distribution': bookmaker_dist,
            'monthly_performance': monthly_performance,
            'final_bankroll': self.current_bankroll,
            'bankroll_growth': f"{((self.current_bankroll / self.initial_bankroll - 1) * 100):.1f}%"
        }
    
    def save_results(self, 
                    results: Dict[str, Any], 
                    output_dir: str = 'models/experiments',
                    filename_prefix: str = 'backtest') -> str:
        """
        Save backtesting results to files
        
        Args:
            results: Backtesting results dictionary
            output_dir: Output directory
            filename_prefix: Prefix for output files
            
        Returns:
            Path to main results file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Main results (JSON)
        main_file = os.path.join(output_dir, f'{filename_prefix}_results_{timestamp}.json')
        
        # Convert numpy types for JSON serialization
        json_results = self._convert_for_json(results)
        
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Bet history (CSV)
        if self.bet_history:
            bet_file = os.path.join(output_dir, f'{filename_prefix}_bets_{timestamp}.csv')
            bet_df = pd.DataFrame(self.bet_history)
            bet_df.to_csv(bet_file, index=False, encoding='utf-8')
        
        # Daily performance (CSV)
        if self.daily_performance:
            daily_file = os.path.join(output_dir, f'{filename_prefix}_daily_{timestamp}.csv')
            daily_df = pd.DataFrame(self.daily_performance)
            daily_df.to_csv(daily_file, index=False, encoding='utf-8')
        
        logger.info(f"💾 Results saved to {main_file}")
        return main_file
    
    def _convert_for_json(self, obj):
        """Convert numpy types and pandas objects to JSON-serializable types"""
        if isinstance(obj, dict):
            # Convert keys and values
            return {self._convert_key_for_json(key): self._convert_for_json(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (date, datetime)):
            return obj.isoformat()
        elif hasattr(obj, 'to_timestamp'):  # pandas Period
            return str(obj)
        elif hasattr(obj, 'isoformat'):  # Other datetime-like objects
            return obj.isoformat()
        else:
            return obj
    
    def _convert_key_for_json(self, key):
        """Convert dictionary keys to JSON-serializable format"""
        if isinstance(key, str):
            return key
        elif isinstance(key, (int, float, bool)):
            return key
        elif hasattr(key, 'to_timestamp'):  # pandas Period
            return str(key)
        elif isinstance(key, (date, datetime)):
            return key.isoformat()
        else:
            return str(key)


# Example usage and testing
if __name__ == "__main__":
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    logger.info("🏒 Starting BacktestingEngine test...")
    
    try:
        # Initialize engine
        engine = BacktestingEngine(
            elo_model_path='models/elo_model_trained_2024.pkl',
            initial_bankroll=10000.0
        )
        
        # Load backtesting data
        data_summary = engine.load_backtesting_data(season='2025')
        
        # Run basic backtest
        results = engine.run_backtest(
            edge_threshold=0.05,
            min_odds=1.30,
            stake_method='fixed',
            stake_size=0.02,
            ev_method='basic'
        )
        
        # Save results
        output_file = engine.save_results(results)
        
        logger.info("✅ BacktestingEngine test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ BacktestingEngine test failed: {e}")
        raise