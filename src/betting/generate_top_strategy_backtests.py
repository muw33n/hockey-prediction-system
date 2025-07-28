#!/usr/bin/env python3
"""
Generate Top Strategy Backtests
Comprehensive backtest generation for top performing strategies

Features:
- Multi-criteria strategy selection (20-25 strategies)
- Detailed backtesting for each selected strategy
- Multiple output formats for risk assessment
- Progress tracking and robust error handling
- Correlation analysis between strategies

Location: src/betting/generate_top_strategy_backtests.py
"""

import pandas as pd
import numpy as np
import os
import logging
import json
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Set
import warnings
from pathlib import Path
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import backtesting engine
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from backtesting_engine import BacktestingEngine

# Setup logging with unique logger name
logger = logging.getLogger('TopStrategyGenerator')
logger.setLevel(logging.INFO)

# Remove existing handlers to avoid duplication
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create handlers
file_handler = logging.FileHandler('logs/top_strategy_generation.log', encoding='utf-8')
console_handler = logging.StreamHandler()

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent propagation to root logger
logger.propagate = False

class TopStrategyGenerator:
    """
    Top Strategy Backtest Generator
    
    Selects top performing strategies from optimization results and generates
    detailed backtests for comprehensive risk analysis
    """
    
    def __init__(self, 
                 elo_model_path: str = 'models/elo_model_trained_2024.pkl',
                 initial_bankroll: float = 10000.0,
                 results_dir: str = 'models/experiments',
                 target_strategies: int = 23):
        """
        Initialize Top Strategy Generator
        
        Args:
            elo_model_path: Path to trained Elo model
            initial_bankroll: Starting bankroll for each simulation
            results_dir: Directory containing optimization results
            target_strategies: Target number of strategies to generate (20-25)
        """
        self.elo_model_path = elo_model_path
        self.initial_bankroll = initial_bankroll
        self.results_dir = results_dir
        self.target_strategies = target_strategies
        
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"🎯 TopStrategyGenerator initialized")
        logger.info(f"   Target strategies: {target_strategies}")
        logger.info(f"   Results directory: {results_dir}")
    
    def load_optimization_results(self) -> pd.DataFrame:
        """
        Load latest optimization results from CSV or JSON
        
        Returns:
            DataFrame with optimization results
        """
        logger.info("📊 Loading optimization results...")
        
        # Try CSV detailed first (preferred)
        csv_pattern = os.path.join(self.results_dir, '*optimization*detailed*.csv')
        csv_files = glob.glob(csv_pattern)
        
        if csv_files:
            # Get latest CSV file
            latest_csv = max(csv_files, key=os.path.getmtime)
            logger.info(f"📄 Loading CSV: {os.path.basename(latest_csv)}")
            
            try:
                df = pd.read_csv(latest_csv)
                logger.info(f"✅ Loaded {len(df)} optimization records from CSV")
                return df
            except Exception as e:
                logger.warning(f"⚠️ Failed to load CSV: {e}")
        
        # Fallback to JSON
        json_pattern = os.path.join(self.results_dir, '*optimization*results*.json')
        json_files = glob.glob(json_pattern)
        
        if json_files:
            latest_json = max(json_files, key=os.path.getmtime)
            logger.info(f"📄 Loading JSON: {os.path.basename(latest_json)}")
            
            try:
                with open(latest_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'optimization_results' in data:
                    # Convert to DataFrame
                    records = []
                    for result in data['optimization_results']:
                        if 'error' not in result:
                            record = {
                                **result.get('parameters', {}),
                                **result.get('performance', {}),
                                **result.get('statistics', {})
                            }
                            records.append(record)
                    
                    if records:
                        df = pd.DataFrame(records)
                        logger.info(f"✅ Loaded {len(df)} optimization records from JSON")
                        return df
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to load JSON: {e}")
        
        raise FileNotFoundError(f"No optimization results found in {self.results_dir}")
    
    def select_top_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select top strategies using multi-criteria approach
        
        Args:
            df: DataFrame with optimization results
            
        Returns:
            DataFrame with selected top strategies
        """
        logger.info(f"🎯 Selecting top {self.target_strategies} strategies...")
        
        # Clean data - only profitable strategies
        profitable = df[df['roi'] > 0].copy()
        logger.info(f"📊 Found {len(profitable)} profitable strategies out of {len(df)} total")
        
        if len(profitable) == 0:
            logger.warning("⚠️ No profitable strategies found - using all strategies")
            profitable = df.copy()
        
        selected_strategies = set()
        selection_details = []
        
        # 1. Top 8 by ROI (best absolute performance)
        if len(profitable) > 0:
            top_roi = profitable.nlargest(8, 'roi')
            roi_indices = set(top_roi.index)
            selected_strategies.update(roi_indices)
            selection_details.append(f"Top 8 by ROI: {len(roi_indices)} strategies")
            logger.info(f"   ✅ Top ROI: {top_roi['roi'].iloc[0]:.2%} to {top_roi['roi'].iloc[-1]:.2%}")
        
        # 2. Top 6 by Sharpe ratio (best risk-adjusted)
        if 'sharpe_ratio' in profitable.columns:
            sharpe_data = profitable.dropna(subset=['sharpe_ratio'])
            if len(sharpe_data) > 0:
                top_sharpe = sharpe_data.nlargest(6, 'sharpe_ratio')
                sharpe_indices = set(top_sharpe.index)
                selected_strategies.update(sharpe_indices)
                selection_details.append(f"Top 6 by Sharpe: {len(sharpe_indices)} strategies")
                logger.info(f"   ✅ Top Sharpe: {top_sharpe['sharpe_ratio'].iloc[0]:.3f} to {top_sharpe['sharpe_ratio'].iloc[-1]:.3f}")
        
        # 3. Top 5 robust strategies (ROI > 3% AND total_bets > 200)
        if 'total_bets' in profitable.columns:
            robust_filter = (profitable['roi'] > 0.03) & (profitable['total_bets'] > 200)
            robust_candidates = profitable[robust_filter]
            if len(robust_candidates) > 0:
                top_robust = robust_candidates.nlargest(5, 'roi')
                robust_indices = set(top_robust.index)
                selected_strategies.update(robust_indices)
                selection_details.append(f"Top 5 robust: {len(robust_indices)} strategies")
                logger.info(f"   ✅ Robust strategies: {len(robust_candidates)} candidates, selected {len(robust_indices)}")
        
        # 4. Top 4 by win rate (consistent performers)
        if 'win_rate' in profitable.columns:
            winrate_filter = (profitable['roi'] > 0) & (profitable.get('total_bets', 0) > 100)
            winrate_candidates = profitable[winrate_filter]
            if len(winrate_candidates) > 0:
                winrate_data = winrate_candidates.dropna(subset=['win_rate'])
                if len(winrate_data) > 0:
                    top_winrate = winrate_data.nlargest(4, 'win_rate')
                    winrate_indices = set(top_winrate.index)
                    selected_strategies.update(winrate_indices)
                    selection_details.append(f"Top 4 by win rate: {len(winrate_indices)} strategies")
                    logger.info(f"   ✅ Top win rates: {top_winrate['win_rate'].iloc[0]:.1%} to {top_winrate['win_rate'].iloc[-1]:.1%}")
        
        # 5. Dark horses (moderate ROI but very high sample size)
        if 'total_bets' in profitable.columns:
            dark_horse_filter = (profitable['roi'] > 0.01) & (profitable['total_bets'] > 500)
            dark_horses = profitable[dark_horse_filter]
            if len(dark_horses) > 0:
                # Select by highest sample size
                top_dark_horses = dark_horses.nlargest(3, 'total_bets')
                dark_horse_indices = set(top_dark_horses.index)
                selected_strategies.update(dark_horse_indices)
                selection_details.append(f"Top 3 dark horses: {len(dark_horse_indices)} strategies")
                logger.info(f"   ✅ Dark horses: {len(dark_horses)} candidates, selected {len(dark_horse_indices)}")
        
        # Get final selected strategies
        final_strategies = profitable.loc[list(selected_strategies)].copy()
        
        # If we have too few, add more by ROI
        if len(final_strategies) < self.target_strategies:
            remaining_needed = self.target_strategies - len(final_strategies)
            remaining_candidates = profitable[~profitable.index.isin(selected_strategies)]
            
            if len(remaining_candidates) > 0:
                additional = remaining_candidates.nlargest(remaining_needed, 'roi')
                final_strategies = pd.concat([final_strategies, additional])
                selection_details.append(f"Additional by ROI: {len(additional)} strategies")
                logger.info(f"   ✅ Added {len(additional)} additional strategies by ROI")
        
        # Sort by ROI for better presentation
        final_strategies = final_strategies.sort_values('roi', ascending=False)
        
        logger.info(f"🎯 Final selection: {len(final_strategies)} strategies")
        for detail in selection_details:
            logger.info(f"   📊 {detail}")
        
        # Summary statistics
        roi_stats = final_strategies['roi'].describe()
        logger.info(f"   📈 ROI range: {roi_stats['min']:.2%} to {roi_stats['max']:.2%}")
        logger.info(f"   📈 Mean ROI: {roi_stats['mean']:.2%}")
        
        return final_strategies
    
    def run_strategy_backtests(self, strategies: pd.DataFrame) -> Dict[str, Any]:
        """
        Run detailed backtests for all selected strategies
        
        Args:
            strategies: DataFrame with selected strategies
            
        Returns:
            Dictionary with all backtest results
        """
        logger.info(f"🚀 Running backtests for {len(strategies)} strategies...")
        
        all_results = {}
        all_bet_records = []
        all_daily_records = []
        all_monthly_records = []
        failed_strategies = []
        
        start_time = time.time()
        
        for i, (idx, strategy) in enumerate(strategies.iterrows(), 1):
            strategy_id = f"strategy_{idx}"
            
            try:
                logger.info(f"   [{i}/{len(strategies)}] Running {strategy_id}...")
                
                # Extract parameters
                params = {
                    'edge_threshold': strategy.get('edge_threshold', 0.05),
                    'min_odds': strategy.get('min_odds', 1.30),
                    'stake_method': strategy.get('stake_method', 'fixed'),
                    'stake_size': strategy.get('stake_size', 0.02),
                    'ev_method': strategy.get('ev_method', 'basic'),
                    'max_stake_pct': strategy.get('max_stake_pct', 0.1)
                }
                
                # Initialize fresh engine
                engine = BacktestingEngine(
                    elo_model_path=self.elo_model_path,
                    initial_bankroll=self.initial_bankroll
                )
                
                # Load data
                engine.load_backtesting_data(season='2025')
                
                # Run backtest
                result = engine.run_backtest(**params)
                
                # Store results
                all_results[strategy_id] = {
                    'strategy_index': idx,
                    'parameters': params,
                    'optimization_metrics': {
                        'opt_roi': strategy.get('roi', 0),
                        'opt_sharpe': strategy.get('sharpe_ratio', 0),
                        'opt_total_bets': strategy.get('total_bets', 0),
                        'opt_win_rate': strategy.get('win_rate', 0)
                    },
                    'backtest_results': result
                }
                
                # Collect bet records with strategy ID
                if 'bet_history' in result:
                    bet_history = result['bet_history']
                    if isinstance(bet_history, list) and len(bet_history) > 0:
                        # Convert list to DataFrame
                        bet_records = pd.DataFrame(bet_history)
                        bet_records['strategy_id'] = strategy_id
                        bet_records['strategy_index'] = idx
                        all_bet_records.append(bet_records)
                    elif isinstance(bet_history, pd.DataFrame):
                        # Already DataFrame
                        bet_records = bet_history.copy()
                        bet_records['strategy_id'] = strategy_id
                        bet_records['strategy_index'] = idx
                        all_bet_records.append(bet_records)
                
                # Collect daily performance records (financial metrics)
                if 'daily_performance' in result:
                    daily_performance = result['daily_performance']
                    if isinstance(daily_performance, list) and len(daily_performance) > 0:
                        # Convert list to DataFrame
                        daily_records = pd.DataFrame(daily_performance)
                        daily_records['strategy_id'] = strategy_id
                        daily_records['strategy_index'] = idx
                        all_daily_records.append(daily_records)
                    elif isinstance(daily_performance, pd.DataFrame):
                        # Already DataFrame
                        daily_records = daily_performance.copy()
                        daily_records['strategy_id'] = strategy_id
                        daily_records['strategy_index'] = idx
                        all_daily_records.append(daily_records)
                
                # Optionally collect gaming day results (game statistics) separately
                if 'gaming_day_results' in result:
                    gaming_day_results = result['gaming_day_results']
                    if isinstance(gaming_day_results, list) and len(gaming_day_results) > 0:
                        # Convert list to DataFrame
                        gaming_records = pd.DataFrame(gaming_day_results)
                        gaming_records['strategy_id'] = strategy_id
                        gaming_records['strategy_index'] = idx
                        # Store in separate list if needed for game analysis
                        # all_gaming_records.append(gaming_records)  # Optional
                
                # Calculate monthly aggregation
                if 'bet_history' in result:
                    bet_history = result['bet_history']
                    if isinstance(bet_history, list) and len(bet_history) > 0:
                        # Convert to DataFrame for monthly calculation
                        bet_df = pd.DataFrame(bet_history)
                        monthly_data = self._calculate_monthly_performance(
                            bet_df, strategy_id, idx
                        )
                        if monthly_data is not None:
                            all_monthly_records.append(monthly_data)
                    elif isinstance(bet_history, pd.DataFrame):
                        monthly_data = self._calculate_monthly_performance(
                            bet_history, strategy_id, idx
                        )
                        if monthly_data is not None:
                            all_monthly_records.append(monthly_data)
                
                # Progress update
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                eta = avg_time * (len(strategies) - i)
                
                logger.info(f"      ✅ Completed in {elapsed/i:.1f}s avg. ETA: {eta/60:.1f}min")
                
            except Exception as e:
                logger.error(f"      ❌ Failed {strategy_id}: {e}")
                failed_strategies.append((strategy_id, str(e)))
                continue  # Skip to next strategy, don't count as successful
        
        total_time = time.time() - start_time
        logger.info(f"🏁 Backtesting completed in {total_time/60:.1f} minutes")
        logger.info(f"   ✅ Successful: {len(all_results)}")
        logger.info(f"   ❌ Failed: {len(failed_strategies)}")
        
        # Combine all data
        combined_results = {
            'generation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_strategies': len(strategies),
                'successful_strategies': len(all_results),
                'failed_strategies': len(failed_strategies),
                'generation_time_minutes': total_time / 60,
                'target_strategies': self.target_strategies,
                'failed_strategy_details': failed_strategies,
                'success_rate': len(all_results) / len(strategies) if len(strategies) > 0 else 0
            },
            'strategy_results': all_results,
            'combined_data': {
                'all_bets': pd.concat(all_bet_records, ignore_index=True) if all_bet_records else pd.DataFrame(),
                'daily_performance': pd.concat(all_daily_records, ignore_index=True) if all_daily_records else pd.DataFrame(),
                'monthly_performance': pd.concat(all_monthly_records, ignore_index=True) if all_monthly_records else pd.DataFrame()
            }
        }
        
        # Log data collection summary
        logger.info(f"📊 Data collection summary:")
        logger.info(f"   Bet records: {len(combined_results['combined_data']['all_bets'])} rows")
        logger.info(f"   Daily performance: {len(combined_results['combined_data']['daily_performance'])} rows")
        logger.info(f"   Monthly performance: {len(combined_results['combined_data']['monthly_performance'])} rows")
        
        if not combined_results['combined_data']['daily_performance'].empty:
            daily_cols = list(combined_results['combined_data']['daily_performance'].columns)
            logger.info(f"   Daily performance columns: {daily_cols}")
        
        return combined_results
    
    def _calculate_monthly_performance(self, bet_history: pd.DataFrame, 
                                     strategy_id: str, strategy_index: int) -> Optional[pd.DataFrame]:
        """Calculate monthly performance aggregation"""
        try:
            if 'date' not in bet_history.columns:
                return None
            
            bet_history = bet_history.copy()
            bet_history['date'] = pd.to_datetime(bet_history['date'])
            bet_history['year_month'] = bet_history['date'].dt.to_period('M')
            
            monthly_agg = bet_history.groupby('year_month').agg({
                'net_result': ['sum', 'count', 'mean'],
                'stake': 'sum',
                'bet_won': ['sum', 'mean']
            }).reset_index()
            
            # Flatten column names
            monthly_agg.columns = [
                'year_month', 'total_pnl', 'total_bets', 'avg_pnl_per_bet',
                'total_staked', 'total_wins', 'win_rate'
            ]
            
            # Add strategy identification
            monthly_agg['strategy_id'] = strategy_id
            monthly_agg['strategy_index'] = strategy_index
            
            # Calculate monthly ROI
            monthly_agg['monthly_roi'] = monthly_agg['total_pnl'] / monthly_agg['total_staked']
            
            return monthly_agg
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate monthly performance for {strategy_id}: {e}")
            return None
    
    def calculate_strategy_correlations(self, combined_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate correlation matrix between strategies
        
        Args:
            combined_results: Combined results from all strategies
            
        Returns:
            Correlation matrix DataFrame
        """
        logger.info("📊 Calculating strategy correlations...")
        
        try:
            daily_data = combined_results['combined_data']['daily_performance']
            
            if daily_data.empty:
                logger.warning("⚠️ No daily data available for correlation calculation")
                return pd.DataFrame()
            
            # Check available columns
            logger.info(f"📋 Available columns in daily data: {list(daily_data.columns)}")
            
            # Determine which column to use for correlation
            value_column = None
            if 'daily_roi' in daily_data.columns:
                value_column = 'daily_roi'
                logger.info("📊 Using 'daily_roi' for correlation calculation")
            elif 'daily_return' in daily_data.columns:
                value_column = 'daily_return'
                logger.info("📊 Using 'daily_return' for correlation calculation")
            elif 'ending_bankroll' in daily_data.columns:
                # Calculate daily returns from bankroll changes
                daily_data = daily_data.sort_values(['strategy_id', 'date'])
                daily_data['daily_return_calc'] = daily_data.groupby('strategy_id')['ending_bankroll'].pct_change()
                value_column = 'daily_return_calc'
                logger.info("📊 Calculated daily returns from bankroll changes")
            else:
                logger.warning("⚠️ No suitable column found for correlation calculation")
                return pd.DataFrame()
            
            # Pivot to get strategy returns by date
            pivot_data = daily_data.pivot(
                index='date', 
                columns='strategy_id', 
                values=value_column
            )
            
            # Calculate correlation matrix
            correlation_matrix = pivot_data.corr()
            
            # Remove NaN strategies (strategies with insufficient data)
            correlation_matrix = correlation_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
            
            if correlation_matrix.empty:
                logger.warning("⚠️ Correlation matrix is empty after cleaning")
                return pd.DataFrame()
            
            logger.info(f"✅ Calculated correlations for {len(correlation_matrix)} strategies")
            
            # Summary statistics
            if len(correlation_matrix) > 1:
                # Get upper triangle values (excluding diagonal)
                mask = np.triu(np.ones_like(correlation_matrix.values, dtype=bool), k=1)
                upper_triangle_values = correlation_matrix.values[mask]
                
                if len(upper_triangle_values) > 0:
                    avg_correlation = np.mean(upper_triangle_values)
                    max_correlation = np.max(upper_triangle_values)
                    min_correlation = np.min(upper_triangle_values)
                    
                    logger.info(f"   📈 Average correlation: {avg_correlation:.3f}")
                    logger.info(f"   📈 Max correlation: {max_correlation:.3f}")
                    logger.info(f"   📈 Min correlation: {min_correlation:.3f}")
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate correlations: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def save_results(self, combined_results: Dict[str, Any], 
                    correlation_matrix: pd.DataFrame) -> Dict[str, str]:
        """
        Save all results to files for risk assessment
        
        Args:
            combined_results: All backtest results
            correlation_matrix: Strategy correlation matrix
            
        Returns:
            Dictionary with output file paths
        """
        logger.info("💾 Saving results...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_files = {}
        
        try:
            # 1. Individual bet records CSV
            bet_data = combined_results['combined_data']['all_bets']
            if not bet_data.empty and len(bet_data) > 0:
                bet_file = os.path.join(self.results_dir, f'backtest_top_strategies_bets_{timestamp}.csv')
                bet_data.to_csv(bet_file, index=False, encoding='utf-8')
                output_files['bet_records'] = bet_file
                logger.info(f"   ✅ Bet records: {os.path.basename(bet_file)} ({len(bet_data):,} records)")
            else:
                logger.warning("   ⚠️ No bet records to save")
            
            # 2. Daily financial performance CSV (for correlation analysis)
            daily_data = combined_results['combined_data']['daily_performance']
            if not daily_data.empty and len(daily_data) > 0:
                daily_file = os.path.join(self.results_dir, f'backtest_top_strategies_daily_financial_{timestamp}.csv')
                daily_data.to_csv(daily_file, index=False, encoding='utf-8')
                output_files['daily_financial'] = daily_file
                logger.info(f"   ✅ Daily financial performance: {os.path.basename(daily_file)} ({len(daily_data):,} records)")
                logger.info(f"       Columns: {list(daily_data.columns)}")
            else:
                logger.warning("   ⚠️ No daily financial performance data to save")
            
            # 3. Monthly performance CSV
            monthly_data = combined_results['combined_data']['monthly_performance']
            if not monthly_data.empty and len(monthly_data) > 0:
                monthly_file = os.path.join(self.results_dir, f'backtest_top_strategies_monthly_{timestamp}.csv')
                monthly_data.to_csv(monthly_file, index=False, encoding='utf-8')
                output_files['monthly_performance'] = monthly_file
                logger.info(f"   ✅ Monthly performance: {os.path.basename(monthly_file)} ({len(monthly_data):,} records)")
            else:
                logger.warning("   ⚠️ No monthly performance data to save")
            
            # 4. Strategy correlation matrix CSV
            if not correlation_matrix.empty:
                corr_file = os.path.join(self.results_dir, f'backtest_strategy_correlation_{timestamp}.csv')
                correlation_matrix.to_csv(corr_file, encoding='utf-8')
                output_files['correlation_matrix'] = corr_file
                logger.info(f"   ✅ Correlation matrix: {os.path.basename(corr_file)} ({correlation_matrix.shape[0]}x{correlation_matrix.shape[1]})")
            
            # 5. Summary metadata JSON
            summary_data = {
                'generation_metadata': combined_results['generation_metadata'],
                'strategy_summary': {},
                'output_files': output_files,
                'correlation_summary': {}
            }
            
            # Add strategy summary
            for strategy_id, strategy_data in combined_results['strategy_results'].items():
                backtest_perf = strategy_data['backtest_results']['performance']
                opt_metrics = strategy_data['optimization_metrics']
                
                summary_data['strategy_summary'][strategy_id] = {
                    'parameters': strategy_data['parameters'],
                    'optimization_roi': opt_metrics.get('opt_roi', 0),
                    'backtest_roi': backtest_perf.get('roi', 0),
                    'total_bets': backtest_perf.get('total_bets', 0),
                    'win_rate': backtest_perf.get('win_rate', 0),
                    'max_drawdown': backtest_perf.get('max_drawdown', 0),
                    'sharpe_ratio': backtest_perf.get('sharpe_ratio', 0)
                }
            
            # Add correlation summary
            if not correlation_matrix.empty:
                # Get upper triangle values (excluding diagonal)
                mask = np.triu(np.ones_like(correlation_matrix.values, dtype=bool), k=1)
                upper_triangle_values = correlation_matrix.values[mask]
                
                if len(upper_triangle_values) > 0:
                    summary_data['correlation_summary'] = {
                        'average_correlation': float(np.mean(upper_triangle_values)),
                        'max_correlation': float(np.max(upper_triangle_values)),
                        'min_correlation': float(np.min(upper_triangle_values)),
                        'std_correlation': float(np.std(upper_triangle_values)),
                        'strategies_count': correlation_matrix.shape[0]
                    }
                else:
                    summary_data['correlation_summary'] = {
                        'note': 'Insufficient data for correlation calculation'
                    }
            else:
                summary_data['correlation_summary'] = {
                    'note': 'No correlation data available'
                }
            
            summary_file = os.path.join(self.results_dir, f'backtest_top_strategies_summary_{timestamp}.json')
            
            # Convert for JSON serialization
            summary_json = self._convert_for_json(summary_data)
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_json, f, indent=2, default=str)
            
            output_files['summary'] = summary_file
            logger.info(f"   ✅ Summary: {os.path.basename(summary_file)}")
            
            logger.info(f"💾 All results saved successfully!")
            return output_files
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            raise
    
    def _convert_for_json(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
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
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'to_timestamp'):  # pandas Period
            return str(obj)
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return obj
    
    def _convert_key_for_json(self, key):
        """Convert dictionary keys to JSON-serializable format"""
        if isinstance(key, (str, int, float, bool)):
            return key
        elif hasattr(key, 'to_timestamp'):  # pandas Period
            return str(key)
        elif hasattr(key, 'isoformat'):  # datetime objects
            return key.isoformat()
        else:
            return str(key)
    
    def run_full_generation(self) -> Dict[str, str]:
        """
        Run complete top strategy generation process
        
        Returns:
            Dictionary with output file paths
        """
        logger.info("🚀 Starting top strategy generation process...")
        
        try:
            # 1. Load optimization results
            optimization_df = self.load_optimization_results()
            
            # 2. Select top strategies
            selected_strategies = self.select_top_strategies(optimization_df)
            
            # 3. Run backtests
            combined_results = self.run_strategy_backtests(selected_strategies)
            
            # 4. Calculate correlations
            correlation_matrix = self.calculate_strategy_correlations(combined_results)
            
            # 5. Save results
            output_files = self.save_results(combined_results, correlation_matrix)
            
            logger.info("🎉 Top strategy generation completed successfully!")
            
            if output_files:
                logger.info("📁 Generated files for risk_assessment.ipynb:")
                for file_type, file_path in output_files.items():
                    logger.info(f"   📄 {file_type}: {os.path.basename(file_path)}")
            
            # Log correlation status
            if not correlation_matrix.empty:
                logger.info(f"✅ Strategy correlations calculated successfully ({correlation_matrix.shape[0]} strategies)")
            else:
                logger.warning("⚠️ Strategy correlations could not be calculated - check daily financial data")
            
            return output_files
            
        except Exception as e:
            logger.error(f"❌ Top strategy generation failed: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    logger.info("🏒 Starting Top Strategy Generation...")
    
    try:
        # Initialize generator
        generator = TopStrategyGenerator(
            elo_model_path='models/elo_model_trained_2024.pkl',
            initial_bankroll=10000.0,
            results_dir='models/experiments',
            target_strategies=23  # 20-25 strategies
        )
        
        # Run full generation process
        output_files = generator.run_full_generation()
        
        logger.info("✅ Top Strategy Generation completed successfully!")
        logger.info("🎯 Ready for risk_assessment.ipynb analysis!")
        
        # Next steps guidance
        logger.info("\n📚 NEXT STEPS:")
        logger.info("1. Run risk_assessment.ipynb notebook")
        logger.info("2. Analyze portfolio correlations and drawdowns")
        logger.info("3. Perform Monte Carlo risk simulations")
        logger.info("4. Implement risk management guidelines")
        
    except Exception as e:
        logger.error(f"❌ Top Strategy Generation failed: {e}")
        raise