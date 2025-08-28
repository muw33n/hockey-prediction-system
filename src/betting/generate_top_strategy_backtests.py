#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Generate Top Strategy Backtests (MIGRATED)
=====================================================================
Comprehensive backtest generation for top performing strategies with enhanced infrastructure.

Features:
- Multi-criteria strategy selection (20-25 strategies)
- Detailed backtesting for each selected strategy
- Multiple output formats for risk assessment
- Enhanced logging with per-component files
- Safe file handling with automatic encoding detection
- Performance monitoring for long operations
- Centralized path management

Location: src/betting/generate_top_strategy_backtests.py
Component: betting -> logs/betting.log
"""

import pandas as pd
import numpy as np
import json
import glob
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Set

# === MIGRACE: Enhanced imports ===
from config.paths import PATHS
from config.logging_config import (
    setup_logging, get_component_logger, 
    PerformanceLogger, LoggingConfig
)
from src.utils.file_handlers import (
    read_csv, write_csv, read_json, write_json,
    save_processed_data, FileHandler
)

# === MIGRACE: Setup enhanced logging (jednou na začátku aplikace) ===
setup_logging(
    log_level='INFO',
    log_to_file=True,
    component_files=True  # Per-component log files
)

# === MIGRACE: Component-specific logger pro betting ===
logger = get_component_logger(__name__, 'betting')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import backtesting engine (package setup handles imports now)
from backtesting_engine import BacktestingEngine


class TopStrategyGenerator:
    """
    Enhanced Top Strategy Backtest Generator
    
    Selects top performing strategies from optimization results and generates
    detailed backtests for comprehensive risk analysis with enhanced infrastructure.
    """
    
    def __init__(self, 
                 elo_model_name: str = 'elo_model_trained_2024',
                 initial_bankroll: float = 10000.0,
                 target_strategies: int = 23):
        """
        Initialize Enhanced Top Strategy Generator
        
        Args:
            elo_model_name: Name of trained Elo model (without .pkl extension)
            initial_bankroll: Starting bankroll for each simulation
            target_strategies: Target number of strategies to generate (20-25)
        """
        # === MIGRACE: Store model name for BacktestingEngine ===
        self.elo_model_name = elo_model_name
        self.elo_model_path = PATHS.get_model_file(elo_model_name, 'pkl')
        self.initial_bankroll = initial_bankroll
        self.target_strategies = target_strategies
        
        # === MIGRACE: Automatické adresáře pomocí PATHS ===
        PATHS.ensure_directories()
        
        # === MIGRACE: Performance logger pro monitoring ===
        self.perf_logger = PerformanceLogger(logger)
        
        logger.info("🎯 Enhanced TopStrategyGenerator initialized")
        logger.info(f"   Target strategies: {target_strategies}")
        logger.info(f"   Model path: {self.elo_model_path.name}")
        logger.info(f"   Results directory: {PATHS.experiments}")
        logger.info(f"   Initial bankroll: ${initial_bankroll:,.0f}")
        
        # Validate model existence
        if not self.elo_model_path.exists():
            logger.warning(f"⚠️ Model file not found: {self.elo_model_path}")
            logger.info(f"   Available models: {[f.name for f in PATHS.trained_models.glob('*.pkl')]}")
    
    def load_optimization_results(self) -> pd.DataFrame:
        """
        Load latest optimization results with enhanced file handling
        
        Returns:
            DataFrame with optimization results
        """
        logger.info("📊 Loading optimization results with safe file handling...")
        
        # === MIGRACE: Performance tracking ===
        self.perf_logger.start_timer('load_optimization_results')
        
        try:
            # === MIGRACE: Použití PATHS místo hardcoded paths ===
            # Try CSV detailed first (preferred)
            csv_pattern = str(PATHS.experiments / '*optimization*detailed*.csv')
            csv_files = glob.glob(csv_pattern)
            
            if csv_files:
                # Get latest CSV file
                latest_csv = max(csv_files, key=lambda x: PATHS.root.joinpath(x).stat().st_mtime)
                logger.info(f"📄 Loading latest CSV: {PATHS.get_relative_path(latest_csv)}")
                
                try:
                    # === MIGRACE: Safe CSV loading s automatic encoding detection ===
                    df = read_csv(latest_csv)
                    logger.info(f"✅ Loaded {len(df)} optimization records from CSV")
                    logger.info(f"   Columns: {list(df.columns)[:5]}...")
                    
                    self.perf_logger.end_timer('load_optimization_results')
                    return df
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load CSV with safe handler: {e}")
            
            # Fallback to JSON
            json_pattern = str(PATHS.experiments / '*optimization*results*.json')
            json_files = glob.glob(json_pattern)
            
            if json_files:
                latest_json = max(json_files, key=lambda x: PATHS.root.joinpath(x).stat().st_mtime)
                logger.info(f"📄 Loading latest JSON: {PATHS.get_relative_path(latest_json)}")
                
                try:
                    # === MIGRACE: Safe JSON loading ===
                    data = read_json(latest_json)
                    
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
                            
                            self.perf_logger.end_timer('load_optimization_results')
                            return df
                            
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load JSON with safe handler: {e}")
            
            # No files found
            search_dirs = [PATHS.experiments, PATHS.models]
            logger.error(f"❌ No optimization results found in: {[str(d) for d in search_dirs]}")
            
            # List available files for debugging
            for search_dir in search_dirs:
                if search_dir.exists():
                    available_files = list(search_dir.glob('*optimization*'))
                    logger.info(f"   Available in {search_dir.name}: {[f.name for f in available_files]}")
            
            raise FileNotFoundError(f"No optimization results found in {PATHS.experiments}")
            
        except Exception as e:
            self.perf_logger.end_timer('load_optimization_results')
            logger.error(f"❌ Failed to load optimization results: {e}")
            raise
    
    def select_top_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select top strategies using multi-criteria approach with enhanced logging
        
        Args:
            df: DataFrame with optimization results
            
        Returns:
            DataFrame with selected top strategies
        """
        logger.info(f"🎯 Selecting top {self.target_strategies} strategies with multi-criteria approach...")
        
        # === MIGRACE: Performance tracking ===
        self.perf_logger.start_timer('strategy_selection')
        
        try:
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
                
                roi_range = f"{top_roi['roi'].iloc[0]:.2%} to {top_roi['roi'].iloc[-1]:.2%}"
                logger.info(f"   ✅ Top ROI range: {roi_range}")
            
            # 2. Top 6 by Sharpe ratio (best risk-adjusted)
            if 'sharpe_ratio' in profitable.columns:
                sharpe_data = profitable.dropna(subset=['sharpe_ratio'])
                if len(sharpe_data) > 0:
                    top_sharpe = sharpe_data.nlargest(6, 'sharpe_ratio')
                    sharpe_indices = set(top_sharpe.index)
                    selected_strategies.update(sharpe_indices)
                    selection_details.append(f"Top 6 by Sharpe: {len(sharpe_indices)} strategies")
                    
                    sharpe_range = f"{top_sharpe['sharpe_ratio'].iloc[0]:.3f} to {top_sharpe['sharpe_ratio'].iloc[-1]:.3f}"
                    logger.info(f"   ✅ Top Sharpe range: {sharpe_range}")
            
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
                        
                        winrate_range = f"{top_winrate['win_rate'].iloc[0]:.1%} to {top_winrate['win_rate'].iloc[-1]:.1%}"
                        logger.info(f"   ✅ Top win rate range: {winrate_range}")
            
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
            
            # === MIGRACE: Detailed logging summary ===
            logger.info(f"🎯 Final selection: {len(final_strategies)} strategies")
            for detail in selection_details:
                logger.info(f"   📊 {detail}")
            
            # Summary statistics
            if len(final_strategies) > 0:
                roi_stats = final_strategies['roi'].describe()
                logger.info(f"   📈 ROI range: {roi_stats['min']:.2%} to {roi_stats['max']:.2%}")
                logger.info(f"   📈 Mean ROI: {roi_stats['mean']:.2%}")
                logger.info(f"   📈 Median ROI: {roi_stats['50%']:.2%}")
                
                if 'total_bets' in final_strategies.columns:
                    bet_stats = final_strategies['total_bets'].describe()
                    logger.info(f"   🎲 Sample sizes: {int(bet_stats['min'])} to {int(bet_stats['max'])} bets")
            
            self.perf_logger.end_timer('strategy_selection')
            return final_strategies
            
        except Exception as e:
            self.perf_logger.end_timer('strategy_selection')
            logger.error(f"❌ Strategy selection failed: {e}")
            raise
    
    def run_strategy_backtests(self, strategies: pd.DataFrame) -> Dict[str, Any]:
        """
        Run detailed backtests for all selected strategies with enhanced monitoring
        
        Args:
            strategies: DataFrame with selected strategies
            
        Returns:
            Dictionary with all backtest results
        """
        logger.info(f"🚀 Running enhanced backtests for {len(strategies)} strategies...")
        
        # === MIGRACE: Performance tracking ===
        self.perf_logger.start_timer('full_backtesting')
        
        all_results = {}
        all_bet_records = []
        all_daily_records = []
        all_monthly_records = []
        failed_strategies = []
        
        start_time = time.time()
        
        for i, (idx, strategy) in enumerate(strategies.iterrows(), 1):
            strategy_id = f"strategy_{idx}"
            
            # === MIGRACE: Per-strategy performance tracking ===
            self.perf_logger.start_timer(f'backtest_{strategy_id}')
            
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
                
                logger.debug(f"      Parameters: {params}")
                
                # Initialize fresh engine
                engine = BacktestingEngine(
                        elo_model_name=self.elo_model_name,
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
                
                # === MIGRACE: Enhanced data collection ===
                self._collect_strategy_data(
                    result, strategy_id, idx,
                    all_bet_records, all_daily_records, all_monthly_records
                )
                
                # Progress update with ETA
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                eta = avg_time * (len(strategies) - i)
                
                self.perf_logger.end_timer(f'backtest_{strategy_id}')
                logger.info(f"      ✅ Completed in {avg_time:.1f}s avg. ETA: {eta/60:.1f}min")
                
            except Exception as e:
                self.perf_logger.end_timer(f'backtest_{strategy_id}')
                logger.error(f"      ❌ Failed {strategy_id}: {e}")
                failed_strategies.append((strategy_id, str(e)))
                continue
        
        total_time = time.time() - start_time
        self.perf_logger.end_timer('full_backtesting')
        
        logger.info(f"🏁 Enhanced backtesting completed in {total_time/60:.1f} minutes")
        logger.info(f"   ✅ Successful: {len(all_results)}")
        logger.info(f"   ❌ Failed: {len(failed_strategies)}")
        
        # === MIGRACE: Enhanced combined results ===
        combined_results = {
            'generation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_strategies': len(strategies),
                'successful_strategies': len(all_results),
                'failed_strategies': len(failed_strategies),
                'generation_time_minutes': total_time / 60,
                'target_strategies': self.target_strategies,
                'failed_strategy_details': failed_strategies,
                'success_rate': len(all_results) / len(strategies) if len(strategies) > 0 else 0,
                'enhanced_infrastructure': True,
                'model_path': str(self.elo_model_path),
                'initial_bankroll': self.initial_bankroll
            },
            'strategy_results': all_results,
            'combined_data': {
                'all_bets': pd.concat(all_bet_records, ignore_index=True) if all_bet_records else pd.DataFrame(),
                'daily_performance': pd.concat(all_daily_records, ignore_index=True) if all_daily_records else pd.DataFrame(),
                'monthly_performance': pd.concat(all_monthly_records, ignore_index=True) if all_monthly_records else pd.DataFrame()
            }
        }
        
        # Enhanced data collection summary
        logger.info(f"📊 Enhanced data collection summary:")
        logger.info(f"   Bet records: {len(combined_results['combined_data']['all_bets']):,} rows")
        logger.info(f"   Daily performance: {len(combined_results['combined_data']['daily_performance']):,} rows")
        logger.info(f"   Monthly performance: {len(combined_results['combined_data']['monthly_performance']):,} rows")
        
        if not combined_results['combined_data']['daily_performance'].empty:
            daily_cols = list(combined_results['combined_data']['daily_performance'].columns)
            logger.info(f"   Daily performance columns: {daily_cols[:5]}...")
        
        return combined_results
    
    def _collect_strategy_data(self, result: Dict[str, Any], strategy_id: str, strategy_index: int,
                              all_bet_records: List, all_daily_records: List, all_monthly_records: List) -> None:
        """
        Enhanced data collection from backtest results
        
        Args:
            result: Backtest result dictionary
            strategy_id: Strategy identifier
            strategy_index: Strategy index
            all_bet_records: List to collect bet records
            all_daily_records: List to collect daily records
            all_monthly_records: List to collect monthly records
        """
        try:
            # Collect bet records with strategy ID
            if 'bet_history' in result:
                bet_history = result['bet_history']
                if isinstance(bet_history, list) and len(bet_history) > 0:
                    bet_records = pd.DataFrame(bet_history)
                elif isinstance(bet_history, pd.DataFrame) and not bet_history.empty:
                    bet_records = bet_history.copy()
                else:
                    return
                
                bet_records['strategy_id'] = strategy_id
                bet_records['strategy_index'] = strategy_index
                all_bet_records.append(bet_records)
            
            # Collect daily performance records
            if 'daily_performance' in result:
                daily_performance = result['daily_performance']
                if isinstance(daily_performance, list) and len(daily_performance) > 0:
                    daily_records = pd.DataFrame(daily_performance)
                elif isinstance(daily_performance, pd.DataFrame) and not daily_performance.empty:
                    daily_records = daily_performance.copy()
                else:
                    daily_records = None
                
                if daily_records is not None:
                    daily_records['strategy_id'] = strategy_id
                    daily_records['strategy_index'] = strategy_index
                    all_daily_records.append(daily_records)
            
            # Calculate monthly aggregation
            if 'bet_history' in result and result['bet_history']:
                bet_history = result['bet_history']
                if isinstance(bet_history, list):
                    bet_df = pd.DataFrame(bet_history)
                elif isinstance(bet_history, pd.DataFrame):
                    bet_df = bet_history
                else:
                    return
                
                monthly_data = self._calculate_monthly_performance(bet_df, strategy_id, strategy_index)
                if monthly_data is not None:
                    all_monthly_records.append(monthly_data)
                    
        except Exception as e:
            logger.warning(f"⚠️ Failed to collect data for {strategy_id}: {e}")
    
    def _calculate_monthly_performance(self, bet_history: pd.DataFrame, 
                                     strategy_id: str, strategy_index: int) -> Optional[pd.DataFrame]:
        """
        Calculate monthly performance aggregation with enhanced error handling
        
        Args:
            bet_history: DataFrame with bet history
            strategy_id: Strategy identifier
            strategy_index: Strategy index
            
        Returns:
            DataFrame with monthly performance or None if calculation fails
        """
        try:
            if 'date' not in bet_history.columns or bet_history.empty:
                return None
            
            bet_history = bet_history.copy()
            bet_history['date'] = pd.to_datetime(bet_history['date'])
            bet_history['year_month'] = bet_history['date'].dt.to_period('M')
            
            # Enhanced aggregation
            agg_functions = {
                'net_result': ['sum', 'count', 'mean'],
                'stake': 'sum'
            }
            
            # Add bet_won if available
            if 'bet_won' in bet_history.columns:
                agg_functions['bet_won'] = ['sum', 'mean']
            
            monthly_agg = bet_history.groupby('year_month').agg(agg_functions).reset_index()
            
            # Flatten column names
            new_columns = ['year_month', 'total_pnl', 'total_bets', 'avg_pnl_per_bet', 'total_staked']
            if 'bet_won' in agg_functions:
                new_columns.extend(['total_wins', 'win_rate'])
            
            monthly_agg.columns = new_columns[:len(monthly_agg.columns)]
            
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
        Calculate correlation matrix between strategies with enhanced error handling
        
        Args:
            combined_results: Combined results from all strategies
            
        Returns:
            Correlation matrix DataFrame
        """
        logger.info("📊 Calculating strategy correlations with enhanced analysis...")
        
        # === MIGRACE: Performance tracking ===
        self.perf_logger.start_timer('correlation_calculation')
        
        try:
            daily_data = combined_results['combined_data']['daily_performance']
            
            if daily_data.empty:
                logger.warning("⚠️ No daily data available for correlation calculation")
                self.perf_logger.end_timer('correlation_calculation')
                return pd.DataFrame()
            
            # Enhanced column detection
            logger.info(f"📋 Available columns in daily data: {list(daily_data.columns)}")
            
            # Determine which column to use for correlation
            value_column = None
            column_priority = ['daily_roi', 'daily_return', 'ending_bankroll']
            
            for col in column_priority:
                if col in daily_data.columns:
                    value_column = col
                    logger.info(f"📊 Using '{col}' for correlation calculation")
                    break
            
            if value_column is None:
                logger.warning("⚠️ No suitable column found for correlation calculation")
                self.perf_logger.end_timer('correlation_calculation')
                return pd.DataFrame()
            
            # Handle ending_bankroll case (calculate returns)
            if value_column == 'ending_bankroll':
                daily_data = daily_data.sort_values(['strategy_id', 'date'])
                daily_data['daily_return_calc'] = daily_data.groupby('strategy_id')['ending_bankroll'].pct_change()
                value_column = 'daily_return_calc'
                logger.info("📊 Calculated daily returns from bankroll changes")
            
            # Pivot to get strategy returns by date
            pivot_data = daily_data.pivot(
                index='date', 
                columns='strategy_id', 
                values=value_column
            )
            
            logger.info(f"📊 Pivot data shape: {pivot_data.shape}")
            
            # Calculate correlation matrix
            correlation_matrix = pivot_data.corr()
            
            # Clean correlation matrix
            correlation_matrix = correlation_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
            
            if correlation_matrix.empty:
                logger.warning("⚠️ Correlation matrix is empty after cleaning")
                self.perf_logger.end_timer('correlation_calculation')
                return pd.DataFrame()
            
            logger.info(f"✅ Calculated correlations for {len(correlation_matrix)} strategies")
            
            # Enhanced summary statistics
            if len(correlation_matrix) > 1:
                mask = np.triu(np.ones_like(correlation_matrix.values, dtype=bool), k=1)
                upper_triangle_values = correlation_matrix.values[mask]
                
                if len(upper_triangle_values) > 0:
                    correlation_stats = {
                        'mean': float(np.mean(upper_triangle_values)),
                        'median': float(np.median(upper_triangle_values)),
                        'std': float(np.std(upper_triangle_values)),
                        'min': float(np.min(upper_triangle_values)),
                        'max': float(np.max(upper_triangle_values))
                    }
                    
                    logger.info(f"   📈 Average correlation: {correlation_stats['mean']:.3f}")
                    logger.info(f"   📈 Median correlation: {correlation_stats['median']:.3f}")
                    logger.info(f"   📈 Correlation range: {correlation_stats['min']:.3f} to {correlation_stats['max']:.3f}")
                    logger.info(f"   📈 Correlation std: {correlation_stats['std']:.3f}")
            
            self.perf_logger.end_timer('correlation_calculation')
            return correlation_matrix
            
        except Exception as e:
            self.perf_logger.end_timer('correlation_calculation')
            logger.error(f"❌ Failed to calculate correlations: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def save_results(self, combined_results: Dict[str, Any], 
                    correlation_matrix: pd.DataFrame) -> Dict[str, str]:
        """
        Save all results to files using enhanced file handling
        
        Args:
            combined_results: All backtest results
            correlation_matrix: Strategy correlation matrix
            
        Returns:
            Dictionary with output file paths
        """
        logger.info("💾 Saving results with enhanced file handling...")
        
        # === MIGRACE: Performance tracking ===
        self.perf_logger.start_timer('save_results')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_files = {}
        
        try:
            # === MIGRACE: Použití PATHS pro output files ===
            
            # 1. Individual bet records CSV
            bet_data = combined_results['combined_data']['all_bets']
            if not bet_data.empty and len(bet_data) > 0:
                bet_filename = f'backtest_top_strategies_bets_{timestamp}'
                bet_file = save_processed_data(bet_data, bet_filename, index=False)
                output_files['bet_records'] = str(bet_file)
                logger.info(f"   ✅ Bet records: {bet_file.name} ({len(bet_data):,} records)")
            else:
                logger.warning("   ⚠️ No bet records to save")
            
            # 2. Daily financial performance CSV
            daily_data = combined_results['combined_data']['daily_performance']
            if not daily_data.empty and len(daily_data) > 0:
                daily_filename = f'backtest_top_strategies_daily_financial_{timestamp}'
                daily_file = save_processed_data(daily_data, daily_filename, index=False)
                output_files['daily_financial'] = str(daily_file)
                logger.info(f"   ✅ Daily performance: {daily_file.name} ({len(daily_data):,} records)")
                logger.info(f"       Columns: {list(daily_data.columns)[:5]}...")
            else:
                logger.warning("   ⚠️ No daily financial performance data to save")
            
            # 3. Monthly performance CSV
            monthly_data = combined_results['combined_data']['monthly_performance']
            if not monthly_data.empty and len(monthly_data) > 0:
                monthly_filename = f'backtest_top_strategies_monthly_{timestamp}'
                monthly_file = save_processed_data(monthly_data, monthly_filename, index=False)
                output_files['monthly_performance'] = str(monthly_file)
                logger.info(f"   ✅ Monthly performance: {monthly_file.name} ({len(monthly_data):,} records)")
            else:
                logger.warning("   ⚠️ No monthly performance data to save")
            
            # 4. Strategy correlation matrix CSV
            if not correlation_matrix.empty:
                corr_file = PATHS.processed_data / f'backtest_strategy_correlation_{timestamp}.csv'
                # === MIGRACE: Safe CSV writing s UTF-8 ===
                write_csv(correlation_matrix, corr_file, index=True)
                output_files['correlation_matrix'] = str(corr_file)
                logger.info(f"   ✅ Correlation matrix: {corr_file.name} ({correlation_matrix.shape[0]}x{correlation_matrix.shape[1]})")
            else:
                logger.warning("   ⚠️ No correlation matrix to save")
            
            # 5. Enhanced summary metadata JSON
            summary_data = self._create_summary_data(combined_results, correlation_matrix, output_files)
            
            summary_file = PATHS.processed_data / f'backtest_top_strategies_summary_{timestamp}.json'
            # === MIGRACE: Safe JSON writing ===
            write_json(summary_data, summary_file, indent=2)
            output_files['summary'] = str(summary_file)
            logger.info(f"   ✅ Enhanced summary: {summary_file.name}")
            
            self.perf_logger.end_timer('save_results')
            logger.info(f"💾 All results saved successfully using enhanced infrastructure!")
            return output_files
            
        except Exception as e:
            self.perf_logger.end_timer('save_results')
            logger.error(f"❌ Failed to save results: {e}")
            raise
    
    def _create_summary_data(self, combined_results: Dict[str, Any], 
                           correlation_matrix: pd.DataFrame, 
                           output_files: Dict[str, str]) -> Dict[str, Any]:
        """
        Create enhanced summary data for JSON export
        
        Args:
            combined_results: All backtest results
            correlation_matrix: Strategy correlation matrix
            output_files: Dictionary with output file paths
            
        Returns:
            Dictionary with summary data
        """
        summary_data = {
            'generation_metadata': combined_results['generation_metadata'],
            'strategy_summary': {},
            'output_files': output_files,
            'correlation_summary': {},
            'enhanced_features': {
                'component_logging': True,
                'safe_file_handling': True,
                'centralized_paths': True,
                'performance_tracking': True,
                'automatic_encoding_detection': True
            }
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
        
        # Enhanced correlation summary
        if not correlation_matrix.empty:
            mask = np.triu(np.ones_like(correlation_matrix.values, dtype=bool), k=1)
            upper_triangle_values = correlation_matrix.values[mask]
            
            if len(upper_triangle_values) > 0:
                summary_data['correlation_summary'] = {
                    'average_correlation': float(np.mean(upper_triangle_values)),
                    'median_correlation': float(np.median(upper_triangle_values)),
                    'max_correlation': float(np.max(upper_triangle_values)),
                    'min_correlation': float(np.min(upper_triangle_values)),
                    'std_correlation': float(np.std(upper_triangle_values)),
                    'strategies_count': correlation_matrix.shape[0],
                    'correlation_pairs': int(len(upper_triangle_values))
                }
            else:
                summary_data['correlation_summary'] = {
                    'note': 'Insufficient data for correlation calculation'
                }
        else:
            summary_data['correlation_summary'] = {
                'note': 'No correlation data available'
            }
        
        return summary_data
    
    def run_full_generation(self) -> Dict[str, str]:
        """
        Run complete enhanced top strategy generation process
        
        Returns:
            Dictionary with output file paths
        """
        logger.info("🚀 Starting enhanced top strategy generation process...")
        logger.info(f"   Component logging: logs/betting.log")
        logger.info(f"   Safe file handling: Enabled")
        logger.info(f"   Centralized paths: {PATHS.root}")
        
        # === MIGRACE: Overall performance tracking ===
        self.perf_logger.start_timer('full_generation')
        
        try:
            # 1. Load optimization results
            logger.info("📊 Step 1/5: Loading optimization results...")
            optimization_df = self.load_optimization_results()
            
            # 2. Select top strategies
            logger.info("🎯 Step 2/5: Selecting top strategies...")
            selected_strategies = self.select_top_strategies(optimization_df)
            
            # 3. Run backtests
            logger.info("🚀 Step 3/5: Running backtests...")
            combined_results = self.run_strategy_backtests(selected_strategies)
            
            # 4. Calculate correlations
            logger.info("📊 Step 4/5: Calculating correlations...")
            correlation_matrix = self.calculate_strategy_correlations(combined_results)
            
            # 5. Save results
            logger.info("💾 Step 5/5: Saving results...")
            output_files = self.save_results(combined_results, correlation_matrix)
            
            self.perf_logger.end_timer('full_generation')
            
            logger.info("🎉 Enhanced top strategy generation completed successfully!")
            logger.info("=" * 60)
            
            if output_files:
                logger.info("📁 Generated files for risk_assessment.ipynb:")
                for file_type, file_path in output_files.items():
                    relative_path = PATHS.get_relative_path(file_path)
                    logger.info(f"   📄 {file_type}: {relative_path}")
            
            # Enhanced correlation status
            if not correlation_matrix.empty:
                logger.info(f"✅ Strategy correlations calculated successfully")
                logger.info(f"   Matrix size: {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}")
            else:
                logger.warning("⚠️ Strategy correlations could not be calculated")
            
            # Performance summary
            total_time = self.perf_logger.timers.get('full_generation', 0)
            if total_time:
                logger.info(f"⏱️  Total processing time: {total_time:.1f} minutes")
            
            logger.info("=" * 60)
            
            return output_files
            
        except Exception as e:
            self.perf_logger.end_timer('full_generation')
            logger.error(f"❌ Enhanced top strategy generation failed: {e}")
            raise


# === MIGRACE: Enhanced main execution ===
def main():
    """Enhanced main function with proper error handling"""
    
    logger.info("🏒 Starting Enhanced Top Strategy Generation...")
    logger.info(f"   Enhanced infrastructure: Active")
    logger.info(f"   Component: betting -> logs/betting.log")
    logger.info(f"   Safe file handling: Enabled")
    logger.info(f"   Performance tracking: Enabled")
    
    try:
        # === MIGRACE: Použití PATHS místo hardcoded values ===
        generator = TopStrategyGenerator(
            elo_model_name='elo_model_trained_2024',  # Name only, PATHS handles path
            initial_bankroll=10000.0,
            target_strategies=23  # 20-25 strategies
        )
        
        # Run full generation process
        output_files = generator.run_full_generation()
        
        logger.info("✅ Enhanced Top Strategy Generation completed successfully!")
        logger.info("🎯 Ready for risk_assessment.ipynb analysis!")
        
        # Enhanced next steps guidance
        logger.info("")
        logger.info("📚 NEXT STEPS:")
        logger.info("1. Open notebooks/analysis/risk_assessment.ipynb")
        logger.info("2. Load generated files for portfolio analysis")
        logger.info("3. Analyze strategy correlations and drawdowns")
        logger.info("4. Perform Monte Carlo risk simulations")
        logger.info("5. Implement risk management guidelines")
        logger.info("")
        logger.info("🔧 ENHANCED FEATURES USED:")
        logger.info(f"   • Component logging: logs/betting.log")
        logger.info(f"   • Safe file handling: UTF-8, automatic encoding detection")
        logger.info(f"   • Centralized paths: {PATHS.processed_data}")
        logger.info(f"   • Performance tracking: Per-operation timing")
        
    except Exception as e:
        logger.error(f"❌ Enhanced Top Strategy Generation failed: {e}")
        logger.error("💡 Check logs/betting.log for detailed error information")
        raise


if __name__ == "__main__":
    main()
