#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Parameter Optimizer (MIGRATED to Enhanced Infrastructure)
==============================================================
Comprehensive grid search optimization for backtesting parameters

MIGRATION ENHANCEMENTS:
- Per-component logging (logs/betting.log)
- Centralized path management via PATHS
- Safe file handling with automatic encoding detection
- Performance monitoring for long-running operations
- Clean imports without sys.path manipulation

Features:
- Multi-dimensional parameter grid search
- Multiple optimization targets (ROI, Sharpe, drawdown)
- Parallel processing for performance
- Robust results analysis and ranking
- Comprehensive reporting and visualization

Location: src/betting/parameter_optimizer.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# === MIGRATION: Enhanced infrastructure imports ===
from config.paths import PATHS
from config.logging_config import setup_logging, get_component_logger, PerformanceLogger
from src.utils.file_handlers import (
    write_json, write_csv, read_json,
    FileHandler
)

# === MIGRATION: Setup enhanced logging (component-specific) ===
# Setup pouze jednou pro celou aplikaci
setup_logging(
    log_level='INFO',
    log_to_file=True,
    component_files=True  # Per-component log files
)

# Component-specific logger pro betting operations
logger = get_component_logger(__name__, 'betting')

# === MIGRATION: Clean import of backtesting engine ===
try:
    from src.betting.backtesting_engine import BacktestingEngine
except ImportError:
    # Fallback pro compatibility
    try:
        from backtesting_engine import BacktestingEngine
    except ImportError as e:
        logger.error(f"Cannot import BacktestingEngine: {e}")
        raise ImportError("BacktestingEngine not available. Please ensure it exists in src/betting/")


class ParameterOptimizer:
    """
    Comprehensive Parameter Optimization for Hockey Backtesting
    
    ENHANCED with:
    - Centralized path management
    - Component-specific logging
    - Performance monitoring
    - Safe file handling
    """
    
    def __init__(self, 
                 elo_model_name: str = 'elo_model_trained_2024',
                 elo_model_type: str = 'pkl',
                 initial_bankroll: float = 10000.0,
                 n_workers: Optional[int] = None):
        """
        Initialize Parameter Optimizer
        
        Args:
            elo_model_name: Name of trained Elo model (without extension)
            elo_model_type: Model file type (pkl, joblib, etc.)
            initial_bankroll: Starting bankroll for each simulation
            n_workers: Number of parallel workers (None = auto-detect)
        """
        # === MIGRATION: Store model parameters for BacktestingEngine ===
        self.elo_model_name = elo_model_name
        self.elo_model_type = elo_model_type
        self.initial_bankroll = initial_bankroll
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        
        # Optimization results storage
        self.optimization_results = []
        self.best_strategies = {}
        
        # Default parameter grids
        self.parameter_grids = self._define_default_grids()
        
        # === MIGRATION: Performance monitoring ===
        self.perf_logger = PerformanceLogger(logger)
        
        logger.info("Parameter Optimizer initialized (Enhanced)")
        logger.info(f"   Workers: {self.n_workers}")
        logger.info(f"   Model: {elo_model_name}.{elo_model_type}")
        logger.info(f"   Bankroll: €{initial_bankroll:,.0f}")
        
    def _define_default_grids(self) -> Dict[str, List]:
        """Define default parameter grids for optimization"""
        
        return {
            'edge_threshold': [0.03, 0.05, 0.08, 0.10, 0.12, 0.15],  # 3% to 15%
            'min_odds': [1.20, 1.30, 1.50, 1.80, 2.00],              # Min odds thresholds
            'stake_method': ['fixed', 'kelly', 'hybrid'],              # Staking methods
            'stake_size': {
                'fixed': [0.01, 0.02, 0.03, 0.05],                   # 1% to 5% for fixed
                'kelly': [0.25, 0.50, 0.75],                         # 25% to 75% Kelly multiplier
                'hybrid': [0.02, 0.03, 0.04]                         # 2% to 4% for hybrid
            },
            'ev_method': ['basic', 'kelly_enhanced', 'confidence_weighted'],  # EV calculations
            'max_stake_pct': [0.10, 0.15, 0.20]                      # Max stake limits
        }
    
    def set_parameter_grid(self, parameter_grids: Dict[str, Any]):
        """
        Set custom parameter grids for optimization
        
        Args:
            parameter_grids: Dictionary defining parameter ranges
        """
        self.parameter_grids.update(parameter_grids)
        logger.info("Custom parameter grids set")
    
    def generate_parameter_combinations(self, 
                                      quick_test: bool = False,
                                      focused_search: bool = False) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations for testing
        
        Args:
            quick_test: If True, use reduced grid for faster testing
            focused_search: If True, focus on promising parameter ranges
            
        Returns:
            List of parameter combination dictionaries
        """
        if quick_test:
            # Reduced grid for quick testing
            combinations = []
            for edge in [0.05, 0.10]:
                for min_odds in [1.30, 1.80]:
                    for ev_method in ['basic', 'kelly_enhanced']:
                        for stake_method in ['fixed', 'kelly']:
                            stake_sizes = [0.02] if stake_method == 'fixed' else [0.50]
                            for stake_size in stake_sizes:
                                combinations.append({
                                    'edge_threshold': edge,
                                    'min_odds': min_odds,
                                    'stake_method': stake_method,
                                    'stake_size': stake_size,
                                    'ev_method': ev_method,
                                    'max_stake_pct': 0.10
                                })
            
        elif focused_search:
            # Focus on higher edge thresholds based on initial results
            combinations = []
            for edge in [0.08, 0.10, 0.12, 0.15]:
                for min_odds in [1.50, 1.80, 2.00]:
                    for ev_method in ['kelly_enhanced', 'confidence_weighted']:
                        for stake_method in ['kelly', 'hybrid']:
                            stake_sizes = self.parameter_grids['stake_size'][stake_method]
                            for stake_size in stake_sizes:
                                for max_stake in [0.10, 0.15]:
                                    combinations.append({
                                        'edge_threshold': edge,
                                        'min_odds': min_odds,
                                        'stake_method': stake_method,
                                        'stake_size': stake_size,
                                        'ev_method': ev_method,
                                        'max_stake_pct': max_stake
                                    })
        else:
            # Full comprehensive grid search
            combinations = []
            
            for edge in self.parameter_grids['edge_threshold']:
                for min_odds in self.parameter_grids['min_odds']:
                    for stake_method in self.parameter_grids['stake_method']:
                        stake_sizes = self.parameter_grids['stake_size'][stake_method]
                        for stake_size in stake_sizes:
                            for ev_method in self.parameter_grids['ev_method']:
                                for max_stake in self.parameter_grids['max_stake_pct']:
                                    combinations.append({
                                        'edge_threshold': edge,
                                        'min_odds': min_odds,
                                        'stake_method': stake_method,
                                        'stake_size': stake_size,
                                        'ev_method': ev_method,
                                        'max_stake_pct': max_stake
                                    })
        
        logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def run_single_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run single backtesting simulation with given parameters
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Results dictionary with parameters and performance metrics
        """
        try:
            # Initialize fresh engine for each test
            engine = BacktestingEngine(
                elo_model_name=self.elo_model_name,
                elo_model_type=self.elo_model_type,
                initial_bankroll=self.initial_bankroll
            )
            
            # Load data (this is fast as it's cached)
            engine.load_backtesting_data(season='2025')
            
            # Run backtest with parameters
            results = engine.run_backtest(**params)
            
            # Extract key metrics
            performance = results['performance']
            statistics = results['statistics']
            
            # Create summary result
            result_summary = {
                'parameters': params.copy(),
                'performance': {
                    'roi': performance['roi'],
                    'win_rate': performance['win_rate'],
                    'profit': performance['profit'],
                    'max_drawdown': performance['max_drawdown'],
                    'sharpe_ratio': performance['sharpe_ratio'],
                    'prediction_accuracy': performance['prediction_accuracy'],
                    'total_return_pct': performance['total_return_pct'],
                    'avg_odds': performance['avg_odds'],
                    'avg_bet_size': performance['avg_bet_size']
                },
                'statistics': {
                    'total_bets': statistics['total_bets'],
                    'winning_bets': statistics['winning_bets'],
                    'total_staked': statistics['total_staked'],
                    'total_returns': statistics['total_returns'],
                    'final_bankroll': engine.current_bankroll
                },
                'risk_metrics': {
                    'max_drawdown': performance['max_drawdown'],
                    'sharpe_ratio': performance['sharpe_ratio'],
                    'largest_loss': performance['largest_loss'],
                    'win_rate': performance['win_rate']
                }
            }
            
            return result_summary
            
        except Exception as e:
            logger.error(f"Backtest failed for params {params}: {e}")
            return {
                'parameters': params.copy(),
                'error': str(e),
                'performance': {'roi': -1.0},  # Penalty for failed runs
                'statistics': {'total_bets': 0}
            }
    
    def optimize_parameters(self, 
                          search_type: str = 'comprehensive',
                          optimization_target: str = 'roi',
                          min_bets_threshold: int = 50) -> Dict[str, Any]:
        """
        Run comprehensive parameter optimization
        
        Args:
            search_type: 'quick', 'focused', or 'comprehensive'
            optimization_target: 'roi', 'sharpe', 'profit', or 'risk_adjusted'
            min_bets_threshold: Minimum number of bets required for valid strategy
            
        Returns:
            Optimization results and best strategies
        """
        logger.info(f"Starting parameter optimization ({search_type} search)")
        logger.info(f"   Target: {optimization_target}")
        logger.info(f"   Min bets threshold: {min_bets_threshold}")
        
        # === MIGRATION: Performance tracking ===
        self.perf_logger.start_timer('parameter_optimization')
        
        # Generate parameter combinations
        if search_type == 'quick':
            combinations = self.generate_parameter_combinations(quick_test=True)
        elif search_type == 'focused':
            combinations = self.generate_parameter_combinations(focused_search=True)
        else:
            combinations = self.generate_parameter_combinations()
        
        total_combinations = len(combinations)
        logger.info(f"Running {total_combinations} backtests on {self.n_workers} workers...")
        
        # Run parallel optimization
        self.optimization_results = []
        completed_count = 0
        
        # Performance tracking for parallel processing
        parallel_timer_name = 'parallel_backtesting'
        self.perf_logger.start_timer(parallel_timer_name)
        
        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all jobs
            future_to_params = {
                executor.submit(self.run_single_backtest, params): params 
                for params in combinations
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_params):
                try:
                    result = future.result()
                    self.optimization_results.append(result)
                    
                    completed_count += 1
                    
                    # Progress logging
                    if completed_count % max(1, total_combinations // 10) == 0:
                        progress_pct = (completed_count / total_combinations) * 100
                        logger.info(f"Progress: {completed_count}/{total_combinations} ({progress_pct:.1f}%)")
                        
                except Exception as e:
                    logger.error(f"Future execution failed: {e}")
                    completed_count += 1
        
        # End performance tracking
        self.perf_logger.end_timer(parallel_timer_name)
        
        logger.info(f"Optimization completed! {len(self.optimization_results)} results collected")
        
        # Filter valid results (minimum bets threshold)
        valid_results = [
            result for result in self.optimization_results 
            if result.get('statistics', {}).get('total_bets', 0) >= min_bets_threshold
            and 'error' not in result
        ]
        
        logger.info(f"Valid strategies: {len(valid_results)}/{len(self.optimization_results)}")
        
        if not valid_results:
            logger.warning("No valid strategies found!")
            return {'best_strategies': {}, 'optimization_results': self.optimization_results}
        
        # Find best strategies by different criteria
        self.best_strategies = self._find_best_strategies(valid_results, optimization_target)
        
        # Generate comprehensive analysis
        analysis = self._generate_optimization_analysis(valid_results)
        
        # End total performance tracking
        self.perf_logger.end_timer('parameter_optimization')
        
        return {
            'best_strategies': self.best_strategies,
            'optimization_results': self.optimization_results,
            'valid_results': valid_results,
            'analysis': analysis,
            'summary': {
                'total_combinations': total_combinations,
                'valid_strategies': len(valid_results),
                'completion_rate': len(self.optimization_results) / total_combinations,
                'optimization_target': optimization_target
            }
        }
    
    def _find_best_strategies(self, 
                            valid_results: List[Dict], 
                            primary_target: str) -> Dict[str, Dict]:
        """Find best strategies by multiple criteria"""
        
        if not valid_results:
            return {}
        
        # Create DataFrame with explicit position tracking
        results_data = []
        for i, result in enumerate(valid_results):
            row = {**result['parameters'], **result['performance'], **result['statistics']}
            row['_position'] = i  # Add position for easy lookup
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        best_strategies = {}
        
        # Best by ROI
        best_roi_idx = results_df['roi'].idxmax()
        best_strategies['best_roi'] = {
            'strategy': valid_results[results_df.loc[best_roi_idx, '_position']],
            'rank_by': 'roi',
            'value': results_df.loc[best_roi_idx, 'roi']
        }
        
        # Best by Sharpe Ratio
        if results_df['sharpe_ratio'].notna().any():
            best_sharpe_idx = results_df['sharpe_ratio'].idxmax()
            best_strategies['best_sharpe'] = {
                'strategy': valid_results[results_df.loc[best_sharpe_idx, '_position']],
                'rank_by': 'sharpe_ratio',
                'value': results_df.loc[best_sharpe_idx, 'sharpe_ratio']
            }
        
        # Best by Risk-Adjusted Return (ROI / Max Drawdown)
        results_df['risk_adjusted_return'] = results_df['roi'] / (results_df['max_drawdown'] + 0.01)
        best_risk_adj_idx = results_df['risk_adjusted_return'].idxmax()
        best_strategies['best_risk_adjusted'] = {
            'strategy': valid_results[results_df.loc[best_risk_adj_idx, '_position']],
            'rank_by': 'risk_adjusted_return',
            'value': results_df.loc[best_risk_adj_idx, 'risk_adjusted_return']
        }
        
        # Best by minimal drawdown (among profitable strategies)
        profitable_mask = results_df['roi'] > 0
        if profitable_mask.any():
            profitable_df = results_df[profitable_mask]
            min_dd_idx = profitable_df['max_drawdown'].idxmin()
            best_strategies['best_low_risk'] = {
                'strategy': valid_results[profitable_df.loc[min_dd_idx, '_position']],
                'rank_by': 'min_drawdown_profitable',
                'value': profitable_df.loc[min_dd_idx, 'max_drawdown']
            }
        
        # Best by activity (most bets among profitable strategies)
        if profitable_mask.any():
            profitable_df = results_df[profitable_mask]
            max_bets_idx = profitable_df['total_bets'].idxmax()
            best_strategies['most_active_profitable'] = {
                'strategy': valid_results[profitable_df.loc[max_bets_idx, '_position']],
                'rank_by': 'most_bets_profitable',
                'value': profitable_df.loc[max_bets_idx, 'total_bets']
            }
        
        # Primary target strategy
        if primary_target != 'roi' and primary_target in results_df.columns:
            if primary_target == 'sharpe':
                primary_target = 'sharpe_ratio'
            elif primary_target == 'risk_adjusted':
                primary_target = 'risk_adjusted_return'
            
            primary_idx = results_df[primary_target].idxmax()
            best_strategies['primary_target'] = {
                'strategy': valid_results[results_df.loc[primary_idx, '_position']],
                'rank_by': primary_target,
                'value': results_df.loc[primary_idx, primary_target]
            }
        
        logger.info(f"Found {len(best_strategies)} best strategy categories")
        return best_strategies
    
    def _generate_optimization_analysis(self, valid_results: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive analysis of optimization results"""
        
        if not valid_results:
            return {'message': 'No valid results to analyze'}
        
        # Convert to DataFrame for analysis
        analysis_data = []
        for result in valid_results:
            row = {**result['parameters'], **result['performance'], **result['statistics']}
            analysis_data.append(row)
        
        df = pd.DataFrame(analysis_data)
        
        # Parameter impact analysis
        parameter_impact = {}
        
        for param in ['edge_threshold', 'min_odds', 'stake_method', 'ev_method']:
            if param in df.columns:
                param_analysis = df.groupby(param).agg({
                    'roi': ['mean', 'std', 'count'],
                    'max_drawdown': 'mean',
                    'total_bets': 'mean',
                    'win_rate': 'mean'
                }).round(4)
                
                # Flatten column names
                param_analysis.columns = ['_'.join(col).strip() for col in param_analysis.columns]
                parameter_impact[param] = param_analysis.to_dict('index')
        
        # Performance distribution
        performance_stats = {
            'roi': {
                'mean': df['roi'].mean(),
                'std': df['roi'].std(),
                'min': df['roi'].min(),
                'max': df['roi'].max(),
                'positive_strategies': (df['roi'] > 0).sum(),
                'negative_strategies': (df['roi'] < 0).sum()
            },
            'total_bets': {
                'mean': df['total_bets'].mean(),
                'std': df['total_bets'].std(),
                'min': df['total_bets'].min(),
                'max': df['total_bets'].max()
            },
            'max_drawdown': {
                'mean': df['max_drawdown'].mean(),
                'std': df['max_drawdown'].std(),
                'min': df['max_drawdown'].min(),
                'max': df['max_drawdown'].max()
            }
        }
        
        # Correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()['roi'].sort_values(ascending=False)
        
        # Top performing parameter combinations
        top_10_strategies = df.nlargest(10, 'roi')[['edge_threshold', 'min_odds', 'stake_method', 
                                                   'ev_method', 'roi', 'max_drawdown', 'total_bets']].to_dict('records')
        
        return {
            'parameter_impact': parameter_impact,
            'performance_statistics': performance_stats,
            'roi_correlation': correlation_matrix.to_dict(),
            'top_strategies': top_10_strategies,
            'total_valid_strategies': len(df),
            'profitable_strategies_count': (df['roi'] > 0).sum(),
            'profitable_strategies_pct': (df['roi'] > 0).mean() * 100
        }
    
    def save_optimization_results(self, 
                                results: Dict[str, Any],
                                filename_prefix: str = 'parameter_optimization') -> str:
        """
        Save optimization results to files using enhanced file handlers
        
        Args:
            results: Optimization results dictionary
            filename_prefix: Prefix for output files
            
        Returns:
            Path to main results file
        """
        # === MIGRATION: Use PATHS for output directory ===
        output_dir = PATHS.experiments
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # === MIGRATION: Use enhanced file handlers ===
        # Main results (JSON)
        main_filename = f'{filename_prefix}_results_{timestamp}.json'
        main_file = output_dir / main_filename
        
        # Convert for JSON serialization
        json_results = self._convert_for_json(results)
        
        try:
            write_json(json_results, main_file)
            logger.info(f"Main results saved to: {main_file}")
        except Exception as e:
            logger.error(f"Failed to save main results: {e}")
            raise
        
        # Detailed results (CSV)
        if results.get('optimization_results'):
            csv_data = []
            for result in results['optimization_results']:
                if 'error' not in result:
                    row = {**result['parameters'], **result['performance'], **result['statistics']}
                    csv_data.append(row)
            
            if csv_data:
                csv_filename = f'{filename_prefix}_detailed_{timestamp}.csv'
                csv_file = output_dir / csv_filename
                
                try:
                    csv_df = pd.DataFrame(csv_data)
                    write_csv(csv_df, csv_file, index=False)
                    logger.info(f"Detailed CSV saved to: {csv_file}")
                except Exception as e:
                    logger.error(f"Failed to save detailed CSV: {e}")
        
        # Best strategies summary (CSV)
        if results.get('best_strategies'):
            best_strategies_data = []
            for category, strategy_info in results['best_strategies'].items():
                strategy = strategy_info['strategy']
                row = {
                    'category': category,
                    'rank_by': strategy_info['rank_by'],
                    'value': strategy_info['value'],
                    **strategy['parameters'],
                    **strategy['performance'],
                    **strategy['statistics']
                }
                best_strategies_data.append(row)
            
            best_filename = f'{filename_prefix}_best_{timestamp}.csv'
            best_file = output_dir / best_filename
            
            try:
                best_df = pd.DataFrame(best_strategies_data)
                write_csv(best_df, best_file, index=False)
                logger.info(f"Best strategies CSV saved to: {best_file}")
            except Exception as e:
                logger.error(f"Failed to save best strategies CSV: {e}")
        
        logger.info(f"Optimization results saved successfully")
        return str(main_file)
    
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
        else:
            return obj
    
    def _convert_key_for_json(self, key):
        """Convert dictionary keys to JSON-serializable format"""
        if isinstance(key, (str, int, float, bool)):
            return key
        else:
            return str(key)
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable summary report"""
        
        if not results.get('best_strategies'):
            return "No optimization results available."
        
        report_lines = []
        report_lines.append("HOCKEY BETTING PARAMETER OPTIMIZATION REPORT")
        report_lines.append("=" * 60)
        
        # Summary statistics
        summary = results.get('summary', {})
        report_lines.append(f"Total combinations tested: {summary.get('total_combinations', 0):,}")
        report_lines.append(f"Valid strategies found: {summary.get('valid_strategies', 0):,}")
        report_lines.append(f"Optimization target: {summary.get('optimization_target', 'N/A')}")
        
        # Best strategies
        report_lines.append("\nBEST STRATEGIES BY CATEGORY:")
        report_lines.append("-" * 40)
        
        for category, strategy_info in results['best_strategies'].items():
            strategy = strategy_info['strategy']
            params = strategy['parameters']
            perf = strategy['performance']
            stats = strategy['statistics']
            
            report_lines.append(f"\n{category.upper().replace('_', ' ')}:")
            report_lines.append(f"   ROI: {perf['roi']:+.2%}")
            report_lines.append(f"   Max Drawdown: {perf['max_drawdown']:.2%}")
            report_lines.append(f"   Total Bets: {stats['total_bets']:,}")
            report_lines.append(f"   Win Rate: {perf['win_rate']:.1%}")
            report_lines.append(f"   Parameters:")
            report_lines.append(f"     - Edge Threshold: {params['edge_threshold']:.1%}")
            report_lines.append(f"     - Min Odds: {params['min_odds']:.2f}")
            report_lines.append(f"     - Stake Method: {params['stake_method']}")
            report_lines.append(f"     - Stake Size: {params['stake_size']:.1%}")
            report_lines.append(f"     - EV Method: {params['ev_method']}")
        
        # Analysis insights
        analysis = results.get('analysis', {})
        if analysis and 'performance_statistics' in analysis:
            perf_stats = analysis['performance_statistics']
            report_lines.append(f"\nOVERALL PERFORMANCE ANALYSIS:")
            report_lines.append("-" * 40)
            report_lines.append(f"Average ROI: {perf_stats['roi']['mean']:+.2%}")
            report_lines.append(f"ROI Range: {perf_stats['roi']['min']:+.2%} to {perf_stats['roi']['max']:+.2%}")
            report_lines.append(f"Profitable strategies: {analysis.get('profitable_strategies_count', 0)} "
                               f"({analysis.get('profitable_strategies_pct', 0):.1f}%)")
        
        return "\n".join(report_lines)


def run_optimization(search_type: str = 'quick', 
                    optimization_target: str = 'roi',
                    min_bets_threshold: int = 20,
                    elo_model_name: str = 'elo_model_trained_2024',
                    elo_model_type: str = 'pkl') -> Dict[str, Any]:
    """
    Enhanced optimization function with flexible parameters
    
    Args:
        search_type: 'quick', 'focused', or 'comprehensive'
        optimization_target: 'roi', 'sharpe', 'risk_adjusted', or 'profit'
        min_bets_threshold: Minimum number of bets for valid strategy
        elo_model_name: Name of Elo model to use (without extension)
        elo_model_type: Model file type (pkl, joblib, etc.)
    
    Returns:
        Optimization results dictionary
    """
    logger.info(f"Starting {search_type} parameter optimization...")
    logger.info(f"   Target: {optimization_target}")
    logger.info(f"   Min bets: {min_bets_threshold}")
    logger.info(f"   Model: {elo_model_name}.{elo_model_type}")
    
    # === MIGRATION: Enhanced initialization ===
    optimizer = ParameterOptimizer(
        elo_model_name=elo_model_name,
        elo_model_type=elo_model_type,
        initial_bankroll=10000.0
    )
    
    # Run optimization with specified parameters
    results = optimizer.optimize_parameters(
        search_type=search_type,
        optimization_target=optimization_target,
        min_bets_threshold=min_bets_threshold
    )
    
    # Save results with target in filename
    output_file = optimizer.save_optimization_results(
        results, 
        filename_prefix=f'optimization_{optimization_target}_{search_type}'
    )
    
    # Generate and print summary
    summary_report = optimizer.generate_summary_report(results)
    print("\n" + summary_report)
    
    logger.info(f"{search_type.title()} optimization ({optimization_target}) completed!")
    logger.info(f"Results saved to {output_file}")
    return results


if __name__ == "__main__":
    # === MIGRATION: Enhanced main execution ===
    
    # Easy configuration parameters
    SEARCH_TYPE = 'quick'               # 'quick', 'focused', 'comprehensive'
    OPTIMIZATION_TARGET = 'roi'         # Options:
                                        #   'roi' - maximize return on investment
                                        #   'sharpe' - maximize risk-adjusted returns (Sharpe ratio)
                                        #   'risk_adjusted' - maximize ROI/drawdown ratio
                                        #   'profit' - maximize absolute profit
    MIN_BETS = 20                       # Minimum bets threshold for valid strategy
    ELO_MODEL_NAME = 'elo_model_trained_2024'  # Model name without extension
    
    # Run optimization with enhanced infrastructure
    results = run_optimization(
        search_type=SEARCH_TYPE,
        optimization_target=OPTIMIZATION_TARGET,
        min_bets_threshold=MIN_BETS,
        elo_model_name=ELO_MODEL_NAME
    )