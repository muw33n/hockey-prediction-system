#!/usr/bin/env python3
"""
Hockey Performance Analyzer
Comprehensive analysis of backtesting results with quarterly breakdown,
risk assessment, strategy comparison, and statistical validation

Features:
- Quarterly performance breakdown for robustness validation
- Statistical significance testing
- Risk metrics and drawdown analysis
- Strategy comparison across multiple configurations
- Model performance validation (predictions vs betting results)
- Time-based performance analysis
- Comprehensive reporting and export

Location: src/betting/performance_analyzer.py
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import glob
from datetime import datetime, date
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/performance_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Comprehensive Performance Analysis for Hockey Backtesting Results
    
    Analyzes backtesting results with focus on statistical robustness,
    risk assessment, and strategy validation
    """
    
    def __init__(self, results_dir: str = 'models/experiments'):
        """
        Initialize Performance Analyzer
        
        Args:
            results_dir: Directory containing backtesting results
        """
        self.results_dir = results_dir
        self.results_cache = {}
        self.analysis_cache = {}
        
        logger.info(f"📊 PerformanceAnalyzer initialized")
        logger.info(f"   Results directory: {results_dir}")
        
    def detect_results_format(self, results: Dict[str, Any]) -> str:
        """
        Detect the format of results file
        
        Args:
            results: Loaded results dictionary
            
        Returns:
            Format type: 'backtest_single', 'optimization_multi', or 'unknown'
        """
        if 'bet_history' in results and 'gaming_day_results' in results:
            return 'backtest_single'
        elif 'optimization_results' in results and 'best_strategies' in results:
            return 'optimization_multi'
        else:
            return 'unknown'
    
    def load_backtest_results(self, 
                            result_file: Optional[str] = None,
                            pattern: str = '*backtest*.json') -> Dict[str, Any]:
        """
        Load backtesting results from JSON file
        
        Args:
            result_file: Specific file to load (None = find latest)
            pattern: File pattern to search for
            
        Returns:
            Loaded results dictionary with format detection
        """
        if result_file is None:
            # Search for multiple patterns to find all relevant files
            patterns = [
                '*backtest*.json',
                '*optimization*.json', 
                '*results*.json'
            ]
            
            all_files = []
            for search_pattern in patterns:
                full_pattern = os.path.join(self.results_dir, search_pattern)
                files = glob.glob(full_pattern)
                all_files.extend(files)
            
            # Remove duplicates and sort by modification time
            all_files = list(set(all_files))
            
            if not all_files:
                raise FileNotFoundError(f"No results files found in: {self.results_dir}")
            
            # Sort by modification time, get latest
            all_files.sort(key=os.path.getmtime, reverse=True)
            
            # Show available files for user info
            logger.info(f"📁 Found {len(all_files)} results files:")
            for i, file in enumerate(all_files[:5]):  # Show top 5
                file_time = datetime.fromtimestamp(os.path.getmtime(file))
                age_indicator = "🆕" if i == 0 else "📄"
                logger.info(f"   {age_indicator} {os.path.basename(file)} ({file_time.strftime('%Y-%m-%d %H:%M')})")
            
            if len(all_files) > 5:
                logger.info(f"   ... and {len(all_files) - 5} more files")
            
            result_file = all_files[0]  # Latest file
            logger.info(f"🎯 Selected: {os.path.basename(result_file)}")
        
        # Load results
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Detect format
        results_format = self.detect_results_format(results)
        results['_format_detected'] = results_format
        
        logger.info(f"✅ Results loaded from {os.path.basename(result_file)}")
        logger.info(f"📊 Format detected: {results_format}")
        
        # Extract key information based on format
        if results_format == 'backtest_single':
            if 'bet_history' in results:
                logger.info(f"   Total bets: {len(results['bet_history']):,}")
            if 'performance' in results:
                roi = results['performance'].get('roi', 0)
                logger.info(f"   ROI: {roi:+.2%}")
        
        elif results_format == 'optimization_multi':
            if 'optimization_results' in results:
                valid_results = [r for r in results['optimization_results'] if 'error' not in r]
                logger.info(f"   Optimization strategies: {len(valid_results):,}")
            if 'best_strategies' in results:
                logger.info(f"   Best strategy categories: {len(results['best_strategies'])}")
                
                # Show best ROI strategy
                best_roi = results['best_strategies'].get('best_roi', {})
                if best_roi and 'strategy' in best_roi:
                    roi = best_roi['strategy'].get('performance', {}).get('roi', 0)
                    logger.info(f"   Best ROI found: {roi:+.2%}")
        
        return results
    
    def analyze_quarterly_performance(self, 
                                    results: Dict[str, Any],
                                    quarters_definition: str = 'calendar') -> Dict[str, Any]:
        """
        Analyze performance broken down by quarters for consistency validation
        
        Args:
            results: Backtesting results dictionary
            quarters_definition: 'calendar' or 'hockey_season'
            
        Returns:
            Quarterly analysis results
        """
        logger.info(f"📅 Analyzing quarterly performance ({quarters_definition} quarters)...")
        
        if not results.get('bet_history'):
            return {'error': 'No bet history found in results'}
        
        # Convert bet history to DataFrame
        bet_df = pd.DataFrame(results['bet_history'])
        bet_df['date'] = pd.to_datetime(bet_df['date'])
        
        # Define quarters
        if quarters_definition == 'hockey_season':
            # Hockey season quarters: Oct-Dec, Jan-Mar, Apr-Jun, Jul-Sep
            def get_hockey_quarter(date):
                month = date.month
                if month in [10, 11, 12]:
                    return f"{date.year}-Q1"  # Start of season
                elif month in [1, 2, 3]:
                    return f"{date.year}-Q2"  # Mid season
                elif month in [4, 5, 6]:
                    return f"{date.year}-Q3"  # Playoffs/End season
                else:  # 7, 8, 9
                    return f"{date.year}-Q4"  # Offseason
            
            bet_df['quarter'] = bet_df['date'].apply(get_hockey_quarter)
        else:
            # Calendar quarters
            bet_df['quarter'] = bet_df['date'].dt.to_period('Q').astype(str)
        
        # Group by quarter and calculate metrics
        quarterly_results = []
        
        for quarter, quarter_bets in bet_df.groupby('quarter'):
            if len(quarter_bets) == 0:
                continue
            
            # Basic metrics
            total_bets = len(quarter_bets)
            winning_bets = quarter_bets['bet_won'].sum()
            win_rate = winning_bets / total_bets
            
            # Financial metrics
            total_staked = quarter_bets['stake'].sum()
            total_returns = quarter_bets['payout'].sum()
            net_profit = total_returns - total_staked
            roi = net_profit / total_staked if total_staked > 0 else 0
            
            # Risk metrics
            bet_results = quarter_bets['net_result'].values
            daily_results = quarter_bets.groupby(quarter_bets['date'].dt.date)['net_result'].sum()
            
            # Calculate drawdown for this quarter
            cumulative_pnl = daily_results.cumsum()
            running_max = cumulative_pnl.cummax()
            drawdown = (cumulative_pnl - running_max) / abs(running_max).replace(0, 1)
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            # Statistical metrics
            if len(bet_results) > 1:
                volatility = np.std(bet_results)
                sharpe_ratio = np.mean(bet_results) / volatility if volatility > 0 else 0
            else:
                volatility = 0
                sharpe_ratio = 0
            
            quarterly_result = {
                'quarter': quarter,
                'total_bets': total_bets,
                'winning_bets': winning_bets,
                'win_rate': win_rate,
                'total_staked': total_staked,
                'total_returns': total_returns,
                'net_profit': net_profit,
                'roi': roi,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'avg_bet_size': quarter_bets['stake'].mean(),
                'avg_odds': quarter_bets['odds'].mean(),
                'start_date': quarter_bets['date'].min(),
                'end_date': quarter_bets['date'].max()
            }
            
            quarterly_results.append(quarterly_result)
        
        # Overall quarterly analysis
        if quarterly_results:
            quarterly_df = pd.DataFrame(quarterly_results)
            
            # Consistency metrics
            roi_consistency = {
                'roi_mean': quarterly_df['roi'].mean(),
                'roi_std': quarterly_df['roi'].std(),
                'roi_min': quarterly_df['roi'].min(),
                'roi_max': quarterly_df['roi'].max(),
                'profitable_quarters': (quarterly_df['roi'] > 0).sum(),
                'total_quarters': len(quarterly_df),
                'profitable_quarter_pct': (quarterly_df['roi'] > 0).mean()
            }
            
            # Statistical tests
            statistical_tests = self._perform_quarterly_statistical_tests(quarterly_df)
            
        else:
            roi_consistency = {}
            statistical_tests = {}
        
        analysis = {
            'quarterly_results': quarterly_results,
            'consistency_metrics': roi_consistency,
            'statistical_tests': statistical_tests,
            'quarters_definition': quarters_definition,
            'analysis_date': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Quarterly analysis completed:")
        logger.info(f"   Quarters analyzed: {len(quarterly_results)}")
        if roi_consistency:
            logger.info(f"   Profitable quarters: {roi_consistency['profitable_quarters']}/{roi_consistency['total_quarters']}")
            logger.info(f"   Average quarterly ROI: {roi_consistency['roi_mean']:+.2%}")
        
        return analysis
    
    def _perform_quarterly_statistical_tests(self, quarterly_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical tests on quarterly performance"""
        
        tests = {}
        
        if len(quarterly_df) < 2:
            return {'note': 'Insufficient quarters for statistical testing'}
        
        # Test for consistency (one-sample t-test against zero)
        roi_values = quarterly_df['roi'].values
        if len(roi_values) > 1:
            t_stat, p_value = stats.ttest_1samp(roi_values, 0)
            tests['roi_significance'] = {
                'test': 'one_sample_t_test',
                'null_hypothesis': 'quarterly ROI = 0',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_at_5pct': p_value < 0.05,
                'interpretation': 'Significantly different from zero' if p_value < 0.05 else 'Not significantly different from zero'
            }
        
        # Test for normality of returns
        if len(roi_values) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(roi_values)
            tests['normality_test'] = {
                'test': 'shapiro_wilk',
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'normally_distributed': shapiro_p > 0.05
            }
        
        # Test for trend (correlation with time)
        if len(roi_values) >= 3:
            time_index = range(len(roi_values))
            correlation, corr_p = stats.pearsonr(time_index, roi_values)
            tests['trend_analysis'] = {
                'test': 'pearson_correlation',
                'correlation': correlation,
                'p_value': corr_p,
                'significant_trend': corr_p < 0.05,
                'trend_direction': 'improving' if correlation > 0 else 'declining' if correlation < 0 else 'stable'
            }
        
        return tests
    
    def analyze_strategy_comparison(self, 
                                  optimization_results: Union[str, Dict, List[str]],
                                  top_n: int = 10) -> Dict[str, Any]:
        """
        Compare multiple strategies from optimization results
        
        Args:
            optimization_results: Path to optimization file, results dict, or list of files
            top_n: Number of top strategies to analyze in detail
            
        Returns:
            Strategy comparison analysis
        """
        logger.info(f"🔄 Analyzing strategy comparison (top {top_n} strategies)...")
        
        # Load optimization results
        if isinstance(optimization_results, str):
            with open(optimization_results, 'r', encoding='utf-8') as f:
                opt_results = json.load(f)
        elif isinstance(optimization_results, list):
            # Multiple files - combine results
            opt_results = {'optimization_results': []}
            for file in optimization_results:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    opt_results['optimization_results'].extend(data.get('optimization_results', []))
        else:
            opt_results = optimization_results
        
        if not opt_results.get('optimization_results'):
            return {'error': 'No optimization results found'}
        
        # Convert to DataFrame for analysis
        strategy_data = []
        for result in opt_results['optimization_results']:
            if 'error' not in result and result.get('statistics', {}).get('total_bets', 0) > 0:
                row = {
                    **result['parameters'],
                    **result['performance'],
                    **result['statistics']
                }
                strategy_data.append(row)
        
        if not strategy_data:
            return {'error': 'No valid strategies found'}
        
        strategy_df = pd.DataFrame(strategy_data)
        
        # Calculate additional metrics
        strategy_df['profit_per_bet'] = strategy_df['profit'] / strategy_df['total_bets']
        strategy_df['risk_adjusted_return'] = strategy_df['roi'] / (strategy_df['max_drawdown'] + 0.01)
        strategy_df['win_loss_ratio'] = strategy_df['win_rate'] / (1 - strategy_df['win_rate'])
        
        # Rank strategies by different criteria
        rankings = {}
        
        metrics_to_rank = ['roi', 'sharpe_ratio', 'risk_adjusted_return', 'profit', 'win_rate']
        for metric in metrics_to_rank:
            if metric in strategy_df.columns:
                rankings[f'top_{metric}'] = strategy_df.nlargest(top_n, metric)[
                    ['edge_threshold', 'min_odds', 'stake_method', 'ev_method', 
                     'roi', 'max_drawdown', 'total_bets', 'win_rate', metric]
                ].to_dict('records')
        
        # Parameter impact analysis
        parameter_analysis = {}
        
        categorical_params = ['stake_method', 'ev_method']
        for param in categorical_params:
            if param in strategy_df.columns:
                param_stats = strategy_df.groupby(param).agg({
                    'roi': ['mean', 'std', 'count'],
                    'max_drawdown': 'mean',
                    'total_bets': 'mean',
                    'win_rate': 'mean'
                }).round(4)
                
                param_stats.columns = ['_'.join(col).strip() for col in param_stats.columns]
                parameter_analysis[param] = param_stats.to_dict('index')
        
        # Numerical parameter correlations
        numerical_params = ['edge_threshold', 'min_odds', 'stake_size']
        correlation_analysis = {}
        
        for param in numerical_params:
            if param in strategy_df.columns:
                correlation_analysis[param] = {
                    'roi_correlation': strategy_df[param].corr(strategy_df['roi']),
                    'drawdown_correlation': strategy_df[param].corr(strategy_df['max_drawdown']),
                    'bets_correlation': strategy_df[param].corr(strategy_df['total_bets'])
                }
        
        # Performance distribution analysis
        performance_distribution = {
            'roi': {
                'percentiles': {
                    '10th': strategy_df['roi'].quantile(0.1),
                    '25th': strategy_df['roi'].quantile(0.25),
                    '50th': strategy_df['roi'].quantile(0.5),
                    '75th': strategy_df['roi'].quantile(0.75),
                    '90th': strategy_df['roi'].quantile(0.9)
                },
                'profitable_strategies': (strategy_df['roi'] > 0).sum(),
                'total_strategies': len(strategy_df)
            },
            'drawdown': {
                'percentiles': {
                    '10th': strategy_df['max_drawdown'].quantile(0.1),
                    '25th': strategy_df['max_drawdown'].quantile(0.25),
                    '50th': strategy_df['max_drawdown'].quantile(0.5),
                    '75th': strategy_df['max_drawdown'].quantile(0.75),
                    '90th': strategy_df['max_drawdown'].quantile(0.9)
                }
            }
        }
        
        analysis = {
            'strategy_rankings': rankings,
            'parameter_analysis': parameter_analysis,
            'correlation_analysis': correlation_analysis,
            'performance_distribution': performance_distribution,
            'total_strategies_analyzed': len(strategy_df),
            'analysis_date': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Strategy comparison completed:")
        logger.info(f"   Strategies analyzed: {len(strategy_df):,}")
        logger.info(f"   Profitable strategies: {(strategy_df['roi'] > 0).sum():,}")
        logger.info(f"   Best ROI: {strategy_df['roi'].max():+.2%}")
        
        return analysis
    
    def analyze_risk_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive risk analysis of backtesting results
        
        Args:
            results: Backtesting results dictionary
            
        Returns:
            Risk analysis results
        """
        logger.info("⚠️ Analyzing comprehensive risk metrics...")
        
        if not results.get('bet_history'):
            return {'error': 'No bet history found in results'}
        
        bet_df = pd.DataFrame(results['bet_history'])
        bet_df['date'] = pd.to_datetime(bet_df['date'])
        
        # Daily P&L calculation
        daily_pnl = bet_df.groupby(bet_df['date'].dt.date).agg({
            'net_result': 'sum',
            'stake': 'sum',
            'payout': 'sum'
        }).reset_index()
        
        daily_pnl['cumulative_pnl'] = daily_pnl['net_result'].cumsum()
        daily_pnl['daily_roi'] = daily_pnl['net_result'] / daily_pnl['stake']
        
        # Drawdown analysis
        running_max = daily_pnl['cumulative_pnl'].cummax()
        drawdown = daily_pnl['cumulative_pnl'] - running_max
        drawdown_pct = drawdown / abs(running_max).replace(0, 1)
        
        # Value at Risk (VaR) calculations
        daily_returns = daily_pnl['daily_roi'].dropna()
        if len(daily_returns) > 0:
            var_95 = np.percentile(daily_returns, 5)  # 5th percentile (95% VaR)
            var_99 = np.percentile(daily_returns, 1)  # 1st percentile (99% VaR)
            
            # Expected Shortfall (Conditional VaR)
            es_95 = daily_returns[daily_returns <= var_95].mean()
            es_99 = daily_returns[daily_returns <= var_99].mean()
        else:
            var_95 = var_99 = es_95 = es_99 = 0
        
        # Longest losing streak
        bet_df['bet_won_clean'] = bet_df['bet_won'].fillna(False)  # Handle NaN values
        bet_df['loss_streak'] = (~bet_df['bet_won_clean']).astype(int)
        bet_df['streak_id'] = (bet_df['bet_won_clean'] != bet_df['bet_won_clean'].shift()).cumsum()
        losing_streaks = bet_df[~bet_df['bet_won_clean']].groupby('streak_id').size()
        max_losing_streak = losing_streaks.max() if len(losing_streaks) > 0 else 0
        
        # Bet size analysis
        stake_analysis = {
            'min_stake': bet_df['stake'].min(),
            'max_stake': bet_df['stake'].max(),
            'avg_stake': bet_df['stake'].mean(),
            'median_stake': bet_df['stake'].median(),
            'stake_std': bet_df['stake'].std(),
            'stake_concentration': (bet_df['stake'] > bet_df['stake'].quantile(0.9)).sum() / len(bet_df)
        }
        
        # Odds distribution analysis
        odds_analysis = {
            'min_odds': bet_df['odds'].min(),
            'max_odds': bet_df['odds'].max(),
            'avg_odds': bet_df['odds'].mean(),
            'median_odds': bet_df['odds'].median(),
            'low_odds_bets': (bet_df['odds'] < 1.5).sum() / len(bet_df),
            'high_odds_bets': (bet_df['odds'] > 2.5).sum() / len(bet_df)
        }
        
        # Win/loss pattern analysis
        win_loss_patterns = {
            'consecutive_wins_max': self._calculate_max_consecutive(bet_df['bet_won_clean']),
            'consecutive_losses_max': self._calculate_max_consecutive(~bet_df['bet_won_clean']),
            'win_after_loss_rate': self._calculate_conditional_probability(bet_df, 'loss_then_win'),
            'loss_after_win_rate': self._calculate_conditional_probability(bet_df, 'win_then_loss')
        }
        
        # Monthly risk metrics
        monthly_risk = {}
        if len(bet_df) > 30:  # At least a month of data
            bet_df['month'] = bet_df['date'].dt.to_period('M')
            monthly_pnl = bet_df.groupby('month')['net_result'].sum()
            
            monthly_risk = {
                'monthly_volatility': monthly_pnl.std(),
                'worst_month': monthly_pnl.min(),
                'best_month': monthly_pnl.max(),
                'negative_months': (monthly_pnl < 0).sum(),
                'total_months': len(monthly_pnl)
            }
        
        risk_analysis = {
            'drawdown_metrics': {
                'max_drawdown_absolute': abs(drawdown.min()),
                'max_drawdown_percent': abs(drawdown_pct.min()),
                'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0,
                'drawdown_periods': (drawdown < 0).sum(),
                'recovery_periods': self._calculate_recovery_periods(drawdown)
            },
            'var_metrics': {
                'var_95_daily': var_95,
                'var_99_daily': var_99,
                'expected_shortfall_95': es_95,
                'expected_shortfall_99': es_99
            },
            'streak_analysis': {
                'max_losing_streak': max_losing_streak,
                'avg_losing_streak': losing_streaks.mean() if len(losing_streaks) > 0 else 0,
                'win_loss_patterns': win_loss_patterns
            },
            'stake_analysis': stake_analysis,
            'odds_analysis': odds_analysis,
            'monthly_risk': monthly_risk,
            'analysis_date': datetime.now().isoformat()
        }
        
        logger.info("✅ Risk analysis completed:")
        logger.info(f"   Max drawdown: {abs(drawdown_pct.min()):.2%}")
        logger.info(f"   VaR (95%): {var_95:+.2%}")
        logger.info(f"   Max losing streak: {max_losing_streak}")
        
        return risk_analysis
    
    def _calculate_max_consecutive(self, series: pd.Series) -> int:
        """Calculate maximum consecutive True values in a boolean series"""
        if len(series) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for value in series:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_conditional_probability(self, bet_df: pd.DataFrame, pattern: str) -> float:
        """Calculate conditional probabilities for win/loss patterns"""
        if len(bet_df) < 2:
            return 0.0
        
        # Use cleaned boolean column
        bet_won_clean = bet_df['bet_won'].fillna(False)
        
        if pattern == 'loss_then_win':
            # P(Win | Previous Loss)
            prev_loss = ~bet_won_clean.shift(1)
            current_win = bet_won_clean
            valid_mask = prev_loss.notna()
            
            if valid_mask.sum() == 0:
                return 0.0
            
            return (prev_loss & current_win & valid_mask).sum() / (prev_loss & valid_mask).sum()
        
        elif pattern == 'win_then_loss':
            # P(Loss | Previous Win)
            prev_win = bet_won_clean.shift(1)
            current_loss = ~bet_won_clean
            valid_mask = prev_win.notna()
            
            if valid_mask.sum() == 0:
                return 0.0
            
            return (prev_win & current_loss & valid_mask).sum() / (prev_win & valid_mask).sum()
        
        return 0.0
    
    def _calculate_recovery_periods(self, drawdown: pd.Series) -> Dict[str, float]:
        """Calculate drawdown recovery statistics"""
        if len(drawdown) == 0:
            return {'avg_recovery_days': 0, 'max_recovery_days': 0}
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []
        
        start_idx = None
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                drawdown_periods.append(i - start_idx)
                start_idx = None
        
        # Handle case where we end in drawdown
        if start_idx is not None:
            drawdown_periods.append(len(drawdown) - start_idx)
        
        if drawdown_periods:
            return {
                'avg_recovery_days': np.mean(drawdown_periods),
                'max_recovery_days': max(drawdown_periods),
                'total_drawdown_periods': len(drawdown_periods)
            }
        else:
            return {'avg_recovery_days': 0, 'max_recovery_days': 0, 'total_drawdown_periods': 0}
    
    def analyze_model_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze model prediction performance separate from betting performance
        
        Args:
            results: Backtesting results dictionary
            
        Returns:
            Model performance analysis
        """
        logger.info("🎯 Analyzing model prediction performance...")
        
        # Extract predictions from gaming day results
        all_predictions = []
        
        if 'gaming_day_results' in results:
            for day_result in results['gaming_day_results']:
                if 'predictions' in day_result:
                    all_predictions.extend(day_result['predictions'])
        
        if not all_predictions:
            return {'error': 'No prediction data found in results'}
        
        pred_df = pd.DataFrame(all_predictions)
        
        # Basic accuracy metrics
        total_predictions = len(pred_df)
        correct_predictions = pred_df['correct_prediction'].sum()
        accuracy = correct_predictions / total_predictions
        
        # Confidence-based analysis
        pred_df['confidence_bucket'] = pd.cut(pred_df['confidence'], 
                                            bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
                                            labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+'])
        
        confidence_analysis = pred_df.groupby('confidence_bucket').agg({
            'correct_prediction': ['count', 'sum', 'mean']
        }).round(3)
        
        confidence_analysis.columns = ['total_predictions', 'correct_predictions', 'accuracy']
        confidence_analysis = confidence_analysis.to_dict('index')
        
        # Probability calibration analysis
        prob_bins = np.arange(0.4, 1.0, 0.05)  # 40% to 95% in 5% increments
        calibration_data = []
        
        for i in range(len(prob_bins) - 1):
            bin_min, bin_max = prob_bins[i], prob_bins[i + 1]
            
            # Home team predictions in this bin
            home_mask = (pred_df['home_win_probability'] >= bin_min) & (pred_df['home_win_probability'] < bin_max)
            home_games = pred_df[home_mask]
            
            if len(home_games) > 0:
                avg_predicted_prob = home_games['home_win_probability'].mean()
                actual_win_rate = (home_games['actual_winner'] == 'HOME').mean()
                calibration_data.append({
                    'bin_center': (bin_min + bin_max) / 2,
                    'predicted_prob': avg_predicted_prob,
                    'actual_win_rate': actual_win_rate,
                    'game_count': len(home_games),
                    'side': 'home'
                })
            
            # Away team predictions in this bin  
            away_mask = (pred_df['away_win_probability'] >= bin_min) & (pred_df['away_win_probability'] < bin_max)
            away_games = pred_df[away_mask]
            
            if len(away_games) > 0:
                avg_predicted_prob = away_games['away_win_probability'].mean()
                actual_win_rate = (away_games['actual_winner'] == 'AWAY').mean()
                calibration_data.append({
                    'bin_center': (bin_min + bin_max) / 2,
                    'predicted_prob': avg_predicted_prob,
                    'actual_win_rate': actual_win_rate,
                    'game_count': len(away_games),
                    'side': 'away'
                })
        
        # Brier Score calculation
        home_brier = np.mean((pred_df['home_win_probability'] - (pred_df['actual_winner'] == 'HOME').astype(int)) ** 2)
        away_brier = np.mean((pred_df['away_win_probability'] - (pred_df['actual_winner'] == 'AWAY').astype(int)) ** 2)
        overall_brier = (home_brier + away_brier) / 2
        
        # Log Loss calculation
        epsilon = 1e-15  # Prevent log(0)
        home_probs_clipped = np.clip(pred_df['home_win_probability'], epsilon, 1 - epsilon)
        away_probs_clipped = np.clip(pred_df['away_win_probability'], epsilon, 1 - epsilon)
        
        home_actual = (pred_df['actual_winner'] == 'HOME').astype(int)
        away_actual = (pred_df['actual_winner'] == 'AWAY').astype(int)
        
        home_log_loss = -np.mean(home_actual * np.log(home_probs_clipped) + (1 - home_actual) * np.log(1 - home_probs_clipped))
        away_log_loss = -np.mean(away_actual * np.log(away_probs_clipped) + (1 - away_actual) * np.log(1 - away_probs_clipped))
        overall_log_loss = (home_log_loss + away_log_loss) / 2
        
        # Performance over time
        pred_df['date'] = pd.to_datetime(pred_df['game_id'].astype(str), errors='coerce')  # This might need adjustment
        if 'date' in pred_df.columns:
            pred_df['month'] = pred_df['date'].dt.to_period('M')
            monthly_accuracy = pred_df.groupby('month')['correct_prediction'].mean().to_dict()
        else:
            monthly_accuracy = {}
        
        model_analysis = {
            'overall_metrics': {
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'brier_score': overall_brier,
                'log_loss': overall_log_loss
            },
            'confidence_analysis': confidence_analysis,
            'calibration_data': calibration_data,
            'monthly_accuracy': monthly_accuracy,
            'analysis_date': datetime.now().isoformat()
        }
        
        logger.info("✅ Model performance analysis completed:")
        logger.info(f"   Overall accuracy: {accuracy:.1%}")
        logger.info(f"   Brier score: {overall_brier:.3f}")
        logger.info(f"   Log loss: {overall_log_loss:.3f}")
        
        return model_analysis
    
    def generate_comprehensive_report(self, 
                                    results: Dict[str, Any],
                                    include_quarterly: bool = True,
                                    include_risk: bool = True,
                                    include_model: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive performance analysis report
        
        Args:
            results: Results dictionary (backtest_single or optimization_multi)
            include_quarterly: Include quarterly breakdown analysis
            include_risk: Include risk analysis
            include_model: Include model performance analysis
            
        Returns:
            Comprehensive analysis report
        """
        logger.info("📋 Generating comprehensive performance report...")
        
        results_format = results.get('_format_detected', 'unknown')
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_components': [],
                'results_format': results_format,
                'results_summary': {}
            }
        }
        
        # Handle different result formats
        if results_format == 'backtest_single':
            # Original analysis for single backtest results
            report['report_metadata']['results_summary'] = {
                'roi': results.get('performance', {}).get('roi', 0),
                'total_bets': results.get('statistics', {}).get('total_bets', 0),
                'win_rate': results.get('performance', {}).get('win_rate', 0),
                'max_drawdown': results.get('performance', {}).get('max_drawdown', 0)
            }
            
            # Original analysis methods
            if include_quarterly:
                try:
                    report['quarterly_analysis'] = self.analyze_quarterly_performance(results)
                    report['report_metadata']['analysis_components'].append('quarterly_breakdown')
                    logger.info("✅ Quarterly analysis included")
                except Exception as e:
                    logger.warning(f"⚠️ Quarterly analysis failed: {e}")
                    report['quarterly_analysis'] = {'error': str(e)}
            
            if include_risk:
                try:
                    report['risk_analysis'] = self.analyze_risk_metrics(results)
                    report['report_metadata']['analysis_components'].append('risk_metrics')
                    logger.info("✅ Risk analysis included")
                except Exception as e:
                    logger.warning(f"⚠️ Risk analysis failed: {e}")
                    report['risk_analysis'] = {'error': str(e)}
            
            if include_model:
                try:
                    report['model_analysis'] = self.analyze_model_performance(results)
                    report['report_metadata']['analysis_components'].append('model_performance')
                    logger.info("✅ Model analysis included")
                except Exception as e:
                    logger.warning(f"⚠️ Model analysis failed: {e}")
                    report['model_analysis'] = {'error': str(e)}
        
        elif results_format == 'optimization_multi':
            # Analysis for optimization results
            try:
                # Strategy comparison analysis
                report['strategy_comparison'] = self.analyze_strategy_comparison(results)
                report['report_metadata']['analysis_components'].append('strategy_comparison')
                logger.info("✅ Strategy comparison analysis included")
                
                # Best strategies summary
                report['best_strategies_analysis'] = self.analyze_best_strategies(results)
                report['report_metadata']['analysis_components'].append('best_strategies')
                logger.info("✅ Best strategies analysis included")
                
                # Parameter sensitivity analysis
                report['parameter_analysis'] = self.analyze_parameter_sensitivity(results)
                report['report_metadata']['analysis_components'].append('parameter_sensitivity')
                logger.info("✅ Parameter sensitivity analysis included")
                
                # Summary from best strategy
                if 'best_strategies' in results and 'best_roi' in results['best_strategies']:
                    best_strategy = results['best_strategies']['best_roi']['strategy']
                    report['report_metadata']['results_summary'] = {
                        'best_roi': best_strategy.get('performance', {}).get('roi', 0),
                        'best_roi_bets': best_strategy.get('statistics', {}).get('total_bets', 0),
                        'best_roi_win_rate': best_strategy.get('performance', {}).get('win_rate', 0),
                        'best_roi_drawdown': best_strategy.get('performance', {}).get('max_drawdown', 0),
                        'total_strategies_tested': len(results.get('optimization_results', []))
                    }
                
            except Exception as e:
                logger.warning(f"⚠️ Optimization analysis failed: {e}")
                report['optimization_analysis'] = {'error': str(e)}
        
        else:
            # Unknown format
            report['analysis_error'] = {
                'message': f'Unknown results format: {results_format}',
                'expected_formats': ['backtest_single', 'optimization_multi'],
                'available_keys': list(results.keys())
            }
            logger.warning(f"⚠️ Unknown results format: {results_format}")
        
        component_count = len(report['report_metadata']['analysis_components'])
        logger.info(f"📊 Comprehensive report generated with {component_count} components")
        
        return report
    
    def save_analysis_results(self, 
                            analysis: Dict[str, Any],
                            output_dir: str = 'models/experiments',
                            filename_prefix: str = 'performance_analysis') -> str:
        """
        Save analysis results to files
        
        Args:
            analysis: Analysis results dictionary
            output_dir: Output directory
            filename_prefix: Prefix for output files
            
        Returns:
            Path to main analysis file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Main analysis (JSON)
        main_file = os.path.join(output_dir, f'{filename_prefix}_{timestamp}.json')
        
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"💾 Analysis results saved to {main_file}")
    def analyze_best_strategies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze best strategies from optimization results
        
        Args:
            results: Optimization results dictionary
            
        Returns:
            Best strategies analysis
        """
        logger.info("🏆 Analyzing best strategies from optimization...")
        
        if not results.get('best_strategies'):
            return {'error': 'No best strategies found in optimization results'}
        
        best_strategies = results['best_strategies']
        analysis = {
            'strategy_categories': {},
            'parameter_patterns': {},
            'performance_comparison': {},
            'recommendations': []
        }
        
        # Analyze each category of best strategies
        for category, strategy_info in best_strategies.items():
            if 'strategy' not in strategy_info:
                continue
                
            strategy = strategy_info['strategy']
            params = strategy.get('parameters', {})
            perf = strategy.get('performance', {})
            stats = strategy.get('statistics', {})
            
            category_analysis = {
                'optimization_criterion': strategy_info.get('rank_by', 'unknown'),
                'criterion_value': strategy_info.get('value', 0),
                'parameters': params,
                'performance_metrics': {
                    'roi': perf.get('roi', 0),
                    'win_rate': perf.get('win_rate', 0),
                    'max_drawdown': perf.get('max_drawdown', 0),
                    'sharpe_ratio': perf.get('sharpe_ratio', 0),
                    'total_bets': stats.get('total_bets', 0),
                    'profit': perf.get('profit', 0)
                }
            }
            
            analysis['strategy_categories'][category] = category_analysis
        
        # Find common parameter patterns among best strategies
        param_values = {}
        for category, strategy_info in best_strategies.items():
            if 'strategy' not in strategy_info:
                continue
            params = strategy_info['strategy'].get('parameters', {})
            for param, value in params.items():
                if param not in param_values:
                    param_values[param] = []
                param_values[param].append(value)
        
        # Calculate parameter frequencies
        for param, values in param_values.items():
            if param in ['edge_threshold', 'min_odds', 'stake_size']:
                # Numerical parameters
                analysis['parameter_patterns'][param] = {
                    'mean': np.mean(values) if values else 0,
                    'median': np.median(values) if values else 0,
                    'min': min(values) if values else 0,
                    'max': max(values) if values else 0,
                    'most_common': max(set(values), key=values.count) if values else None
                }
            else:
                # Categorical parameters
                from collections import Counter
                value_counts = Counter(values)
                analysis['parameter_patterns'][param] = {
                    'most_common': value_counts.most_common(1)[0] if value_counts else None,
                    'distribution': dict(value_counts)
                }
        
        # Performance comparison between categories
        roi_values = []
        drawdown_values = []
        for category, cat_analysis in analysis['strategy_categories'].items():
            roi = cat_analysis['performance_metrics']['roi']
            drawdown = cat_analysis['performance_metrics']['max_drawdown']
            roi_values.append(roi)
            drawdown_values.append(drawdown)
        
        if roi_values:
            analysis['performance_comparison'] = {
                'roi_range': {'min': min(roi_values), 'max': max(roi_values)},
                'drawdown_range': {'min': min(drawdown_values), 'max': max(drawdown_values)},
                'best_overall_roi': max(roi_values),
                'lowest_risk': min(drawdown_values)
            }
        
        # Generate recommendations
        recommendations = []
        
        # Edge threshold recommendation
        edge_pattern = analysis['parameter_patterns'].get('edge_threshold', {})
        if edge_pattern:
            recommended_edge = edge_pattern.get('median', 0)
            recommendations.append({
                'parameter': 'edge_threshold',
                'recommendation': f"Use edge threshold around {recommended_edge:.1%}",
                'reasoning': f"Median of best strategies: {recommended_edge:.1%}"
            })
        
        # Stake method recommendation
        stake_pattern = analysis['parameter_patterns'].get('stake_method', {})
        if stake_pattern and 'most_common' in stake_pattern:
            most_common_stake = stake_pattern['most_common']
            if most_common_stake:
                recommendations.append({
                    'parameter': 'stake_method',
                    'recommendation': f"Use {most_common_stake[0]} staking method",
                    'reasoning': f"Used in {most_common_stake[1]} out of {len(best_strategies)} best strategies"
                })
        
        analysis['recommendations'] = recommendations
        
        logger.info(f"✅ Best strategies analysis completed:")
        logger.info(f"   Categories analyzed: {len(analysis['strategy_categories'])}")
        logger.info(f"   Recommendations generated: {len(recommendations)}")
        
        return analysis
    
    def analyze_parameter_sensitivity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze parameter sensitivity from optimization results
        
        Args:
            results: Optimization results dictionary
            
        Returns:
            Parameter sensitivity analysis
        """
        logger.info("📈 Analyzing parameter sensitivity...")
        
        if not results.get('optimization_results'):
            return {'error': 'No optimization results found'}
        
        # Convert optimization results to DataFrame
        opt_data = []
        for result in results['optimization_results']:
            if 'error' not in result and result.get('statistics', {}).get('total_bets', 0) > 0:
                row = {
                    **result['parameters'],
                    **result['performance'],
                    **result['statistics']
                }
                opt_data.append(row)
        
        if not opt_data:
            return {'error': 'No valid optimization results found'}
        
        df = pd.DataFrame(opt_data)
        
        analysis = {
            'parameter_correlations': {},
            'performance_by_parameter': {},
            'optimal_ranges': {},
            'sensitivity_ranking': {}
        }
        
        # Calculate correlations with ROI
        numeric_params = ['edge_threshold', 'min_odds', 'stake_size']
        target_metrics = ['roi', 'max_drawdown', 'sharpe_ratio', 'total_bets']
        
        correlations = {}
        for param in numeric_params:
            if param in df.columns:
                param_correlations = {}
                for metric in target_metrics:
                    if metric in df.columns:
                        corr = df[param].corr(df[metric])
                        param_correlations[metric] = corr if not pd.isna(corr) else 0
                correlations[param] = param_correlations
        
        analysis['parameter_correlations'] = correlations
        
        # Performance by parameter value
        for param in numeric_params + ['stake_method', 'ev_method']:
            if param not in df.columns:
                continue
                
            if param in numeric_params:
                # Numerical parameters - create bins
                try:
                    bins = pd.qcut(df[param], q=5, duplicates='drop')
                    grouped = df.groupby(bins)
                except:
                    # Fallback to equal-width bins
                    bins = pd.cut(df[param], bins=5)
                    grouped = df.groupby(bins)
            else:
                # Categorical parameters
                grouped = df.groupby(param)
            
            param_performance = {}
            for group_name, group_data in grouped:
                if len(group_data) > 0:
                    param_performance[str(group_name)] = {
                        'count': len(group_data),
                        'avg_roi': group_data['roi'].mean(),
                        'avg_drawdown': group_data['max_drawdown'].mean(),
                        'avg_bets': group_data['total_bets'].mean(),
                        'profitable_strategies': (group_data['roi'] > 0).sum()
                    }
            
            analysis['performance_by_parameter'][param] = param_performance
        
        # Find optimal ranges for numerical parameters
        for param in numeric_params:
            if param not in df.columns:
                continue
            
            # Find range of top 10% strategies by ROI
            top_10_pct = df.nlargest(max(1, len(df) // 10), 'roi')
            if len(top_10_pct) > 0:
                analysis['optimal_ranges'][param] = {
                    'min': top_10_pct[param].min(),
                    'max': top_10_pct[param].max(),
                    'median': top_10_pct[param].median(),
                    'recommendation': f"{top_10_pct[param].median():.3f}"
                }
        
        # Rank parameters by their impact on ROI (absolute correlation)
        roi_impacts = {}
        for param, corrs in correlations.items():
            roi_impact = abs(corrs.get('roi', 0))
            roi_impacts[param] = roi_impact
        
        # Sort by impact
        sensitivity_ranking = sorted(roi_impacts.items(), key=lambda x: x[1], reverse=True)
        analysis['sensitivity_ranking'] = {
            'ranking': sensitivity_ranking,
            'most_impactful': sensitivity_ranking[0] if sensitivity_ranking else None,
            'least_impactful': sensitivity_ranking[-1] if sensitivity_ranking else None
        }
        
        logger.info(f"✅ Parameter sensitivity analysis completed:")
        logger.info(f"   Parameters analyzed: {len(correlations)}")
        if sensitivity_ranking:
            logger.info(f"   Most impactful parameter: {sensitivity_ranking[0][0]} (correlation: {sensitivity_ranking[0][1]:.3f})")
        
        return analysis


def run_comprehensive_analysis(results_file: Optional[str] = None, 
                              show_available_files: bool = False):
    """
    Run comprehensive performance analysis
    
    Args:
        results_file: Specific file to analyze (None = auto-select latest)
        show_available_files: If True, show all available files and prompt for selection
    """
    logger.info("🚀 Starting comprehensive performance analysis...")
    
    analyzer = PerformanceAnalyzer()
    
    # Show available files if requested
    if show_available_files and results_file is None:
        patterns = ['*backtest*.json', '*optimization*.json', '*results*.json']
        all_files = []
        for pattern in patterns:
            full_pattern = os.path.join(analyzer.results_dir, pattern)
            files = glob.glob(full_pattern)
            all_files.extend(files)
        
        all_files = list(set(all_files))
        all_files.sort(key=os.path.getmtime, reverse=True)
        
        if all_files:
            print("\n📁 Available results files:")
            for i, file in enumerate(all_files):
                file_time = datetime.fromtimestamp(os.path.getmtime(file))
                print(f"   {i+1}. {os.path.basename(file)} ({file_time.strftime('%Y-%m-%d %H:%M')})")
            
            try:
                choice = input(f"\nSelect file (1-{len(all_files)}) or press Enter for latest: ").strip()
                if choice:
                    idx = int(choice) - 1
                    if 0 <= idx < len(all_files):
                        results_file = all_files[idx]
                        print(f"Selected: {os.path.basename(results_file)}")
            except (ValueError, IndexError):
                print("Invalid selection, using latest file")
    
    # Load results
    results = analyzer.load_backtest_results(results_file)
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(
        results,
        include_quarterly=True,
        include_risk=True,
        include_model=True
    )
    
    # Save results
    output_file = analyzer.save_analysis_results(report)
    
    logger.info("✅ Comprehensive analysis completed!")
    return report, output_file


if __name__ == "__main__":
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # 🎯 CONFIGURATION:
    SHOW_FILE_SELECTION = True  # Set to True to see all available files
    SPECIFIC_FILE = None        # Or specify exact file: 'backtest_results_20250721_123456.json'
    
    # Run comprehensive analysis
    report, output_file = run_comprehensive_analysis(
        results_file=SPECIFIC_FILE,
        show_available_files=SHOW_FILE_SELECTION
    )