"""
Statistical tests for performance analysis.

Pure statistical testing functions extracted from performance_analyzer.py.
Uses scipy.stats for statistical calculations.

Location: src/betting/statistical_tests.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy import stats


def perform_quarterly_statistical_tests(quarterly_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform statistical tests on quarterly performance data.

    Args:
        quarterly_df: DataFrame with quarterly performance metrics
                     Must contain 'roi' column

    Returns:
        Dictionary with statistical test results
    """
    tests = {}

    if len(quarterly_df) < 2:
        return {'note': 'Insufficient quarters for statistical testing'}

    try:
        roi_values = quarterly_df['roi'].values

        # Test for significance: one-sample t-test against zero
        if len(roi_values) > 1:
            t_stat, p_value = stats.ttest_1samp(roi_values, 0)
            tests['roi_significance'] = {
                'test': 'one_sample_t_test',
                'null_hypothesis': 'quarterly ROI = 0',
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_at_5pct': p_value < 0.05,
                'interpretation': (
                    'Significantly different from zero' if p_value < 0.05
                    else 'Not significantly different from zero'
                )
            }

        # Test for normality of returns
        if len(roi_values) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(roi_values)
            tests['normality_test'] = {
                'test': 'shapiro_wilk',
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'normally_distributed': shapiro_p > 0.05
            }

        # Test for trend (correlation with time)
        if len(roi_values) >= 3:
            time_index = range(len(roi_values))
            correlation, corr_p = stats.pearsonr(time_index, roi_values)
            tests['trend_analysis'] = {
                'test': 'pearson_correlation',
                'correlation': float(correlation),
                'p_value': float(corr_p),
                'significant_trend': corr_p < 0.05,
                'trend_direction': (
                    'improving' if correlation > 0
                    else 'declining' if correlation < 0
                    else 'stable'
                )
            }

    except Exception as e:
        tests['error'] = str(e)

    return tests


def calculate_roi_confidence_interval(roi_values: np.ndarray,
                                      confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Calculate confidence interval for ROI values.

    Args:
        roi_values: Array of ROI values
        confidence_level: Confidence level (default 0.95)

    Returns:
        Dictionary with confidence interval bounds
    """
    if len(roi_values) < 2:
        return {'error': 'Insufficient data for confidence interval'}

    mean_roi = np.mean(roi_values)
    std_roi = np.std(roi_values, ddof=1)
    n = len(roi_values)

    # t-distribution for small samples
    t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 1)
    margin_error = t_critical * (std_roi / np.sqrt(n))

    return {
        'mean': float(mean_roi),
        'std': float(std_roi),
        'lower_bound': float(mean_roi - margin_error),
        'upper_bound': float(mean_roi + margin_error),
        'confidence_level': confidence_level,
        'sample_size': n
    }


def test_strategy_difference(strategy1_returns: np.ndarray,
                             strategy2_returns: np.ndarray) -> Dict[str, Any]:
    """
    Test if two strategies have significantly different returns.

    Args:
        strategy1_returns: Returns from first strategy
        strategy2_returns: Returns from second strategy

    Returns:
        Dictionary with comparison test results
    """
    if len(strategy1_returns) < 2 or len(strategy2_returns) < 2:
        return {'error': 'Insufficient data for comparison'}

    # Welch's t-test (does not assume equal variances)
    t_stat, p_value = stats.ttest_ind(strategy1_returns, strategy2_returns, equal_var=False)

    # Mann-Whitney U test (non-parametric alternative)
    u_stat, u_pvalue = stats.mannwhitneyu(strategy1_returns, strategy2_returns,
                                          alternative='two-sided')

    return {
        'welch_t_test': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_difference': p_value < 0.05
        },
        'mann_whitney_u': {
            'u_statistic': float(u_stat),
            'p_value': float(u_pvalue),
            'significant_difference': u_pvalue < 0.05
        },
        'strategy1_mean': float(np.mean(strategy1_returns)),
        'strategy2_mean': float(np.mean(strategy2_returns)),
        'mean_difference': float(np.mean(strategy1_returns) - np.mean(strategy2_returns))
    }


def test_randomness(returns: np.ndarray) -> Dict[str, Any]:
    """
    Test if returns show patterns (non-randomness).

    Args:
        returns: Array of returns

    Returns:
        Dictionary with randomness test results
    """
    if len(returns) < 10:
        return {'error': 'Insufficient data for randomness testing'}

    tests = {}

    # Runs test for randomness
    median = np.median(returns)
    runs = 1
    above_median = returns[0] > median

    for ret in returns[1:]:
        if (ret > median) != above_median:
            runs += 1
            above_median = ret > median

    n_above = np.sum(returns > median)
    n_below = len(returns) - n_above

    # Expected runs and standard deviation
    expected_runs = (2 * n_above * n_below) / len(returns) + 1
    std_runs = np.sqrt(
        (2 * n_above * n_below * (2 * n_above * n_below - len(returns))) /
        (len(returns) ** 2 * (len(returns) - 1))
    )

    z_stat = (runs - expected_runs) / std_runs if std_runs > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    tests['runs_test'] = {
        'observed_runs': runs,
        'expected_runs': float(expected_runs),
        'z_statistic': float(z_stat),
        'p_value': float(p_value),
        'is_random': p_value > 0.05
    }

    # Autocorrelation at lag 1
    if len(returns) > 2:
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        tests['autocorrelation_lag1'] = {
            'value': float(autocorr) if not np.isnan(autocorr) else 0.0,
            'significant': abs(autocorr) > 2 / np.sqrt(len(returns))
        }

    return tests
