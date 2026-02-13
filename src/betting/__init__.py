"""
Betting module - Betting analysis, backtesting, and risk management.

Location: src/betting/__init__.py
"""

from src.betting.ev_calculations import (
    calculate_ev_variants,
    calculate_stake_size
)
from src.betting.statistical_tests import (
    perform_quarterly_statistical_tests,
    calculate_roi_confidence_interval,
    test_strategy_difference,
    test_randomness
)
from src.betting.risk_metrics import (
    calculate_var_metrics,
    calculate_drawdown_metrics,
    calculate_recovery_periods,
    calculate_max_consecutive,
    calculate_streak_analysis,
    calculate_conditional_probability,
    calculate_stake_analysis,
    calculate_odds_analysis,
    calculate_monthly_risk,
    calculate_sharpe_ratio,
    calculate_sortino_ratio
)

__all__ = [
    # EV calculations
    'calculate_ev_variants',
    'calculate_stake_size',
    # Statistical tests
    'perform_quarterly_statistical_tests',
    'calculate_roi_confidence_interval',
    'test_strategy_difference',
    'test_randomness',
    # Risk metrics
    'calculate_var_metrics',
    'calculate_drawdown_metrics',
    'calculate_recovery_periods',
    'calculate_max_consecutive',
    'calculate_streak_analysis',
    'calculate_conditional_probability',
    'calculate_stake_analysis',
    'calculate_odds_analysis',
    'calculate_monthly_risk',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio'
]
