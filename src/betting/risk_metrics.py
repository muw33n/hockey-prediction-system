"""
Risk metrics calculations for betting performance analysis.

Pure risk metric functions extracted from performance_analyzer.py.
Calculates VaR, drawdowns, streaks, and other risk measures.

Location: src/betting/risk_metrics.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


def calculate_var_metrics(daily_returns: pd.Series) -> Dict[str, float]:
    """
    Calculate Value at Risk (VaR) and Expected Shortfall metrics.

    Args:
        daily_returns: Series of daily returns/ROI values

    Returns:
        Dictionary with VaR and ES metrics
    """
    daily_returns = daily_returns.dropna()

    if len(daily_returns) == 0:
        return {
            'var_95_daily': 0.0,
            'var_99_daily': 0.0,
            'expected_shortfall_95': 0.0,
            'expected_shortfall_99': 0.0
        }

    var_95 = float(np.percentile(daily_returns, 5))  # 5th percentile (95% VaR)
    var_99 = float(np.percentile(daily_returns, 1))  # 1st percentile (99% VaR)

    # Expected Shortfall (Conditional VaR) - average loss beyond VaR
    es_95_values = daily_returns[daily_returns <= var_95]
    es_99_values = daily_returns[daily_returns <= var_99]

    es_95 = float(es_95_values.mean()) if len(es_95_values) > 0 else var_95
    es_99 = float(es_99_values.mean()) if len(es_99_values) > 0 else var_99

    return {
        'var_95_daily': var_95,
        'var_99_daily': var_99,
        'expected_shortfall_95': es_95,
        'expected_shortfall_99': es_99
    }


def calculate_drawdown_metrics(cumulative_pnl: pd.Series) -> Dict[str, Any]:
    """
    Calculate drawdown metrics from cumulative P&L series.

    Args:
        cumulative_pnl: Series of cumulative profit/loss

    Returns:
        Dictionary with drawdown metrics
    """
    if len(cumulative_pnl) == 0:
        return {
            'max_drawdown_absolute': 0.0,
            'max_drawdown_percent': 0.0,
            'current_drawdown': 0.0,
            'drawdown_periods': 0,
            'recovery_periods': {'avg_recovery_days': 0, 'max_recovery_days': 0}
        }

    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    drawdown_pct = drawdown / abs(running_max).replace(0, 1)

    recovery_stats = calculate_recovery_periods(drawdown)

    return {
        'max_drawdown_absolute': float(abs(drawdown.min())),
        'max_drawdown_percent': float(abs(drawdown_pct.min())),
        'current_drawdown': float(drawdown.iloc[-1]) if len(drawdown) > 0 else 0.0,
        'drawdown_periods': int((drawdown < 0).sum()),
        'recovery_periods': recovery_stats
    }


def calculate_recovery_periods(drawdown: pd.Series) -> Dict[str, float]:
    """
    Calculate drawdown recovery statistics.

    Args:
        drawdown: Series of drawdown values (negative when in drawdown)

    Returns:
        Dictionary with recovery period statistics
    """
    if len(drawdown) == 0:
        return {'avg_recovery_days': 0, 'max_recovery_days': 0, 'total_drawdown_periods': 0}

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
            'avg_recovery_days': float(np.mean(drawdown_periods)),
            'max_recovery_days': int(max(drawdown_periods)),
            'total_drawdown_periods': len(drawdown_periods)
        }

    return {'avg_recovery_days': 0, 'max_recovery_days': 0, 'total_drawdown_periods': 0}


def calculate_max_consecutive(series: pd.Series) -> int:
    """
    Calculate maximum consecutive True values in a boolean series.

    Args:
        series: Boolean pandas Series

    Returns:
        Maximum count of consecutive True values
    """
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


def calculate_streak_analysis(bet_won: pd.Series) -> Dict[str, Any]:
    """
    Analyze winning and losing streaks.

    Args:
        bet_won: Boolean series indicating bet wins

    Returns:
        Dictionary with streak analysis
    """
    bet_won_clean = bet_won.fillna(False).astype(bool)

    # Calculate losing streaks
    streak_id = (bet_won_clean != bet_won_clean.shift()).cumsum()
    losing_streaks = bet_won_clean[~bet_won_clean].groupby(streak_id[~bet_won_clean]).size()
    winning_streaks = bet_won_clean[bet_won_clean].groupby(streak_id[bet_won_clean]).size()

    return {
        'max_losing_streak': int(losing_streaks.max()) if len(losing_streaks) > 0 else 0,
        'avg_losing_streak': float(losing_streaks.mean()) if len(losing_streaks) > 0 else 0,
        'max_winning_streak': int(winning_streaks.max()) if len(winning_streaks) > 0 else 0,
        'avg_winning_streak': float(winning_streaks.mean()) if len(winning_streaks) > 0 else 0,
        'consecutive_wins_max': calculate_max_consecutive(bet_won_clean),
        'consecutive_losses_max': calculate_max_consecutive(~bet_won_clean)
    }


def calculate_conditional_probability(bet_won: pd.Series, pattern: str) -> float:
    """
    Calculate conditional probabilities for win/loss patterns.

    Args:
        bet_won: Boolean series indicating bet wins
        pattern: 'loss_then_win' or 'win_then_loss'

    Returns:
        Conditional probability (0.0 to 1.0)
    """
    if len(bet_won) < 2:
        return 0.0

    bet_won_clean = bet_won.fillna(False).astype(bool)

    if pattern == 'loss_then_win':
        # P(Win | Previous Loss)
        prev_loss = ~bet_won_clean.shift(1)
        current_win = bet_won_clean
        valid_mask = prev_loss.notna()

        if valid_mask.sum() == 0:
            return 0.0

        denominator = (prev_loss & valid_mask).sum()
        if denominator == 0:
            return 0.0

        return float((prev_loss & current_win & valid_mask).sum() / denominator)

    elif pattern == 'win_then_loss':
        # P(Loss | Previous Win)
        prev_win = bet_won_clean.shift(1)
        current_loss = ~bet_won_clean
        valid_mask = prev_win.notna()

        if valid_mask.sum() == 0:
            return 0.0

        denominator = (prev_win & valid_mask).sum()
        if denominator == 0:
            return 0.0

        return float((prev_win & current_loss & valid_mask).sum() / denominator)

    return 0.0


def calculate_stake_analysis(stakes: pd.Series) -> Dict[str, float]:
    """
    Analyze stake size distribution.

    Args:
        stakes: Series of stake amounts

    Returns:
        Dictionary with stake analysis
    """
    return {
        'min_stake': float(stakes.min()),
        'max_stake': float(stakes.max()),
        'avg_stake': float(stakes.mean()),
        'median_stake': float(stakes.median()),
        'stake_std': float(stakes.std()),
        'stake_concentration': float((stakes > stakes.quantile(0.9)).sum() / len(stakes))
    }


def calculate_odds_analysis(odds: pd.Series) -> Dict[str, float]:
    """
    Analyze odds distribution.

    Args:
        odds: Series of odds values

    Returns:
        Dictionary with odds analysis
    """
    return {
        'min_odds': float(odds.min()),
        'max_odds': float(odds.max()),
        'avg_odds': float(odds.mean()),
        'median_odds': float(odds.median()),
        'low_odds_bets': float((odds < 1.5).sum() / len(odds)),
        'high_odds_bets': float((odds > 2.5).sum() / len(odds))
    }


def calculate_monthly_risk(bet_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate monthly risk metrics.

    Args:
        bet_df: DataFrame with bet data including 'date' and 'net_result' columns

    Returns:
        Dictionary with monthly risk metrics
    """
    if len(bet_df) < 30:
        return {}

    bet_df = bet_df.copy()
    bet_df['month'] = pd.to_datetime(bet_df['date']).dt.to_period('M')
    monthly_pnl = bet_df.groupby('month')['net_result'].sum()

    return {
        'monthly_volatility': float(monthly_pnl.std()),
        'worst_month': float(monthly_pnl.min()),
        'best_month': float(monthly_pnl.max()),
        'negative_months': int((monthly_pnl < 0).sum()),
        'total_months': len(monthly_pnl)
    }


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                          periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    return float(np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std())


def calculate_sortino_ratio(returns: pd.Series, target_return: float = 0.0,
                           periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sortino ratio (uses downside deviation).

    Args:
        returns: Series of periodic returns
        target_return: Target return (default 0)
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return float('inf')  # No downside - infinite ratio

    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    if downside_std == 0:
        return 0.0

    return float(np.sqrt(periods_per_year) * (returns.mean() - target_return) / downside_std)
