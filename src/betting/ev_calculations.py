"""
Pure Expected Value and stake sizing calculations — no dependencies.
Used for unit testing and can be imported without side effects.
"""

from typing import Dict, Optional


def calculate_ev_variants(model_prob: float,
                          bookmaker_odds: float,
                          confidence: Optional[float] = None,
                          max_kelly: float = 0.25) -> Dict[str, float]:
    """
    Calculate multiple Expected Value variants.

    Args:
        model_prob: Model's predicted probability
        bookmaker_odds: Bookmaker's decimal odds
        confidence: Model confidence (0-1), calculated if None
        max_kelly: Maximum Kelly fraction cap

    Returns:
        Dictionary with EV calculations
    """
    if confidence is None:
        confidence = abs(model_prob - 0.5) * 2

    basic_ev = (model_prob * bookmaker_odds) - 1

    if bookmaker_odds > 1.0:
        kelly_fraction = (bookmaker_odds * model_prob - 1) / (bookmaker_odds - 1)
        kelly_fraction = max(0, min(kelly_fraction, max_kelly))
        kelly_enhanced_ev = basic_ev * kelly_fraction if kelly_fraction > 0 else 0
    else:
        kelly_enhanced_ev = 0
        kelly_fraction = 0

    confidence_weighted_ev = basic_ev * (0.5 + 0.5 * confidence)

    return {
        'basic_ev': basic_ev,
        'kelly_enhanced_ev': kelly_enhanced_ev,
        'confidence_weighted_ev': confidence_weighted_ev,
        'kelly_fraction': kelly_fraction,
        'confidence': confidence
    }


def calculate_stake_size(ev_value: float,
                         odds: float,
                         model_prob: float,
                         current_bankroll: float,
                         stake_method: str = 'fixed',
                         stake_size: float = 0.02,
                         max_stake_pct: float = 0.10,
                         min_stake: float = 1.0) -> float:
    """
    Calculate stake size based on specified method.

    Args:
        ev_value: Expected value of the bet
        odds: Bookmaker odds
        model_prob: Model's predicted probability
        current_bankroll: Current bankroll amount
        stake_method: 'fixed', 'kelly', or 'hybrid'
        stake_size: Fixed percentage or Kelly multiplier
        max_stake_pct: Maximum stake as percentage of bankroll
        min_stake: Minimum bet amount

    Returns:
        Stake amount in currency units
    """
    if ev_value <= 0:
        return 0.0

    if stake_method == 'fixed':
        stake_amount = current_bankroll * stake_size

    elif stake_method == 'kelly':
        if odds > 1.0:
            kelly_fraction = (odds * model_prob - 1) / (odds - 1)
            kelly_fraction = max(0, kelly_fraction * stake_size)
            stake_amount = current_bankroll * kelly_fraction
        else:
            stake_amount = 0.0

    elif stake_method == 'hybrid':
        base_stake = current_bankroll * (stake_size * 0.5)
        kelly_fraction = (odds * model_prob - 1) / (odds - 1) if odds > 1.0 else 0
        kelly_adjustment = current_bankroll * kelly_fraction * (stake_size * 0.5)
        stake_amount = base_stake + max(0, kelly_adjustment)
    else:
        raise ValueError(f"Unknown stake method: {stake_method}")

    max_stake = current_bankroll * max_stake_pct
    stake_amount = min(stake_amount, max_stake)
    stake_amount = max(stake_amount, min_stake) if stake_amount > 0 else 0.0

    return round(stake_amount, 2)
