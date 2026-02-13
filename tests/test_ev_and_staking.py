"""
Unit testy pro EV kalkulace a stake sizing — čisté funkce bez DB závislostí.
"""

import pytest
from src.betting.ev_calculations import calculate_ev_variants, calculate_stake_size


# ──────────────────────────────────────────────
# calculate_ev_variants()
# ──────────────────────────────────────────────

class TestCalculateEvVariants:
    """Testy pro výpočet Expected Value variant."""

    def test_positive_ev_value_bet(self):
        """When model_prob * odds > 1, basic EV should be positive."""
        result = calculate_ev_variants(0.55, 2.0)
        assert result['basic_ev'] == pytest.approx(0.10)

    def test_negative_ev_no_value(self):
        """When model_prob * odds < 1, basic EV should be negative."""
        result = calculate_ev_variants(0.40, 2.0)
        assert result['basic_ev'] == pytest.approx(-0.20)

    def test_zero_ev_fair_odds(self):
        """When model_prob * odds = 1, EV should be zero."""
        result = calculate_ev_variants(0.50, 2.0)
        assert result['basic_ev'] == pytest.approx(0.0)

    def test_kelly_fraction_positive_value(self):
        """Kelly fraction should be positive for value bets."""
        result = calculate_ev_variants(0.60, 2.0)
        assert result['kelly_fraction'] == pytest.approx(0.20)

    def test_kelly_fraction_capped(self):
        """Kelly fraction should not exceed max_kelly."""
        result = calculate_ev_variants(0.9, 5.0, max_kelly=0.25)
        assert result['kelly_fraction'] == pytest.approx(0.25)

    def test_kelly_fraction_zero_for_no_value(self):
        """Kelly fraction should be 0 when there's no value."""
        result = calculate_ev_variants(0.40, 2.0)
        assert result['kelly_fraction'] == pytest.approx(0.0)

    def test_kelly_enhanced_ev(self):
        """Kelly-enhanced EV should be basic_ev * kelly_fraction."""
        result = calculate_ev_variants(0.60, 2.0)
        expected_kelly_ev = result['basic_ev'] * result['kelly_fraction']
        assert result['kelly_enhanced_ev'] == pytest.approx(expected_kelly_ev)

    def test_confidence_auto_calculated(self):
        """When confidence is None, it should be calculated as abs(prob - 0.5) * 2."""
        result = calculate_ev_variants(0.70, 2.0)
        expected_conf = abs(0.70 - 0.5) * 2
        assert result['confidence'] == pytest.approx(expected_conf)

    def test_confidence_provided(self):
        """When confidence is provided explicitly, it should be used."""
        result = calculate_ev_variants(0.60, 2.0, confidence=0.8)
        assert result['confidence'] == pytest.approx(0.8)

    def test_confidence_weighted_ev(self):
        """Confidence-weighted EV = basic_ev * (0.5 + 0.5 * confidence)."""
        result = calculate_ev_variants(0.60, 2.0, confidence=0.8)
        expected = result['basic_ev'] * (0.5 + 0.5 * 0.8)
        assert result['confidence_weighted_ev'] == pytest.approx(expected)

    def test_odds_equal_one(self):
        """Edge case: odds = 1.0 → kelly_enhanced_ev should be 0."""
        result = calculate_ev_variants(0.60, 1.0)
        assert result['kelly_enhanced_ev'] == 0
        assert result['kelly_fraction'] == 0

    def test_high_odds_favorite(self):
        """Strong favorite with high odds."""
        result = calculate_ev_variants(0.80, 1.50)
        assert result['basic_ev'] == pytest.approx(0.20)

    def test_all_keys_present(self):
        """Result should contain all expected keys."""
        result = calculate_ev_variants(0.55, 2.0)
        expected_keys = {'basic_ev', 'kelly_enhanced_ev', 'confidence_weighted_ev',
                         'kelly_fraction', 'confidence'}
        assert set(result.keys()) == expected_keys


# ──────────────────────────────────────────────
# calculate_stake_size()
# ──────────────────────────────────────────────

class TestCalculateStakeSize:
    """Testy pro výpočet velikosti sázky."""

    # --- Fixed staking ---

    def test_fixed_stake_basic(self):
        """Fixed stake = bankroll * stake_size."""
        stake = calculate_stake_size(
            ev_value=0.10, odds=2.0, model_prob=0.55,
            current_bankroll=10000.0,
            stake_method='fixed', stake_size=0.02
        )
        assert stake == pytest.approx(200.0)

    def test_fixed_stake_different_bankroll(self):
        stake = calculate_stake_size(
            ev_value=0.10, odds=2.0, model_prob=0.55,
            current_bankroll=5000.0,
            stake_method='fixed', stake_size=0.02
        )
        assert stake == pytest.approx(100.0)

    # --- Zero/Negative EV ---

    def test_zero_ev_returns_zero(self):
        """When EV <= 0, stake should be 0."""
        stake = calculate_stake_size(
            ev_value=0.0, odds=2.0, model_prob=0.50,
            current_bankroll=10000.0,
            stake_method='fixed', stake_size=0.02
        )
        assert stake == 0.0

    def test_negative_ev_returns_zero(self):
        stake = calculate_stake_size(
            ev_value=-0.10, odds=2.0, model_prob=0.45,
            current_bankroll=10000.0,
            stake_method='fixed', stake_size=0.02
        )
        assert stake == 0.0

    # --- Kelly staking ---

    def test_kelly_stake_calculation(self):
        """Kelly stake = bankroll * kelly_fraction * stake_size."""
        stake = calculate_stake_size(
            ev_value=0.10, odds=2.0, model_prob=0.60,
            current_bankroll=10000.0,
            stake_method='kelly', stake_size=0.5
        )
        assert stake == pytest.approx(1000.0)

    def test_kelly_no_value_returns_zero(self):
        """Kelly with no value should return 0."""
        stake = calculate_stake_size(
            ev_value=0.01, odds=2.0, model_prob=0.45,
            current_bankroll=10000.0,
            stake_method='kelly', stake_size=0.5
        )
        assert stake == 0.0

    def test_kelly_odds_lte_one(self):
        """Kelly with odds <= 1.0 should return 0."""
        stake = calculate_stake_size(
            ev_value=0.10, odds=1.0, model_prob=0.60,
            current_bankroll=10000.0,
            stake_method='kelly', stake_size=0.5
        )
        assert stake == 0.0

    # --- Hybrid staking ---

    def test_hybrid_stake_calculation(self):
        """Hybrid = base_stake + kelly_adjustment."""
        stake = calculate_stake_size(
            ev_value=0.10, odds=2.0, model_prob=0.60,
            current_bankroll=10000.0,
            stake_method='hybrid', stake_size=0.04
        )
        assert stake == pytest.approx(240.0)

    # --- Max stake cap ---

    def test_max_stake_cap(self):
        """Stake should not exceed max_stake_pct of bankroll."""
        stake = calculate_stake_size(
            ev_value=0.10, odds=2.0, model_prob=0.60,
            current_bankroll=10000.0,
            stake_method='fixed', stake_size=0.20,
            max_stake_pct=0.05
        )
        assert stake == pytest.approx(500.0)

    # --- Minimum stake ---

    def test_minimum_stake(self):
        """Very small stake should be bumped to min_stake."""
        stake = calculate_stake_size(
            ev_value=0.10, odds=2.0, model_prob=0.55,
            current_bankroll=10.0,
            stake_method='fixed', stake_size=0.02,
            min_stake=1.0
        )
        assert stake == pytest.approx(1.0)

    # --- Unknown method ---

    def test_unknown_method_raises(self):
        """Unknown stake method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown stake method"):
            calculate_stake_size(
                ev_value=0.10, odds=2.0, model_prob=0.55,
                current_bankroll=10000.0,
                stake_method='unknown'
            )

    # --- Rounding ---

    def test_stake_rounded_to_2_decimals(self):
        """Stake should be rounded to 2 decimal places."""
        stake = calculate_stake_size(
            ev_value=0.10, odds=2.0, model_prob=0.55,
            current_bankroll=10000.0,
            stake_method='fixed', stake_size=0.0333
        )
        assert stake == round(stake, 2)
