"""
Unit testy pro Elo výpočty — čisté funkce bez DB závislostí.
"""

import pytest
from src.models.elo_calculations import (
    expected_score,
    game_result_to_score,
    calculate_rating_update,
    apply_season_regression,
)


# ──────────────────────────────────────────────
# expected_score()
# ──────────────────────────────────────────────

class TestExpectedScore:
    """Testy pro Elo expected score (win probability) formuli."""

    def test_equal_ratings_no_advantage(self):
        assert expected_score(1500, 1500, home_advantage=0) == pytest.approx(0.5)

    def test_equal_ratings_with_home_advantage(self):
        result = expected_score(1500, 1500, home_advantage=50)
        assert result > 0.5
        assert result == pytest.approx(1 / (1 + 10 ** (-50 / 400)), abs=1e-6)

    def test_higher_rating_wins(self):
        result = expected_score(1600, 1400)
        assert result > 0.5

    def test_lower_rating_loses(self):
        result = expected_score(1400, 1600)
        assert result < 0.5

    def test_symmetry(self):
        """P(A beats B) + P(B beats A) = 1."""
        p_a = expected_score(1600, 1400)
        p_b = expected_score(1400, 1600)
        assert p_a + p_b == pytest.approx(1.0, abs=1e-10)

    def test_400_point_difference(self):
        """Elo standard: 400-point difference ~ 10:1 odds (0.909)."""
        result = expected_score(1900, 1500)
        assert result == pytest.approx(10 / 11, abs=1e-6)

    def test_extreme_difference(self):
        result = expected_score(2000, 1000)
        assert result > 0.99

    def test_result_between_0_and_1(self):
        for diff in [-800, -400, -100, 0, 100, 400, 800]:
            result = expected_score(1500 + diff, 1500)
            assert 0.0 < result < 1.0


# ──────────────────────────────────────────────
# game_result_to_score()
# ──────────────────────────────────────────────

class TestGameResultToScore:
    """Testy pro konverzi výsledku zápasu na Elo skóre."""

    def test_home_win_regulation(self):
        score, result_type = game_result_to_score(3, 1)
        assert score == 1.0
        assert result_type == 'HOME_WIN_REG'

    def test_away_win_regulation(self):
        score, result_type = game_result_to_score(1, 3)
        assert score == 0.0
        assert result_type == 'AWAY_WIN_REG'

    def test_home_win_overtime(self):
        score, result_type = game_result_to_score(4, 3, 'OT')
        assert score == 0.6
        assert result_type == 'HOME_WIN_OT'

    def test_home_win_shootout(self):
        score, result_type = game_result_to_score(2, 1, 'SO')
        assert score == 0.6
        assert result_type == 'HOME_WIN_SO'

    def test_away_win_overtime(self):
        score, result_type = game_result_to_score(3, 4, 'OT')
        assert score == 0.4
        assert result_type == 'AWAY_WIN_OT'

    def test_away_win_shootout(self):
        score, result_type = game_result_to_score(1, 2, 'SO')
        assert score == 0.4
        assert result_type == 'AWAY_WIN_SO'

    def test_tie(self):
        score, result_type = game_result_to_score(3, 3)
        assert score == 0.5
        assert result_type == 'TIE'

    def test_ot_values_sum_to_one(self):
        """OT/SO výhra + porážka by měla dát 1.0 (zero-sum)."""
        win_score, _ = game_result_to_score(4, 3, 'OT')
        loss_score, _ = game_result_to_score(3, 4, 'OT')
        assert win_score + loss_score == pytest.approx(1.0)


# ──────────────────────────────────────────────
# calculate_rating_update()
# ──────────────────────────────────────────────

class TestRatingUpdate:
    """Testy pro aktualizaci Elo ratingů po zápase."""

    def test_winner_rating_increases(self):
        new_a, new_b = calculate_rating_update(1500.0, 1500.0, actual_score=1.0, k_factor=32.0)
        assert new_a > 1500.0
        assert new_b < 1500.0

    def test_loser_rating_decreases(self):
        new_a, new_b = calculate_rating_update(1500.0, 1500.0, actual_score=0.0, k_factor=32.0)
        assert new_a < 1500.0
        assert new_b > 1500.0

    def test_ratings_zero_sum(self):
        """Rating changes should sum to approximately zero."""
        new_a, new_b = calculate_rating_update(1500.0, 1500.0, actual_score=1.0, k_factor=32.0)
        change_a = new_a - 1500.0
        change_b = new_b - 1500.0
        assert change_a + change_b == pytest.approx(0.0, abs=1e-10)

    def test_equal_teams_expected_change(self):
        """Two equal teams: winner gains k/2, loser loses k/2."""
        new_a, _ = calculate_rating_update(1500.0, 1500.0, actual_score=1.0, k_factor=32.0)
        assert new_a == pytest.approx(1516.0, abs=1e-6)

    def test_upset_gives_larger_change(self):
        """Upset (weaker team wins) should produce larger rating changes."""
        new_a_upset, _ = calculate_rating_update(1400.0, 1600.0, actual_score=1.0, k_factor=32.0)
        upset_change = new_a_upset - 1400.0

        new_a_normal, _ = calculate_rating_update(1500.0, 1500.0, actual_score=1.0, k_factor=32.0)
        normal_change = new_a_normal - 1500.0

        assert upset_change > normal_change

    def test_k_multiplier(self):
        """k_multiplier should scale rating changes proportionally."""
        new_a_1x, _ = calculate_rating_update(1500.0, 1500.0, actual_score=1.0, k_factor=32.0, k_multiplier=1.0)
        change_1x = new_a_1x - 1500.0

        new_a_2x, _ = calculate_rating_update(1500.0, 1500.0, actual_score=1.0, k_factor=32.0, k_multiplier=2.0)
        change_2x = new_a_2x - 1500.0

        assert change_2x == pytest.approx(change_1x * 2.0, abs=1e-6)

    def test_home_advantage_effect(self):
        """Home advantage should reduce gains for home winner."""
        new_a_no_ha, _ = calculate_rating_update(1500.0, 1500.0, actual_score=1.0, k_factor=32.0, home_advantage=0)
        change_no_ha = new_a_no_ha - 1500.0

        new_a_ha, _ = calculate_rating_update(1500.0, 1500.0, actual_score=1.0, k_factor=32.0, home_advantage=50)
        change_ha = new_a_ha - 1500.0

        assert change_ha < change_no_ha


# ──────────────────────────────────────────────
# apply_season_regression()
# ──────────────────────────────────────────────

class TestSeasonRegression:
    """Testy pro regresi ratingů k průměru mezi sezónami."""

    def test_regression_moves_toward_mean(self):
        ratings = {1: 1600.0, 2: 1400.0}
        regressed = apply_season_regression(ratings, 0.3)
        assert regressed[1] < 1600.0
        assert regressed[2] > 1400.0

    def test_regression_preserves_mean(self):
        """Season regression should not change the overall mean."""
        ratings = {1: 1600.0, 2: 1500.0, 3: 1400.0}
        mean_before = sum(ratings.values()) / len(ratings)
        regressed = apply_season_regression(ratings, 0.3)
        mean_after = sum(regressed.values()) / len(regressed)
        assert mean_before == pytest.approx(mean_after, abs=1e-10)

    def test_zero_regression_no_change(self):
        ratings = {1: 1600.0, 2: 1400.0}
        regressed = apply_season_regression(ratings, 0.0)
        assert regressed[1] == pytest.approx(1600.0)
        assert regressed[2] == pytest.approx(1400.0)

    def test_full_regression_resets_to_mean(self):
        ratings = {1: 1600.0, 2: 1400.0}
        regressed = apply_season_regression(ratings, 1.0)
        assert regressed[1] == pytest.approx(1500.0)
        assert regressed[2] == pytest.approx(1500.0)

    def test_regression_amount(self):
        """Check exact regression formula: new = old + regression * (mean - old)."""
        ratings = {1: 1600.0, 2: 1400.0}
        regressed = apply_season_regression(ratings, 0.3)
        # mean = 1500
        # new_1 = 1600 + 0.3 * (1500 - 1600) = 1600 - 30 = 1570
        # new_2 = 1400 + 0.3 * (1500 - 1400) = 1400 + 30 = 1430
        assert regressed[1] == pytest.approx(1570.0)
        assert regressed[2] == pytest.approx(1430.0)

    def test_empty_ratings_returns_empty(self):
        regressed = apply_season_regression({}, 0.3)
        assert regressed == {}
