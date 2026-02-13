"""
Pure Elo calculation functions — no dependencies on DB or config.
Used for unit testing and can be imported without side effects.
"""

from typing import Tuple


def expected_score(rating_a: float, rating_b: float, home_advantage: float = 0) -> float:
    """
    Calculate expected score for team A against team B using Elo formula.

    Args:
        rating_a: Elo rating of team A
        rating_b: Elo rating of team B
        home_advantage: Additional rating for home team

    Returns:
        Expected win probability for team A (0-1)
    """
    adjusted_rating_a = rating_a + home_advantage
    rating_diff = adjusted_rating_a - rating_b
    return 1 / (1 + 10 ** (-rating_diff / 400))


def game_result_to_score(home_score: int, away_score: int,
                         overtime_shootout: str = '') -> Tuple[float, str]:
    """
    Convert game result to Elo score format.

    Args:
        home_score: Final home team score
        away_score: Final away team score
        overtime_shootout: 'OT', 'SO', or empty string

    Returns:
        Tuple (home_team_score, result_type)
        home_team_score: 1.0 win, 0.0 loss, 0.6 OT/SO win, 0.4 OT/SO loss
    """
    if home_score > away_score:
        if overtime_shootout in ['OT', 'SO']:
            return 0.6, f'HOME_WIN_{overtime_shootout}'
        else:
            return 1.0, 'HOME_WIN_REG'
    elif away_score > home_score:
        if overtime_shootout in ['OT', 'SO']:
            return 0.4, f'AWAY_WIN_{overtime_shootout}'
        else:
            return 0.0, 'AWAY_WIN_REG'
    else:
        return 0.5, 'TIE'


def calculate_rating_update(rating_a: float, rating_b: float,
                            actual_score: float, k_factor: float,
                            home_advantage: float = 0,
                            k_multiplier: float = 1.0) -> Tuple[float, float]:
    """
    Calculate new Elo ratings after a game.

    Args:
        rating_a: Current rating of team A
        rating_b: Current rating of team B
        actual_score: 1 if team A won, 0 if team B won, 0.5/0.6/0.4 for OT/SO
        k_factor: Base K-factor (learning rate)
        home_advantage: Bonus for home team
        k_multiplier: Multiplier for K-factor (e.g., for playoffs)

    Returns:
        Tuple (new_rating_a, new_rating_b)
    """
    expected_a = expected_score(rating_a, rating_b, home_advantage)
    k = k_factor * k_multiplier
    change_a = k * (actual_score - expected_a)
    change_b = k * ((1 - actual_score) - (1 - expected_a))

    return rating_a + change_a, rating_b + change_b


def apply_season_regression(ratings: dict, regression_factor: float) -> dict:
    """
    Apply mean regression to ratings between seasons.

    Args:
        ratings: Dictionary of {team_id: rating}
        regression_factor: Factor 0-1, where 1 = full regression to mean

    Returns:
        New dictionary with regressed ratings
    """
    if not ratings:
        return {}

    values = list(ratings.values())
    mean_rating = sum(values) / len(values)

    return {
        team_id: rating + regression_factor * (mean_rating - rating)
        for team_id, rating in ratings.items()
    }
