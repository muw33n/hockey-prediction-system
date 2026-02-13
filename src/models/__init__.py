"""
Models module - Prediction models and rating systems.

Location: src/models/__init__.py
"""

from src.models.elo_calculations import (
    expected_score,
    game_result_to_score,
    calculate_rating_update,
    apply_season_regression
)

__all__ = [
    'expected_score',
    'game_result_to_score',
    'calculate_rating_update',
    'apply_season_regression'
]
