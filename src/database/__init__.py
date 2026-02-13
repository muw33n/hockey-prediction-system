"""
Database module - Database operations and schema management.

Location: src/database/__init__.py
"""

from src.database.connection import DatabaseConnectionManager
from src.database.team_mapper import (
    NHL_FRANCHISES,
    NHL_CURRENT_TEAMS,
    NHL_ARIZONA_HISTORY,
    TEAM_NAME_MAPPINGS,
    FRANCHISE_MAPPINGS,
    normalize_team_name,
    resolve_jets_name,
    get_franchise_mapping,
    get_team_column_candidates,
    get_nhl_team_indicators
)

__all__ = [
    'DatabaseConnectionManager',
    'NHL_FRANCHISES',
    'NHL_CURRENT_TEAMS',
    'NHL_ARIZONA_HISTORY',
    'TEAM_NAME_MAPPINGS',
    'FRANCHISE_MAPPINGS',
    'normalize_team_name',
    'resolve_jets_name',
    'get_franchise_mapping',
    'get_team_column_candidates',
    'get_nhl_team_indicators'
]
