"""
Team Mapper - NHL team name mapping and franchise data.

Extracted from database_setup.py for separation of concerns.
Location: src/database/team_mapper.py
"""

from typing import Dict, List, Tuple, Optional
from datetime import date
import pandas as pd


# NHL Franchises data: (id, name, founded_date, founded_city)
NHL_FRANCHISES: List[Tuple[int, str, str, str]] = [
    (1, 'Boston Bruins Franchise', '1924-11-01', 'Boston'),
    (2, 'Buffalo Sabres Franchise', '1970-05-12', 'Buffalo'),
    (3, 'Detroit Red Wings Franchise', '1926-05-15', 'Detroit'),
    (4, 'Florida Panthers Franchise', '1993-06-14', 'Miami'),
    (5, 'Montreal Canadiens Franchise', '1909-12-04', 'Montreal'),
    (6, 'Ottawa Senators Franchise', '1992-12-16', 'Ottawa'),
    (7, 'Tampa Bay Lightning Franchise', '1992-12-16', 'Tampa Bay'),
    (8, 'Toronto Maple Leafs Franchise', '1917-11-26', 'Toronto'),
    (9, 'Carolina Hurricanes Franchise', '1979-06-22', 'Hartford'),
    (10, 'Columbus Blue Jackets Franchise', '2000-06-25', 'Columbus'),
    (11, 'New Jersey Devils Franchise', '1974-06-11', 'Kansas City'),
    (12, 'New York Islanders Franchise', '1972-11-08', 'Uniondale'),
    (13, 'New York Rangers Franchise', '1926-05-15', 'New York'),
    (14, 'Philadelphia Flyers Franchise', '1967-06-05', 'Philadelphia'),
    (15, 'Pittsburgh Penguins Franchise', '1967-06-05', 'Pittsburgh'),
    (16, 'Washington Capitals Franchise', '1974-06-11', 'Washington'),
    (17, 'Chicago Blackhawks Franchise', '1926-05-15', 'Chicago'),
    (18, 'Colorado Avalanche Franchise', '1979-06-22', 'Quebec City'),
    (19, 'Dallas Stars Franchise', '1967-06-05', 'Minneapolis'),
    (20, 'Minnesota Wild Franchise', '2000-06-25', 'Saint Paul'),
    (21, 'Nashville Predators Franchise', '1998-06-25', 'Nashville'),
    (22, 'St. Louis Blues Franchise', '1967-06-05', 'St. Louis'),
    (23, 'Arizona/Utah Franchise', '1979-06-22', 'Winnipeg'),
    (24, 'Winnipeg Jets Franchise', '1999-06-25', 'Atlanta'),
    (25, 'Anaheim Ducks Franchise', '1993-06-15', 'Anaheim'),
    (26, 'Calgary Flames Franchise', '1972-06-06', 'Atlanta'),
    (27, 'Edmonton Oilers Franchise', '1979-06-22', 'Edmonton'),
    (28, 'Los Angeles Kings Franchise', '1967-06-05', 'Los Angeles'),
    (29, 'San Jose Sharks Franchise', '1991-05-09', 'San Jose'),
    (30, 'Seattle Kraken Franchise', '2021-07-21', 'Seattle'),
    (31, 'Vancouver Canucks Franchise', '1970-05-12', 'Vancouver'),
    (32, 'Vegas Golden Knights Franchise', '2017-06-22', 'Las Vegas'),
]

# Current NHL teams: (franchise_id, name, city, conference, division, abbreviation, effective_from)
NHL_CURRENT_TEAMS: List[Tuple[int, str, str, str, str, str, str]] = [
    (1, 'Boston Bruins', 'Boston', 'Eastern', 'Atlantic', 'BOS', '1924-11-01'),
    (2, 'Buffalo Sabres', 'Buffalo', 'Eastern', 'Atlantic', 'BUF', '1970-05-12'),
    (3, 'Detroit Red Wings', 'Detroit', 'Eastern', 'Atlantic', 'DET', '1932-10-05'),
    (4, 'Florida Panthers', 'Sunrise', 'Eastern', 'Atlantic', 'FLA', '1993-10-06'),
    (5, 'Montreal Canadiens', 'Montreal', 'Eastern', 'Atlantic', 'MTL', '1917-11-26'),
    (6, 'Ottawa Senators', 'Ottawa', 'Eastern', 'Atlantic', 'OTT', '1992-10-08'),
    (7, 'Tampa Bay Lightning', 'Tampa Bay', 'Eastern', 'Atlantic', 'TBL', '1992-10-07'),
    (8, 'Toronto Maple Leafs', 'Toronto', 'Eastern', 'Atlantic', 'TOR', '1927-02-17'),
    (9, 'Carolina Hurricanes', 'Raleigh', 'Eastern', 'Metropolitan', 'CAR', '1997-10-29'),
    (10, 'Columbus Blue Jackets', 'Columbus', 'Eastern', 'Metropolitan', 'CBJ', '2000-10-07'),
    (11, 'New Jersey Devils', 'Newark', 'Eastern', 'Metropolitan', 'NJD', '1982-10-05'),
    (12, 'New York Islanders', 'Elmont', 'Eastern', 'Metropolitan', 'NYI', '1972-10-07'),
    (13, 'New York Rangers', 'New York', 'Eastern', 'Metropolitan', 'NYR', '1926-11-16'),
    (14, 'Philadelphia Flyers', 'Philadelphia', 'Eastern', 'Metropolitan', 'PHI', '1967-10-11'),
    (15, 'Pittsburgh Penguins', 'Pittsburgh', 'Eastern', 'Metropolitan', 'PIT', '1967-10-11'),
    (16, 'Washington Capitals', 'Washington', 'Eastern', 'Metropolitan', 'WSH', '1974-10-09'),
    (17, 'Chicago Blackhawks', 'Chicago', 'Western', 'Central', 'CHI', '1926-11-17'),
    (18, 'Colorado Avalanche', 'Denver', 'Western', 'Central', 'COL', '1995-10-06'),
    (19, 'Dallas Stars', 'Dallas', 'Western', 'Central', 'DAL', '1993-10-05'),
    (20, 'Minnesota Wild', 'Saint Paul', 'Western', 'Central', 'MIN', '2000-10-06'),
    (21, 'Nashville Predators', 'Nashville', 'Western', 'Central', 'NSH', '1998-10-10'),
    (22, 'St. Louis Blues', 'St. Louis', 'Western', 'Central', 'STL', '1967-10-11'),
    (23, 'Utah Mammoth', 'Salt Lake City', 'Western', 'Central', 'UTA', '2024-04-18'),
    (24, 'Winnipeg Jets', 'Winnipeg', 'Western', 'Central', 'WPG', '2011-10-09'),
    (25, 'Anaheim Ducks', 'Anaheim', 'Western', 'Pacific', 'ANA', '1993-10-08'),
    (26, 'Calgary Flames', 'Calgary', 'Western', 'Pacific', 'CGY', '1980-10-09'),
    (27, 'Edmonton Oilers', 'Edmonton', 'Western', 'Pacific', 'EDM', '1979-10-10'),
    (28, 'Los Angeles Kings', 'Los Angeles', 'Western', 'Pacific', 'LAK', '1967-10-14'),
    (29, 'San Jose Sharks', 'San Jose', 'Western', 'Pacific', 'SJS', '1991-10-04'),
    (30, 'Seattle Kraken', 'Seattle', 'Western', 'Pacific', 'SEA', '2021-10-12'),
    (31, 'Vancouver Canucks', 'Vancouver', 'Western', 'Pacific', 'VAN', '1970-10-09'),
    (32, 'Vegas Golden Knights', 'Las Vegas', 'Western', 'Pacific', 'VGK', '2017-10-06'),
]

# Historical Arizona/Utah transitions
NHL_ARIZONA_HISTORY: List[Tuple[int, str, str, str, str, str, str, str]] = [
    (23, 'Winnipeg Jets', 'Winnipeg', 'Western', 'Smythe', 'WPG', '1979-10-10', '1996-04-13'),
    (23, 'Phoenix Coyotes', 'Phoenix', 'Western', 'Pacific', 'PHX', '1996-04-13', '2014-06-27'),
    (23, 'Arizona Coyotes', 'Glendale', 'Western', 'Pacific', 'ARI', '2014-06-27', '2024-04-18'),
]

# Team name normalization mappings
TEAM_NAME_MAPPINGS: Dict[str, str] = {
    'Utah Hockey Club': 'Utah Mammoth',
    'Arizona Coyotes': 'Utah Mammoth',
    'Phoenix Coyotes': 'Utah Mammoth',
}

# Franchise-based mappings for historical team resolution
FRANCHISE_MAPPINGS: Dict[str, str] = {
    'Arizona Coyotes': 'Utah Mammoth',
    'Utah Hockey Club': 'Utah Mammoth',
    'Phoenix Coyotes': 'Utah Mammoth',
}


def normalize_team_name(team_name: str) -> str:
    """
    Normalize team name to current naming convention.

    Args:
        team_name: Raw team name string

    Returns:
        Normalized team name
    """
    team_name = team_name.strip()
    return TEAM_NAME_MAPPINGS.get(team_name, team_name)


def resolve_jets_name(game_date) -> str:
    """
    Resolve Jets franchise based on game date.

    The original Winnipeg Jets (1979-1996) became Phoenix/Arizona Coyotes.
    The current Winnipeg Jets (2011-present) were originally Atlanta Thrashers.

    Args:
        game_date: Date of the game

    Returns:
        Team name for the Jets franchise
    """
    cutoff_date = pd.to_datetime('2011-05-31').date()
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date).date()

    if game_date <= cutoff_date:
        return 'Utah Mammoth'  # Old Jets -> Arizona/Utah franchise
    else:
        return 'Winnipeg Jets'  # New Jets (former Thrashers)


def get_franchise_mapping(team_name: str, game_date) -> Optional[str]:
    """
    Get current team name for a franchise based on game date.

    Args:
        team_name: Historical team name
        game_date: Date of the game

    Returns:
        Current team name or None if not a franchise mapping
    """
    team_name = team_name.strip()

    # Check franchise mappings first (before normalization)
    if team_name in FRANCHISE_MAPPINGS:
        return FRANCHISE_MAPPINGS[team_name]

    # Handle special Jets case
    if team_name == 'Winnipeg Jets':
        return resolve_jets_name(game_date)

    return None


def get_team_column_candidates() -> List[str]:
    """
    Get list of potential team name column names for CSV detection.

    Returns:
        List of column name candidates
    """
    return ['Team', '', ':', 'team', 'Team Name', 'Tm']


def get_nhl_team_indicators() -> List[str]:
    """
    Get list of NHL team name indicators for column detection.

    Returns:
        List of team name substrings
    """
    return ['Bruins', 'Rangers', 'Kings', 'Wings', 'Leafs', 'Flames', 'Stars', 'Wild']
