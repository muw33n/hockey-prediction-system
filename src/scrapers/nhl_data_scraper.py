#!/usr/bin/env python3
"""
NHL Data Scraper for Hockey-Reference.com (MIGRATED)
=====================================================
Collects game results, team stats, and standings data.

MIGRACE: Enhanced infrastructure s per-component logging,
centralized paths a safe file handling.

Location: src/scrapers/nhl_data_scraper.py
"""

import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re

# === MIGRACE: Enhanced infrastructure imports ===
from config.paths import PATHS
from config.logging_config import get_component_logger, PerformanceLogger
from src.utils.file_handlers import write_csv, write_json

# === MIGRACE: Per-component logger pro scraping ===
logger = get_component_logger(__name__, 'scraping')


class NHLScraper:
    """Scraper for NHL data from Hockey-Reference.com"""

    def __init__(self, delay: float = 2.0):
        """
        Initialize scraper with enhanced logging.

        Args:
            delay: Delay between requests (seconds)
        """
        self.base_url = "https://www.hockey-reference.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.delay = delay
        self.perf_logger = PerformanceLogger(logger)

        logger.info("NHLScraper initialized")
        logger.info(f"  Base URL: {self.base_url}")
        logger.info(f"  Request delay: {self.delay}s")

    def _make_request(self, url: str) -> Optional[BeautifulSoup]:
        """Make HTTP request with error handling and rate limiting"""
        try:
            logger.debug(f"Fetching: {url}")
            time.sleep(self.delay)

            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            return BeautifulSoup(response.content, 'html.parser')

        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def _parse_game_time(self, time_cell) -> Optional[str]:
        """Parse game time from the time cell"""
        if not time_cell:
            return None

        time_text = time_cell.text.strip()
        if not time_text or time_text in ['', 'Time']:
            return None

        try:
            formats_to_try = [
                '%I:%M %p',
                '%I %p',
                '%H:%M',
                '%H'
            ]

            for fmt in formats_to_try:
                try:
                    parsed_time = datetime.strptime(time_text, fmt).time()
                    return parsed_time.strftime('%H:%M')
                except ValueError:
                    continue

            logger.warning(f"Could not parse time format: '{time_text}'")
            return None

        except Exception as e:
            logger.warning(f"Error parsing time '{time_text}': {e}")
            return None

    def _extract_boxscore_url(self, date_cell) -> str:
        """Extract boxscore URL from the date cell link"""
        try:
            date_link = date_cell.find('a')
            if date_link and date_link.get('href'):
                relative_url = date_link.get('href')
                if relative_url.startswith('/'):
                    return f"{self.base_url}{relative_url}"
                return relative_url
            return ""
        except Exception as e:
            logger.warning(f"Could not extract boxscore URL: {e}")
            return ""

    def get_season_schedule(self, season: str) -> pd.DataFrame:
        """
        Get all games for a specific season.

        Args:
            season: Season year (e.g., '2024' for 2023-24 season)

        Returns:
            DataFrame with game results
        """
        self.perf_logger.start_timer(f'scrape_season_{season}')

        url = f"{self.base_url}/leagues/NHL_{season}_games.html"
        soup = self._make_request(url)

        if not soup:
            return pd.DataFrame()

        games_table = soup.find('table', {'id': 'games'})
        if not games_table:
            logger.error(f"No games table found for season {season}")
            return pd.DataFrame()

        games_data = []
        rows = games_table.find('tbody').find_all('tr')

        for row in rows:
            if row.get('class') and 'thead' in row.get('class'):
                continue

            cells = row.find_all(['td', 'th'])
            if len(cells) < 7:
                continue

            try:
                date_cell = cells[0]
                time_cell = cells[1]
                visitor_cell = cells[2]
                home_cell = cells[4]

                date_text = date_cell.text.strip()
                if not date_text or date_text in ['Date', '']:
                    continue

                game_date = datetime.strptime(date_text, '%Y-%m-%d').date()
                boxscore_url = self._extract_boxscore_url(date_cell)
                game_time = self._parse_game_time(time_cell)

                if game_time:
                    hour, minute = game_time.split(':')
                    game_datetime = datetime.combine(
                        game_date,
                        datetime.min.time().replace(hour=int(hour), minute=int(minute))
                    )
                else:
                    game_datetime = datetime.combine(
                        game_date,
                        datetime.min.time().replace(hour=19, minute=0)
                    )
                game_datetime_iso = game_datetime.strftime('%Y-%m-%d %H:%M:%S')

                visitor_team = visitor_cell.find('a')
                home_team = home_cell.find('a')

                if not visitor_team or not home_team:
                    continue

                visitor_name = visitor_team.text.strip()
                home_name = home_team.text.strip()

                visitor_score = None
                home_score = None
                visitor_score_cell = cells[3]
                home_score_cell = cells[5]
                ot_cell_idx = 6

                try:
                    visitor_score = int(visitor_score_cell.text.strip())
                    home_score = int(home_score_cell.text.strip())
                except (ValueError, AttributeError):
                    pass

                status = 'completed' if visitor_score is not None else 'scheduled'

                ot_so = ''
                if len(cells) > ot_cell_idx:
                    ot_so = cells[ot_cell_idx].text.strip()

                game_data = {
                    'datetime': game_datetime_iso,
                    'date': game_date,
                    'season': season,
                    'visitor_team': visitor_name,
                    'home_team': home_name,
                    'visitor_score': visitor_score,
                    'home_score': home_score,
                    'overtime_shootout': ot_so,
                    'status': status,
                    'boxscore_url': boxscore_url,
                    'scraped_at': datetime.now()
                }

                games_data.append(game_data)

            except Exception as e:
                logger.warning(f"Error parsing row: {e}")
                continue

        df = pd.DataFrame(games_data)

        self.perf_logger.end_timer(f'scrape_season_{season}')
        logger.info(f"Scraped {len(df)} games for season {season}")

        return df

    def get_team_stats(self, season: str) -> pd.DataFrame:
        """Get team statistics for a specific season"""
        url = f"{self.base_url}/leagues/NHL_{season}.html"
        soup = self._make_request(url)

        if not soup:
            return pd.DataFrame()

        stats_table = soup.find('table', {'id': 'stats'})
        if not stats_table:
            logger.debug(f"Stats table not found directly, checking comments for season {season}")
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))

            for comment in comments:
                if 'id="stats"' in comment:
                    comment_soup = BeautifulSoup(comment, 'html.parser')
                    stats_table = comment_soup.find('table', {'id': 'stats'})
                    if stats_table:
                        break

        if not stats_table:
            logger.error(f"No team stats table found for season {season}")
            return pd.DataFrame()

        stats_data = []

        header_rows = stats_table.find('thead').find_all('tr')
        headers = []
        for header_row in header_rows:
            headers1 = [th.text.strip() for th in header_row.find_all('th')]
            if len(headers1) >= 6:
                headers = headers1

        rows = stats_table.find('tbody').find_all('tr')

        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) != len(headers):
                continue

            row_data = {}
            for i, cell in enumerate(cells):
                if i < len(headers):
                    if headers[i] == 'Team':
                        team_link = cell.find('a')
                        value = team_link.text.strip() if team_link else cell.text.strip()
                    else:
                        value = cell.text.strip()
                    row_data[headers[i]] = value

            row_data['season'] = season
            row_data['scraped_at'] = datetime.now()
            stats_data.append(row_data)

        df = pd.DataFrame(stats_data)
        logger.info(f"Scraped stats for {len(df)} teams in season {season}")

        return df

    def get_standings(self, season: str) -> pd.DataFrame:
        """Get standings for a specific season"""
        url = f"{self.base_url}/leagues/NHL_{season}.html"
        soup = self._make_request(url)

        if not soup:
            return pd.DataFrame()

        standings_data = []

        for conference in ['EAS', 'WES']:
            table_id = f"standings_{conference.split()[0]}"
            table = soup.find('table', {'id': table_id})

            if not table:
                continue

            header_row = table.find('thead').find('tr')
            headers = [th.text.strip() for th in header_row.find_all('th')]

            rows = table.find('tbody').find_all('tr')

            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) != len(headers):
                    continue

                row_data = {}
                for i, cell in enumerate(cells):
                    if i < len(headers):
                        if headers[i] == 'Team':
                            team_link = cell.find('a')
                            value = team_link.text.strip() if team_link else cell.text.strip()
                        else:
                            value = cell.text.strip()
                        row_data[headers[i]] = value

                row_data['conference'] = conference
                row_data['season'] = season
                row_data['scraped_at'] = datetime.now()
                standings_data.append(row_data)

        df = pd.DataFrame(standings_data)
        logger.info(f"Scraped standings for {len(df)} teams in season {season}")

        return df

    def scrape_multiple_seasons(self, start_season: int, end_season: int) -> Dict[str, pd.DataFrame]:
        """Scrape multiple seasons of data"""
        all_games = []
        all_stats = []
        all_standings = []

        for season_year in range(start_season, end_season + 1):
            season_str = str(season_year)
            logger.info(f"Scraping season {season_year-1}-{str(season_year)[2:]}")

            games_df = self.get_season_schedule(season_str)
            if not games_df.empty:
                all_games.append(games_df)

            stats_df = self.get_team_stats(season_str)
            if not stats_df.empty:
                all_stats.append(stats_df)

            standings_df = self.get_standings(season_str)
            if not standings_df.empty:
                all_standings.append(standings_df)

            time.sleep(5)

        result = {}

        if all_games:
            result['games'] = pd.concat(all_games, ignore_index=True)
        if all_stats:
            result['team_stats'] = pd.concat(all_stats, ignore_index=True)
        if all_standings:
            result['standings'] = pd.concat(all_standings, ignore_index=True)

        return result


def main():
    """Main function to run the scraper with enhanced infrastructure"""

    # === MIGRACE: Use PATHS for directories ===
    PATHS.ensure_directories()

    scraper = NHLScraper()

    # Calculate seasons to scrape
    current_year = datetime.now().year
    if datetime.now().month >= 9:
        current_year += 1

    start_season = current_year - 3
    end_season = current_year

    logger.info(f"Starting NHL data scraping for seasons {start_season-1}-{start_season} to {end_season-1}-{end_season}")

    # Scrape data
    data = scraper.scrape_multiple_seasons(start_season, end_season)

    # === MIGRACE: Save using PATHS and safe file handlers ===
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    for data_type, df in data.items():
        filepath = PATHS.raw_data / f"nhl_{data_type}_{timestamp}.csv"
        write_csv(df, filepath)
        logger.info(f"Saved {len(df)} records to {filepath}")

    # Save summary
    summary = {
        'scrape_date': datetime.now().isoformat(),
        'seasons_scraped': f"{start_season}-{end_season}",
        'total_games': len(data.get('games', [])),
        'total_teams': len(data.get('team_stats', [])),
        'data_files': list(data.keys()),
    }

    summary_path = PATHS.raw_data / f"nhl_scrape_summary_{timestamp}.json"
    write_json(summary, summary_path)

    logger.info("NHL data scraping completed successfully!")

    # Display statistics
    for data_type, df in data.items():
        logger.info(f"{data_type.upper()} SUMMARY:")
        logger.info(f"  Total records: {len(df)}")
        if data_type == 'games':
            completed_games = df[df['status'] == 'completed']
            logger.info(f"  Completed games: {len(completed_games)}")
            logger.info(f"  Scheduled games: {len(df) - len(completed_games)}")
        logger.info(f"  Seasons: {df['season'].min()} - {df['season'].max()}")


if __name__ == "__main__":
    main()
