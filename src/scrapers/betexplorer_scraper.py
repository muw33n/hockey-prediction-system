#!/usr/bin/env python3
"""
BetExplorer Scraper pro historicke kurzy NHL (MIGRATED)
========================================================
Stahuje historicke sazkove kurzy z betexplorer.com pro NHL zapasy.

MIGRACE: Enhanced infrastructure s per-component logging,
centralized paths a safe file handling.

Location: src/scrapers/betexplorer_scraper.py
"""

import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Tuple
import re
from urllib.parse import urljoin, urlparse
import json
import random

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# === MIGRACE: Enhanced infrastructure imports ===
from config.paths import PATHS
from config.logging_config import get_component_logger, PerformanceLogger
from src.utils.file_handlers import write_csv, write_json

# === MIGRACE: Per-component logger pro scraping ===
logger = get_component_logger(__name__, 'scraping')


class BetExplorerScraper:
    """Scraper pro historicke kurzy z BetExplorer.com"""

    def __init__(self, use_selenium: bool = True, headless: bool = True):
        """
        Inicializace scraperu s enhanced logging.

        Args:
            use_selenium: Pouzit Selenium pro dynamicky obsah
            headless: Spustit prohlizec v headless rezimu
        """
        self.base_url = "https://www.betexplorer.com"
        self.use_selenium = use_selenium

        # HTTP session for basic requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'cs,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

        # Selenium setup
        self.driver = None
        if self.use_selenium:
            self._setup_selenium(headless)

        # Rate limiting
        self.delay_range = (2, 5)
        self.last_request_time = 0

        # Team mapping
        self.team_mapping = self._load_team_mapping()

        # Performance monitoring
        self.perf_logger = PerformanceLogger(logger)

        # Cache
        self.cache = {}

        logger.info("BetExplorerScraper initialized")
        logger.info(f"  Selenium: {self.use_selenium}")

    def _setup_selenium(self, headless: bool = True):
        """Setup Selenium WebDriver"""
        try:
            chrome_options = Options()
            if headless:
                chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)

            logger.info("Selenium WebDriver initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize Selenium: {e}")
            logger.warning("Continuing with requests only...")
            self.use_selenium = False
            self.driver = None

    def _load_team_mapping(self) -> Dict[str, str]:
        """Load team name mapping"""
        return {
            'Anaheim Ducks': 'Anaheim Ducks',
            'Arizona Coyotes': 'Arizona Coyotes',
            'Boston Bruins': 'Boston Bruins',
            'Buffalo Sabres': 'Buffalo Sabres',
            'Calgary Flames': 'Calgary Flames',
            'Carolina Hurricanes': 'Carolina Hurricanes',
            'Chicago Blackhawks': 'Chicago Blackhawks',
            'Colorado Avalanche': 'Colorado Avalanche',
            'Columbus Blue Jackets': 'Columbus Blue Jackets',
            'Dallas Stars': 'Dallas Stars',
            'Detroit Red Wings': 'Detroit Red Wings',
            'Edmonton Oilers': 'Edmonton Oilers',
            'Florida Panthers': 'Florida Panthers',
            'Los Angeles Kings': 'Los Angeles Kings',
            'Minnesota Wild': 'Minnesota Wild',
            'Montreal Canadiens': 'Montreal Canadiens',
            'Nashville Predators': 'Nashville Predators',
            'New Jersey Devils': 'New Jersey Devils',
            'New York Islanders': 'New York Islanders',
            'New York Rangers': 'New York Rangers',
            'Ottawa Senators': 'Ottawa Senators',
            'Philadelphia Flyers': 'Philadelphia Flyers',
            'Pittsburgh Penguins': 'Pittsburgh Penguins',
            'San Jose Sharks': 'San Jose Sharks',
            'Seattle Kraken': 'Seattle Kraken',
            'St. Louis Blues': 'St. Louis Blues',
            'Tampa Bay Lightning': 'Tampa Bay Lightning',
            'Toronto Maple Leafs': 'Toronto Maple Leafs',
            'Utah Hockey Club': 'Utah Hockey Club',
            'Vancouver Canucks': 'Vancouver Canucks',
            'Vegas Golden Knights': 'Vegas Golden Knights',
            'Washington Capitals': 'Washington Capitals',
            'Winnipeg Jets': 'Winnipeg Jets'
        }

    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        delay = random.uniform(*self.delay_range)
        if time_since_last < delay:
            sleep_time = delay - time_since_last
            logger.debug(f"Rate limiting: waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, url: str, use_selenium: bool = None) -> Optional[BeautifulSoup]:
        """Make HTTP request with rate limiting"""
        self._rate_limit()

        if use_selenium is None:
            use_selenium = self.use_selenium

        try:
            logger.debug(f"Fetching: {url}")

            if use_selenium and self.driver:
                self.driver.get(url)
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                html = self.driver.page_source
                return BeautifulSoup(html, 'html.parser')
            else:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return BeautifulSoup(response.content, 'html.parser')

        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def get_season_results_urls(self, season: str) -> List[str]:
        """Get URLs for all matches in a season"""
        season_url = f"{self.base_url}/hockey/usa/nhl-{season}/results/"

        logger.info(f"Loading results for season {season}")

        soup = self._make_request(season_url)
        if not soup:
            logger.error(f"Failed to load results for season {season}")
            return []

        stage_element = soup.find(attrs={'title': 'Main season game statistics'})
        if not stage_element:
            logger.error("Element with title='Main season game statistics' not found")
            return []

        stage_href = None
        if stage_element.name == 'a' and stage_element.get('href'):
            stage_href = stage_element['href']
        else:
            parent = stage_element.find_parent('a')
            if parent and parent.get('href'):
                stage_href = parent['href']

        if not stage_href:
            logger.error("href for Main season game statistics not found")
            return []

        stage_url = urljoin(season_url, stage_href)
        if '?' in stage_url:
            results_url = f"{stage_url}&month=all"
        else:
            results_url = f"{stage_url}?month=all"

        logger.info(f"Using stage URL: {results_url}")

        soup = self._make_request(results_url)
        if not soup:
            logger.error(f"Failed to load complete results for season {season}")
            return []

        match_urls = []

        results_table = soup.find('table', {'class': 'table-main'})
        if not results_table:
            logger.error("Results table not found")
            return []

        for row in results_table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 5:
                match_link = None
                for cell in cells:
                    link = cell.find('a')
                    if link and '/hockey/usa/nhl' in link.get('href', ''):
                        match_link = urljoin(self.base_url, link['href'])
                        break

                if match_link:
                    match_urls.append(match_link)

        logger.info(f"Found {len(match_urls)} matches for season {season}")
        return match_urls

    def extract_match_odds(self, match_url: str) -> Optional[Dict]:
        """Extract odds for a specific match"""
        soup = self._make_request(match_url)
        if not soup:
            return None

        try:
            match_info = self._extract_match_info(soup, match_url)
            if not match_info:
                return None

            odds_data = self._extract_odds_from_page(soup, match_url)

            result = {
                **match_info,
                'odds': odds_data,
                'scraped_at': datetime.now(),
                'source_url': match_url
            }

            return result

        except Exception as e:
            logger.error(f"Error extracting odds from {match_url}: {e}")
            return None

    def _extract_match_info(self, soup: BeautifulSoup, match_url: str) -> Optional[Dict]:
        """Extract basic match information"""
        try:
            team_elements = soup.find_all('h2', {'class': 'list-details__item__title'})

            if len(team_elements) < 2:
                logger.error("Teams not found on page")
                return None

            home_team = team_elements[0].get_text(strip=True)
            away_team = team_elements[1].get_text(strip=True)

            home_team = self.team_mapping.get(home_team, home_team)
            away_team = self.team_mapping.get(away_team, away_team)

            match_datetime = None
            date_elem = soup.find(attrs={'data-dt': True})
            if date_elem:
                date_string = date_elem.get('data-dt')
                date_parts = date_string.split(',')
                if len(date_parts) >= 5:
                    try:
                        day, month, year = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
                        hour, minute = int(date_parts[3]), int(date_parts[4])
                        match_datetime = datetime(year, month, day, hour, minute)
                    except (ValueError, IndexError):
                        logger.warning(f"Failed to parse datetime: {date_string}")
                elif len(date_parts) >= 3:
                    try:
                        day, month, year = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
                        match_datetime = datetime(year, month, day)
                    except (ValueError, IndexError):
                        logger.warning(f"Failed to parse date: {date_string}")

            home_score, away_score = None, None
            score_elem = soup.find(id='js-score')
            if score_elem:
                score_text = score_elem.get_text(strip=True)
                if score_text and ':' in score_text:
                    try:
                        scores = score_text.split(':')
                        if len(scores) == 2:
                            home_score = int(scores[0].strip())
                            away_score = int(scores[1].strip())
                    except ValueError:
                        logger.warning(f"Failed to parse score: {score_text}")

            status = 'scheduled'

            finished_elem = soup.find(id='isFinished')
            if finished_elem and finished_elem.get('value', '0') == '1':
                status = 'completed'

            live_elem = soup.find(id='isLive')
            if live_elem and live_elem.get('value', ''):
                status = 'live'

            if home_score is not None and away_score is not None and status == 'scheduled':
                status = 'completed'

            partial_score = None
            partial_elem = soup.find(id='js-partial')
            if partial_elem:
                partial_score = partial_elem.get_text(strip=True)

            result = {
                'home_team': home_team,
                'away_team': away_team,
                'match_datetime': match_datetime,
                'match_date': match_datetime.date() if match_datetime else None,
                'home_score': home_score,
                'away_score': away_score,
                'status': status
            }

            if partial_score:
                result['partial_score'] = partial_score

            logger.debug(f"Extracted match info: {result}")
            return result

        except Exception as e:
            logger.error(f"Error extracting match info: {e}")
            return None

    def _extract_odds_from_page(self, soup: BeautifulSoup, match_url: str) -> List[Dict]:
        """Extract odds from match page using API"""
        odds_data = []

        try:
            match_id = self._extract_match_id_from_url(match_url)
            if not match_id:
                logger.error(f"Failed to extract match ID from URL: {match_url}")
                return []

            logger.debug(f"Extracted match ID: {match_id}")

            market_types = [
                ('HA', 'moneyline_2way', 2),
            ]

            for market_code, market_name, expected_odds_count in market_types:
                try:
                    market_odds = self._fetch_match_odds(match_id, market_code, market_name, expected_odds_count)
                    if market_odds:
                        odds_data.extend(market_odds)
                except Exception as e:
                    logger.warning(f"Error fetching odds for market {market_name}: {e}")
                    continue

            logger.info(f"Extracted {len(odds_data)} odds for match {match_id}")

        except Exception as e:
            logger.error(f"Error extracting odds: {e}")

        return odds_data

    def _extract_match_id_from_url(self, match_url: str) -> Optional[str]:
        """Extract match ID from URL"""
        try:
            url_parts = match_url.rstrip('/').split('/')

            for part in reversed(url_parts):
                if len(part) == 8 and part.isalnum():
                    return part

            match_id_pattern = r'/([a-zA-Z0-9]{8})/?(?:odds/?)?$'
            match = re.search(match_id_pattern, match_url)
            if match:
                return match.group(1)

            logger.error(f"Match ID not found in URL: {match_url}")
            return None

        except Exception as e:
            logger.error(f"Error extracting match ID: {e}")
            return None

    def _fetch_match_odds(self, match_id: str, market_code: str, market_name: str, expected_odds_count: int) -> List[Dict]:
        """Fetch odds for a specific match and market"""
        try:
            odds_url = f"{self.base_url}/match-odds-old/{match_id}/1/{market_code}/1/en/"

            logger.debug(f"Fetching odds from: {odds_url}")

            response = self.session.get(odds_url, timeout=30)
            response.raise_for_status()

            try:
                data = response.json()
            except ValueError as e:
                logger.error(f"Invalid JSON response from {odds_url}: {e}")
                return []

            if 'odds' not in data:
                logger.warning(f"Key 'odds' not found in response for {market_name}")
                return []

            return self._parse_odds_response(data, market_name, expected_odds_count)

        except Exception as e:
            logger.error(f"Error fetching odds for {market_name}: {e}")
            return []

    def _parse_odds_response(self, response_data: Dict, market_name: str, expected_odds_count: int) -> List[Dict]:
        """Parse JSON response containing HTML odds"""
        odds_list = []

        try:
            html_content = response_data.get('odds', '')
            if not html_content:
                logger.warning(f"Empty HTML content for {market_name}")
                return []

            soup = BeautifulSoup(html_content, 'html.parser')

            odds_table = soup.find('table', {'class': 'table-main'})
            if not odds_table:
                logger.warning(f"Odds table not found for {market_name}")
                return []

            tbody = odds_table.find('tbody')
            if not tbody:
                logger.warning(f"tbody not found for {market_name}")
                return []

            for row in tbody.find_all('tr'):
                try:
                    odds_entry = self._parse_bookmaker_row(row, market_name, expected_odds_count)
                    if odds_entry:
                        odds_list.append(odds_entry)
                except Exception as e:
                    logger.warning(f"Error parsing odds row: {e}")
                    continue

            try:
                average_odds = self._parse_average_odds(odds_table, market_name, expected_odds_count)
                if average_odds:
                    odds_list.append(average_odds)
            except Exception as e:
                logger.warning(f"Error parsing average odds: {e}")

            logger.debug(f"Parsed {len(odds_list)} odds for {market_name}")

        except Exception as e:
            logger.error(f"Error parsing HTML odds: {e}")

        return odds_list

    def _parse_bookmaker_row(self, row, market_name: str, expected_odds_count: int) -> Optional[Dict]:
        """Parse bookmaker odds row"""
        try:
            bookmaker_link = row.find('a', class_=lambda x: x and 'in-bookmaker-logo-link' in x)
            if bookmaker_link:
                bookmaker_name = bookmaker_link.get_text(strip=True)
            else:
                bookmaker_span = row.find('span', class_=lambda x: x and 'in-bookmaker-logo' in x)
                if bookmaker_span:
                    bookmaker_name = bookmaker_span.get('title', 'Unknown')
                else:
                    logger.warning("Bookmaker name not found in row")
                    return None

            odds_cells = row.find_all('td', {'data-odd': True})

            if len(odds_cells) < expected_odds_count:
                logger.warning(f"Not enough odds for {bookmaker_name}")
                return None

            odds_info = {}

            if expected_odds_count == 2:  # HA (Home/Away)
                odds_info['home_odd'] = float(odds_cells[0].get('data-odd'))
                odds_info['away_odd'] = float(odds_cells[1].get('data-odd'))

                for i, cell in enumerate(odds_cells[:2]):
                    prefix = 'home' if i == 0 else 'away'
                    self._add_additional_odds_info(cell, odds_info, prefix)

            elif expected_odds_count == 3:  # 1X2
                odds_info['home_odd'] = float(odds_cells[0].get('data-odd'))
                odds_info['draw_odd'] = float(odds_cells[1].get('data-odd'))
                odds_info['away_odd'] = float(odds_cells[2].get('data-odd'))

            return {
                'bookmaker': bookmaker_name,
                'market_type': market_name,
                'odds': odds_info,
                'timestamp': datetime.now(),
                'source': 'betexplorer_api'
            }

        except Exception as e:
            logger.error(f"Error parsing bookmaker row: {e}")
            return None

    def _add_additional_odds_info(self, cell, odds_info: dict, prefix: str):
        """Add additional odds info (opening odds, dates)"""
        opening_odd = cell.get('data-opening-odd')
        opening_date = cell.get('data-opening-date')

        if opening_odd:
            try:
                odds_info[f'{prefix}_opening_odd'] = float(opening_odd)
            except ValueError:
                pass

        if opening_date:
            parsed_date = self._parse_betexplorer_datetime(opening_date)
            if parsed_date:
                odds_info[f'{prefix}_opening_datetime'] = parsed_date

    def _parse_betexplorer_datetime(self, date_string: str) -> Optional[datetime]:
        """Parse datetime from betexplorer format"""
        try:
            date_parts = date_string.split(',')
            if len(date_parts) >= 5:
                day, month, year = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
                hour, minute = int(date_parts[3]), int(date_parts[4])
                return datetime(year, month, day, hour, minute)
            elif len(date_parts) >= 3:
                day, month, year = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
                return datetime(year, month, day)
            return None
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse betexplorer date: {date_string}")
            return None

    def _parse_average_odds(self, table, market_name: str, expected_odds_count: int) -> Optional[Dict]:
        """Parse average odds from tfoot"""
        try:
            tfoot = table.find('tfoot')
            if not tfoot:
                return None

            avg_cells = tfoot.find_all('td', {'data-odd': True})

            if len(avg_cells) < expected_odds_count:
                return None

            odds_info = {}

            if expected_odds_count == 2:
                odds_info['home_odd'] = float(avg_cells[0].get('data-odd'))
                odds_info['away_odd'] = float(avg_cells[1].get('data-odd'))
            elif expected_odds_count == 3:
                odds_info['home_odd'] = float(avg_cells[0].get('data-odd'))
                odds_info['draw_odd'] = float(avg_cells[1].get('data-odd'))
                odds_info['away_odd'] = float(avg_cells[2].get('data-odd'))

            return {
                'bookmaker': 'Average',
                'market_type': market_name,
                'odds': odds_info,
                'timestamp': datetime.now(),
                'source': 'betexplorer_api'
            }

        except Exception as e:
            logger.error(f"Error parsing average odds: {e}")
            return None

    def scrape_season_odds(self, season: str, max_matches: Optional[int] = None) -> List[Dict]:
        """Download odds for an entire season"""
        self.perf_logger.start_timer(f'scrape_season_{season}')

        logger.info(f"Starting odds download for season {season}")

        match_urls = self.get_season_results_urls(season)
        if not match_urls:
            logger.error(f"No matches found for season {season}")
            return []

        if max_matches:
            match_urls = match_urls[:max_matches]
            logger.info(f"Limiting to first {max_matches} matches")

        season_odds = []
        success_count = 0

        for i, match_url in enumerate(match_urls, 1):
            logger.info(f"Processing match {i}/{len(match_urls)}: {match_url}")

            try:
                match_odds = self.extract_match_odds(match_url)
                if match_odds:
                    match_odds['season'] = season
                    season_odds.append(match_odds)
                    success_count += 1

                    if success_count % 10 == 0:
                        logger.info(f"Successfully processed {success_count} matches")

            except Exception as e:
                logger.error(f"Error processing match {match_url}: {e}")
                continue

        self.perf_logger.end_timer(f'scrape_season_{season}')
        logger.info(f"Completed! Processed {success_count}/{len(match_urls)} matches for season {season}")
        return season_odds

    def scrape_multiple_seasons(self, seasons: List[str], max_matches_per_season: Optional[int] = None) -> Dict[str, List[Dict]]:
        """Download odds for multiple seasons"""
        all_seasons_data = {}

        for season in seasons:
            logger.info(f"{'='*50}")
            logger.info(f"DOWNLOADING SEASON {season}")
            logger.info(f"{'='*50}")

            try:
                season_odds = self.scrape_season_odds(season, max_matches_per_season)
                all_seasons_data[season] = season_odds

                # Save incremental results
                self.save_season_data(season, season_odds)

            except Exception as e:
                logger.error(f"Error downloading season {season}: {e}")
                all_seasons_data[season] = []

        return all_seasons_data

    def save_season_data(self, season: str, season_data: List[Dict]) -> str:
        """Save season data to CSV file using enhanced infrastructure"""
        # === MIGRACE: Use PATHS for output directory ===
        output_dir = PATHS.odds_data
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"nhl_odds_{season}_{timestamp}.csv"
        filepath = output_dir / filename

        if not season_data:
            logger.warning(f"No data to save for season {season}")
            return ""

        # Flatten data for DataFrame
        flattened_data = []

        for match in season_data:
            base_match_info = {
                'season': match.get('season'),
                'match_date': match.get('match_date'),
                'match_datetime': match.get('match_datetime'),
                'home_team': match.get('home_team'),
                'away_team': match.get('away_team'),
                'home_score': match.get('home_score'),
                'away_score': match.get('away_score'),
                'status': match.get('status'),
                'source_url': match.get('source_url'),
                'scraped_at': match.get('scraped_at')
            }

            if match.get('odds'):
                for odds_entry in match['odds']:
                    row = {**base_match_info}
                    row['bookmaker'] = odds_entry.get('bookmaker')
                    row['market_type'] = odds_entry.get('market_type')

                    odds_dict = odds_entry.get('odds', {})
                    for market, odd_value in odds_dict.items():
                        row[f'odds_{market}'] = odd_value

                    flattened_data.append(row)
            else:
                flattened_data.append(base_match_info)

        # === MIGRACE: Save using safe file handler ===
        df = pd.DataFrame(flattened_data)
        write_csv(df, filepath)

        logger.info(f"Data saved to {filepath} ({len(df)} rows)")
        return str(filepath)

    def close(self):
        """Close scraper and WebDriver"""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")


def main():
    """Main function to run scraper with enhanced infrastructure"""

    # === MIGRACE: Use PATHS for directories ===
    PATHS.ensure_directories()

    # Configuration
    SEASONS = ['2021-2022', '2022-2023', '2023-2024', '2024-2025']
    MAX_MATCHES_PER_SEASON = 50  # For testing - set to None for all
    USE_SELENIUM = True

    logger.info("Starting BetExplorer scraper for NHL odds")
    logger.info(f"Seasons: {', '.join(SEASONS)}")
    logger.info(f"Max matches per season: {MAX_MATCHES_PER_SEASON or 'all'}")
    logger.info("Market types: HA (2-way Moneyline)")

    scraper = BetExplorerScraper(use_selenium=USE_SELENIUM)

    try:
        all_data = scraper.scrape_multiple_seasons(
            seasons=SEASONS,
            max_matches_per_season=MAX_MATCHES_PER_SEASON
        )

        logger.info("=" * 60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 60)

        total_matches = 0
        for season, season_data in all_data.items():
            count = len(season_data)
            total_matches += count
            logger.info(f"  {season}: {count} matches")

        logger.info(f"  TOTAL: {total_matches} matches")

        # Save combined data
        if total_matches > 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            combined_file = PATHS.odds_data / f"nhl_odds_combined_{timestamp}.json"

            write_json(all_data, combined_file)
            logger.info(f"Combined data saved to {combined_file}")

        logger.info("Download completed successfully!")

    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"Error during download: {e}")
        raise
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
