#!/usr/bin/env python3
"""
BetExplorer Scraper pro historické kurzy NHL
Stahuje historické sázkové kurzy z betexplorer.com pro NHL zápasy.

Autor: Hockey Prediction System
Datum: 2025
"""

import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time
import os
from datetime import datetime, timedelta, date
import logging
from typing import List, Dict, Optional, Tuple
import re
from urllib.parse import urljoin, urlparse
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/betexplorer_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BetExplorerScraper:
    """Scraper pro historické kurzy z BetExplorer.com"""
    
    def __init__(self, use_selenium: bool = True, headless: bool = True):
        """
        Inicializace scraperu
        
        Args:
            use_selenium: Použít Selenium pro dynamický obsah
            headless: Spustit prohlížeč v headless režimu
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
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Selenium setup
        self.driver = None
        if self.use_selenium:
            self._setup_selenium(headless)
        
        # Rate limiting
        self.delay_range = (2, 5)  # Random delays between requests
        self.last_request_time = 0
        
        # Team mapping (betexplorer name -> our name)
        self.team_mapping = self._load_team_mapping()
        
        # Cache for results
        self.cache = {}
        
    def _setup_selenium(self, headless: bool = True):
        """Nastavení Selenium WebDriveru"""
        try:
            chrome_options = Options()
            if headless:
                chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            
            logger.info("✅ Selenium WebDriver úspěšně inicializován")
            
        except Exception as e:
            logger.warning(f"⚠️ Nepodařilo se inicializovat Selenium: {e}")
            logger.warning("Pokračuji pouze s requests...")
            self.use_selenium = False
            self.driver = None
    
    def _load_team_mapping(self) -> Dict[str, str]:
        """Načte mapování názvů týmů"""
        # Basic mapping of NHL teams (betexplorer -> standard name)
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
            'Vancouver Canucks': 'Vancouver Canucks',
            'Vegas Golden Knights': 'Vegas Golden Knights',
            'Washington Capitals': 'Washington Capitals',
            'Winnipeg Jets': 'Winnipeg Jets'
        }
    
    def _rate_limit(self):
        """Implementuje rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        delay = random.uniform(*self.delay_range)
        if time_since_last < delay:
            sleep_time = delay - time_since_last
            logger.debug(f"Rate limiting: čekám {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, use_selenium: bool = None) -> Optional[BeautifulSoup]:
        """
        Provede HTTP požadavek s rate limitingem
        
        Args:
            url: URL pro stažení
            use_selenium: Použít Selenium místo requests
        
        Returns:
            BeautifulSoup objekt nebo None
        """
        self._rate_limit()
        
        if use_selenium is None:
            use_selenium = self.use_selenium
        
        try:
            logger.debug(f"Stahuji: {url}")
            
            if use_selenium and self.driver:
                self.driver.get(url)
                
                # Wait for the page to load
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
            logger.error(f"Chyba při stahování {url}: {e}")
            return None
    
    def get_season_results_urls(self, season: str) -> List[str]:
        """
        Získá URL všech výsledků pro danou sezónu
        
        Args:
            season: Sezóna ve formátu '2021-2022'
        
        Returns:
            Seznam URL jednotlivých zápasů
        """
        # Convert season to betexplorer format
        season_formatted = season.replace('-', '-20') if len(season.split('-')[0]) == 4 else season
        season_url = f"{self.base_url}/hockey/usa/nhl-{season_formatted}/results/"
        
        logger.info(f"Načítám výsledky pro sezónu {season}")
        
        soup = self._make_request(season_url)
        if not soup:
            logger.error(f"Nepodařilo se načíst výsledky pro sezónu {season}")
            return []
        
        match_urls = []
        
        # Find the table with the results
        results_table = soup.find('table', {'class': 'table-main'})
        if not results_table:
            logger.error("Nenalezena tabulka s výsledky")
            return []
        
        # Extract links to individual matches
        for row in results_table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 5:  # At least 5 columns expected
                # Search for a match link
                match_link = None
                for cell in cells:
                    link = cell.find('a')
                    if link and '/hockey/usa/nhl' in link.get('href', ''):
                        match_link = urljoin(self.base_url, link['href'])
                        break
                
                if match_link:
                    match_urls.append(match_link)
        
        logger.info(f"Nalezeno {len(match_urls)} zápasů pro sezónu {season}")
        return match_urls
    
    def extract_match_odds(self, match_url: str) -> Optional[Dict]:
        """
        Extrahuje kurzy pro konkrétní zápas
        
        Args:
            match_url: URL zápasu na betexplorer
        
        Returns:
            Slovník s kurzy a informacemi o zápase
        """
        soup = self._make_request(match_url)
        if not soup:
            return None
        
        try:
            # Extract basic match information
            match_info = self._extract_match_info(soup, match_url)
            if not match_info:
                return None
            
            # Extract odds
            odds_data = self._extract_odds_from_page(soup)
            
            # Combine information
            result = {
                **match_info,
                'odds': odds_data,
                'scraped_at': datetime.now(),
                'source_url': match_url
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Chyba při extrakci kurzů z {match_url}: {e}")
            return None
    
    def _extract_match_info(self, soup: BeautifulSoup, match_url: str) -> Optional[Dict]:
        """Extrahuje základní informace o zápase"""
        try:
            # Extract teams from the page
            title = soup.find('h1')
            if not title:
                logger.error("Nenalezen titulek stránky")
                return None
            
            title_text = title.get_text(strip=True)
            
            # Parse team names (format: "Team A - Team B")
            if ' - ' in title_text:
                teams = title_text.split(' - ')
                if len(teams) >= 2:
                    away_team = teams[0].strip()
                    home_team = teams[1].strip()
                else:
                    logger.error(f"Neočekávaný formát titulku: {title_text}")
                    return None
            else:
                logger.error(f"Neočekávaný formát titulku: {title_text}")
                return None
            
            # Map team names
            away_team = self.team_mapping.get(away_team, away_team)
            home_team = self.team_mapping.get(home_team, home_team)
            
            # Extract the date and result
            date_elem = soup.find('p', {'class': 'list-details'})
            match_date = None
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                # Parse date (various formats)
                match_date = self._parse_date(date_text)
            
            # Extract the result
            score_elem = soup.find('p', {'class': 'result'})
            home_score, away_score = None, None
            if score_elem:
                score_text = score_elem.get_text(strip=True)
                scores = self._parse_score(score_text)
                if scores:
                    home_score, away_score = scores
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'match_date': match_date,
                'home_score': home_score,
                'away_score': away_score,
                'status': 'completed' if home_score is not None else 'scheduled'
            }
            
        except Exception as e:
            logger.error(f"Chyba při extrakci informací o zápase: {e}")
            return None
    
    def _extract_odds_from_page(self, soup: BeautifulSoup) -> List[Dict]:
        """Extrahuje kurzy ze stránky zápasu"""
        odds_data = []
        
        try:
            # Search odds tables
            odds_tables = soup.find_all('table', {'class': 'table-main'})
            
            for table in odds_tables:
                # Find the table header
                header = table.find('thead')
                if not header:
                    continue
                
                # Extract columns
                header_row = header.find('tr')
                if not header_row:
                    continue
                
                columns = [th.get_text(strip=True) for th in header_row.find_all('th')]
                
                # Process rows with data
                tbody = table.find('tbody')
                if not tbody:
                    continue
                
                for row in tbody.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) < len(columns):
                        continue
                    
                    # Extract data from a row
                    row_data = {}
                    for i, cell in enumerate(cells):
                        if i < len(columns):
                            cell_text = cell.get_text(strip=True)
                            row_data[columns[i]] = cell_text
                    
                    # If the row contains odds, add it
                    if self._is_odds_row(row_data):
                        odds_entry = self._process_odds_row(row_data)
                        if odds_entry:
                            odds_data.append(odds_entry)
            
        except Exception as e:
            logger.error(f"Chyba při extrakci kurzů: {e}")
        
        return odds_data
    
    def _is_odds_row(self, row_data: Dict) -> bool:
        """Určí, zda řádek obsahuje kurzy"""
        # Search for columns with numeric values ​​(odds)
        for key, value in row_data.items():
            try:
                float_val = float(value.replace(',', '.'))
                if 1.0 <= float_val <= 100.0:  # Typical course range
                    return True
            except ValueError:
                continue
        return False
    
    def _process_odds_row(self, row_data: Dict) -> Optional[Dict]:
        """Zpracuje řádek s kurzy"""
        try:
            # Identify the bookmaker (usually the first column)
            bookmaker = None
            odds = {}
            
            for key, value in row_data.items():
                if not bookmaker and not self._is_numeric_odds(value):
                    bookmaker = value
                elif self._is_numeric_odds(value):
                    odds[key] = float(value.replace(',', '.'))
            
            if bookmaker and odds:
                return {
                    'bookmaker': bookmaker,
                    'odds': odds,
                    'market_type': 'main'  # Expand for different types of markets
                }
            
        except Exception as e:
            logger.error(f"Chyba při zpracování řádku kurzů: {e}")
        
        return None
    
    def _is_numeric_odds(self, value: str) -> bool:
        """Zkontroluje, zda hodnota vypadá jako kurz"""
        try:
            float_val = float(value.replace(',', '.'))
            return 1.0 <= float_val <= 100.0
        except ValueError:
            return False
    
    def _parse_date(self, date_text: str) -> Optional[date]:
        """Parsuje datum z různých formátů"""
        # Different data formats that betexplorer uses
        date_formats = [
            "%d.%m.%Y",
            "%d/%m/%Y", 
            "%Y-%m-%d",
            "%d.%m.%y",
            "%d/%m/%y"
        ]
        
        # Clear text
        date_clean = re.sub(r'[^\d./\-]', '', date_text.strip())
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_clean, fmt).date()
                return parsed_date
            except ValueError:
                continue
        
        logger.warning(f"Nepodařilo se parsovat datum: {date_text}")
        return None
    
    def _parse_score(self, score_text: str) -> Optional[Tuple[int, int]]:
        """Parsuje výsledek zápasu"""
        try:
            # Expected format: "3:2" or "3-2"
            score_clean = re.sub(r'[^\d:\-]', '', score_text.strip())
            
            if ':' in score_clean:
                parts = score_clean.split(':')
            elif '-' in score_clean:
                parts = score_clean.split('-')
            else:
                return None
            
            if len(parts) == 2:
                home_score = int(parts[0])
                away_score = int(parts[1])
                return (home_score, away_score)
                
        except (ValueError, IndexError):
            pass
        
        logger.warning(f"Nepodařilo se parsovat výsledek: {score_text}")
        return None
    
    def scrape_season_odds(self, season: str, max_matches: Optional[int] = None) -> List[Dict]:
        """
        Stáhne kurzy pro celou sezónu
        
        Args:
            season: Sezóna ve formátu '2021-2022'
            max_matches: Maximum zápasů ke stažení (pro testování)
        
        Returns:
            Seznam slovníků s kurzy a informacemi o zápasech
        """
        logger.info(f"🏒 Začínám stahování kurzů pro sezónu {season}")
        
        # Get URLs of all matches
        match_urls = self.get_season_results_urls(season)
        if not match_urls:
            logger.error(f"Nenalezeny žádné zápasy pro sezónu {season}")
            return []
        
        if max_matches:
            match_urls = match_urls[:max_matches]
            logger.info(f"Omezuji na prvních {max_matches} zápasů")
        
        # Download odds for each match
        season_odds = []
        success_count = 0
        
        for i, match_url in enumerate(match_urls, 1):
            logger.info(f"Zpracovávám zápas {i}/{len(match_urls)}: {match_url}")
            
            try:
                match_odds = self.extract_match_odds(match_url)
                if match_odds:
                    match_odds['season'] = season
                    season_odds.append(match_odds)
                    success_count += 1
                    
                    if success_count % 10 == 0:
                        logger.info(f"✅ Úspěšně zpracováno {success_count} zápasů")
                
            except Exception as e:
                logger.error(f"Chyba při zpracování zápasu {match_url}: {e}")
                continue
        
        logger.info(f"🎉 Dokončeno! Zpracováno {success_count}/{len(match_urls)} zápasů pro sezónu {season}")
        return season_odds
    
    def scrape_multiple_seasons(self, seasons: List[str], max_matches_per_season: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        Stáhne kurzy pro více sezón
        
        Args:
            seasons: Seznam sezón ve formátu ['2021-2022', '2022-2023']
            max_matches_per_season: Maximum zápasů na sezónu
        
        Returns:
            Slovník s kurzy pro každou sezónu
        """
        all_seasons_data = {}
        
        for season in seasons:
            logger.info(f"\n{'='*50}")
            logger.info(f"STAHOVÁNÍ SEZÓNY {season}")
            logger.info(f"{'='*50}")
            
            try:
                season_odds = self.scrape_season_odds(season, max_matches_per_season)
                all_seasons_data[season] = season_odds
                
                # Save running results
                self.save_season_data(season, season_odds)
                
            except Exception as e:
                logger.error(f"Chyba při stahování sezóny {season}: {e}")
                all_seasons_data[season] = []
        
        return all_seasons_data
    
    def save_season_data(self, season: str, season_data: List[Dict], 
                        output_dir: str = "data/odds") -> str:
        """
        Uloží data sezóny do CSV souboru
        
        Args:
            season: Název sezóny
            season_data: Data sezóny
            output_dir: Výstupní adresář
        
        Returns:
            Cesta k uloženému souboru
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"nhl_odds_{season}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        if not season_data:
            logger.warning(f"Žádná data k uložení pro sezónu {season}")
            return ""
        
        # Převeď data do DataFrame
        flattened_data = []
        
        for match in season_data:
            base_match_info = {
                'season': match.get('season'),
                'match_date': match.get('match_date'),
                'home_team': match.get('home_team'),
                'away_team': match.get('away_team'),
                'home_score': match.get('home_score'),
                'away_score': match.get('away_score'),
                'status': match.get('status'),
                'source_url': match.get('source_url'),
                'scraped_at': match.get('scraped_at')
            }
            
            # Expand odds
            if match.get('odds'):
                for odds_entry in match['odds']:
                    row = {**base_match_info}
                    row['bookmaker'] = odds_entry.get('bookmaker')
                    row['market_type'] = odds_entry.get('market_type')
                    
                    # Add odds
                    odds_dict = odds_entry.get('odds', {})
                    for market, odd_value in odds_dict.items():
                        row[f'odds_{market}'] = odd_value
                    
                    flattened_data.append(row)
            else:
                # No odds match
                flattened_data.append(base_match_info)
        
        # Save to CSV
        df = pd.DataFrame(flattened_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        logger.info(f"💾 Data uložena do {filepath} ({len(df)} řádků)")
        return filepath
    
    def close(self):
        """Ukončí scraper a uzavře WebDriver"""
        if self.driver:
            self.driver.quit()
            logger.info("🔒 WebDriver uzavřen")


def main():
    """Hlavní funkce pro spuštění scraperu"""
    
    # Create the necessary directories
    os.makedirs('data/odds', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Configuration
    SEASONS = ['2021-2022', '2022-2023', '2023-2024']  # Seasons to download
    MAX_MATCHES_PER_SEASON = 50  # For testing (50) - set to None for all matches
    USE_SELENIUM = True  # Use Selenium for dynamic content
    
    logger.info("🏒 Spouštím BetExplorer scraper pro NHL kurzy")
    logger.info(f"Sezóny: {', '.join(SEASONS)}")
    logger.info(f"Max zápasů na sezónu: {MAX_MATCHES_PER_SEASON or 'všechny'}")
    
    # Initialize scraper
    scraper = BetExplorerScraper(use_selenium=USE_SELENIUM)
    
    try:
        # Download data for all seasons
        all_data = scraper.scrape_multiple_seasons(
            seasons=SEASONS,
            max_matches_per_season=MAX_MATCHES_PER_SEASON
        )
        
        # Results summary
        logger.info("\n" + "="*60)
        logger.info("📊 SOUHRN STAHOVÁNÍ")
        logger.info("="*60)
        
        total_matches = 0
        for season, season_data in all_data.items():
            count = len(season_data)
            total_matches += count
            logger.info(f"  {season}: {count} zápasů")
        
        logger.info(f"  CELKEM: {total_matches} zápasů")
        
        # Save combined data
        if total_matches > 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            combined_file = f"data/odds/nhl_odds_combined_{timestamp}.json"
            
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"💾 Kombinovaná data uložena do {combined_file}")
        
        logger.info("🎉 Stahování dokončeno úspěšně!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Stahování přerušeno uživatelem")
    except Exception as e:
        logger.error(f"❌ Chyba při stahování: {e}")
        raise
    finally:
        # Close scraper
        scraper.close()


if __name__ == "__main__":
    main()