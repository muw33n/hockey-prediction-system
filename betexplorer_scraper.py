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
        #season_formatted = season.replace('-', '-20') if len(season.split('-')[0]) == 4 else season
        season_url = f"{self.base_url}/hockey/usa/nhl-{season}/results/"
        
        logger.info(f"Načítám výsledky pro sezónu {season}")
        
        soup = self._make_request(season_url)
        if not soup:
            logger.error(f"Nepodařilo se načíst výsledky pro sezónu {season}")
            return []

        # Najdi element s title="Main season game statistics" a získej stage URL
        stage_element = soup.find(attrs={'title': 'Main season game statistics'})
        if not stage_element:
            logger.error("Nenalezen element s title='Main season game statistics'")
            return []
        
        # Získej href z elementu (může být buď přímo href nebo v parent elementu)
        stage_href = None
        if stage_element.name == 'a' and stage_element.get('href'):
            stage_href = stage_element['href']
        else:
            # Hledej parent element, který je link
            parent = stage_element.find_parent('a')
            if parent and parent.get('href'):
                stage_href = parent['href']
        
        if not stage_href:
            logger.error("Nenalezen href pro Main season game statistics")
            return []
        
        # Vytvoř novou URL s &month=all
        stage_url = urljoin(season_url, stage_href)
        if '?' in stage_url:
            results_url = f"{stage_url}&month=all"
        else:
            results_url = f"{stage_url}?month=all"
        
        logger.info(f"Používám stage URL: {results_url}")
        
        # Načti stránku s kompletními výsledky
        soup = self._make_request(results_url)
        if not soup:
            logger.error(f"Nepodařilo se načíst kompletní výsledky pro sezónu {season}")
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
            odds_data = self._extract_odds_from_page(soup, match_url)
            
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
            # Extrahuj týmy z list-details struktury
            team_elements = soup.find_all('h2', {'class': 'list-details__item__title'})
            
            if len(team_elements) < 2:
                logger.error("Nenalezeny týmy na stránce")
                return None
            
            # První tým je domácí, druhý je hostující (podle pořadí v HTML)
            home_team = team_elements[0].get_text(strip=True)
            away_team = team_elements[1].get_text(strip=True)
            
            # Mapuj názvy týmů
            home_team = self.team_mapping.get(home_team, home_team)
            away_team = self.team_mapping.get(away_team, away_team)
            
            # Extrahuj datum z data-dt atributu
            match_date = None
            date_elem = soup.find(attrs={'data-dt': True})
            if date_elem:
                date_string = date_elem.get('data-dt')
                # Format: "31,10,2021,21,00"
                date_parts = date_string.split(',')
                if len(date_parts) >= 3:
                    try:
                        day = int(date_parts[0])
                        month = int(date_parts[1])
                        year = int(date_parts[2])
                        match_date = date(year, month, day)
                    except (ValueError, IndexError):
                        logger.warning(f"Nepodařilo se parsovat datum: {date_string}")
            
            # Extrahuj výsledek z js-score elementu
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
                        logger.warning(f"Nepodařilo se parsovat skóre: {score_text}")
            
            # Určení statusu zápasu
            status = 'scheduled'  # Výchozí hodnota
            
            # Zkontroluj isFinished hodnotu
            finished_elem = soup.find(id='isFinished')
            if finished_elem:
                finished_value = finished_elem.get('value', '0')
                if finished_value == '1':
                    status = 'completed'
            
            # Zkontroluj isLive hodnotu
            live_elem = soup.find(id='isLive')
            if live_elem:
                live_value = live_elem.get('value', '')
                if live_value and live_value != '':
                    status = 'live'
            
            # Pokud máme skóre ale status není completed, pravděpodobně je zápas dokončen
            if home_score is not None and away_score is not None and status == 'scheduled':
                status = 'completed'
            
            # Extrahuj podrobnosti skóre (periody) pokud existují
            partial_score = None
            partial_elem = soup.find(id='js-partial')
            if partial_elem:
                partial_score = partial_elem.get_text(strip=True)
            
            result = {
                'home_team': home_team,
                'away_team': away_team,
                'match_date': match_date,
                'home_score': home_score,
                'away_score': away_score,
                'status': status
            }
            
            # Přidej podrobnosti skóre pokud existují
            if partial_score:
                result['partial_score'] = partial_score
            
            logger.debug(f"Extrahované informace: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Chyba při extrakci informací o zápase: {e}")
            return None

    def _extract_odds_from_page(self, soup: BeautifulSoup, match_url: str) -> List[Dict]:
        """
        Extrahuje kurzy ze stránky zápasu pomocí API
        
        Args:
            soup: BeautifulSoup objekt stránky (pro fallback)
            match_url: URL zápasu pro extrakci ID
        
        Returns:
            Seznam slovníků s kurzy
        """
        odds_data = []
        
        try:
            # Extrahuj match ID z URL
            match_id = self._extract_match_id_from_url(match_url)
            if not match_id:
                logger.error(f"Nepodařilo se extrahovat match ID z URL: {match_url}")
                return []
            
            logger.debug(f"Extrahován match ID: {match_id}")
            
            # Získej kurzy pro různé typy trhů
            market_types = [
                ('HA', 'moneyline_2way'),  # Domácí/Hosté (2-way)
                ('1x2', '1x2')            # 1X2 (3-way) - pro úplnost
            ]
            
            for market_code, market_name in market_types:
                try:
                    market_odds = self._fetch_match_odds(match_id, market_code, market_name)
                    if market_odds:
                        odds_data.extend(market_odds)
                except Exception as e:
                    logger.warning(f"Chyba při načítání kurzů pro trh {market_name}: {e}")
                    continue
            
            logger.info(f"Extrahováno {len(odds_data)} kurzů pro zápas {match_id}")
            
        except Exception as e:
            logger.error(f"Chyba při extrakci kurzů: {e}")
        
        return odds_data
    
    def _extract_match_id_from_url(self, match_url: str) -> Optional[str]:
        """
        Extrahuje ID zápasu z URL
        
        Args:
            match_url: URL zápasu (např. https://www.betexplorer.com/hockey/usa/nhl/buffalo-sabres-philadelphia-flyers/z1sKAia5/)
        
        Returns:
            8-znakové ID zápasu nebo None
        """
        try:
            # URL formát: .../team1-team2/MATCH_ID/
            # Nebo: .../team1-team2/MATCH_ID/odds/
            
            # Rozděl URL podle '/'
            url_parts = match_url.rstrip('/').split('/')
            
            # Hledej 8-znakový identifikátor
            for part in reversed(url_parts):  # Začni od konce
                if len(part) == 8 and part.isalnum():
                    return part
            
            # Fallback: regex pro 8 znaků
            import re
            match_id_pattern = r'/([a-zA-Z0-9]{8})/?(?:odds/?)?'
            match = re.search(match_id_pattern, match_url)
            if match:
                return match.group(1)
            
            logger.error(f"Match ID nenalezeno v URL: {match_url}")
            return None
            
        except Exception as e:
            logger.error(f"Chyba při extrakci match ID: {e}")
            return None

    def _fetch_match_odds(self, match_id: str, market_code: str, market_name: str) -> List[Dict]:
        """Načte kurzy pro konkrétní zápas a trh"""
        try:
            # Sestav API URL
            odds_url = f"{self.base_url}/match-odds-old/{match_id}/1/{market_code}/1/en/"
            
            logger.debug(f"Načítám kurzy z: {odds_url}")
            
            # Proveď API požadavek
            response = self.session.get(odds_url, timeout=30)
            response.raise_for_status()
            
            # Parsuj JSON odpověď
            try:
                data = response.json()
            except ValueError as e:
                logger.error(f"Neplatná JSON odpověď z {odds_url}: {e}")
                return []
            
            # Zpracuj kurzy z odpovědi
            if 'odds' not in data:
                logger.warning(f"Klíč 'odds' nenalezen v odpovědi pro {market_name}")
                return []
            
            return self._parse_odds_response(data, market_name)
            
        except Exception as e:
            logger.error(f"Chyba při načítání kurzů pro {market_name}: {e}")
            return []
        
    def _parse_odds_response(self, response_data: Dict, market_name: str) -> List[Dict]:
        """Parsuje JSON odpověď s kurzy (obsahuje HTML fragment)"""
        odds_list = []
        
        try:
            # Získej HTML fragment z klíče "odds"
            html_content = response_data.get('odds', '')
            if not html_content:
                logger.warning(f"Prázdný HTML fragment pro {market_name}")
                return []
            
            # Parsuj HTML fragment
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Najdi tabulku s kurzy
            odds_table = soup.find('table', class_='table-main')
            if not odds_table:
                logger.warning(f"Tabulka s kurzy nenalezena pro {market_name}")
                return []
            
            # Získej header pro identifikaci sloupců
            header_row = odds_table.find('thead')
            if header_row:
                header_cells = header_row.find_all('th')
                logger.debug(f"Header sloupce: {[th.get_text(strip=True) for th in header_cells]}")
            
            # Parsuj řádky s kurzy
            tbody = odds_table.find('tbody')
            if not tbody:
                logger.warning(f"Tělo tabulky nenalezeno pro {market_name}")
                return []
            
            for row in tbody.find_all('tr'):
                try:
                    # Extrahuj název bookmaker
                    bookmaker_cell = row.find('a', class_='in-bookmaker-logo-link')
                    if not bookmaker_cell:
                        continue

                    bookmaker_name = bookmaker_cell.get_text(strip=True)
                    
                    # Najdi buňky s kurzy (mají data-odd atribut)
                    odds_cells = row.find_all('td', attrs={'data-odd': True})
                    
                    if len(odds_cells) < 2:
                        logger.debug(f"Nedostatek kurzů pro {bookmaker_name}")
                        continue
                    
                    # Extrahuj kurzy a další metadata
                    home_cell = odds_cells[0]  # První kurz = domácí
                    away_cell = odds_cells[1]  # Druhý kurz = hosté
                    
                    home_odds = float(home_cell.get('data-odd'))
                    away_odds = float(away_cell.get('data-odd'))
                    
                    # Extrahuj datum posledního updatu
                    home_created = home_cell.get('data-created', '')
                    away_created = away_cell.get('data-created', '')
                    
                    # Extrahuj opening odds
                    home_opening = home_cell.get('data-opening-odd')
                    away_opening = away_cell.get('data-opening-odd')
                    opening_date = home_cell.get('data-opening-date', '')
                    
                    # Sestavení záznamu
                    odds_entry = {
                        'bookmaker': bookmaker_name,
                        'market_type': market_name,
                        'odds': {
                            'home': home_odds,
                            'away': away_odds
                        },
                        'timestamp': datetime.now(),
                        'last_updated': self._parse_betexplorer_datetime(home_created),
                        'metadata': {
                            'home_opening_odds': float(home_opening) if home_opening else None,
                            'away_opening_odds': float(away_opening) if away_opening else None,
                            'opening_date': self._parse_betexplorer_datetime(opening_date),
                            'bookmaker_id': home_cell.get('data-bookie-id'),
                            'bet_url': home_cell.get('data-bet-url')
                        }
                    }
                    
                    odds_list.append(odds_entry)
                    logger.debug(f"Parsován kurz: {bookmaker_name} - Domácí: {home_odds}, Hosté: {away_odds}")
                    
                except Exception as e:
                    logger.warning(f"Chyba při parsování řádku kurzů: {e}")
                    continue

            # Pokus o extrakci průměrných kurzů
            tfoot = odds_table.find('tfoot')
            if tfoot:
                avg_cells = tfoot.find_all('td', attrs={'data-odd': True})
                if len(avg_cells) >= 2:
                    try:
                        avg_home = float(avg_cells[0].get('data-odd'))
                        avg_away = float(avg_cells[1].get('data-odd'))

                        odds_entry = {
                            'bookmaker': 'Average',
                            'market_type': market_name,
                            'odds': {
                                'home': avg_home,
                                'away': avg_away
                            },
                            'timestamp': datetime.now(),
                            'metadata': {
                                'is_average': True
                            }
                        }
                        odds_list.append(odds_entry)
                        logger.debug(f"Parsován průměr: Domácí: {avg_home}, Hosté: {avg_away}")
                        
                    except Exception as e:
                        logger.warning(f"Chyba při parsování průměrných kurzů: {e}")
            
            logger.info(f"Parsováno {len(odds_list)} kurzů pro {market_name}")
            
        except Exception as e:
            logger.error(f"Chyba při parsování kurzů: {e}")
        
        return odds_list
 
    def _parse_betexplorer_datetime(self, datetime_str: str) -> Optional[datetime]:
        """Parsuje datum/čas z betexplorer formátu"""
        if not datetime_str:
            return None
        
        try:
            # Format: "18,04,2025,00,48" (den,měsíc,rok,hodina,minuta)
            parts = datetime_str.split(',')
            if len(parts) >= 5:
                day = int(parts[0])
                month = int(parts[1])
                year = int(parts[2])
                hour = int(parts[3])
                minute = int(parts[4])
                
                return datetime(year, month, day, hour, minute)
            elif len(parts) >= 3:
                day = int(parts[0])
                month = int(parts[1])
                year = int(parts[2])
                
                return datetime(year, month, day)
                
        except (ValueError, IndexError):
            logger.warning(f"Nepodařilo se parsovat datum: {datetime_str}")
        
        return None
    
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
