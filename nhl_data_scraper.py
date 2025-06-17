#!/usr/bin/env python3
"""
NHL Data Scraper for Hockey-Reference.com
Collects game results, team stats, and standings data.
"""

import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NHLScraper:
    """Scraper for NHL data from Hockey-Reference.com"""
    
    def __init__(self):
        self.base_url = "https://www.hockey-reference.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.delay = 2  # Respectful delay between requests
        
    def _make_request(self, url: str) -> Optional[BeautifulSoup]:
        """Make HTTP request with error handling and rate limiting"""
        try:
            logger.info(f"Fetching: {url}")
            time.sleep(self.delay)  # Be respectful to the server
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            return BeautifulSoup(response.content, 'html.parser')
            
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def _parse_game_time(self, time_cell) -> Optional[str]:
        """
        Parse game time from the time cell
        Returns time in HH:MM format or None if not found
        """
        if not time_cell:
            return None
            
        time_text = time_cell.text.strip()
        if not time_text or time_text in ['', 'Time']:
            return None
        
        try:
            # Try common Hockey-Reference formats
            formats_to_try = [
                '%I:%M %p',    # "1:00 PM", "7:30 AM"
                '%I %p',       # "7 PM", "1 AM"  
                '%H:%M',       # "19:00", "13:30"
                '%H'           # "19", "13"
            ]
            
            for fmt in formats_to_try:
                try:
                    parsed_time = datetime.strptime(time_text, fmt).time()
                    return parsed_time.strftime('%H:%M')
                except ValueError:
                    continue
            
            # If nothing worked, return None
            logger.warning(f"Could not parse time format: '{time_text}'")
            return None
            
        except Exception as e:
            logger.warning(f"Error parsing time '{time_text}': {e}")
            return None
    
    def _extract_boxscore_url(self, date_cell) -> str:
        """
        Extract boxscore URL from the date cell link
        
        Args:
            date_cell: BeautifulSoup cell containing the date with link
            
        Returns:
            Full boxscore URL or empty string if not found
        """
        try:
            # Look for the link in the date cell
            date_link = date_cell.find('a')
            if date_link and date_link.get('href'):
                relative_url = date_link.get('href')
                # Convert relative URL to absolute URL
                if relative_url.startswith('/'):
                    return f"{self.base_url}{relative_url}"
                else:
                    return relative_url
            
            return ""
            
        except Exception as e:
            logger.warning(f"Could not extract boxscore URL: {e}")
            return ""
            
    def get_season_schedule(self, season: str) -> pd.DataFrame:
        """
        Get all games for a specific season with enhanced datetime and boxscore URLs
        
        Args:
            season: Season year (e.g., '2024' for 2023-24 season)
            
        Returns:
            DataFrame with game results including datetime and boxscore URLs
        """
        url = f"{self.base_url}/leagues/NHL_{season}_games.html"
        soup = self._make_request(url)
        
        if not soup:
            return pd.DataFrame()
        
        # Find the games table
        games_table = soup.find('table', {'id': 'games'})
        if not games_table:
            logger.error(f"No games table found for season {season}")
            return pd.DataFrame()
        
        games_data = []
        
        # Parse table rows
        rows = games_table.find('tbody').find_all('tr')
        
        for row in rows:
            # Skip header rows
            if row.get('class') and 'thead' in row.get('class'):
                continue
                
            cells = row.find_all(['td', 'th'])
            if len(cells) < 7:  # Minimum expected columns
                continue
            
            try:
                # Extract game data - column positions may vary
                date_cell = cells[0]
                time_cell = cells[1]# if len(cells) > 7 else None
                visitor_cell = cells[2]# if len(cells) > 7 else cells[1]
                home_cell = cells[4]# if len(cells) > 7 else cells[3]
                
                # Parse date and extract boxscore URL from the same cell
                date_text = date_cell.text.strip()
                if not date_text or date_text in ['Date', '']:
                    continue
                    
                game_date = datetime.strptime(date_text, '%Y-%m-%d').date()
                
                # Extract boxscore URL from date cell
                boxscore_url = self._extract_boxscore_url(date_cell)
                
                # Parse time
                game_time = self._parse_game_time(time_cell)
                
                # Create datetime object
                if game_time:
                    hour, minute = game_time.split(':')
                    game_datetime = datetime.combine(
                        game_date, 
                        datetime.min.time().replace(hour=int(hour), minute=int(minute))
                    )
                    game_datetime_iso = game_datetime.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    # Default time if not found (most NHL games are around 7-8 PM)
                    game_datetime = datetime.combine(
                        game_date,
                        datetime.min.time().replace(hour=19, minute=0)  # Default 7:00 PM
                    )
                    game_datetime_iso = game_datetime.strftime('%Y-%m-%d %H:%M:%S')
                
                # Extract team names and scores
                visitor_team = visitor_cell.find('a')
                home_team = home_cell.find('a')
                
                if not visitor_team or not home_team:
                    continue
                    
                visitor_name = visitor_team.text.strip()
                home_name = home_team.text.strip()
                
                # Extract scores (if game is completed)
                visitor_score = None
                home_score = None
                
                # Adjust indices based on whether time column exists
                #if len(cells) > 7:  # Time column exists
                visitor_score_cell = cells[3]
                home_score_cell = cells[5]
                ot_cell_idx = 6
                #else:  # No time column
                 #   visitor_score_cell = cells[2]
                  #  home_score_cell = cells[4]
                   # ot_cell_idx = 5
                    
                try:
                    visitor_score = int(visitor_score_cell.text.strip())
                    home_score = int(home_score_cell.text.strip())
                except (ValueError, AttributeError):
                    # Game not yet played or in progress
                    pass
                
                # Determine game status
                status = 'completed' if visitor_score is not None else 'scheduled'
                
                # Overtime/Shootout info
                ot_so = ''
                if len(cells) > ot_cell_idx:
                    ot_cell = cells[ot_cell_idx]
                    ot_so = ot_cell.text.strip()
                
                game_data = {
                    'datetime': game_datetime_iso,  # ISO format with time
                    'date': game_date,              # Keep original date for compatibility
                    'season': season,
                    'visitor_team': visitor_name,
                    'home_team': home_name,
                    'visitor_score': visitor_score,
                    'home_score': home_score,
                    'overtime_shootout': ot_so,
                    'status': status,
                    'boxscore_url': boxscore_url,   # New boxscore URL
                    'scraped_at': datetime.now()
                }
                
                games_data.append(game_data)
                
            except Exception as e:
                logger.warning(f"Error parsing row: {e}")
                continue
        
        df = pd.DataFrame(games_data)
        logger.info(f"Scraped {len(df)} games for season {season}")
        
        return df
    
    def get_team_stats(self, season: str) -> pd.DataFrame:
        """
        Get team statistics for a specific season
        
        Args:
            season: Season year
            
        Returns:
            DataFrame with team statistics
        """
        url = f"{self.base_url}/leagues/NHL_{season}.html"
        soup = self._make_request(url)
        
        if not soup:
            return pd.DataFrame()
        
        # Find team stats table
        stats_table = soup.find('table', {'id': 'stats'})
        if not stats_table:
            logger.error(f"No team stats table found for season {season}")
            logger.error(f"Trying to find in comments for season {season}")
            
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            
            if not comments:
                return pd.DataFrame()
            
            for comment in comments:
                if 'id="stats"' in comment:
                    # Parsing HTML from comments
                    comment_soup = BeautifulSoup(comment, 'html.parser')
                    stats_table_1 = comment_soup.find('table', {'id': 'stats'})
                    
                    if stats_table_1:
                        stats_table = stats_table_1
                        break
        
        if not stats_table:
            logger.error(f"No team stats table found for season {season} even in comments")
            return pd.DataFrame()
        
        stats_data = []
        
        # Parse header to get column names
        header_rows = stats_table.find('thead').find_all('tr')
        headers = []
        for header_row in header_rows:
            headers1 = [th.text.strip() for th in header_row.find_all('th')]
            if len(headers1) < 6:
                continue
            headers = headers1            
        
        # Parse data rows
        rows = stats_table.find('tbody').find_all('tr')
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) != len(headers):
                continue
                
            row_data = {}
            for i, cell in enumerate(cells):
                if i < len(headers):
                    # Clean up team names (remove links)
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
        """
        Get standings for a specific season
        
        Args:
            season: Season year
            
        Returns:
            DataFrame with standings
        """
        url = f"{self.base_url}/leagues/NHL_{season}.html"
        soup = self._make_request(url)
        
        if not soup:
            return pd.DataFrame()
        
        standings_data = []
        
        # Find both conference tables
        for conference in ['EAS', 'WES']:
            table_id = f"standings_{conference.split()[0]}"
            table = soup.find('table', {'id': table_id})
            
            if not table:
                continue
                
            # Parse header
            header_row = table.find('thead').find('tr')
            headers = [th.text.strip() for th in header_row.find_all('th')]
            
            # Parse rows
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
        """
        Scrape multiple seasons of data
        
        Args:
            start_season: Starting season year
            end_season: Ending season year
            
        Returns:
            Dictionary with DataFrames for each data type
        """
        all_games = []
        all_stats = []
        all_standings = []
        
        for season_year in range(start_season, end_season + 1):
            season_str = str(season_year)
            logger.info(f"Scraping season {season_year-1}-{str(season_year)[2:]}")
            
            # Get games
            games_df = self.get_season_schedule(season_str)
            if not games_df.empty:
                all_games.append(games_df)
            
            # Get team stats
            stats_df = self.get_team_stats(season_str)
            if not stats_df.empty:
                all_stats.append(stats_df)
            
            # Get standings
            standings_df = self.get_standings(season_str)
            if not standings_df.empty:
                all_standings.append(standings_df)
            
            # Be extra respectful between seasons
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
    """Main function to run the scraper"""
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    scraper = NHLScraper()
    
    # Scrape last 3 seasons (2022-23, 2023-24, 2024-25)
    current_year = datetime.now().year
    if datetime.now().month >= 9:  # New season starts in September
        current_year += 1
    
    start_season = current_year - 3
    end_season = current_year
    
    logger.info(f"Starting NHL data scraping for seasons {start_season-1}-{start_season-1+1} to {end_season-1}-{str(end_season)[2:]}")
    
    # Scrape data
    data = scraper.scrape_multiple_seasons(start_season, end_season)
    
    # Save to CSV files
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    for data_type, df in data.items():
        filename = f"data/raw/nhl_{data_type}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(df)} records to {filename}")
    
    # Save combined summary
    summary = {
        'scrape_date': datetime.now(),
        'seasons_scraped': f"{start_season}-{end_season}",
        'total_games': len(data.get('games', [])),
        'total_teams': len(data.get('team_stats', [])),
        'data_files': list(data.keys()),
        'enhancements': 'datetime_iso_format,boxscore_urls'
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"data/raw/nhl_scrape_summary_{timestamp}.csv", index=False)
    
    logger.info("NHL data scraping completed successfully!")
    
    # Display basic statistics
    for data_type, df in data.items():
        logger.info(f"\n{data_type.upper()} SUMMARY:")
        logger.info(f"  Total records: {len(df)}")
        if data_type == 'games':
            completed_games = df[df['status'] == 'completed']
            logger.info(f"  Completed games: {len(completed_games)}")
            logger.info(f"  Scheduled games: {len(df) - len(completed_games)}")
            logger.info(f"  Enhanced features: datetime (ISO), boxscore URLs")
        
        logger.info(f"  Date range: {df['season'].min()} - {df['season'].max()}")

if __name__ == "__main__":
    main()