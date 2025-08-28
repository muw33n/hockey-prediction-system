#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Database Setup (ENHANCED MIGRATED)
==============================================================
Database setup s enhanced infrastructure: per-component logging,
safe file handling, centralized paths a performance monitoring.

Umístění: src/database/database_setup.py

MIGRATION NOTES:
- Enhanced per-component logging (database.log)
- Safe file handlers s automatic encoding detection  
- Performance monitoring pro kritické operace
- PATHS integration for all file operations
- Robust error handling s detailed logging
"""

import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import re
from typing import List, Dict, Optional, Set
from pathlib import Path

# === ENHANCED MIGRATION: Centralized imports ===
from config.paths import PATHS
from config.logging_config import (
    setup_logging, 
    get_component_logger, 
    PerformanceLogger
)
from src.utils.file_handlers import (
    read_csv,
    load_latest_games_data,
    FileHandler
)

# === ENHANCED MIGRATION: Setup per-component logging ===
setup_logging(
    log_level='INFO',
    log_to_file=True,
    component_files=True  # Key: per-component log files
)

# Component-specific logger for database operations
logger = get_component_logger(__name__, 'database')


class EnhancedDatabaseManager:
    """
    Enhanced Database Manager s kompletní migrací na enhanced infrastructure.
    
    Features:
    - Per-component logging (database.log)
    - Safe file handling s encoding detection
    - Performance monitoring
    - PATHS integration
    - Robust error handling
    """
    
    def __init__(self):
        """Inicializace s enhanced infrastructure"""
        
        # === MIGRATION: Performance logger setup ===
        self.perf_logger = PerformanceLogger(logger)
        
        # Database konfigurace - simplified (odstranil settings dependency)
        # TODO: Move to environment variables or config file
        self.database_url = self._get_database_url()
        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # === MIGRATION: Enhanced paths s PATHS integration ===
        self.data_paths = {
            'nhl_data': PATHS.raw_data,
            'odds_data': PATHS.odds_data
        }
        
        # Zajištění existence adresářů
        PATHS.ensure_directories()
        
        # Enhanced startup logging
        self.log_enhanced_startup()
        
        # Team name mapping (unchanged, business logic preserved)
        self.team_name_mapping = {
            'Arizona Coyotes': 'Utah Mammoth',
            'Utah Hockey Club': 'Utah Mammoth'
        }
        
        logger.info("🎯 Enhanced DatabaseManager initialized successfully")
        logger.info(f"   Database: {self._get_db_info()}")
        logger.info(f"   NHL Data: {self.data_paths['nhl_data']}")
        logger.info(f"   Odds Data: {self.data_paths['odds_data']}")
    
    def _get_database_url(self) -> str:
        """Get database URL - simplified configuration"""
        import os
        
        # Try environment variables first
        if 'DATABASE_URL' in os.environ:
            return os.environ['DATABASE_URL']
        
        # Fallback to default configuration
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        database = os.getenv('DB_NAME', 'hockey_db')
        username = os.getenv('DB_USER', 'hockey_user')
        password = os.getenv('DB_PASSWORD', 'hockey_pass')
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    def _get_db_info(self) -> str:
        """Get database info for logging"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.database_url)
            return f"{parsed.hostname}:{parsed.port}/{parsed.path[1:]}"
        except:
            return "database connection"
    
    def log_enhanced_startup(self):
        """Enhanced startup logging s file discovery"""
        
        logger.info("🔍 Enhanced startup - scanning for data files...")
        
        # === MIGRATION: Use PATHS for file discovery ===
        file_patterns = [
            ('Games', 'nhl_games_*.csv', self.data_paths['nhl_data']),
            ('Team Stats', 'nhl_team_stats_*.csv', self.data_paths['nhl_data']), 
            ('Standings', 'nhl_standings_*.csv', self.data_paths['nhl_data']),
            ('Odds', 'nhl_odds_*.csv', self.data_paths['odds_data'])
        ]
        
        for file_type, pattern, directory in file_patterns:
            try:
                files = list(directory.glob(pattern))
                if files:
                    files.sort(reverse=True)  # Nejnovější první
                    logger.info(f"  📁 {file_type}: {len(files)} files found")
                    
                    # Show latest files with enhanced timestamp extraction
                    for i, file in enumerate(files[:3]):
                        timestamp = self.extract_timestamp_from_filename(file)
                        age_indicator = "🆕" if i == 0 else "📄"
                        logger.info(f"    {age_indicator} {file.name} ({timestamp})")
                    
                    if len(files) > 3:
                        logger.info(f"    ... and {len(files) - 3} more files")
                else:
                    logger.warning(f"  📁 {file_type}: No files found matching '{pattern}'")
                    
            except Exception as e:
                logger.error(f"Error scanning {file_type}: {e}")
    
    def extract_timestamp_from_filename(self, filename: Path) -> str:
        """Enhanced timestamp extraction s better error handling"""
        try:
            # Extract timestamp from pattern: nhl_games_20250616_190551.csv
            match = re.search(r'_(\d{8}_\d{6})\.csv$', str(filename))
            if match:
                timestamp_str = match.group(1)
                # Convert to readable format
                date_part = timestamp_str[:8]
                time_part = timestamp_str[9:]
                formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                return f"{formatted_date} {formatted_time}"
            return "Unknown timestamp"
        except Exception as e:
            logger.debug(f"Timestamp extraction failed for {filename}: {e}")
            return "Invalid timestamp"
    
    def extract_sort_key(self, filepath: Path) -> str:
        """Enhanced sort key extraction"""
        try:
            match = re.search(r'_(\d{8}_\d{6})\.csv$', filepath.name)
            return match.group(1) if match else "00000000_000000"
        except Exception:
            return "00000000_000000"
    
    def check_permissions(self) -> bool:
        """Enhanced permission check s better error reporting"""
        
        self.perf_logger.start_timer('permission_check')
        
        try:
            with self.engine.connect() as conn:
                # Test table creation and drop
                conn.execute(text("CREATE TABLE IF NOT EXISTS test_permissions (id SERIAL PRIMARY KEY)"))
                conn.execute(text("DROP TABLE IF EXISTS test_permissions"))
                conn.commit()
                
                duration = self.perf_logger.end_timer('permission_check')
                logger.info("✅ Database permissions verified")
                return True
                
        except Exception as e:
            self.perf_logger.end_timer('permission_check')
            logger.error(f"❌ Permission check failed: {e}")
            
            # Enhanced troubleshooting guidance
            if "permission denied" in str(e).lower():
                logger.error("💡 Permission issue detected. Run these commands as postgres user:")
                logger.error("   GRANT ALL ON SCHEMA public TO hockey_user;")
                logger.error("   GRANT ALL ON ALL TABLES IN SCHEMA public TO hockey_user;")
                logger.error("   GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO hockey_user;")
            
            return False
    
    def create_tables(self) -> bool:
        """Enhanced table creation s performance monitoring"""
        
        if not self.check_permissions():
            logger.error("❌ Insufficient permissions. Please fix permissions first.")
            return False
        
        self.perf_logger.start_timer('table_creation')
        
        # Table creation SQL (unchanged - preserve business logic)
        create_tables_sql = """
        -- Drop existing tables if they exist
        DROP TABLE IF EXISTS value_bets CASCADE;
        DROP TABLE IF EXISTS predictions CASCADE;
        DROP TABLE IF EXISTS odds CASCADE;
        DROP TABLE IF EXISTS game_urls CASCADE;
        DROP TABLE IF EXISTS goalie_stats CASCADE;
        DROP TABLE IF EXISTS player_stats CASCADE;
        DROP TABLE IF EXISTS team_stats CASCADE;
        DROP TABLE IF EXISTS games CASCADE;
        DROP TABLE IF EXISTS team_history CASCADE;
        DROP TABLE IF EXISTS teams CASCADE;
        DROP TABLE IF EXISTS venues CASCADE;
        DROP TABLE IF EXISTS franchises CASCADE;
        DROP TABLE IF EXISTS leagues CASCADE;
        
        -- Leagues table
        CREATE TABLE leagues (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            country VARCHAR(50),
            level INTEGER,
            season_start INTEGER,
            season_end INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Franchises table - core entity that survives relocations
        CREATE TABLE franchises (
            id SERIAL PRIMARY KEY,
            franchise_name VARCHAR(100) NOT NULL,
            founded_date DATE,
            founded_city VARCHAR(100),
            is_active BOOLEAN DEFAULT TRUE,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Simplified venues table (to be populated in future)
        CREATE TABLE venues (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            city VARCHAR(100),
            country VARCHAR(50),
            capacity INTEGER,
            latitude DECIMAL(10,8),
            longitude DECIMAL(11,8),
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Enhanced teams table
        CREATE TABLE teams (
            id SERIAL PRIMARY KEY,
            franchise_id INTEGER REFERENCES franchises(id),
            name VARCHAR(100) NOT NULL,
            city VARCHAR(100),
            league_id INTEGER REFERENCES leagues(id),
            conference VARCHAR(50),
            division VARCHAR(50),
            abbreviation VARCHAR(10),
            effective_from DATE NOT NULL,
            effective_to DATE,
            is_current BOOLEAN DEFAULT FALSE,
            change_reason VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(franchise_id, effective_from)
        );
        
        -- Team history tracking
        CREATE TABLE team_history (
            id SERIAL PRIMARY KEY,
            franchise_id INTEGER REFERENCES franchises(id),
            from_team_id INTEGER REFERENCES teams(id),
            to_team_id INTEGER REFERENCES teams(id),
            change_date DATE NOT NULL,
            change_type VARCHAR(50) NOT NULL,
            from_city VARCHAR(100),
            to_city VARCHAR(100),
            from_name VARCHAR(100),
            to_name VARCHAR(100),
            description TEXT,
            source VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Games table with simplified venue reference
        CREATE TABLE games (
            id SERIAL PRIMARY KEY,
            date DATE NOT NULL,
            datetime_et TIMESTAMP WITHOUT TIME ZONE,
            season INTEGER NOT NULL,
            league_id INTEGER REFERENCES leagues(id),
            home_team_id INTEGER REFERENCES teams(id),
            away_team_id INTEGER REFERENCES teams(id),
            venue_id INTEGER REFERENCES venues(id),
            home_score INTEGER,
            away_score INTEGER,
            overtime_shootout VARCHAR(20),
            status VARCHAR(20) DEFAULT 'scheduled',
            game_type VARCHAR(50) DEFAULT 'regular',
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            data_source VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, home_team_id, away_team_id)
        );
        
        -- Game URLs table
        CREATE TABLE game_urls (
            id SERIAL PRIMARY KEY,
            game_id INTEGER REFERENCES games(id) ON DELETE CASCADE,
            url_type VARCHAR(50) NOT NULL,
            url TEXT NOT NULL,
            source VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(game_id, url_type, source)
        );
        
        -- Odds table for moneyline 2-way betting
        CREATE TABLE odds (
            id SERIAL PRIMARY KEY,
            game_id INTEGER REFERENCES games(id) ON DELETE CASCADE,
            bookmaker VARCHAR(100) NOT NULL,
            market_type VARCHAR(50) NOT NULL,
            home_odd DECIMAL(8,4),
            home_opening_odd DECIMAL(8,4),
            home_opening_datetime TIMESTAMP,
            home_last_update TIMESTAMP,
            away_odd DECIMAL(8,4),
            away_opening_odd DECIMAL(8,4),
            away_opening_datetime TIMESTAMP,
            away_last_update TIMESTAMP,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            data_source VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(game_id, bookmaker, market_type)
        );
        
        -- Team stats table
        CREATE TABLE team_stats (
            id SERIAL PRIMARY KEY,
            team_id INTEGER REFERENCES teams(id),
            season INTEGER NOT NULL,
            games_played INTEGER,
            wins INTEGER,
            losses INTEGER,
            overtime_losses INTEGER,
            points INTEGER,
            points_percentage DECIMAL(5,3),
            goals_for INTEGER,
            goals_against INTEGER,
            shootout_wins INTEGER,
            shootout_losses INTEGER,
            srs DECIMAL(6,3),
            sos DECIMAL(6,3),
            goals_for_per_game DECIMAL(5,2),
            goals_against_per_game DECIMAL(5,2),
            power_play_goals INTEGER,
            power_play_opportunities INTEGER,
            power_play_percentage DECIMAL(5,2),
            penalty_kill_percentage DECIMAL(5,2),
            short_handed_goals INTEGER,
            short_handed_goals_allowed INTEGER,
            shots INTEGER,
            shot_percentage DECIMAL(5,2),
            shots_against INTEGER,
            save_percentage DECIMAL(5,3),
            shutouts INTEGER,
            penalties_per_game DECIMAL(5,2),
            opponent_penalties_per_game DECIMAL(5,2),
            average_age DECIMAL(4,1),
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(team_id, season)
        );
        
        -- Additional tables (goalie_stats, player_stats, predictions, value_bets)
        CREATE TABLE goalie_stats (
            id SERIAL PRIMARY KEY,
            team_id INTEGER REFERENCES teams(id),
            season INTEGER NOT NULL,
            player_name VARCHAR(100),
            games_played INTEGER,
            wins INTEGER,
            losses INTEGER,
            overtime_losses INTEGER,
            saves INTEGER,
            shots_against INTEGER,
            save_percentage DECIMAL(5,3),
            goals_against_average DECIMAL(5,2),
            shutouts INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE player_stats (
            id SERIAL PRIMARY KEY,
            team_id INTEGER REFERENCES teams(id),
            season INTEGER NOT NULL,
            player_name VARCHAR(100),
            position VARCHAR(10),
            games_played INTEGER,
            goals INTEGER,
            assists INTEGER,
            points INTEGER,
            plus_minus INTEGER,
            penalty_minutes INTEGER,
            shots INTEGER,
            shooting_percentage DECIMAL(5,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE predictions (
            id SERIAL PRIMARY KEY,
            game_id INTEGER REFERENCES games(id),
            model_name VARCHAR(100),
            model_version VARCHAR(50),
            home_win_probability DECIMAL(5,4),
            away_win_probability DECIMAL(5,4),
            prediction_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            features_used TEXT,
            confidence_score DECIMAL(5,4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE value_bets (
            id SERIAL PRIMARY KEY,
            game_id INTEGER REFERENCES games(id),
            odds_id INTEGER REFERENCES odds(id),
            prediction_id INTEGER REFERENCES predictions(id),
            bet_type VARCHAR(50),
            recommended_stake DECIMAL(10,2),
            expected_value DECIMAL(8,4),
            kelly_percentage DECIMAL(5,4),
            confidence_level VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better performance
        CREATE INDEX idx_games_date ON games(date);
        CREATE INDEX idx_games_season ON games(season);
        CREATE INDEX idx_games_teams ON games(home_team_id, away_team_id);
        CREATE INDEX idx_games_venue ON games(venue_id);
        CREATE INDEX idx_odds_game_bookmaker ON odds(game_id, bookmaker);
        CREATE INDEX idx_team_stats_season ON team_stats(season);
        CREATE INDEX idx_predictions_game ON predictions(game_id);
        CREATE INDEX idx_value_bets_game ON value_bets(game_id);
        CREATE INDEX idx_teams_franchise ON teams(franchise_id);
        CREATE INDEX idx_teams_effective_dates ON teams(effective_from, effective_to);
        CREATE INDEX idx_teams_current ON teams(is_current) WHERE is_current = TRUE;
        CREATE INDEX idx_team_history_franchise ON team_history(franchise_id);
        CREATE INDEX idx_team_history_date ON team_history(change_date);
        CREATE INDEX idx_venues_active ON venues(is_active) WHERE is_active = TRUE;
        
        -- Helper views
        CREATE VIEW current_teams AS
        SELECT t.*, f.franchise_name 
        FROM teams t 
        JOIN franchises f ON t.franchise_id = f.id 
        WHERE t.is_current = TRUE;
        
        CREATE OR REPLACE FUNCTION get_team_for_date(franchise_id_param INT, game_date DATE)
        RETURNS TABLE(team_id INT, team_name VARCHAR, city VARCHAR, abbreviation VARCHAR) AS $$
        BEGIN
            RETURN QUERY
            SELECT t.id, t.name, t.city, t.abbreviation
            FROM teams t
            WHERE t.franchise_id = franchise_id_param
              AND t.effective_from <= game_date
              AND (t.effective_to IS NULL OR t.effective_to > game_date);
        END;
        $$ LANGUAGE plpgsql;
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_tables_sql))
                conn.commit()
                
                duration = self.perf_logger.end_timer('table_creation')
                logger.info("✅ All database tables created successfully")
                return True
                
        except Exception as e:
            self.perf_logger.end_timer('table_creation')
            logger.error(f"❌ Error creating tables: {e}")
            
            if "permission denied" in str(e).lower():
                logger.error("💡 Permission issue detected. Please run these commands as postgres user:")
                logger.error("   GRANT ALL ON SCHEMA public TO hockey_user;")
                logger.error("   GRANT ALL ON ALL TABLES IN SCHEMA public TO hockey_user;")
                logger.error("   GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO hockey_user;")
            
            return False
    
    def insert_initial_data(self) -> bool:
        """Enhanced initial data insertion s performance tracking"""
        
        self.perf_logger.start_timer('initial_data_insertion')
        
        try:
            with self.engine.connect() as conn:
                # Insert NHL league
                nhl_league_sql = """
                INSERT INTO leagues (name, country, level, season_start, season_end)
                VALUES ('NHL', 'North America', 1, 2021, 2025)
                ON CONFLICT DO NOTHING
                RETURNING id;
                """
                
                result = conn.execute(text(nhl_league_sql))
                league_row = result.fetchone()
                
                if league_row:
                    nhl_league_id = league_row[0]
                else:
                    result = conn.execute(text("SELECT id FROM leagues WHERE name = 'NHL'"))
                    nhl_league_id = result.fetchone()[0]
                
                # Insert franchises (business logic unchanged)
                franchises_data = [
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
                
                franchises_inserted = 0
                for franchise_id, name, founded_date, founded_city in franchises_data:
                    franchise_sql = """
                    INSERT INTO franchises (id, franchise_name, founded_date, founded_city, is_active)
                    VALUES (:id, :name, :founded_date, :founded_city, TRUE)
                    ON CONFLICT (id) DO NOTHING;
                    """
                    conn.execute(text(franchise_sql), {
                        'id': franchise_id,
                        'name': name,
                        'founded_date': founded_date,
                        'founded_city': founded_city
                    })
                    franchises_inserted += 1
                
                # Insert current teams (business logic unchanged)
                current_teams = [
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
                
                teams_inserted = 0
                for franchise_id, name, city, conference, division, abbrev, effective_from in current_teams:
                    try:
                        team_sql = """
                        INSERT INTO teams (franchise_id, name, city, league_id, conference, division, 
                                         abbreviation, effective_from, is_current)
                        VALUES (:franchise_id, :name, :city, :league_id, :conference, :division, 
                               :abbreviation, :effective_from, TRUE)
                        ON CONFLICT (franchise_id, effective_from) DO NOTHING;
                        """
                        
                        conn.execute(text(team_sql), {
                            'franchise_id': franchise_id,
                            'name': name,
                            'city': city,
                            'league_id': nhl_league_id,
                            'conference': conference,
                            'division': division,
                            'abbreviation': abbrev,
                            'effective_from': effective_from
                        })
                        teams_inserted += 1
                    except Exception as e:
                        logger.warning(f"Failed to insert team {name}: {e}")
                
                # Insert historical Arizona/Utah transitions
                historical_arizona_teams = [
                    (23, 'Winnipeg Jets', 'Winnipeg', 'Western', 'Smythe', 'WPG', '1979-10-10', '1996-04-13'),
                    (23, 'Phoenix Coyotes', 'Phoenix', 'Western', 'Pacific', 'PHX', '1996-04-13', '2014-06-27'),
                    (23, 'Arizona Coyotes', 'Glendale', 'Western', 'Pacific', 'ARI', '2014-06-27', '2024-04-18'),
                ]
                
                for franchise_id, name, city, conference, division, abbrev, effective_from, effective_to in historical_arizona_teams:
                    team_sql = """
                    INSERT INTO teams (franchise_id, name, city, league_id, conference, division, 
                                     abbreviation, effective_from, effective_to, is_current)
                    VALUES (:franchise_id, :name, :city, :league_id, :conference, :division, 
                           :abbreviation, :effective_from, :effective_to, FALSE)
                    ON CONFLICT (franchise_id, effective_from) DO NOTHING;
                    """
                    
                    conn.execute(text(team_sql), {
                        'franchise_id': franchise_id,
                        'name': name,
                        'city': city,
                        'league_id': nhl_league_id,
                        'conference': conference,
                        'division': division,
                        'abbreviation': abbrev,
                        'effective_from': effective_from,
                        'effective_to': effective_to
                    })
                
                conn.commit()
                
                duration = self.perf_logger.end_timer('initial_data_insertion')
                logger.info(f"✅ Inserted NHL league, {franchises_inserted} franchises and {teams_inserted} current teams")
                logger.info("✅ Historical team transitions configured (Arizona/Utah)")
                return True
                
        except Exception as e:
            self.perf_logger.end_timer('initial_data_insertion')
            logger.error(f"❌ Error inserting initial data: {e}")
            return False
    
    def detect_team_name_column(self, df: pd.DataFrame) -> str:
        """Enhanced team column detection (unchanged business logic)"""
        team_column_candidates = [
            'Team', '', ':', 'team', 'Team Name', 'Tm'
        ]
        
        for col_name in team_column_candidates:
            if col_name in df.columns:
                sample_values = df[col_name].dropna().astype(str).str.strip()
                sample_values = sample_values[sample_values != '']
                
                if len(sample_values) > 0:
                    first_value = sample_values.iloc[0]
                    if len(first_value) > 2 and not first_value.isdigit():
                        logger.info(f"🎯 Detected team name column: '{col_name}'")
                        logger.info(f"   Sample values: {list(sample_values.head(3))}")
                        return col_name
        
        # Fallback detection
        logger.warning("⚠️ Standard team name columns not found, attempting auto-detection...")
        
        for col_name in df.columns:
            if df[col_name].dtype == 'object':
                sample_values = df[col_name].dropna().astype(str).str.strip()
                sample_values = sample_values[sample_values != '']
                
                if len(sample_values) > 0:
                    first_value = sample_values.iloc[0]
                    nhl_indicators = ['Bruins', 'Rangers', 'Kings', 'Wings', 'Leafs', 'Flames', 'Stars', 'Wild']
                    if any(indicator in first_value for indicator in nhl_indicators):
                        logger.info(f"🎯 Auto-detected team name column: '{col_name}'")
                        logger.info(f"   Sample values: {list(sample_values.head(3))}")
                        return col_name
        
        # Final fallback
        string_columns = df.select_dtypes(include=['object']).columns.tolist()
        if string_columns:
            fallback_col = string_columns[0]
            logger.warning(f"⚠️ Using fallback column: '{fallback_col}'")
            return fallback_col
        
        raise ValueError("❌ Could not detect team name column in CSV")
    
    def get_team_id_for_date(self, team_name: str, game_date: str, conn) -> Optional[int]:
        """Enhanced team ID resolution (business logic unchanged)"""
        normalized_name = self.normalize_team_name(team_name)
        
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date).date()
        
        # Try current team first
        current_team_sql = """
        SELECT id FROM teams 
        WHERE name = :team_name AND is_current = TRUE
        """
        result = conn.execute(text(current_team_sql), {'team_name': normalized_name})
        current_team = result.fetchone()
        
        if current_team:
            return current_team[0]
        
        # Try historical team for specific date
        historical_team_sql = """
        SELECT id FROM teams 
        WHERE name = :team_name 
          AND effective_from <= :game_date
          AND (effective_to IS NULL OR effective_to > :game_date)
        """
        result = conn.execute(text(historical_team_sql), {
            'team_name': normalized_name,
            'game_date': game_date
        })
        historical_team = result.fetchone()
        
        if historical_team:
            return historical_team[0]
        
        # Try franchise mapping
        franchise_mapping = {
            'Arizona Coyotes': 'Utah Mammoth',
            'Utah Hockey Club': 'Utah Mammoth',
            'Phoenix Coyotes': 'Utah Mammoth',
            'Winnipeg Jets': self._resolve_jets_name(game_date),
        }
        
        if normalized_name in franchise_mapping:
            mapped_name = franchise_mapping[normalized_name]
            return self.get_team_id_for_date(mapped_name, game_date, conn)
        
        logger.debug(f"Team not found: {team_name} (normalized: {normalized_name}) for date {game_date}")
        return None
    
    def _resolve_jets_name(self, game_date) -> str:
        """Resolve Jets franchise (business logic unchanged)"""
        cutoff_date = pd.to_datetime('2011-05-31').date()
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date).date()
        
        if game_date <= cutoff_date:
            return 'Utah Mammoth'
        else:
            return 'Winnipeg Jets'
    
    def normalize_team_name(self, team_name: str) -> str:
        """Enhanced team name normalization"""
        team_name = team_name.strip()
        
        basic_mappings = {
            'Utah Hockey Club': 'Utah Mammoth',
            'Arizona Coyotes': 'Utah Mammoth',
        }
        
        return basic_mappings.get(team_name, team_name)
    
    def find_latest_data_files(self, base_name: str, limit: int = 1) -> List[Path]:
        """Enhanced file discovery s PATHS integration"""
        
        try:
            # === MIGRATION: Use PATHS for directory determination ===
            if base_name.startswith('nhl_odds'):
                search_dir = PATHS.odds_data
            else:
                search_dir = PATHS.raw_data
            
            files = list(search_dir.glob(f"{base_name}_*.csv"))
            
            if not files:
                logger.warning(f"No files found matching pattern: {base_name}_*.csv in {search_dir}")
                return []
            
            # Sort by timestamp
            files.sort(key=self.extract_sort_key, reverse=True)
            
            # Enhanced logging
            logger.info(f"📄 Found {len(files)} {base_name} files in {search_dir}:")
            for i, file in enumerate(files[:limit]):
                timestamp = self.extract_timestamp_from_filename(file)
                age_indicator = "🆕" if i == 0 else "📄"
                logger.info(f"  {age_indicator} {file.name} (scraped: {timestamp})")
            
            selected_files = files[:limit]
            
            if selected_files:
                latest_file = selected_files[0]
                timestamp = self.extract_timestamp_from_filename(latest_file)
                logger.info(f"✅ Using latest {base_name} file: {latest_file.name} (scraped: {timestamp})")
            
            return selected_files
            
        except Exception as e:
            logger.error(f"Error finding files for {base_name}: {e}")
            return []
    
    def verify_file_integrity(self, file_path: Path) -> bool:
        """Enhanced file verification s safe file handlers"""
        
        try:
            # === MIGRATION: Use enhanced safe file handlers ===
            df = read_csv(file_path, nrows=5)
            
            if df.empty:
                logger.warning(f"File appears to be empty: {file_path}")
                return False
            
            # Count total rows safely
            with open(file_path, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for line in f) - 1
            
            logger.info(f"📊 File validation: {file_path.name}")
            logger.info(f"  Rows: {total_rows:,}")
            logger.info(f"  Columns: {len(df.columns)}")
            logger.info(f"  Columns: {', '.join(df.columns[:8])}{'...' if len(df.columns) > 8 else ''}")
            
            return True
            
        except Exception as e:
            logger.error(f"File integrity check failed for {file_path}: {e}")
            return False
    
    def import_scraped_data(self) -> bool:
        """Enhanced data import s performance monitoring"""
        
        self.perf_logger.start_timer('full_data_import')
        
        try:
            # Import games
            self.perf_logger.start_timer('games_import')
            games_files = self.find_latest_data_files('nhl_games')
            for file_path in games_files:
                logger.info(f"Importing games from {file_path.name}...")
                self.import_games_data(file_path)
            self.perf_logger.end_timer('games_import')
            
            if not games_files:
                logger.warning(f"No games files found in {self.data_paths['nhl_data']}")
            
            # Import team stats (comprehensive)
            self.perf_logger.start_timer('team_stats_import')
            stats_files = self.find_latest_data_files('nhl_team_stats')
            teams_with_stats = set()
            
            for file_path in stats_files:
                logger.info(f"Importing comprehensive team stats from {file_path.name}...")
                imported_teams = self.import_team_stats_data(file_path, return_teams=True)
                teams_with_stats.update(imported_teams)
            self.perf_logger.end_timer('team_stats_import')
            
            # Import standings as fallback
            self.perf_logger.start_timer('standings_fallback_import')
            standings_files = self.find_latest_data_files('nhl_standings')
            for file_path in standings_files:
                logger.info(f"Importing standings as fallback from {file_path.name}...")
                self.import_standings_as_fallback(file_path, exclude_teams=teams_with_stats)
            self.perf_logger.end_timer('standings_fallback_import')
            
            # Report strategy results
            self.report_import_strategy_results()
            
            # Import odds
            self.perf_logger.start_timer('odds_import')
            odds_files = list(self.data_paths['odds_data'].glob('nhl_odds_*.csv'))
            
            if odds_files:
                odds_files.sort(key=self.extract_sort_key, reverse=True)
                logger.info(f"📊 Found {len(odds_files)} odds files in {self.data_paths['odds_data']}")
                
                for file_path in odds_files:
                    timestamp = self.extract_timestamp_from_filename(file_path)
                    logger.info(f"Importing odds from {file_path.name} (scraped: {timestamp})...")
                    self.import_odds_data(file_path)
            else:
                logger.warning(f"No odds files found in {self.data_paths['odds_data']}")
            self.perf_logger.end_timer('odds_import')
            
            total_duration = self.perf_logger.end_timer('full_data_import')
            logger.info(f"✅ Complete data import finished in {total_duration:.2f}s")
            
            return True
            
        except Exception as e:
            self.perf_logger.end_timer('full_data_import')
            logger.error(f"❌ Error importing scraped data: {e}")
            return False
    
    def import_games_data(self, file_path: Path):
        """Enhanced games import (core logic unchanged)"""
        
        if not self.verify_file_integrity(file_path):
            logger.error(f"Skipping import of {file_path} due to integrity issues")
            return
        
        try:
            # === MIGRATION: Use safe file handlers ===
            df = read_csv(file_path)
            games_imported = 0
            games_skipped = 0
            
            logger.info(f"🎯 Starting import of {len(df)} games from {file_path.name}")
            
            with self.engine.connect() as conn:
                for _, row in df.iterrows():
                    try:
                        game_date = pd.to_datetime(row['date']).date()
                        
                        home_team_id = self.get_team_id_for_date(row['home_team'], game_date, conn)
                        away_team_id = self.get_team_id_for_date(row['visitor_team'], game_date, conn)
                        
                        if not home_team_id or not away_team_id:
                            logger.debug(f"Teams not found: {row['home_team']}, {row['visitor_team']} for {game_date}")
                            games_skipped += 1
                            continue
                        
                        datetime_et = pd.to_datetime(row.get('datetime')) if pd.notna(row.get('datetime')) else None
                        
                        game_sql = """
                        INSERT INTO games (date, datetime_et, season, league_id, home_team_id, away_team_id,
                                         venue_id, home_score, away_score, overtime_shootout, status, data_source)
                        VALUES (:date, :datetime_et, :season, 1, :home_team_id, :away_team_id,
                               NULL, :home_score, :away_score, :overtime_shootout, :status, 'hockey-reference')
                        ON CONFLICT (date, home_team_id, away_team_id) DO UPDATE SET
                            datetime_et = EXCLUDED.datetime_et,
                            home_score = EXCLUDED.home_score,
                            away_score = EXCLUDED.away_score,
                            status = EXCLUDED.status
                        RETURNING id;
                        """
                        
                        result = conn.execute(text(game_sql), {
                            'date': game_date,
                            'datetime_et': datetime_et,
                            'season': int(row['season']),
                            'home_team_id': home_team_id,
                            'away_team_id': away_team_id,
                            'home_score': int(row['home_score']) if pd.notna(row['home_score']) else None,
                            'away_score': int(row['visitor_score']) if pd.notna(row['visitor_score']) else None,
                            'overtime_shootout': row.get('overtime_shootout'),
                            'status': row.get('status', 'completed')
                        })
                        
                        game_row = result.fetchone()
                        if game_row:
                            game_id = game_row[0]
                            
                            # Insert boxscore URL if available
                            if pd.notna(row.get('boxscore_url')):
                                url_sql = """
                                INSERT INTO game_urls (game_id, url_type, url, source)
                                VALUES (:game_id, 'boxscore', :url, 'hockey-reference')
                                ON CONFLICT (game_id, url_type, source) DO NOTHING;
                                """
                                conn.execute(text(url_sql), {
                                    'game_id': game_id,
                                    'url': row['boxscore_url']
                                })
                        
                        games_imported += 1
                        
                        if games_imported % 100 == 0:
                            logger.info(f"  📈 Progress: {games_imported} games imported...")
                        
                    except Exception as e:
                        logger.error(f"Error importing game row: {e}")
                        games_skipped += 1
                        continue
                
                conn.commit()
                logger.info(f"✅ Games import completed:")
                logger.info(f"  📊 Imported: {games_imported} games")
                logger.info(f"  ⚠️  Skipped: {games_skipped} games")
                logger.info(f"  📁 Source: {file_path.name}")
                
        except Exception as e:
            logger.error(f"❌ Error importing games data: {e}")
    
    def import_team_stats_data(self, file_path: Path, return_teams: bool = False):
        """Enhanced team stats import (core logic preserved)"""
        
        if not self.verify_file_integrity(file_path):
            logger.error(f"Skipping import of {file_path} due to integrity issues")
            return set() if return_teams else None
        
        try:
            # === MIGRATION: Use safe file handlers ===
            df = read_csv(file_path)
            
            team_column = self.detect_team_name_column(df)
            
            stats_imported = 0
            stats_skipped = 0
            processed_teams = set()
            
            logger.info(f"📈 Starting import of {len(df)} comprehensive team stats from {file_path.name}")
            logger.info(f"🎯 Using team name column: '{team_column}'")
            
            with self.engine.connect() as conn:
                for _, row in df.iterrows():
                    try:
                        team_name = str(row.get(team_column, '')).replace('*', '').strip()
                        
                        if not team_name:
                            logger.debug(f"Empty team name in row, skipping")
                            stats_skipped += 1
                            continue
                        
                        season = int(row['season'])
                        season_start_date = pd.to_datetime(f'{season-1}-10-01').date()
                        
                        team_id = self.get_team_id_for_date(team_name, season_start_date, conn)
                        if not team_id:
                            logger.debug(f"Team not found: {team_name} for season {season}")
                            stats_skipped += 1
                            continue
                        
                        # Comprehensive team stats SQL (unchanged)
                        stats_sql = """
                        INSERT INTO team_stats (
                            team_id, season, games_played, wins, losses, overtime_losses, points,
                            points_percentage, goals_for, goals_against, shootout_wins, shootout_losses,
                            srs, sos, goals_for_per_game, goals_against_per_game,
                            power_play_goals, power_play_opportunities, power_play_percentage,
                            penalty_kill_percentage, short_handed_goals, short_handed_goals_allowed,
                            shots, shot_percentage, shots_against, save_percentage, shutouts,
                            penalties_per_game, opponent_penalties_per_game, average_age
                        )
                        VALUES (
                            :team_id, :season, :gp, :w, :l, :ol, :pts, :pts_pct, :gf, :ga, :sow, :sol,
                            :srs, :sos, :gf_per_g, :ga_per_g, :pp, :ppo, :pp_pct, :pk_pct, :sh, :sha,
                            :shots, :shot_pct, :sa, :sv_pct, :so, :pim_per_g, :opim_per_g, :avg_age
                        )
                        ON CONFLICT (team_id, season) DO UPDATE SET
                            games_played = EXCLUDED.games_played,
                            wins = EXCLUDED.wins,
                            losses = EXCLUDED.losses,
                            overtime_losses = EXCLUDED.overtime_losses,
                            points = EXCLUDED.points,
                            points_percentage = EXCLUDED.points_percentage,
                            goals_for = EXCLUDED.goals_for,
                            goals_against = EXCLUDED.goals_against,
                            power_play_percentage = EXCLUDED.power_play_percentage,
                            penalty_kill_percentage = EXCLUDED.penalty_kill_percentage,
                            shot_percentage = EXCLUDED.shot_percentage,
                            save_percentage = EXCLUDED.save_percentage,
                            shutouts = EXCLUDED.shutouts,
                            average_age = EXCLUDED.average_age;
                        """
                        
                        conn.execute(text(stats_sql), {
                            'team_id': team_id,
                            'season': season,
                            'gp': int(row['GP']) if pd.notna(row['GP']) else None,
                            'w': int(row['W']) if pd.notna(row['W']) else None,
                            'l': int(row['L']) if pd.notna(row['L']) else None,
                            'ol': int(row['OL']) if pd.notna(row['OL']) else None,
                            'pts': int(row['PTS']) if pd.notna(row['PTS']) else None,
                            'pts_pct': float(row['PTS%']) if pd.notna(row['PTS%']) else None,
                            'gf': int(row['GF']) if pd.notna(row['GF']) else None,
                            'ga': int(row['GA']) if pd.notna(row['GA']) else None,
                            'sow': int(row['SOW']) if pd.notna(row['SOW']) else None,
                            'sol': int(row['SOL']) if pd.notna(row['SOL']) else None,
                            'srs': float(row['SRS']) if pd.notna(row['SRS']) else None,
                            'sos': float(row['SOS']) if pd.notna(row['SOS']) else None,
                            'gf_per_g': float(row['GF/G']) if pd.notna(row['GF/G']) else None,
                            'ga_per_g': float(row['GA/G']) if pd.notna(row['GA/G']) else None,
                            'pp': int(row['PP']) if pd.notna(row['PP']) else None,
                            'ppo': int(row['PPO']) if pd.notna(row['PPO']) else None,
                            'pp_pct': float(row['PP%']) if pd.notna(row['PP%']) else None,
                            'pk_pct': float(row['PK%']) if pd.notna(row['PK%']) else None,
                            'sh': int(row['SH']) if pd.notna(row['SH']) else None,
                            'sha': int(row['SHA']) if pd.notna(row['SHA']) else None,
                            'shots': int(row['S']) if pd.notna(row['S']) else None,
                            'shot_pct': float(row['S%']) if pd.notna(row['S%']) else None,
                            'sa': int(row['SA']) if pd.notna(row['SA']) else None,
                            'sv_pct': float(row['SV%']) if pd.notna(row['SV%']) else None,
                            'so': int(row['SO']) if pd.notna(row['SO']) else None,
                            'pim_per_g': float(row['PIM/G']) if pd.notna(row['PIM/G']) else None,
                            'opim_per_g': float(row['oPIM/G']) if pd.notna(row['oPIM/G']) else None,
                            'avg_age': float(row['AvAge']) if pd.notna(row['AvAge']) else None
                        })
                        
                        stats_imported += 1
                        processed_teams.add((team_id, season))
                        
                    except Exception as e:
                        logger.error(f"Error importing team stats row: {e}")
                        stats_skipped += 1
                        continue
                
                conn.commit()
                logger.info(f"✅ Comprehensive team stats import completed:")
                logger.info(f"  📊 Imported: {stats_imported} team stats (COMPREHENSIVE)")
                logger.info(f"  ⚠️  Skipped: {stats_skipped} team stats")
                logger.info(f"  📁 Source: {file_path.name}")
                
                if return_teams:
                    return processed_teams
                
        except Exception as e:
            logger.error(f"❌ Error importing team stats: {e}")
            return set() if return_teams else None
    
    def import_standings_as_fallback(self, file_path: Path, exclude_teams: Set = None):
        """Enhanced standings fallback import"""
        
        if exclude_teams is None:
            exclude_teams = set()
        
        try:
            # === MIGRATION: Use safe file handlers ===
            df = read_csv(file_path)
            
            team_column = self.detect_team_name_column(df)
            
            standings_imported = 0
            standings_skipped_existing = 0
            standings_skipped_missing = 0
            
            logger.info(f"📄 Processing {len(df)} standings as FALLBACK data from {file_path.name}")
            logger.info(f"🎯 Using team name column: '{team_column}'")
            logger.info(f"🚫 Excluding {len(exclude_teams)} teams already covered by comprehensive stats")
            
            with self.engine.connect() as conn:
                for _, row in df.iterrows():
                    try:
                        team_name = str(row.get(team_column, '')).replace('*', '').strip()
                        
                        if not team_name:
                            standings_skipped_missing += 1
                            continue
                        
                        season = int(row['season'])
                        season_start_date = pd.to_datetime(f'{season-1}-10-01').date()
                        
                        team_id = self.get_team_id_for_date(team_name, season_start_date, conn)
                        if not team_id:
                            standings_skipped_missing += 1
                            continue
                        
                        # Skip if already covered
                        if (team_id, season) in exclude_teams:
                            standings_skipped_existing += 1
                            logger.debug(f"Skipping {team_name} {season} - already covered by team_stats")
                            continue
                        
                        # Double-check for existing stats
                        existing_stats = conn.execute(text("""
                            SELECT id FROM team_stats WHERE team_id = :team_id AND season = :season
                        """), {'team_id': team_id, 'season': season}).fetchone()
                        
                        if existing_stats:
                            standings_skipped_existing += 1
                            logger.debug(f"Skipping {team_name} {season} - team_stats record exists")
                            continue
                        
                        # Create basic team_stats record (fallback)
                        fallback_sql = """
                        INSERT INTO team_stats (
                            team_id, season, games_played, wins, losses, overtime_losses, points,
                            points_percentage, goals_for, goals_against, srs, sos
                        )
                        VALUES (
                            :team_id, :season, :gp, :w, :l, :ol, :pts, :pts_pct, :gf, :ga, :srs, :sos
                        );
                        """
                        
                        conn.execute(text(fallback_sql), {
                            'team_id': team_id,
                            'season': season,
                            'gp': int(row['GP']) if pd.notna(row['GP']) else None,
                            'w': int(row['W']) if pd.notna(row['W']) else None,
                            'l': int(row['L']) if pd.notna(row['L']) else None,
                            'ol': int(row['OL']) if pd.notna(row['OL']) else None,
                            'pts': int(row['PTS']) if pd.notna(row['PTS']) else None,
                            'pts_pct': float(row['PTS%']) if pd.notna(row['PTS%']) else None,
                            'gf': int(row['GF']) if pd.notna(row['GF']) else None,
                            'ga': int(row['GA']) if pd.notna(row['GA']) else None,
                            'srs': float(row['SRS']) if pd.notna(row['SRS']) else None,
                            'sos': float(row['SOS']) if pd.notna(row['SOS']) else None
                        })
                        
                        standings_imported += 1
                        
                    except Exception as e:
                        logger.error(f"Error importing standings fallback row: {e}")
                        continue
                
                conn.commit()
                logger.info(f"✅ Standings fallback import completed:")
                logger.info(f"  📊 Imported: {standings_imported} basic team stats (FALLBACK)")
                logger.info(f"  ⭐  Skipped (already covered): {standings_skipped_existing}")
                logger.info(f"  ⚠️  Skipped (missing teams): {standings_skipped_missing}")
                logger.info(f"  📁 Source: {file_path.name}")
                
        except Exception as e:
            logger.error(f"❌ Error importing standings as fallback: {e}")
    
    def report_import_strategy_results(self):
        """Enhanced import strategy reporting"""
        
        try:
            with self.engine.connect() as conn:
                comprehensive_count = conn.execute(text("""
                    SELECT COUNT(*) FROM team_stats 
                    WHERE power_play_percentage IS NOT NULL
                """)).fetchone()[0]
                
                basic_count = conn.execute(text("""
                    SELECT COUNT(*) FROM team_stats 
                    WHERE power_play_percentage IS NULL
                """)).fetchone()[0]
                
                total_count = comprehensive_count + basic_count
                
                logger.info(f"\n📋 ENHANCED IMPORT STRATEGY RESULTS:")
                logger.info(f"  👑 Total team-season records: {total_count}")
                logger.info(f"  📈 Comprehensive stats (from team_stats): {comprehensive_count}")
                logger.info(f"  📊 Basic stats (from standings fallback): {basic_count}")
                
                if total_count > 0:
                    comprehensive_pct = (comprehensive_count / total_count) * 100
                    logger.info(f"  ✅ Data completeness: {comprehensive_pct:.1f}% comprehensive")
                
                if basic_count > 0:
                    basic_teams = conn.execute(text("""
                        SELECT t.name, ts.season 
                        FROM team_stats ts
                        JOIN teams t ON ts.team_id = t.id
                        WHERE ts.power_play_percentage IS NULL
                        ORDER BY ts.season, t.name
                        LIMIT 10
                    """)).fetchall()
                    
                    logger.info(f"  📝 Teams with basic data (sample):")
                    for team_name, season in basic_teams:
                        logger.info(f"    {team_name} ({season})")
                    
                    if basic_count > 10:
                        logger.info(f"    ... and {basic_count - 10} more")
                
        except Exception as e:
            logger.error(f"Error generating import strategy report: {e}")
    
    def import_odds_data(self, file_path: Path):
        """Enhanced odds import s timezone conversion"""
        
        try:
            # === MIGRATION: Use safe file handlers ===
            df = read_csv(file_path)
            odds_imported = 0
            odds_skipped_teams = 0
            odds_skipped_games = 0
            
            logger.info(f"💰 Starting odds import from {file_path.name}")
            logger.info(f"📊 Processing {len(df)} odds records...")
            
            with self.engine.connect() as conn:
                for index, row in df.iterrows():
                    try:
                        match_datetime_str = row['match_datetime']
                        
                        # Parse datetime from CSV (assumes CET/CEST timezone)
                        match_datetime_cet = pd.to_datetime(match_datetime_str)
                        
                        # Convert from Central European Time to Eastern Time
                        match_datetime_et = self.convert_cet_to_et(match_datetime_cet)
                        
                        game_date = match_datetime_et.date()
                        
                        home_team_id = self.get_team_id_for_date(row['home_team'], game_date, conn)
                        away_team_id = self.get_team_id_for_date(row['away_team'], game_date, conn)
                        
                        if not home_team_id or not away_team_id:
                            logger.debug(f"Teams not found: {row['home_team']} vs {row['away_team']} for {game_date}")
                            odds_skipped_teams += 1
                            continue
                        
                        # Find matching game with wider tolerance
                        game_id = self.find_matching_game(
                            conn, home_team_id, away_team_id, match_datetime_et
                        )
                        
                        if not game_id:
                            logger.debug(f"No matching game found for {row['home_team']} vs {row['away_team']} at {match_datetime_et}")
                            odds_skipped_games += 1
                            continue
                        
                        # Insert odds with timezone-corrected datetimes
                        self.insert_odds_record(conn, row, game_id, match_datetime_et)
                        
                        # Insert betting URL
                        if pd.notna(row.get('source_url')):
                            self.insert_betting_url(conn, game_id, row['source_url'])
                        
                        odds_imported += 1
                        
                        if odds_imported % 50 == 0:
                            logger.info(f"  📈 Progress: {odds_imported} odds imported...")
                        
                    except Exception as e:
                        logger.error(f"Error importing odds row {index}: {e}")
                        logger.debug(f"Problematic row data: {dict(row)}")
                        continue
                
                conn.commit()
                
                total_processed = odds_imported + odds_skipped_teams + odds_skipped_games
                logger.info(f"✅ Odds import completed:")
                logger.info(f"  📊 Processed: {total_processed} records")
                logger.info(f"  💰 Imported: {odds_imported} odds")
                logger.info(f"  ⚠️  Skipped (teams not found): {odds_skipped_teams}")
                logger.info(f"  ⚠️  Skipped (games not matched): {odds_skipped_games}")
                logger.info(f"  📁 Source: {file_path.name}")
                logger.info(f"  🌍 Timezone: CET/CEST → Eastern Time conversion applied")
                
                if odds_skipped_games > 0:
                    self.log_sample_unmatched_games(conn, df.head(10))
                
        except Exception as e:
            logger.error(f"❌ Error importing odds data: {e}")
    
    def convert_cet_to_et(self, cet_datetime):
        """Convert Central European Time to Eastern Time"""
        try:
            import pytz
            
            cet_tz = pytz.timezone('Europe/Prague')
            et_tz = pytz.timezone('US/Eastern')
            
            if cet_datetime.tzinfo is None:
                cet_datetime = cet_tz.localize(cet_datetime)
            
            et_datetime = cet_datetime.astimezone(et_tz)
            
            return et_datetime.replace(tzinfo=None)
            
        except Exception as e:
            logger.warning(f"Timezone conversion failed: {e}, using offset approximation")
            return cet_datetime - pd.Timedelta(hours=6)
    
    def find_matching_game(self, conn, home_team_id: int, away_team_id: int, match_datetime_et):
        """Find matching game with enhanced tolerance strategies"""
        
        # Strategy 1: Exact datetime match (±1 hour)
        game_sql_exact = """
        SELECT id, datetime_et, 
               ABS(EXTRACT(EPOCH FROM (datetime_et - :match_datetime))) as time_diff_seconds
        FROM games 
        WHERE home_team_id = :home_team_id 
          AND away_team_id = :away_team_id
          AND ABS(EXTRACT(EPOCH FROM (datetime_et - :match_datetime))) < 3600
        ORDER BY time_diff_seconds
        LIMIT 1;
        """
        
        result = conn.execute(text(game_sql_exact), {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'match_datetime': match_datetime_et
        })
        
        game_row = result.fetchone()
        if game_row:
            logger.debug(f"Found exact match: game_id={game_row[0]}, time_diff={game_row[2]:.0f}s")
            return game_row[0]
        
        # Strategy 2: Same date match (±12 hours) 
        game_sql_date = """
        SELECT id, datetime_et,
               ABS(EXTRACT(EPOCH FROM (datetime_et - :match_datetime))) as time_diff_seconds
        FROM games 
        WHERE home_team_id = :home_team_id 
          AND away_team_id = :away_team_id
          AND ABS(EXTRACT(EPOCH FROM (datetime_et - :match_datetime))) < 43200
        ORDER BY time_diff_seconds
        LIMIT 1;
        """
        
        result = conn.execute(text(game_sql_date), {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'match_datetime': match_datetime_et
        })
        
        game_row = result.fetchone()
        if game_row:
            time_diff_hours = game_row[2] / 3600
            logger.debug(f"Found date match: game_id={game_row[0]}, time_diff={time_diff_hours:.1f}h")
            return game_row[0]
        
        # Strategy 3: Date-only match (ignore time completely)
        game_date = match_datetime_et.date()
        game_sql_dateonly = """
        SELECT id, datetime_et
        FROM games 
        WHERE home_team_id = :home_team_id 
          AND away_team_id = :away_team_id
          AND date = :game_date
        LIMIT 1;
        """
        
        result = conn.execute(text(game_sql_dateonly), {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'game_date': game_date
        })
        
        game_row = result.fetchone()
        if game_row:
            logger.debug(f"Found date-only match: game_id={game_row[0]}")
            return game_row[0]
        
        return None
    
    def insert_odds_record(self, conn, row, game_id: int, match_datetime_et):
        """Insert odds record with proper timezone processing"""
        
        odds_sql = """
        INSERT INTO odds (game_id, bookmaker, market_type, home_odd, away_odd,
                        home_opening_odd, away_opening_odd, home_opening_datetime,
                        away_opening_datetime, data_source)
        VALUES (:game_id, :bookmaker, :market_type, :home_odd, :away_odd,
               :home_opening_odd, :away_opening_odd, :home_opening_datetime,
               :away_opening_datetime, 'betexplorer')
        ON CONFLICT (game_id, bookmaker, market_type) DO UPDATE SET
            home_odd = EXCLUDED.home_odd,
            away_odd = EXCLUDED.away_odd,
            home_opening_odd = EXCLUDED.home_opening_odd,
            away_opening_odd = EXCLUDED.away_opening_odd;
        """
        
        # Convert opening datetimes from CET to ET if they exist
        home_opening_datetime = None
        away_opening_datetime = None
        
        if pd.notna(row.get('odds_home_opening_datetime')):
            home_opening_cet = pd.to_datetime(row['odds_home_opening_datetime'])
            home_opening_datetime = self.convert_cet_to_et(home_opening_cet)
        
        if pd.notna(row.get('odds_away_opening_datetime')):
            away_opening_cet = pd.to_datetime(row['odds_away_opening_datetime'])
            away_opening_datetime = self.convert_cet_to_et(away_opening_cet)
        
        conn.execute(text(odds_sql), {
            'game_id': game_id,
            'bookmaker': row['bookmaker'],
            'market_type': row['market_type'],
            'home_odd': float(row['odds_home_odd']) if pd.notna(row['odds_home_odd']) else None,
            'away_odd': float(row['odds_away_odd']) if pd.notna(row['odds_away_odd']) else None,
            'home_opening_odd': float(row['odds_home_opening_odd']) if pd.notna(row['odds_home_opening_odd']) else None,
            'away_opening_odd': float(row['odds_away_opening_odd']) if pd.notna(row['odds_away_opening_odd']) else None,
            'home_opening_datetime': home_opening_datetime,
            'away_opening_datetime': away_opening_datetime
        })
    
    def insert_betting_url(self, conn, game_id: int, source_url: str):
        """Insert betting URL for game"""
        
        url_sql = """
        INSERT INTO game_urls (game_id, url_type, url, source)
        VALUES (:game_id, 'betting', :url, 'betexplorer')
        ON CONFLICT (game_id, url_type, source) DO NOTHING;
        """
        
        conn.execute(text(url_sql), {
            'game_id': game_id,
            'url': source_url
        })
    
    def log_sample_unmatched_games(self, conn, sample_df):
        """Log sample unmatched games for debugging"""
        
        logger.info("🔍 Sample unmatched games (for debugging):")
        
        for _, row in sample_df.iterrows():
            try:
                match_datetime_cet = pd.to_datetime(row['match_datetime'])
                match_datetime_et = self.convert_cet_to_et(match_datetime_cet)
                
                games_sql = """
                SELECT date, datetime_et, home_score, away_score
                FROM games g
                JOIN teams ht ON g.home_team_id = ht.id 
                JOIN teams at ON g.away_team_id = at.id
                WHERE ht.name LIKE :home_team_pattern
                  AND at.name LIKE :away_team_pattern
                  AND ABS(EXTRACT(EPOCH FROM (g.datetime_et - :match_datetime))) < 86400
                ORDER BY ABS(EXTRACT(EPOCH FROM (g.datetime_et - :match_datetime)))
                LIMIT 3;
                """
                
                result = conn.execute(text(games_sql), {
                    'home_team_pattern': f"%{row['home_team'][:10]}%",
                    'away_team_pattern': f"%{row['away_team'][:10]}%", 
                    'match_datetime': match_datetime_et
                })
                
                games = result.fetchall()
                
                logger.info(f"  🔍 {row['home_team']} vs {row['away_team']}")
                logger.info(f"     CET: {match_datetime_cet} → ET: {match_datetime_et}")
                
                if games:
                    logger.info(f"     Similar games found:")
                    for game in games:
                        logger.info(f"       {game[0]} {game[1]} (Score: {game[2]}-{game[3]})")
                else:
                    logger.info(f"     No similar games found in database")
                    
            except Exception as e:
                logger.debug(f"Error in debugging sample: {e}")
                continue
    
    def get_data_summary(self):
        """Enhanced data summary with performance tracking"""
        
        self.perf_logger.start_timer('data_summary_generation')
        
        try:
            with self.engine.connect() as conn:
                # Games summary
                games_result = conn.execute(text("""
                    SELECT season, COUNT(*) as total_games, 
                           COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_games,
                           MIN(date) as first_game, MAX(date) as last_game
                    FROM games 
                    GROUP BY season 
                    ORDER BY season
                """))
                
                logger.info("\n📊 ENHANCED GAMES SUMMARY:")
                for row in games_result:
                    logger.info(f"  Season {row[0]}: {row[1]} total games, {row[2]} completed ({row[3]} to {row[4]})")
                
                # Venue assignment summary
                games_with_venues = conn.execute(text("""
                    SELECT COUNT(*) FROM games WHERE venue_id IS NOT NULL
                """)).fetchone()[0]
                total_games = conn.execute(text("SELECT COUNT(*) FROM games")).fetchone()[0]
                
                logger.info(f"\n🏟️ VENUE ASSIGNMENT:")
                logger.info(f"  Games with venue assigned: {games_with_venues}/{total_games}")
                logger.info(f"  Venue assignment: To be completed in future")
                
                # Enhanced franchises and teams summary
                franchise_result = conn.execute(text("""
                    SELECT COUNT(*) as total_franchises,
                           COUNT(CASE WHEN is_active = TRUE THEN 1 END) as active_franchises
                    FROM franchises
                """))
                franchise_data = franchise_result.fetchone()
                
                teams_result = conn.execute(text("""
                    SELECT COUNT(*) as total_teams,
                           COUNT(CASE WHEN is_current = TRUE THEN 1 END) as current_teams,
                           COUNT(CASE WHEN is_current = FALSE THEN 1 END) as historical_teams
                    FROM teams
                """))
                teams_data = teams_result.fetchone()
                
                logger.info(f"\n🏒 ENHANCED FRANCHISES & TEAMS:")
                logger.info(f"  {franchise_data[0]} total franchises ({franchise_data[1]} active)")
                logger.info(f"  {teams_data[0]} total team identities ({teams_data[1]} current, {teams_data[2]} historical)")
                
                # Historical changes summary
                history_result = conn.execute(text("""
                    SELECT change_type, COUNT(*) as count
                    FROM team_history 
                    GROUP BY change_type
                    ORDER BY count DESC
                """))
                
                logger.info("\n📜 TEAM HISTORY:")
                for row in history_result:
                    logger.info(f"  {row[0]}: {row[1]} changes")
                
                # Key franchise transitions (Utah/Arizona)
                utah_history = conn.execute(text("""
                    SELECT 
                        t_from.name as from_name,
                        t_to.name as to_name,
                        th.change_date,
                        th.change_type
                    FROM team_history th
                    JOIN teams t_from ON th.from_team_id = t_from.id
                    JOIN teams t_to ON th.to_team_id = t_to.id
                    WHERE th.franchise_id = 23  -- Utah/Arizona franchise
                    ORDER BY th.change_date
                """))
                
                logger.info("\n🦣 UTAH MAMMOTH FRANCHISE HISTORY:")
                for row in utah_history:
                    logger.info(f"  {row[2]}: {row[0]} → {row[1]} ({row[3]})")
                
                # Enhanced odds summary
                odds_result = conn.execute(text("""
                    SELECT market_type, COUNT(*) as records, COUNT(DISTINCT bookmaker) as bookmakers
                    FROM odds 
                    GROUP BY market_type
                """))
                
                logger.info("\n💰 ENHANCED ODDS SUMMARY:")
                for row in odds_result:
                    logger.info(f"  {row[0]}: {row[1]} records from {row[2]} bookmakers")
                
                # URLs summary
                urls_result = conn.execute(text("""
                    SELECT url_type, source, COUNT(*) as urls
                    FROM game_urls
                    GROUP BY url_type, source
                    ORDER BY url_type, source
                """))
                
                logger.info("\n🔗 ENHANCED URLS SUMMARY:")
                for row in urls_result:
                    logger.info(f"  {row[0]} ({row[1]}): {row[2]} URLs")
                
                # Enhanced team stats summary
                stats_result = conn.execute(text("""
                    SELECT season, COUNT(*) as teams_with_stats
                    FROM team_stats 
                    GROUP BY season 
                    ORDER BY season
                """))
                
                logger.info("\n📈 ENHANCED TEAM STATS SUMMARY:")
                for row in stats_result:
                    logger.info(f"  Season {row[0]}: {row[1]} teams with stats")
            
            duration = self.perf_logger.end_timer('data_summary_generation')
            logger.info(f"\n✅ Enhanced data summary completed in {duration:.3f}s")
                    
        except Exception as e:
            self.perf_logger.end_timer('data_summary_generation')
            logger.error(f"❌ Error generating enhanced summary: {e}")


def enhanced_main():
    """Enhanced main function s comprehensive error handling a performance monitoring"""
    
    # Ensure directories exist
    PATHS.ensure_directories()
    
    logger.info("🏒 Starting NHL Database Setup with ENHANCED INFRASTRUCTURE...")
    logger.info("=" * 80)
    logger.info("ENHANCED FEATURES:")
    logger.info("  ✅ Per-component logging (database.log)")
    logger.info("  ✅ Safe file handling with encoding detection")
    logger.info("  ✅ Performance monitoring for all operations")
    logger.info("  ✅ PATHS integration for all file operations")
    logger.info("  ✅ Robust error handling with detailed logging")
    logger.info("=" * 80)
    
    logger.info(f"📂 Enhanced file locations:")
    logger.info(f"  NHL Data: {PATHS.raw_data}")
    logger.info(f"  Odds Data: {PATHS.odds_data}")
    logger.info(f"  Logs: {PATHS.logs}")
    
    # Performance logger for main function
    main_perf_logger = PerformanceLogger(logger)
    main_perf_logger.start_timer('total_setup')
    
    try:
        # Initialize enhanced database manager
        logger.info("\n🔧 Initializing Enhanced DatabaseManager...")
        main_perf_logger.start_timer('manager_initialization')
        db_manager = EnhancedDatabaseManager()
        main_perf_logger.end_timer('manager_initialization')
        
        # Create tables
        logger.info("\n🔧 Creating database tables with enhanced monitoring...")
        if not db_manager.create_tables():
            logger.error("❌ Failed to create tables. Please check permissions.")
            return
        
        # Insert initial data
        logger.info("\n🏒 Inserting initial leagues, franchises and teams...")
        if not db_manager.insert_initial_data():
            logger.error("❌ Failed to insert initial data.")
            return
        
        # Import scraped data with enhanced processing
        logger.info("\n📊 Importing scraped NHL data with ENHANCED hierarchical strategy...")  
        if not db_manager.import_scraped_data():
            logger.error("❌ Failed to import scraped data.")
            logger.info(f"💡 Make sure your data files are in the correct directories:")
            logger.info(f"   • NHL data: {PATHS.raw_data}")
            logger.info(f"   • Odds data: {PATHS.odds_data}")
            return
        
        # Generate enhanced summary
        logger.info("\n📋 Generating enhanced data summary...")
        db_manager.get_data_summary()
        
        total_duration = main_perf_logger.end_timer('total_setup')
        
        logger.info("\n" + "="*80)
        logger.info("🎉 ENHANCED Database setup completed successfully!")
        logger.info(f"⏱️  Total setup time: {total_duration:.2f}s")
        logger.info("\n📋 ENHANCED MIGRATION IMPROVEMENTS:")
        logger.info("  ✅ Per-component logging (logs/database.log)")
        logger.info("  ✅ Safe file handlers with automatic encoding detection")
        logger.info("  ✅ Performance monitoring for all major operations")
        logger.info("  ✅ PATHS integration replacing all hardcoded paths")
        logger.info("  ✅ Enhanced error handling with detailed context")
        logger.info("  ✅ Preserved all business logic and functionality")
        logger.info("  ✅ UTF-8 encoding consistency throughout")
        logger.info("  ✅ Robust file discovery and validation")
        logger.info("="*80)
        
    except Exception as e:
        main_perf_logger.end_timer('total_setup')
        logger.error(f"❌ Enhanced database setup failed: {e}")
        
        # Enhanced troubleshooting
        logger.info("\n💡 ENHANCED TROUBLESHOOTING:")
        logger.info("  1. Check database connection settings")
        logger.info(f"  2. Ensure data files are in: {PATHS.raw_data} and {PATHS.odds_data}")
        logger.info("  3. Verify file naming follows pattern: nhl_*_YYYYMMDD_HHMMSS.csv")
        logger.info("  4. Check database permissions and disk space")
        logger.info(f"  5. Review component-specific logs in: {PATHS.logs}/database.log")
        logger.info("  6. Verify encoding of CSV files (should be UTF-8 compatible)")
        
        raise


if __name__ == "__main__":
    enhanced_main()