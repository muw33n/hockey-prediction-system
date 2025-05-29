#!/usr/bin/env python3
"""
Database setup with better permission handling
"""

import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
import logging
from datetime import datetime
import glob
from typing import List, Dict, Optional

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/database.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations with better permission handling"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)
        
    def check_permissions(self):
        """Check if we have necessary permissions"""
        try:
            with self.engine.connect() as conn:
                # Try to create a test table
                conn.execute(text("CREATE TABLE IF NOT EXISTS test_permissions (id SERIAL PRIMARY KEY)"))
                conn.execute(text("DROP TABLE IF EXISTS test_permissions"))
                conn.commit()
                logger.info("✅ Database permissions OK")
                return True
        except Exception as e:
            logger.error(f"❌ Permission check failed: {e}")
            logger.error("💡 Please run the permission fix SQL commands as postgres user")
            return False
    
    def create_tables(self):
        """Create all necessary tables with error handling"""
        
        # Check permissions first
        if not self.check_permissions():
            logger.error("❌ Insufficient permissions. Please fix permissions first.")
            return False
        
        # SQL for creating tables (same as before but with better error handling)
        create_tables_sql = """
        -- Drop existing tables if they exist
        DROP TABLE IF EXISTS value_bets CASCADE;
        DROP TABLE IF EXISTS predictions CASCADE;
        DROP TABLE IF EXISTS odds CASCADE;
        DROP TABLE IF EXISTS goalie_stats CASCADE;
        DROP TABLE IF EXISTS player_stats CASCADE;
        DROP TABLE IF EXISTS team_stats CASCADE;
        DROP TABLE IF EXISTS games CASCADE;
        DROP TABLE IF EXISTS teams CASCADE;
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
        
        -- Teams table
        CREATE TABLE teams (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            city VARCHAR(100),
            league_id INTEGER REFERENCES leagues(id),
            conference VARCHAR(50),
            division VARCHAR(50),
            abbreviation VARCHAR(10),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Games table
        CREATE TABLE games (
            id SERIAL PRIMARY KEY,
            date DATE NOT NULL,
            season VARCHAR(10) NOT NULL,
            league_id INTEGER REFERENCES leagues(id),
            home_team_id INTEGER REFERENCES teams(id),
            away_team_id INTEGER REFERENCES teams(id),
            home_score INTEGER,
            away_score INTEGER,
            overtime_shootout VARCHAR(10),
            status VARCHAR(20) DEFAULT 'scheduled',
            stage VARCHAR(20) DEFAULT 'regular',
            scraped_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Team statistics table
        CREATE TABLE team_stats (
            id SERIAL PRIMARY KEY,
            team_id INTEGER REFERENCES teams(id),
            season VARCHAR(10) NOT NULL,
            games_played INTEGER,
            wins INTEGER,
            losses INTEGER,
            overtime_losses INTEGER,
            points INTEGER,
            goals_for INTEGER,
            goals_against INTEGER,
            goal_differential INTEGER,
            shots_for INTEGER,
            shots_against INTEGER,
            power_play_goals INTEGER,
            power_play_opportunities INTEGER,
            penalty_kill_percentage DECIMAL(5,2),
            face_off_win_percentage DECIMAL(5,2),
            scraped_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Player statistics table
        CREATE TABLE player_stats (
            id SERIAL PRIMARY KEY,
            player_name VARCHAR(100) NOT NULL,
            team_id INTEGER REFERENCES teams(id),
            season VARCHAR(10) NOT NULL,
            position VARCHAR(10),
            games_played INTEGER,
            goals INTEGER,
            assists INTEGER,
            points INTEGER,
            plus_minus INTEGER,
            penalty_minutes INTEGER,
            shots INTEGER,
            shooting_percentage DECIMAL(5,2),
            scraped_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Goalie statistics table
        CREATE TABLE goalie_stats (
            id SERIAL PRIMARY KEY,
            player_name VARCHAR(100) NOT NULL,
            team_id INTEGER REFERENCES teams(id),
            season VARCHAR(10) NOT NULL,
            games_played INTEGER,
            wins INTEGER,
            losses INTEGER,
            overtime_losses INTEGER,
            saves INTEGER,
            shots_against INTEGER,
            save_percentage DECIMAL(5,3),
            goals_against_average DECIMAL(4,2),
            shutouts INTEGER,
            scraped_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Odds table
        CREATE TABLE odds (
            id SERIAL PRIMARY KEY,
            game_id INTEGER REFERENCES games(id),
            bookmaker VARCHAR(50) NOT NULL,
            market_type VARCHAR(50) NOT NULL,
            selection VARCHAR(100) NOT NULL,
            odds_decimal DECIMAL(6,2) NOT NULL,
            odds_american INTEGER,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Predictions table
        CREATE TABLE predictions (
            id SERIAL PRIMARY KEY,
            game_id INTEGER REFERENCES games(id),
            model_name VARCHAR(50) NOT NULL,
            model_version VARCHAR(20),
            home_win_prob DECIMAL(4,3),
            away_win_prob DECIMAL(4,3),
            over_under_prob DECIMAL(4,3),
            predicted_home_score DECIMAL(3,1),
            predicted_away_score DECIMAL(3,1),
            confidence_score DECIMAL(4,3),
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Value bets table
        CREATE TABLE value_bets (
            id SERIAL PRIMARY KEY,
            game_id INTEGER REFERENCES games(id),
            bookmaker VARCHAR(50) NOT NULL,
            market_type VARCHAR(50) NOT NULL,
            selection VARCHAR(100) NOT NULL,
            our_probability DECIMAL(4,3) NOT NULL,
            bookmaker_probability DECIMAL(4,3) NOT NULL,
            bookmaker_odds DECIMAL(6,2) NOT NULL,
            expected_value DECIMAL(6,3) NOT NULL,
            kelly_fraction DECIMAL(6,4),
            recommended_stake DECIMAL(8,2),
            bet_placed BOOLEAN DEFAULT FALSE,
            result VARCHAR(20),
            profit_loss DECIMAL(8,2),
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes
        CREATE INDEX idx_games_date ON games(date);
        CREATE INDEX idx_games_season ON games(season);
        CREATE INDEX idx_games_teams ON games(home_team_id, away_team_id);
        CREATE INDEX idx_team_stats_season ON team_stats(season);
        CREATE INDEX idx_odds_game_id ON odds(game_id);
        CREATE INDEX idx_predictions_game_id ON predictions(game_id);
        CREATE INDEX idx_value_bets_game_id ON value_bets(game_id);
        CREATE INDEX idx_value_bets_expected_value ON value_bets(expected_value DESC);
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_tables_sql))
                conn.commit()
                logger.info("✅ Database tables created successfully!")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error creating tables: {e}")
            if "permission denied" in str(e).lower():
                logger.error("💡 Permission issue detected. Please run these commands as postgres user:")
                logger.error("   GRANT ALL ON SCHEMA public TO hockey_user;")
                logger.error("   GRANT ALL ON ALL TABLES IN SCHEMA public TO hockey_user;")
                logger.error("   GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO hockey_user;")
            return False
    
    def insert_initial_data(self):
        """Insert initial leagues and teams data"""
        
        try:
            # Insert NHL league
            nhl_league_sql = """
            INSERT INTO leagues (name, country, level, season_start, season_end)
            VALUES ('NHL', 'North America', 1, 2023, 2024)
            ON CONFLICT DO NOTHING
            RETURNING id;
            """
            
            # NHL teams data (same as before)
            nhl_teams = [
                # Eastern Conference - Atlantic Division
                ('Boston Bruins', 'Boston', 'Eastern', 'Atlantic', 'BOS'),
                ('Buffalo Sabres', 'Buffalo', 'Eastern', 'Atlantic', 'BUF'),
                ('Detroit Red Wings', 'Detroit', 'Eastern', 'Atlantic', 'DET'),
                ('Florida Panthers', 'Sunrise', 'Eastern', 'Atlantic', 'FLA'),
                ('Montreal Canadiens', 'Montreal', 'Eastern', 'Atlantic', 'MTL'),
                ('Ottawa Senators', 'Ottawa', 'Eastern', 'Atlantic', 'OTT'),
                ('Tampa Bay Lightning', 'Tampa Bay', 'Eastern', 'Atlantic', 'TBL'),
                ('Toronto Maple Leafs', 'Toronto', 'Eastern', 'Atlantic', 'TOR'),
                
                # Eastern Conference - Metropolitan Division
                ('Carolina Hurricanes', 'Raleigh', 'Eastern', 'Metropolitan', 'CAR'),
                ('Columbus Blue Jackets', 'Columbus', 'Eastern', 'Metropolitan', 'CBJ'),
                ('New Jersey Devils', 'Newark', 'Eastern', 'Metropolitan', 'NJD'),
                ('New York Islanders', 'Elmont', 'Eastern', 'Metropolitan', 'NYI'),
                ('New York Rangers', 'New York', 'Eastern', 'Metropolitan', 'NYR'),
                ('Philadelphia Flyers', 'Philadelphia', 'Eastern', 'Metropolitan', 'PHI'),
                ('Pittsburgh Penguins', 'Pittsburgh', 'Eastern', 'Metropolitan', 'PIT'),
                ('Washington Capitals', 'Washington', 'Eastern', 'Metropolitan', 'WSH'),
                
                # Western Conference - Central Division
                ('Arizona Coyotes', 'Tempe', 'Western', 'Central', 'ARI'),
                ('Chicago Blackhawks', 'Chicago', 'Western', 'Central', 'CHI'),
                ('Colorado Avalanche', 'Denver', 'Western', 'Central', 'COL'),
                ('Dallas Stars', 'Dallas', 'Western', 'Central', 'DAL'),
                ('Minnesota Wild', 'Saint Paul', 'Western', 'Central', 'MIN'),
                ('Nashville Predators', 'Nashville', 'Western', 'Central', 'NSH'),
                ('St. Louis Blues', 'St. Louis', 'Western', 'Central', 'STL'),
                ('Winnipeg Jets', 'Winnipeg', 'Western', 'Central', 'WPG'),
                
                # Western Conference - Pacific Division
                ('Anaheim Ducks', 'Anaheim', 'Western', 'Pacific', 'ANA'),
                ('Calgary Flames', 'Calgary', 'Western', 'Pacific', 'CGY'),
                ('Edmonton Oilers', 'Edmonton', 'Western', 'Pacific', 'EDM'),
                ('Los Angeles Kings', 'Los Angeles', 'Western', 'Pacific', 'LAK'),
                ('San Jose Sharks', 'San Jose', 'Western', 'Pacific', 'SJS'),
                ('Seattle Kraken', 'Seattle', 'Western', 'Pacific', 'SEA'),
                ('Vancouver Canucks', 'Vancouver', 'Western', 'Pacific', 'VAN'),
                ('Vegas Golden Knights', 'Las Vegas', 'Western', 'Pacific', 'VGK'),
            ]
            
            with self.engine.connect() as conn:
                # Insert NHL league
                result = conn.execute(text(nhl_league_sql))
                league_row = result.fetchone()
                
                if league_row:
                    nhl_league_id = league_row[0]
                else:
                    # League already exists, get its ID
                    result = conn.execute(text("SELECT id FROM leagues WHERE name = 'NHL'"))
                    nhl_league_id = result.fetchone()[0]
                
                # Insert teams with better error handling
                teams_inserted = 0
                for name, city, conference, division, abbrev in nhl_teams:
                    try:
                        team_sql = """
                        INSERT INTO teams (name, city, league_id, conference, division, abbreviation)
                        VALUES (:name, :city, :league_id, :conference, :division, :abbreviation)
                        ON CONFLICT DO NOTHING;
                        """
                        conn.execute(text(team_sql), {
                            'name': name,
                            'city': city,
                            'league_id': nhl_league_id,
                            'conference': conference,
                            'division': division,
                            'abbreviation': abbrev
                        })
                        teams_inserted += 1
                    except Exception as e:
                        logger.warning(f"Failed to insert team {name}: {e}")
                
                conn.commit()
                logger.info(f"✅ Inserted NHL league and {teams_inserted} teams")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error inserting initial data: {e}")
            return False
    
    # Rest of the methods remain the same...
    def import_scraped_data(self, data_directory: str = "data/raw"):
        """Import scraped CSV data into database"""
        
        try:
            # Find latest CSV files
            csv_files = glob.glob(f"{data_directory}/nhl_*.csv")
            if not csv_files:
                logger.warning("No CSV files found to import")
                return False
            
            # Get the latest timestamp
            latest_files = [f for f in csv_files if 'summary' not in f]
            if not latest_files:
                logger.warning("No data CSV files found")
                return False
                
            latest_timestamp = max([f.split('_')[-1].replace('.csv', '') for f in latest_files])
            
            # Import games data
            games_file = f"{data_directory}/nhl_games_{latest_timestamp}.csv"
            if os.path.exists(games_file):
                self._import_games(games_file)
            
            # Import team stats data  
            stats_file = f"{data_directory}/nhl_team_stats_{latest_timestamp}.csv"
            if os.path.exists(stats_file):
                self._import_team_stats(stats_file)
                
            logger.info("✅ Data import completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error importing data: {e}")
            return False
    
    def _import_games(self, filename: str):
        """Import games data from CSV with better error handling"""
        
        try:
            df = pd.read_csv(filename)
            logger.info(f"Importing {len(df)} games from {filename}")
            
            # Get team name to ID mapping
            team_mapping = self._get_team_mapping()
            
            if not team_mapping:
                logger.error("No teams found in database. Please ensure teams are inserted first.")
                return
            
            # Prepare data for insertion
            games_data = []
            skipped = 0
            
            for _, row in df.iterrows():
                home_team_id = team_mapping.get(row['home_team'])
                away_team_id = team_mapping.get(row['visitor_team'])
                
                if home_team_id and away_team_id:
                    games_data.append({
                        'date': row['date'],
                        'season': row['season'],
                        'league_id': 1,  # NHL
                        'home_team_id': home_team_id,
                        'away_team_id': away_team_id,
                        'home_score': row['home_score'] if pd.notna(row['home_score']) else None,
                        'away_score': row['visitor_score'] if pd.notna(row['visitor_score']) else None,
                        'overtime_shootout': row.get('overtime_shootout', '') if pd.notna(row.get('overtime_shootout', '')) else '',
                        'status': row['status'],
                        'scraped_at': row['scraped_at']
                    })
                else:
                    skipped += 1
                    if skipped <= 5:  # Log first few skipped entries
                        logger.warning(f"Skipped game: {row['visitor_team']} vs {row['home_team']} (teams not found)")
            
            # Insert data
            if games_data:
                games_df = pd.DataFrame(games_data)
                games_df.to_sql('games', self.engine, if_exists='append', index=False)
                logger.info(f"✅ Imported {len(games_data)} games (skipped {skipped})")
            else:
                logger.warning("No games data to import")
                
        except Exception as e:
            logger.error(f"❌ Error importing games: {e}")
    
    def _import_team_stats(self, filename: str):
        """Import team stats data from CSV with better error handling"""
        
        try:
            if not os.path.exists(filename):
                logger.warning(f"Team stats file not found: {filename}")
                return
                
            df = pd.read_csv(filename)
            logger.info(f"Importing {len(df)} team stats from {filename}")
            
            # Get team name to ID mapping
            team_mapping = self._get_team_mapping()
            
            # Prepare data for insertion
            stats_data = []
            skipped = 0
            
            for _, row in df.iterrows():
                team_id = team_mapping.get(row.get('Team', ''))
                
                if team_id:
                    stats_data.append({
                        'team_id': team_id,
                        'season': row['season'],
                        'games_played': self._safe_int(row.get('GP')),
                        'wins': self._safe_int(row.get('W')),
                        'losses': self._safe_int(row.get('L')),
                        'overtime_losses': self._safe_int(row.get('OL')),
                        'points': self._safe_int(row.get('PTS')),
                        'goals_for': self._safe_int(row.get('GF')),
                        'goals_against': self._safe_int(row.get('GA')),
                        'scraped_at': row['scraped_at']
                    })
                else:
                    skipped += 1
                    if skipped <= 5:
                        logger.warning(f"Skipped team stats: {row.get('Team', 'Unknown')} (team not found)")
            
            # Insert data
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_sql('team_stats', self.engine, if_exists='append', index=False)
                logger.info(f"✅ Imported {len(stats_data)} team stats (skipped {skipped})")
            else:
                logger.warning("No team stats data to import")
                
        except Exception as e:
            logger.error(f"❌ Error importing team stats: {e}")
    
    def _get_team_mapping(self) -> Dict[str, int]:
        """Get mapping from team names to team IDs"""
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT id, name FROM teams"))
                return {row[1]: row[0] for row in result.fetchall()}
        except Exception as e:
            logger.error(f"❌ Error getting team mapping: {e}")
            return {}
    
    def _safe_int(self, value) -> Optional[int]:
        """Safely convert value to int"""
        try:
            return int(value) if pd.notna(value) else None
        except (ValueError, TypeError):
            return None
    
    def get_data_summary(self):
        """Get summary of imported data with error handling"""
        
        try:
            with self.engine.connect() as conn:
                # Games summary
                games_result = conn.execute(text("""
                    SELECT 
                        season,
                        COUNT(*) as total_games,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_games,
                        MIN(date) as first_game,
                        MAX(date) as last_game
                    FROM games 
                    GROUP BY season 
                    ORDER BY season
                """))
                
                logger.info("\n📊 GAMES SUMMARY:")
                for row in games_result:
                    logger.info(f"  Season {row[0]}: {row[1]} total games, {row[2]} completed ({row[3]} to {row[4]})")
                
                # Teams summary
                teams_result = conn.execute(text("SELECT COUNT(*) FROM teams"))
                teams_count = teams_result.fetchone()[0]
                logger.info(f"\n🏒 TEAMS: {teams_count} teams imported")
                
                # Team stats summary
                stats_result = conn.execute(text("""
                    SELECT season, COUNT(*) as teams_with_stats
                    FROM team_stats 
                    GROUP BY season 
                    ORDER BY season
                """))
                
                logger.info("\n📈 TEAM STATS SUMMARY:")
                for row in stats_result:
                    logger.info(f"  Season {row[0]}: {row[1]} teams with stats")
                    
        except Exception as e:
            logger.error(f"❌ Error generating summary: {e}")

def main():
    """Main function with better error handling"""
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    logger.info("🏒 Starting database setup and data import...")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Create tables
        logger.info("Creating database tables...")
        if not db_manager.create_tables():
            logger.error("❌ Failed to create tables. Please check permissions.")
            return
        
        # Insert initial data
        logger.info("Inserting initial leagues and teams...")
        if not db_manager.insert_initial_data():
            logger.error("❌ Failed to insert initial data.")
            return
        
        # Import scraped data
        logger.info("Importing scraped NHL data...")  
        if not db_manager.import_scraped_data():
            logger.error("❌ Failed to import scraped data.")
            return
        
        # Display summary
        logger.info("Generating data summary...")
        db_manager.get_data_summary()
        
        logger.info("🎉 Database setup and import completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        raise

if __name__ == "__main__":
    main()