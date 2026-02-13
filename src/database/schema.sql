-- Hockey Prediction System - Database Schema
-- ============================================
-- SQL DDL definitions for the hockey prediction database.
-- Extracted from database_setup.py for separation of concerns.
--
-- Location: src/database/schema.sql

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

-- Goalie stats table
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

-- Player stats table
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

-- Predictions table
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

-- Value bets table
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

-- Helper function to get team for a specific date
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
