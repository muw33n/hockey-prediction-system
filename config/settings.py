#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Centrální konfigurace
================================================
Jednotná správa všech nastavení a parametrů systému.

Umístění: config/settings.py
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import json
import logging

# Načti .env soubor
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Konfigurace databáze"""
    host: str = field(default_factory=lambda: os.getenv('DB_HOST', 'localhost'))
    port: int = field(default_factory=lambda: int(os.getenv('DB_PORT', '5432')))
    database: str = field(default_factory=lambda: os.getenv('DB_NAME', 'hockey_predictions'))
    user: str = field(default_factory=lambda: os.getenv('DB_USER', 'postgres'))
    password: str = field(default_factory=lambda: os.getenv('DB_PASSWORD', ''))
    schema: str = field(default_factory=lambda: os.getenv('DB_SCHEMA', 'public'))
    
    @property
    def connection_string(self) -> str:
        """Vrátí connection string pro SQLAlchemy"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def connection_params(self) -> dict:
        """Vrátí parametry pro psycopg2"""
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user,
            'password': self.password
        }


@dataclass
class ModelConfig:
    """Konfigurace modelů"""
    # Elo model parameters
    elo_k_factor: float = field(default_factory=lambda: float(os.getenv('ELO_K_FACTOR', '32.0')))
    elo_home_advantage: float = field(default_factory=lambda: float(os.getenv('ELO_HOME_ADVANTAGE', '100.0')))
    elo_initial_rating: float = field(default_factory=lambda: float(os.getenv('ELO_INITIAL_RATING', '1500.0')))
    elo_season_regression: float = field(default_factory=lambda: float(os.getenv('ELO_SEASON_REGRESSION', '0.25')))
    
    # ML model parameters
    xgboost_n_estimators: int = field(default_factory=lambda: int(os.getenv('XGB_N_ESTIMATORS', '100')))
    xgboost_max_depth: int = field(default_factory=lambda: int(os.getenv('XGB_MAX_DEPTH', '6')))
    xgboost_learning_rate: float = field(default_factory=lambda: float(os.getenv('XGB_LEARNING_RATE', '0.3')))
    
    # Training parameters
    test_size: float = field(default_factory=lambda: float(os.getenv('TEST_SIZE', '0.2')))
    random_state: int = field(default_factory=lambda: int(os.getenv('RANDOM_STATE', '42')))
    cv_folds: int = field(default_factory=lambda: int(os.getenv('CV_FOLDS', '5')))


@dataclass
class BacktestingConfig:
    """Konfigurace backtestingu"""
    initial_bankroll: float = field(default_factory=lambda: float(os.getenv('INITIAL_BANKROLL', '10000.0')))
    max_bet_percentage: float = field(default_factory=lambda: float(os.getenv('MAX_BET_PERCENTAGE', '0.05')))
    min_bet_amount: float = field(default_factory=lambda: float(os.getenv('MIN_BET_AMOUNT', '10.0')))
    max_bet_amount: float = field(default_factory=lambda: float(os.getenv('MAX_BET_AMOUNT', '500.0')))
    
    # Value betting thresholds
    min_edge_threshold: float = field(default_factory=lambda: float(os.getenv('MIN_EDGE_THRESHOLD', '0.02')))
    min_odds: float = field(default_factory=lambda: float(os.getenv('MIN_ODDS', '1.5')))
    max_odds: float = field(default_factory=lambda: float(os.getenv('MAX_ODDS', '5.0')))
    
    # Kelly criterion
    kelly_fraction: float = field(default_factory=lambda: float(os.getenv('KELLY_FRACTION', '0.25')))
    use_kelly: bool = field(default_factory=lambda: os.getenv('USE_KELLY', 'true').lower() == 'true')


@dataclass
class ScrapingConfig:
    """Konfigurace scrapingu"""
    user_agent: str = field(default_factory=lambda: os.getenv(
        'USER_AGENT',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    ))
    request_timeout: int = field(default_factory=lambda: int(os.getenv('REQUEST_TIMEOUT', '30')))
    retry_count: int = field(default_factory=lambda: int(os.getenv('RETRY_COUNT', '3')))
    delay_between_requests: float = field(default_factory=lambda: float(os.getenv('REQUEST_DELAY', '1.0')))
    
    # API keys
    odds_api_key: Optional[str] = field(default_factory=lambda: os.getenv('ODDS_API_KEY'))
    
    # Bookmaker APIs
    bookmaker_apis: Dict[str, str] = field(default_factory=lambda: {
        'tipsport': os.getenv('TIPSPORT_API_KEY', ''),
        'fortuna': os.getenv('FORTUNA_API_KEY', ''),
        'betano': os.getenv('BETANO_API_KEY', '')
    })


@dataclass
class LoggingConfig:
    """Konfigurace logování"""
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    log_format: str = field(default_factory=lambda: os.getenv(
        'LOG_FORMAT',
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    ))
    log_to_file: bool = field(default_factory=lambda: os.getenv('LOG_TO_FILE', 'true').lower() == 'true')
    log_file_max_bytes: int = field(default_factory=lambda: int(os.getenv('LOG_FILE_MAX_BYTES', '10485760')))
    log_file_backup_count: int = field(default_factory=lambda: int(os.getenv('LOG_FILE_BACKUP_COUNT', '5')))


@dataclass
class NotificationConfig:
    """Konfigurace notifikací"""
    enable_notifications: bool = field(default_factory=lambda: os.getenv('ENABLE_NOTIFICATIONS', 'false').lower() == 'true')
    
    # Email
    smtp_host: Optional[str] = field(default_factory=lambda: os.getenv('SMTP_HOST'))
    smtp_port: int = field(default_factory=lambda: int(os.getenv('SMTP_PORT', '587')))
    smtp_user: Optional[str] = field(default_factory=lambda: os.getenv('SMTP_USER'))
    smtp_password: Optional[str] = field(default_factory=lambda: os.getenv('SMTP_PASSWORD'))
    notification_email: Optional[str] = field(default_factory=lambda: os.getenv('NOTIFICATION_EMAIL'))
    
    # Discord/Slack
    discord_webhook: Optional[str] = field(default_factory=lambda: os.getenv('DISCORD_WEBHOOK'))
    slack_webhook: Optional[str] = field(default_factory=lambda: os.getenv('SLACK_WEBHOOK'))


@dataclass
class FeatureFlags:
    """Feature flags pro postupné nasazování funkcí"""
    enable_live_trading: bool = field(default_factory=lambda: os.getenv('ENABLE_LIVE_TRADING', 'false').lower() == 'true')
    enable_ml_models: bool = field(default_factory=lambda: os.getenv('ENABLE_ML_MODELS', 'false').lower() == 'true')
    enable_multi_league: bool = field(default_factory=lambda: os.getenv('ENABLE_MULTI_LEAGUE', 'false').lower() == 'true')
    enable_advanced_analytics: bool = field(default_factory=lambda: os.getenv('ENABLE_ADVANCED_ANALYTICS', 'false').lower() == 'true')
    dry_run_mode: bool = field(default_factory=lambda: os.getenv('DRY_RUN_MODE', 'true').lower() == 'true')


class Settings:
    """Hlavní třída pro správu všech nastavení"""
    
    def __init__(self):
        """Inicializace všech konfigurací"""
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.backtesting = BacktestingConfig()
        self.scraping = ScrapingConfig()
        self.logging = LoggingConfig()
        self.notifications = NotificationConfig()
        self.features = FeatureFlags()
        
        # Metadata
        self.version = os.getenv('APP_VERSION', '1.0.0')
        self.environment = os.getenv('ENVIRONMENT', 'development')
        
        # Validace konfigurace při inicializaci
        self._validate()
    
    def _validate(self):
        """Validace konfigurace"""
        errors = []
        
        # Kontrola databázového připojení
        if not self.database.database:
            errors.append("Database name is not configured")
        
        # Kontrola API klíčů pro produkci
        if self.environment == 'production':
            if self.features.enable_live_trading and not self.scraping.odds_api_key:
                errors.append("Odds API key is required for live trading")
            
            if self.notifications.enable_notifications:
                if not (self.notifications.smtp_host or self.notifications.discord_webhook or self.notifications.slack_webhook):
                    errors.append("At least one notification method must be configured")
        
        # Log varování pro chybějící konfigurace
        if errors:
            for error in errors:
                logger.warning(f"Configuration warning: {error}")
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Převede konfiguraci na dictionary.
        
        Args:
            include_sensitive: Pokud False, vynechá citlivé údaje (default: False)
        """
        config_dict = {
            'version': self.version,
            'environment': self.environment,
            'database': {
                'host': self.database.host,
                'port': self.database.port,
                'database': self.database.database,
                # Heslo NIKDY neukládáme
                'user': self.database.user if include_sensitive else '***'
            },
            'model': {
                'elo_k_factor': self.model.elo_k_factor,
                'elo_home_advantage': self.model.elo_home_advantage,
                'elo_initial_rating': self.model.elo_initial_rating,
                'elo_season_regression': self.model.elo_season_regression
            },
            'backtesting': {
                'initial_bankroll': self.backtesting.initial_bankroll,
                'max_bet_percentage': self.backtesting.max_bet_percentage,
                'min_edge_threshold': self.backtesting.min_edge_threshold,
                'use_kelly': self.backtesting.use_kelly,
                'kelly_fraction': self.backtesting.kelly_fraction
            },
            'scraping': {
                'request_timeout': self.scraping.request_timeout,
                'retry_count': self.scraping.retry_count,
                'delay_between_requests': self.scraping.delay_between_requests,
                # API klíče NIKDY neukládáme
                'has_odds_api_key': bool(self.scraping.odds_api_key) if not include_sensitive else '***'
            },
            'logging': {
                'log_level': self.logging.log_level,
                'log_to_file': self.logging.log_to_file
            },
            'features': {
                'enable_live_trading': self.features.enable_live_trading,
                'enable_ml_models': self.features.enable_ml_models,
                'enable_multi_league': self.features.enable_multi_league,
                'dry_run_mode': self.features.dry_run_mode
            }
        }
        
        return config_dict
    
    def save_to_file(self, filepath: Optional[Path] = None):
        """Uloží konfiguraci do JSON souboru (BEZ citlivých údajů)"""
        if filepath is None:
            # Oprava importu pro standalone spuštění
            try:
                from config.paths import PATHS
                filepath = PATHS.config / 'current_settings.json'
            except ModuleNotFoundError:
                # Fallback když spouštíme přímo
                filepath = Path(__file__).parent / 'current_settings.json'
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Uložit BEZ citlivých údajů
        config_data = self.to_dict(include_sensitive=False)
        
        # Přidat header s varováním
        config_data['_warning'] = "This file does NOT contain sensitive data. Passwords and API keys are stored in .env file."
        config_data['_generated_at'] = os.getenv('TZ', 'UTC') + ' ' + str(Path(__file__).stat().st_mtime)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Settings saved to {filepath} (sensitive data excluded)")
    
    def __str__(self) -> str:
        """String reprezentace konfigurace"""
        return f"""
Hockey Prediction System Settings
=================================
Version: {self.version}
Environment: {self.environment}
Database: {self.database.database} @ {self.database.host}:{self.database.port}
Dry Run Mode: {self.features.dry_run_mode}
Live Trading: {self.features.enable_live_trading}
ML Models: {self.features.enable_ml_models}
"""


# === Singleton instance ===
settings = Settings()


# === Helper functions ===

def get_setting(path: str, default: Any = None) -> Any:
    """
    Získá nastavení podle cesty (dot notation).
    
    Args:
        path: Cesta k nastavení (např. 'database.host')
        default: Výchozí hodnota
        
    Returns:
        Hodnota nastavení
    """
    parts = path.split('.')
    obj = settings
    
    try:
        for part in parts:
            obj = getattr(obj, part)
        return obj
    except AttributeError:
        return default


def update_setting(path: str, value: Any) -> bool:
    """
    Aktualizuje nastavení (pouze pro runtime, neukládá do .env).
    
    Args:
        path: Cesta k nastavení
        value: Nová hodnota
        
    Returns:
        bool: True pokud úspěšné
    """
    parts = path.split('.')
    
    if len(parts) < 2:
        return False
    
    # Najdi parent objekt
    obj = settings
    for part in parts[:-1]:
        obj = getattr(obj, part, None)
        if obj is None:
            return False
    
    # Nastav hodnotu
    try:
        setattr(obj, parts[-1], value)
        logger.info(f"Updated setting {path} = {value}")
        return True
    except Exception as e:
        logger.error(f"Failed to update setting {path}: {e}")
        return False


# === Test funkce ===

if __name__ == "__main__":
    # Zobraz aktuální konfiguraci
    print(settings)
    
    # Test získání nastavení
    print("\n📊 Database settings:")
    print(f"  Host: {get_setting('database.host')}")
    print(f"  Port: {get_setting('database.port')}")
    print(f"  Database: {get_setting('database.database')}")
    print(f"  Connection string: {settings.database.connection_string}")
    
    print("\n🎯 Model settings:")
    print(f"  Elo K-factor: {settings.model.elo_k_factor}")
    print(f"  Elo home advantage: {settings.model.elo_home_advantage}")
    
    print("\n💰 Backtesting settings:")
    print(f"  Initial bankroll: ${settings.backtesting.initial_bankroll:,.2f}")
    print(f"  Max bet: {settings.backtesting.max_bet_percentage:.1%}")
    print(f"  Use Kelly: {settings.backtesting.use_kelly}")
    
    print("\n🚀 Feature flags:")
    print(f"  Live trading: {settings.features.enable_live_trading}")
    print(f"  ML models: {settings.features.enable_ml_models}")
    print(f"  Dry run: {settings.features.dry_run_mode}")
    
    # Test uložení do souboru
    if input("\n💾 Save settings to file? (y/n): ").lower() == 'y':
        settings.save_to_file()
        print("✅ Settings saved!")