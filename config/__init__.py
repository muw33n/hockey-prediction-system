#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Config Package
==========================================
Centralizovaný přístup ke všem konfiguračním komponentům.

Umístění: config/__init__.py

Použití:
    from config import PATHS, settings, setup_logging, get_logger
    
    # Místo:
    # from config.paths import PATHS
    # from config.settings import settings
    # from config.logging_config import setup_logging, get_logger
"""

# === Import order je kritický kvůli dependencies ===

# 1. PATHS - žádné dependencies, základní singleton
from .paths import PATHS, setup_project_paths

# 2. LOGGING - žádné dependencies na project komponenty
from .logging_config import (
    setup_logging,
    get_logger,
    LoggingConfig,
    ColoredFormatter,
    PerformanceLogger
)

# 3. SETTINGS - může používat PATHS (importováno výše)
from .settings import (
    settings,
    get_setting,
    update_setting,
    DatabaseConfig,
    ModelConfig,
    BacktestingConfig,
    ScrapingConfig,
    LoggingConfig as SettingsLoggingConfig,  # Rename kvůli konfliktu
    NotificationConfig,
    FeatureFlags
)

# === Package metadata ===
__version__ = "1.0.0"
__author__ = "Hockey Prediction System"

# === Explicitní export control ===
__all__ = [
    # Core singletons - nejčastější použití
    'PATHS',
    'settings',
    
    # Logging functions - často používané
    'setup_logging',
    'get_logger',
    
    # Path utilities
    'setup_project_paths',
    
    # Settings utilities
    'get_setting',
    'update_setting',
    
    # Advanced classes - pro pokročilé použití
    'LoggingConfig',
    'ColoredFormatter', 
    'PerformanceLogger',
    
    # Settings dataclasses - pro type hints a advanced config
    'DatabaseConfig',
    'ModelConfig',
    'BacktestingConfig',
    'ScrapingConfig',
    'SettingsLoggingConfig',
    'NotificationConfig',
    'FeatureFlags',
]

# === Package-level inicializace ===
def initialize_config(log_level: str = None, ensure_dirs: bool = True):
    """
    Inicializuj kompletní konfiguraci projektu.
    
    Args:
        log_level: Úroveň loggingu (override default)
        ensure_dirs: Zda vytvořit chybějící adresáře
        
    Returns:
        bool: True pokud úspěšně inicializováno
        
    Usage:
        from config import initialize_config
        initialize_config(log_level='DEBUG')
    """
    try:
        # 1. Zajisti existenci adresářů
        if ensure_dirs:
            PATHS.ensure_directories()
        
        # 2. Setup loggingu s custom level nebo z settings
        setup_logging(
            log_level=log_level or settings.logging.log_level,
            log_dir=PATHS.logs,
            log_to_file=settings.logging.log_to_file,
            colorize=True
        )
        
        # 3. Validuj konfiguraci
        validation = PATHS.validate()
        
        if validation['is_valid']:
            logger = get_logger(__name__)
            logger.info("✅ Config package initialized successfully")
            logger.info(f"   Root: {PATHS.root}")
            logger.info(f"   Environment: {settings.environment}")
            logger.info(f"   Log level: {log_level or settings.logging.log_level}")
            return True
        else:
            print("❌ Config validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Config initialization failed: {e}")
        return False

# === Convenience functions pro common patterns ===
def get_data_file(data_type: str, latest: bool = True):
    """
    Shortcut pro získání datových souborů.
    
    Args:
        data_type: Typ dat ('games', 'odds', 'standings', 'team_stats')  
        latest: Pokud True, vrátí nejnovější soubor
        
    Returns:
        Path: Cesta k souboru
        
    Usage:
        from config import get_data_file
        latest_games = get_data_file('games')
    """
    return PATHS.get_data_file(data_type, latest=latest)

def get_model_path(model_name: str):
    """
    Shortcut pro cesty k modelům.
    
    Args:
        model_name: Název modelu (např. 'elo_model.pkl')
        
    Returns:
        Path: Cesta k modelu
        
    Usage:
        from config import get_model_path
        model_file = get_model_path('elo_model.pkl')
    """
    return PATHS.trained_models / model_name

def get_experiment_dir(experiment_name: str = None):
    """
    Shortcut pro experiment adresáře.
    
    Args:
        experiment_name: Název experimentu (optional)
        
    Returns:
        Path: Cesta k experiment adresáři
        
    Usage:
        from config import get_experiment_dir
        exp_dir = get_experiment_dir('backtest_20250101')
    """
    if experiment_name:
        return PATHS.experiments / experiment_name
    return PATHS.experiments

# === Package level validace při importu ===
def _validate_config_on_import():
    """Validace konfigurace při importu package"""
    try:
        # Rychlá validace bez side-effectů
        validation = PATHS.validate()
        if not validation['is_valid']:
            import warnings
            warnings.warn(
                "Config package imported but project structure validation failed. "
                "Run initialize_config() to setup properly.",
                ImportWarning
            )
    except Exception:
        # Tichá chyba při importu - nechceme zlomit import
        pass

# Spusť validaci při importu
_validate_config_on_import()

# === Auto-completion hints pro IDEs ===
if False:  # Type checking only
    from pathlib import Path
    
    # IDE hint: PATHS obsahuje tyto atributy
    PATHS: 'ProjectPaths'
    PATHS.root: Path
    PATHS.data: Path  
    PATHS.raw_data: Path
    PATHS.logs: Path
    # ... a další