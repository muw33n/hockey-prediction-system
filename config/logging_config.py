#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Centrální konfigurace logování (Enhanced)
==================================================================
Jednotné nastavení logování pro celý projekt s per-component log files.

Umístění: config/logging_config.py
"""

import logging
import logging.config
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
import os
import sys
from datetime import datetime


class LoggingConfig:
    """Centrální správa logování s per-component support"""
    
    # Barevné kódy pro konzolový výstup (Windows i Linux)
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green  
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    # Definované komponenty s vlastními log soubory
    COMPONENTS = {
        'database': 'database.log',
        'models': 'models.log', 
        'betting': 'betting.log',
        'scraping': 'scraping.log',
        'features': 'features.log',
        'analysis': 'analysis.log',
        'utils': 'utils.log',
        'notebooks': 'notebooks.log'
    }
    
    _initialized = False
    _component_loggers = {}
    
    @staticmethod
    def setup_logging(
        log_level: Optional[str] = None,
        log_dir: Optional[Path] = None,
        log_to_file: Optional[bool] = None,
        log_to_console: bool = True,
        colorize: bool = True,
        component_files: bool = True
    ) -> None:
        """
        Nastaví jednotné logování pro celý projekt s per-component files.
        
        Args:
            log_level: Úroveň logování (DEBUG, INFO, WARNING, ERROR)
            log_dir: Adresář pro log soubory
            log_to_file: Zda logovat do souboru
            log_to_console: Zda logovat do konzole
            colorize: Zda používat barvy v konzoli
            component_files: Zda vytvářet separátní soubory per-component
        """
        
        if LoggingConfig._initialized:
            return
        
        # Načti konfiguraci z prostředí nebo použij výchozí
        if log_level is None:
            log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        if log_to_file is None:
            log_to_file = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
        
        # Nastav adresář pro logy
        if log_dir is None:
            try:
                from config.paths import PATHS
                log_dir = PATHS.logs
            except ImportError:
                # Fallback pro standalone spuštění
                log_dir = Path.cwd() / 'logs'
        
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Připrav handlery
        handlers = {}
        
        # Konzolový handler
        if log_to_console:
            console_handler = {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'colored' if colorize else 'standard',
                'stream': 'ext://sys.stdout'
            }
            handlers['console'] = console_handler
        
        # Souborové handlers
        if log_to_file:
            # Hlavní log soubor (všechno)
            main_log_file = log_dir / 'hockey_system.log'
            handlers['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',  # Do hlavního souboru vše
                'formatter': 'detailed',
                'filename': str(main_log_file),
                'maxBytes': int(os.getenv('LOG_FILE_MAX_BYTES', '10485760')),  # 10MB
                'backupCount': int(os.getenv('LOG_FILE_BACKUP_COUNT', '5')),
                'encoding': 'utf-8'
            }
            
            # Error log soubor (pouze ERROR a výše)
            error_log_file = log_dir / 'errors.log'
            handlers['error_file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': str(error_log_file),
                'maxBytes': 5242880,  # 5MB
                'backupCount': 3,
                'encoding': 'utf-8'
            }
            
            # Per-component log files
            if component_files:
                for component, filename in LoggingConfig.COMPONENTS.items():
                    component_log_file = log_dir / filename
                    handler_name = f'{component}_file'
                    
                    handlers[handler_name] = {
                        'class': 'logging.handlers.RotatingFileHandler',
                        'level': 'DEBUG',
                        'formatter': 'component',
                        'filename': str(component_log_file),
                        'maxBytes': 5242880,  # 5MB per component
                        'backupCount': 3,
                        'encoding': 'utf-8'
                    }
        
        # Konfigurace formátovačů
        formatters = {
            'standard': {
                'format': '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'component': {
                'format': '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'colored': {
                '()': 'config.logging_config.ColoredFormatter',
                'format': '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                'datefmt': '%H:%M:%S'
            }
        }
        
        # Sestavení konfigurace
        loggers_config = {
            '': {  # Root logger
                'handlers': ['console', 'file', 'error_file'] if log_to_file else ['console'],
                'level': 'DEBUG'
            }
        }
        
        # Per-component loggery s vlastními soubory
        if component_files and log_to_file:
            for component in LoggingConfig.COMPONENTS:
                component_handlers = ['console', f'{component}_file', 'error_file']
                loggers_config[component] = {
                    'handlers': component_handlers,
                    'level': log_level,
                    'propagate': False  # Nepropaguj do root loggeru (duplicity)
                }
        
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': formatters,
            'handlers': handlers,
            'loggers': loggers_config
        }
        
        # Aplikuj konfiguraci
        logging.config.dictConfig(logging_config)
        
        LoggingConfig._initialized = True
        
        # Log úvodní zprávu
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("Hockey Prediction System - Enhanced Logging Initialized")
        logger.info(f"   Log Level: {log_level}")
        logger.info(f"   Log Directory: {log_dir}")
        logger.info(f"   File Logging: {'Enabled' if log_to_file else 'Disabled'}")
        logger.info(f"   Component Files: {'Enabled' if component_files else 'Disabled'}")
        logger.info(f"   Console Logging: {'Enabled' if log_to_console else 'Disabled'}")
        logger.info("=" * 60)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Získá logger pro daný modul (legacy method).
        
        Args:
            name: Název modulu (obvykle __name__)
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)
    
    @staticmethod
    def get_component_logger(name: str, component: Optional[str] = None) -> logging.Logger:
        """
        Získá logger pro konkrétní komponentu s vlastním log souborem.
        
        Args:
            name: Název modulu (obvykle __name__)
            component: Komponenta ('database', 'models', 'betting', etc.)
            
        Returns:
            Logger instance
            
        Usage:
            logger = get_component_logger(__name__, 'database')
            logger = get_component_logger(__name__, 'models')
        """
        # Auto-detect komponenty ze jména modulu
        if component is None:
            component = LoggingConfig._detect_component_from_name(name)
        
        # Pokud komponenta není definovaná, použij root logger
        if component not in LoggingConfig.COMPONENTS:
            return logging.getLogger(name)
        
        # Vytvoř nebo získej component logger
        logger_name = f"{component}.{name.split('.')[-1]}"  # např. "database.database_setup"
        
        if logger_name not in LoggingConfig._component_loggers:
            logger = logging.getLogger(component)  # Použij component jako parent
            child_logger = logger.getChild(name.split('.')[-1])
            LoggingConfig._component_loggers[logger_name] = child_logger
        
        return LoggingConfig._component_loggers[logger_name]
    
    @staticmethod
    def _detect_component_from_name(name: str) -> Optional[str]:
        """
        Automaticky detekuje komponentu z názvu modulu.
        
        Args:
            name: Module name (např. 'src.database.database_setup')
            
        Returns:
            Component name nebo None
        """
        name_lower = name.lower()
        
        # Mapování module patterns na komponenty
        patterns = {
            'database': ['database', 'db'],
            'models': ['models', 'model', 'elo', 'rating'],
            'betting': ['betting', 'backtest', 'value'],
            'scraping': ['scraping', 'scraper', 'data_collection'],
            'features': ['features', 'feature', 'engineering'],
            'analysis': ['analysis', 'analyze'],
            'utils': ['utils', 'utility', 'helper'],
            'notebooks': ['notebook', 'ipynb']
        }
        
        for component, keywords in patterns.items():
            if any(keyword in name_lower for keyword in keywords):
                return component
        
        return None
    
    @staticmethod
    def setup_notebook_logging():
        """Speciální nastavení pro Jupyter notebooks"""
        import sys
        
        # Odstraní existující handlery
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Jednoduchý formát pro notebooks
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s | %(message)s',
            stream=sys.stdout
        )
    
    @staticmethod
    def log_exception(logger: logging.Logger, exception: Exception, 
                     context: Optional[str] = None) -> None:
        """
        Zaloguje výjimku s kontextem.
        
        Args:
            logger: Logger instance
            exception: Výjimka k zalogování
            context: Dodatečný kontext
        """
        import traceback
        
        error_msg = f"Exception occurred: {type(exception).__name__}: {str(exception)}"
        if context:
            error_msg = f"{context} - {error_msg}"
        
        logger.error(error_msg)
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
    
    @staticmethod
    def create_run_logger(run_name: str, component: Optional[str] = None) -> logging.Logger:
        """
        Vytvoří speciální logger pro konkrétní běh/experiment.
        
        Args:
            run_name: Název běhu (např. 'backtest_20250101_120000')
            component: Komponenta pro organizaci
            
        Returns:
            Logger pro tento běh
        """
        try:
            from config.paths import PATHS
            log_dir = PATHS.logs / 'runs'
        except ImportError:
            log_dir = Path.cwd() / 'logs' / 'runs'
        
        if component:
            log_dir = log_dir / component
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Vytvoř logger
        logger_name = f"run.{component}.{run_name}" if component else f"run.{run_name}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        # Soubor pro tento běh
        log_file = log_dir / f"{run_name}.log"
        
        # Handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Formát
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
        return logger


class ColoredFormatter(logging.Formatter):
    """Formátovač s barevným výstupem pro konzoli"""
    
    def format(self, record):
        # Původní formátování
        msg = super().format(record)
        
        # Přidej barvy podle úrovně
        if record.levelname in LoggingConfig.COLORS:
            levelname_color = (
                LoggingConfig.COLORS[record.levelname] + 
                record.levelname + 
                LoggingConfig.COLORS['RESET']
            )
            msg = msg.replace(record.levelname, levelname_color)
        
        return msg


class PerformanceLogger:
    """Logger pro měření výkonu operací"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers = {}
    
    def start_timer(self, operation: str) -> None:
        """Začne měřit čas operace"""
        from time import time
        self.timers[operation] = time()
        self.logger.debug(f"Started: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """Ukončí měření a vrátí dobu trvání"""
        from time import time
        
        if operation not in self.timers:
            self.logger.warning(f"Timer for '{operation}' was not started")
            return 0.0
        
        duration = time() - self.timers[operation]
        del self.timers[operation]
        
        if duration < 1:
            self.logger.info(f"Completed: {operation} ({duration*1000:.1f}ms)")
        else:
            self.logger.info(f"Completed: {operation} ({duration:.2f}s)")
        
        return duration


# === Enhanced convenience functions ===

def setup_logging(**kwargs):
    """Zkratka pro nastavení logování"""
    LoggingConfig.setup_logging(**kwargs)


def get_logger(name: str = None) -> logging.Logger:
    """Zkratka pro získání loggeru (legacy)"""
    if name is None:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'unknown')
    return LoggingConfig.get_logger(name)


def get_component_logger(name: str = None, component: str = None) -> logging.Logger:
    """Zkratka pro získání component loggeru"""
    if name is None:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'unknown')
    return LoggingConfig.get_component_logger(name, component)


# === Test ===

if __name__ == "__main__":
    print("Testing enhanced logging configuration...")
    
    # Setup enhanced logging
    LoggingConfig.setup_logging(
        log_level='DEBUG',
        log_to_file=True,
        colorize=True,
        component_files=True
    )
    
    # Test různých komponent
    database_logger = get_component_logger(__name__, 'database')
    models_logger = get_component_logger(__name__, 'models')  
    betting_logger = get_component_logger(__name__, 'betting')
    
    database_logger.info("Database operation completed")
    models_logger.info("Model training finished")
    betting_logger.info("Backtest analysis done")
    
    # Test auto-detection
    auto_logger = get_component_logger('src.database.database_setup')
    auto_logger.info("Auto-detected as database component")
    
    # Test performance logger
    perf_logger = PerformanceLogger(models_logger)
    
    import time
    perf_logger.start_timer("test_operation")
    time.sleep(0.1)
    perf_logger.end_timer("test_operation")
    
    # Test run logger
    run_logger = LoggingConfig.create_run_logger("test_run_20250126", "models")
    run_logger.info("Test run completed successfully")
    
    print("\nEnhanced logging test completed! Check logs/ directory for component-specific files.")