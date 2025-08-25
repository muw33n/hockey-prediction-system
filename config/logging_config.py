#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Centrální konfigurace logování
=========================================================
Jednotné nastavení logování pro celý projekt.

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
    """Centrální správa logování"""
    
    # Barevné kódy pro konzolový výstup (Windows i Linux)
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green  
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    @staticmethod
    def setup_logging(
        log_level: Optional[str] = None,
        log_dir: Optional[Path] = None,
        log_to_file: Optional[bool] = None,
        log_to_console: bool = True,
        colorize: bool = True
    ) -> None:
        """
        Nastaví jednotné logování pro celý projekt.
        
        Args:
            log_level: Úroveň logování (DEBUG, INFO, WARNING, ERROR)
            log_dir: Adresář pro log soubory
            log_to_file: Zda logovat do souboru
            log_to_console: Zda logovat do konzole
            colorize: Zda používat barvy v konzoli
        """
        
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
        
        # Souborový handler
        if log_to_file:
            # Hlavní log soubor
            main_log_file = log_dir / 'hockey_system.log'
            file_handler = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',  # Do souboru vše
                'formatter': 'detailed',
                'filename': str(main_log_file),
                'maxBytes': int(os.getenv('LOG_FILE_MAX_BYTES', '10485760')),  # 10MB
                'backupCount': int(os.getenv('LOG_FILE_BACKUP_COUNT', '5')),
                'encoding': 'utf-8'
            }
            handlers['file'] = file_handler
            
            # Error log soubor (pouze ERROR a výše)
            error_log_file = log_dir / 'errors.log'
            error_handler = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': str(error_log_file),
                'maxBytes': 5242880,  # 5MB
                'backupCount': 3,
                'encoding': 'utf-8'
            }
            handlers['error_file'] = error_handler
        
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
            'colored': {
                '()': 'config.logging_config.ColoredFormatter',
                'format': '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                'datefmt': '%H:%M:%S'
            }
        }
        
        # Sestavení konfigurace
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': formatters,
            'handlers': handlers,
            'loggers': {
                '': {  # Root logger
                    'handlers': list(handlers.keys()),
                    'level': 'DEBUG'
                },
                # Specifické loggery pro různé moduly
                'database': {
                    'handlers': list(handlers.keys()),
                    'level': log_level,
                    'propagate': False
                },
                'models': {
                    'handlers': list(handlers.keys()),
                    'level': log_level,
                    'propagate': False
                },
                'betting': {
                    'handlers': list(handlers.keys()),
                    'level': log_level,
                    'propagate': False
                },
                'scraping': {
                    'handlers': list(handlers.keys()),
                    'level': log_level,
                    'propagate': False
                }
            }
        }
        
        # Aplikuj konfiguraci
        logging.config.dictConfig(logging_config)
        
        # Log úvodní zprávu
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("🏒 Hockey Prediction System - Logging Initialized")
        logger.info(f"   Log Level: {log_level}")
        logger.info(f"   Log Directory: {log_dir}")
        logger.info(f"   File Logging: {'Enabled' if log_to_file else 'Disabled'}")
        logger.info(f"   Console Logging: {'Enabled' if log_to_console else 'Disabled'}")
        logger.info("=" * 60)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Získá logger pro daný modul.
        
        Args:
            name: Název modulu (obvykle __name__)
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)
    
    @staticmethod
    def setup_notebook_logging():
        """Speciální nastavení pro Jupyter notebooks"""
        import sys
        
        # Odstranění existujících handlerů
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
    def create_run_logger(run_name: str) -> logging.Logger:
        """
        Vytvoří speciální logger pro konkrétní běh/experiment.
        
        Args:
            run_name: Název běhu (např. 'backtest_20250101_120000')
            
        Returns:
            Logger pro tento běh
        """
        try:
            from config.paths import PATHS
            log_dir = PATHS.logs / 'runs'
        except ImportError:
            log_dir = Path.cwd() / 'logs' / 'runs'
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Vytvoř logger
        logger = logging.getLogger(run_name)
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
        self.logger.debug(f"⏱️ Started: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """Ukončí měření a vrátí dobu trvání"""
        from time import time
        
        if operation not in self.timers:
            self.logger.warning(f"Timer for '{operation}' was not started")
            return 0.0
        
        duration = time() - self.timers[operation]
        del self.timers[operation]
        
        if duration < 1:
            self.logger.info(f"⏱️ Completed: {operation} ({duration*1000:.1f}ms)")
        else:
            self.logger.info(f"⏱️ Completed: {operation} ({duration:.2f}s)")
        
        return duration


# === Convenience funkce ===

def setup_logging(**kwargs):
    """Zkratka pro nastavení logování"""
    LoggingConfig.setup_logging(**kwargs)


def get_logger(name: str = None) -> logging.Logger:
    """Zkratka pro získání loggeru"""
    if name is None:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'unknown')
    return LoggingConfig.get_logger(name)


# === Test ===

if __name__ == "__main__":
    print("🧪 Testing logging configuration...")
    
    # Setup logging
    LoggingConfig.setup_logging(
        log_level='DEBUG',
        log_to_file=True,
        colorize=True
    )
    
    # Test různých úrovní
    logger = get_logger(__name__)
    
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
    
    # Test performance loggeru
    perf_logger = PerformanceLogger(logger)
    
    import time
    perf_logger.start_timer("test_operation")
    time.sleep(0.5)
    perf_logger.end_timer("test_operation")
    
    # Test exception loggingu
    try:
        1 / 0
    except Exception as e:
        LoggingConfig.log_exception(logger, e, "Testing exception logging")
    
    print("\n✅ Logging test completed! Check logs/ directory for output files.")