#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Notebook Helper (ENHANCED)
===================================================
Jednoduchý setup pro Jupyter notebooky s enhanced infrastructure.
Automaticky konfiguruje prostředí a importuje základní knihovny.
Integrace s per-component logging a safe file handlers.

Umístění: src/utils/notebook_helper.py
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    import pandas as pd
    from config.paths import ProjectPaths
    from config.logging_config import PerformanceLogger

# Suppress common warnings v noteboocích
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def setup_notebook_environment(
    project_root: Optional[str] = None,
    quiet: bool = False,
    component: str = 'notebooks'
) -> Dict[str, Any]:
    """
    Automatický setup pro Jupyter notebooky s enhanced infrastructure.
    
    Args:
        project_root: Cesta k root adresáři (optional)
        quiet: Potlačí výstupní zprávy
        component: Component name pro logging (default: 'notebooks')
        
    Returns:
        dict: Slovník s konfigurací prostředí
        
    Usage:
        # V první buňce notebooku:
        from src.utils.notebook_helper import setup_notebook_environment
        env = setup_notebook_environment()
        PATHS = env['PATHS']
        logger = env['logger']
    """
    
    # Set project root if provided
    if project_root:
        os.environ['HOCKEY_PROJECT_ROOT'] = str(Path(project_root).resolve())
    
    # Import PATHS (this handles sys.path automatically)
    try:
        from config.paths import PATHS, setup_project_paths
    except ImportError:
        # Fallback: try to find and add project root manually
        if not project_root:
            # Pokus najít root automaticky
            current = Path.cwd()
            for _ in range(10):
                if (current / 'requirements.txt').exists() or (current / 'setup.py').exists():
                    project_root = str(current)
                    os.environ['HOCKEY_PROJECT_ROOT'] = project_root
                    break
                if current.parent == current:
                    break
                current = current.parent
        
        if project_root:
            sys.path.insert(0, str(Path(project_root)))
            sys.path.insert(0, str(Path(project_root) / 'src'))
            
            try:
                from config.paths import PATHS, setup_project_paths
            except ImportError as e:
                raise ImportError(
                    f"Cannot import paths module: {e}\n"
                    f"Please ensure project structure is correct.\n"
                    f"Project root: {project_root}"
                )
        else:
            raise ImportError(
                "Cannot find project root. Please specify project_root parameter."
            )
    
    # Setup enhanced logging
    try:
        from config.logging_config import setup_logging, get_component_logger
        
        # Setup enhanced logging s per-component files
        setup_logging(
            log_level='INFO',
            log_to_file=True,
            log_to_console=not quiet,
            component_files=True  # Key: per-component log files
        )
        
        # Get component-specific logger pro notebooks
        logger = get_component_logger(f"notebook.{Path.cwd().name}", component)
        
    except ImportError:
        # Fallback logging
        import logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
        logger = logging.getLogger('notebook')
    
    # Setup paths and ensure directories
    paths_valid = setup_project_paths(verbose=not quiet)
    
    # Configure matplotlib pro Jupyter
    try:
        import matplotlib.pyplot as plt
        plt.style.use('default')
        # Set backend pro inline zobrazování
        if 'ipykernel' in sys.modules:
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['figure.dpi'] = 100
    except ImportError:
        pass
    
    # Configure pandas display options
    try:
        import pandas as pd
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
    except ImportError:
        pass
    
    # Return enhanced environment info
    env_info = {
        'PATHS': PATHS,
        'logger': logger,
        'project_root': str(PATHS.root),
        'paths_valid': paths_valid,
        'python_version': sys.version,
        'working_directory': str(Path.cwd()),
        'component': component,
        'sys_path_added': True
    }
    
    if not quiet:
        print("Hockey Prediction System - Enhanced Notebook Environment")
        print("=" * 55)
        print(f"Project root: {PATHS.root}")
        print(f"Paths configured: {paths_valid}")
        print(f"Component logging: {component} -> logs/{component}.log")
        print(f"Working directory: {Path.cwd()}")
        print(f"Python version: {sys.version.split()[0]}")
        print("\nReady to use! Access paths via: PATHS.data, PATHS.models, etc.")
        print("Enhanced logging active with per-component files.")
    
    return env_info


def quick_setup(component: str = 'notebooks') -> 'ProjectPaths':
    """
    Rychlý setup bez verbose outputu s enhanced logging.
    
    Args:
        component: Component name pro logging
        
    Returns:
        ProjectPaths: PATHS objekt
        
    Usage:
        from src.utils.notebook_helper import quick_setup
        PATHS = quick_setup('analysis')  # Creates logs/analysis.log
    """
    env = setup_notebook_environment(quiet=True, component=component)
    return env['PATHS']


def import_common_libraries(include_ml: bool = True, include_viz: bool = True) -> Dict[str, Any]:
    """
    Importuje běžně používané knihovny pro analýzu.
    
    Args:
        include_ml: Zda importovat ML knihovny
        include_viz: Zda importovat vizualizační knihovny
        
    Returns:
        dict: Slovník s importovanými moduly
        
    Usage:
        from src.utils.notebook_helper import import_common_libraries
        libs = import_common_libraries()
        pd = libs['pd']
        np = libs['np']
    """
    libraries = {}
    
    # Core data libraries
    try:
        import pandas as pd
        import numpy as np
        libraries.update({'pd': pd, 'np': np})
    except ImportError as e:
        print(f"Failed to import core libraries: {e}")
    
    # Visualization libraries
    if include_viz:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            libraries.update({'plt': plt, 'sns': sns})
            
            # Configure seaborn
            sns.set_style("whitegrid")
            sns.set_context("notebook", font_scale=1.1)
        except ImportError:
            print("Visualization libraries not available")
    
    # ML libraries
    if include_ml:
        try:
            from sklearn import metrics
            from sklearn.model_selection import train_test_split
            libraries.update({'metrics': metrics, 'train_test_split': train_test_split})
        except ImportError:
            print("ML libraries not available")
    
    # Date/time utilities
    try:
        from datetime import datetime, timedelta
        libraries.update({'datetime': datetime, 'timedelta': timedelta})
    except ImportError:
        pass
    
    return libraries


def setup_logging_for_notebook(notebook_name: str = "notebook", component: str = 'notebooks') -> 'logging.Logger':
    """
    Nastaví enhanced logging pro notebook s per-component support.
    
    Args:
        notebook_name: Název notebooku
        component: Component kategorie
        
    Returns:
        logging.Logger: Konfigurovaný logger
        
    Usage:
        from src.utils.notebook_helper import setup_logging_for_notebook
        logger = setup_logging_for_notebook("risk_analysis", "analysis")
        logger.info("Starting risk analysis...")
    """
    try:
        from config.logging_config import setup_logging, get_component_logger
        
        # Setup enhanced logging s per-component files
        setup_logging(
            log_level='INFO',
            log_to_file=True,
            component_files=True
        )
        
        # Vrať component-specific logger
        return get_component_logger(f"notebook.{notebook_name}", component)
        
    except ImportError:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
        return logging.getLogger(f"notebook.{notebook_name}")


def load_sample_data(data_type: str = "games", limit: int = 1000, **kwargs) -> 'pd.DataFrame':
    """
    Načte vzorková data pro rychlé testování s enhanced file handling.
    
    Args:
        data_type: Typ dat ('games', 'odds', etc.)
        limit: Počet řádků k načtení
        **kwargs: Additional parameters pro file loading
        
    Returns:
        pd.DataFrame: Načtená data
        
    Usage:
        from src.utils.notebook_helper import load_sample_data
        games = load_sample_data('games', limit=500)
    """
    import pandas as pd
    
    try:
        from config.paths import PATHS
        from src.utils.file_handlers import read_csv
        
        # Použij enhanced file handlers s automatic encoding detection
        data_file = PATHS.get_data_file(data_type, latest=True)
        
        # Safe loading s limit
        df = read_csv(data_file, nrows=limit, **kwargs)
        print(f"Loaded {len(df)} rows from {data_file.name} (safe encoding detection)")
        return df
        
    except Exception as e:
        print(f"Failed to load {data_type} data: {e}")
        # Return empty DataFrame s očekávanou strukturou
        if data_type == 'games':
            return pd.DataFrame(columns=['date', 'home_team', 'away_team', 'home_score', 'away_score'])
        else:
            return pd.DataFrame()


def export_notebook_results(
    data: 'pd.DataFrame', 
    filename: str, 
    export_type: str = "csv",
    **kwargs
) -> Path:
    """
    Exportuje výsledky z notebooku s enhanced file handling.
    
    Args:
        data: DataFrame k exportu
        filename: Název souboru (bez přípony)
        export_type: Typ exportu (csv, excel, parquet)
        **kwargs: Additional parameters
        
    Returns:
        Path: Cesta k exportovanému souboru
        
    Usage:
        results_path = export_notebook_results(df, "analysis_results", "csv")
    """
    try:
        from src.utils.file_handlers import save_processed_data, write_json
        from config.paths import PATHS
        
        if export_type.lower() == "csv":
            # Použij enhanced save_processed_data s automatic timestamp
            return save_processed_data(data, filename, **kwargs)
        
        elif export_type.lower() in ["excel", "xlsx"]:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_with_timestamp = f"{filename}_{timestamp}.xlsx"
            file_path = PATHS.processed_data / filename_with_timestamp
            
            data.to_excel(file_path, index=False, **kwargs)
            print(f"Exported to: {file_path}")
            return file_path
            
        else:
            raise ValueError(f"Unsupported export type: {export_type}")
        
    except Exception as e:
        print(f"Export failed: {e}")
        # Fallback to current directory
        fallback_path = Path(f"{filename}.{export_type}")
        if export_type.lower() == "csv":
            data.to_csv(fallback_path, index=False, encoding='utf-8', **kwargs)
        print(f"Exported to fallback location: {fallback_path}")
        return fallback_path


def create_performance_tracker(logger: 'logging.Logger') -> 'PerformanceLogger':
    """
    Vytvoří performance tracker pro notebook s enhanced logging.
    
    Args:
        logger: Logger instance
        
    Returns:
        PerformanceLogger: Performance tracker
        
    Usage:
        perf = create_performance_tracker(logger)
        perf.start_timer('data_processing')
        # ... processing ...
        perf.end_timer('data_processing')
    """
    try:
        from config.logging_config import PerformanceLogger
        return PerformanceLogger(logger)
    except ImportError:
        # Fallback simple timer
        class SimpleTimer:
            def __init__(self, logger):
                self.logger = logger
                self.timers = {}
            
            def start_timer(self, name):
                import time
                self.timers[name] = time.time()
                self.logger.info(f"Started: {name}")
            
            def end_timer(self, name):
                import time
                if name in self.timers:
                    duration = time.time() - self.timers[name]
                    self.logger.info(f"Completed: {name} ({duration:.2f}s)")
                    del self.timers[name]
                    return duration
                return 0.0
        
        return SimpleTimer(logger)


def notebook_setup_complete(
    project_root: Optional[str] = None,
    import_libraries: bool = True,
    sample_data: Optional[str] = None,
    quiet: bool = False,
    component: str = 'notebooks'
) -> Dict[str, Any]:
    """
    Kompletní enhanced setup pro notebook - paths, logging, libraries, sample data.
    
    Args:
        project_root: Cesta k root adresáři
        import_libraries: Zda importovat běžné knihovny
        sample_data: Typ vzorových dat k načtení
        quiet: Potlačí výstupní zprávy
        component: Component name pro logging
        
    Returns:
        dict: Kompletní enhanced prostředí pro notebook
        
    Usage:
        # V první buňce notebooku:
        from src.utils.notebook_helper import notebook_setup_complete
        env = notebook_setup_complete(sample_data='games', component='analysis')
        
        # Použití:
        PATHS = env['PATHS']
        logger = env['logger']
        pd = env['pd']
        plt = env['plt']
        games_df = env['sample_data']
        perf = env['performance_tracker']
    """
    # Setup enhanced environment
    env = setup_notebook_environment(project_root, quiet, component)
    
    # Import libraries
    if import_libraries:
        libs = import_common_libraries()
        env.update(libs)
    
    # Enhanced file handlers integration
    try:
        from src.utils.file_handlers import (
            read_csv, write_csv, read_json, write_json,
            load_latest_games_data, save_processed_data
        )
        env.update({
            'read_csv': read_csv,
            'write_csv': write_csv,
            'load_latest_games_data': load_latest_games_data,
            'save_processed_data': save_processed_data
        })
    except ImportError:
        if not quiet:
            print("Enhanced file handlers not available")
    
    # Performance tracking
    env['performance_tracker'] = create_performance_tracker(env['logger'])
    
    # Load sample data s enhanced handling
    if sample_data:
        try:
            sample_df = load_sample_data(sample_data, limit=1000)
            env['sample_data'] = sample_df
        except Exception as e:
            if not quiet:
                print(f"Could not load sample data: {e}")
    
    if not quiet:
        print("\nEnhanced notebook setup complete!")
        print(f"Available in 'env': {list(env.keys())}")
        print(f"Logger component: {component}")
        print("Enhanced file handling and performance tracking ready.")
    
    return env


if __name__ == "__main__":
    # Test notebook helper
    print("Testing notebook helper...")
    
    # Test basic setup
    env = setup_notebook_environment()
    print(f"Root: {env['project_root']}")
    
    # Test library import
    libs = import_common_libraries()
    print(f"Libraries: {list(libs.keys())}")
    
    # Test sample data loading
    try:
        df = load_sample_data('games', limit=10)
        print(f"Sample data shape: {df.shape}")
    except Exception as e:
        print(f"Sample data test failed: {e}")
