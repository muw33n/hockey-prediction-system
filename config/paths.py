#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Centrální správa cest (Enhanced)
=========================================================
Jednotný systém pro správu všech cest v projektu.
Řeší problémy s nekonzistentními cestami napříč moduly.

Umístění: config/paths.py
"""

from pathlib import Path
import os
import sys
import re
from typing import List, Optional, Union, Dict
import logging


class ProjectPaths:
    """
    Centrální správa všech cest v projektu.
    
    Použití:
        from config.paths import PATHS
        
        # Přístup k cestám
        data_file = PATHS.raw_data / 'nhl_games_20250101_120000.csv'
        model_file = PATHS.trained_models / 'elo_model.pkl'
        log_file = PATHS.get_log_file('training')
    """
    
    def __init__(self):
        """Inicializace všech projektových cest"""
        # Najdi root adresář projektu
        self.root = self._find_project_root()
        
        # === Data paths ===
        self.data = self.root / 'data'
        self.raw_data = self.data / 'raw'
        self.processed_data = self.data / 'processed'
        self.odds_data = self.data / 'odds'
        self.external_data = self.data / 'external'
        
        # === Model paths ===
        self.models = self.root / 'models'
        self.trained_models = self.models / 'trained'
        self.experiments = self.models / 'experiments'
        self.model_charts = self.experiments / 'charts'
        
        # === Source code paths ===
        self.src = self.root / 'src'
        self.src_data = self.src / 'data'
        self.src_features = self.src / 'features'
        self.src_models = self.src / 'models'
        self.src_betting = self.src / 'betting'
        self.src_utils = self.src / 'utils'
        self.src_analysis = self.src / 'analysis'
        self.src_database = self.src / 'database'
        
        # === Notebook paths ===
        self.notebooks = self.root / 'notebooks'
        self.notebooks_eda = self.notebooks / 'eda'
        self.notebooks_modeling = self.notebooks / 'modeling'
        self.notebooks_analysis = self.notebooks / 'analysis'
        
        # === Config paths ===
        self.config = self.root / 'config'
        
        # === Other paths ===
        self.logs = self.root / 'logs'
        self.tests = self.root / 'tests'
        
        # === Files ===
        self.env_file = self.root / '.env'
        self.requirements_file = self.root / 'requirements.txt'
        self.readme_file = self.root / 'README.md'
        self.setup_file = self.root / 'setup.py'
        
        # Ensure src is in Python path
        if str(self.src) not in sys.path:
            sys.path.insert(0, str(self.src))
    
    def _find_project_root(self) -> Path:
        """
        Inteligentně najde root adresář projektu.
        Enhanced s lepší robustností a více strategiemi.
        
        Returns:
            Path: Cesta k root adresáři projektu
            
        Raises:
            RuntimeError: Pokud nelze najít root adresář
        """
        # Strategie 1: Environment variable (nejvyšší priorita)
        if 'HOCKEY_PROJECT_ROOT' in os.environ:
            root = Path(os.environ['HOCKEY_PROJECT_ROOT'])
            if root.exists() and self._is_valid_project_root(root):
                return root.resolve()
        
        # Strategie 2: Pokud jsme spouštěni jako modul
        if __package__:
            current = Path(__file__).resolve().parent
            while current != current.parent:
                if self._is_valid_project_root(current):
                    return current
                current = current.parent
        
        # Strategie 3: Hledej od aktuálního souboru nahoru
        current = Path(__file__).resolve().parent  # Start from config/
        for _ in range(10):  # Max 10 úrovní nahoru
            if self._is_valid_project_root(current):
                return current
            if current.parent == current:
                break
            current = current.parent
        
        # Strategie 4: Zkus z working directory
        cwd = Path.cwd()
        if self._is_valid_project_root(cwd):
            return cwd.resolve()
        
        # Strategie 5: Zkus parent adresáře z CWD
        current = cwd.parent
        for _ in range(5):
            if self._is_valid_project_root(current):
                return current.resolve()
            if current.parent == current:
                break
            current = current.parent
        
        raise RuntimeError(
            "Cannot find project root directory. "
            "Please ensure you're running from within the project "
            "or set HOCKEY_PROJECT_ROOT environment variable.\n"
            f"Searched from: {Path(__file__).parent}, {cwd}"
        )
    
    def _is_valid_project_root(self, path: Path) -> bool:
        """
        Ověří, zda je adresář validní root projektu.
        
        Args:
            path: Cesta k ověření
            
        Returns:
            bool: True pokud je validní root
        """
        if not path.exists():
            return False
            
        # Musí obsahovat requirements.txt nebo setup.py
        required_files = [
            path / 'requirements.txt',
            path / 'setup.py'
        ]
        
        if not any(f.exists() for f in required_files):
            return False
        
        # Musí obsahovat config/ nebo src/ adresář
        required_dirs = [
            path / 'config',
            path / 'src'
        ]
        
        if not any(d.exists() and d.is_dir() for d in required_dirs):
            return False
        
        return True
    
    def ensure_directories(self):
        """Vytvoří všechny potřebné adresáře, pokud neexistují"""
        directories = [
            self.data, self.raw_data, self.processed_data, self.odds_data,
            self.external_data, self.models, self.trained_models,
            self.experiments, self.model_charts, self.logs,
            self.src_utils, self.src_analysis, self.src_database, self.config,
            self.notebooks_eda, self.notebooks_modeling, self.notebooks_analysis,
            self.tests
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_latest_file(self, directory: Path, pattern: str) -> Path:
        """
        Najde nejnovější soubor podle vzoru v daném adresáři.
        
        Args:
            directory: Adresář k prohledání
            pattern: Vzor názvu souboru (např. 'nhl_games_*.csv')
            
        Returns:
            Path: Cesta k nejnovějšímu souboru
            
        Raises:
            FileNotFoundError: Pokud není nalezen žádný soubor
        """
        files = list(directory.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No files found matching {pattern} in {directory}")
        
        # Extrahuj timestamp a seřaď
        def extract_timestamp(filepath):
            match = re.search(r'_(\d{8}_\d{6})', str(filepath))
            return match.group(1) if match else '00000000_000000'
        
        latest_file = max(files, key=extract_timestamp)
        return latest_file
    
    def get_data_file(self, data_type: str, latest: bool = True) -> Union[Path, List[Path]]:
        """
        Získá cestu k datovému souboru.
        
        Args:
            data_type: Typ dat ('games', 'odds', 'standings', 'team_stats')
            latest: Pokud True, vrátí nejnovější soubor
            
        Returns:
            Path nebo List[Path]: Cesta k souboru nebo seznam cest
        """
        patterns = {
            'games': 'nhl_games_*.csv',
            'odds': 'nhl_odds_*.csv',
            'standings': 'nhl_standings_*.csv',
            'team_stats': 'nhl_team_stats_*.csv'
        }
        
        if data_type not in patterns:
            raise ValueError(f"Unknown data type: {data_type}. Available: {list(patterns.keys())}")
        
        directory = self.raw_data if data_type != 'odds' else self.odds_data
        
        if latest:
            return self.get_latest_file(directory, patterns[data_type])
        else:
            return list(directory.glob(patterns[data_type]))
    
    def get_log_file(self, log_name: str, with_timestamp: bool = True) -> Path:
        """
        Vytvoří cestu k log souboru s automatickým timestampem.
        
        Args:
            log_name: Název log souboru (bez přípony)
            with_timestamp: Zda přidat timestamp
            
        Returns:
            Path: Cesta k log souboru
        """
        if with_timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{log_name}_{timestamp}.log"
        else:
            filename = f"{log_name}.log"
        
        return self.logs / filename
    
    def get_model_file(self, model_name: str, model_type: str = "pkl") -> Path:
        """
        Vytvoří cestu k model souboru.
        
        Args:
            model_name: Název modelu
            model_type: Typ souboru (pkl, joblib, h5, onnx)
            
        Returns:
            Path: Cesta k model souboru
        """
        filename = f"{model_name}.{model_type}"
        return self.trained_models / filename
    
    def get_experiment_dir(self, experiment_name: str) -> Path:
        """
        Vytvoří adresář pro experiment a vrátí cestu.
        
        Args:
            experiment_name: Název experimentu
            
        Returns:
            Path: Cesta k experiment adresáři
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.experiments / f"{experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def get_chart_file(self, chart_name: str, chart_type: str = "png") -> Path:
        """
        Vytvoří cestu k chart souboru.
        
        Args:
            chart_name: Název grafu
            chart_type: Typ souboru (png, jpg, svg, pdf)
            
        Returns:
            Path: Cesta k chart souboru
        """
        filename = f"{chart_name}.{chart_type}"
        return self.model_charts / filename
    
    def find_files(self, pattern: str, directory: Optional[Path] = None) -> List[Path]:
        """
        Najde všechny soubory odpovídající vzoru.
        
        Args:
            pattern: Glob pattern (např. "*.csv", "model_*.pkl")
            directory: Adresář k prohledání (default: root)
            
        Returns:
            List[Path]: Seznam nalezených souborů
        """
        search_dir = directory or self.root
        return list(search_dir.rglob(pattern))
    
    def get_relative_path(self, file_path: Union[str, Path]) -> Path:
        """
        Konvertuje absolutní cestu na relativní vzhledem k root.
        
        Args:
            file_path: Cesta k souboru
            
        Returns:
            Path: Relativní cesta
        """
        file_path = Path(file_path)
        try:
            return file_path.relative_to(self.root)
        except ValueError:
            return file_path
    
    def __str__(self) -> str:
        """String reprezentace s přehledem cest"""
        return f"""
Hockey Prediction System - Project Paths
========================================
Root: {self.root}
Data: {self.data}
Models: {self.models}
Source: {self.src}
Logs: {self.logs}
Config: {self.config}
Notebooks: {self.notebooks}
"""
    
    def validate(self) -> Dict[str, any]:
        """
        Validuje existenci klíčových adresářů a souborů.
        
        Returns:
            dict: Slovník s výsledky validace
        """
        validation = {
            'root_exists': self.root.exists(),
            'requirements_exists': self.requirements_file.exists(),
            'setup_exists': self.setup_file.exists(),
            'env_exists': self.env_file.exists(),
            'data_dir_exists': self.data.exists(),
            'src_dir_exists': self.src.exists(),
            'config_dir_exists': self.config.exists(),
            'critical_dirs_missing': [],
            'warnings': []
        }
        
        # Kontrola kritických adresářů
        critical_dirs = [self.data, self.src, self.config, self.logs]
        for dir_path in critical_dirs:
            if not dir_path.exists():
                validation['critical_dirs_missing'].append(str(dir_path))
        
        # Kontrola doporučených adresářů
        recommended_dirs = [self.models, self.notebooks, self.tests]
        for dir_path in recommended_dirs:
            if not dir_path.exists():
                validation['warnings'].append(f"Recommended directory missing: {dir_path}")
        
        # Kontrola Python path
        if str(self.src) not in sys.path:
            validation['warnings'].append("Source directory not in Python path")
        
        validation['is_valid'] = (
            validation['root_exists'] and 
            (validation['requirements_exists'] or validation['setup_exists']) and
            len(validation['critical_dirs_missing']) == 0
        )
        
        return validation


# === Singleton instance ===
PATHS = ProjectPaths()


# === Helper functions ===
def setup_project_paths(verbose: bool = True) -> bool:
    """
    Setup funkce pro inicializaci projektových cest.
    
    Args:
        verbose: Zda vypisovat informace
        
    Returns:
        bool: True pokud je setup úspěšný
    """
    PATHS.ensure_directories()
    validation = PATHS.validate()
    
    if verbose:
        if not validation['is_valid']:
            print("⚠️ Project structure validation failed:")
            if not validation['root_exists']:
                print("  ❌ Root directory not found")
            if not validation['requirements_exists'] and not validation['setup_exists']:
                print("  ❌ requirements.txt or setup.py not found")
            if validation['critical_dirs_missing']:
                print(f"  ❌ Missing directories: {validation['critical_dirs_missing']}")
        else:
            print("✅ Project paths configured successfully")
            print(f"   Root: {PATHS.root}")
        
        # Warnings
        if validation['warnings']:
            print("\n⚠️ Warnings:")
            for warning in validation['warnings']:
                print(f"  • {warning}")
    
    return validation['is_valid']





if __name__ == "__main__":
    # Test paths configuration
    print(PATHS)
    
    # Validate structure
    validation = PATHS.validate()
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
    
    # Setup directories
    if input("\nCreate missing directories? (y/n): ").lower() == 'y':
        PATHS.ensure_directories()
        print("✅ Directories created")
    
    # Test getting latest files
    try:
        latest_games = PATHS.get_data_file('games', latest=True)
        print(f"\nLatest games file: {latest_games}")
    except FileNotFoundError as e:
        print(f"\n⚠️ {e}")
    
    # Test new methods
    print(f"\nLog file example: {PATHS.get_log_file('training')}")
    print(f"Model file example: {PATHS.get_model_file('elo_model')}")
    print(f"Chart file example: {PATHS.get_chart_file('accuracy_plot')}")