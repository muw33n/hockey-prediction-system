#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Centrální správa cest
================================================
Jednotný systém pro správu všech cest v projektu.
Řeší problémy s nekonzistentními cestami napříč moduly.

Umístění: config/paths.py
"""

from pathlib import Path
import os
import sys


class ProjectPaths:
    """
    Centrální správa všech cest v projektu.
    
    Použití:
        from config.paths import PATHS
        
        # Přístup k cestám
        data_file = PATHS.raw_data / 'nhl_games_20250101_120000.csv'
        model_file = PATHS.trained_models / 'elo_model.pkl'
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
        
        # Ensure src is in Python path
        if str(self.src) not in sys.path:
            sys.path.insert(0, str(self.src))
    
    def _find_project_root(self) -> Path:
        """
        Inteligentně najde root adresář projektu.
        
        Returns:
            Path: Cesta k root adresáři projektu
            
        Raises:
            RuntimeError: Pokud nelze najít root adresář
        """
        # Strategie 1: Pokud jsme spouštěni jako modul
        if __package__:
            current = Path(__file__).resolve().parent
            while current != current.parent:
                if (current / 'requirements.txt').exists():
                    return current
                current = current.parent
        
        # Strategie 2: Hledej od aktuálního souboru nahoru
        current = Path(__file__).resolve()
        for _ in range(10):  # Max 10 úrovní nahoru
            if (current / 'requirements.txt').exists():
                return current
            if current.parent == current:
                break
            current = current.parent
        
        # Strategie 3: Zkus z working directory
        cwd = Path.cwd()
        if (cwd / 'requirements.txt').exists():
            return cwd
        
        # Strategie 4: Environment variable
        if 'HOCKEY_PROJECT_ROOT' in os.environ:
            root = Path(os.environ['HOCKEY_PROJECT_ROOT'])
            if root.exists():
                return root
        
        raise RuntimeError(
            "Cannot find project root directory. "
            "Please ensure you're running from within the project "
            "or set HOCKEY_PROJECT_ROOT environment variable."
        )
    
    def ensure_directories(self):
        """Vytvoří všechny potřebné adresáře, pokud neexistují"""
        directories = [
            self.data, self.raw_data, self.processed_data, self.odds_data,
            self.external_data, self.models, self.trained_models,
            self.experiments, self.model_charts, self.logs,
            self.src_utils, self.src_analysis, self.config,
            self.notebooks_eda, self.notebooks_modeling, self.notebooks_analysis
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
        import glob
        import re
        
        files = list(directory.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No files found matching {pattern} in {directory}")
        
        # Extrahuj timestamp a seřaď
        def extract_timestamp(filepath):
            match = re.search(r'_(\d{8}_\d{6})', str(filepath))
            return match.group(1) if match else '00000000_000000'
        
        latest_file = max(files, key=extract_timestamp)
        return latest_file
    
    def get_data_file(self, data_type: str, latest: bool = True) -> Path:
        """
        Získá cestu k datovému souboru.
        
        Args:
            data_type: Typ dat ('games', 'odds', 'standings', 'team_stats')
            latest: Pokud True, vrátí nejnovější soubor
            
        Returns:
            Path: Cesta k souboru
        """
        patterns = {
            'games': 'nhl_games_*.csv',
            'odds': 'nhl_odds_*.csv',
            'standings': 'nhl_standings_*.csv',
            'team_stats': 'nhl_team_stats_*.csv'
        }
        
        if data_type not in patterns:
            raise ValueError(f"Unknown data type: {data_type}")
        
        directory = self.raw_data if data_type != 'odds' else self.odds_data
        
        if latest:
            return self.get_latest_file(directory, patterns[data_type])
        else:
            # Vrať všechny soubory
            return list(directory.glob(patterns[data_type]))
    
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
    
    def validate(self) -> dict:
        """
        Validuje existenci klíčových adresářů a souborů.
        
        Returns:
            dict: Slovník s výsledky validace
        """
        validation = {
            'root_exists': self.root.exists(),
            'requirements_exists': self.requirements_file.exists(),
            'env_exists': self.env_file.exists(),
            'data_dir_exists': self.data.exists(),
            'src_dir_exists': self.src.exists(),
            'critical_dirs_missing': []
        }
        
        critical_dirs = [self.data, self.src, self.config]
        for dir_path in critical_dirs:
            if not dir_path.exists():
                validation['critical_dirs_missing'].append(str(dir_path))
        
        validation['is_valid'] = (
            validation['root_exists'] and 
            validation['requirements_exists'] and
            len(validation['critical_dirs_missing']) == 0
        )
        
        return validation


# === Singleton instance ===
PATHS = ProjectPaths()


# === Helper functions ===
def setup_project_paths():
    """Setup funkce pro inicializaci projektových cest"""
    PATHS.ensure_directories()
    validation = PATHS.validate()
    
    if not validation['is_valid']:
        print("⚠️ Project structure validation failed:")
        if not validation['root_exists']:
            print("  ❌ Root directory not found")
        if not validation['requirements_exists']:
            print("  ❌ requirements.txt not found")
        if validation['critical_dirs_missing']:
            print(f"  ❌ Missing directories: {validation['critical_dirs_missing']}")
    else:
        print("✅ Project paths configured successfully")
        print(f"   Root: {PATHS.root}")
    
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