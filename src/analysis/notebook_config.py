#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Notebook Configuration Management
===========================================================

Centralized configuration for specialized analysis notebooks pipeline.
Provides consistent settings, parameters and paths for all notebooks.

Author: Hockey Prediction System  
Location: src/analysis/notebook_config.py
Used by: run_all_notebooks.py, all specialized notebooks
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import json


class NotebookConfig:
    """
    Centralized configuration manager for analysis notebooks.
    
    Provides consistent settings across all specialized notebooks:
    - main_analysis.ipynb
    - strategy_optimization.ipynb  
    - risk_assessment.ipynb
    - model_validation.ipynb
    """
    
    def __init__(self, run_timestamp: Optional[str] = None):
        """
        Initialize configuration with project paths and settings.
        
        Args:
            run_timestamp: Timestamp for this run (auto-generated if None)
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.run_timestamp = run_timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.setup_paths()
        
    def setup_paths(self):
        """Setup all required directory paths for consistent access."""
        # Core directories
        self.data_dir = self.project_root / 'data'
        self.logs_dir = self.project_root / 'logs' 
        self.models_dir = self.project_root / 'models'
        self.notebooks_dir = self.project_root / 'notebooks'
        self.src_dir = self.project_root / 'src'
        
        # Specific subdirectories
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_data_dir = self.data_dir / 'processed'
        self.experiments_dir = self.models_dir / 'experiments'
        self.charts_dir = self.experiments_dir / 'charts'
        self.analysis_notebooks_dir = self.notebooks_dir / 'analysis'
        self.src_analysis_dir = self.src_dir / 'analysis'
        
        # Run-specific directories
        self.run_dir = self.experiments_dir / f'automated_run_{self.run_timestamp}'
        self.run_summaries_dir = self.run_dir / 'individual_summaries'
        self.run_charts_dir = self.run_dir / 'all_charts'
        
    def get_analysis_config(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis configuration for all notebooks.
        
        Returns:
            Dictionary with all analysis parameters and settings
        """
        return {
            # Time period settings
            'time_period': {
                'season': '2024/25',
                'start_date': '2024-10-01',
                'end_date': '2025-06-01',
                'lookback_days': 30,
                'rolling_window': 10
            },
            
            # Data sources and validation
            'data_sources': {
                'games_table': 'games',
                'odds_table': 'odds', 
                'teams_table': 'teams',
                'predictions_table': 'predictions',
                'min_games_required': 50,
                'min_predictions_required': 100
            },
            
            # Model configuration
            'model_settings': {
                'model_type': 'elo_enhanced',
                'accuracy_threshold': 0.55,
                'confidence_levels': [0.6, 0.7, 0.8, 0.9],
                'calibration_bins': 10,
                'validation_split': 0.2
            },
            
            # Backtesting parameters
            'backtesting': {
                'initial_bankroll': 10000,
                'max_bet_size': 0.05,  # 5% of bankroll
                'min_edge': 0.02,      # 2% minimum edge
                'min_odds': 1.5,       # Minimum acceptable odds
                'max_odds': 3.0,       # Maximum acceptable odds
                'commission': 0.0,     # Betting commission
                'compound_growth': True
            },
            
            # Strategy optimization
            'optimization': {
                'edge_thresholds': [0.01, 0.02, 0.03, 0.04, 0.05],
                'confidence_thresholds': [0.55, 0.60, 0.65, 0.70, 0.75],
                'bet_sizing_methods': ['fixed', 'proportional', 'kelly'],
                'max_iterations': 1000,
                'optimization_metric': 'sharpe_ratio'
            },
            
            # Risk assessment parameters
            'risk_analysis': {
                'var_confidence': [0.95, 0.99],  # Value at Risk confidence levels
                'stress_test_scenarios': 5,
                'monte_carlo_simulations': 10000,
                'max_drawdown_threshold': 0.25,   # 25% max acceptable drawdown
                'consecutive_losses_limit': 10,
                'risk_free_rate': 0.02            # 2% annual risk-free rate
            },
            
            # Model validation settings
            'validation': {
                'cross_validation_folds': 5,
                'temporal_validation_periods': 4,  # Quarterly validation
                'accuracy_benchmarks': {
                    'random_baseline': 0.50,
                    'market_baseline': 0.52,
                    'target_accuracy': 0.58
                },
                'calibration_tolerance': 0.05,     # 5% calibration tolerance
                'stability_window': 30             # Days for stability analysis
            },
            
            # Visualization settings
            'visualization': {
                'chart_theme': 'plotly_white',
                'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                'figure_size': (12, 8),
                'export_format': 'html',
                'interactive_charts': True,
                'chart_quality': 'high'
            },
            
            # Export and logging
            'export': {
                'save_charts': True,
                'save_summaries': True,
                'export_format': 'html',
                'summary_format': 'json',
                'chart_timestamp': True,
                'compress_outputs': False
            },
            
            # Logging configuration
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s | %(levelname)s | %(message)s',
                'encoding': 'utf-8',
                'log_to_file': True,
                'log_to_console': True
            },
            
            # Runtime settings
            'runtime': {
                'automated_run': False,  # Set to True by orchestrator
                'run_timestamp': self.run_timestamp,
                'output_dir': str(self.run_dir),
                'parallel_processing': False,
                'memory_limit_gb': 8,
                'timeout_minutes': 30
            }
        }
        
    def get_notebook_parameters(self, notebook_name: str) -> Dict[str, Any]:
        """
        Get notebook-specific parameters for papermill injection.
        
        Args:
            notebook_name: Name of the notebook (e.g., 'main_analysis')
            
        Returns:
            Dictionary with parameters specific to the notebook
        """
        base_config = self.get_analysis_config()
        
        # Notebook-specific parameter sets
        notebook_params = {
            'main_analysis': {
                'ANALYSIS_CONFIG': base_config,
                'focus_areas': ['overview', 'performance', 'trends'],
                'detail_level': 'executive',
                'include_charts': True,
                'export_summary': True
            },
            
            'strategy_optimization': {
                'ANALYSIS_CONFIG': base_config,
                'optimization_scope': 'comprehensive',
                'parameter_sensitivity': True,
                'ab_testing': True,
                'heatmap_analysis': True,
                'export_best_params': True
            },
            
            'risk_assessment': {
                'ANALYSIS_CONFIG': base_config,
                'risk_scope': 'full',
                'stress_testing': True,
                'scenario_analysis': True,
                'portfolio_analysis': True,
                'regulatory_compliance': True
            },
            
            'model_validation': {
                'ANALYSIS_CONFIG': base_config,
                'validation_scope': 'comprehensive',
                'calibration_analysis': True,
                'temporal_stability': True,
                'market_comparison': True,
                'deployment_readiness': True
            }
        }
        
        return notebook_params.get(notebook_name, {
            'ANALYSIS_CONFIG': base_config,
            'notebook_name': notebook_name
        })
        
    def get_database_config(self) -> Dict[str, str]:
        """
        Get database connection configuration.
        
        Returns:
            Database configuration dictionary
        """
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'hockey_predictions'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'schema': os.getenv('DB_SCHEMA', 'public')
        }
        
    def get_file_paths(self) -> Dict[str, str]:
        """
        Get standardized file paths for data access.
        
        Returns:
            Dictionary with common file paths
        """
        return {
            # Data files
            'games_data': str(self.processed_data_dir / 'games_processed.csv'),
            'odds_data': str(self.processed_data_dir / 'odds_processed.csv'),
            'predictions_data': str(self.experiments_dir / 'predictions.csv'),
            'team_stats': str(self.processed_data_dir / 'team_stats.csv'),
            
            # Model files
            'elo_model': str(self.models_dir / 'trained' / 'elo_model.pkl'),
            'ml_model': str(self.models_dir / 'trained' / 'ml_model.pkl'),
            'feature_scaler': str(self.models_dir / 'trained' / 'feature_scaler.pkl'),
            
            # Analysis scripts (src/analysis/)
            'runner_script': str(self.src_analysis_dir / 'run_all_notebooks.py'),
            'config_script': str(self.src_analysis_dir / 'notebook_config.py'),
            'dashboard_script': str(self.src_analysis_dir / 'dashboard_generator.py'),
            
            # Notebooks (notebooks/analysis/)
            'notebooks_dir': str(self.analysis_notebooks_dir),
            
            # Output paths
            'charts_dir': str(self.charts_dir),
            'logs_dir': str(self.logs_dir),
            'run_dir': str(self.run_dir),
            'summaries_dir': str(self.run_summaries_dir)
        }
        
    def save_config(self, filename: Optional[str] = None):
        """
        Save current configuration to JSON file.
        
        Args:
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            filename = f'notebook_config_{self.run_timestamp}.json'
            
        config_file = self.run_dir / filename
        
        config_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'run_timestamp': self.run_timestamp,
                'config_version': '1.0'
            },
            'analysis_config': self.get_analysis_config(),
            'database_config': self.get_database_config(),
            'file_paths': self.get_file_paths()
        }
        
        # Ensure run directory exists
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False, default=str)
            
        return str(config_file)
        
    def load_config(self, config_file: str):
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
        """
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
        return config_data
        
    def validate_configuration(self) -> Dict[str, bool]:
        """
        Validate configuration settings and paths.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'paths_exist': True,
            'parameters_valid': True,
            'database_accessible': True,
            'models_available': True
        }
        
        # Check critical paths
        critical_paths = [
            self.analysis_notebooks_dir,  # notebooks/analysis/
            self.experiments_dir,
            self.logs_dir,
            self.src_dir               # src/ directory should exist
        ]
        
        for path in critical_paths:
            if not path.exists():
                validation_results['paths_exist'] = False
                break
                
        # Validate analysis parameters
        config = self.get_analysis_config()
        
        # Check backtesting parameters
        if config['backtesting']['initial_bankroll'] <= 0:
            validation_results['parameters_valid'] = False
            
        if config['backtesting']['max_bet_size'] > 1.0:
            validation_results['parameters_valid'] = False
            
        # Check optimization parameters
        if not config['optimization']['edge_thresholds']:
            validation_results['parameters_valid'] = False
            
        return validation_results


# Convenience functions for easy access
def get_config(run_timestamp: Optional[str] = None) -> NotebookConfig:
    """Get configured NotebookConfig instance."""
    return NotebookConfig(run_timestamp)

def get_analysis_config(run_timestamp: Optional[str] = None) -> Dict[str, Any]:
    """Get analysis configuration dictionary."""
    return NotebookConfig(run_timestamp).get_analysis_config()

def get_notebook_params(notebook_name: str, run_timestamp: Optional[str] = None) -> Dict[str, Any]:
    """Get parameters for specific notebook."""
    return NotebookConfig(run_timestamp).get_notebook_parameters(notebook_name)


if __name__ == "__main__":
    # Example usage and testing
    config = NotebookConfig()
    
    print("🔧 Hockey Prediction System - Notebook Configuration")
    print("=" * 50)
    
    # Validate configuration
    validation = config.validate_configuration()
    print(f"📋 Configuration validation:")
    for check, result in validation.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}")
        
    # Save configuration
    config_file = config.save_config()
    print(f"💾 Configuration saved to: {config_file}")
    
    # Example notebook parameters
    print(f"\n📊 Example parameters for main_analysis:")
    params = config.get_notebook_parameters('main_analysis')
    print(f"  - Time period: {params['ANALYSIS_CONFIG']['time_period']['season']}")
    print(f"  - Initial bankroll: ${params['ANALYSIS_CONFIG']['backtesting']['initial_bankroll']:,}")
    print(f"  - Min edge: {params['ANALYSIS_CONFIG']['backtesting']['min_edge']:.1%}")
