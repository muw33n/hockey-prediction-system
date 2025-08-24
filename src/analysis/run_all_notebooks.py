#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Automated Analysis Runner
===================================================

Main orchestrator script for running complete specialized notebooks pipeline.
Executes all 4 analysis notebooks sequentially with consolidated reporting.

Author: Hockey Prediction System
Location: src/analysis/run_all_notebooks.py
Dependencies: papermill, nbconvert, pathlib
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    import papermill as pm
    import nbformat
    from nbconvert import HTMLExporter
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("📦 Install with: pip install papermill nbconvert nbformat")
    sys.exit(1)


class NotebookRunner:
    """
    Automated runner for specialized analysis notebooks pipeline.
    
    Executes sequence: main_analysis → strategy_optimization → risk_assessment → model_validation
    With consolidated reporting and error resilience.
    """
    
    def __init__(self):
        """Initialize runner with project paths and configuration."""
        self.project_root = Path(__file__).parent.parent.parent
        self.setup_paths()
        self.setup_logging()
        
        # Notebook execution sequence
        self.notebooks = [
            {
                'name': 'main_analysis',
                'filename': 'main_analysis.ipynb',
                'description': 'Core backtesting overview'
            },
            {
                'name': 'strategy_optimization', 
                'filename': 'strategy_optimization.ipynb',
                'description': 'Parameter sensitivity & A/B testing'
            },
            {
                'name': 'risk_assessment',
                'filename': 'risk_assessment.ipynb', 
                'description': 'Comprehensive risk analysis'
            },
            {
                'name': 'model_validation',
                'filename': 'model_validation.ipynb',
                'description': 'Prediction accuracy & calibration'
            }
        ]
        
        self.execution_stats = {
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'notebooks_executed': 0,
            'notebooks_failed': 0,
            'success_rate': 0.0,
            'errors': []
        }
        
    def setup_paths(self):
        """Setup all required directory paths."""
        # Notebooks jsou v notebooks/analysis/
        self.notebooks_dir = self.project_root / 'notebooks' / 'analysis'
        self.logs_dir = self.project_root / 'logs'
        self.experiments_dir = self.project_root / 'models' / 'experiments'
        self.charts_dir = self.experiments_dir / 'charts'
        
        # Ensure src/analysis/ exists for scripts
        self.src_analysis_dir = self.project_root / 'src' / 'analysis'
        self.src_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this run
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.experiments_dir / f'automated_run_{self.run_timestamp}'
        
        # Create output directories
        for directory in [self.logs_dir, self.run_dir, 
                         self.run_dir / 'individual_summaries',
                         self.run_dir / 'all_charts']:
            directory.mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self):
        """Setup UTF-8 encoded logging for detailed progress tracking."""
        log_file = self.logs_dir / 'automated_runner.log'
        execution_log = self.run_dir / 'execution_log.txt'
        
        # Configure logging with UTF-8 encoding
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.FileHandler(execution_log, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def validate_environment(self) -> bool:
        """
        Validate and create all required directories and components.
        
        Returns:
            bool: True if environment is ready, False otherwise
        """
        self.logger.info("🔍 Validating and preparing environment...")
        
        validation_results = {
            'notebooks_directory': self.notebooks_dir.exists(),
            'required_notebooks': True,
            'data_availability': True,
            'dependencies': True,
            'directories_created': True
        }
        
        # Create all required directories
        required_directories = [
            self.project_root / 'logs',
            self.project_root / 'models',
            self.project_root / 'models' / 'experiments', 
            self.project_root / 'models' / 'experiments' / 'charts',
            self.project_root / 'models' / 'trained',
            self.project_root / 'data',
            self.project_root / 'data' / 'processed',
            self.run_dir,
            self.run_dir / 'individual_summaries',
            self.run_dir / 'all_charts'
        ]
        
        self.logger.info("📁 Creating required directories...")
        for directory in required_directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"✅ Directory ready: {directory.relative_to(self.project_root)}")
            except Exception as e:
                self.logger.error(f"❌ Failed to create directory {directory}: {e}")
                validation_results['directories_created'] = False
        
        # Check all required notebooks exist
        missing_notebooks = []
        for notebook in self.notebooks:
            notebook_path = self.notebooks_dir / notebook['filename']
            if not notebook_path.exists():
                missing_notebooks.append(notebook['filename'])
                validation_results['required_notebooks'] = False
                
        if missing_notebooks:
            self.logger.error(f"❌ Missing notebooks: {missing_notebooks}")
            
        # Check basic data availability (warn but don't fail)
        data_dirs = [
            self.project_root / 'data' / 'processed',
            self.experiments_dir
        ]
        
        for data_dir in data_dirs:
            if not data_dir.exists():
                self.logger.warning(f"⚠️ Data directory empty: {data_dir}")
                # Don't fail validation for missing data - notebooks should handle this
        
        # Report validation results
        all_valid = all(validation_results.values())
        
        if all_valid:
            self.logger.info("✅ Environment validation and setup completed")
        else:
            self.logger.error("❌ Environment validation failed")
            for check, result in validation_results.items():
                status = "✅" if result else "❌"
                self.logger.error(f"  {status} {check}")
                
        # Write validation report
        validation_report = self.run_dir / 'data_validation_report.txt'
        with open(validation_report, 'w', encoding='utf-8') as f:
            f.write(f"Data Validation Report - {self.run_timestamp}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("DIRECTORY CREATION:\n")
            for directory in required_directories:
                status = "CREATED" if directory.exists() else "FAILED"
                f.write(f"  {directory.relative_to(self.project_root)}: {status}\n")
            
            f.write(f"\nVALIDATION CHECKS:\n")
            for check, result in validation_results.items():
                status = "PASS" if result else "FAIL"
                f.write(f"  {check}: {status}\n")
                
            if missing_notebooks:
                f.write(f"\nMissing notebooks: {missing_notebooks}\n")
                
        return all_valid
        
    def execute_notebook(self, notebook: Dict) -> Tuple[bool, Optional[str]]:
        """
        Execute single notebook with comprehensive error handling.
        
        Args:
            notebook: Notebook configuration dictionary
            
        Returns:
            Tuple of (success, error_message)
        """
        notebook_name = notebook['name']
        notebook_file = notebook['filename']
        description = notebook['description']
        
        self.logger.info(f"📊 Starting {notebook_name}: {description}")
        
        input_path = self.notebooks_dir / notebook_file
        output_path = self.run_dir / f"{notebook_name}_executed.ipynb"
        
        # Log paths for debugging
        self.logger.info(f"Input Notebook:  {input_path}")
        self.logger.info(f"Output Notebook: {output_path}")
        self.logger.info(f"Working directory: {self.project_root}")
        
        try:
            # Execute notebook with papermill
            start_time = datetime.now()
            
            # Set environment variables as fallback for notebooks
            import os
            os.environ['HOCKEY_PROJECT_ROOT'] = str(self.project_root)
            os.environ['HOCKEY_CHARTS_DIR'] = str(self.charts_dir)
            os.environ['HOCKEY_LOGS_DIR'] = str(self.logs_dir)
            os.environ['HOCKEY_RESULTS_DIR'] = str(self.experiments_dir)
            
            # Enhanced parameters for notebooks - use absolute paths
            notebook_parameters = {
                'automated_run': True,
                'run_timestamp': self.run_timestamp,
                'output_dir': str(self.run_dir),
                'project_root': str(self.project_root),
                'charts_dir': str(self.charts_dir),
                'logs_dir': str(self.logs_dir),
                # Add absolute paths that notebooks can use
                'RESULTS_DIR': str(self.experiments_dir),
                'CHARTS_EXPORT_DIR': str(self.charts_dir),
                'LOGS_DIR': str(self.logs_dir),
                'PROJECT_ROOT': str(self.project_root)
            }
            
            # Log environment setup for debugging
            self.logger.info(f"📁 Environment variables set:")
            self.logger.info(f"   HOCKEY_PROJECT_ROOT: {os.environ.get('HOCKEY_PROJECT_ROOT')}")
            self.logger.info(f"   HOCKEY_CHARTS_DIR: {os.environ.get('HOCKEY_CHARTS_DIR')}")
            self.logger.info(f"   HOCKEY_LOGS_DIR: {os.environ.get('HOCKEY_LOGS_DIR')}")
            
            pm.execute_notebook(
                input_path=str(input_path),
                output_path=str(output_path),
                parameters=notebook_parameters,
                cwd=str(self.project_root),
                progress_bar=False,
                log_output=True
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info(f"✅ {notebook_name} completed in {duration:.1f}s")
            
            # Copy any generated outputs
            self.collect_notebook_outputs(notebook_name)
            
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to execute {notebook_name}: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self.logger.error(f"📝 Stack trace: {traceback.format_exc()}")
            
            # Log additional debugging info
            self.logger.error(f"📁 Working directory exists: {self.project_root.exists()}")
            self.logger.error(f"📔 Input notebook exists: {input_path.exists()}")
            self.logger.error(f"📁 Output directory exists: {self.run_dir.exists()}")
            
            self.execution_stats['errors'].append({
                'notebook': notebook_name,
                'error': error_msg,
                'timestamp': datetime.now().isoformat(),
                'traceback': traceback.format_exc(),
                'paths': {
                    'input': str(input_path),
                    'output': str(output_path),
                    'working_dir': str(self.project_root)
                }
            })
            
            return False, error_msg
            
    def collect_notebook_outputs(self, notebook_name: str):
        """
        Collect outputs from executed notebook (charts, summaries) with improved error handling.
        
        Args:
            notebook_name: Name of the executed notebook
        """
        try:
            # Look for JSON summaries and HTML charts in charts directory
            summary_pattern = f"{notebook_name}_*summary*.json"
            chart_pattern = f"{notebook_name}_*.html"
            
            charts_source = self.charts_dir
            summaries_dest = self.run_dir / 'individual_summaries'
            charts_dest = self.run_dir / 'all_charts'
            
            # Ensure destination directories exist
            summaries_dest.mkdir(parents=True, exist_ok=True)
            charts_dest.mkdir(parents=True, exist_ok=True)
            
            collected_files = 0
            
            # Copy summary files
            if charts_source.exists():
                for summary_file in charts_source.glob(summary_pattern):
                    try:
                        dest_file = summaries_dest / summary_file.name
                        dest_file.write_text(summary_file.read_text(encoding='utf-8'), encoding='utf-8')
                        self.logger.info(f"📄 Collected summary: {summary_file.name}")
                        collected_files += 1
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to copy summary {summary_file.name}: {e}")
                        
                # Copy chart files  
                for chart_file in charts_source.glob(chart_pattern):
                    try:
                        dest_file = charts_dest / chart_file.name
                        dest_file.write_text(chart_file.read_text(encoding='utf-8'), encoding='utf-8')
                        self.logger.info(f"📈 Collected chart: {chart_file.name}")
                        collected_files += 1
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to copy chart {chart_file.name}: {e}")
                        
            if collected_files == 0:
                self.logger.info(f"ℹ️ No output files found for {notebook_name} (this is normal if notebook failed early)")
            else:
                self.logger.info(f"✅ Collected {collected_files} output files for {notebook_name}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to collect outputs for {notebook_name}: {e}")
            self.logger.warning(f"📁 Charts directory exists: {self.charts_dir.exists()}")
            self.logger.warning(f"📁 Charts directory path: {self.charts_dir}")
            
    def run_all_notebooks(self) -> bool:
        """
        Execute complete notebooks pipeline sequentially.
        
        Returns:
            bool: True if at least one notebook succeeded
        """
        self.logger.info("🚀 Starting automated notebooks pipeline")
        self.logger.info(f"📁 Output directory: {self.run_dir}")
        
        self.execution_stats['start_time'] = datetime.now()
        
        success_count = 0
        total_notebooks = len(self.notebooks)
        
        for i, notebook in enumerate(self.notebooks, 1):
            self.logger.info(f"📋 Progress: {i}/{total_notebooks} notebooks")
            
            success, error = self.execute_notebook(notebook)
            
            if success:
                success_count += 1
                self.execution_stats['notebooks_executed'] += 1
            else:
                self.execution_stats['notebooks_failed'] += 1
                
        self.execution_stats['end_time'] = datetime.now()
        self.execution_stats['total_duration'] = (
            self.execution_stats['end_time'] - self.execution_stats['start_time']
        ).total_seconds()
        self.execution_stats['success_rate'] = success_count / total_notebooks
        
        # Log final results
        self.logger.info(f"📊 Pipeline completed: {success_count}/{total_notebooks} notebooks succeeded")
        self.logger.info(f"⏱️ Total execution time: {self.execution_stats['total_duration']:.1f}s")
        self.logger.info(f"📈 Success rate: {self.execution_stats['success_rate']:.1%}")
        
        return success_count > 0
        
    def generate_execution_metadata(self):
        """Generate metadata file with execution statistics."""
        metadata = {
            'run_info': {
                'timestamp': self.run_timestamp,
                'start_time': self.execution_stats['start_time'].isoformat(),
                'end_time': self.execution_stats['end_time'].isoformat(),
                'duration_seconds': self.execution_stats['total_duration']
            },
            'execution_stats': {
                'total_notebooks': len(self.notebooks),
                'notebooks_executed': self.execution_stats['notebooks_executed'],
                'notebooks_failed': self.execution_stats['notebooks_failed'],
                'success_rate': self.execution_stats['success_rate']
            },
            'notebooks': self.notebooks,
            'errors': self.execution_stats['errors'],
            'output_structure': {
                'run_directory': str(self.run_dir),
                'summaries_directory': str(self.run_dir / 'individual_summaries'),
                'charts_directory': str(self.run_dir / 'all_charts')
            }
        }
        
        metadata_file = self.run_dir / 'execution_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
        self.logger.info(f"📄 Generated execution metadata: {metadata_file}")
        
    def run(self) -> int:
        """
        Main execution method with enhanced error reporting.
        
        Returns:
            int: Exit code (0 for success, 1 for failure)
        """
        try:
            self.logger.info("🤖 Hockey Prediction System - Automated Analysis Runner")
            self.logger.info("=" * 60)
            
            # Step 1: Environment validation and directory creation
            if not self.validate_environment():
                self.logger.error("❌ Environment validation failed. Aborting.")
                return 1
                
            # Step 2: Execute notebooks pipeline  
            pipeline_success = self.run_all_notebooks()
            
            # Step 3: Generate metadata
            self.generate_execution_metadata()
            
            # Step 4: Generate master dashboard regardless of success/failure
            try:
                # Import dashboard generator from same directory
                import sys
                dashboard_module_path = self.project_root / 'src' / 'analysis' / 'dashboard_generator.py'
                if dashboard_module_path.exists():
                    sys.path.insert(0, str(self.project_root / 'src' / 'analysis'))
                    from dashboard_generator import generate_master_dashboard
                    dashboard_path = generate_master_dashboard(str(self.run_dir), self.run_timestamp)
                    self.logger.info(f"📊 Master dashboard generated: {dashboard_path}")
                else:
                    self.logger.warning("⚠️ Dashboard generator not found - skipping dashboard creation")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to generate master dashboard: {e}")
                self.logger.warning("📊 You can generate it manually later if needed")
            
            # Step 5: Final status and recommendations
            if pipeline_success:
                success_rate = self.execution_stats['success_rate']
                notebooks_executed = self.execution_stats['notebooks_executed']
                
                if success_rate == 1.0:
                    self.logger.info("🎉 Automated pipeline completed successfully!")
                    self.logger.info("✅ All notebooks executed without errors")
                elif success_rate >= 0.5:
                    self.logger.info("🎉 Automated pipeline completed with partial success!")
                    self.logger.info(f"✅ {notebooks_executed}/{len(self.notebooks)} notebooks succeeded")
                    self.logger.info("💡 Check individual notebook logs for failed executions")
                else:
                    self.logger.warning("⚠️ Pipeline completed with limited success")
                    self.logger.warning(f"⚠️ Only {notebooks_executed}/{len(self.notebooks)} notebooks succeeded")
                    
                self.logger.info(f"📁 Results available in: {self.run_dir}")
                self.logger.info(f"📊 Master dashboard: {self.run_dir / 'master_dashboard.html'}")
                self.logger.info(f"📝 Execution log: {self.run_dir / 'execution_log.txt'}")
                self.logger.info(f"📋 View results: file://{(self.run_dir / 'master_dashboard.html').resolve()}")
                return 0
            else:
                self.logger.error("❌ Pipeline failed - no notebooks executed successfully")
                self.logger.error("🔍 Check environment setup and notebook dependencies")
                self.logger.info(f"📁 Partial results and logs available in: {self.run_dir}")
                return 1
                
        except Exception as e:
            self.logger.error(f"💥 Critical error in pipeline execution: {e}")
            self.logger.error(f"📝 Stack trace: {traceback.format_exc()}")
            return 1


def main():
    """Main entry point for automated runner."""
    runner = NotebookRunner()
    exit_code = runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
