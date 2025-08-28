#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Model Validation Data Generator (MIGRATED)
===================================================================
Enhanced wrapper kolem BacktestingEngine pro model validation s:
- Per-component logging (betting component)
- Centralized paths via PATHS system
- Safe file handling s automatic encoding detection
- Performance monitoring pro kritické operace
- Robust error handling s detailed logging

Umístění: src/betting/model_validation_generator.py
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# === MIGRACE: Enhanced imports ===
from config.paths import PATHS
from config.logging_config import setup_logging, get_component_logger, PerformanceLogger
from src.utils.file_handlers import write_csv, write_json, read_json, save_processed_data

# === MIGRACE: Setup enhanced logging (jednou na začátku aplikace) ===
setup_logging(
    log_level='INFO',
    log_to_file=True,
    component_files=True  # Per-component log files
)

# === MIGRACE: Component-specific logger pro betting ===
logger = get_component_logger(__name__, 'betting')


class ModelValidationDataGenerator:
    """
    Enhanced wrapper kolem BacktestingEngine pro model validation.
    
    Přístup: Používá BacktestingEngine s parametry které neudělají žádné sázky,
    pak exportuje všechny predictions jako validation dataset.
    
    Enhanced features:
    - Centralized paths via PATHS
    - Per-component logging
    - Performance monitoring  
    - Safe file operations
    - Robust error handling
    """
    
    def __init__(self, 
                 elo_model_name: str = 'elo_model_trained_2024',
                 elo_model_type: str = 'pkl', 
                 season: str = '2025',
                 output_dir: Optional[str] = None):
        """
        Initialize enhanced validation generator.
        
        Args:
            elo_model_name: Name of Elo model (without extension)
            elo_model_type: Model file type (pkl, joblib, etc.)
            season: Season to process (default '2025' = 2024-25)
            output_dir: Output directory (None = use PATHS.experiments)
        """
        # === MIGRACE: Enhanced performance monitoring ===
        self.perf_logger = PerformanceLogger(logger)
        
        self.perf_logger.start_timer('initialization')
        
        # Store model info
        self.elo_model_name = elo_model_name
        self.elo_model_type = elo_model_type
        
        # === MIGRACE: Verify model exists pomocí PATHS ===
        try:
            model_path = PATHS.get_model_file(elo_model_name, elo_model_type)
            if not model_path.exists():
                raise FileNotFoundError(f"Elo model not found: {model_path}")
            logger.info(f"Verified Elo model exists: {model_path.name}")
        except Exception as e:
            logger.error(f"Failed to verify Elo model: {e}")
            raise ValueError(f"Cannot find Elo model: {elo_model_name}.{elo_model_type}")
        
        # === MIGRACE: Output directory pomocí PATHS ===
        if output_dir is None:
            self.output_dir = PATHS.experiments
        else:
            self.output_dir = PATHS.root / output_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.season = season
        
        # === MIGRACE: Import BacktestingEngine s enhanced error handling ===
        try:
            from betting.backtesting_engine import BacktestingEngine
            
            # Initialize BacktestingEngine with correct parameters
            self.engine = BacktestingEngine(
                elo_model_name=elo_model_name,
                elo_model_type=elo_model_type,
                initial_bankroll=10000.0
            )
            
            logger.info("BacktestingEngine initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import BacktestingEngine: {e}")
            logger.error("Please ensure betting module structure is correct")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize BacktestingEngine: {e}")
            raise
        
        # Log initialization summary
        logger.info("=" * 60)
        logger.info("Enhanced Model Validation Generator Initialized")
        logger.info(f"   Season: {season}")
        logger.info(f"   Elo model: {elo_model_name}.{elo_model_type}")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info(f"   Component logging: betting -> logs/betting.log")
        logger.info("=" * 60)
        
        self.perf_logger.end_timer('initialization')
    
    def generate_validation_dataset(self) -> bool:
        """
        Hlavní metoda - použije BacktestingEngine s no-betting parametry.
        Enhanced s performance monitoring a detailed logging.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.perf_logger.start_timer('complete_generation')
            
            logger.info("🚀 Starting enhanced model validation dataset generation")
            logger.info("💡 Using BacktestingEngine with no-betting parameters")
            
            # 1. Load data using BacktestingEngine
            logger.info(f"📊 Loading data for season {self.season}...")
            self.perf_logger.start_timer('data_loading')
            
            load_result = self.engine.load_backtesting_data(season=self.season)
            
            self.perf_logger.end_timer('data_loading')
            
            logger.info("✅ Data loaded successfully:")
            logger.info(f"   Games: {load_result['games_loaded']:,}")
            logger.info(f"   Odds records: {load_result['odds_records']:,}")
            logger.info(f"   Bookmakers: {load_result['unique_bookmakers']}")
            
            # 2. Run "backtesting" s no-betting parameters
            logger.info("🔄 Running BacktestingEngine with no-betting parameters...")
            self.perf_logger.start_timer('backtesting')
            
            # Parametry navržené tak, aby neudělaly žádné sázky:
            # - edge_threshold=99.0 (99% edge required - nemožné)
            # - min_odds=99.0 (minimální kurz 99.0 - nemožné)
            results = self.engine.run_backtest(
                edge_threshold=99.0,  # Žádné sázky nebudou mít 99% edge
                min_odds=99.0,        # Žádné kurzy nebudou 99.0+
                stake_method='fixed',
                stake_size=0.01,
                ev_method='basic'
            )
            
            self.perf_logger.end_timer('backtesting')
            
            # 3. Extract predictions z results
            logger.info("🔍 Extracting predictions from BacktestingEngine results...")
            self.perf_logger.start_timer('predictions_extraction')
            
            all_predictions = self._extract_predictions_from_results(results)
            
            self.perf_logger.end_timer('predictions_extraction')
            
            if not all_predictions:
                logger.error("❌ No predictions found in BacktestingEngine results")
                return False
            
            logger.info(f"✅ Extracted {len(all_predictions)} predictions")
            
            # 4. Převeď predictions na validation format a exportuj
            logger.info("📤 Exporting validation dataset...")
            success = self._export_validation_data(all_predictions, results)
            
            self.perf_logger.end_timer('complete_generation')
            
            if success:
                logger.info("✅ Model validation dataset generation completed successfully")
            else:
                logger.error("❌ Export phase failed")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error in dataset generation: {e}")
            logger.debug(f"Traceback: {e}", exc_info=True)
            return False
    
    def _extract_predictions_from_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extrahuj všechny predictions z BacktestingEngine results.
        Enhanced s detailed logging.
        
        Args:
            results: BacktestingEngine results dictionary
            
        Returns:
            List of prediction dictionaries
        """
        all_predictions = []
        
        # BacktestingEngine ukládá predictions v gaming_day_results
        if 'gaming_day_results' not in results:
            logger.error("No gaming_day_results found in BacktestingEngine results")
            return []
        
        gaming_days_processed = 0
        predictions_found = 0
        
        for day_result in results['gaming_day_results']:
            gaming_days_processed += 1
            
            if 'predictions' in day_result:
                day_date = day_result['date']
                day_predictions = day_result['predictions']
                
                # Přidej date do každého prediction
                for prediction in day_predictions:
                    prediction['date'] = day_date
                    all_predictions.append(prediction)
                
                predictions_found += len(day_predictions)
                
                if gaming_days_processed % 50 == 0:  # Progress logging
                    logger.info(f"Processed {gaming_days_processed} gaming days, "
                               f"{predictions_found} predictions so far...")
        
        logger.info(f"Extraction complete: {gaming_days_processed} days processed, "
                   f"{predictions_found} predictions extracted")
        
        return all_predictions
    
    def _export_validation_data(self, 
                               predictions: List[Dict[str, Any]], 
                               full_results: Dict[str, Any]) -> bool:
        """
        Export predictions jako validation dataset s enhanced file handling.
        
        Args:
            predictions: List of prediction dictionaries
            full_results: Full BacktestingEngine results
            
        Returns:
            True if export successful
        """
        try:
            self.perf_logger.start_timer('data_export')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Převeď predictions na DataFrame
            logger.info("🔄 Converting predictions to DataFrame...")
            validation_df = pd.DataFrame(predictions)
            
            # Přidej odds data pro každou hru
            logger.info("💰 Enriching with odds data...")
            self.perf_logger.start_timer('odds_enrichment')
            validation_df = self._enrich_with_odds_data(validation_df)
            self.perf_logger.end_timer('odds_enrichment')
            
            # Přidej team names (pokud nejsou už tam)
            logger.info("🏒 Enriching with team names...")
            validation_df = self._enrich_with_team_names(validation_df)
            
            # Převeď na model validation format
            logger.info("📋 Converting to validation format...")
            validation_df = self._convert_to_validation_format(validation_df)
            
            # === MIGRACE: Enhanced export paths pomocí PATHS ===
            csv_filename = f'model_validation_complete_{self.season}_{timestamp}.csv'
            json_filename = f'model_validation_summary_{self.season}_{timestamp}.json'
            
            csv_path = self.output_dir / csv_filename
            json_path = self.output_dir / json_filename
            
            # === MIGRACE: Safe CSV export ===
            logger.info("💾 Exporting CSV with safe file handling...")
            write_csv(validation_df, csv_path, index=False)
            logger.info(f"📄 CSV exported: {csv_path}")
            
            # Generate summary
            logger.info("📊 Calculating validation summary...")
            summary_stats = self._calculate_validation_summary(validation_df)
            
            # === MIGRACE: Safe JSON export ===
            logger.info("💾 Exporting JSON summary with safe file handling...")
            write_json(summary_stats, json_path)
            logger.info(f"📄 JSON summary exported: {json_path}")
            
            # Display summary
            self._display_generation_summary(validation_df, summary_stats)
            
            self.perf_logger.end_timer('data_export')
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting validation data: {e}")
            logger.debug(f"Export error traceback: {e}", exc_info=True)
            return False
    
    def _enrich_with_odds_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Přidej odds data pro každou hru pomocí BacktestingEngine.get_best_odds().
        Enhanced s progress tracking a error handling.
        """
        logger.info("💰 Enriching with odds data using BacktestingEngine...")
        
        odds_data = []
        games_processed = 0
        games_with_odds = 0
        
        total_games = len(df)
        
        for _, row in df.iterrows():
            games_processed += 1
            game_id = row['game_id']
            
            try:
                # Použij BacktestingEngine metodu (tato už funguje správně!)
                odds_info = self.engine.get_best_odds(game_id)
                
                if odds_info['has_odds']:
                    games_with_odds += 1
                    odds_record = {
                        'home_odds': float(odds_info['home_odd']),
                        'away_odds': float(odds_info['away_odd']),
                        'home_implied_prob': 1.0 / odds_info['home_odd'],
                        'away_implied_prob': 1.0 / odds_info['away_odd'],
                        'best_home_bookmaker': odds_info['home_bookmaker'],
                        'best_away_bookmaker': odds_info['away_bookmaker'],
                        'has_odds_data': True
                    }
                else:
                    odds_record = {
                        'home_odds': None,
                        'away_odds': None,
                        'home_implied_prob': None,
                        'away_implied_prob': None,
                        'best_home_bookmaker': None,
                        'best_away_bookmaker': None,
                        'has_odds_data': False
                    }
                
                odds_data.append(odds_record)
                
                # Progress logging každých 100 her
                if games_processed % 100 == 0:
                    logger.info(f"Odds enrichment progress: {games_processed}/{total_games} "
                               f"({games_processed/total_games:.1%})")
                
            except Exception as e:
                logger.warning(f"Error getting odds for game {game_id}: {e}")
                # Přidej prázdný záznam
                odds_data.append({
                    'home_odds': None,
                    'away_odds': None,
                    'home_implied_prob': None,
                    'away_implied_prob': None,
                    'best_home_bookmaker': None,
                    'best_away_bookmaker': None,
                    'has_odds_data': False
                })
        
        # Přidej odds data do DataFrame
        odds_df = pd.DataFrame(odds_data)
        result_df = pd.concat([df, odds_df], axis=1)
        
        odds_coverage = games_with_odds / total_games
        logger.info(f"💰 Odds enrichment complete: {games_with_odds}/{total_games} "
                   f"games have odds ({odds_coverage:.1%} coverage)")
        
        return result_df
    
    def _enrich_with_team_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Přidej team names pokud nejsou už tam"""
        
        if 'home_team_name' in df.columns:
            logger.info("🏒 Team names already present in data")
            return df  # Already have team names
        
        logger.info("🏒 Adding team names from BacktestingEngine...")
        
        try:
            # Get team names from BacktestingEngine's games_df
            games_df = self.engine.games_df[['id', 'home_team_name', 'away_team_name']].copy()
            games_df = games_df.rename(columns={'id': 'game_id'})
            
            # Merge with validation data
            result_df = df.merge(games_df, on='game_id', how='left')
            
            # Check merge success
            missing_names = result_df['home_team_name'].isna().sum()
            if missing_names > 0:
                logger.warning(f"⚠️ {missing_names} games missing team names after merge")
            else:
                logger.info("✅ All games have team names")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error enriching with team names: {e}")
            return df  # Return original df if enrichment fails
    
    def _convert_to_validation_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Převeď BacktestingEngine predictions na model validation format.
        Enhanced s detailed column mapping.
        """
        logger.info("🔄 Converting to validation format...")
        
        # Ensure required columns exist and rename if necessary
        column_mapping = {
            'actual_home_score': 'home_score',
            'actual_away_score': 'away_score',
            'actual_winner': 'actual_winner',
            'correct_prediction': 'prediction_correct'
        }
        
        # Apply column mapping
        df = df.rename(columns=column_mapping)
        
        # Add model confidence (if not exists)
        if 'model_confidence' not in df.columns:
            logger.info("📊 Calculating model confidence...")
            df['model_confidence'] = df.apply(
                lambda row: abs(max(row['home_win_probability'], 
                                  row['away_win_probability']) - 0.5), 
                axis=1
            )
        
        # Add season column if not exist
        if 'season' not in df.columns:
            df['season'] = self.season
        
        # Select and order columns for validation format
        validation_columns = [
            'game_id', 'date', 'season',
            'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name',
            'home_score', 'away_score', 'actual_winner',
            'predicted_winner', 'home_win_probability', 'away_win_probability',
            'rating_difference', 'home_rating', 'away_rating',
            'model_confidence', 'prediction_correct',
            'home_odds', 'away_odds', 'home_implied_prob', 'away_implied_prob',
            'best_home_bookmaker', 'best_away_bookmaker', 'has_odds_data'
        ]
        
        # Keep only columns that exist in df
        available_columns = [col for col in validation_columns if col in df.columns]
        missing_columns = [col for col in validation_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"⚠️ Some validation columns missing: {missing_columns}")
        
        result_df = df[available_columns].copy()
        
        logger.info(f"✅ Converted to validation format: {len(result_df)} records, "
                   f"{len(available_columns)} columns")
        
        return result_df
    
    def _calculate_validation_summary(self, validation_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate validation summary statistics s enhanced metrics"""
        
        logger.info("📊 Calculating comprehensive validation summary...")
        
        # Basic metrics
        total_games = len(validation_df)
        correct_predictions = validation_df['prediction_correct'].sum() if 'prediction_correct' in validation_df.columns else 0
        accuracy = correct_predictions / total_games if total_games > 0 else 0
        
        # Odds coverage
        games_with_odds = validation_df['has_odds_data'].sum() if 'has_odds_data' in validation_df.columns else 0
        odds_coverage = games_with_odds / total_games if total_games > 0 else 0
        
        # Confidence analysis
        avg_confidence = validation_df['model_confidence'].mean() if 'model_confidence' in validation_df.columns else 0
        
        # Enhanced metrics
        summary = {
            'generation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'season': self.season,
                'total_records': total_games,
                'method': 'Enhanced_BacktestingEngine_wrapper',
                'enhanced_features': [
                    'per_component_logging',
                    'centralized_paths',
                    'safe_file_handling',
                    'performance_monitoring'
                ]
            },
            'model_performance': {
                'overall_accuracy': accuracy,
                'correct_predictions': int(correct_predictions),
                'total_predictions': total_games,
                'average_confidence': avg_confidence
            },
            'data_coverage': {
                'games_with_odds': int(games_with_odds),
                'odds_coverage': odds_coverage,
                'date_range_start': validation_df['date'].min() if 'date' in validation_df.columns else None,
                'date_range_end': validation_df['date'].max() if 'date' in validation_df.columns else None
            },
            'export_info': {
                'output_directory': str(self.output_dir),
                'files_generated': [
                    'model_validation_complete_*.csv',
                    'model_validation_summary_*.json'
                ],
                'enhanced_infrastructure': 'PATHS + per_component_logging + safe_file_handlers'
            }
        }
        
        return summary
    
    def _display_generation_summary(self, 
                                  validation_df: pd.DataFrame, 
                                  summary_stats: Dict[str, Any]):
        """Display enhanced generation summary"""
        
        print("\n" + "="*80)
        print("🏒 MODEL VALIDATION DATASET - ENHANCED GENERATION COMPLETE")
        print("="*80)
        
        meta = summary_stats['generation_metadata']
        perf = summary_stats['model_performance']
        coverage = summary_stats['data_coverage']
        export_info = summary_stats['export_info']
        
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Total records: {meta['total_records']:,}")
        print(f"   Season: {meta['season']}")
        print(f"   Method: {meta['method']}")
        
        print(f"\n🎯 MODEL PERFORMANCE:")
        print(f"   Overall accuracy: {perf['overall_accuracy']:.1%}")
        print(f"   Correct predictions: {perf['correct_predictions']}/{perf['total_predictions']}")
        print(f"   Average confidence: {perf['average_confidence']:.3f}")
        
        print(f"\n💰 ODDS DATA COVERAGE:")
        print(f"   Games with odds: {coverage['games_with_odds']}")
        print(f"   Odds coverage: {coverage['odds_coverage']:.1%}")
        
        if coverage['date_range_start']:
            print(f"\n📅 DATE RANGE:")
            print(f"   From: {coverage['date_range_start']}")
            print(f"   To: {coverage['date_range_end']}")
        
        print(f"\n🚀 ENHANCED INFRASTRUCTURE:")
        for feature in meta['enhanced_features']:
            print(f"   ✅ {feature}")
        
        print(f"\n📁 EXPORT LOCATION:")
        print(f"   Directory: {export_info['output_directory']}")
        print(f"   Infrastructure: {export_info['enhanced_infrastructure']}")
        
        print(f"\n✅ Ready for enhanced model validation analysis!")
        print("="*80)


def main():
    """
    Enhanced main execution function s comprehensive error handling.
    """
    
    print("🏒 Hockey Prediction System - Enhanced Model Validation Generator")
    print("="*80)
    print("🚀 Enhanced features: per-component logging, safe file handling, performance monitoring")
    print("💡 Využívá BacktestingEngine s no-betting parametry")
    print("💰 Kurzy se načítají automaticky z ověřené BacktestingEngine logiky")
    
    try:
        # === MIGRACE: Configuration pomocí PATHS ===
        # Model configuration
        elo_model_name = 'elo_model_trained_2024'
        elo_model_type = 'pkl'
        
        # Verify model exists
        try:
            model_path = PATHS.get_model_file(elo_model_name, elo_model_type)
            if not model_path.exists():
                logger.error(f"❌ Model file not found: {model_path}")
                logger.error("Please ensure the trained Elo model exists")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to locate model via PATHS: {e}")
            return False
        
        # Season configuration
        season = '2025'  # 2024-25 season
        
        # Enhanced initialization
        logger.info("🔧 Initializing enhanced model validation generator...")
        generator = ModelValidationDataGenerator(
            elo_model_name=elo_model_name,
            elo_model_type=elo_model_type,
            season=season
        )
        
        # Generate validation dataset with enhanced features
        logger.info(f"🔄 Generating enhanced validation dataset for season {season}...")
        success = generator.generate_validation_dataset()
        
        if success:
            print("\n✅ Enhanced model validation dataset generated successfully!")
            print("🎯 Ready for enhanced model_validation.ipynb analysis")
            print("💰 Kurzy jsou nyní správně načtené z BacktestingEngine!")
            print("📊 Enhanced logging available in logs/betting.log")
            logger.info("✅ Enhanced model validation generation completed successfully")
            return True
        else:
            print("\n❌ Dataset generation failed")
            logger.error("❌ Enhanced dataset generation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in enhanced main execution: {e}")
        logger.debug(f"Main execution error traceback: {e}", exc_info=True)
        print(f"❌ Error in enhanced main execution: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)