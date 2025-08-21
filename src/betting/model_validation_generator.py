#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Model Validation Data Generator (NOVÉ ŘEŠENÍ)

Jednoduchý přístup: Použij BacktestingEngine jak je, jen "vypni" betting logiku
a exportuj všechny predictions místo value bets.

BacktestingEngine už správně načítá kurzy, takže jen převezmeme jeho výstup.

Location: src/betting/generate_model_validation_data.py
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import json

# Add project root to Python path (we're in src/betting/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

try:
    from betting.backtesting_engine import BacktestingEngine
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   Please ensure src directory structure is correct")
    sys.exit(1)


class ModelValidationDataGenerator:
    """
    Jednoduchý wrapper kolem BacktestingEngine pro model validation
    
    Přístup: Použij BacktestingEngine s parametry, které neudělají žádné sázky,
    pak exportuj všechny predictions jako validation dataset.
    """
    
    def __init__(self, elo_model_path: str = None, season: str = '2025', 
                 output_dir: Path = None):
        """
        Args:
            elo_model_path: Path to trained Elo model pickle
            season: Season to process (default '2025' = 2024-25)
            output_dir: Output directory for results
        """
        # Use BacktestingEngine directly
        model_path = elo_model_path or 'models/elo_model_trained_2024.pkl'
        self.engine = BacktestingEngine(elo_model_path=model_path, initial_bankroll=10000.0)
        
        self.season = season
        self.output_dir = output_dir or PROJECT_ROOT / 'models' / 'experiments'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"ModelValidationDataGenerator initialized")
        self.logger.info(f"Season: {season}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self):
        """Setup logging"""
        log_dir = PROJECT_ROOT / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('model_validation')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            file_handler = logging.FileHandler(
                log_dir / 'model_validation_generator.log', 
                encoding='utf-8'
            )
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(file_handler)
    
    def generate_validation_dataset(self) -> bool:
        """
        Hlavní metoda - použije BacktestingEngine s parametry, které neudělají sázky
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("🚀 Starting model validation dataset generation")
            self.logger.info("💡 Using BacktestingEngine with no-betting parameters")
            
            # 1. Load data using BacktestingEngine (toto už funguje správně)
            self.logger.info(f"📊 Loading data for season {self.season}...")
            load_result = self.engine.load_backtesting_data(season=self.season)
            
            self.logger.info(f"✅ Data loaded:")
            self.logger.info(f"   Games: {load_result['games_loaded']:,}")
            self.logger.info(f"   Odds records: {load_result['odds_records']:,}")
            self.logger.info(f"   Bookmakers: {load_result['unique_bookmakers']}")
            
            # 2. Run "backtesting" s parametry, které neudělají žádné sázky
            self.logger.info("🔄 Running BacktestingEngine with no-betting parameters...")
            
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
            
            # 3. Extrahuj predictions z results
            all_predictions = self._extract_predictions_from_results(results)
            
            if not all_predictions:
                self.logger.error("❌ No predictions found in BacktestingEngine results")
                return False
            
            self.logger.info(f"✅ Extracted {len(all_predictions)} predictions")
            
            # 4. Převeď predictions na validation format a exportuj
            return self._export_validation_data(all_predictions, results)
            
        except Exception as e:
            self.logger.error(f"❌ Error in dataset generation: {e}")
            return False
    
    def _extract_predictions_from_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extrahuj všechny predictions z BacktestingEngine results
        
        Args:
            results: BacktestingEngine results dictionary
            
        Returns:
            List of prediction dictionaries
        """
        all_predictions = []
        
        # BacktestingEngine ukládá predictions v gaming_day_results
        if 'gaming_day_results' not in results:
            self.logger.error("No gaming_day_results found in BacktestingEngine results")
            return []
        
        for day_result in results['gaming_day_results']:
            if 'predictions' in day_result:
                # Přidej date do každého prediction
                day_date = day_result['date']
                for prediction in day_result['predictions']:
                    prediction['date'] = day_date
                    all_predictions.extend([prediction])
        
        return all_predictions
    
    def _export_validation_data(self, predictions: List[Dict[str, Any]], 
                               full_results: Dict[str, Any]) -> bool:
        """
        Export predictions jako validation dataset s odds
        
        Args:
            predictions: List of prediction dictionaries
            full_results: Full BacktestingEngine results
            
        Returns:
            True if export successful
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Převeď predictions na DataFrame
            validation_df = pd.DataFrame(predictions)
            
            # Přidej odds data pro každou hru
            validation_df = self._enrich_with_odds_data(validation_df)
            
            # Přidej team names (pokud nejsou už tam)
            validation_df = self._enrich_with_team_names(validation_df)
            
            # Převeď na model validation format
            validation_df = self._convert_to_validation_format(validation_df)
            
            # Export paths
            csv_path = self.output_dir / f'model_validation_complete_{self.season}_{timestamp}.csv'
            json_path = self.output_dir / f'model_validation_summary_{self.season}_{timestamp}.json'
            
            # Export CSV
            validation_df.to_csv(csv_path, index=False, encoding='utf-8')
            self.logger.info(f"📁 Exported CSV: {csv_path}")
            
            # Generate summary
            summary_stats = self._calculate_validation_summary(validation_df)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            self.logger.info(f"📁 Exported JSON summary: {json_path}")
            
            # Display summary
            self._display_generation_summary(validation_df, summary_stats)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting validation data: {e}")
            return False
    
    def _enrich_with_odds_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Přidej odds data pro každou hru použitím BacktestingEngine.get_best_odds()
        """
        self.logger.info("💰 Enriching with odds data...")
        
        odds_data = []
        
        for _, row in df.iterrows():
            game_id = row['game_id']
            
            try:
                # Použij BacktestingEngine metodu (tato už funguje správně!)
                odds_info = self.engine.get_best_odds(game_id)
                
                if odds_info['has_odds']:
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
                
            except Exception as e:
                self.logger.warning(f"Error getting odds for game {game_id}: {e}")
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
        
        games_with_odds = result_df['has_odds_data'].sum()
        self.logger.info(f"💰 Odds data added: {games_with_odds}/{len(result_df)} games have odds")
        
        return result_df
    
    def _enrich_with_team_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Přidej team names pokud nejsou už tam"""
        
        if 'home_team_name' in df.columns:
            return df  # Already have team names
        
        self.logger.info("🏒 Adding team names...")
        
        # Get team names from BacktestingEngine's games_df
        games_df = self.engine.games_df[['id', 'home_team_name', 'away_team_name']].copy()
        games_df = games_df.rename(columns={'id': 'game_id'})
        
        # Merge with validation data
        result_df = df.merge(games_df, on='game_id', how='left')
        
        return result_df
    
    def _convert_to_validation_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Převeď BacktestingEngine predictions na model validation format
        """
        self.logger.info("🔄 Converting to validation format...")
        
        # Ensure required columns exist and rename if necessary
        column_mapping = {
            'actual_home_score': 'home_score',
            'actual_away_score': 'away_score',
            'actual_winner': 'actual_winner',
            'correct_prediction': 'prediction_correct'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Add model confidence (if not exists)
        if 'model_confidence' not in df.columns:
            df['model_confidence'] = df.apply(
                lambda row: abs(max(row['home_win_probability'], 
                                  row['away_win_probability']) - 0.5), 
                axis=1
            )
        
        # Add season and team_id columns if not exist
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
        result_df = df[available_columns].copy()
        
        self.logger.info(f"✅ Converted to validation format: {len(result_df)} records, {len(available_columns)} columns")
        
        return result_df
    
    def _calculate_validation_summary(self, validation_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate validation summary statistics"""
        
        # Basic metrics
        total_games = len(validation_df)
        correct_predictions = validation_df['prediction_correct'].sum() if 'prediction_correct' in validation_df.columns else 0
        accuracy = correct_predictions / total_games if total_games > 0 else 0
        
        # Odds coverage
        games_with_odds = validation_df['has_odds_data'].sum() if 'has_odds_data' in validation_df.columns else 0
        odds_coverage = games_with_odds / total_games if total_games > 0 else 0
        
        # Confidence analysis
        avg_confidence = validation_df['model_confidence'].mean() if 'model_confidence' in validation_df.columns else 0
        
        return {
            'generation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'season': self.season,
                'total_records': total_games,
                'method': 'BacktestingEngine_wrapper'
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
            }
        }
    
    def _display_generation_summary(self, validation_df: pd.DataFrame, summary_stats: Dict[str, Any]):
        """Display generation summary"""
        
        print("\n" + "="*80)
        print("🏒 MODEL VALIDATION DATASET - GENERATION COMPLETE")
        print("="*80)
        
        meta = summary_stats['generation_metadata']
        perf = summary_stats['model_performance']
        coverage = summary_stats['data_coverage']
        
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Total records: {meta['total_records']:,}")
        print(f"   Season: {meta['season']}")
        print(f"   Method: BacktestingEngine wrapper")
        
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
        
        print(f"\n✅ Ready for model validation analysis!")
        print("="*80)


def main():
    """Main execution function"""
    
    print("🏒 Hockey Prediction System - Model Validation Data Generator (NOVÉ ŘEŠENÍ)")
    print("="*80)
    print("💡 Využívá BacktestingEngine s no-betting parametry")
    print("💰 Kurzy se načítají automaticky z ověřené BacktestingEngine logiky")
    
    # Configuration
    MODEL_PATH = PROJECT_ROOT / 'models' / 'elo_model_trained_2024.pkl'
    SEASON = '2025'  # 2024-25 season
    
    # Validate model file exists
    if not MODEL_PATH.exists():
        print(f"❌ Model file not found: {MODEL_PATH}")
        print(f"   Please ensure the trained Elo model exists")
        return False
    
    try:
        # Initialize generator
        print("🔧 Initializing model validation generator...")
        generator = ModelValidationDataGenerator(
            elo_model_path=str(MODEL_PATH),
            season=SEASON
        )
        
        # Generate validation dataset
        print(f"🔄 Generating validation dataset for season {SEASON}...")
        success = generator.generate_validation_dataset()
        
        if success:
            print("\n✅ Model validation dataset generated successfully!")
            print("🎯 Ready for model_validation.ipynb analysis")
            print("💰 Kurzy jsou nyní správně načtené z BacktestingEngine!")
            return True
        else:
            print("\n❌ Dataset generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)