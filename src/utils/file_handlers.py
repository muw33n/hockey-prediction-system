#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Bezpečné čtení a zápis souborů (MIGRATED)
===================================================================
Řeší problémy s kódováním a zajišťuje konzistentní práci se soubory.
Používá centralized paths + component-specific logging.

Umístění: src/utils/file_handlers.py
"""

import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Any, List, Optional, Union
import chardet
import csv

# === MIGRACE: Centralized imports ===
from config.paths import PATHS
from config.logging_config import get_component_logger

# === MIGRACE: Component-specific logger pro utils ===
logger = get_component_logger(__name__, 'utils')


class FileHandler:
    """Centrální handler pro bezpečné čtení a zápis souborů"""
    
    # Podporovaná kódování (v pořadí priority)
    ENCODINGS = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1252', 'latin-1']
    
    @classmethod
    def detect_encoding(cls, filepath: Union[str, Path]) -> str:
        """
        Detekuje kódování souboru.
        
        Args:
            filepath: Cesta k souboru
            
        Returns:
            str: Detekované kódování
        """
        filepath = Path(filepath)
        
        # Načti první KB dat pro detekci
        with open(filepath, 'rb') as f:
            raw_data = f.read(1024)
        
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        
        logger.debug(f"Detected encoding for {filepath.name}: {encoding} (confidence: {confidence:.2%})")
        
        # Pokud je confidence nízká, zkus UTF-8
        if confidence < 0.7:
            logger.warning(f"Low confidence encoding detection for {filepath.name}, trying UTF-8")
            return 'utf-8'
        
        return encoding
    
    @classmethod
    def read_csv_safe(cls, 
                     filepath: Union[str, Path], 
                     encoding: Optional[str] = None,
                     **kwargs) -> pd.DataFrame:
        """
        Bezpečně načte CSV soubor s automatickou detekcí kódování.
        
        Args:
            filepath: Cesta k CSV souboru
            encoding: Explicitní kódování (pokud None, detekuje automaticky)
            **kwargs: Další parametry pro pd.read_csv
            
        Returns:
            pd.DataFrame: Načtená data
            
        Raises:
            ValueError: Pokud nelze načíst soubor žádným kódováním
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Pokud není specifikované kódování, zkus detekci
        if encoding is None:
            try:
                encoding = cls.detect_encoding(filepath)
                df = pd.read_csv(filepath, encoding=encoding, **kwargs)
                logger.info(f"Successfully loaded {filepath.name} with detected encoding: {encoding}")
                return df
            except Exception as e:
                logger.warning(f"Failed with detected encoding {encoding}: {e}")
        
        # Zkus různá kódování
        errors = []
        for enc in cls.ENCODINGS:
            try:
                df = pd.read_csv(filepath, encoding=enc, **kwargs)
                logger.info(f"Successfully loaded {filepath.name} with encoding: {enc}")
                return df
            except UnicodeDecodeError as e:
                errors.append(f"{enc}: {str(e)[:50]}")
                continue
            except Exception as e:
                errors.append(f"{enc}: {str(e)[:50]}")
                continue
        
        # Pokud nic nefunguje, vyhoď chybu
        error_msg = f"Cannot read {filepath} with any encoding. Tried: {', '.join(cls.ENCODINGS)}"
        logger.error(error_msg)
        for error in errors:
            logger.debug(f"  {error}")
        raise ValueError(error_msg)
    
    @classmethod
    def write_csv_safe(cls,
                      df: pd.DataFrame,
                      filepath: Union[str, Path],
                      encoding: str = 'utf-8',
                      **kwargs) -> None:
        """
        Bezpečně uloží DataFrame do CSV.
        
        Args:
            df: DataFrame k uložení
            filepath: Cesta k výstupnímu souboru
            encoding: Kódování (default: utf-8)
            **kwargs: Další parametry pro df.to_csv
        """
        filepath = Path(filepath)
        
        # Vytvoř adresář pokud neexistuje
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            df.to_csv(filepath, encoding=encoding, **kwargs)
            logger.info(f"Successfully saved {len(df)} rows to {filepath.name}")
        except Exception as e:
            logger.error(f"Failed to save CSV to {filepath}: {e}")
            raise
    
    @classmethod
    def read_json_safe(cls,
                      filepath: Union[str, Path],
                      encoding: Optional[str] = None) -> dict:
        """
        Bezpečně načte JSON soubor.
        
        Args:
            filepath: Cesta k JSON souboru
            encoding: Kódování (pokud None, detekuje automaticky)
            
        Returns:
            dict: Načtená data
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Detekuj kódování pokud není zadané
        if encoding is None:
            encoding = cls.detect_encoding(filepath)
        
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                data = json.load(f)
            logger.info(f"Successfully loaded JSON from {filepath.name}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to read JSON from {filepath}: {e}")
            raise
    
    @classmethod
    def write_json_safe(cls,
                       data: dict,
                       filepath: Union[str, Path],
                       encoding: str = 'utf-8',
                       indent: int = 2,
                       ensure_ascii: bool = False) -> None:
        """
        Bezpečně uloží data do JSON souboru.
        
        Args:
            data: Data k uložení
            filepath: Cesta k výstupnímu souboru
            encoding: Kódování
            indent: Odsazení pro pretty print
            ensure_ascii: Pokud False, zachová Unicode znaky
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding=encoding) as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            logger.info(f"Successfully saved JSON to {filepath.name}")
        except Exception as e:
            logger.error(f"Failed to save JSON to {filepath}: {e}")
            raise
    
    @classmethod
    def load_pickle_safe(cls, filepath: Union[str, Path]) -> Any:
        """
        Bezpečně načte pickle soubor.
        
        Args:
            filepath: Cesta k pickle souboru
            
        Returns:
            Any: Načtený objekt
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
            logger.info(f"Successfully loaded pickle from {filepath.name}")
            return obj
        except Exception as e:
            logger.error(f"Failed to load pickle from {filepath}: {e}")
            raise
    
    @classmethod
    def save_pickle_safe(cls, obj: Any, filepath: Union[str, Path]) -> None:
        """
        Bezpečně uloží objekt do pickle souboru.
        
        Args:
            obj: Objekt k uložení
            filepath: Cesta k výstupnímu souboru
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
            logger.info(f"Successfully saved pickle to {filepath.name}")
        except Exception as e:
            logger.error(f"Failed to save pickle to {filepath}: {e}")
            raise
    
    @classmethod
    def read_text_safe(cls,
                      filepath: Union[str, Path],
                      encoding: Optional[str] = None) -> str:
        """
        Bezpečně načte textový soubor.
        
        Args:
            filepath: Cesta k textovému souboru
            encoding: Kódování
            
        Returns:
            str: Obsah souboru
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if encoding is None:
            encoding = cls.detect_encoding(filepath)
        
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            logger.info(f"Successfully read text from {filepath.name}")
            return content
        except Exception as e:
            logger.error(f"Failed to read text from {filepath}: {e}")
            raise
    
    @classmethod
    def write_text_safe(cls,
                       content: str,
                       filepath: Union[str, Path],
                       encoding: str = 'utf-8') -> None:
        """
        Bezpečně uloží text do souboru.
        
        Args:
            content: Text k uložení
            filepath: Cesta k výstupnímu souboru
            encoding: Kódování
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding=encoding) as f:
                f.write(content)
            logger.info(f"Successfully wrote text to {filepath.name}")
        except Exception as e:
            logger.error(f"Failed to write text to {filepath}: {e}")
            raise


# === Convenience functions ===

def read_csv(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Zkratka pro bezpečné čtení CSV"""
    return FileHandler.read_csv_safe(filepath, **kwargs)


def write_csv(df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> None:
    """Zkratka pro bezpečné uložení CSV"""
    FileHandler.write_csv_safe(df, filepath, **kwargs)


def read_json(filepath: Union[str, Path], **kwargs) -> dict:
    """Zkratka pro bezpečné čtení JSON"""
    return FileHandler.read_json_safe(filepath, **kwargs)


def write_json(data: dict, filepath: Union[str, Path], **kwargs) -> None:
    """Zkratka pro bezpečné uložení JSON"""
    FileHandler.write_json_safe(data, filepath, **kwargs)


def load_model(filepath: Union[str, Path]) -> Any:
    """Zkratka pro načtení modelu"""
    return FileHandler.load_pickle_safe(filepath)


def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """Zkratka pro uložení modelu"""
    FileHandler.save_pickle_safe(model, filepath)


# === MIGRACE: Enhanced data loading helpers s PATHS ===

def load_latest_games_data(**kwargs) -> pd.DataFrame:
    """
    Načte nejnovější games data s použitím PATHS.
    
    Returns:
        pd.DataFrame: Games data
    """
    try:
        filepath = PATHS.get_data_file('games', latest=True)
        logger.info(f"Loading latest games data from: {filepath.name}")
        return read_csv(filepath, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load games data: {e}")
        raise


def load_latest_odds_data(**kwargs) -> pd.DataFrame:
    """
    Načte nejnovější odds data s použitím PATHS.
    
    Returns:
        pd.DataFrame: Odds data
    """
    try:
        filepath = PATHS.get_data_file('odds', latest=True) 
        logger.info(f"Loading latest odds data from: {filepath.name}")
        return read_csv(filepath, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load odds data: {e}")
        raise


def save_processed_data(df: pd.DataFrame, filename: str, **kwargs) -> Path:
    """
    Uloží zpracovaná data do processed_data adresáře.
    
    Args:
        df: DataFrame k uložení
        filename: Název souboru (bez přípony)
        **kwargs: Další parametry pro write_csv
        
    Returns:
        Path: Cesta k uloženému souboru
    """
    from datetime import datetime
    
    # Přidej timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_with_timestamp = f"{filename}_{timestamp}.csv"
    
    # Ulož do processed_data
    filepath = PATHS.processed_data / filename_with_timestamp
    
    write_csv(df, filepath, **kwargs)
    logger.info(f"Saved processed data to: {filepath}")
    
    return filepath


def load_model_safe(model_name: str, model_type: str = "pkl") -> Any:
    """
    Načte model s použitím PATHS a bezpečného loading.
    
    Args:
        model_name: Název modelu
        model_type: Typ souboru
        
    Returns:
        Any: Načtený model
    """
    try:
        filepath = PATHS.get_model_file(model_name, model_type)
        logger.info(f"Loading model from: {filepath.name}")
        
        if model_type == "pkl":
            return load_model(filepath)
        elif model_type == "json":
            return read_json(filepath)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise


def save_model_safe(model: Any, model_name: str, model_type: str = "pkl") -> Path:
    """
    Uloží model s použitím PATHS a bezpečného saving.
    
    Args:
        model: Model k uložení
        model_name: Název modelu
        model_type: Typ souboru
        
    Returns:
        Path: Cesta k uloženému modelu
    """
    try:
        filepath = PATHS.get_model_file(model_name, model_type)
        logger.info(f"Saving model to: {filepath.name}")
        
        if model_type == "pkl":
            save_model(model, filepath)
        elif model_type == "json":
            write_json(model, filepath)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to save model {model_name}: {e}")
        raise


# === Test funkce (MIGRATED) ===

if __name__ == "__main__":
    # === MIGRACE: Setup logging pomocí centralized systému ===
    from config.logging_config import setup_logging
    
    setup_logging(
        log_level='DEBUG',
        log_to_file=True,
        component_files=True
    )
    
    logger.info("Testing migrated file handlers...")
    
    # === MIGRACE: Test s PATHS integrací ===
    
    # Test detekce kódování na real data files
    try:
        games_file = PATHS.get_data_file('games', latest=True)
        logger.info(f"Testing encoding detection for: {games_file.name}")
        
        encoding = FileHandler.detect_encoding(games_file)
        logger.info(f"Detected encoding: {encoding}")
        
        # Test načtení CSV pomocí PATHS
        df = load_latest_games_data()
        logger.info(f"Loaded {len(df)} games, {len(df.columns)} columns")
        logger.info(f"Columns: {', '.join(df.columns[:5])}...")
        
    except FileNotFoundError:
        logger.warning("No games data file found, skipping test")
    except Exception as e:
        logger.error(f"Games data test failed: {e}")
    
    # Test JSON operations
    logger.info("Testing JSON operations...")
    test_data = {
        "team": "Utah Mammoth",
        "rating": 1523.5,
        "české_znaky": "Řeřicha žluťoučký"
    }
    
    # === MIGRACE: Použij PATHS pro test file ===
    test_json_path = PATHS.processed_data / "test_output.json"
    
    try:
        write_json(test_data, test_json_path)
        loaded_data = read_json(test_json_path)
        assert loaded_data == test_data
        logger.info("JSON write/read successful")
        
        # Cleanup
        test_json_path.unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"JSON test failed: {e}")
    
    # Test processed data saving
    try:
        import pandas as pd
        test_df = pd.DataFrame({
            'team': ['Team A', 'Team B'],
            'rating': [1500, 1600]
        })
        
        saved_path = save_processed_data(test_df, "test_data", index=False)
        logger.info(f"Test processed data saved to: {saved_path}")
        
        # Cleanup
        saved_path.unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"Processed data test failed: {e}")
    
    logger.info("Migrated file handler tests completed!")
    
    # Log summary
    logger.info("=" * 50)
    logger.info("MIGRATION SUMMARY:")
    logger.info("- Using component-specific logging (utils.log)")
    logger.info("- PATHS integration for automatic file discovery") 
    logger.info("- Enhanced helper functions with timestamp support")
    logger.info("- UTF-8 encoding by default")
    logger.info("- Backwards compatible API")
    logger.info("=" * 50)