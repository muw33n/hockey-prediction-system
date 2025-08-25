#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Bezpečné čtení a zápis souborů
=========================================================
Řeší problémy s kódováním a zajišťuje konzistentní práci se soubory.

Umístění: src/utils/file_handlers.py
"""

import pandas as pd
import json
import pickle
import logging
from pathlib import Path
from typing import Any, List, Optional, Union
import chardet
import csv

logger = logging.getLogger(__name__)


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
                logger.info(f"✅ Successfully loaded {filepath.name} with detected encoding: {encoding}")
                return df
            except Exception as e:
                logger.warning(f"Failed with detected encoding {encoding}: {e}")
        
        # Zkus různá kódování
        errors = []
        for enc in cls.ENCODINGS:
            try:
                df = pd.read_csv(filepath, encoding=enc, **kwargs)
                logger.info(f"✅ Successfully loaded {filepath.name} with encoding: {enc}")
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
            logger.info(f"✅ Successfully saved {len(df)} rows to {filepath.name}")
        except Exception as e:
            logger.error(f"❌ Failed to save CSV to {filepath}: {e}")
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
            logger.info(f"✅ Successfully loaded JSON from {filepath.name}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to read JSON from {filepath}: {e}")
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
            logger.info(f"✅ Successfully saved JSON to {filepath.name}")
        except Exception as e:
            logger.error(f"❌ Failed to save JSON to {filepath}: {e}")
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
            logger.info(f"✅ Successfully loaded pickle from {filepath.name}")
            return obj
        except Exception as e:
            logger.error(f"❌ Failed to load pickle from {filepath}: {e}")
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
            logger.info(f"✅ Successfully saved pickle to {filepath.name}")
        except Exception as e:
            logger.error(f"❌ Failed to save pickle to {filepath}: {e}")
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
            logger.info(f"✅ Successfully read text from {filepath.name}")
            return content
        except Exception as e:
            logger.error(f"❌ Failed to read text from {filepath}: {e}")
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
            logger.info(f"✅ Successfully wrote text to {filepath.name}")
        except Exception as e:
            logger.error(f"❌ Failed to write text to {filepath}: {e}")
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


# === Test funkce ===

if __name__ == "__main__":
    # Setup logging pro test
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s | %(message)s'
    )
    
    # Test detekce kódování
    test_file = Path("../../data/raw/nhl_games_20250101_120000.csv")
    if test_file.exists():
        print(f"\n🔍 Testing encoding detection for: {test_file.name}")
        encoding = FileHandler.detect_encoding(test_file)
        print(f"   Detected encoding: {encoding}")
        
        # Test načtení CSV
        print(f"\n📖 Testing CSV reading...")
        try:
            df = read_csv(test_file)
            print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            print(f"   Columns: {', '.join(df.columns[:5])}...")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    else:
        print(f"⚠️ Test file not found: {test_file}")
    
    # Test JSON
    print("\n📝 Testing JSON operations...")
    test_data = {
        "team": "Utah Mammoth",
        "rating": 1523.5,
        "české_znaky": "Řeřicha žluťoučký"
    }
    
    test_json_path = Path("test_output.json")
    try:
        write_json(test_data, test_json_path)
        loaded_data = read_json(test_json_path)
        assert loaded_data == test_data
        print("   ✅ JSON write/read successful")
        test_json_path.unlink()  # Clean up
    except Exception as e:
        print(f"   ❌ JSON test failed: {e}")
    
    print("\n✅ File handler tests completed!")