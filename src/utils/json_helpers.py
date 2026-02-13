"""
JSON helpers - Conversion utilities for JSON serialization.

Handles numpy types, pandas objects, and datetime conversions for JSON.
Extracted from backtesting_engine.py for reuse across the project.

Location: src/utils/json_helpers.py
"""

import numpy as np
from datetime import date, datetime
from typing import Any, Dict, List, Union


def convert_for_json(obj: Any) -> Any:
    """
    Convert numpy types and pandas objects to JSON-serializable types.

    Args:
        obj: Object to convert (dict, list, numpy types, datetime, etc.)

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {convert_key_for_json(key): convert_for_json(value)
                for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif hasattr(obj, 'to_timestamp'):  # pandas Period
        return str(obj)
    elif hasattr(obj, 'isoformat'):  # Other datetime-like objects
        return obj.isoformat()
    else:
        return obj


def convert_key_for_json(key: Any) -> Union[str, int, float, bool]:
    """
    Convert dictionary keys to JSON-serializable format.

    Args:
        key: Dictionary key to convert

    Returns:
        JSON-serializable key (str, int, float, or bool)
    """
    if isinstance(key, str):
        return key
    elif isinstance(key, (int, float, bool)):
        return key
    elif hasattr(key, 'to_timestamp'):  # pandas Period
        return str(key)
    elif isinstance(key, (date, datetime)):
        return key.isoformat()
    else:
        return str(key)


def convert_numpy_types(data: Dict) -> Dict:
    """
    Recursively convert all numpy types in a dictionary to Python native types.

    Args:
        data: Dictionary potentially containing numpy types

    Returns:
        Dictionary with all numpy types converted to native Python types
    """
    return convert_for_json(data)


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float, returning default if conversion fails.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    try:
        if value is None or (hasattr(value, '__len__') and len(value) == 0):
            return default
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to int, returning default if conversion fails.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Integer value or default
    """
    try:
        if value is None:
            return default
        return int(value)
    except (ValueError, TypeError):
        return default
