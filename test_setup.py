#!/usr/bin/env python3
"""
Test script to verify the setup is working correctly.
"""

import sys
import os
from datetime import datetime

def test_imports():
    """Test if all required packages can be imported."""
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import xgboost as xgb
        import requests
        import psycopg2
        import sqlalchemy
        from dotenv import load_dotenv
        
        print("✅ All required packages imported successfully!")
        print(f"   - Pandas: {pd.__version__}")
        print(f"   - NumPy: {np.__version__}")
        print(f"   - Scikit-learn: {sklearn.__version__}")
        print(f"   - XGBoost: {xgb.__version__}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def test_environment():
    """Test environment variables."""
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['DATABASE_URL', 'DB_HOST', 'DB_NAME']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    else:
        print("✅ Environment variables loaded successfully!")
        return True

def test_database_connection():
    """Test database connection."""
    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()
        
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', 5432),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        
        cursor = conn.cursor()
        cursor.execute('SELECT version();')
        version = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        print("✅ Database connection successful!")
        print(f"   PostgreSQL version: {version[0]}")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def main():
    """Run all setup tests."""
    print("🏒 Hockey Prediction System - Setup Test")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Test time: {datetime.now()}")
    print()
    
    tests = [
        ("Package imports", test_imports),
        ("Environment variables", test_environment),
        ("Database connection", test_database_connection)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        result = test_func()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Setup completed successfully! Ready to start coding!")
    else:
        print("🔧 Some issues need to be resolved before continuing.")

if __name__ == "__main__":
    main()