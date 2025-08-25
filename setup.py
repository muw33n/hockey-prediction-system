#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Package Setup
=======================================
Enables pip install -e . for development installation.
Properly configures project as Python package.

Location: setup.py (root directory)
Usage: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read version from config/__init__.py or version file
def get_version():
    """Extract version from package files"""
    version_file = Path("config") / "settings.py"
    if version_file.exists():
        with open(version_file, 'r', encoding='utf-8') as f:
            content = f.read()
            version_match = re.search(r"VERSION\s*=\s*['\"]([^'\"]*)['\"]", content)
            if version_match:
                return version_match.group(1)
    
    # Fallback version
    return "0.1.0"

# Read long description from README
def get_long_description():
    """Read project description from README.md"""
    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Hockey Prediction System - NHL betting predictions with ML models"

# Read requirements from requirements.txt
def get_requirements():
    """Parse requirements.txt for dependencies"""
    requirements_path = Path("requirements.txt")
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
            # Filter out comments and empty lines
            requirements = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    requirements.append(line)
            return requirements
    
    # Fallback minimal requirements
    return [
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "psycopg2-binary>=2.9.0",
        "python-dotenv>=0.19.0",
        "requests>=2.27.0",
        "beautifulsoup4>=4.11.0"
    ]

setup(
    # === Basic package info ===
    name="hockey-prediction-system",
    version=get_version(),
    description="NHL hockey prediction system with ML models and value betting",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    # === Author info ===
    author="Hockey Prediction System",
    author_email="developer@hockeypredictions.com",
    url="https://github.com/your-username/hockey-prediction-system",
    
    # === Package discovery ===
    packages=find_packages(where="."),
    package_dir={"": "."},
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "config": ["*.py", "*.json", "*.yaml"],
        "data": ["*.csv", "*.json"],
        "": ["*.md", "*.txt", "*.yml", "*.yaml"]
    },
    
    # === Dependencies ===
    python_requires=">=3.8",
    install_requires=get_requirements(),
    
    # Optional dependencies for different use cases
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0"
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "ipywidgets>=7.6.0"
        ],
        "visualization": [
            "plotly>=5.0.0",
            "streamlit>=1.10.0",
            "dash>=2.5.0"
        ],
        "deployment": [
            "docker>=6.0.0",
            "gunicorn>=20.1.0",
            "uvicorn>=0.18.0"
        ]
    },
    
    # === Entry points ===
    entry_points={
        "console_scripts": [
            # Main prediction commands
            "hockey-predict=src.models.prediction_runner:main",
            "hockey-backtest=src.betting.backtesting_runner:main",
            "hockey-train=src.models.model_trainer:main",
            
            # Data collection commands
            "hockey-scrape=src.data.data_scraper:main",
            "hockey-update=src.data.data_updater:main",
            
            # Database management
            "hockey-db-setup=src.database.database_setup:main",
            "hockey-db-migrate=src.database.migrations:main",
            
            # Utility commands
            "hockey-validate=src.utils.model_validator:main",
            "hockey-export=src.utils.data_exporter:main",
        ]
    },
    
    # === Classification ===
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    
    # === Additional metadata ===
    keywords="hockey nhl prediction betting machine-learning elo-rating value-betting",
    project_urls={
        "Documentation": "https://github.com/your-username/hockey-prediction-system/wiki",
        "Source": "https://github.com/your-username/hockey-prediction-system",
        "Tracker": "https://github.com/your-username/hockey-prediction-system/issues",
    },
    
    # === Package configuration ===
    zip_safe=False,  # Allow access to package data
    
    # Platform compatibility
    platforms=["Windows", "Linux", "Mac OS-X"],
    
    # Minimum setuptools version
    setup_requires=["setuptools>=45.0.0"],
)