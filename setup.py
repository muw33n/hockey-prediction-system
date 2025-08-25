#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Setup Configuration
==============================================
Instalační skript pro správné fungování importů.

Umístění: hockey-prediction-system/setup.py (ROOT projektu)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Načti README pokud existuje
readme_file = Path("README.md")
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Načti requirements
requirements_file = Path("requirements.txt")
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="hockey-prediction-system",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered NHL betting prediction system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hockey-prediction-system",
    
    # Packages configuration
    packages=find_packages(include=["src", "src.*", "config", "config.*"]),
    package_dir={
        "": ".",  # Root directory je současný adresář
    },
    
    # Include non-Python files
    package_data={
        "config": ["*.json", "*.yaml", "*.yml"],
    },
    
    # Dependencies
    install_requires=requirements,
    
    # Python version
    python_requires=">=3.8",
    
    # Entry points (volitelné - pro spouštění z příkazové řádky)
    entry_points={
        "console_scripts": [
            "hockey-train=src.models.elo_rating_model:main",
            "hockey-backtest=src.betting.backtesting_engine:main",
            "hockey-db-setup=database_setup:main",
            "hockey-analyze=src.analysis.run_all_notebooks:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    
    # Additional metadata
    keywords="nhl hockey betting prediction machine-learning elo-rating",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/hockey-prediction-system/issues",
        "Source": "https://github.com/yourusername/hockey-prediction-system",
    },
)