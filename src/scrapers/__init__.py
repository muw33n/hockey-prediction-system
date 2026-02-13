"""
Scrapers module for hockey prediction system.
Contains web scrapers for NHL and betting data.
"""

from src.scrapers.nhl_data_scraper import NHLScraper
from src.scrapers.betexplorer_scraper import BetExplorerScraper

__all__ = ['NHLScraper', 'BetExplorerScraper']
