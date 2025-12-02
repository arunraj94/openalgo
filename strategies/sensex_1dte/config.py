# config.py
# Directional Credit Spread Strategy for Sensex 1 DTE
# Strategy: Supertrend + ATR + ADX with Pivot Point Strike Selection

from dataclasses import dataclass
from datetime import time as dt_time
import os

@dataclass
class SensexCreditSpreadConfig:
    """
    Directional Credit Spread Strategy for Sensex 1DTE
    - Bull Put Spread (Bullish) / Bear Call Spread (Bearish)
    - Signal: Supertrend Direction
    - Filters: ATR (volatility), ADX (trend strength)
    - Strike Selection: Pivot Points (S1/R1) from Yahoo Finance
    """
    
    # ---- Environment ----
    PLACE_ORDERS: bool = False  # Set to True for live trading
    OPENALGO_API_KEY: str = os.environ.get('OPENALGO_API_KEY', '487218d8093857f95d8ac73b833b8d1dcd97e83ba36fb3dc70ef5091eacce2b8')
    OPENALGO_HOST: str = os.environ.get('OPENALGO_HOST', 'http://35.200.139.131/')
    OPENALGO_WS: str = os.environ.get('OPENALGO_WS', 'ws://35.200.139.131/ws')
    
    # ---- Strategy ----
    STRATEGY_NAME: str = "SENSEX_DIRECTIONAL_CREDIT_SPREAD"
    STRATEGY_TYPE: str = "CREDIT_SPREAD"
    
    # ---- Underlying ----
    UNDERLYING_SYMBOL: str = "SENSEX"
    UNDERLYING_EXCHANGE: str = "BFO"  # BSE Futures & Options
    YAHOO_SYMBOL: str = "^BSESN"  # Yahoo Finance symbol for Sensex
    LOTSIZE: int = 10  # Sensex lot size
    
    # ---- Entry Timing ----
    ENTRY_START_TIME: dt_time = dt_time(9, 35)
    ENTRY_END_TIME: dt_time = dt_time(14, 45)
    
    # ---- Indicators (5-minute timeframe) ----
    TIMEFRAME: str = "5minute"
    
    # Supertrend
    SUPERTREND_PERIOD: int = 10
    SUPERTREND_MULTIPLIER: float = 3.0
    
    # ATR
    ATR_PERIOD: int = 14
    
    # ADX
    ADX_PERIOD: int = 14
    ADX_MIN: float = 18.0  # Minimum ADX for entry
    ADX_STRONG: float = 25.0  # Strong trend threshold
    
    # ---- Strike Selection ----
    # Strikes are selected based on Pivot Points (S1 for BPS, R1 for BCS)
    SPREAD_WIDTH_MIN: int = 300  # Minimum spread width (points)
    SPREAD_WIDTH_MAX: int = 500  # Maximum spread width (points)
    
    # ---- Entry Filters ----
    NO_FLIP_CANDLES: int = 3  # No Supertrend flip in last N candles
    
    # ---- Multiple Entries (Scaling) ----
    ALLOW_MULTIPLE_ENTRIES: bool = True
    MAX_ENTRIES: int = 3
    ENTRY_LOTS: list = None  # Will be set to [3, 2, 1] in __post_init__
    MIN_ENTRY_GAP_MINUTES: int = 10  # Minimum gap between entries
    SCALING_ADX_MIN: float = 20.0  # ADX required for 2nd/3rd entry
    SCALING_ADX_STRONG: float = 25.0  # ADX for 3rd entry
    NO_SCALING_AFTER_TIME: dt_time = dt_time(12, 0)  # No new entries after 12 PM
    
    # ---- Stop Loss Rules ----
    SL_OPTION_A_PCT: float = 50.0  # 50% loss on spread premium
    SL_OPTION_B_MULTIPLIER: float = 2.0  # Spread value doubles
    
    # Use whichever hits first
    USE_SL_OPTION_A: bool = True
    USE_SL_OPTION_B: bool = True
    
    # ---- Target Rules ----
    TARGET_MIN_PCT: float = 40.0  # Book at 40-60% profit
    TARGET_MAX_PCT: float = 60.0
    
    # ---- Exit Rules ----
    MANDATORY_EXIT_TIME: dt_time = dt_time(15, 0)  # Exit all at 3:00 PM
    
    # ---- Re-entry After SL ----
    SL_COOLDOWN_MINUTES: int = 30  # Wait 30 minutes after SL hit
    
    # ---- Position Sizing ----
    INITIAL_LOTS: int = 3  # First entry
    
    # ---- Logging ----
    DB_PATH: str = "./sensex_1dte_credit_spread.db"
    LOG_FILE: str = "./sensex_1dte.log"
    
    def __post_init__(self):
        """Initialize entry lots if not provided"""
        if self.ENTRY_LOTS is None:
            self.ENTRY_LOTS = [3, 2, 1]

# Instance
config = SensexCreditSpreadConfig()
