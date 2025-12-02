# config_iron_condor.py
# Wide OTM Iron Condor Strategy for Sensex 1DTE
# Strategy: Range-bound, Low Volatility, High Win Rate

from dataclasses import dataclass
from datetime import time as dt_time
import os

@dataclass
class SensexIronCondorConfig:
    """
    Wide OTM Iron Condor Strategy for Sensex 1DTE
    - Designed for low volatility, rangebound days
    - One-time entry, no adjustments, no scaling
    - High win rate, low RR strategy
    - Perfect complement to directional credit spread
    """
    
    # ---- Environment ----
    PLACE_ORDERS: bool = False  # Set to True for live trading
    OPENALGO_API_KEY: str = os.environ.get('OPENALGO_API_KEY', '')
    OPENALGO_HOST: str = os.environ.get('OPENALGO_HOST', '')
    OPENALGO_WS: str = os.environ.get('OPENALGO_WS', '')
    
    # ---- Strategy ----
    STRATEGY_NAME: str = "SENSEX_IRON_CONDOR"
    STRATEGY_TYPE: str = "IRON_CONDOR"
    
    # ---- Underlying ----
    UNDERLYING_SYMBOL: str = "SENSEX"
    UNDERLYING_EXCHANGE: str = "BFO"  # BSE Futures & Options
    YAHOO_SYMBOL: str = "^BSESN"  # Yahoo Finance symbol for Sensex
    LOTSIZE: int = 10  # Sensex lot size
    
    # ---- Entry Timing (CRITICAL: Early morning only) ----
    ENTRY_START_TIME: dt_time = dt_time(9, 25)
    ENTRY_END_TIME: dt_time = dt_time(10, 15)
    
    # ---- Entry Filters ----
    # ATR for volatility check (only needs 2-3 candles, works at 9:25 AM)
    ATR_PERIOD: int = 14
    MAX_ATR_THRESHOLD: int = 250  # Skip if ATR > 250 (high volatility)
    
    # Gap Filter
    MAX_OPENING_GAP_POINTS: int = 200  # Skip if overnight gap > 200 points
    
    # NOTE: No ADX filter - ADX requires 14-20 candles to calculate
    # By 9:25-9:35 AM, we only have 2-4 candles available
    # Early morning + small gap + low ATR = rangebound assumption
    
    # ---- Strike Selection ----
    # Wide OTM strikes to reduce gamma risk
    # Use ATR-based or fixed distance from spot
    
    # Option 1: Fixed distance (points from spot)
    SELL_CE_DISTANCE: int = 350  # Sell CE at Spot + 350
    BUY_CE_DISTANCE: int = 650   # Buy CE at Spot + 650
    SELL_PE_DISTANCE: int = 350  # Sell PE at Spot - 350
    BUY_PE_DISTANCE: int = 650   # Buy PE at Spot - 650
    
    # Option 2: ATR-based (more adaptive)
    USE_ATR_BASED_STRIKES: bool = True
    SELL_STRIKE_ATR_MULTIPLE: float = 1.5  # Sell at Spot ± (1.5 * ATR)
    BUY_STRIKE_ATR_MULTIPLE: float = 2.5   # Buy at Spot ± (2.5 * ATR)
    
    # Strike rounding
    STRIKE_STEP: int = 100  # Round to nearest 100
    
    # ---- Position Sizing & Tranche Scaling ----
    # Tranche-based entry: Scale into position as premium expands
    ALLOW_MULTIPLE_ENTRIES: bool = True
    MAX_ENTRIES: int = 3  # Max 3 tranches
    ENTRY_LOTS: list = None  # Will be set to [3, 2, 1] in __post_init__
    
    # Tranche trigger: Enter next tranche when total IC premium increases
    TRANCHE_PREMIUM_INCREASE_PCT: float = 10.0  # 10% increase triggers next entry
    MIN_TRANCHE_GAP_MINUTES: int = 5  # Minimum 5 minutes between tranches
    NO_MORE_TRANCHES_AFTER: dt_time = dt_time(14, 0)  # No tranches after 2 PM
    
    # ---- Stop Loss ----
    # SL = 50% increase in total IC credit received
    # Example: Credit = ₹250, SL triggers when debit = ₹375
    SL_CREDIT_INCREASE_PCT: float = 50.0
    
    # Alternative SL: Total premium doubles
    USE_PREMIUM_DOUBLE_SL: bool = True
    SL_PREMIUM_MULTIPLIER: float = 2.0
    
    # ---- Target ----
    # Target = Capture 40-60% of credit
    # Example: Credit = ₹250, exit when remaining value = ₹100-150
    TARGET_MIN_PCT: float = 40.0
    TARGET_MAX_PCT: float = 60.0
    
    # ---- Exit Rules ----
    MANDATORY_EXIT_TIME: dt_time = dt_time(15, 0)  # Exit at 3:00 PM
    
    # ---- NO ADJUSTMENTS POLICY (Except Tranche Scaling) ----
    ALLOW_ADJUSTMENTS: bool = False  # No leg adjustments
    ALLOW_RE_ENTRY: bool = False  # No re-entry after exit
    # Note: ALLOW_MULTIPLE_ENTRIES = True for tranche scaling only
    
    # ---- Risk Management ----
    MAX_LOSS_PER_IC: int = 6000  # Maximum loss per Iron Condor (₹)
    EXPECTED_CREDIT: int = 250   # Expected credit per lot (₹)
    
    # ---- Indicator Timeframe ----
    TIMEFRAME: str = "5minute"
    INDICATOR_LOOKBACK: int = 3  # Only need 3 candles for ATR (not 20 for ADX)
    
    # ---- Logging ----
    DB_PATH: str = "./sensex_1dte_iron_condor.db"
    LOG_FILE: str = "./sensex_1dte_iron_condor.log"
    
    def __post_init__(self):
        """Initialize entry lots if not provided"""
        if self.ENTRY_LOTS is None:
            self.ENTRY_LOTS = [3, 2, 1]  # Scaled lot sizes for tranches

# Instance
config = SensexIronCondorConfig()
