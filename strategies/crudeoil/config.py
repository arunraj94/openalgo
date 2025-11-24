# config.py
# Centralized configuration for the Unified Short Straddle Bot (MCX:CRUDEOILM)

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import time as dt_time
import os


@dataclass
class TimeWindow:
    start: dt_time
    end: dt_time


@dataclass
class Config:

	

    # ---- Environment ----
    PLACE_ORDERS: bool = False  # enable only when going live

    OPENALGO_API_KEY: str = os.environ.get('OPENALGO_API_KEY', '')   
    OPENALGO_HOST: str = os.environ.get('OPENALGO_HOST', 'http://34.14.208.231/')
    OPENALGO_WS: str = os.environ.get('OPENALGO_WS', 'ws://34.14.208.231/ws')

    # ---- Trading Windows (IST) ----
    # CRUDEOILM major volatility cycles
    ALLOWED_WINDOWS: List[Tuple[dt_time, dt_time]] = field(default_factory=lambda: [
        (dt_time(13, 0), dt_time(23, 30)),  # Continuous session
    ])

    # ---- Underlying ----
    UNDERLYING_SYMBOL: str = "CRUDEOILM"
    UNDERLYING_EXCHANGE: str = "MCX"
    LOTSIZE: int = 10

    # ---- ATR Settings ----
    ATR_SHORT_PERIOD: int = 3
    ATR_LONG_PERIOD: int = 20

    # ---- COMPRESSION Detector ----
    N_BODY_AVG: int = 10
    BODY_PCT_THRESHOLD: float = 0.55
    MIN_CONSEC: int = 1
    BB_LEN: int = 20
    BBWIDTH_LOOKBACK: int = 60
    BBWIDTH_PERCENTILE: int = 40      # 30% works better for CRUDE
    VWAP_DEV_PCT: float = 0.25        # ±0.25% around anchor price

    # ---- VWAP Reversion ----
    VWAP_REV_LOOKBACK: int = 10
    VWAP_REV_DEV_PCT: float = 0.5
    VWAP_REV_CLOSE_PCT: float = 0.25

    # ---- Range-Break Reversal ----
    RANGE_BREAK_LOOKBACK: int = 15
    RANGE_BREAK_REVERT_PCT: float = 0.15

    # ---- Trend Filter (ADX) ----
    ADX_PERIOD: int = 14
    ADX_THRESHOLD: int = 25  # Above this = Trending

    # ---- RSI Divergence ----
    RSI_DIV_LOOKBACK: int = 10
    RSI_PERIOD: int = 14  # RSI calculation period for mean reversion


    # ---- Directional Strategy ----
    EMA_FAST: int = 9
    EMA_SLOW: int = 21
    SUPERTREND_PERIOD: int = 10
    SUPERTREND_MULTIPLIER: int = 3
    TREND_LOOKBACK: int = 5

    # ---- Opening Range Breakout ----
    ORB_SESSIONS: List[Tuple[dt_time, dt_time]] = field(default_factory=lambda: [
        (dt_time(15, 0), dt_time(15, 30)),  # Afternoon session
        # (dt_time(9, 0), dt_time(9, 30)),  # Morning session (optional)
    ])
    ORB_RANGE_DURATION_MINUTES: int = 15  # First 15 min = range
    ORB_BREAKOUT_BUFFER_PCT: float = 0.1  # 0.1% buffer to confirm breakout
    ORB_MIN_RANGE_SIZE: float = 10.0  # Min range in points (avoid low vol)
    ORB_MAX_RANGE_SIZE: float = 100.0  # Max range in points (avoid gaps)
    ORB_BODY_STRENGTH_PCT: float = 0.6  # Candle body must be 60% of range

    # ---- US Session Proximity ----
    US_SESSION_START = dt_time(20, 0)
    US_PROXIMITY_MINUTES: int = 20

    # ---- Position Sizing ----
    ACCOUNT_CAPITAL: float = 1000000.0   # 10L
    MARGIN_PER_LEG: float = 30000.0      # approx
    LEGS_PER_STRADDLE: int = 2
    MAX_ALLOC_PER_TRADE_PCT: float = 0.1
    MAX_TOTAL_ALLOC_PCT: float = 0.8

    # Lots = K / ATR (dynamic sizing)
    K_ATR_SCALER: float = 500.0

    # ---- Stop-Loss Logic ----
    # Directional Strategy
    DIRECTIONAL_SPREAD_STRIKES = 2  # Width of the spread in strikes (e.g., 2 strikes away)
    
    # Time-variant SL% helps reduce noise
    TIME_BASE_SL_BUCKETS: List[Tuple[dt_time, dt_time, float]] = field(default_factory=lambda: [
        (dt_time(15, 0), dt_time(17, 0), 14.0),    # calm period
        (dt_time(17, 0), dt_time(19, 0), 17.0),    # pre-US choppiness
        (dt_time(19, 0), dt_time(21, 0), 22.0),    # US open volatility
        (dt_time(21, 0), dt_time(23, 30), 20.0),   # late session
    ])

    MARKET_START_TIME = dt_time(9, 15)
    MARKET_END_TIME = dt_time(23, 30)
    INTRADAY_SQUAREOFF_TIME = dt_time(23, 20)  # Force exit all positions

    ATR_MULTIPLIER_HIGH: float = 1.2  # ATR expanding rapidly
    ATR_MULTIPLIER_LOW: float = 0.8   # ATR extremely low

    TARGET_PCT_OF_CREDIT: float = 0.50
    TRAIL_FACTOR: float = 2.5
    TRAIL_MIN_BUFFER: float = 1.50
    
    # ---- Dynamic Trailing Based on Profit Level ----
    USE_DYNAMIC_TRAILING: bool = True  # Enable profit-based dynamic trailing
    # Profit thresholds and corresponding trail factors
    # Format: (min_profit_pct, trail_factor, min_buffer)
    DYNAMIC_TRAIL_LEVELS: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (0,   3.0, 2.0),   # 0-10% profit: Wide buffer (let it run)
        (10,  2.5, 1.5),   # 10-25% profit: Medium buffer
        (25,  2.0, 1.2),   # 25-40% profit: Tighter buffer
        (40,  1.5, 0.75),  # >40% profit: Tightest buffer (protect gains)
    ])

    # ---- Early Exit: no movement / no decay ----
    NO_MOVE_WAIT_MINUTES: int = 15
    MIN_DECAY_PCT: float = 3.0
    MOVE_THRESHOLD_MULT: float = 0.5

    # ---- Risk Control ----
    MAX_TRADES_PER_DAY: Optional[int] = None   # unlimited unless you set a number
    
    # Global Risk & Correlation
    MAX_DAILY_LOSS: float = 10000.0
    DAILY_PROFIT_LOCK: float = 25000.0
    CORRELATION_REDUCTION_FACTOR: float = 0.5 # Reduce size by 50% if portfolio is correlated

    # ML Signal Overlay
    ENABLE_ML_FILTER: bool = True
    ML_CONFIDENCE_THRESHOLD: float = 0.6
    ML_MODEL_PATH: str = "ml_model.pkl"

    # ---- Volatility Expansion (Option BUYING Strategy) ----
    # WARNING: Do NOT run this simultaneously with short straddle strategy
    # This strategy BUYS options during extreme compression (opposite of selling)
    VOLEXP_ENABLED: bool = False  # Set to True to enable (disabled by default)
    
    # Extreme compression detection (tighter than selling thresholds)
    VOLEXP_EXTREME_BBWIDTH_PERCENTILE: int = 10  # Bottom 10% (vs 25-40% for selling)
    VOLEXP_EXTREME_ATR_RATIO: float = 0.55  # ATR short/long (vs 0.7-0.8 for selling)
    VOLEXP_EXTREME_VWAP_DEV: float = 0.15  # ±0.15% (vs ±0.25% for selling)
    
    # Consolidation breakout detection
    VOLEXP_CONSOLIDATION_CANDLES: int = 8  # Minimum consecutive small-body candles
    VOLEXP_BODY_THRESHOLD: float = 0.30  # Body must be < 30% of average
    VOLEXP_N_BODY_AVG: int = 10  # Lookback for average body calculation
    
    # Pre-event timing (US open proximity)
    VOLEXP_PRE_EVENT_MINUTES_BEFORE: int = 30  # Start window 30min before event
    VOLEXP_PRE_EVENT_MINUTES_AFTER: int = 45  # End window 45min before event
    
    # Position sizing (conservative for option buying)
    VOLEXP_POSITION_SIZE_PCT: float = 0.10  # 10% of capital per trade
    VOLEXP_MAX_TRADES_PER_DAY: int = 2  # Maximum 2 trades per day
    
    # Profit targets and stop loss
    VOLEXP_TARGET_PREMIUM_PCT: float = 50.0  # 50% gain on premium paid
    VOLEXP_STOP_PREMIUM_PCT: float = -40.0  # -40% stop loss on premium
    
    # Time-based exit
    VOLEXP_TIME_EXIT_MINUTES: int = 30  # Exit if no meaningful move
    VOLEXP_MIN_UNDERLYING_MOVE_PCT: float = 0.20  # 0.20% minimum move required
    VOLEXP_SESSION_CLOSE_EXIT_TIME: dt_time = dt_time(23, 15)  # Auto-exit time

    # --------------------------------------------------------
    # Mean Reversion Scalper Settings
    # --------------------------------------------------------
    MEANREV_RSI_HIGH: int = 80
    MEANREV_RSI_LOW: int = 20
    MEANREV_BB_DEV: float = 2.5  # Wider bands for extremes
    MEANREV_TARGET_POINTS: float = 15.0
    MEANREV_STOP_POINTS: float = 10.0
    MEANREV_TIME_EXIT_MINUTES: int = 15  # Auto-exit time

    # ---- Database ----

    DB_PATH: str = "./trades.db"

    # ---- Logging ----
    VERBOSE: bool = True

    # Spread filters
    MAX_OPTION_SPREAD_PCT: float = 3.0     # % of option price (recommended 2.5–4%)
    MAX_PREMIUM_SKEW_PCT: float = 25.0      # CE vs PE premium difference limit


# Convenience: default config instance
default_config = Config()


if __name__ == "__main__":
    print("Default config:")
    print(default_config)