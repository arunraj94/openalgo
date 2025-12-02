# config_0dte_short_straddle.py
# Nifty 0 DTE Short Straddle with Whipsaw Protection
# Advanced naked straddle selling with progressive lock-in system

from dataclasses import dataclass, field
from datetime import time as dt_time
import os

@dataclass
class ShortStraddleConfig:
    """
    Nifty 0 DTE Short Straddle Configuration
    
    Strategy:
    - Sell ATM Call + ATM Put in range-bound markets
    - Multiple entry windows (10:00, 11:30, 1:45)
    - Whipsaw protection with progressive lock-in
    - Automatic leg management when one SL hits
    """
    
    # ---- Environment ----
    PLACE_ORDERS: bool = True  # ⚠️ REAL TRADING ENABLED ⚠️
    OPENALGO_API_KEY: str = os.environ.get('OPENALGO_API_KEY', '')
    OPENALGO_HOST: str = os.environ.get('OPENALGO_HOST', 'http://localhost:5000/')
    OPENALGO_WS: str = os.environ.get('OPENALGO_WS', 'ws://localhost:5000/ws')
    
    # ---- Strategy Info ----
    STRATEGY_NAME: str = "NIFTY_0DTE_SHORT_STRADDLE"
    STRATEGY_TYPE: str = "NAKED_STRADDLE"
    VERSION: str = "2.0_WHIPSAW_PROTECTED"
    
    # ---- Underlying ----
    UNDERLYING_SYMBOL: str = "NIFTY"
    UNDERLYING_EXCHANGE: str = "NFO"
    LOTSIZE: int = 75
    MAX_LOTS: int = 2  # Start conservative - adjust based on capital
    
    # ---- Expiry Validation ----
    EXPIRY_DAY: int = 3  # Thursday (0=Monday, 3=Thursday)
    SKIP_IF_NOT_EXPIRY: bool = True  # Only trade on expiry day
    
    # ---- Entry Windows ----
    # Extended windows for flexibility
    ENTRY_WINDOWS: list = field(default_factory=lambda: [
        {
            'name': 'WINDOW_1_MORNING',
            'start_time': dt_time(10, 0),
            'end_time': dt_time(10, 30),
            'enabled': True,
            'max_range': 60  # Max 60-point range in first 45 min
        },
        {
            'name': 'WINDOW_2_MIDDAY',
            'start_time': dt_time(11, 30),
            'end_time': dt_time(12, 30),
            'enabled': True,
            'max_range': 120  # Max 120-point morning range
        },
        {
            'name': 'WINDOW_3_AFTERNOON',
            'start_time': dt_time(13, 45),
            'end_time': dt_time(14, 30),
            'enabled': False,  # Disabled by default - risky close to expiry
            'max_range': 200  # Looser for afternoon
        }
    ])
    
    # ---- Entry Conditions (Indicators) ----
    USE_INDICATORS: bool = True
    TIMEFRAME: str = "5minute"
    
    # RSI (Sideways market)
    RSI_PERIOD: int = 14
    RSI_LOWER: float = 40.0
    RSI_UPPER: float = 60.0
    
    # ADX (Non-trending)
    ADX_PERIOD: int = 14
    ADX_MAX: float = 25.0  # Should be < 25 for entry
    
    # ATR (Volatility check)
    ATR_PERIOD: int = 14
    
    # Volume
    VOLUME_SPIKE_THRESHOLD: float = 2.0  # Avoid if vol > 2x average
    
    # ---- Entry Structure ----
    SELL_STRIKE_TYPE: str = "ATM"  # Sell exact ATM
    STRIKE_ROUNDING: int = 50  # Round to nearest 50
    
    # Minimum Premium Collection
    MIN_PREMIUM_COLLECTED: int = 120  # Min 120 points total to enter
    
    # ---- Risk Management ----
    
    # Individual Leg SL
    SL_PCT_PER_LEG: float = 40.0  # 40% loss on either Call or Put
    
    # Combined Position SL
    SL_PCT_COMBINED: float = 30.0  # 30% total premium loss
    
    # Profit Target
    PROFIT_TARGET_PCT: float = 50.0  # Book at 50% profit (consistent target)
    
    # ---- Whipsaw Protection ----
    ENABLE_WHIPSAW_PROTECTION: bool = True
    
    # Progressive Lock-In Rule (50-30-15 Rule)
    LOCK_IN_RULES: dict = field(default_factory=lambda: {
        'premium_gt_50': {'buffer_points': 10},  # If premium > 50, lock 10 pts
        'premium_30_50': {'buffer_points': 20},  # If 30-50, lock 20 pts
        'premium_20_30': {'buffer_points': 15},  # If 20-30, lock 15 pts
        'premium_lt_20': {'buffer_points': 10},  # If < 20, lock 10 pts
        'premium_lt_10': {'close_immediately': True}  # Close at market
    })
    
    # Trailing SL
    TRAIL_EVERY_POINTS: int = 10  # Update SL every 10-point decay
    TRAIL_ENABLE_AFTER_MINUTES: int = 15  # Start trailing after 15 min
    
    # Whipsaw Detection Signals
    WHIPSAW_SIGNALS: dict = field(default_factory=lambda: {
        'premium_rise_pct': 20.0,  # Close if premium rises 20%
        'reversal_candle_size': 40,  # Large opposite candle (points)
        'rsi_reversal_threshold': 25,  # RSI jumps 25+ points
        'volume_spike_multiplier': 2.0,  # Volume 2x average
        'time_cutoff': dt_time(14, 30)  # After 2:30 PM, close at any profit
    })
    
    # Leg Management Strategy
    LEG_MANAGEMENT_STRATEGY: str = "CONVERT_TO_NAKED"  # or "CLOSE_BOTH"
    # CONVERT_TO_NAKED: Keep winning leg, trail with lock-in (RECOMMENDED)
    # CLOSE_BOTH: Close both legs when one hits SL (conservative)
    
    # ---- Exit Rules ----
    MANDATORY_EXIT_TIME: dt_time = dt_time(15, 0)  # Force close by 3:00 PM
    PROFIT_EXIT_TIME: dt_time = dt_time(14, 30)  # If profitable at 2:30, book it
    AUTO_CLOSE_AT_EXPIRY: bool = True
    
    # ---- Position Limits ----
    MAX_ENTRIES_PER_DAY: int = 5  # Only 1 straddle per day (conservative)
    MAX_LOSS_PER_DAY: float = 10000.0  # Max ₹10,000 loss per day
    
    # ---- Market Timing ----
    MARKET_OPEN: dt_time = dt_time(9, 15)
    MARKET_CLOSE: dt_time = dt_time(15, 30)
    
    # ---- Monitoring ----
    MONITOR_FREQUENCY_SECONDS: int = 5  # Check positions every 5 sec
    LOG_CANDLE_COMPLETION: bool = True
    REPORT_PNL_EVERY_MINUTES: int = 15  # Log P&L every 15 min
    
    # ---- Logging & Database ----
    DB_PATH: str = "./nifty_expiry_straddle.db"
    LOG_FILE: str = "./logs/nifty_expiry/short_straddle.log"
    VERBOSE: bool = True
    
    # ---- Safety Overrides ----
    # Set to True during testing, False for live trading
    PAPER_TRADE_MODE: bool = False  # ⚠️ FALSE = REAL TRADING ⚠️
    REQUIRE_MANUAL_CONFIRMATION: bool = False  # Set True for extra safety

# Singleton instance
short_straddle_config = ShortStraddleConfig()

if __name__ == "__main__":
    print("=" * 70)
    print("NIFTY 0DTE SHORT STRADDLE CONFIGURATION")
    print("=" * 70)
    print(f"\n🎯 Strategy: {short_straddle_config.STRATEGY_NAME}")
    print(f"📦 Version: {short_straddle_config.VERSION}")
    print(f"📅 Expiry Day: {['Mon','Tue','Wed','Thu','Fri'][short_straddle_config.EXPIRY_DAY]}")
    print(f"💰 Lot Size: {short_straddle_config.LOTSIZE} x {short_straddle_config.MAX_LOTS} lots")
    
    print(f"\n⏰ Entry Windows:")
    for window in short_straddle_config.ENTRY_WINDOWS:
        status = "✅ ENABLED" if window['enabled'] else "❌ DISABLED"
        print(f"   {window['name']}: {window['start_time']} - {window['end_time']} {status}")
        print(f"      Max Range: {window['max_range']} points")
    
    print(f"\n📊 Entry Conditions:")
    print(f"   RSI: {short_straddle_config.RSI_LOWER} - {short_straddle_config.RSI_UPPER}")
    print(f"   ADX: < {short_straddle_config.ADX_MAX}")
    print(f"   Min Premium: {short_straddle_config.MIN_PREMIUM_COLLECTED} points")
    
    print(f"\n🛡️ Risk Management:")
    print(f"   Individual Leg SL: {short_straddle_config.SL_PCT_PER_LEG}%")
    print(f"   Combined SL: {short_straddle_config.SL_PCT_COMBINED}%")
    print(f"   Profit Target: {short_straddle_config.PROFIT_TARGET_PCT}%")
    print(f"   Max Daily Loss: ₹{short_straddle_config.MAX_LOSS_PER_DAY:,.0f}")
    
    print(f"\n🔄 Whipsaw Protection: {'ENABLED ✅' if short_straddle_config.ENABLE_WHIPSAW_PROTECTION else 'DISABLED'}")
    if short_straddle_config.ENABLE_WHIPSAW_PROTECTION:
        print(f"   Progressive Lock-In: Active (50-30-15 Rule)")
        print(f"   Trail Every: {short_straddle_config.TRAIL_EVERY_POINTS} points")
        print(f"   Trail After: {short_straddle_config.TRAIL_ENABLE_AFTER_MINUTES} minutes")
        print(f"   Leg Management: {short_straddle_config.LEG_MANAGEMENT_STRATEGY}")
    
    print(f"\n⚠️")
    print(f"⚠️  REAL TRADING MODE: {'ENABLED ⚠️⚠️⚠️' if short_straddle_config.PLACE_ORDERS else 'DISABLED (Paper)'}")
    print(f"⚠️  Paper Mode: {'YES (Safe)' if short_straddle_config.PAPER_TRADE_MODE else 'NO (LIVE MONEY!)'}")
    print(f"⚠️")
    
    if short_straddle_config.PLACE_ORDERS and not short_straddle_config.PAPER_TRADE_MODE:
        print(f"\n🚨 WARNING: This will place REAL orders with REAL money!")
        print(f"🚨 Ensure you understand the risks before running!")
        print(f"🚨 Start with small position sizes (1-2 lots)")
    
    print("\n" + "=" * 70)
