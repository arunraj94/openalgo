# config_crude_ratio_spread.py
# Protected Ratio Spread for Crude Oil - Evening Session (3:15 PM - 11:00 PM IST)
# DEFINED RISK - No unlimited loss through protective wings + strict stop losses

from dataclasses import dataclass, field
from typing import List, Tuple
from datetime import time as dt_time
import os


@dataclass
class CrudeRatioSpreadConfig:
    """
    Protected Ratio Spread Configuration for Crude Oil
    
    Key Features:
    - Defined Risk (protective wings + stop losses)
    - Evening session only (3:15 PM - 11:00 PM)
    - Maximum loss capped at percentage of capital
    - Phased entries (1:1 → 1:2 with protection)
    """
    
    # ---- Environment ----
    PLACE_ORDERS: bool = False
    OPENALGO_API_KEY: str = os.environ.get('OPENALGO_API_KEY', '')
    OPENALGO_HOST: str = os.environ.get('OPENALGO_HOST', 'http://localhost:5000/')
    OPENALGO_WS: str = os.environ.get('OPENALGO_WS', 'ws://localhost:5000/ws')
    
    # ---- Strategy Settings ----
    STRATEGY_NAME: str = "CRUDE_PROTECTED_RATIO_SPREAD"
    STRATEGY_TYPE: str = "PROTECTED_RATIO"  # NOT naked ratio
    
    # ---- Underlying ----
    UNDERLYING_SYMBOL: str = "CRUDEOIL"
    UNDERLYING_EXCHANGE: str = "MCX"
    LOTSIZE: int = 10
    
    # ---- Trading Hours (EVENING SESSION ONLY) ----
    SESSION_START_TIME: dt_time = dt_time(15, 15)  # 3:15 PM (after Nifty close)
    SESSION_END_TIME: dt_time = dt_time(23, 0)     # 11:00 PM (mandatory exit)
    
    # Entry windows within evening session
    ENTRY_WINDOW_START: dt_time = dt_time(15, 30)  # Wait 15min for volatility to settle
    ENTRY_WINDOW_END: dt_time = dt_time(20, 0)     # Last entry by 8 PM
    
    # CRITICAL: Exit before EIA report (Wednesday 8 PM)
    EIA_REPORT_DAY: int = 2  # 2 = Wednesday
    EIA_EXIT_TIME: dt_time = dt_time(19, 30)  # Exit 30min before EIA on Wed
    
    # ---- Position Structure (PROTECTED) ----
    # We use a "broken wing" ratio spread with protection
    MAX_RATIO: int = 2  # Maximum 1:2 (NOT 1:3 for safety)
    
    # Phase 1: Conservative 1:1 Credit Spread (DEFINED RISK)
    PHASE_1: dict = field(default_factory=lambda: {
        'name': 'PHASE_1_CREDIT_SPREAD',
        'time': dt_time(15, 30),
        'structure': {
            'buy_1': 'ATM',          # Buy 1 ATM
            'sell_1': 'ATM+50',      # Sell 1 ATM+50
        },
        'risk': 'DEFINED',           # Max loss = spread width
        'trigger': {
            'trend_confirmed': True,
            'min_move_from_315pm': 20,  # Crude moved 20+ points
        }
    })
    
    # Phase 2: Add ratio leg BUT with protection (STILL DEFINED RISK)
    PHASE_2: dict = field(default_factory=lambda: {
        'name': 'PHASE_2_PROTECTED_RATIO',
        'time': dt_time(18, 0),
        'structure': {
            'sell_1_more': 'ATM+50',   # Sell 1 more (now 1:2 ratio)
            'buy_protection': 'ATM+150',  # Buy far OTM protection (CRITICAL!)
        },
        'risk': 'DEFINED',  # Protected by far OTM wing
        'trigger': {
            'phase1_profit_pct': 15,   # Phase 1 must be +15%
            'spot_moved_points': (30, 80),  # 30-80 point move
            'trend_still_strong': True,
        }
    })
    
    # ---- Strike Selection (Crude Oil specific) ----
    STRIKE_SPACING: int = 50  # Crude oil strikes in 50-point increments
    
    # For Bullish (Call Ratio):
    CALL_BUY_STRIKE_OFFSET: int = 0       # ATM
    CALL_SELL_STRIKE_1_OFFSET: int = 50   # ATM+50
    CALL_SELL_STRIKE_2_OFFSET: int = 50   # ATM+50 (same)
    CALL_PROTECTION_OFFSET: int = 150     # ATM+150 (far OTM wing)
    
    # For Bearish (Put Ratio):
    PUT_BUY_STRIKE_OFFSET: int = 0        # ATM
    PUT_SELL_STRIKE_1_OFFSET: int = -50   # ATM-50
    PUT_SELL_STRIKE_2_OFFSET: int = -50   # ATM-50 (same)
    PUT_PROTECTION_OFFSET: int = -150     # ATM-150 (far OTM wing)
    
    # ---- Risk Management (CRITICAL - NO UNLIMITED LOSS) ----
    
    # Maximum Loss Limits
    MAX_LOSS_PERCENT: float = 3.0  # Max 3% loss per trade
    MAX_LOSS_RUPEES: float = 15000.0  # Or ₹15,000, whichever is lower
    
    # Stop Loss Levels (Multiple layers)
    STOP_LOSS_LAYERS: dict = field(default_factory=lambda: {
        # Layer 1: Quick stop if wrong
        'quick_sl_pct': 30,  # Exit if -30% of invested capital
        'quick_sl_time': 30,  # Within 30 minutes
        
        # Layer 2: Spot-based stop
        'spot_buffer_points': 80,  # Exit if spot breaches short + 80pts
        
        # Layer 3: Time-based trailing
        'trail_after_profit_pct': 20,  # Trail SL after +20% profit
        'trail_buffer_pct': 10,  # Trail 10% below high
        
        # Layer 4: Max total loss
        'absolute_max_loss': 15000,  # Never lose more than ₹15k
    })
    
    # ---- Exit Rules ----
    # Profit Targets
    TARGET_PROFIT_PCT: float = 50  # 50% of max potential
    PARTIAL_EXIT_AT_PCT: float = 30  # Book 50% at 30% profit
    
    # Time Exits
    EXIT_TIME: dt_time = dt_time(23, 0)  # 11:00 PM MANDATORY
    NO_ENTRY_AFTER: dt_time = dt_time(20, 0)  # 8:00 PM cutoff
    
    # EIA Report Exit (Wednesday Only)
    EXIT_BEFORE_EIA: bool = True
    
    # ---- Trend Detection ----
    USE_DIRECTIONAL_FILTERS: bool = True
    
    # EMA Settings
    EMA_FAST: int = 9
    EMA_SLOW: int = 21
    MIN_EMA_SEPARATION_POINTS: float = 15  # EMAs must be 15+ pts apart
    
    # ADX Settings
    ADX_PERIOD: int = 14
    ADX_THRESHOLD: float = 25  # Strong trend required
    
    # Supertrend
    SUPERTREND_PERIOD: int = 10
    SUPERTREND_MULTIPLIER: float = 3
    
    # ---- Position Sizing ----
    ACCOUNT_CAPITAL: float = 500000.0
    MAX_CAPITAL_PER_TRADE_PCT: float = 0.10  # 10% max (₹50,000)
    MARGIN_PER_SPREAD: float = 40000.0
    
    # Conservative sizing for evening session
    BASE_LOTS: int = 1
    MAX_LOTS: int = 2  # Never exceed 2 lots
    
    # ---- VIX / Volatility Filter ----
    # Crude oil doesn't have VIX, use ATR instead
    USE_ATR_FILTER: bool = True
    MIN_ATR_POINTS: float = 30  # Need at least 30 point ATR
    MAX_ATR_POINTS: float = 150  # Too volatile if >150
    
    # ---- Trade Management Rules ----
    
    # Monitoring Frequency
    CHECK_INTERVAL_SECONDS: int = 30  # Check every 30 seconds
    
    # Adjustment Rules
    ENABLE_ADJUSTMENTS: bool = True
    ADJUST_IF_SPOT_WITHIN: int = 30  # Adjust if spot within 30pts of short
    
    # Rolling Logic
    ROLL_THREATENED_LEG: bool = True
    ROLL_BUFFER_POINTS: int = 50  # Roll if spot within 50pts
    
    # ---- Blackout Periods ----
    BLACKOUT_EVENTS: List[str] = field(default_factory=lambda: [
        'EIA_REPORT',     # Wednesday 8 PM
        'OPEC_MEETING',   # Manual flag
        'GEOPOLITICAL',   # Manual flag
    ])
    
    # ---- Logging ----
    DB_PATH: str = "./crude_ratio_trades.db"
    VERBOSE: bool = True
    LOG_EACH_PHASE: bool = True


# Instance
crude_ratio_config = CrudeRatioSpreadConfig()


# ========================================
# TRADE MANAGEMENT LOGIC SUMMARY
# ========================================

"""
🎯 PROTECTED RATIO SPREAD - TRADE MANAGEMENT LOGIC

┌─────────────────────────────────────────────────────────────┐
│  ENTRY FLOW (3:15 PM - 8:00 PM)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  3:15 PM: Market opens (Nifty closed, evening session)    │
│  3:30 PM: Check for trend (EMA, ADX, Supertrend)          │
│           ↓                                                 │
│  ✅ Trend Confirmed → Enter Phase 1                        │
│           ↓                                                 │
│  PHASE 1 (3:30 PM):                                        │
│  ├─ BUY 1 lot ATM Call/Put                                │
│  └─ SELL 1 lot ATM+50 Call/Put                            │
│     → 1:1 Credit Spread (DEFINED RISK)                     │
│     → Max Loss: ₹5,000 (spread width - premium)           │
│           ↓                                                 │
│  6:00 PM: Check Phase 1 P&L                                │
│           ├─ If +15% profit → Proceed to Phase 2          │
│           └─ If negative → Exit, don't scale              │
│           ↓                                                 │
│  PHASE 2 (6:00 PM):                                        │
│  ├─ SELL 1 more lot ATM+50 (now 1:2 ratio)               │
│  └─ BUY 1 lot ATM+150 (PROTECTION - CRITICAL!)            │
│     → Protected Ratio Spread                               │
│     → Max Loss: STILL DEFINED (capped by protection)      │
│           ↓                                                 │
│  8:00 PM: Last entry cutoff                                │
│  11:00 PM: FORCED EXIT (all positions)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  STOP LOSS LAYERS (Multi-layered protection)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Layer 1: QUICK STOP (First 30 minutes)                    │
│  ├─ If position loses >30% in first 30 min                │
│  └─ → EXIT IMMEDIATELY (wrong trade)                       │
│                                                             │
│  Layer 2: SPOT-BASED STOP                                  │
│  ├─ For Call Ratio: Exit if spot > (short strike + 80)    │
│  ├─ For Put Ratio: Exit if spot < (short strike - 80)     │
│  └─ → Prevents runaway losses                              │
│                                                             │
│  Layer 3: PERCENTAGE STOP                                  │
│  ├─ If total loss reaches 3% of capital (₹15,000)         │
│  └─ → EXIT ALL (absolute max loss)                         │
│                                                             │
│  Layer 4: TIME STOP                                         │
│  ├─ Wednesday 7:30 PM (before EIA report at 8 PM)         │
│  ├─ Daily 11:00 PM (end of session)                       │
│  └─ → NO OVERNIGHT HOLDING                                 │
│                                                             │
│  Layer 5: PROTECTIVE WING (Phase 2 only)                  │
│  ├─ Far OTM option (ATM+150 or ATM-150)                   │
│  └─ → Caps maximum loss even if spot gaps                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PROFIT MANAGEMENT                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Target: 50% of max potential profit                       │
│                                                             │
│  Partial Exit:                                             │
│  ├─ At +30% profit: Book 50% of position                  │
│  └─ Let remaining 50% run to 50% target                   │
│                                                             │
│  Trailing Stop (activated at +20% profit):                │
│  ├─ Trail stop loss 10% below highest profit              │
│  └─ Locks in minimum 10% profit after +20% achieved       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  MAXIMUM LOSS CALCULATION                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Example: Crude at 7000 (Bullish Setup)                   │
│                                                             │
│  Phase 1 (1:1 Credit Spread):                              │
│  ├─ Buy 7000 Call @ ₹120 = -₹12,000                       │
│  ├─ Sell 7050 Call @ ₹80 = +₹8,000                        │
│  └─ Net Debit: ₹4,000                                      │
│     Max Loss: ₹4,000 + (50 × 10) = ₹9,000 ✅              │
│                                                             │
│  Phase 2 (1:2 Protected Ratio):                            │
│  ├─ Sell 1 more 7050 Call @ ₹60 = +₹6,000                │
│  ├─ Buy 7150 Call @ ₹20 = -₹2,000 (PROTECTION!)           │
│  └─ Additional Net: +₹4,000                                │
│                                                             │
│  TOTAL POSITION:                                            │
│  ├─ Buy 1 × 7000 Call                                      │
│  ├─ Sell 2 × 7050 Call                                     │
│  └─ Buy 1 × 7150 Call (PROTECTION WING)                    │
│                                                             │
│  Max Profit: If spot at 7050 = ~₹12,000                   │
│  Max Loss: If spot at 7150+ = ₹6,000 (CAPPED!) ✅         │
│                                                             │
│  WITHOUT Protection Wing:                                   │
│  Max Loss: UNLIMITED ❌                                     │
│                                                             │
│  WITH Protection Wing:                                      │
│  Max Loss: ₹6,000 (1.2% of ₹5L capital) ✅                │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  SPECIAL RULES FOR EVENING SESSION                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Why 3:15 PM - 11:00 PM?                                   │
│  ├─ Nifty closed → Less correlation                        │
│  ├─ Global cues (US markets) drive crude                   │
│  ├─ Extended time for trend to develop                     │
│  └─ Less competition from retail (most close by 8 PM)      │
│                                                             │
│  Wednesday EIA Report (8:00 PM):                           │
│  ├─ MUST exit all positions by 7:30 PM                    │
│  ├─ No new entries after 6:00 PM on Wednesday             │
│  └─ Inventory data causes extreme volatility               │
│                                                             │
│  End of Day (11:00 PM):                                    │
│  ├─ FORCED EXIT at 11:00 PM (no exceptions)               │
│  ├─ Market closes 11:30 PM, exit 30min early              │
│  └─ NO OVERNIGHT HOLDING (intraday only)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

🎯 KEY TAKEAWAYS:

1. ✅ DEFINED RISK at all times (protection wings)
2. ✅ Max loss capped at 3% (₹15,000)
3. ✅ Multi-layered stop losses
4. ✅ Evening session only (3:15 PM - 11:00 PM)
5. ✅ Mandatory exit at 11:00 PM
6. ✅ EIA report protection (Wednesday)
7. ✅ Phased entries with profit requirements
8. ✅ Conservative 1:2 max ratio (not 1:3)
"""

if __name__ == "__main__":
    print("=" * 60)
    print("CRUDE OIL PROTECTED RATIO SPREAD CONFIGURATION")
    print("=" * 60)
    
    print(f"\n⏰ Trading Hours:")
    print(f"  Session: {crude_ratio_config.SESSION_START_TIME} - {crude_ratio_config.SESSION_END_TIME}")
    print(f"  Entry Window: {crude_ratio_config.ENTRY_WINDOW_START} - {crude_ratio_config.ENTRY_WINDOW_END}")
    print(f"  Mandatory Exit: {crude_ratio_config.EXIT_TIME}")
    
    print(f"\n🛡️ Risk Management:")
    print(f"  Max Loss %: {crude_ratio_config.MAX_LOSS_PERCENT}%")
    print(f"  Max Loss ₹: ₹{crude_ratio_config.MAX_LOSS_RUPEES:,.0f}")
    print(f"  Strategy Type: {crude_ratio_config.STRATEGY_TYPE} (DEFINED RISK)")
    
    print(f"\n📊 Position Structure:")
    print(f"  Phase 1: 1:1 Credit Spread (ATM vs ATM+50)")
    print(f"  Phase 2: 1:2 Protected Ratio + ATM+150 wing")
    print(f"  Max Ratio: 1:{crude_ratio_config.MAX_RATIO}")
    
    print(f"\n🚨 Stop Loss Layers:")
    for key, value in crude_ratio_config.STOP_LOSS_LAYERS.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ NO UNLIMITED LOSS - All positions protected!")
    print("=" * 60)
