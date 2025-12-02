# run_credit_spread.py
# Sensex 1DTE Directional Credit Spread Strategy
# Supertrend + ATR + ADX with Yahoo Finance Pivot Points

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
import yfinance as yf
import asyncio
import traceback
import sqlite3
import threading
import pytz

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from openalgo_client import OpenAlgoClientWrapper

IST = pytz.timezone("Asia/Kolkata")

# ================================================================================
# INDICATOR CALCULATIONS
# ================================================================================

def calculate_supertrend(df, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Returns: df with 'supertrend' and 'supertrend_signal' columns
    supertrend_signal: 1 = GREEN (Bullish), -1 = RED (Bearish)
    """
    hl2 = (df['high'] + df['low']) / 2
    df['atr'] = calculate_atr(df, period)
    
    df['upperband'] = hl2 + (multiplier * df['atr'])
    df['lowerband'] = hl2 - (multiplier * df['atr'])
    
    df['in_uptrend'] = True
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['upperband'].iloc[i-1]:
            df.loc[df.index[i], 'in_uptrend'] = True
        elif df['close'].iloc[i] < df['lowerband'].iloc[i-1]:
            df.loc[df.index[i], 'in_uptrend'] = False
        else:
            df.loc[df.index[i], 'in_uptrend'] = df['in_uptrend'].iloc[i-1]
            
            if df['in_uptrend'].iloc[i] and df['lowerband'].iloc[i] < df['lowerband'].iloc[i-1]:
                df.loc[df.index[i], 'lowerband'] = df['lowerband'].iloc[i-1]
                
            if not df['in_uptrend'].iloc[i] and df['upperband'].iloc[i] > df['upperband'].iloc[i-1]:
                df.loc[df.index[i], 'upperband'] = df['upperband'].iloc[i-1]
    
    df['supertrend'] = df.apply(
        lambda row: row['lowerband'] if row['in_uptrend'] else row['upperband'],
        axis=1
    )
    
    df['supertrend_signal'] = df['in_uptrend'].apply(lambda x: 1 if x else -1)
    
    return df

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    return true_range.rolling(period).mean()

def calculate_adx(df, period=14):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = calculate_atr(df, 1) * period  # True Range
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)
    
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period).mean()
    
    # Protect against NaN values
    adx = adx.fillna(0)
    
    return adx

# ================================================================================
# PIVOT POINT CALCULATIONS
# ================================================================================

def fetch_previous_day_ohlc(yahoo_symbol="^BSESN"):
    """
    Fetch previous day OHLC from Yahoo Finance
    Returns: dict with 'high', 'low', 'close'
    """
    try:
        df = yf.download(yahoo_symbol, period="2d", interval="1d", progress=False)
        
        if len(df) < 2:
            print("[ERROR] Not enough data from Yahoo Finance")
            return None
        
        prev_high = float(df.iloc[-2]["High"])
        prev_low = float(df.iloc[-2]["Low"])
        prev_close = float(df.iloc[-2]["Close"])
        
        print(f"[YAHOO] Previous Day - H: {prev_high}, L: {prev_low}, C: {prev_close}")
        
        return {
            'high': prev_high,
            'low': prev_low,
            'close': prev_close
        }
    except Exception as e:
        print(f"[ERROR] Failed to fetch Yahoo data: {e}")
        traceback.print_exc()
        return None

def calculate_pivot_points(prev_data):
    """
    Calculate Pivot Points from previous day OHLC
    Returns: dict with P, S1, R1
    """
    H = prev_data['high']
    L = prev_data['low']
    C = prev_data['close']
    
    P = (H + L + C) / 3
    S1 = (2 * P) - H
    R1 = (2 * P) - L
    
    print(f"[PIVOTS] P: {P:.2f}, S1: {S1:.2f}, R1: {R1:.2f}")
    
    return {
        'P': P,
        'S1': S1,
        'R1': R1
    }

# ================================================================================
# STRATEGY STATE MANAGEMENT
# ================================================================================

class StrategyState:
    """Manages strategy state and positions"""
    
    def __init__(self):
        self.positions = []  # List of active positions
        self.last_entry_time = None
        self.last_sl_time = None
        self.last_sl_side = None  # 'BULL' or 'BEAR'
        self.df = None  # Current dataframe with indicators
        self.pivot_points = None
        self.prev_day_data = None
        
    def can_enter(self, current_time):
        """Check if new entry is allowed"""
        # Time window check
        if not (config.ENTRY_START_TIME <= current_time.time() <= config.ENTRY_END_TIME):
            return False
        
        # Max entries check
        if len(self.positions) >= config.MAX_ENTRIES:
            return False
        
        # Entry gap check
        if self.last_entry_time:
            gap = (current_time - self.last_entry_time).total_seconds() / 60
            if gap < config.MIN_ENTRY_GAP_MINUTES:
                return False
        
        # No scaling after 12 PM
        if len(self.positions) > 0 and current_time.time() >= config.NO_SCALING_AFTER_TIME:
            return False
        
        # SL cooldown check
        if self.last_sl_time:
            cooldown = (current_time - self.last_sl_time).total_seconds() / 60
            if cooldown < config.SL_COOLDOWN_MINUTES:
                print(f"[INFO] SL Cooldown active: {cooldown:.1f}/{config.SL_COOLDOWN_MINUTES} minutes")
                return False
        
        return True
    
    def get_entry_lots(self):
        """Get lot size for current entry"""
        entry_number = len(self.positions)
        if entry_number < len(config.ENTRY_LOTS):
            return config.ENTRY_LOTS[entry_number]
        return 1
    
    def add_position(self, position):
        """Add a new position"""
        self.positions.append(position)
        self.last_entry_time = datetime.now()
    
    def remove_position(self, position):
        """Remove a position"""
        if position in self.positions:
            self.positions.remove(position)
    
    def record_sl_hit(self, side):
        """Record stop loss hit"""
        self.last_sl_time = datetime.now()
        self.last_sl_side = side

# ================================================================================
# ENTRY SIGNAL LOGIC
# ================================================================================

def check_bullish_entry(df, pivot_points, state):
    """
    Check if Bullish Entry (Bull Put Spread) conditions are met
    Returns: True if all conditions satisfied
    """
    if len(df) < config.NO_FLIP_CANDLES + 1:
        return False
    
    latest = df.iloc[-1]
    
    # 1. Supertrend = GREEN
    if latest['supertrend_signal'] != 1:
        return False
    
    # 2. ATR stable or decreasing
    atr_current = latest['atr']
    atr_prev = df.iloc[-2]['atr']
    if atr_current > atr_prev:
        print("[BULL] ATR increasing, skip entry")
        return False
    
    # 3. ADX > 18
    adx_current = latest['adx']
    if adx_current < config.ADX_MIN:
        print(f"[BULL] ADX too low: {adx_current:.2f}")
        return False
    
    # If this is 2nd or 3rd entry, require stronger ADX
    if len(state.positions) >= 1:
        required_adx = config.SCALING_ADX_MIN
        if len(state.positions) >= 2:
            required_adx = config.SCALING_ADX_STRONG
        
        if adx_current < required_adx:
            print(f"[BULL] ADX not strong enough for scaling: {adx_current:.2f} < {required_adx}")
            return False
    
    # 4. No Supertrend flip in last 3 candles
    for i in range(-config.NO_FLIP_CANDLES, 0):
        if df.iloc[i]['supertrend_signal'] != 1:
            print("[BULL] Supertrend flipped recently")
            return False
    
    # 5. Price above Pivot (optional but recommended)
    if latest['close'] < pivot_points['P']:
        print(f"[BULL] Price below Pivot: {latest['close']:.2f} < {pivot_points['P']:.2f}")
        return False
    
    # 6. Check if current positions are in profit (for scaling)
    if len(state.positions) > 0:
        any_in_loss = any(p['current_pnl'] < 0 for p in state.positions if 'current_pnl' in p)
        if any_in_loss:
            print("[BULL] Existing position in loss, no scaling")
            return False
    
    print(f"[BULL SIGNAL] ✅ All conditions met - ADX: {adx_current:.2f}, ATR: {atr_current:.2f}")
    return True

def check_bearish_entry(df, pivot_points, state):
    """
    Check if Bearish Entry (Bear Call Spread) conditions are met
    Returns: True if all conditions satisfied
    """
    if len(df) < config.NO_FLIP_CANDLES + 1:
        return False
    
    latest = df.iloc[-1]
    
    # 1. Supertrend = RED
    if latest['supertrend_signal'] != -1:
        return False
    
    # 2. ATR stable or decreasing
    atr_current = latest['atr']
    atr_prev = df.iloc[-2]['atr']
    if atr_current > atr_prev:
        print("[BEAR] ATR increasing, skip entry")
        return False
    
    # 3. ADX > 18
    adx_current = latest['adx']
    if adx_current < config.ADX_MIN:
        print(f"[BEAR] ADX too low: {adx_current:.2f}")
        return False
    
    # If this is 2nd or 3rd entry, require stronger ADX
    if len(state.positions) >= 1:
        required_adx = config.SCALING_ADX_MIN
        if len(state.positions) >= 2:
            required_adx = config.SCALING_ADX_STRONG
        
        if adx_current < required_adx:
            print(f"[BEAR] ADX not strong enough for scaling: {adx_current:.2f} < {required_adx}")
            return False
    
    # 4. No Supertrend flip in last 3 candles
    for i in range(-config.NO_FLIP_CANDLES, 0):
        if df.iloc[i]['supertrend_signal'] != -1:
            print("[BEAR] Supertrend flipped recently")
            return False
    
    # 5. Price below Pivot (recommended)
    if latest['close'] > pivot_points['P']:
        print(f"[BEAR] Price above Pivot: {latest['close']:.2f} > {pivot_points['P']:.2f}")
        return False
    
    # 6. Check if current positions are in profit (for scaling)
    if len(state.positions) > 0:
        any_in_loss = any(p['current_pnl'] < 0 for p in state.positions if 'current_pnl' in p)
        if any_in_loss:
            print("[BEAR] Existing position in loss, no scaling")
            return False
    
    print(f"[BEAR SIGNAL] ✅ All conditions met - ADX: {adx_current:.2f}, ATR: {atr_current:.2f}")
    return True

# ================================================================================
# STRIKE SELECTION
# ================================================================================

def select_bull_put_strikes(S1, current_price):
    """
    Select strikes for Bull Put Spread
    Sell PE near S1, Buy lower PE (300-500 points below)
    Returns: (sell_strike, buy_strike)
    """
    # Round S1 to nearest 100
    sell_strike = round(S1 / 100) * 100
    
    # Calculate spread width (use midpoint)
    spread_width = (config.SPREAD_WIDTH_MIN + config.SPREAD_WIDTH_MAX) / 2
    buy_strike = sell_strike - spread_width
    
    print(f"[BPS] Sell {sell_strike} PE, Buy {buy_strike} PE (width: {spread_width})")
    return int(sell_strike), int(buy_strike)

def select_bear_call_strikes(R1, current_price):
    """
    Select strikes for Bear Call Spread
    Sell CE near R1, Buy higher CE (300-500 points above)
    Returns: (sell_strike, buy_strike)
    """
    # Round R1 to nearest 100
    sell_strike = round(R1 / 100) * 100
    
    # Calculate spread width (use midpoint)
    spread_width = (config.SPREAD_WIDTH_MIN + config.SPREAD_WIDTH_MAX) / 2
    buy_strike = sell_strike + spread_width
    
    print(f"[BCS] Sell {sell_strike} CE, Buy {buy_strike} CE (width: {spread_width})")
    return int(sell_strike), int(buy_strike)

# ================================================================================
# ORDER PLACEMENT
# ================================================================================

async def place_bull_put_spread(client, sell_strike, buy_strike, lots, expiry):
    """
    Place Bull Put Spread
    Returns: position dict or None
    """
    try:
        quantity = lots * config.LOTSIZE
        
        # Build option symbols
        sell_symbol = f"{config.UNDERLYING_SYMBOL}{expiry}{sell_strike}PE"
        buy_symbol = f"{config.UNDERLYING_SYMBOL}{expiry}{buy_strike}PE"
        
        print(f"[BPS] Placing spread: Sell {sell_symbol}, Buy {buy_symbol} x {quantity}")
        
        if not config.PLACE_ORDERS:
            print("[SIM] Simulated order placement")
            return {
                'type': 'BPS',
                'sell_strike': sell_strike,
                'buy_strike': buy_strike,
                'sell_symbol': sell_symbol,
                'buy_symbol': buy_symbol,
                'lots': lots,
                'quantity': quantity,
                'entry_time': datetime.now(),
                'credit_received': 50,  # Simulated
                'status': 'ACTIVE'
            }
        
        # Place actual orders via client using gather for simultaneous execution
        sell_order, buy_order = await asyncio.gather(
            client.async_place_orders(sell_symbol, 'SELL', quantity, config.STRATEGY_NAME),
            client.async_place_orders(buy_symbol, 'BUY', quantity, config.STRATEGY_NAME)
        )
        
        # Check if both orders completed
        sell_complete = sell_order and sell_order.get('order_status') == 'complete'
        buy_complete = buy_order and buy_order.get('order_status') == 'complete'
        
        if not sell_complete or not buy_complete:
            print(f"[BPS] ❌ Order execution failed - Sell: {sell_complete}, Buy: {buy_complete}")
            return None
        
        sell_price = float(sell_order.get('average_price', 0))
        buy_price = float(buy_order.get('average_price', 0))
        credit = sell_price - buy_price
        
        # Place SL order for the sold leg
        # SL percentage based on config (using Option A percentage)
        sl_pct = config.SL_OPTION_A_PCT
        sl_order = await client.async_sl_order(sell_order, sl_pct, strategy_tag=config.STRATEGY_NAME)
        
        if not sl_order:
            print(f"[BPS] ⚠️ SL order placement failed for {sell_symbol}")
        
        position = {
            'type': 'BPS',
            'sell_strike': sell_strike,
            'buy_strike': buy_strike,
            'sell_symbol': sell_symbol,
            'buy_symbol': buy_symbol,
            'lots': lots,
            'quantity': quantity,
            'sell_order': sell_order,
            'buy_order': buy_order,
            'sl_order': sl_order,
            'sell_price': sell_price,
            'buy_price': buy_price,
            'credit_received': credit,
            'entry_time': datetime.now(),
            'status': 'ACTIVE'
        }
        
        print(f"[BPS] ✅ Spread placed - Credit: ₹{credit:.2f}, SL: {sl_pct}%")
        return position
        
    except Exception as e:
        print(f"[ERROR] Failed to place BPS: {e}")
        traceback.print_exc()
    
    return None

async def place_bear_call_spread(client, sell_strike, buy_strike, lots, expiry):
    """
    Place Bear Call Spread
    Returns: position dict or None
    """
    try:
        quantity = lots * config.LOTSIZE
        
        # Build option symbols
        sell_symbol = f"{config.UNDERLYING_SYMBOL}{expiry}{sell_strike}CE"
        buy_symbol = f"{config.UNDERLYING_SYMBOL}{expiry}{buy_strike}CE"
        
        print(f"[BCS] Placing spread: Sell {sell_symbol}, Buy {buy_symbol} x {quantity}")
        
        if not config.PLACE_ORDERS:
            print("[SIM] Simulated order placement")
            return {
                'type': 'BCS',
                'sell_strike': sell_strike,
                'buy_strike': buy_strike,
                'sell_symbol': sell_symbol,
                'buy_symbol': buy_symbol,
                'lots': lots,
                'quantity': quantity,
                'entry_time': datetime.now(),
                'credit_received': 50,  # Simulated
                'status': 'ACTIVE'
            }
        
        # Place actual orders via client using gather for simultaneous execution
        sell_order, buy_order = await asyncio.gather(
            client.async_place_orders(sell_symbol, 'SELL', quantity, config.STRATEGY_NAME),
            client.async_place_orders(buy_symbol, 'BUY', quantity, config.STRATEGY_NAME)
        )
        
        # Check if both orders completed
        sell_complete = sell_order and sell_order.get('order_status') == 'complete'
        buy_complete = buy_order and buy_order.get('order_status') == 'complete'
        
        if not sell_complete or not buy_complete:
            print(f"[BCS] ❌ Order execution failed - Sell: {sell_complete}, Buy: {buy_complete}")
            return None
        
        sell_price = float(sell_order.get('average_price', 0))
        buy_price = float(buy_order.get('average_price', 0))
        credit = sell_price - buy_price
        
        # Place SL order for the sold leg
        # SL percentage based on config (using Option A percentage)
        sl_pct = config.SL_OPTION_A_PCT
        sl_order = await client.async_sl_order(sell_order, sl_pct, strategy_tag=config.STRATEGY_NAME)
        
        if not sl_order:
            print(f"[BCS] ⚠️ SL order placement failed for {sell_symbol}")
        
        position = {
            'type': 'BCS',
            'sell_strike': sell_strike,
            'buy_strike': buy_strike,
            'sell_symbol': sell_symbol,
            'buy_symbol': buy_symbol,
            'lots': lots,
            'quantity': quantity,
            'sell_order': sell_order,
            'buy_order': buy_order,
            'sl_order': sl_order,
            'sell_price': sell_price,
            'buy_price': buy_price,
            'credit_received': credit,
            'entry_time': datetime.now(),
            'status': 'ACTIVE'
        }
        
        print(f"[BCS] ✅ Spread placed - Credit: ₹{credit:.2f}, SL: {sl_pct}%")
        return position
        
    except Exception as e:
        print(f"[ERROR] Failed to place BCS: {e}")
        traceback.print_exc()
    
    return None

# ================================================================================
# EXIT LOGIC
# ================================================================================

async def check_exit_conditions(client, position, current_ltp_sell, current_ltp_buy):
    """
    Check if position should be exited
    Returns: (should_exit, reason)
    """
    if position['status'] != 'ACTIVE':
        return False, None
    
    current_spread_value = current_ltp_sell - current_ltp_buy
    credit = position['credit_received']
    
    # Calculate P&L
    pnl = credit - current_spread_value
    pnl_pct = (pnl / credit) * 100 if credit > 0 else 0
    
    position['current_pnl'] = pnl
    position['current_pnl_pct'] = pnl_pct
    
    # Check Stop Loss
    if config.USE_SL_OPTION_A:
        # Option A: 50% loss
        if pnl_pct <= -config.SL_OPTION_A_PCT:
            return True, f"SL_A: {pnl_pct:.1f}% loss"
    
    if config.USE_SL_OPTION_B:
        # Option B: Spread value doubles
        if current_spread_value >= credit * config.SL_OPTION_B_MULTIPLIER:
            return True, f"SL_B: Spread value doubled ({current_spread_value:.2f} >= {credit * config.SL_OPTION_B_MULTIPLIER:.2f})"
    
    # Check Target
    if pnl_pct >= config.TARGET_MIN_PCT:
        return True, f"TARGET: {pnl_pct:.1f}% profit"
    
    # Check Time Exit
    current_time = datetime.now().time()
    if current_time >= config.MANDATORY_EXIT_TIME:
        return True, f"TIME_EXIT: {current_time}"
    
    return False, None

async def exit_position(client, position, reason):
    """
    Exit a credit spread position
    """
    try:
        print(f"[EXIT] Closing {position['type']} - Reason: {reason}")
        
        if not config.PLACE_ORDERS:
            position['status'] = 'CLOSED'
            position['exit_time'] = datetime.now()
            position['exit_reason'] = reason
            print(f"[SIM] Position closed")
            return True
        
        quantity = position['quantity']
        
        # Cancel SL order if exists
        sl_order = position.get('sl_order')
        if sl_order:
            try:
                sl_id = sl_order.get('orderid')
                if sl_id and sl_id.get('order_status', '').lower() == 'trigger pending':
                    print(f"[EXIT] Cancelling SL order {sl_id}")
                    await client.client.cancelorder_async(
                        order_id=sl_id,
                        strategy=config.STRATEGY_NAME
                    )
            except Exception as e:
                print(f"[WARN] Failed to cancel SL order: {e}")

        if reason == "SL_ORDER_HIT":
            sell_long_order = await client.async_place_orders(
                position['buy_symbol'],
                'SELL',
                quantity,
                config.STRATEGY_NAME
            )
            position['status'] = 'CLOSED'
            position['exit_time'] = datetime.now()
            position['exit_reason'] = reason
            print(f"[EXIT] ✅ Position closed successfully")
            return True
        else:
            # Exit positions simultaneously
            buy_back_order, sell_long_order = await asyncio.gather(
                client.async_place_orders(
                    position['sell_symbol'],
                    'BUY',
                    quantity,
                    config.STRATEGY_NAME
                ),
                client.async_place_orders(
                    position['buy_symbol'],
                    'SELL',
                    quantity,
                    config.STRATEGY_NAME
                )
            )
        
            if buy_back_order and sell_long_order:
                position['status'] = 'CLOSED'
                position['exit_time'] = datetime.now()
                position['exit_reason'] = reason
                print(f"[EXIT] ✅ Position closed successfully")
                return True
        
    except Exception as e:
        print(f"[ERROR] Failed to exit position: {e}")
        traceback.print_exc()
    
    return False

# ================================================================================
# MAIN STRATEGY CLASS
# ================================================================================

class SensexCreditSpreadRunner:
    """Main strategy runner with WebSocket integration"""
    
    def __init__(self):
        self.config = config
        
        # OpenAlgo client
        self.client = OpenAlgoClientWrapper(
            api_key=self.config.OPENALGO_API_KEY,
            host=self.config.OPENALGO_HOST,
            ws_url=self.config.OPENALGO_WS,
            on_ltp_callback=self.on_ltp
        )
        
        # Strategy state
        self.state = StrategyState()
        
        # OHLC data for 5-minute candles
        self.ohlc = []
        self.ohlc_df = None
        self.curr_min = None
        self.open = self.high = self.low = self.close = None
        
        # Options symbols and LTP tracking
        self.spot_symbol = None  # Sensex spot/future symbol
        self.ce_symbols = {}  # {strike: symbol_dict}
        self.pe_symbols = {}  # {strike: symbol_dict}
        self.ltp_cache = {}  # {symbol: ltp}
        
        # ATM tracking
        self.current_atm = None
        self.nearest_expiry = None
        
        # Threading lock
        self.lock = threading.Lock()
        
    def now_ist(self):
        """Get current IST time"""
        return datetime.now(pytz.utc).astimezone(IST)
    
    def in_allowed_window(self):
        """Check if current time is in allowed entry window"""
        now = self.now_ist().time()
        return self.config.ENTRY_START_TIME <= now <= self.config.ENTRY_END_TIME
    
    def initialize_symbols(self):
        """
        Initialize Sensex spot and options symbols for ±20 strikes from ATM
        Similar to crude oil get_option_symbols pattern
        """
        print("[INIT] Fetching Sensex symbols...")
        
        # Set exchange and symbol
        self.client.exchange = self.config.UNDERLYING_EXCHANGE
        self.client.symbol = self.config.UNDERLYING_SYMBOL
        self.client.strike_step = 100  # Sensex strike step
        
        # Get nearest expiry for options
        try:
            self.nearest_expiry = self.client.get_nearest_expiry(
                symbol=self.config.UNDERLYING_SYMBOL,
                exchange=self.config.UNDERLYING_EXCHANGE
            )
            print(f"[INIT] Nearest expiry: {self.nearest_expiry}")
        except Exception as e:
            print(f"[ERROR] Failed to get expiry: {e}")
            return None
        
        # Get quote for spot to determine ATM
        try:
            quote = self.client.client.quotes(
                symbol=self.config.UNDERLYING_SYMBOL,
                exchange="BSE"  # BSE for Sensex spot
            )
            print(f"[DEBUG] Spot Quote: {quote}")
            
            spot_price = 0
            if quote and 'data' in quote and 'ltp' in quote['data']:
                spot_price = float(quote["data"]["ltp"])
            
            print(f"[INIT] Sensex spot price: {spot_price}")
            
            if spot_price <= 0:
                print("[WARN] Invalid spot price from OpenAlgo. Using Yahoo Finance previous close as fallback.")
                if self.state.prev_day_data:
                    spot_price = self.state.prev_day_data['close']
                    print(f"[INIT] Fallback Spot Price: {spot_price}")
                else:
                    print("[ERROR] No fallback data available. Cannot determine ATM.")
                    return None
            
            # Calculate ATM
            self.current_atm = self.client.get_ATM_strike(spot_price)
            print(f"[INIT] ATM Strike: {self.current_atm}")
            
            # Set spot symbol for subscriptions
            self.spot_symbol = {'exchange': 'BSE', 'symbol': self.config.UNDERLYING_SYMBOL}
            
        except Exception as e:
            print(f"[ERROR] Failed to get spot price: {e}")
            traceback.print_exc()
            return None
        
        # Build strike range (±20 from ATM)
        strike_range = [
            self.current_atm + (i * self.client.strike_step) 
            for i in range(-20, 21)
        ]
        
        # Build CE and PE symbols
        subscription_symbols = []
        
        for strike in strike_range:
            # CE symbol
            ce_sym = self.client.build_option_symbol(
                self.config.UNDERLYING_SYMBOL,
                self.nearest_expiry,
                strike,
                "CE",
                exchange=self.config.UNDERLYING_EXCHANGE
            )
            self.ce_symbols[strike] = ce_sym
            subscription_symbols.append(ce_sym)
            
            # PE symbol
            pe_sym = self.client.build_option_symbol(
                self.config.UNDERLYING_SYMBOL,
                self.nearest_expiry,
                strike,
                "PE",
                exchange=self.config.UNDERLYING_EXCHANGE
            )
            self.pe_symbols[strike] = pe_sym
            subscription_symbols.append(pe_sym)
        
        # Add spot symbol
        subscription_symbols.append(self.spot_symbol)
        
        print(f"[INIT] Total subscription symbols: {len(subscription_symbols)}")
        print(f"[INIT] CE strikes: {len(self.ce_symbols)}, PE strikes: {len(self.pe_symbols)}")
        
        return subscription_symbols
    
    def update_ohlc_df(self):
        """Update OHLC dataframe from candle list"""
        if len(self.ohlc) == 0:
            self.ohlc_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])
        else:
            self.ohlc_df = pd.DataFrame(self.ohlc)
    
    def compute_indicators(self):
        """Calculate all indicators on current OHLC data"""
        if self.ohlc_df is None or len(self.ohlc_df) < max(config.SUPERTREND_PERIOD, config.ADX_PERIOD) + 1:
            return False
        
        try:
            # Calculate Supertrend
            self.ohlc_df = calculate_supertrend(
                self.ohlc_df,
                period=config.SUPERTREND_PERIOD,
                multiplier=config.SUPERTREND_MULTIPLIER
            )
            
            # Calculate ADX
            self.ohlc_df['adx'] = calculate_adx(
                self.ohlc_df,
                period=config.ADX_PERIOD
            )
            
            # Store in state
            self.state.df = self.ohlc_df
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Indicator calculation failed: {e}")
            traceback.print_exc()
            return False
    
    def on_underlying_tick(self, tick):
        """Handle Sensex spot tick to build OHLC candles"""
        ltp = tick.get('ltp')
        ts_ms = tick.get('timestamp')
        
        if ltp is None or ts_ms is None:
            return
        
        dt = datetime.fromtimestamp(ts_ms/1000.0, pytz.utc).astimezone(IST)
        
        # Build 5-minute candles
        minute = dt.replace(second=0, microsecond=0)
        candle_5min = minute.replace(minute=(minute.minute // 5) * 5)
        
        with self.lock:
            # Initialize first candle
            if self.curr_min is None:
                self.curr_min = candle_5min
                self.open = self.high = self.low = self.close = ltp
                return
            
            # New candle completed
            if candle_5min != self.curr_min:
                # Finalize previous candle
                row = {
                    'timestamp': self.curr_min,
                    'open': self.open,
                    'high': self.high,
                    'low': self.low,
                    'close': self.close
                }
                
                print(f"[5-MIN] Candle: O={self.open:.1f} H={self.high:.1f} L={self.low:.1f} C={self.close:.1f}")
                
                self.ohlc.append(row)
                self.update_ohlc_df()
                
                # Calculate indicators
                if self.compute_indicators():
                    # Check for entry signals
                    if self.in_allowed_window():
                        self.check_entry_signals()
                
                # Reset for new candle
                self.curr_min = candle_5min
                self.open = self.high = self.low = self.close = ltp
            else:
                # Update running candle
                self.close = ltp
                self.high = max(self.high, ltp)
                self.low = min(self.low, ltp)
            
            # Check exits on every tick
            if len(self.state.positions) > 0:
                self.check_exit_conditions_all()
    
    def on_option_tick(self, symbol, ltp):
        """Handle option tick - update LTP cache"""
        self.ltp_cache[symbol] = ltp
    
    def on_ltp(self, data):
        """Main LTP callback for all subscribed symbols"""
        if data.get('type') != 'market_data':
            return
        
        symbol = data.get('symbol')
        exch = data.get('exchange')
        tick = data.get('data', {})
        ltp = tick.get('ltp')
        
        if ltp is None:
            return
        
        # Spot/underlying tick
        if symbol == self.config.UNDERLYING_SYMBOL:
            self.on_underlying_tick(tick)
        # Options tick
        elif exch == self.config.UNDERLYING_EXCHANGE:
            self.on_option_tick(symbol, ltp)
    
    def check_entry_signals(self):
        """Check for bullish or bearish entry signals"""
        if not self.state.can_enter(self.now_ist()):
            return
        
        if self.state.df is None or len(self.state.df) == 0:
            return
        
        current_spot = self.close  # Latest spot price
        
        # Check bullish signal
        if check_bullish_entry(self.state.df, self.state.pivot_points, self.state):
            print("[ENTRY] Bullish signal detected!")
            lots = self.state.get_entry_lots()
            sell_strike, buy_strike = select_bull_put_strikes(
                self.state.pivot_points['S1'],
                current_spot
            )
            
            # Place order in separate thread to avoid blocking
            threading.Thread(
                target=self._run_async_order,
                args=('BPS', sell_strike, buy_strike, lots)
            ).start()
            
        # Check bearish signal
        elif check_bearish_entry(self.state.df, self.state.pivot_points, self.state):
            print("[ENTRY] Bearish signal detected!")
            lots = self.state.get_entry_lots()
            sell_strike, buy_strike = select_bear_call_strikes(
                self.state.pivot_points['R1'],
                current_spot
            )
            
            # Place order in separate thread
            threading.Thread(
                target=self._run_async_order,
                args=('BCS', sell_strike, buy_strike, lots)
            ).start()
    
    def _run_async_order(self, spread_type, sell_strike, buy_strike, lots):
        """Helper to run async order placement"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            if spread_type == 'BPS':
                position = loop.run_until_complete(
                    place_bull_put_spread(
                        self.client, sell_strike, buy_strike, lots, self.nearest_expiry
                    )
                )
            else:  # BCS
                position = loop.run_until_complete(
                    place_bear_call_spread(
                        self.client, sell_strike, buy_strike, lots, self.nearest_expiry
                    )
                )
            
            if position:
                with self.lock:
                    self.state.add_position(position)
                print(f"[POSITION] Added {spread_type} position: {position['sell_strike']}/{position['buy_strike']}")
        
        except Exception as e:
            print(f"[ERROR] Order placement failed: {e}")
            traceback.print_exc()
        finally:
            loop.close()
    
    def check_exit_conditions_all(self):
        """Check exit conditions for all active positions"""
        if len(self.state.positions) == 0:
            return
        
        for position in self.state.positions[:]:  # Copy to allow modification
            if position['status'] != 'ACTIVE':
                continue
            
            # Get current LTPs
            sell_symbol = position['sell_symbol']
            buy_symbol = position['buy_symbol']
            
            current_ltp_sell = self.ltp_cache.get(sell_symbol, 0)
            current_ltp_buy = self.ltp_cache.get(buy_symbol, 0)
            
            if current_ltp_sell == 0 or current_ltp_buy == 0:
                continue  # Wait for valid LTP
            
            # Check if SL order was triggered (priority check)
            if position.get('sl_order'):
                sl_order_id = position['sl_order'].get('orderid')
                if sl_order_id:
                    sl_info = self.client.get_order_info_of_order(sl_order_id)
                    if sl_info and sl_info.get('order_status', '').lower() != 'trigger pending':
                        # SL hit - close the hedge and exit
                        print(f"[SL HIT] Stop loss triggered for {position['type']} - {position['sell_symbol']}")
                        
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            success = loop.run_until_complete(
                                exit_position(self.client, position, "SL_ORDER_HIT")
                            )
                            if success:
                                with self.lock:
                                    self.state.remove_position(position)
                                self.state.record_sl_hit(position['type'])
                                print(f"[EXIT] ✅ Position closed due to SL hit")
                        except Exception as e:
                            print(f"[ERROR] SL exit failed: {e}")
                        finally:
                            loop.close()
                        continue
            
            # Run async check for spread-value based exits
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                should_exit, reason = loop.run_until_complete(
                    check_exit_conditions(
                        self.client, position, current_ltp_sell, current_ltp_buy
                    )
                )
                
                if should_exit:
                    print(f"[EXIT] Exit triggered for {position['type']}: {reason}")
                    
                    success = loop.run_until_complete(
                        exit_position(self.client, position, reason)
                    )
                    
                    if success:
                        with self.lock:
                            self.state.remove_position(position)
                        
                        if 'SL' in reason:
                            self.state.record_sl_hit(position['type'])
                        
                        print(f"[EXIT] ✅ Position closed: {reason}")
            
            except Exception as e:
                print(f"[ERROR] Exit check failed: {e}")
                traceback.print_exc()
            finally:
                loop.close()
    
    def start(self):
        """Start the strategy runner"""
        print("="*80)
        print("SENSEX 1DTE DIRECTIONAL CREDIT SPREAD STRATEGY")
        print("="*80)
        print(f"Strategy: {self.config.STRATEGY_NAME}")
        print(f"Underlying: {self.config.UNDERLYING_SYMBOL}")
        print(f"Exchange: {self.config.UNDERLYING_EXCHANGE}")
        print(f"Entry Window: {self.config.ENTRY_START_TIME} - {self.config.ENTRY_END_TIME}")
        print(f"Exit Time: {self.config.MANDATORY_EXIT_TIME}")
        print(f"Max Entries: {self.config.MAX_ENTRIES}")
        print(f"Entry Lots: {self.config.ENTRY_LOTS}")
        print("="*80)
        
        # Fetch previous day data from Yahoo
        print("\n[INIT] Fetching previous day data from Yahoo Finance...")
        self.state.prev_day_data = fetch_previous_day_ohlc(self.config.YAHOO_SYMBOL)
        
        if not self.state.prev_day_data:
            print("[ERROR] Failed to fetch previous day data. Exiting.")
            return
        
        # Calculate pivot points
        self.state.pivot_points = calculate_pivot_points(self.state.prev_day_data)
        
        # Connect to OpenAlgo
        print("\n[INIT] Connecting to OpenAlgo...")
        self.client.connect()
        
        # Initialize and subscribe to symbols
        symbols = self.initialize_symbols()
        
        if not symbols:
            print("[ERROR] Failed to initialize symbols. Exiting.")
            self.client.disconnect()
            return
        
        # Subscribe to LTP
        print(f"\n[INIT] Subscribing to {len(symbols)} symbols...")
        self.client.subscribe_ltp(symbols)
        self.client.subscribe_depth(symbols)
        self.client.subscribe_orderbook()
        
        print("[INIT] ✅ Strategy initialized and listening for data...")
        print(f"[INIT] Pivot Points - P: {self.state.pivot_points['P']:.2f}, S1: {self.state.pivot_points['S1']:.2f}, R1: {self.state.pivot_points['R1']:.2f}")
        print("\n[RUNNING] Waiting for 5-minute candles to build...\n")
        
        # Main loop
        try:
            while True:
                time.sleep(1)
                
                # Check for EOD exit
                now = self.now_ist().time()
                if now >= dt_time(15, 30):
                    print("\n[EOD] End of day reached. Exiting...")
                    break
                
        except KeyboardInterrupt:
            print("\n[STOP] Strategy stopped by user")
        finally:
            print("\n[SHUTDOWN] Disconnecting...")
            self.client.disconnect()
            print("[COMPLETE] Strategy execution completed")

# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    runner = SensexCreditSpreadRunner()
    runner.start()
