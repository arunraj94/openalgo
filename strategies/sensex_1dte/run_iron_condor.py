# run_iron_condor.py
# Sensex 1DTE Wide OTM Iron Condor Strategy
# Entry: 9:25-9:35 AM, Rangebound Days Only (ADX < 18)
# No Adjustments, No Scaling, One Trade Per Day

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

from config_iron_condor import config
from openalgo_client import OpenAlgoClientWrapper

IST = pytz.timezone("Asia/Kolkata")

# ================================================================================
# INDICATOR CALCULATIONS
# ================================================================================

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
# PREVIOUS DAY DATA
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

# ================================================================================
# STRATEGY STATE MANAGEMENT
# ================================================================================

class IronCondorState:
    """Manages Iron Condor state with tranche-based scaling"""
    
    def __init__(self):
        self.positions = []  # List of IC positions (up to 3 tranches)
        self.entry_count = 0  # Number of tranches entered
        self.last_entry_time = None  # Time of last tranche entry
        self.initial_premium = None  # Baseline premium for scaling triggers
        self.df = None  # Current dataframe with indicators
        self.prev_day_data = None
        
    def can_enter(self, current_time):
        """Check if new tranche entry is allowed"""
        # Max tranches reached
        if self.entry_count >= config.MAX_ENTRIES:
            return False
        
        # Already have max active positions
        active_positions = [p for p in self.positions if p['status'] == 'ACTIVE']
        if len(active_positions) >= config.MAX_ENTRIES:
            return False
        
        # First entry: Time window check (9:25-9:35 AM)
        if self.entry_count == 0:
            if not (config.ENTRY_START_TIME <= current_time.time() <= config.ENTRY_END_TIME):
                return False
        
        # Subsequent tranches: No entries after 2 PM
        else:
            if current_time.time() >= config.NO_MORE_TRANCHES_AFTER:
                return False
            
            # Minimum gap between tranches
            if self.last_entry_time:
                gap_minutes = (current_time - self.last_entry_time).total_seconds() / 60
                if gap_minutes < config.MIN_TRANCHE_GAP_MINUTES:
                    return False
        
        return True
    
    def get_entry_lots(self):
        """Get lot size for current tranche"""
        if self.entry_count < len(config.ENTRY_LOTS):
            return config.ENTRY_LOTS[self.entry_count]
        return 1  # Fallback
    
    def add_position(self, position):
        """Add IC position (tranche)"""
        self.positions.append(position)
        self.entry_count += 1
        self.last_entry_time = datetime.now()
        
        # Set initial premium baseline from first tranche
        if self.initial_premium is None:
            self.initial_premium = position['credit_received']
    
    def remove_position(self, position):
        """Remove a position"""
        if position in self.positions:
            self.positions.remove(position)
    
    def get_total_credit(self):
        """Get total credit from all active positions"""
        active = [p for p in self.positions if p['status'] == 'ACTIVE']
        return sum(p['credit_received'] for p in active)

# ================================================================================
# ENTRY SIGNAL LOGIC
# ================================================================================

def check_iron_condor_entry(df, spot_price, prev_day_close):
    """
    Check if Iron Condor entry conditions are met
    Returns: (can_enter, reason)
    
    NOTE: This runs at 9:25-9:35 AM when we only have 2-4 candles
    ADX requires 14-20 candles, so we use simpler filters:
    1. Opening gap < 200 points (immediate calculation)
    2. ATR not expanding rapidly (needs only 2-3 candles)
    3. Early morning = rangebound assumption
    """
    # 1. Opening gap filter (works immediately)
    opening_gap = abs(spot_price - prev_day_close)
    if opening_gap > config.MAX_OPENING_GAP_POINTS:
        return False, f"Opening gap too large: {opening_gap:.0f} > {config.MAX_OPENING_GAP_POINTS}"
    
    # 2. ATR volatility check (only if we have enough data)
    atr_current = 0
    if len(df) >= 3:
        latest = df.iloc[-1]
        atr_current = latest.get('atr', 0)
        
        # Skip if ATR is too high (volatility spike)
        if atr_current > config.MAX_ATR_THRESHOLD:
            return False, f"High volatility: ATR {atr_current:.0f} > {config.MAX_ATR_THRESHOLD}"
    else:
        # Not enough data for ATR yet
        return False, "Waiting for ATR data (need 3+ candles)"
    
    # 3. All conditions met - early morning + small gap + low ATR = rangebound
    print(f"[IC SIGNAL] ✅ Conditions met - ATR: {atr_current:.2f}, Gap: {opening_gap:.0f}")
    return True, "All conditions met"

# ================================================================================
# STRIKE SELECTION
# ================================================================================

def select_iron_condor_strikes(spot_price, atr_value=None):
    """
    Select strikes for Iron Condor
    Returns: (sell_ce, buy_ce, sell_pe, buy_pe)
    """
    if config.USE_ATR_BASED_STRIKES and atr_value:
        # ATR-based strike selection
        sell_distance = int(atr_value * config.SELL_STRIKE_ATR_MULTIPLE)
        buy_distance = int(atr_value * config.BUY_STRIKE_ATR_MULTIPLE)
        
        # Round to strike step
        sell_distance = round(sell_distance / config.STRIKE_STEP) * config.STRIKE_STEP
        buy_distance = round(buy_distance / config.STRIKE_STEP) * config.STRIKE_STEP
        
        print(f"[IC] ATR-based strikes - ATR: {atr_value:.0f}, Sell: ±{sell_distance}, Buy: ±{buy_distance}")
    else:
        # Fixed distance
        sell_distance = config.SELL_CE_DISTANCE
        buy_distance = config.BUY_CE_DISTANCE
        
        print(f"[IC] Fixed strikes - Sell: ±{sell_distance}, Buy: ±{buy_distance}")
    
    # Calculate strikes
    sell_ce = round((spot_price + sell_distance) / config.STRIKE_STEP) * config.STRIKE_STEP
    buy_ce = round((spot_price + buy_distance) / config.STRIKE_STEP) * config.STRIKE_STEP
    sell_pe = round((spot_price - sell_distance) / config.STRIKE_STEP) * config.STRIKE_STEP
    buy_pe = round((spot_price - buy_distance) / config.STRIKE_STEP) * config.STRIKE_STEP
    
    print(f"[IC] Strikes - Sell {sell_ce} CE, Buy {buy_ce} CE, Sell {sell_pe} PE, Buy {buy_pe} PE")
    print(f"[IC] Breakevens - Upper: ~{sell_ce + 25}, Lower: ~{sell_pe - 25}")
    
    return int(sell_ce), int(buy_ce), int(sell_pe), int(buy_pe)

# ================================================================================
# ORDER PLACEMENT
# ================================================================================

async def place_iron_condor(client, sell_ce, buy_ce, sell_pe, buy_pe, lots, expiry):
    """
    Place Iron Condor (4 legs simultaneously)
    Returns: position dict or None
    """
    try:
        quantity = lots * config.LOTSIZE
        
        # Build option symbols
        sell_ce_symbol = f"{config.UNDERLYING_SYMBOL}{expiry}{sell_ce}CE"
        buy_ce_symbol = f"{config.UNDERLYING_SYMBOL}{expiry}{buy_ce}CE"
        sell_pe_symbol = f"{config.UNDERLYING_SYMBOL}{expiry}{sell_pe}PE"
        buy_pe_symbol = f"{config.UNDERLYING_SYMBOL}{expiry}{buy_pe}PE"
        
        print(f"[IC] Placing Iron Condor:")
        print(f"  CE Spread: Sell {sell_ce_symbol}, Buy {buy_ce_symbol}")
        print(f"  PE Spread: Sell {sell_pe_symbol}, Buy {buy_pe_symbol}")
        print(f"  Quantity: {quantity} per leg")
        
        if not config.PLACE_ORDERS:
            print("[SIM] Simulated order placement")
            return {
                'type': 'IC',
                'sell_ce': sell_ce,
                'buy_ce': buy_ce,
                'sell_pe': sell_pe,
                'buy_pe': buy_pe,
                'sell_ce_symbol': sell_ce_symbol,
                'buy_ce_symbol': buy_ce_symbol,
                'sell_pe_symbol': sell_pe_symbol,
                'buy_pe_symbol': buy_pe_symbol,
                'lots': lots,
                'quantity': quantity,
                'entry_time': datetime.now(),
                'credit_received': 250,  # Simulated
                'status': 'ACTIVE'
            }
        
        # Place all 4 legs simultaneously
        sell_ce_order, buy_ce_order, sell_pe_order, buy_pe_order = await asyncio.gather(
            client.async_place_orders(sell_ce_symbol, 'SELL', quantity, config.STRATEGY_NAME),
            client.async_place_orders(buy_ce_symbol, 'BUY', quantity, config.STRATEGY_NAME),
            client.async_place_orders(sell_pe_symbol, 'SELL', quantity, config.STRATEGY_NAME),
            client.async_place_orders(buy_pe_symbol, 'BUY', quantity, config.STRATEGY_NAME)
        )
        
        # Check if all orders completed
        all_complete = all([
            sell_ce_order and sell_ce_order.get('order_status') == 'complete',
            buy_ce_order and buy_ce_order.get('order_status') == 'complete',
            sell_pe_order and sell_pe_order.get('order_status') == 'complete',
            buy_pe_order and buy_pe_order.get('order_status') == 'complete'
        ])
        
        if not all_complete:
            print(f"[IC] ❌ Order execution failed - Not all legs filled")
            return None
        
        # Calculate total credit
        sell_ce_price = float(sell_ce_order.get('average_price', 0))
        buy_ce_price = float(buy_ce_order.get('average_price', 0))
        sell_pe_price = float(sell_pe_order.get('average_price', 0))
        buy_pe_price = float(buy_pe_order.get('average_price', 0))
        
        ce_credit = sell_ce_price - buy_ce_price
        pe_credit = sell_pe_price - buy_pe_price
        total_credit = ce_credit + pe_credit
        
        # Place SL orders for both sold legs
        sl_pct = config.SL_CREDIT_INCREASE_PCT
        
        sl_ce_order = await client.async_sl_order(sell_ce_order, sl_pct, strategy_tag=config.STRATEGY_NAME)
        sl_pe_order = await client.async_sl_order(sell_pe_order, sl_pct, strategy_tag=config.STRATEGY_NAME)
        
        if not sl_ce_order:
            print(f"[IC] ⚠️ SL order placement failed for {sell_ce_symbol}")
        if not sl_pe_order:
            print(f"[IC] ⚠️ SL order placement failed for {sell_pe_symbol}")
        
        position = {
            'type': 'IC',
            'sell_ce': sell_ce,
            'buy_ce': buy_ce,
            'sell_pe': sell_pe,
            'buy_pe': buy_pe,
            'sell_ce_symbol': sell_ce_symbol,
            'buy_ce_symbol': buy_ce_symbol,
            'sell_pe_symbol': sell_pe_symbol,
            'buy_pe_symbol': buy_pe_symbol,
            'lots': lots,
            'quantity': quantity,
            'sell_ce_order': sell_ce_order,
            'buy_ce_order': buy_ce_order,
            'sell_pe_order': sell_pe_order,
            'buy_pe_order': buy_pe_order,
            'sl_ce_order': sl_ce_order,
            'sl_pe_order': sl_pe_order,
            'sell_ce_price': sell_ce_price,
            'buy_ce_price': buy_ce_price,
            'sell_pe_price': sell_pe_price,
            'buy_pe_price': buy_pe_price,
            'ce_credit': ce_credit,
            'pe_credit': pe_credit,
            'credit_received': total_credit,
            'entry_time': datetime.now(),
            'status': 'ACTIVE'
        }
        
        print(f"[IC] ✅ Iron Condor placed")
        print(f"  CE Credit: ₹{ce_credit:.2f}, PE Credit: ₹{pe_credit:.2f}")
        print(f"  Total Credit: ₹{total_credit:.2f}")
        
        return position
        
    except Exception as e:
        print(f"[ERROR] Failed to place Iron Condor: {e}")
        traceback.print_exc()
    
    return None

# ================================================================================
# EXIT LOGIC
# ================================================================================

async def check_exit_conditions(client, position, ltp_sell_ce, ltp_buy_ce, ltp_sell_pe, ltp_buy_pe):
    """
    Check if Iron Condor should be exited
    Returns: (should_exit, reason)
    """
    if position['status'] != 'ACTIVE':
        return False, None
    
    # Current spread values
    ce_spread_value = ltp_sell_ce - ltp_buy_ce
    pe_spread_value = ltp_sell_pe - ltp_buy_pe
    total_spread_value = ce_spread_value + pe_spread_value
    
    credit = position['credit_received']
    
    # Calculate P&L
    pnl = credit - total_spread_value
    pnl_pct = (pnl / credit) * 100 if credit > 0 else 0
    
    position['current_pnl'] = pnl
    position['current_pnl_pct'] = pnl_pct
    
    # Check Stop Loss: 50% increase in credit
    sl_threshold = credit * (1 + config.SL_CREDIT_INCREASE_PCT / 100)
    if total_spread_value >= sl_threshold:
        return True, f"SL: Spread value increased 50% ({total_spread_value:.2f} >= {sl_threshold:.2f})"
    
    # Alternative SL: Premium doubles
    if config.USE_PREMIUM_DOUBLE_SL:
        if total_spread_value >= credit * config.SL_PREMIUM_MULTIPLIER:
            return True, f"SL: Premium doubled ({total_spread_value:.2f} >= {credit * config.SL_PREMIUM_MULTIPLIER:.2f})"
    
    # Check Target: 40-60% credit capture
    if pnl_pct >= config.TARGET_MIN_PCT:
        return True, f"TARGET: {pnl_pct:.1f}% profit"
    
    # Check Time Exit
    current_time = datetime.now().time()
    if current_time >= config.MANDATORY_EXIT_TIME:
        return True, f"TIME_EXIT: {current_time}"
    
    return False, None

async def exit_iron_condor(client, position, reason):
    """
    Exit Iron Condor (close all 4 legs)
    """
    try:
        print(f"[EXIT] Closing Iron Condor - Reason: {reason}")
        
        if not config.PLACE_ORDERS:
            position['status'] = 'CLOSED'
            position['exit_time'] = datetime.now()
            position['exit_reason'] = reason
            print(f"[SIM] Iron Condor closed")
            return True
        
        quantity = position['quantity']
        
        # Cancel SL orders if exist
        for sl_key in ['sl_ce_order', 'sl_pe_order']:
            sl_order = position.get(sl_key)
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
        
        # Exit all 4 legs simultaneously
        close_ce_sell, close_ce_buy, close_pe_sell, close_pe_buy = await asyncio.gather(
            client.async_place_orders(position['sell_ce_symbol'], 'BUY', quantity, config.STRATEGY_NAME),
            client.async_place_orders(position['buy_ce_symbol'], 'SELL', quantity, config.STRATEGY_NAME),
            client.async_place_orders(position['sell_pe_symbol'], 'BUY', quantity, config.STRATEGY_NAME),
            client.async_place_orders(position['buy_pe_symbol'], 'SELL', quantity, config.STRATEGY_NAME)
        )
        
        all_closed = all([close_ce_sell, close_ce_buy, close_pe_sell, close_pe_buy])
        
        if all_closed:
            position['status'] = 'CLOSED'
            position['exit_time'] = datetime.now()
            position['exit_reason'] = reason
            print(f"[EXIT] ✅ Iron Condor closed successfully")
            return True
        else:
            print(f"[EXIT] ⚠️ Partial exit - some legs failed to close")
            return False
        
    except Exception as e:
        print(f"[ERROR] Failed to exit Iron Condor: {e}")
        traceback.print_exc()
    
    return False

# ================================================================================
# MAIN STRATEGY CLASS
# ================================================================================

class SensexIronCondorRunner:
    """Main Iron Condor strategy runner with WebSocket integration"""
    
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
        self.state = IronCondorState()
        
        # OHLC data for 5-minute candles
        self.ohlc = []
        self.ohlc_df = None
        self.curr_min = None
        self.open = self.high = self.low = self.close = None
        
        # Options symbols and LTP tracking
        self.spot_symbol = None
        self.ce_symbols = {}
        self.pe_symbols = {}
        self.ltp_cache = {}
        
        # ATM tracking
        self.current_atm = None
        self.nearest_expiry = None
        
        # Threading lock
        self.lock = threading.Lock()
        
    def now_ist(self):
        """Get current IST time"""
        return datetime.now(pytz.utc).astimezone(IST)
    
    def in_entry_window(self):
        """Check if current time is in entry window"""
        now = self.now_ist().time()
        return self.config.ENTRY_START_TIME <= now <= self.config.ENTRY_END_TIME
    
    def initialize_symbols(self):
        """Initialize Sensex spot and options symbols"""
        print("[INIT] Fetching Sensex symbols...")
        
        # Set exchange and symbol
        self.client.exchange = self.config.UNDERLYING_EXCHANGE
        self.client.symbol = self.config.UNDERLYING_SYMBOL
        self.client.strike_step = 100
        
        # Get nearest expiry
        try:
            self.nearest_expiry = self.client.get_nearest_expiry(
                symbol=self.config.UNDERLYING_SYMBOL,
                exchange=self.config.UNDERLYING_EXCHANGE
            )
            print(f"[INIT] Nearest expiry: {self.nearest_expiry}")
        except Exception as e:
            print(f"[ERROR] Failed to get expiry: {e}")
            return None
        
        # Get spot price
        try:
            quote = self.client.client.quotes(
                symbol=self.config.UNDERLYING_SYMBOL,
                exchange="BSE"
            )
            print(f"[DEBUG] Spot Quote: {quote}")
            
            spot_price = 0
            if quote and 'data' in quote and 'ltp' in quote['data']:
                spot_price = float(quote["data"]["ltp"])
            
            print(f"[INIT] Sensex spot price: {spot_price}")
            
            if spot_price <= 0:
                print("[WARN] Invalid spot price. Using Yahoo Finance fallback.")
                if self.state.prev_day_data:
                    spot_price = self.state.prev_day_data['close']
                    print(f"[INIT] Fallback Spot Price: {spot_price}")
                else:
                    print("[ERROR] No fallback data. Cannot determine ATM.")
                    return None
            
            self.current_atm = self.client.get_ATM_strike(spot_price)
            print(f"[INIT] ATM Strike: {self.current_atm}")
            
            self.spot_symbol = {'exchange': 'BSE', 'symbol': self.config.UNDERLYING_SYMBOL}
            
        except Exception as e:
            print(f"[ERROR] Failed to get spot price: {e}")
            traceback.print_exc()
            return None
        
        # Build strike range (±30 from ATM)
        strike_range = [
            self.current_atm + (i * self.client.strike_step) 
            for i in range(-30, 31)
        ]
        
        # Build CE and PE symbols
        subscription_symbols = []
        
        for strike in strike_range:
            ce_sym = self.client.build_option_symbol(
                self.config.UNDERLYING_SYMBOL,
                self.nearest_expiry,
                strike,
                "CE",
                exchange=self.config.UNDERLYING_EXCHANGE
            )
            self.ce_symbols[strike] = ce_sym
            subscription_symbols.append(ce_sym)
            
            pe_sym = self.client.build_option_symbol(
                self.config.UNDERLYING_SYMBOL,
                self.nearest_expiry,
                strike,
                "PE",
                exchange=self.config.UNDERLYING_EXCHANGE
            )
            self.pe_symbols[strike] = pe_sym
            subscription_symbols.append(pe_sym)
        
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
        """Calculate ATR indicator on current OHLC data"""
        # Only need 3+ candles for ATR (not 14-20 like ADX)
        if self.ohlc_df is None or len(self.ohlc_df) < 3:
            return False
        
        try:
            # Calculate ATR (only indicator needed for early morning entry)
            self.ohlc_df['atr'] = calculate_atr(
                self.ohlc_df,
                period=config.ATR_PERIOD
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
                    # Check for entry signal
                    if self.in_entry_window():
                        self.check_entry_signal()
                
                # Reset for new candle
                self.curr_min = candle_5min
                self.open = self.high = self.low = self.close = ltp
            else:
                # Update running candle
                self.close = ltp
                self.high = max(self.high, ltp)
                self.low = min(self.low, ltp)
            
            # Check exits on every tick for all active positions
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
    
    def check_entry_signal(self):
        """Check for Iron Condor entry signal (first tranche or scaling)"""
        if not self.state.can_enter(self.now_ist()):
            return
        
        if self.state.df is None or len(self.state.df) == 0:
            return
        
        current_spot = self.close
        
        # FIRST TRANCHE: Entry conditions (9:25-9:35 AM)
        if self.state.entry_count == 0:
            can_enter, reason = check_iron_condor_entry(
                self.state.df,
                current_spot,
                self.state.prev_day_data['close']
            )
            
            if not can_enter:
                print(f"[INFO] Entry conditions not met: {reason}")
                return
            
            print(f"[ENTRY] 🎯 Tranche 1 entry signal - Initial IC setup")
        
        # SUBSEQUENT TRANCHES: Premium expansion trigger
        else:
            # Check if total IC premium has increased by 10%
            current_total_premium = self.get_total_ic_premium()
            
            if current_total_premium <= 0 or self.state.initial_premium is None:
                return  # Wait for valid premium data
            
            premium_increase_pct = ((current_total_premium - self.state.initial_premium) / 
                                   self.state.initial_premium) * 100
            
            if premium_increase_pct < config.TRANCHE_PREMIUM_INCREASE_PCT:
                return  # Not enough premium expansion yet
            
            print(f"[ENTRY] 📈 Tranche {self.state.entry_count + 1} trigger - Premium increase: {premium_increase_pct:.1f}%")
            
            # Update baseline for next tranche
            self.state.initial_premium = current_total_premium
        
        # Get lot size for this tranche
        lots = self.state.get_entry_lots()
        
        # Get ATR for strike selection
        atr_value = self.state.df.iloc[-1]['atr']
        
        # Select strikes
        sell_ce, buy_ce, sell_pe, buy_pe = select_iron_condor_strikes(current_spot, atr_value)
        
        # Place order in separate thread
        threading.Thread(
            target=self._run_async_order,
            args=(sell_ce, buy_ce, sell_pe, buy_pe, lots)
        ).start()
    
    def get_total_ic_premium(self):
        """Calculate total current premium of all active IC positions"""
        total = 0
        
        for position in self.state.positions:
            if position['status'] != 'ACTIVE':
                continue
            
            # Get LTPs for all 4 legs
            ltp_sell_ce = self.ltp_cache.get(position['sell_ce_symbol'], 0)
            ltp_buy_ce = self.ltp_cache.get(position['buy_ce_symbol'], 0)
            ltp_sell_pe = self.ltp_cache.get(position['sell_pe_symbol'], 0)
            ltp_buy_pe = self.ltp_cache.get(position['buy_pe_symbol'], 0)
            
            if any([ltp_sell_ce <= 0, ltp_buy_ce <= 0, ltp_sell_pe <= 0, ltp_buy_pe <= 0]):
                continue  # Skip if LTPs not available
            
            # Current spread value (what we'd pay to close)
            ce_spread = ltp_sell_ce - ltp_buy_ce
            pe_spread = ltp_sell_pe - ltp_buy_pe
            total += ce_spread + pe_spread
        
        return total
    
    def _run_async_order(self, sell_ce, buy_ce, sell_pe, buy_pe, lots):
        """Helper to run async order placement"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            position = loop.run_until_complete(
                place_iron_condor(
                    self.client, sell_ce, buy_ce, sell_pe, buy_pe, lots, self.nearest_expiry
                )
            )
            
            if position:
                with self.lock:
                    self.state.add_position(position)
                print(f"[POSITION] Added Tranche {self.state.entry_count}: CE {position['sell_ce']}/{position['buy_ce']}, PE {position['sell_pe']}/{position['buy_pe']}, Lots: {lots}")
        
        except Exception as e:
            print(f"[ERROR] Order placement failed: {e}")
            traceback.print_exc()
        finally:
            loop.close()
    
    def check_exit_conditions_all(self):
        """Check exit conditions for all active IC positions"""
        if len(self.state.positions) == 0:
            return
        
        for position in self.state.positions[:]:  # Copy to allow modification
            if position['status'] != 'ACTIVE':
                continue
            
            # Get current LTPs for all 4 legs
            ltp_sell_ce = self.ltp_cache.get(position['sell_ce_symbol'], 0)
            ltp_buy_ce = self.ltp_cache.get(position['buy_ce_symbol'], 0)
            ltp_sell_pe = self.ltp_cache.get(position['sell_pe_symbol'], 0)
            ltp_buy_pe = self.ltp_cache.get(position['buy_pe_symbol'], 0)
            
            if any([ltp_sell_ce == 0, ltp_buy_ce == 0, ltp_sell_pe == 0, ltp_buy_pe == 0]):
                continue  # Wait for valid LTPs
            
            # Check if SL orders were triggered (priority check)
            for sl_key in ['sl_ce_order', 'sl_pe_order']:
                sl_order = position.get(sl_key)
                if sl_order:
                    sl_order_id = sl_order.get('orderid')
                    if sl_order_id:
                        sl_info = self.client.get_order_info_of_order(sl_order_id)
                        if sl_info and sl_info.get('order_status', '').lower() != 'trigger pending':
                            print(f"[SL HIT] Stop loss triggered - {sl_key} for Tranche")
                            
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                success = loop.run_until_complete(
                                    exit_iron_condor(self.client, position, "SL_ORDER_HIT")
                                )
                                if success:
                                    with self.lock:
                                        self.state.remove_position(position)
                                    print(f"[EXIT] ✅ Tranche closed due to SL")
                            except Exception as e:
                                print(f"[ERROR] SL exit failed: {e}")
                            finally:
                                loop.close()
                            return  # Process one at a time
            
            # Check spread-value based exits
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                should_exit, reason = loop.run_until_complete(
                    check_exit_conditions(
                        self.client, position, ltp_sell_ce, ltp_buy_ce, ltp_sell_pe, ltp_buy_pe
                    )
                )
                
                if should_exit:
                    print(f"[EXIT] Exit triggered: {reason}")
                    
                    success = loop.run_until_complete(
                        exit_iron_condor(self.client, position, reason)
                    )
                    
                    if success:
                        with self.lock:
                            self.state.remove_position(position)
                        print(f"[EXIT] ✅ Tranche closed: {reason}")
            
            except Exception as e:
                print(f"[ERROR] Exit check failed: {e}")
                traceback.print_exc()
            finally:
                loop.close()
    
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
    
    def check_entry_signal(self):
        """Check for Iron Condor entry signal (first tranche or scaling)"""
        if not self.state.can_enter(self.now_ist()):
            return
        
        if self.state.df is None or len(self.state.df) == 0:
            return
        
        current_spot = self.close
        
        # FIRST TRANCHE: Entry conditions (9:25-9:35 AM)
        if self.state.entry_count == 0:
            can_enter, reason = check_iron_condor_entry(
                self.state.df,
                current_spot,
                self.state.prev_day_data['close']
            )
            
            if not can_enter:
                print(f"[INFO] Entry conditions not met: {reason}")
                return
            
            print(f"[ENTRY] 🎯 Tranche 1 entry signal - Initial IC setup")
        
        # SUBSEQUENT TRANCHES: Premium expansion trigger
        else:
            # Check if total IC premium has increased by 10%
            current_total_premium = self.get_total_ic_premium()
            
            if current_total_premium <= 0 or self.state.initial_premium is None:
                return  # Wait for valid premium data
            
            premium_increase_pct = ((current_total_premium - self.state.initial_premium) / 
                                   self.state.initial_premium) * 100
            
            if premium_increase_pct < config.TRANCHE_PREMIUM_INCREASE_PCT:
                return  # Not enough premium expansion yet
            
            print(f"[ENTRY] 📈 Tranche {self.state.entry_count + 1} trigger - Premium increase: {premium_increase_pct:.1f}%")
            
            # Update baseline for next tranche
            self.state.initial_premium = current_total_premium
        
        # Get lot size for this tranche
        lots = self.state.get_entry_lots()
        
        # Get ATR for strike selection
        atr_value = self.state.df.iloc[-1]['atr']
        
        # Select strikes
        sell_ce, buy_ce, sell_pe, buy_pe = select_iron_condor_strikes(current_spot, atr_value)
        
        # Place order in separate thread
        threading.Thread(
            target=self._run_async_order,
            args=(sell_ce, buy_ce, sell_pe, buy_pe, lots)
        ).start()
    
    def get_total_ic_premium(self):
        """Calculate total current premium of all active IC positions"""
        total = 0
        
        for position in self.state.positions:
            if position['status'] != 'ACTIVE':
                continue
            
            # Get LTPs for all 4 legs
            ltp_sell_ce = self.ltp_cache.get(position['sell_ce_symbol'], 0)
            ltp_buy_ce = self.ltp_cache.get(position['buy_ce_symbol'], 0)
            ltp_sell_pe = self.ltp_cache.get(position['sell_pe_symbol'], 0)
            ltp_buy_pe = self.ltp_cache.get(position['buy_pe_symbol'], 0)
            
            if any([ltp_sell_ce <= 0, ltp_buy_ce <= 0, ltp_sell_pe <= 0, ltp_buy_pe <= 0]):
                continue  # Skip if LTPs not available
            
            # Current spread value (what we'd pay to close)
            ce_spread = ltp_sell_ce - ltp_buy_ce
            pe_spread = ltp_sell_pe - ltp_buy_pe
            total += ce_spread + pe_spread
        
        return total
    
    def _run_async_order(self, sell_ce, buy_ce, sell_pe, buy_pe, lots):
        """Helper to run async order placement"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            position = loop.run_until_complete(
                place_iron_condor(
                    self.client, sell_ce, buy_ce, sell_pe, buy_pe, lots, self.nearest_expiry
                )
            )
            
            if position:
                with self.lock:
                    self.state.add_position(position)
                print(f"[POSITION] Added Iron Condor: CE {position['sell_ce']}/{position['buy_ce']}, PE {position['sell_pe']}/{position['buy_pe']}")
        
        except Exception as e:
            print(f"[ERROR] Order placement failed: {e}")
            traceback.print_exc()
        finally:
            loop.close()
    
    def check_exit_conditions_ic(self):
        """Check exit conditions for Iron Condor"""
        position = self.state.position
        
        if not position or position['status'] != 'ACTIVE':
            return
        
        # Get current LTPs for all 4 legs
        ltp_sell_ce = self.ltp_cache.get(position['sell_ce_symbol'], 0)
        ltp_buy_ce = self.ltp_cache.get(position['buy_ce_symbol'], 0)
        ltp_sell_pe = self.ltp_cache.get(position['sell_pe_symbol'], 0)
        ltp_buy_pe = self.ltp_cache.get(position['buy_pe_symbol'], 0)
        
        if any([ltp_sell_ce == 0, ltp_buy_ce == 0, ltp_sell_pe == 0, ltp_buy_pe == 0]):
            return
        
        # Check if SL orders were triggered
        for sl_key in ['sl_ce_order', 'sl_pe_order']:
            sl_order = position.get(sl_key)
            if sl_order:
                sl_order_id = sl_order.get('orderid')
                if sl_order_id:
                    sl_info = self.client.get_order_info_of_order(sl_order_id)
                    if sl_info and sl_info.get('order_status', '').lower() != 'trigger pending':
                        print(f"[SL HIT] Stop loss triggered - {sl_key}")
                        
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            success = loop.run_until_complete(
                                exit_iron_condor(self.client, position, "SL_ORDER_HIT")
                            )
                            if success:
                                with self.lock:
                                    self.state.clear_position()
                                print(f"[EXIT] ✅ Iron Condor closed due to SL")
                        except Exception as e:
                            print(f"[ERROR] SL exit failed: {e}")
                        finally:
                            loop.close()
                        return
        
        # Check spread-value based exits
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            should_exit, reason = loop.run_until_complete(
                check_exit_conditions(
                    self.client, position, ltp_sell_ce, ltp_buy_ce, ltp_sell_pe, ltp_buy_pe
                )
            )
            
            if should_exit:
                print(f"[EXIT] Exit triggered: {reason}")
                
                success = loop.run_until_complete(
                    exit_iron_condor(self.client, position, reason)
                )
                
                if success:
                    with self.lock:
                        self.state.clear_position()
                    print(f"[EXIT] ✅ Iron Condor closed: {reason}")
        
        except Exception as e:
            print(f"[ERROR] Exit check failed: {e}")
            traceback.print_exc()
        finally:
            loop.close()
    
    def start(self):
        """Start the strategy runner"""
        print("="*80)
        print("SENSEX 1DTE WIDE OTM IRON CONDOR STRATEGY - TRANCHE SCALING")
        print("="*80)
        print(f"Strategy: {self.config.STRATEGY_NAME}")
        print(f"Underlying: {self.config.UNDERLYING_SYMBOL}")
        print(f"Exchange: {self.config.UNDERLYING_EXCHANGE}")
        print(f"Entry Window: {self.config.ENTRY_START_TIME} - {self.config.ENTRY_END_TIME}")
        print(f"Exit Time: {self.config.MANDATORY_EXIT_TIME}")
        print(f"ATR Threshold: < {self.config.MAX_ATR_THRESHOLD}")
        print(f"Max Gap: {self.config.MAX_OPENING_GAP_POINTS} points")
        print(f"")
        print(f"🎯 TRANCHE SCALING:")
        print(f"  Max Tranches: {self.config.MAX_ENTRIES}")
        print(f"  Lot Sizes: {self.config.ENTRY_LOTS} (scaled down)")
        print(f"  Trigger: {self.config.TRANCHE_PREMIUM_INCREASE_PCT}% premium increase")
        print(f"  Cutoff: {self.config.NO_MORE_TRANCHES_AFTER} (no more tranches)")
        print("="*80)
        
        # Fetch previous day data
        print("\n[INIT] Fetching previous day data from Yahoo Finance...")
        self.state.prev_day_data = fetch_previous_day_ohlc(self.config.YAHOO_SYMBOL)
        
        if not self.state.prev_day_data:
            print("[ERROR] Failed to fetch previous day data. Exiting.")
            return
        
        # Connect to OpenAlgo
        print("\n[INIT] Connecting to OpenAlgo...")
        self.client.connect()
        
        # Initialize and subscribe to symbols
        symbols = self.initialize_symbols()
        
        if not symbols:
            print("[ERROR] Failed to initialize symbols. Exiting.")
            self.client.disconnect()
            return
        
        # Subscribe to LTP, depth, and orderbook (matching run_credit_spread.py)
        print(f"\n[INIT] Subscribing to {len(symbols)} symbols...")
        self.client.subscribe_ltp(symbols)
        self.client.subscribe_depth(symbols)
        self.client.subscribe_orderbook()
        
        print("[INIT] ✅ Strategy initialized and listening for data...")
        print(f"[INIT] Previous Close: {self.state.prev_day_data['close']:.2f}")
        print("\n[RUNNING] Waiting for entry window (9:25-9:35 AM)...\n")
        
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
    runner = SensexIronCondorRunner()
    runner.start()
