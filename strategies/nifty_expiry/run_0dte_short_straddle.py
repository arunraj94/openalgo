# run_0dte_short_straddle.py
# Nifty 0 DTE Short Straddle Runner with Whipsaw Protection
# Progressive lock-in system + automatic leg management

import sys
import os
import threading
import asyncio
import logging
import time as time_module
from datetime import datetime, time as dt_time
from collections import deque
import pytz

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nifty_expiry.config_0dte_short_straddle import short_straddle_config as cfg
from openalgo_client import OpenAlgoClientWrapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(cfg.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

IST = pytz.timezone('Asia/Kolkata')


class ShortStraddleRunner:
    """
    Nifty 0 DTE Short Straddle with Whipsaw Protection
    
   

 Features:
    - Multiple entry windows (10:00, 11:30, 1:45)
    - RSI + ADX filtering
    - Progressive lock-in when one leg hits SL
    - Whipsaw detection (20% premium rise, RSI reversal, etc)
    - Automatic trailing stop loss
    """
    
    def __init__(self, client=None):
        self.cfg = cfg
        
        # Validate expiry day
        today = datetime.now(IST).weekday()
        if today != self.cfg.EXPIRY_DAY:
            logger.error(f"❌ Today is NOT expiry day (expected {self.cfg.EXPIRY_DAY}, got {today})")
            if self.cfg.SKIP_IF_NOT_EXPIRY:
                raise ValueError("Not an expiry day - strategy skipped")
        
        # Client
        if client:
            self.client = client
            self.client.add_listener(self.on_ltp)
            logger.info("Using shared OpenAlgo client")
        else:
            self.client = OpenAlgoClientWrapper(
                api_key=self.cfg.OPENALGO_API_KEY,
                host=self.cfg.OPENALGO_HOST,
                ws_url=self.cfg.OPENALGO_WS,
                on_ltp_callback=self.on_ltp
            )
        
        self.client.symbol = self.cfg.UNDERLYING_SYMBOL
        self.client.exchange = self.cfg.UNDERLYING_EXCHANGE
        self.client.strike_step = 50
        
        # Market Data (5-min candles)
        self.ohlc = []
        self.ohlc_df = None
        self.curr_min = None
        self.open = self.high = self.low = self.close = None
        
        # Indicators
        self.rsi = None
        self.adx = None
        self.atr = None
        self.prev_rsi = None  # For whipsaw detection
        
        # Volume tracking
        self.volume_history = deque(maxlen=20)
        self.avg_volume = 0
        
        # Position state
        self.straddle_active = False
        self.call_leg = None  # {'symbol', 'strike', 'entry_price', 'current_price', 'qty', 'status', 'sl_price'}
        self.put_leg = None
        self.entry_time = None
        self.entry_premium_total = 0
        self.daily_loss = 0.0
        
        # Whipsaw protection state
        self.leg_closed_by_sl = None  # 'CALL' or 'PUT' if one leg hit SL
        self.remaining_leg_low_premium = None  # Track lowest premium for trailing
        self.last_trail_time = None
        
        # Entry tracking
        self.entries_taken_today = 0
        self.entry_windows_used = []
        
        self.lock = threading.Lock()
        
        logger.info("=" * 70)
        logger.info("NIFTY 0DTE SHORT STRADDLE - WHIPSAW PROTECTED")
        logger.info("=" * 70)
        logger.info(f"Version: {self.cfg.VERSION}")
        logger.info(f"⚠️  REAL TRADING: {'ENABLED' if self.cfg.PLACE_ORDERS else 'DISABLED'}")
        logger.info(f"📅 Expiry Day: Thursday")
        logger.info(f"💰 Max Lots: {self.cfg.MAX_LOTS}")
        logger.info("=" * 70)
    
    def now_ist(self):
        return datetime.now(pytz.utc).astimezone(IST)
    
    def _run_async(self, coro):
        """Helper to run async functions"""
        return asyncio.run(coro)
    
    def in_trading_window(self):
        now = self.now_ist().time()
        return self.cfg.MARKET_OPEN <= now <= self.cfg.MARKET_CLOSE
    
    # ==================================================================
    # INDICATOR CALCULATION
    # ==================================================================
    
    def update_ohlc_df(self):
        import pandas as pd
        if len(self.ohlc) == 0:
            self.ohlc_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        else:
            self.ohlc_df = pd.DataFrame(self.ohlc)
    
    def compute_indicators(self):
        """Calculate RSI, ADX, ATR"""
        if self.ohlc_df is None or len(self.ohlc_df) < max(self.cfg.RSI_PERIOD, self.cfg.ADX_PERIOD, self.cfg.ATR_PERIOD) + 2:
            return
        
        try:
            import pandas_ta as ta
            
            # RSI
            self.ohlc_df['rsi'] = ta.rsi(self.ohlc_df['close'], length=self.cfg.RSI_PERIOD)
            
            # ADX
            adx_df = ta.adx(self.ohlc_df['high'], self.ohlc_df['low'], self.ohlc_df['close'], length=self.cfg.ADX_PERIOD)
            if adx_df is not None and not adx_df.empty:
                adx_col = f"ADX_{self.cfg.ADX_PERIOD}"
                if adx_col in adx_df.columns:
                    self.ohlc_df['adx'] = adx_df[adx_col]
            
            # ATR
            self.ohlc_df['atr'] = ta.atr(self.ohlc_df['high'], self.ohlc_df['low'], self.ohlc_df['close'], length=self.cfg.ATR_PERIOD)
            
            last_candle = self.ohlc_df.iloc[-1]
            self.prev_rsi = self.rsi  # Store previous for whipsaw detection
            self.rsi = last_candle.get('rsi')
            self.adx = last_candle.get('adx')
            self.atr = last_candle.get('atr')
            
            if self.cfg.VERBOSE:
                logger.info(f"[INDICATORS] RSI: {self.rsi:.1f}, ADX: {self.adx:.1f}, ATR: {self.atr:.1f}")
        
        except Exception as e:
            logger.error(f"Indicator error: {e}")
    
    # ==================================================================
    # ENTRY LOGIC
    # ==================================================================
    
    def get_current_entry_window(self):
        """Check if we're in an active entry window"""
        now = self.now_ist().time()
        
        for window in self.cfg.ENTRY_WINDOWS:
            if not window['enabled']:
                continue
            if window['name'] in self.entry_windows_used:
                continue  # Already used this window
            if window['start_time'] <= now <= window['end_time']:
                return window
        return None
    
    def check_range_condition(self, window):
        """Check if market range is within limits for entry"""
        if self.ohlc_df is None or len(self.ohlc_df) == 0:
            return False, "No data"
        
        # For Window 1: Check first 45 mins (9:15-10:00)
        if window['name'] == 'WINDOW_1_MORNING':
            first_45_data = self.ohlc_df[self.ohlc_df['timestamp'] >= datetime.combine(datetime.today(), dt_time(9, 15)).replace(tzinfo=IST)]
            if len(first_45_data) > 0:
                range_45 = first_45_data['high'].max() - first_45_data['low'].min()
                if range_45 > window['max_range']:
                    return False, f"45-min range too large ({range_45:.0f} pts)"
        
        # For Window 2: Check morning range (9:15-11:30)
        elif window['name'] == 'WINDOW_2_MIDDAY':
            morning_range = self.ohlc_df['high'].max() - self.ohlc_df['low'].min()
            if morning_range > window['max_range']:
                return False, f"Morning range too large ({morning_range:.0f} pts)"
        
        return True, "Range OK"
    
    def check_entry_conditions(self, window):
        """Check all entry conditions for given window"""
        
        # Check if already took max entries
        if self.entries_taken_today >= self.cfg.MAX_ENTRIES_PER_DAY:
            return False, "Max entries reached"
        
        # Check daily loss limit
        if self.daily_loss >= self.cfg.MAX_LOSS_PER_DAY:
            return False, "Daily loss limit hit"
        
        # Range check
        range_ok, range_msg = self.check_range_condition(window)
        if not range_ok:
            return False, range_msg
        
        # Indicator checks (if enabled)
        if self.cfg.USE_INDICATORS:
            if self.rsi is None or self.adx is None:
                return False, "Indicators not ready"
            
            # RSI (sideways market)
            if not (self.cfg.RSI_LOWER <= self.rsi <= self.cfg.RSI_UPPER):
                return False, f"RSI not sideways ({self.rsi:.1f})"
            
            # ADX (non-trending)
            if self.adx >= self.cfg.ADX_MAX:
                return False, f"ADX too high ({self.adx:.1f})"
        
        # Volume check (avoid spikes)
        if self.avg_volume > 0:
            if len(self.ohlc_df) > 0:
                recent_vol = self.ohlc_df.iloc[-1].get('volume', 0)
                if recent_vol > self.avg_volume * self.cfg.VOLUME_SPIKE_THRESHOLD:
                    return False, "Volume spike detected"
        
        return True, "All conditions met"
    
    # ==================================================================
    # ORDER PLACEMENT
    # ==================================================================
    
    async def place_straddle_async(self, atm_strike, qty):
        """Place ATM straddle (Call + Put)"""
        
        expiry = self.client.nearest_expiry
        
        # Build symbols
        call_info = self.client.build_option_symbol(
            self.cfg.UNDERLYING_SYMBOL, expiry, atm_strike, 'CE')
        put_info = self.client.build_option_symbol(
            self.cfg.UNDERLYING_SYMBOL, expiry, atm_strike, 'PE')
        
        call_symbol = call_info['symbol']
        put_symbol = put_info['symbol']
        
        logger.info(f"Placing Straddle: {call_symbol} + {put_symbol}")
        
        # Place both legs in parallel
        call_order, put_order = await asyncio.gather(
            self.client.async_place_orders(call_symbol, 'SELL', qty, strategy_tag="NIFTY_STRADDLE"),
            self.client.async_place_orders(put_symbol, 'SELL', qty, strategy_tag="NIFTY_STRADDLE")
        )
        
        return call_order, put_order, call_symbol, put_symbol
    
    def enter_straddle(self):
        """Enter short straddle position"""
        logger.info("=" * 70)
        logger.info("ENTERING SHORT STRADDLE")
        logger.info("=" * 70)
        
        ltp = self.client.get_exchange_ltp()
        atm = self.client.get_ATM_strike(ltp)
        
        logger.info(f"Spot: {ltp:.2f}, ATM: {atm}")
        
        qty = self.cfg.LOTSIZE * self.cfg.MAX_LOTS
        
        try:
            call_order, put_order, call_symbol, put_symbol = self._run_async(
                self.place_straddle_async(atm, qty)
            )
            
            # Validate
            if not call_order or call_order.get('order_status') != 'complete':
                logger.error("❌ Call order failed")
                return
            if not put_order or put_order.get('order_status') != 'complete':
                logger.error("❌ Put order failed")
                return
            
            # Get prices
            call_price = float(call_order.get('average_price', call_order.get('price', 0)))
            put_price = float(put_order.get('average_price', put_order.get('price', 0)))
            
            total_premium = call_price + put_price
            
            # Check minimum premium
            if total_premium < self.cfg.MIN_PREMIUM_COLLECTED:
                logger.warning(f"⚠️  Premium ({total_premium:.0f}) less than minimum ({self.cfg.MIN_PREMIUM_COLLECTED})")
                # Could exit here if desired
            
            # Store legs
            self.call_leg = {
                'symbol': call_symbol,
                'strike': atm,
                'entry_price': call_price,
                'current_price': call_price,
                'qty': qty,
                'status': 'OPEN',
                'sl_price': call_price * (1 + self.cfg.SL_PCT_PER_LEG / 100.0),
                'order': call_order
            }
            
            self.put_leg = {
                'symbol': put_symbol,
                'strike': atm,
                'entry_price': put_price,
                'current_price': put_price,
                'qty': qty,
                'status': 'OPEN',
                'sl_price': put_price * (1 + self.cfg.SL_PCT_PER_LEG / 100.0),
                'order': put_order
            }
            
            # ------------------------------------------------------------------
            # PLACE STOP LOSS ORDERS
            # ------------------------------------------------------------------
            logger.info("🛡️ Placing Stop Loss Orders...")
            
            try:
                ce_sl_order, pe_sl_order = self._run_async(asyncio.gather(
                    self.client.async_sl_order(call_order, self.cfg.SL_PCT_PER_LEG, strategy_tag="NIFTY_STRADDLE_SL"),
                    self.client.async_sl_order(put_order, self.cfg.SL_PCT_PER_LEG, strategy_tag="NIFTY_STRADDLE_SL")
                ))
                
                if ce_sl_order:
                    self.call_leg['sl_order'] = ce_sl_order
                    logger.info(f"✅ Call SL Placed: {ce_sl_order.get('orderid')}")
                else:
                    logger.error("❌ Failed to place Call SL")
                    
                if pe_sl_order:
                    self.put_leg['sl_order'] = pe_sl_order
                    logger.info(f"✅ Put SL Placed: {pe_sl_order.get('orderid')}")
                else:
                    logger.error("❌ Failed to place Put SL")
                    
            except Exception as e:
                logger.error(f"❌ Error placing SL orders: {e}")
            
            self.entry_premium_total = total_premium * qty
            self.entry_time = self.now_ist()
            self.straddle_active = True
            self.entries_taken_today += 1
            
            logger.info(f"✅ Straddle Entered:")
            logger.info(f"   Call {atm}CE @ {call_price:.1f} (SL: {self.call_leg['sl_price']:.1f})")
            logger.info(f"   Put {atm}PE @ {put_price:.1f} (SL: {self.put_leg['sl_price']:.1f})")
            logger.info(f"   Total Premium: ₹{self.entry_premium_total:,.0f}")
            logger.info("=" * 70)
        
        except Exception as e:
            logger.error(f"❌ Entry failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================================================================
    # POSITION MANAGEMENT & WHIPSAW PROTECTION
    # ==================================================================
    
    def calculate_pnl(self):
        """Calculate current P&L"""
        if not self.straddle_active:
            return 0, 0
        
        call_pnl = 0
        put_pnl = 0
        
        if self.call_leg and self.call_leg['status'] == 'OPEN':
            call_pnl = (self.call_leg['entry_price'] - self.call_leg['current_price']) * self.call_leg['qty']
        
        if self.put_leg and self.put_leg['status'] == 'OPEN':
            put_pnl = (self.put_leg['entry_price'] - self.put_leg['current_price']) * self.put_leg['qty']
        
        total_pnl = call_pnl + put_pnl
        pnl_pct = (total_pnl / self.entry_premium_total) * 100 if self.entry_premium_total > 0 else 0
        
        return total_pnl, pnl_pct
    
    def apply_progressive_lockin(self, leg):
        """
        Apply 50-30-15 rule to set SL with profit buffer
        Returns new SL price
        """
        current_premium = leg['current_price']
        entry_premium = leg['entry_price']
        
        # Determine buffer based on current premium
        if current_premium >= 50:
            buffer = self.cfg.LOCK_IN_RULES['premium_gt_50']['buffer_points']
        elif 30 <= current_premium < 50:
            buffer = self.cfg.LOCK_IN_RULES['premium_30_50']['buffer_points']
        elif 20 <= current_premium < 30:
            buffer = self.cfg.LOCK_IN_RULES['premium_20_30']['buffer_points']
        elif current_premium < 20:
            if current_premium < 10:
                # Close immediately
                return 'CLOSE_NOW'
            buffer = self.cfg.LOCK_IN_RULES['premium_lt_20']['buffer_points']
        
        # New SL = entry_price - buffer (locks profit)
        new_sl = entry_premium - buffer
        
        logger.info(f"🔒 Progressive Lock-In: Premium {current_premium:.0f}, Buffer {buffer}, New SL {new_sl:.0f}")
        
        return new_sl
    
    def detect_whipsaw_signals(self, leg, leg_type):
        """
        Detect if whipsaw is happening (market reversing)
        Returns True if should close immediately
        """
        if not self.cfg.ENABLE_WHIPSAW_PROTECTION:
            return False
        
        current_premium = leg['current_price']
        low_premium = self.remaining_leg_low_premium or current_premium
        
        # Signal 1: Premium rising 20%+
        if low_premium > 0:
            rise_pct = ((current_premium - low_premium) / low_premium) * 100
            if rise_pct >= self.cfg.WHIPSAW_SIGNALS['premium_rise_pct']:
                logger.warning(f"🚨 WHIPSAW SIGNAL 1: Premium rose {rise_pct:.1f}%")
                return True
       
        # Signal 2: RSI reversal
        if self.prev_rsi and self.rsi:
            rsi_jump = abs(self.rsi - self.prev_rsi)
            if rsi_jump >= self.cfg.WHIPSAW_SIGNALS['rsi_reversal_threshold']:
                logger.warning(f"🚨 WHIPSAW SIGNAL 3: RSI jumped {rsi_jump:.0f} points")
                return True
        
        # Signal 4: Time cutoff
        now = self.now_ist().time()
        if now >= self.cfg.WHIPSAW_SIGNALS['time_cutoff']:
            if current_premium < leg['entry_price']:  # Any profit
                logger.warning(f"🚨 WHIPSAW SIGNAL 4: After 2:30 PM, closing at profit")
                return True
        
        return False
    
    async def modify_broker_sl_async(self, leg, new_price):
        """Modify the broker SL order to new price"""
        if not leg.get('sl_order'):
            return
        
        order_id = leg['sl_order']['orderid']
        symbol = leg['symbol']
        qty = leg['qty']
        
        # For SELL position, SL is BUY.
        # Trigger = new_price
        # Limit = new_price + 5% (to ensure fill)
        trigger_price = round(new_price, 1)
        limit_price = round(new_price * 1.05, 1)
        
        try:
            # Use client's modify method
            # Note: Using client.client directly or wrapper if available. 
            # The wrapper has modifyorder_async but it takes many args.
            # Let's use the wrapper's method.
            
            await self.client.client.modifyorder_async(
                strategy="NIFTY_STRADDLE_SL",
                order_id=order_id,
                symbol=symbol,
                action='BUY',
                exchange=self.cfg.UNDERLYING_EXCHANGE,
                price_type="SL",
                product='NRML',
                quantity=qty,
                price=limit_price,
                trigger_price=trigger_price
            )
            logger.info(f"✅ Broker SL Modified: {trigger_price:.1f}")
        except Exception as e:
            logger.error(f"❌ Failed to modify SL: {e}")

    def manage_single_leg(self, leg, leg_type):
        """
        Manage remaining leg after one leg hit SL
        Apply progressive lock-in and whipsaw detection
        """
        if not leg or leg['status'] != 'OPEN':
            return
        
        current_premium = leg['current_price']
        
        # Track lowest premium for whipsaw detection
        if self.remaining_leg_low_premium is None:
            self.remaining_leg_low_premium = current_premium
        else:
            self.remaining_leg_low_premium = min(self.remaining_leg_low_premium, current_premium)
        
        # Check whipsaw signals
        if self.detect_whipsaw_signals(leg, leg_type):
            logger.warning(f"⚠️  Whipsaw detected! Closing {leg_type} immediately")
            self.close_leg(leg, leg_type, "WHIPSAW_DETECTED")
            return
        
        # Apply progressive lock-in
        new_sl = self.apply_progressive_lockin(leg)
        
        if new_sl == 'CLOSE_NOW':
            logger.info(f"✂️  Premium < 10, closing {leg_type} at market")
            self.close_leg(leg, leg_type, "PREMIUM_LT_10")
            return
        
        # Update SL (trail)
        if new_sl < leg['sl_price']:
            logger.info(f"📈 Trailing {leg_type} SL: {leg['sl_price']:.0f} → {new_sl:.0f}")
            current_price = leg.get('current_price', 0)
            if current_price >= new_sl:
                logger.warning(f"⚠️ New SL {new_sl:.1f} already breached by CMP {current_price:.1f}. Exiting {leg_type} now.")
                self.close_leg(leg, leg_type, "SL_BREACHED_IMMEDIATELY")
            else:
                # Update internal SL price
                leg['sl_price'] = new_sl
                logger.info(f"📉 Updating {leg_type} SL to {new_sl:.1f}")
                
                # Modify the actual Broker SL Order
                self._run_async(self.modify_broker_sl_async(leg, new_sl))
        
        # Check if SL hit (Broker Status)
        if leg.get('sl_order'):
            sl_id = leg['sl_order']['orderid']
            info = self.client.get_order_info_of_order(sl_id)
            if info and info.get('order_status','').lower() != 'trigger pending':
                 self.handle_sl_execution(leg, leg_type, info)
                 return
    
    def close_leg(self, leg, leg_type, reason):
        """Close a single leg"""
        logger.info(f"Closing {leg_type} leg: {reason}")
        
        try:
            # 1. Cancel SL Order first if it exists
            if 'sl_order' in leg and leg['sl_order']:
                sl_id = leg['sl_order'].get('orderid')
                if sl_id:
                    logger.info(f"Cancelling SL order {sl_id} for {leg_type}...")
                    try:
                        self._run_async(self.client.client.cancelorder_async(sl_id, strategy_tag="NIFTY_STRADDLE_EXIT"))
                    except Exception as e:
                        logger.error(f"Failed to cancel SL {sl_id}: {e}")
            
            # 2. Place exit order
            qty = leg['qty']
            symbol = leg['symbol']
            
            # BUY to close SELL position
            order = self._run_async(
                self.client.async_place_orders(symbol, 'BUY', qty, strategy_tag="NIFTY_STRADDLE_EXIT")
            )
            
            if order and order.get('order_status') == 'complete':
                exit_price = float(order.get('average_price', order.get('price', leg['current_price'])))
                pnl = (leg['entry_price'] - exit_price) * qty
                
                logger.info(f"✅ {leg_type} closed @ {exit_price:.1f}, P&L: ₹{pnl:,.0f}")
                
                leg['status'] = 'CLOSED'
                leg['exit_price'] = exit_price
                leg['exit_reason'] = reason
                
                # Update daily loss
                if pnl < 0:
                    self.daily_loss += abs(pnl)
            
        except Exception as e:
            logger.error(f"❌ Failed to close {leg_type}: {e}")
    
    def handle_sl_execution(self, leg, leg_type, order_info):
        """Handle case where SL order was executed by broker"""
        logger.warning(f"🛑 {leg_type} SL Executed by Broker")
        
        exit_price = float(order_info.get('average_price', order_info.get('price', leg['current_price'])))
        pnl = (leg['entry_price'] - exit_price) * leg['qty']
        
        logger.info(f"✅ {leg_type} SL closed @ {exit_price:.1f}, P&L: ₹{pnl:,.0f}")
        
        leg['status'] = 'CLOSED'
        leg['exit_price'] = exit_price
        leg['exit_reason'] = "SL_HIT_BROKER"
        
        if pnl < 0:
            self.daily_loss += abs(pnl)
            
        self.leg_closed_by_sl = leg_type
        
        # Manage the other leg
        other_leg = self.put_leg if leg_type == 'CALL' else self.call_leg
        other_type = 'PUT' if leg_type == 'CALL' else 'CALL'
        
        if other_leg['status'] == 'OPEN':
            if self.cfg.LEG_MANAGEMENT_STRATEGY == "CONVERT_TO_NAKED":
                logger.info(f"🔄 Converting to NAKED {other_type} position")
                
                # Calculate new SL based on progressive lock-in
                new_sl = self.apply_progressive_lockin(other_leg)
                
                if new_sl == 'CLOSE_NOW':
                    logger.info(f"✂️ Premium < 10, closing {other_type} immediately")
                    self.close_leg(other_leg, other_type, "PREMIUM_LT_10")
                else:
                    # Check if current price has already breached the new SL
                    current_price = other_leg.get('current_price', 0)
                    if current_price >= new_sl:
                        logger.warning(f"⚠️ New SL {new_sl:.1f} already breached by CMP {current_price:.1f}. Exiting {other_type} now.")
                        self.close_leg(other_leg, other_type, "SL_BREACHED_IMMEDIATELY")
                    else:
                        # Update internal SL price
                        other_leg['sl_price'] = new_sl
                        logger.info(f"📉 Updating {other_type} SL to {new_sl:.1f}")
                        
                        # Modify the actual Broker SL Order
                        self._run_async(self.modify_broker_sl_async(other_leg, new_sl))
            else:
                logger.info(f"🔄 Closing {other_type} as well (CLOSE_BOTH strategy)")
                self.close_leg(other_leg, other_type, "PAIRED_CLOSURE")

    def manage_position(self):
        """Main position management logic"""
        if not self.straddle_active:
            return
        
        total_pnl, pnl_pct = self.calculate_pnl()
        
        # Check if one leg already closed + managing other
        if self.leg_closed_by_sl:
            remaining_leg = self.call_leg if self.leg_closed_by_sl == 'PUT' else self.put_leg
            leg_type = 'CALL' if self.leg_closed_by_sl == 'PUT' else 'PUT'
            
            self.manage_single_leg(remaining_leg, leg_type)
            
            # Check if both legs now closed
            if self.call_leg['status'] == 'CLOSED' and self.put_leg['status'] == 'CLOSED':
                self.straddle_active = False
                logger.info(f"✅ Straddle fully closed. Final P&L: ₹{total_pnl:,.0f} ({pnl_pct:.1f}%)")
            
            return
        
        # Both legs still active - check normal SLs and targets
        
        # Profit target
        if pnl_pct >= self.cfg.PROFIT_TARGET_PCT:
            logger.info(f"🎯 PROFIT TARGET HIT: {pnl_pct:.1f}%")
            self.exit_straddle("PROFIT_TARGET")
            return
        
        # Combined SL
        if pnl_pct <= -self.cfg.SL_PCT_COMBINED:
            logger.info(f"🛑 COMBINED SL HIT: {pnl_pct:.1f}%")
            self.exit_straddle("COMBINED_SL")
            return
        
        # ------------------------------------------------------------------
        # CHECK BROKER SL ORDER STATUS (Instead of local price check)
        # ------------------------------------------------------------------
        
        # Check CALL SL
        if self.call_leg['status'] == 'OPEN' and self.call_leg.get('sl_order'):
            sl_id = self.call_leg['sl_order']['orderid']
            info = self.client.get_order_info_of_order(sl_id)
            if info and info.get('order_status','').lower() != 'trigger pending':
                 self.handle_sl_execution(self.call_leg, 'CALL', info)
                 return

        # Check PUT SL
        if self.put_leg['status'] == 'OPEN' and self.put_leg.get('sl_order'):
            sl_id = self.put_leg['sl_order']['orderid']
            info = self.client.get_order_info_of_order(sl_id)
            if info and info.get('order_status','').lower() != 'trigger pending':
                 self.handle_sl_execution(self.put_leg, 'PUT', info)
                 return
        
        # Time-based exits
        now = self.now_ist().time()
        
        if now >= self.cfg.MANDATORY_EXIT_TIME:
            logger.info(f"⏰ MANDATORY EXIT: {now}")
            self.exit_straddle("TIME_EXIT")
            return
        
        if now >= self.cfg.PROFIT_EXIT_TIME and pnl_pct > 10:
            logger.info(f"🕐 PROFIT EXIT at {now}: {pnl_pct:.1f}%")
            self.exit_straddle("EARLY_PROFIT")
            return
    
    def exit_straddle(self, reason):
        """Exit both legs of straddle"""
        logger.info("=" * 70)
        logger.info(f"EXITING STRADDLE: {reason}")
        logger.info("=" * 70)
        
        if self.call_leg and self.call_leg['status'] == 'OPEN':
            self.close_leg(self.call_leg, 'CALL', reason)
        
        if self.put_leg and self.put_leg['status'] == 'OPEN':
            self.close_leg(self.put_leg, 'PUT', reason)
        
        total_pnl, pnl_pct = self.calculate_pnl()
        logger.info(f"Final P&L: ₹{total_pnl:,.0f} ({pnl_pct:.1f}%)")
        logger.info("=" * 70)
        
        self.straddle_active = False
    
    # ==================================================================
    # TICK PROCESSING
    # ==================================================================
    
    def on_underlying_tick(self, tick):
        """Process underlying (futures) tick"""
        ltp = tick.get('ltp')
        if ltp is None:
            return
        
        dt = self.now_ist()
        minute = dt.replace(second=0, microsecond=0)
        candle_5min = minute.replace(minute=(minute.minute // 5) * 5)
        
        with self.lock:
            # 5-min candle logic
            if self.curr_min is None:
                self.curr_min = candle_5min
                self.open = self.high = self.low = self.close = ltp
                return
            
            if candle_5min != self.curr_min:
                # New candle
                row = {
                    'timestamp': self.curr_min,
                    'open': self.open,
                    'high': self.high,
                    'low': self.low,
                    'close': self.close,
                    'volume': 0  # Would need actual volume data
                }
                self.ohlc.append(row)
                self.update_ohlc_df()
                self.compute_indicators()
                
                if self.cfg.LOG_CANDLE_COMPLETION:
                    logger.info(f"[5-MIN] O={self.open:.0f} H={self.high:.0f} L={self.low:.0f} C={self.close:.0f}")
                
                # Entry/Management logic
                if self.straddle_active:
                    self.manage_position()
                else:
                    # Check for entry
                    window = self.get_current_entry_window()
                    if window:
                        can_enter, msg = self.check_entry_conditions(window)
                        if can_enter:
                            logger.info(f"✅ Entry signal in {window['name']}")
                            self.entry_windows_used.append(window['name'])
                            self.enter_straddle()
                        else:
                            logger.debug(f"Entry check: {msg}")
                
                self.curr_min = candle_5min
                self.open = self.high = self.low = self.close = ltp
            else:
                # Update current candle
                self.close = ltp
                self.high = max(self.high, ltp)
                self.low = min(self.low, ltp)
                
                # Monitor position every tick if active
                if self.straddle_active:
                    self.manage_position()
    
    def on_option_tick(self, symbol, ltp):
        """Update option leg prices"""
        if self.call_leg and symbol == self.call_leg['symbol']:
            self.call_leg['current_price'] = ltp
        if self.put_leg and symbol == self.put_leg['symbol']:
            self.put_leg['current_price'] = ltp
    
    def on_ltp(self, data):
        """Main LTP callback"""
        if data.get('type') != 'market_data':
            return
        
        symbol = data.get('symbol')
        exch = data.get('exchange')
        tick = data.get('data', {})
        
        if exch != self.cfg.UNDERLYING_EXCHANGE:
            return
        
        if symbol == self.cfg.UNDERLYING_SYMBOL:
            self.on_underlying_tick(tick)
        else:
            self.on_option_tick(symbol, tick.get('ltp'))
    
    # ==================================================================
    # START / MAIN LOOP
    # ==================================================================
    
    def start(self):
        """Start the strategy"""
        logger.info("🚀 Starting Nifty 0DTE Short Straddle...")
        logger.info(f"⚠️  REAL TRADING: {self.cfg.PLACE_ORDERS}")
        
        self.client.connect()
        
        if not self.client.connected:
            logger.error("❌ Failed to connect")
            return
        
        # Get expiry and subscribe to underlying index
        self.client.nearest_expiry = self.client.get_nearest_expiry(self.cfg.UNDERLYING_SYMBOL)
        logger.info(f"Underlying: {self.cfg.UNDERLYING_SYMBOL}")
        logger.info(f"Expiry: {self.client.nearest_expiry}")
        
        # Set future_symbol to Underlying for get_exchange_ltp() to work with Spot
        self.client.future_symbol = self.cfg.UNDERLYING_SYMBOL
        
        # Subscribe to underlying index (spot)
        self.client.subscribe_ltp([self.cfg.UNDERLYING_SYMBOL])
        self.client.subscribe_depth([self.cfg.UNDERLYING_SYMBOL])
        
        # Subscribe to options (Custom logic for Nifty Spot)
        try:
            # Get Spot Price
            spot_quote = self.client.get_quote(self.cfg.UNDERLYING_SYMBOL)
            if spot_quote:
                spot_price = spot_quote.get('ltp')
                atm_strike = self.client.get_ATM_strike(spot_price)
                
                # Generate symbols around ATM
                strike_range = [atm_strike + (i * 50) for i in range(-20, 21)]
                expiry = self.client.nearest_expiry
                
                option_symbols = []
                for strike in strike_range:
                    ce = self.client.build_option_symbol(self.cfg.UNDERLYING_SYMBOL, expiry, strike, "CE")
                    pe = self.client.build_option_symbol(self.cfg.UNDERLYING_SYMBOL, expiry, strike, "PE")
                    option_symbols.append(ce['symbol'])
                    option_symbols.append(pe['symbol'])
                
                logger.info(f"Subscribing to {len(option_symbols)} option symbols (ATM {atm_strike})...")
                self.client.subscribe_ltp(option_symbols)
                self.client.subscribe_depth(option_symbols)
            else:
                logger.warning("⚠️ Could not fetch Spot Quote. Waiting for tick data...")
                
        except Exception as e:
            logger.error(f"Error subscribing to options: {e}")
        
        logger.info("✅ Subscribed to market data")
        logger.info(f"🕐 Waiting for entry windows...")
        
        try:
            while self.in_trading_window():
                time_module.sleep(60)
            
            logger.info("⏰ Market closed")
            
            if self.straddle_active:
                self.exit_straddle("MARKET_CLOSE")
        
        except KeyboardInterrupt:
            logger.info("⏹️  Stopped by user")
        finally:
            if self.straddle_active:
                self.exit_straddle("MANUAL_EXIT")
            self.client.disconnect()


if __name__ == "__main__":
    runner = ShortStraddleRunner()
    runner.start()
