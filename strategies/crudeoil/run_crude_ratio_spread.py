# run_crude_ratio_spread.py
# Protected Ratio Spread Runner for Crude Oil - Evening Session
# 5-minute OHLC for signals, 1-minute for execution
# DEFINED RISK - No unlimited loss

import time
from datetime import datetime, timedelta
import pytz
import threading
import asyncio
import pandas as pd
from typing import Optional, Literal

# Import modules
from config_crude_ratio_spread import crude_ratio_config as cfg
from entries_directional import DirectionalEntries
from db_logger import init_db, log_entry, log_exit
from openalgo_client import OpenAlgoClientWrapper
from position_sizing import compute_lots_from_config
from logger import setup_logger
from strategy_coordinator import get_coordinator

# Setup
logger = setup_logger("CRUDE_RATIO_SPREAD")
IST = pytz.timezone("Asia/Kolkata")

logger.info("🚀 Crude Oil Protected Ratio Spread Runner Starting...")
logger.info(f"   Session: {cfg.SESSION_START_TIME} - {cfg.SESSION_END_TIME}")
logger.info(f"   OHLC Timeframe: 5-minute (signals)")
logger.info(f"   Max Loss: {cfg.MAX_LOSS_PERCENT}% or ₹{cfg.MAX_LOSS_RUPEES}")


class ProtectedRatioSpreadRunner:
    """
    Protected Ratio Spread Strategy Runner
    
    Features:
    - 5-minute OHLC for trend signals
    - 1-minute ticks for execution
    - Multi-layered stop losses
    - Phase 1 (1:1) + Phase 2 (1:2 + protection wing)
    - Defined risk at all times
    """
    
    def __init__(self, config=cfg):
        self.cfg = config
        
        # Directional signal detector
        self.detector = DirectionalEntries()
        
        # Database
        self.db_session = init_db(self.cfg.DB_PATH)
        
        # Strategy Coordinator (prevent conflict with directional strategy)
        self.coordinator = get_coordinator()
        
        # OpenAlgo client
        self.client = OpenAlgoClientWrapper(
            api_key=self.cfg.OPENALGO_API_KEY,
            host=self.cfg.OPENALGO_HOST,
            ws_url=self.cfg.OPENALGO_WS,
            on_ltp_callback=self.on_ltp
        )
        
        # OHLC Data (5-minute candles)
        self.ohlc_5min = []
        self.ohlc_5min_df = None
        self.current_5min_candle = None
        self.current_5min_open = None
        self.current_5min_high = None
        self.current_5min_low = None
        self.current_5min_close = None
        
        # Session tracking
        self.session_start_price = None
        self.current_spot = None
        
        # Position State
        self.phase = 0  # 0=no position, 1=Phase 1, 2=Phase 2
        self.direction = None  # "BULLISH" or "BEARISH"
        self.entry_time = None
        self.phase1_entry_spot = None
        
        # Position Legs
        self.long_leg = None  # Buy ATM
        self.short_leg_1 = None  # Sell ATM+50 (Phase 1)
        self.short_leg_2 = None  # Sell ATM+50 (Phase 2)
        self.protection_wing = None  # Buy ATM+150 (Phase 2)
        
        # P&L Tracking
        self.phase1_pnl = 0.0
        self.total_pnl = 0.0
        self.highest_profit = 0.0
        self.entry_capital = 0.0
        
        # Stop Loss State
        self.trailing_active = False
        self.quick_sl_triggered = False
        
        # Trade Management
        self.current_trade_id = None
        self.last_check_time = None
        
        self.lock = threading.Lock()
    
    # ========================================
    # Utility Methods
    # ========================================
    
    def now_ist(self):
        return datetime.now(pytz.utc).astimezone(IST)
    
    def in_session(self):
        """Check if in evening session"""
        now = self.now_ist().time()
        return self.cfg.SESSION_START_TIME <= now < self.cfg.SESSION_END_TIME
    
    def in_entry_window(self):
        """Check if in entry window"""
        now = self.now_ist().time()
        return self.cfg.ENTRY_WINDOW_START <= now < self.cfg.ENTRY_WINDOW_END
    
    def is_wednesday_eia_risk(self):
        """Check if Wednesday EIA report risk"""
        now = self.now_ist()
        if now.weekday() == cfg.EIA_REPORT_DAY:  # Wednesday
            if now.time() >= cfg.EIA_EXIT_TIME:
                return True
        return False
    
    def minutes_since_entry(self):
        """Minutes since position entry"""
        if not self.entry_time:
            return 0
        return (self.now_ist() - self.entry_time).total_seconds() / 60
    
    # ========================================
    # OHLC Management (5-minute candles)
    # ========================================
    
    def update_5min_candle(self, ltp, timestamp):
        """Update 5-minute OHLC candle"""
        candle_time = timestamp.replace(second=0, microsecond=0)
        candle_5min = candle_time.replace(minute=(candle_time.minute // 5) * 5)
        
        with self.lock:
            if self.current_5min_candle is None:
                # First candle
                self.current_5min_candle = candle_5min
                self.current_5min_open = ltp
                self.current_5min_high = ltp
                self.current_5min_low = ltp
                self.current_5min_close = ltp
                return False  # Not complete
            
            if candle_5min != self.current_5min_candle:
                # Candle complete, save it
                completed_candle = {
                    'timestamp': self.current_5min_candle,
                    'open': self.current_5min_open,
                    'high': self.current_5min_high,
                    'low': self.current_5min_low,
                    'close': self.current_5min_close,
                }
                self.ohlc_5min.append(completed_candle)
                
                # Update DataFrame
                self.ohlc_5min_df = pd.DataFrame(self.ohlc_5min)
                
                # Update detector
                self.detector.update_ohlc(self.ohlc_5min_df)
                
                logger.info(f"[5MIN] Candle Complete: {self.current_5min_candle.strftime('%H:%M')} "
                           f"O={self.current_5min_open} H={self.current_5min_high} "
                           f"L={self.current_5min_low} C={self.current_5min_close}")
                
                # Start new candle
                self.current_5min_candle = candle_5min
                self.current_5min_open = ltp
                self.current_5min_high = ltp
                self.current_5min_low = ltp
                self.current_5min_close = ltp
                
                return True  # Candle completed
            else:
                # Update current candle
                self.current_5min_close = ltp
                self.current_5min_high = max(self.current_5min_high, ltp)
                self.current_5min_low = min(self.current_5min_low, ltp)
                return False
    
    # ========================================
    # Entry Logic
    # ========================================
    
    def check_entry_conditions(self):
        """Check if should enter Phase 1"""
        # Already in position
        if self.phase > 0:
            return False, "Already in position"
        
        # Check time window
        if not self.in_entry_window():
            return False, "Outside entry window"
        
        # Check Wednesday EIA
        if self.is_wednesday_eia_risk():
            return False, "Wednesday EIA risk - no new entries"
        
        # Need minimum data
        if self.ohlc_5min_df is None or len(self.ohlc_5min_df) < 30:
            return False, f"Insufficient data: {len(self.ohlc_5min_df) if self.ohlc_5min_df is not None else 0} candles"
        
        # Get directional signal
        direction = self.detector.evaluate_trend_direction()
        
        if direction is None:
            return False, "No clear trend signal"
        
        # Check movement from session start
        if self.session_start_price and self.current_spot:
            move = abs(self.current_spot - self.session_start_price)
            if move < 20:
                return False, f"Insufficient movement: {move:.1f} points (need 20+)"
        
        logger.info(f"✅ Entry Conditions Met: {direction} trend confirmed")
        return True, direction
    
    def enter_phase1(self, direction: Literal["BULLISH", "BEARISH"]):
        """Enter Phase 1: 1:1 Credit Spread (DEFINED RISK)"""
        # CHECK STRATEGY COORDINATOR: Prevent conflict with Directional Strategy
        can_enter, reason = self.coordinator.can_enter_same_direction('ratio_spread', direction)
        if not can_enter:
            logger.warning(f"⛔ Skipping Ratio Spread Entry: {reason}")
            return False
        
        logger.info(f"🔵 Entering Phase 1: {direction} Credit Spread")
        
        ltp = self.client.get_exchange_ltp()
        atm = self.client.get_ATM_strike(ltp)
        
        # Determine strikes
        if direction == "BULLISH":
            # Call Credit Spread
            long_strike = atm
            short_strike = atm + cfg.CALL_SELL_STRIKE_1_OFFSET
            opt_type = "CE"
        else:
            # Put Credit Spread
            long_strike = atm
            short_strike = atm + cfg.PUT_SELL_STRIKE_1_OFFSET  # -50
            opt_type = "PE"
        
        # Build symbols
        long_symbol_info = self.client.build_option_symbol(
            cfg.UNDERLYING_SYMBOL, self.client.nearest_expiry, long_strike, opt_type
        )
        short_symbol_info = self.client.build_option_symbol(
            cfg.UNDERLYING_SYMBOL, self.client.nearest_expiry, short_strike, opt_type
        )
        
        long_symbol = long_symbol_info['symbol']
        short_symbol = short_symbol_info['symbol']
        
        logger.info(f"   Long: {long_symbol}")
        logger.info(f"   Short: {short_symbol}")
        
        # Calculate lots
        lots = cfg.BASE_LOTS  # Start with 1 lot
        quantity = lots * cfg.LOTSIZE
        
        async def place_phase1():
            # Place both legs
            long_order, short_order = await asyncio.gather(
                self.client.async_place_orders(long_symbol, 'BUY', quantity, strategy_tag="RATIO_P1"),
                self.client.async_place_orders(short_symbol, 'SELL', quantity, strategy_tag="RATIO_P1")
            )
            
            if not long_order or long_order.get('order_status') != 'complete':
                logger.error(f"Phase 1 Long leg failed: {long_order}")
                return False
            
            if not short_order or short_order.get('order_status') != 'complete':
                logger.error(f"Phase 1 Short leg failed: {short_order}")
                # Cleanup long
                await self.client.async_place_orders(long_symbol, 'SELL', quantity, strategy_tag="RATIO_CLEANUP")
                return False
            
            # Store leg details
            long_price = float(long_order.get('average_price', long_order.get('price', 0)))
            short_price = float(short_order.get('average_price', short_order.get('price', 0)))
            
            self.long_leg = {
                'symbol': long_symbol,
                'strike': long_strike,
                'type': opt_type,
                'buy_price': long_price,
                'current_price': long_price,
                'qty': quantity,
                'status': 'OPEN',
            }
            
            self.short_leg_1 = {
                'symbol': short_symbol,
                'strike': short_strike,
                'type': opt_type,
                'sell_price': short_price,
                'current_price': short_price,
                'qty': quantity,
                'status': 'OPEN',
            }
            
            # Update state
            self.phase = 1
            self.direction = direction
            self.entry_time = self.now_ist()
            self.phase1_entry_spot = ltp
            self.entry_capital = abs(long_price - short_price) * quantity
            
            # Log entry
            try:
                self.current_trade_id = log_entry(
                    self.db_session, short_symbol, long_symbol,
                    short_price, long_price, lots, f"RATIO_P1_{direction}"
                )
            except Exception as e:
                logger.error(f"DB log error: {e}")
            
            # MARK STRATEGY AS ACTIVE in coordinator
            self.coordinator.mark_entry('ratio_spread', direction=direction)
            logger.info(f"✅ Ratio Spread marked as active ({direction})")
            
            logger.info(f"✅ Phase 1 Entered: Capital = ₹{self.entry_capital:.0f}")
            return True
        
        return asyncio.run(place_phase1())
    
    def check_phase2_conditions(self):
        """Check if should enter Phase 2"""
        if self.phase != 1:
            return False, "Not in Phase 1"
        
        # Check time
        if not self.in_entry_window():
            return False, "Outside entry window"
        
        # Check Wednesday EIA
        if self.is_wednesday_eia_risk():
            return False, "Wednesday EIA risk"
        
        # Calculate Phase 1 P&L %
        self.update_phase1_pnl()
        
        phase1_pnl_pct = (self.phase1_pnl / self.entry_capital * 100) if self.entry_capital > 0 else 0
        
        if phase1_pnl_pct < cfg.PHASE_2['trigger']['phase1_in_profit']:
            return False, f"Phase 1 profit insufficient: {phase1_pnl_pct:.1f}% (need 15%)"
        
        # Check spot movement
        if self.phase1_entry_spot and self.current_spot:
            move = abs(self.current_spot - self.phase1_entry_spot)
            min_move, max_move = cfg.PHASE_2['trigger']['spot_moved_points']
            
            if move < min_move:
                return False, f"Insufficient move: {move:.1f} pts (need {min_move})"
            if move > max_move:
                return False, f"Moved too much: {move:.1f} pts (max {max_move})"
        
        # Check trend still intact
        direction = self.detector.evaluate_trend_direction()
        if direction != self.direction:
            return False, f"Trend changed: was {self.direction}, now {direction}"
        
        logger.info(f"✅ Phase 2 Conditions Met: P1 profit={phase1_pnl_pct:.1f}%, trend intact")
        return True, "Ready for Phase 2"
    
    def enter_phase2(self):
        """Enter Phase 2: Add ratio leg + PROTECTION WING (CRITICAL!)"""
        logger.info(f"🔵 Entering Phase 2: Protected Ratio (1:2 + wing)")
        
        ltp = self.client.get_exchange_ltp()
        atm = self.client.get_ATM_strike(ltp)
        
        # Determine strikes
        if self.direction == "BULLISH":
            short_strike_2 = atm + cfg.CALL_SELL_STRIKE_2_OFFSET  # Same as Phase 1
            protection_strike = atm + cfg.CALL_PROTECTION_OFFSET  # ATM+150
            opt_type = "CE"
        else:
            short_strike_2 = atm + cfg.PUT_SELL_STRIKE_2_OFFSET
            protection_strike = atm + cfg.PUT_PROTECTION_OFFSET
            opt_type = "PE"
        
        # Build symbols
        short_symbol_2_info = self.client.build_option_symbol(
            cfg.UNDERLYING_SYMBOL, self.client.nearest_expiry, short_strike_2, opt_type
        )
        protection_symbol_info = self.client.build_option_symbol(
            cfg.UNDERLYING_SYMBOL, self.client.nearest_expiry, protection_strike, opt_type
        )
        
        short_symbol_2 = short_symbol_2_info['symbol']
        protection_symbol = protection_symbol_info['symbol']
        
        logger.info(f"   Short (2nd): {short_symbol_2}")
        logger.info(f"   Protection Wing: {protection_symbol}")
        
        quantity = cfg.BASE_LOTS * cfg.LOTSIZE
        
        async def place_phase2():
            # CRITICAL: Place protection FIRST, then short
            protection_order, short_order_2 = await asyncio.gather(
                self.client.async_place_orders(protection_symbol, 'BUY', quantity, strategy_tag="RATIO_P2_WING"),
                self.client.async_place_orders(short_symbol_2, 'SELL', quantity, strategy_tag="RATIO_P2"),
            )
            
            if not protection_order or protection_order.get('order_status') != 'complete':
                logger.error(f"❌ PROTECTION WING FAILED: {protection_order}")
                logger.error("   CANNOT PROCEED WITHOUT PROTECTION - SKIPPING PHASE 2")
                return False
            
            if not short_order_2 or short_order_2.get('order_status') != 'complete':
                logger.warning(f"Short leg 2 failed: {short_order_2}")
                # Cleanup protection
                await self.client.async_place_orders(protection_symbol, 'SELL', quantity, strategy_tag="RATIO_CLEANUP")
                return False
            
            # Store leg details
            short_price_2 = float(short_order_2.get('average_price', short_order_2.get('price', 0)))
            protection_price = float(protection_order.get('average_price', protection_order.get('price', 0)))
            
            self.short_leg_2 = {
                'symbol': short_symbol_2,
                'strike': short_strike_2,
                'type': opt_type,
                'sell_price': short_price_2,
                'current_price': short_price_2,
                'qty': quantity,
                'status': 'OPEN',
            }
            
            self.protection_wing = {
                'symbol': protection_symbol,
                'strike': protection_strike,
                'type': opt_type,
                'buy_price': protection_price,
                'current_price': protection_price,
                'qty': quantity,
                'status': 'OPEN',
            }
            
            # Update state
            self.phase = 2
            
            logger.info(f"✅ Phase 2 Entered: Now 1:2 Protected Ratio")
            logger.info(f"   🛡️ PROTECTION WING ACTIVE at {protection_strike}")
            return True
        
        return asyncio.run(place_phase2())
    
    # ========================================
    # P&L Tracking
    # ========================================
    
    def update_phase1_pnl(self):
        """Calculate Phase 1 P&L"""
        if not self.long_leg or not self.short_leg_1:
            self.phase1_pnl = 0.0
            return
        
        long_pnl = (self.long_leg['current_price'] - self.long_leg['buy_price']) * self.long_leg['qty']
        short_pnl = (self.short_leg_1['sell_price'] - self.short_leg_1['current_price']) * self.short_leg_1['qty']
        
        self.phase1_pnl = long_pnl + short_pnl
    
    def update_total_pnl(self):
        """Calculate total position P&L"""
        total = 0.0
        
        # Long leg
        if self.long_leg and self.long_leg['status'] == 'OPEN':
            total += (self.long_leg['current_price'] - self.long_leg['buy_price']) * self.long_leg['qty']
        
        # Short leg 1
        if self.short_leg_1 and self.short_leg_1['status'] == 'OPEN':
            total += (self.short_leg_1['sell_price'] - self.short_leg_1['current_price']) * self.short_leg_1['qty']
        
        # Short leg 2 (Phase 2)
        if self.short_leg_2 and self.short_leg_2['status'] == 'OPEN':
            total += (self.short_leg_2['sell_price'] - self.short_leg_2['current_price']) * self.short_leg_2['qty']
        
        # Protection wing (Phase 2)
        if self.protection_wing and self.protection_wing['status'] == 'OPEN':
            total += (self.protection_wing['current_price'] - self.protection_wing['buy_price']) * self.protection_wing['qty']
        
        self.total_pnl = total
        self.highest_profit = max(self.highest_profit, total)
    
    # ========================================
    # Risk Management - 5 Layers
    # ========================================
    
    def check_stop_losses(self):
        """Check all 5 layers of stop losses"""
        if self.phase == 0:
            return False, ""
        
        self.update_total_pnl()
        minutes_in_trade = self.minutes_since_entry()
        
        # Layer 1: Quick Stop (First 30 minutes)
        if minutes_in_trade < cfg.STOP_LOSS_LAYERS['quick_sl_time']:
            loss_pct = (self.total_pnl / self.entry_capital * 100) if self.entry_capital > 0 else 0
            if loss_pct < -cfg.STOP_LOSS_LAYERS['quick_sl_pct']:
                return True, f"Quick SL: -{abs(loss_pct):.1f}% in {minutes_in_trade:.0f}min"
        
        # Layer 2: Spot-Based Stop
        if self.current_spot and self.short_leg_1:
            short_strike = self.short_leg_1['strike']
            buffer = cfg.STOP_LOSS_LAYERS['spot_buffer_points']
            
            if self.direction == "BULLISH":
                if self.current_spot > (short_strike + buffer):
                    return True, f"Spot SL: {self.current_spot} > {short_strike + buffer}"
            else:
                if self.current_spot < (short_strike - buffer):
                    return True, f"Spot SL: {self.current_spot} < {short_strike - buffer}"
        
        # Layer 3: Absolute Max Loss
        if self.total_pnl < -cfg.STOP_LOSS_LAYERS['absolute_max_loss']:
            return True, f"Max Loss Hit: ₹{abs(self.total_pnl):.0f}"
        
        # Layer 4: Percentage Max Loss
        max_loss_rupees = self.cfg.ACCOUNT_CAPITAL * (self.cfg.MAX_LOSS_PERCENT / 100)
        if abs(self.total_pnl) > max_loss_rupees:
            return True, f"Max Loss %: {abs(self.total_pnl):.0f} > {max_loss_rupees:.0f}"
        
        return False, ""
    
    def check_profit_targets(self):
        """Check profit targets and trailing"""
        if self.phase == 0:
            return False, ""
        
        self.update_total_pnl()
        profit_pct = (self.total_pnl / self.entry_capital * 100) if self.entry_capital > 0 else 0
        
        # Full exit at target
        if profit_pct >= cfg.TARGET_PROFIT_PCT:
            return True, f"Profit Target: {profit_pct:.1f}%"
        
        # Trailing stop (after 20% profit)
        if profit_pct >= cfg.STOP_LOSS_LAYERS['trail_after_profit_pct']:
            if not self.trailing_active:
                logger.info(f"🎯 Trailing SL activated at +{profit_pct:.1f}%")
                self.trailing_active = True
            
            # Trail 10% below highest
            trail_threshold = self.highest_profit * (1 - cfg.STOP_LOSS_LAYERS['trail_buffer_pct'] / 100)
            if self.total_pnl < trail_threshold:
                return True, f"Trailing SL: ₹{self.total_pnl:.0f} < ₹{trail_threshold:.0f}"
        
        return False, ""
    
    def check_time_exits(self):
        """Check time-based exits"""
        now = self.now_ist().time()
        
        # Wednesday EIA
        if self.is_wednesday_eia_risk():
            return True, "Wednesday EIA Exit"
        
        # End of session
        if now >= cfg.EXIT_TIME:
            return True, "End of Session (11 PM)"
        
        return False, ""
    
    # ========================================
    # Position Management
    # ========================================
    
    def manage_position(self):
        """Main position management logic"""
        if self.phase == 0:
            return
        
        # Check exits (stops, profits, time)
        should_exit, reason = self.check_stop_losses()
        if should_exit:
            logger.warning(f"🛑 Stop Loss: {reason}")
            self.exit_all_positions(reason)
            return
        
        should_exit, reason = self.check_profit_targets()
        if should_exit:
            logger.info(f"🎯 Profit Exit: {reason}")
            self.exit_all_positions(reason)
            return
        
        should_exit, reason = self.check_time_exits()
        if should_exit:
            logger.warning(f"⏰ Time Exit: {reason}")
            self.exit_all_positions(reason)
            return
        
        # Check Phase 2 entry (if in Phase 1)
        if self.phase == 1:
            can_enter, reason = self.check_phase2_conditions()
            if can_enter:
                self.enter_phase2()
    
    def exit_all_positions(self, reason=""):
        """Exit all position legs"""
        logger.info(f"🔴 Exiting All Positions: {reason}")
        
        async def close_all():
            tasks = []
            
            # Close long leg
            if self.long_leg and self.long_leg['status'] == 'OPEN':
                tasks.append(self.client.async_place_orders(
                    self.long_leg['symbol'], 'SELL', self.long_leg['qty'], strategy_tag="RATIO_EXIT"
                ))
                self.long_leg['status'] = 'CLOSED'
            
            # Close short leg 1
            if self.short_leg_1 and self.short_leg_1['status'] == 'OPEN':
                tasks.append(self.client.async_place_orders(
                    self.short_leg_1['symbol'], 'BUY', self.short_leg_1['qty'], strategy_tag="RATIO_EXIT"
                ))
                self.short_leg_1['status'] = 'CLOSED'
            
            # Close short leg 2 (if Phase 2)
            if self.short_leg_2 and self.short_leg_2['status'] == 'OPEN':
                tasks.append(self.client.async_place_orders(
                    self.short_leg_2['symbol'], 'BUY', self.short_leg_2['qty'], strategy_tag="RATIO_EXIT"
                ))
                self.short_leg_2['status'] = 'CLOSED'
            
            # Close protection wing (if Phase 2)
            if self.protection_wing and self.protection_wing['status'] == 'OPEN':
                tasks.append(self.client.async_place_orders(
                    self.protection_wing['symbol'], 'SELL', self.protection_wing['qty'], strategy_tag="RATIO_EXIT"
                ))
                self.protection_wing['status'] = 'CLOSED'
            
            if tasks:
                await asyncio.gather(*tasks)
            
            # Log exit
            try:
                log_exit(self.db_session, self.current_trade_id,
                        ce_buy=None, pe_buy=None, realized_pnl=self.total_pnl, notes=reason)
            except Exception as e:
                logger.error(f"DB log error: {e}")
            
            # MARK STRATEGY AS INACTIVE in coordinator
            self.coordinator.mark_exit('ratio_spread')
            logger.info("✅ Ratio Spread marked as inactive")
            
            logger.info(f"✅ All positions closed. Final P&L: ₹{self.total_pnl:.0f}")
        
        asyncio.run(close_all())
        
        # Reset state
        self.reset_position()
    
    def reset_position(self):
        """Reset position state"""
        self.phase = 0
        self.direction = None
        self.entry_time = None
        self.phase1_entry_spot = None
        self.long_leg = None
        self.short_leg_1 = None
        self.short_leg_2 = None
        self.protection_wing = None
        self.phase1_pnl = 0.0
        self.total_pnl = 0.0
        self.highest_profit = 0.0
        self.entry_capital = 0.0
        self.trailing_active = False
        self.current_trade_id = None
    
    # ========================================
    # Tick Handling
    # ========================================
    
    def on_underlying_tick(self, tick):
        """Handle underlying (future) tick"""
        ltp = tick.get('ltp')
        ts_ms = tick.get('timestamp')
        if ltp is None:
            return
        
        timestamp = datetime.fromtimestamp(ts_ms / 1000.0, pytz.utc).astimezone(IST)
        
        # Store session start price
        if self.session_start_price is None and self.in_session():
            self.session_start_price = ltp
            logger.info(f"📍 Session start price: {ltp}")
        
        self.current_spot = ltp
        
        # Update 5-minute candle
        candle_complete = self.update_5min_candle(ltp, timestamp)
        
        if candle_complete:
            # On new 5-min candle, check entry/manage position
            if self.phase == 0:
                # Check entry
                can_enter, result = self.check_entry_conditions()
                if can_enter:
                    self.enter_phase1(result)  # result = direction
            else:
                # Manage existing position
                self.manage_position()
    
    def on_option_tick(self, symbol, ltp):
        """Update option leg prices"""
        if self.long_leg and symbol == self.long_leg['symbol']:
            self.long_leg['current_price'] = ltp
        if self.short_leg_1 and symbol == self.short_leg_1['symbol']:
            self.short_leg_1['current_price'] = ltp
        if self.short_leg_2 and symbol == self.short_leg_2['symbol']:
            self.short_leg_2['current_price'] = ltp
        if self.protection_wing and symbol == self.protection_wing['symbol']:
            self.protection_wing['current_price'] = ltp
    
    def on_ltp(self, data):
        """WebSocket LTP callback"""
        if data.get('type') != 'market_data':
            return
        
        symbol = data.get('symbol')
        exchange = data.get('exchange')
        tick = data.get('data', {})
        
        if exchange != cfg.UNDERLYING_EXCHANGE:
            return
        
        if symbol == self.client.future_symbol:
            self.on_underlying_tick(tick)
        else:
            self.on_option_tick(symbol, tick.get('ltp'))
    
    # ========================================
    # Main Runner
    # ========================================
    
    def start(self):
        """Start the runner"""
        logger.info("🚀 Starting Protected Ratio Spread Runner...")
        
        # Connect to WebSocket
        self.client.connect()
        
        # Subscribe to symbols
        subs_syms = self.client.get_option_symbols(cfg.UNDERLYING_SYMBOL, cfg.UNDERLYING_EXCHANGE)
        if subs_syms:
            self.client.subscribe_ltp(subs_syms)
            self.client.subscribe_depth(subs_syms)
            self.client.subscribe_orderbook()
        else:
            logger.error("Failed to get option symbols")
            return
        
        logger.info("✅ Listening for signals...")
        logger.info(f"   📊 OHLC: 5-minute candles")
        logger.info(f"   ⏰ Session: {cfg.SESSION_START_TIME} - {cfg.SESSION_END_TIME}")
        logger.info(f"   🛡️ Max Loss: {cfg.MAX_LOSS_PERCENT}% or ₹{cfg.MAX_LOSS_RUPEES}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            if self.phase > 0:
                self.exit_all_positions("Manual shutdown")
            self.client.disconnect()


if __name__ == '__main__':
    runner = ProtectedRatioSpreadRunner()
    runner.start()
