# run_directional.py
# Runner for Directional Strategy (Trend Following)
# Uses EMA, Supertrend, and ADX to identify trends and enters Credit Spreads (Bull Put / Bear Call)

import time
from datetime import datetime
import pytz
import threading
import asyncio
import os

# modules
from config import default_config as cfg
from entries_directional import DirectionalEntries
from db_logger import init_db, log_entry, log_exit
from openalgo_client import OpenAlgoClientWrapper
from position_sizing import compute_lots_from_config
from logger import setup_logger
from ml_engine import MLEngine
from strategy_coordinator import get_coordinator

# Setup Logger
logger = setup_logger("DIRECTIONAL")

# ATR helper (reused)
try:
    from openalgo import ta
    def compute_atr(high, low, close, period=3):
        return float(ta.atr(high, low, close, period=period).iloc[-1])
except Exception:
    import pandas as pd
    def compute_atr(high, low, close, period=3):
        h = pd.Series(high)
        l = pd.Series(low)
        c = pd.Series(close)
        prev_close = c.shift(1)
        tr1 = h - l
        tr2 = (h - prev_close).abs()
        tr3 = (l - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return float(atr.iloc[-1])


IST = pytz.timezone("Asia/Kolkata")

logger.info("🚀 Directional Strategy Runner Starting (Credit Spreads)...")


class DirectionalRun:
    def __init__(self, config=cfg):
        self.cfg = config
        self.detectors = DirectionalEntries()
        self.ml_engine = MLEngine(model_path=self.cfg.ML_MODEL_PATH) if getattr(self.cfg, 'ENABLE_ML_FILTER', False) else None
        self.db_session = init_db(self.cfg.DB_PATH)
        
        # Strategy Coordinator (prevent conflict with ratio spread)
        self.coordinator = get_coordinator()
        
        self.client = OpenAlgoClientWrapper(
            api_key=self.cfg.OPENALGO_API_KEY,
            host=self.cfg.OPENALGO_HOST,
            ws_url=self.cfg.OPENALGO_WS,
            on_ltp_callback=self.on_ltp
        )
        
        # OHLC
        self.ohlc = []
        self.ohlc_df = None
        self.curr_min = None
        self.open = self.high = self.low = self.close = None
        
        # ATRs
        self.atr_short = None
        self.atr_long = None

        # Active Position (Spread)
        self.active_position = False
        self.short_leg = None
        self.long_leg = None
        self.current_trade_id = None
        self.entry_time = None
        self.position_direction = None  # "BULLISH" or "BEARISH" - track position direction
        self.highest_profit = 0.0
        self.trailing_active = False
        
        self.lock = threading.Lock()

    def _run_async(self, coro):
        return asyncio.run(coro)

    def now_ist(self):
        return datetime.now(pytz.utc).astimezone(IST)

    def in_allowed_window(self):
        now = self.now_ist().time()
        for s, e in self.cfg.ALLOWED_WINDOWS:
            if s <= now < e:
                return True
        return False

    def update_ohlc_df(self):
        import pandas as pd
        if len(self.ohlc) == 0:
            self.ohlc_df = pd.DataFrame(columns=['timestamp','open','high','low','close'])
        else:
            self.ohlc_df = pd.DataFrame(self.ohlc)
    
    def minutes_since_entry(self):
        """Calculate minutes since position entry"""
        if not self.entry_time:
            return 0
        return (self.now_ist() - self.entry_time).total_seconds() / 60.0
    
    def calculate_pnl_pct(self):
        """Calculate current P&L as percentage of entry capital"""
        if not self.short_leg or not self.long_leg:
            return 0.0
        
        # Calculate spread P&L
        short_pnl = (self.short_leg['sell_price'] - self.short_leg['current_price']) * self.short_leg['qty']
        long_pnl = (self.long_leg['current_price'] - self.long_leg['buy_price']) * self.long_leg['qty']
        total_pnl = short_pnl + long_pnl
        
        # Entry capital (max risk on spread)
        entry_capital = abs(self.short_leg['sell_price'] - self.long_leg['buy_price']) * self.short_leg['qty']
        
        if entry_capital > 0:
            return (total_pnl / entry_capital) * 100.0
        return 0.0

    def compute_atrs(self):
        if self.ohlc_df is None or len(self.ohlc_df) < max(self.cfg.ATR_LONG_PERIOD, self.cfg.ATR_SHORT_PERIOD) + 1:
            self.atr_short = None
            self.atr_long = None
            return

        highs = self.ohlc_df['high']
        lows = self.ohlc_df['low']
        closes = self.ohlc_df['close']
        try:
            self.atr_short = compute_atr(highs, lows, closes, period=self.cfg.ATR_SHORT_PERIOD)
            self.atr_long = compute_atr(highs, lows, closes, period=self.cfg.ATR_LONG_PERIOD)
        except Exception:
            self.atr_short = self.atr_long = None

    # --------------------------------------------------------
    # Liquidity Check
    # --------------------------------------------------------
    def check_liquidity(self, symbol):
        # 1. Try Market Depth
        try:
            depth = self.client.get_market_depth(symbol)
            if depth:
                bids = depth.get('bids', [])
                asks = depth.get('asks', [])
                if bids and asks:
                    best_bid = bids[0]['price']
                    best_ask = asks[0]['price']
                    if best_bid > 0 and best_ask > 0:
                        spread = best_ask - best_bid
                        threshold = max(0.9, best_ask * 0.02)
                        if spread > threshold:
                            logger.warning(f"Liquidity Warning: Spread {spread:.2f} > {threshold:.2f} for {symbol}")
                            return False
                        return True
        except Exception as e:
            logger.error(f"Depth Check Error: {e}")

        # 2. Fallback to Quote
        try:
            quote = self.client.get_quote(symbol)
            if quote:
                ask = float(quote.get('ask', 0))
                bid = float(quote.get('bid', 0))
                if ask > 0 and bid > 0:
                    logger.info(f"Liquidity Verified via Quote for {symbol} (Bid: {bid}, Ask: {ask})")
                    return True
                else:
                    logger.warning(f"Liquidity Fail: Quote has zero bid/ask for {symbol} (Bid: {bid}, Ask: {ask})")
        except Exception as e:
            logger.error(f"Quote Check Error: {e}")
            
        logger.warning(f"Liquidity Fail: No valid depth or quote for {symbol}")
        return False

    # --------------------------------------------------------
    # Trade Execution (Credit Spread)
    # --------------------------------------------------------
    def enter_trade(self, direction, lots):
        # CHECK STRATEGY COORDINATOR: Prevent conflict with Ratio Spread
        can_enter, reason = self.coordinator.can_enter_same_direction('directional', direction)
        if not can_enter:
            logger.warning(f"⛔ Skipping Directional Entry: {reason}")
            return
        
        ltp = self.client.get_exchange_ltp()
        atm = self.client.get_ATM_strike(ltp)
        
        # Determine Strikes for Credit Spread
        spread_width = getattr(self.cfg, 'DIRECTIONAL_SPREAD_STRIKES', 2)
        
        if direction == "BULLISH":
            # Bull Put Spread: Sell ATM PE, Buy OTM PE
            short_strike = atm
            long_strike = atm - (spread_width * 100) 
            opt_type = "PE"
            strategy_name = "BULL_PUT_SPREAD"
        else: # BEARISH
            # Bear Call Spread: Sell ATM CE, Buy OTM CE
            short_strike = atm
            long_strike = atm + (spread_width * 100)
            opt_type = "CE"
            strategy_name = "BEAR_CALL_SPREAD"

        short_symbol_info = self.client.build_option_symbol(
            self.cfg.UNDERLYING_SYMBOL, self.client.nearest_expiry, short_strike, opt_type
        )
        long_symbol_info = self.client.build_option_symbol(
            self.cfg.UNDERLYING_SYMBOL, self.client.nearest_expiry, long_strike, opt_type
        )
        
        short_symbol = short_symbol_info['symbol']
        long_symbol = long_symbol_info['symbol']

        logger.info(f"Entering {strategy_name}: Sell {short_symbol}, Buy {long_symbol}")
        
        # ML Filter Check
        if self.ml_engine:
            confidence = self.ml_engine.predict()
            if confidence < self.cfg.ML_CONFIDENCE_THRESHOLD:
                logger.warning(f"⛔ ML Filter Blocked Trade. Confidence: {confidence:.2f} < {self.cfg.ML_CONFIDENCE_THRESHOLD}")
                return
            logger.info(f"✅ ML Filter Passed. Confidence: {confidence:.2f}")
        
        # Check Liquidity for both
        if not self.check_liquidity(short_symbol) or not self.check_liquidity(long_symbol):
            logger.warning("Skipping Spread: Poor Liquidity")
            return

        async def place():
            quantity = lots * self.cfg.LOTSIZE
            
            # Place both legs in parallel to minimize slippage
            long_order, short_order = await asyncio.gather(
                self.client.async_place_orders(long_symbol, 'BUY', quantity, strategy_tag="DIRECTIONAL"),
                self.client.async_place_orders(short_symbol, 'SELL', quantity, strategy_tag="DIRECTIONAL")
            )
            
            # Check Order Status
            if not long_order or long_order.get('order_status') != 'complete':
                logger.error(f"Long Leg Failed/Rejected: {long_order}")
                # Cleanup: If short leg was somehow placed (unlikely due to await), cancel it
                if short_order and short_order.get('order_status') == 'open':
                     await self.client.async_cancel_order(short_order['orderid'])
                return

            if not short_order or short_order.get('order_status') != 'complete':
                logger.error(f"Short Leg Failed/Rejected: {short_order}")
                # Cleanup: Close long leg
                if long_order and long_order.get('order_status') == 'complete':
                     await self.client.async_place_orders(long_symbol, 'SELL', quantity, strategy_tag="DIRECTIONAL_CLEANUP")
                return

            logger.info(f"Long Leg Order: {long_order}")
            logger.info(f"Short Leg Order: {short_order}")
            
            # 3. Place SL on Short Leg (e.g., 30% of premium)
            sl_pct = 30.0 
            sl_order = await self.client.async_sl_order(short_order, sl_pct, strategy_tag="DIRECTIONAL")
            logger.info(f"Short Leg SL: {sl_order}")
            
            # Ensure Float Prices
            try:
                short_price = float(short_order.get('average_price', short_order.get('price', 0.0)))
                long_price = float(long_order.get('average_price', long_order.get('price', 0.0)))
            except Exception as e:
                logger.error(f"Price Conversion Error: {e}")
                short_price = 0.0
                long_price = 0.0

            self.short_leg = {
                'symbol': short_symbol,
                'sell_price': short_price,
                'qty': quantity,
                'status': 'OPEN',
                'current_price': short_price,
                'sl_order': sl_order,
                'type': opt_type
            }
            
            self.long_leg = {
                'symbol': long_symbol,
                'buy_price': long_price,
                'qty': quantity,
                'status': 'OPEN',
                'current_price': long_price
            }
            
            self.active_position = True
            self.entry_time = self.now_ist()
            self.position_direction = direction  # Track position direction
            self.highest_profit = 0.0
            self.trailing_active = False
            
            try:
                # Log entry
                self.current_trade_id = log_entry(
                    self.db_session, short_symbol, long_symbol, 
                    short_price, long_price, lots, f"{strategy_name}_{direction}"
                )
            except Exception as e:
                logger.error(f"DB Log Error: {e}")
            
            # MARK STRATEGY AS ACTIVE in coordinator
            self.coordinator.mark_entry('directional', direction=direction)
            logger.info(f"✅ Directional strategy marked as active ({direction})")

        self._run_async(place())

    def exit_trade(self, reason=""):
        if not self.active_position:
            return

        logger.info(f"Exiting Spread: {reason}")
        
        async def close():
            # Cancel SL on Short Leg
            if self.short_leg and self.short_leg.get('sl_order'):
                try:
                    await self.client.async_cancel_order(self.short_leg['sl_order']['orderid'])
                except: pass
            
            # Close Short Leg (Buy Back)
            if self.short_leg and self.short_leg['status'] == 'OPEN':
                await self.client.async_place_orders(
                    self.short_leg['symbol'], 'BUY', self.short_leg['qty'], strategy_tag="DIRECTIONAL"
                )
            
            # Close Long Leg (Sell)
            if self.long_leg and self.long_leg['status'] == 'OPEN':
                await self.client.async_place_orders(
                    self.long_leg['symbol'], 'SELL', self.long_leg['qty'], strategy_tag="DIRECTIONAL"
                )
            
            # Log exit
            try:
                # Calculate PnL
                realized = 0.0
                if self.short_leg:
                    realized += (self.short_leg['sell_price'] - self.short_leg['current_price']) * self.short_leg['qty']
                if self.long_leg:
                    realized += (self.long_leg['current_price'] - self.long_leg['buy_price']) * self.long_leg['qty']
                
                log_exit(self.db_session, self.current_trade_id, 
                         ce_buy=None, pe_buy=None, realized_pnl=realized, notes=reason)
            except Exception as e:
                logger.error(f"Exit Log Error: {e}")
            
            # MARK STRATEGY AS INACTIVE in coordinator
            self.coordinator.mark_exit('directional')
            logger.info("✅ Directional strategy marked as inactive")
            
            self.active_position = False
            self.short_leg = None
            self.long_leg = None
            self.current_trade_id = None
            self.position_direction = None  # Reset direction

        self._run_async(close())

    # --------------------------------------------------------
    # Trade Management
    # --------------------------------------------------------
    def manage_trade(self):
        """Enhanced trade management with trend reversal detection"""
        if not self.active_position or not self.short_leg:
            return
        
        # ========================================
        # 1. TREND REVERSAL EXIT (New Feature!)
        # ========================================
        current_trend = self.detectors.evaluate_trend_direction()
        
        if current_trend and current_trend != self.position_direction:
            # Opposite trend detected!
            pnl_pct = self.calculate_pnl_pct()
            minutes_in_trade = self.minutes_since_entry()
            
            logger.warning(f"⚠️ TREND REVERSAL DETECTED!")
            logger.info(f"   Position Direction: {self.position_direction}")
            logger.info(f"   New Trend: {current_trend}")
            logger.info(f"   Current P&L: {pnl_pct:.1f}%")
            logger.info(f"   Time in Trade: {minutes_in_trade:.0f} minutes")
            
            # Decision Logic:
            # Exit if in loss (cut losses early, don't wait for SL)
            if pnl_pct < 0:
                logger.warning(f"   ❌ In Loss ({pnl_pct:.1f}%) - Exiting Early")
                logger.info(f"   💡 Better to cut loss now and enter new {current_trend} trend")
                self.exit_trade(f"Trend Reversal: {self.position_direction}→{current_trend}, P&L={pnl_pct:.1f}%")
                return
            
            # Exit if small profit (< 10%) and trend reversed
            elif pnl_pct < 10.0:
                logger.warning(f"   ⚠️ Small Profit ({pnl_pct:.1f}%) but Trend Reversed")
                logger.info(f"   💡 Locking profit and freeing up for new {current_trend} trade")
                self.exit_trade(f"Trend Reversal (Small Profit): {pnl_pct:.1f}%")
                return
            
            # Keep position if in good profit (>= 10%)
            else:
                logger.info(f"   ✅ Good Profit ({pnl_pct:.1f}%) - Keeping Position")
                logger.info(f"   💡 Will manage with trailing SL instead")
        
        # ========================================
        # 2. STOP LOSS CHECK
        # ========================================
        # Check SL hit on Short Leg
        if self.short_leg.get('sl_order'):
            info = self.client.get_order_info_of_order(self.short_leg['sl_order']['orderid'])
            if info and info.get('order_status','').lower() != 'trigger pending':
                logger.info("Short Leg SL Hit! Exiting Spread.")
                self.exit_trade("SL Hit")
                return
        
        # ========================================
        # 3. TRAILING LOGIC
        # ========================================
        # Trailing Logic (Based on Short Leg Decay)
        curr_price = self.short_leg['current_price']
        sell_price = self.short_leg['sell_price']
        
        # Profit % on the Short Leg (decay)
        decay_pct = (sell_price - curr_price) / sell_price * 100.0
        
        # 1. Move to Breakeven at 25% decay
        if decay_pct >= 25.0 and not self.trailing_active:
            logger.info("Short Leg Decayed > 25%: Moving SL to Cost")
            self._run_async(self.client.modify_sl_to_cost(self.short_leg, sell_price - 1.0)) 
            self.trailing_active = True
            
        # 2. Trail aggressively after 50% decay
        if decay_pct >= 50.0:
            new_sl = curr_price + (curr_price * 0.20) 
            current_sl = self.short_leg.get('stop_price', 99999)
            
            if new_sl < current_sl: 
                logger.info(f"Trailing SL to {new_sl:.2f}")
                self.short_leg['stop_price'] = new_sl
                self._run_async(self.client.modify_sl_to_cost(self.short_leg, new_sl))

    # --------------------------------------------------------
    # Tick Handling
    # --------------------------------------------------------
    def on_underlying_tick(self, tick):
        ltp = tick.get('ltp')
        ts_ms = tick.get('timestamp')
        if ltp is None: return

        dt = datetime.fromtimestamp(ts_ms/1000.0, pytz.utc).astimezone(IST)
        
        # CHANGED: Use 5-minute candles instead of 1-minute
        minute = dt.replace(second=0, microsecond=0)
        candle_5min = minute.replace(minute=(minute.minute // 5) * 5)

        with self.lock:
            if self.curr_min is None:
                self.curr_min = candle_5min
                self.open = self.high = self.low = self.close = ltp
                return

            if candle_5min != self.curr_min:
                # Finalize 5-minute candle
                row = {'timestamp': self.curr_min, 'open': self.open, 'high': self.high, 'low': self.low, 'close': self.close}
                self.ohlc.append(row)
                self.update_ohlc_df()
                self.compute_atrs()
                
                # Update detectors with 5-min data
                self.detectors.update_ohlc(self.ohlc_df)
                self.detectors.latest_atr_short = self.atr_short
                self.detectors.latest_atr_long = self.atr_long
                
                self.check_intraday_squareoff()
                
                if self.active_position:
                    self.manage_trade()
                else:
                    # ENTRY LOGIC - Now using 5-min candles
                    if self.in_allowed_window():
                        # Directional signal from entries_directional.py
                        # Uses EMA, ADX, Supertrend on 5-min data (better signals!)
                        direction = self.detectors.evaluate_trend_direction()
                        
                        if direction:  # "BULLISH" or "BEARISH"
                            logger.info(f"✅ 5-MIN Signal: {direction} trend confirmed")
                            lots = compute_lots_from_config(self.atr_short, self.cfg, legs_count=1, signal_direction=direction)
                            if lots > 0:
                                self.enter_trade(direction, lots)

                self.curr_min = candle_5min
                self.open = self.high = self.low = self.close = ltp
            else:
                # Update current 5-min candle
                self.close = ltp
                self.high = max(self.high, ltp)
                self.low = min(self.low, ltp)
                
                if self.active_position:
                    self.manage_trade()

    def on_option_tick(self, symbol, ltp):
        if self.short_leg and symbol == self.short_leg['symbol']:
            self.short_leg['current_price'] = ltp
        if self.long_leg and symbol == self.long_leg['symbol']:
            self.long_leg['current_price'] = ltp

    def on_ltp(self, data):
        if data.get('type') != 'market_data': return
        symbol = data.get('symbol')
        exch = data.get('exchange')
        tick = data.get('data', {})
        
        if exch != self.cfg.UNDERLYING_EXCHANGE: return
        
        if symbol == self.client.future_symbol:
            self.on_underlying_tick(tick)
        else:
            self.on_option_tick(symbol, tick.get('ltp'))

    def check_intraday_squareoff(self):
        now = self.now_ist().time()
        if now >= self.cfg.INTRADAY_SQUAREOFF_TIME:
            if self.active_position:
                logger.warning(f"⏰ Intraday Square-off Time Reached ({now}). Closing all positions.")
                self.exit_trade("Intraday Square-off")

    def start(self):
        self.client.connect()
        subs_sym = self.client.get_option_symbols(self.cfg.UNDERLYING_SYMBOL, self.cfg.UNDERLYING_EXCHANGE)
        if subs_sym:  # Check for None
            self.client.subscribe_ltp(subs_sym)
            self.client.subscribe_depth(subs_sym)
            self.client.subscribe_orderbook()
        else:
            logger.error("Failed to get option symbols. Cannot subscribe.")
            return
        logger.info("Directional Runner (Credit Spread) Listening...")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            self.client.disconnect()

if __name__ == '__main__':
    runner = DirectionalRun()
    runner.start()
