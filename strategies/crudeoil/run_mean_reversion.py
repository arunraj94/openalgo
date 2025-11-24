# run_mean_reversion.py
# Runner for Mean Reversion Scalper Strategy
# Refactored to use Credit Spreads (Bull Put / Bear Call)

import time
from datetime import datetime
import pytz
import threading
import asyncio
import os

# modules
from config import default_config as cfg
from entries_mean_reversion import MeanReversionEntries
from db_logger import init_db, log_entry, log_exit
from openalgo_client import OpenAlgoClientWrapper
from position_sizing import compute_lots_from_config
from logger import setup_logger
from ml_engine import MLEngine

# Setup Logger
logger = setup_logger("MEANREV")

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

logger.info("⚡ Mean Reversion Scalper Runner Starting (Credit Spreads)...")


class MeanRevRun:
    def __init__(self, config=cfg):
        self.cfg = config
        self.detector = MeanReversionEntries(config=config)
        self.ml_engine = MLEngine(model_path=self.cfg.ML_MODEL_PATH) if getattr(self.cfg, 'ENABLE_ML_FILTER', False) else None
        self.db_session = init_db(self.cfg.DB_PATH)
        
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

        # Active Position (Spread)
        self.active_position = False
        self.short_leg = None
        self.long_leg = None
        self.current_trade_id = None
        self.entry_time = None
        self.position_signal = None  # Track entry signal ("BUY" or "SELL")
        self.spread_credit = 0.0
        
        self.lock = threading.Lock()

    def _run_async(self, coro):
        return asyncio.run(coro)

    def now_ist(self):
        return datetime.now(pytz.utc).astimezone(IST)

    def update_ohlc_df(self):
        import pandas as pd
        if len(self.ohlc) == 0:
            self.ohlc_df = pd.DataFrame(columns=['timestamp','open','high','low','close'])
        else:
            self.ohlc_df = pd.DataFrame(self.ohlc)

    def compute_atrs(self):
        if self.ohlc_df is None or len(self.ohlc_df) < self.cfg.ATR_SHORT_PERIOD + 1:
            self.atr_short = None
            return

        highs = self.ohlc_df['high']
        lows = self.ohlc_df['low']
        closes = self.ohlc_df['close']
        try:
            self.atr_short = compute_atr(highs, lows, closes, period=self.cfg.ATR_SHORT_PERIOD)
        except Exception:
            self.atr_short = None

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
    def enter_trade(self, signal, lots):
        ltp = self.client.get_exchange_ltp()
        atm = self.client.get_ATM_strike(ltp)
        
        # Determine Strikes for Credit Spread
        # Default spread width 200 points if not in config
        spread_width = getattr(self.cfg, 'DIRECTIONAL_SPREAD_STRIKES', 2)
        
        if signal == "BUY":
            # Signal is BUY (Bullish) -> Sell Bull Put Spread
            # Sell ATM PE, Buy OTM PE
            short_strike = atm
            long_strike = atm - (spread_width * 100) 
            opt_type = "PE"
            strategy_name = "MEANREV_BULL_PUT"
            log_signal = "BULLISH"
        else: # SELL (Bearish)
            # Signal is SELL (Bearish) -> Sell Bear Call Spread
            # Sell ATM CE, Buy OTM CE
            short_strike = atm
            long_strike = atm + (spread_width * 100)
            opt_type = "CE"
            strategy_name = "MEANREV_BEAR_CALL"
            log_signal = "BEARISH"

        short_sym_info = self.client.build_option_symbol(
            self.cfg.UNDERLYING_SYMBOL, self.client.nearest_expiry, short_strike, opt_type
        )
        long_sym_info = self.client.build_option_symbol(
            self.cfg.UNDERLYING_SYMBOL, self.client.nearest_expiry, long_strike, opt_type
        )
        
        short_symbol = short_sym_info['symbol']
        long_symbol = long_sym_info['symbol']

        logger.info(f"MeanRev Entry {log_signal}: Sell {short_symbol}, Buy {long_symbol} ({lots} lots)")

        # ML Filter Check
        if self.ml_engine:
            confidence = self.ml_engine.predict()
            if confidence < self.cfg.ML_CONFIDENCE_THRESHOLD:
                logger.warning(f"⛔ ML Filter Blocked Trade. Confidence: {confidence:.2f} < {self.cfg.ML_CONFIDENCE_THRESHOLD}")
                return
            logger.info(f"✅ ML Filter Passed. Confidence: {confidence:.2f}")
        
        # Liquidity Check
        if not self.check_liquidity(short_symbol) or not self.check_liquidity(long_symbol):
            logger.warning("Skipping MeanRev Spread: Poor Liquidity")
            return

        async def place():
            quantity = lots * self.cfg.LOTSIZE
            
            # Place Both Legs in Parallel (minimize slippage)
            long_order, short_order = await asyncio.gather(
                self.client.async_place_orders(long_symbol, 'BUY', quantity, strategy_tag="MEANREV"),
                self.client.async_place_orders(short_symbol, 'SELL', quantity, strategy_tag="MEANREV")
            )
            
            # Check Order Status
            if not long_order or long_order.get('order_status') != 'complete':
                logger.error(f"Long Leg Failed/Rejected: {long_order}")
                # Cleanup: If short leg was somehow placed (unlikely due to await), cancel it
                if short_order and short_order.get('order_status') == 'open' or short_order.get('order_status') == 'trigger pending':
                     await self.client.async_cancel_order(short_order['orderid'])
                return

            if not short_order or short_order.get('order_status') != 'complete':
                logger.error(f"Short Leg Failed/Rejected: {short_order}")
                # Cleanup: Close long leg
                if long_order and long_order.get('order_status') == 'complete':
                     await self.client.async_place_orders(long_symbol, 'SELL', quantity, strategy_tag="MEANREV_CLEANUP")
                return

            # SL on Short Leg (e.g., 40% of premium)
            sl_price = float(short_order.get('average_price', short_order.get('price', 0.0))) * 1.40
            sl_order = await self.client.async_sl_order(short_order, 40.0, strategy_tag="MEANREV") # % based
            
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
                'sl_order': sl_order
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
            self.position_signal = signal  # Track entry signal
            self.spread_credit = (short_price - long_price) * quantity
            
            try:
                self.current_trade_id = log_entry(
                    self.db_session, short_symbol, long_symbol, 
                    short_price, long_price, lots, f"MEANREV_{log_signal}"
                )
            except Exception as e:
                logger.error(f"DB Log Error: {e}")

        self._run_async(place())

    def exit_trade(self, reason=""):
        if not self.active_position:
            return

        logger.info(f"Exiting MeanRev trade: {reason}")
        
        async def close():
            # Cancel SL on short leg
            if self.short_leg and self.short_leg.get('sl_order'):
                try:
                    await self.client.async_cancel_order(self.short_leg['sl_order']['orderid'])
                except: pass
            
            # Close Short Leg (Buy back)
            if self.short_leg:
                await self.client.async_place_orders(
                    self.short_leg['symbol'], 'BUY', self.short_leg['qty'], strategy_tag="MEANREV"
                )
            
            # Close Long Leg (Sell)
            if self.long_leg:
                await self.client.async_place_orders(
                    self.long_leg['symbol'], 'SELL', self.long_leg['qty'], strategy_tag="MEANREV"
                )
            
            try:
                # Calculate PnL
                realized = 0.0
                if self.short_leg:
                    realized += (self.short_leg['sell_price'] - self.short_leg['current_price']) * self.short_leg['qty']
                if self.long_leg:
                    realized += (self.long_leg['current_price'] - self.long_leg['buy_price']) * self.long_leg['qty']

                log_exit(self.db_session, self.current_trade_id, 
                         ce_buy=None, pe_buy=None, realized_pnl=realized, notes=reason)
            except: pass
                
            self.active_position = False
            self.short_leg = None
            self.long_leg = None
            self.current_trade_id = None

        self._run_async(close())

    # --------------------------------------------------------
    # Trade Management
    # --------------------------------------------------------
    def manage_trade(self):
        """Enhanced trade management with signal reversal detection"""
        if not self.active_position or not self.short_leg:
            return
        
        # ========================================
        # 1. CHECK FOR SIGNAL REVERSAL (New!)
        # ========================================
        # If we entered on OVERSOLD, check if now OVERBOUGHT (and vice versa)
        current_signal = self.detector.evaluate_signal()
        
        if current_signal and current_signal != self.position_signal:
            # Opposite signal detected!
            short_pnl = (self.short_leg['sell_price'] - self.short_leg['current_price']) * self.short_leg['qty']
            long_pnl = (self.long_leg['current_price'] - self.long_leg['buy_price']) * self.long_leg['qty']
            total_pnl = short_pnl + long_pnl
            
            logger.warning(f"⚠️ SIGNAL REVERSAL DETECTED!")
            logger.info(f"   Entry Signal: {self.position_signal}")
            logger.info(f"   Current Signal: {current_signal}")
            logger.info(f"   Current P&L: ₹{total_pnl:.0f}")
            
            # Exit if in loss (don't wait for full SL)
            if total_pnl < 0:
                logger.warning(f"   ❌ In Loss - Exiting Early (Before SL)")
                logger.info(f"   💡 Mean reversion failed, opposite extreme detected")
                self.exit_trade(f"Signal Reversal: {self.position_signal}→{current_signal}, Loss=₹{total_pnl:.0f}")
                return
            
            # Exit if small profit (<20% of target)
            elif total_pnl < (self.spread_credit * 0.20):
                logger.warning(f"   ⚠️ Small Profit but Signal Reversed")
                logger.info(f"   💡 Locking profit early")
                self.exit_trade(f"Signal Reversal (Small Profit): ₹{total_pnl:.0f}")
                return
            
            # Keep if good profit (>=20%)
            else:
                logger.info(f"   ✅ Good Profit (₹{total_pnl:.0f}) - Keeping Position")
        
        # ========================================
        # 2. STOP LOSS CHECK
        # ========================================
        # Check SL Hit on Short Leg via Order Status
        info = self.client.get_order_info_of_order(self.short_leg['sl_order']['orderid'])
        if info and info.get('order_status','').lower() != 'trigger pending':
            logger.info("MeanRev Short Leg SL Hit!")
            self.exit_trade("SL Hit")
            return

        # Calculate Spread PnL
        short_pnl = (self.short_leg['sell_price'] - self.short_leg['current_price']) * self.short_leg['qty']
        long_pnl = (self.long_leg['current_price'] - self.long_leg['buy_price']) * self.long_leg['qty']
        total_pnl = short_pnl + long_pnl
        
        # Target: 50% of Max Credit
        if total_pnl >= (self.spread_credit * 0.5):
            self.exit_trade("Target Hit (50% of Credit)")
            return
            
        # Time Exit (End of session or Stale)
        now = self.now_ist()
        elapsed_mins = (now - self.entry_time).total_seconds() / 60.0
        
        if elapsed_mins > self.cfg.MEANREV_TIME_EXIT_MINUTES:
            self.exit_trade("Time Exit (Stale Scalp)")
            return

        if now.time() >= datetime.strptime("23:15", "%H:%M").time():
            self.exit_trade("Session End Exit")

    # --------------------------------------------------------
    # Tick Handling
    # --------------------------------------------------------
    def on_underlying_tick(self, tick):
        ltp = tick.get('ltp')
        ts_ms = tick.get('timestamp')
        if ltp is None: return

        dt = datetime.fromtimestamp(ts_ms/1000.0, pytz.utc).astimezone(IST)
        
        # CHANGED: Use 5-minute candles for better mean reversion signals
        # VWAP deviation on 5-min is more meaningful than 1-min noise
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
                
                # Update MeanRev logic with 5-min data
                self.detector.update_ohlc(self.ohlc_df)
                
                # ENTRY LOGIC - Now using 5-min VWAP deviation
                if not self.active_position:
                    signal = self.detector.evaluate_signal()
                    if signal:
                        logger.info(f"[5-MIN] MeanRev Signal: {signal}")
                        # Mean Reversion: BUY = expect UP move, SELL = expect DOWN move
                        direction = "BULLISH" if signal == "BUY" else "BEARISH"
                        lots = compute_lots_from_config(self.atr_short, self.cfg, legs_count=1, signal_direction=direction)
                        if lots > 0:
                            self.enter_trade(signal, lots)
                
                # EXIT LOGIC
                self.check_intraday_squareoff()

                self.curr_min = candle_5min
                self.open = self.high = self.low = self.close = ltp
            else:
                # Update current 5-min candle
                self.close = ltp
                self.high = max(self.high, ltp)
                self.low = min(self.low, ltp)
                
                if self.active_position:
                    self.manage_trade()

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

    def on_option_tick(self, symbol, ltp):
        if self.short_leg and symbol == self.short_leg['symbol']:
            self.short_leg['current_price'] = ltp
        if self.long_leg and symbol == self.long_leg['symbol']:
            self.long_leg['current_price'] = ltp

    def check_intraday_squareoff(self):
        now = self.now_ist().time()
        if now >= self.cfg.INTRADAY_SQUAREOFF_TIME:
            if self.active_position:
                logger.warning(f"⏰ Intraday Square-off Time Reached ({now}). Closing all positions.")
                self.exit_trade("Intraday Square-off")

    def start(self):
        self.client.connect()
        subs_sym = self.client.get_option_symbols(self.cfg.UNDERLYING_SYMBOL, self.cfg.UNDERLYING_EXCHANGE)
        self.client.subscribe_ltp(subs_sym)
        self.client.subscribe_depth(subs_sym)
        self.client.subscribe_orderbook()
        logger.info("MeanRev Runner Listening...")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            self.client.disconnect()

if __name__ == '__main__':
    runner = MeanRevRun()
    runner.start()
