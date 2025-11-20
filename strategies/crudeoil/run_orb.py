# run_orb.py
# Runner for Opening Range Breakout (ORB) Strategy
# Refactored to use Credit Spreads (Bull Put / Bear Call)

import time
from datetime import datetime
import pytz
import threading
import asyncio
import os

# modules
from config import default_config as cfg
from entries_orb import OpeningRangeBreakout
from db_logger import init_db, log_entry, log_exit
from openalgo_client import OpenAlgoClientWrapper
from position_sizing import compute_lots_from_config
from logger import setup_logger
from ml_engine import MLEngine

# Setup Logger
logger = setup_logger("ORB")

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

logger.info("🌅 ORB Strategy Runner Starting (Credit Spreads)...")


class ORBRun:
    def __init__(self, config=cfg):
        self.cfg = config
        self.orb = OpeningRangeBreakout()
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
        spread_width = getattr(self.cfg, 'DIRECTIONAL_SPREAD_STRIKES', 2)
        
        if signal == "BULLISH":
            # Bull Put Spread: Sell ATM PE, Buy OTM PE
            short_strike = atm
            long_strike = atm - (spread_width * 100) 
            opt_type = "PE"
            strategy_name = "ORB_BULL_PUT"
        else: # BEARISH
            # Bear Call Spread: Sell ATM CE, Buy OTM CE
            short_strike = atm
            long_strike = atm + (spread_width * 100)
            opt_type = "CE"
            strategy_name = "ORB_BEAR_CALL"

        short_sym_info = self.client.build_option_symbol(
            self.cfg.UNDERLYING_SYMBOL, self.client.nearest_expiry, short_strike, opt_type
        )
        long_sym_info = self.client.build_option_symbol(
            self.cfg.UNDERLYING_SYMBOL, self.client.nearest_expiry, long_strike, opt_type
        )
        
        short_symbol = short_sym_info['symbol']
        long_symbol = long_sym_info['symbol']

        logger.info(f"ORB Entry {signal}: Sell {short_symbol}, Buy {long_symbol} ({lots} lots)")

        # ML Filter Check
        if self.ml_engine:
            confidence = self.ml_engine.predict()
            if confidence < self.cfg.ML_CONFIDENCE_THRESHOLD:
                logger.warning(f"⛔ ML Filter Blocked Trade. Confidence: {confidence:.2f} < {self.cfg.ML_CONFIDENCE_THRESHOLD}")
                return
            logger.info(f"✅ ML Filter Passed. Confidence: {confidence:.2f}")
        
        # Liquidity Check
        if not self.check_liquidity(short_symbol) or not self.check_liquidity(long_symbol):
            logger.warning("Skipping ORB Spread: Poor Liquidity")
            return

        async def place():
            quantity = lots * self.cfg.LOTSIZE
            
            # 1. Buy Hedge (Long Leg) First
            long_order = await self.client.async_place_orders(long_symbol, 'BUY', quantity, strategy_tag="ORB")
            
            # 2. Sell Premium (Short Leg)
            short_order = await self.client.async_place_orders(short_symbol, 'SELL', quantity, strategy_tag="ORB")
            
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
                     await self.client.async_place_orders(long_symbol, 'SELL', quantity, strategy_tag="ORB_CLEANUP")
                return

            # SL on Short Leg (e.g., 40% of premium)
            sl_order = await self.client.async_sl_order(short_order, 40.0, strategy_tag="ORB") # % based
            
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
            self.spread_credit = (short_price - long_price) * quantity
            
            try:
                self.current_trade_id = log_entry(
                    self.db_session, short_symbol, long_symbol, 
                    short_price, long_price, lots, f"ORB_{signal}"
                )
            except Exception as e:
                logger.error(f"DB Log Error: {e}")

        self._run_async(place())

    def exit_trade(self, reason=""):
        if not self.active_position:
            return

        logger.info(f"Exiting ORB trade: {reason}")
        
        async def close():
            # Cancel SL on short leg
            if self.short_leg and self.short_leg.get('sl_order'):
                try:
                    await self.client.async_cancel_order(self.short_leg['sl_order']['orderid'])
                except: pass
            
            # Close Short Leg (Buy back)
            if self.short_leg:
                await self.client.async_place_orders(
                    self.short_leg['symbol'], 'BUY', self.short_leg['qty'], strategy_tag="ORB"
                )
            
            # Close Long Leg (Sell)
            if self.long_leg:
                await self.client.async_place_orders(
                    self.long_leg['symbol'], 'SELL', self.long_leg['qty'], strategy_tag="ORB"
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
        if not self.active_position or not self.short_leg:
            return
            
        # Check SL Hit on Short Leg via Order Status
        info = self.client.get_order_info_of_order(self.short_leg['sl_order']['orderid'])
        if info and info.get('order_status','').lower() != 'open':
            logger.info("ORB Short Leg SL Hit!")
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
            
        # Time Exit (End of session)
        now = self.now_ist().time()
        if now >= datetime.strptime("23:15", "%H:%M").time():
            self.exit_trade("Session End Exit")

    # --------------------------------------------------------
    # Tick Handling
    # --------------------------------------------------------
    def on_underlying_tick(self, tick):
        ltp = tick.get('ltp')
        ts_ms = tick.get('timestamp')
        if ltp is None: return

        dt = datetime.fromtimestamp(ts_ms/1000.0, pytz.utc).astimezone(IST)
        minute = dt.replace(second=0, microsecond=0)

        with self.lock:
            if self.curr_min is None:
                self.curr_min = minute
                self.open = self.high = self.low = self.close = ltp
                return

            if minute != self.curr_min:
                # Finalize candle
                row = {'timestamp': self.curr_min, 'open': self.open, 'high': self.high, 'low': self.low, 'close': self.close}
                self.ohlc.append(row)
                self.update_ohlc_df()
                self.compute_atrs()
                
                # Update ORB logic
                self.orb.update_ohlc(self.ohlc_df)
                
                # ENTRY LOGIC
                if not self.active_position:
                    signal = self.orb.evaluate_orb_signal()
                    if signal:
                        logger.info(f"ORB Signal: {signal}")
                        # Pass signal direction for correlation sizing
                        lots = compute_lots_from_config(self.atr_short, self.cfg, legs_count=1, signal_direction=signal)
                        if lots > 0:
                            self.enter_trade(signal, lots)
                
                # EXIT LOGIC
                self.check_intraday_squareoff()

                self.curr_min = minute
                self.open = self.high = self.low = self.close = ltp
            else:
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
        logger.info("ORB Runner Listening...")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            self.client.disconnect()

if __name__ == '__main__':
    runner = ORBRun()
    runner.start()
