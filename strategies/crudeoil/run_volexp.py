# run_volexp.py
# Runner for Volatility Expansion Strategy (Option Buying)
# Buys ATM Straddles during extreme compression

import time
from datetime import datetime
import pytz
import threading
import asyncio
import os

# modules
from config import default_config as cfg
from entries_volatility_expansion import VolatilityExpansion
from db_logger import init_db, log_entry, log_exit
from openalgo_client import OpenAlgoClientWrapper
from position_sizing import compute_lots_from_config
from logger import setup_logger
from ml_engine import MLEngine

# Setup Logger
logger = setup_logger("VOLEXP")

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

logger.info("💥 Volatility Expansion Runner Starting...")


class VolExpRun:
    def __init__(self, config=cfg):
        self.cfg = config
        self.detector = VolatilityExpansion(config=config)
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
        self.atr_long = None

        # Active Position
        self.active_position = False
        self.ce = None
        self.pe = None
        self.current_trade_id = None
        self.entry_time = None
        self.total_premium_paid = 0.0
        self.highest_profit_pct = 0.0  # Track peak profit for trailing
        self.trailing_active = False   # Trailing stop activated?
        
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
            if best_bid <= 0 or best_ask <= 0:
                return False
                
            spread = best_ask - best_bid
            max_spread_ticks = 0.25
            max_spread_pct = best_ask * 0.02
            threshold = max(max_spread_ticks, max_spread_pct)
            if spread > threshold:
                logger.warning(f"Liquidity Warning: Spread {spread:.2f} > {threshold:.2f} for {symbol}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Liquidity Check Error: {e}")
            return False

    # --------------------------------------------------------
    # Trade Management
    # --------------------------------------------------------
    def manage_trade(self):
        """Enhanced trade management with trailing profit protection"""
        if not self.active_position: 
            return

        # Calculate PnL
        current_pnl = 0.0
        invested = 0.0
        
        if self.ce:
            pnl = (self.ce['current_price'] - self.ce['buy_price']) * self.ce['qty']
            current_pnl += pnl
            invested += self.ce['buy_price'] * self.ce['qty']
            
        if self.pe:
            pnl = (self.pe['current_price'] - self.pe['buy_price']) * self.pe['qty']
            current_pnl += pnl
            invested += self.pe['buy_price'] * self.pe['qty']
            
        if invested > 0:
            profit_pct = (current_pnl / invested) * 100.0
        else:
            profit_pct = 0.0
        
        # ========================================
        # 1. TRAILING PROFIT PROTECTION (New!)
        # ========================================
        TRAILING_ACTIVATION = 10.0  # Activate trailing at +10%
        TRAILING_OFFSET = 10.0      # Trail by 10% from peak
        
        if profit_pct >= TRAILING_ACTIVATION:
            # Update highest profit
            if profit_pct > self.highest_profit_pct:
                self.highest_profit_pct = profit_pct
                logger.info(f"📈 New Peak Profit: {profit_pct:.1f}%")
                
                if not self.trailing_active:
                    self.trailing_active = True
                    logger.info(f"✅ Trailing Stop ACTIVATED (Peak: {profit_pct:.1f}%)")
            
            # Calculate trailing threshold
            trailing_threshold = self.highest_profit_pct - TRAILING_OFFSET
            
            # Check if profit dropped below trailing threshold
            if profit_pct <= trailing_threshold:
                logger.warning(f"📉 Trailing Stop Hit!")
                logger.info(f"   Peak Profit: {self.highest_profit_pct:.1f}%")
                logger.info(f"   Current Profit: {profit_pct:.1f}%")
                logger.info(f"   Drop from Peak: {self.highest_profit_pct - profit_pct:.1f}%")
                self.exit_trade(f"Trailing Stop: Lock Profit at {profit_pct:.1f}% (Peak: {self.highest_profit_pct:.1f}%)")
                return
        
        # ========================================
        # 2. PROFIT TARGET
        # ========================================
        if profit_pct >= self.cfg.VOLEXP_TARGET_PREMIUM_PCT:
            self.exit_trade(f"Target Hit ({self.cfg.VOLEXP_TARGET_PREMIUM_PCT:.0f}%)")
            return
        
        # ========================================
        # 3. STOP LOSS
        # ========================================
        if profit_pct <= self.cfg.VOLEXP_STOP_PREMIUM_PCT:
            self.exit_trade(f"Stop Loss Hit ({self.cfg.VOLEXP_STOP_PREMIUM_PCT:.0f}%)")
            return
        
        # ========================================
        # 4. TIME EXIT
        # ========================================
        elapsed_mins = (self.now_ist() - self.entry_time).total_seconds() / 60.0
        if elapsed_mins > self.cfg.VOLEXP_TIME_EXIT_MINUTES and profit_pct < 5.0:
            self.exit_trade("Time Exit (No Move)")

    def place_trade(self, signal, lots):
        logger.info(f"VolExp Signal: {signal} -> Buying Straddle {lots} lots")

        # ML Filter Check
        if self.ml_engine:
            confidence = self.ml_engine.predict()
            if confidence < self.cfg.ML_CONFIDENCE_THRESHOLD:
                logger.warning(f"⛔ ML Filter Blocked Trade. Confidence: {confidence:.2f} < {self.cfg.ML_CONFIDENCE_THRESHOLD}")
                return
            logger.info(f"✅ ML Filter Passed. Confidence: {confidence:.2f}")

        ltp = self.client.get_exchange_ltp()
        strike = self.client.get_ATM_strike(ltp)
        
        ce_sym_info = self.client.build_option_symbol(self.cfg.UNDERLYING_SYMBOL, self.client.nearest_expiry, strike, "CE")
        pe_sym_info = self.client.build_option_symbol(self.cfg.UNDERLYING_SYMBOL, self.client.nearest_expiry, strike, "PE")
        
        ce_symbol = ce_sym_info['symbol']
        pe_symbol = pe_sym_info['symbol']
        
        if not self.check_liquidity(ce_symbol) or not self.check_liquidity(pe_symbol):
            logger.warning("Skipping VolExp: Poor Liquidity")
            return

        async def place():
            qty = lots * self.cfg.LOTSIZE
            
            # Place Both Orders in Parallel (minimize slippage)
            ce_order, pe_order = await asyncio.gather(
                self.client.async_place_orders(ce_symbol, 'BUY', qty, strategy_tag="VOLEXP"),
                self.client.async_place_orders(pe_symbol, 'BUY', qty, strategy_tag="VOLEXP")
            )
            
            self.ce = {
                'symbol': ce_symbol, 'buy_price': ce_order['price'], 'qty': qty, 
                'current_price': ce_order['price']
            }
            self.pe = {
                'symbol': pe_symbol, 'buy_price': pe_order['price'], 'qty': qty, 
                'current_price': pe_order['price']
            }
            
            self.active_position = True
            self.entry_time = self.now_ist()
            self.highest_profit_pct = 0.0  # Reset profit tracking
            self.trailing_active = False   # Reset trailing flag
            
            try:
                self.current_trade_id = log_entry(
                    self.db_session, ce_symbol, pe_symbol, 
                    ce_order['price'], pe_order['price'], lots, f"VOLEXP_{signal}"
                )
            except Exception as e:
                logger.error(f"DB Log Error: {e}")

        self._run_async(place())

    def exit_trade(self, reason=""):
        if not self.active_position: return
        logger.info(f"Exiting VolExp: {reason}")
        
        async def close():
            if self.ce:
                await self.client.async_place_orders(self.ce['symbol'], 'SELL', self.ce['qty'], strategy_tag="VOLEXP")
            if self.pe:
                await self.client.async_place_orders(self.pe['symbol'], 'SELL', self.pe['qty'], strategy_tag="VOLEXP")
                
            try:
                realized = 0.0
                if self.ce: realized += (self.ce['current_price'] - self.ce['buy_price']) * self.ce['qty']
                if self.pe: realized += (self.pe['current_price'] - self.pe['buy_price']) * self.pe['qty']
                
                log_exit(self.db_session, self.current_trade_id, 
                         ce_buy=None, pe_buy=None, realized_pnl=realized, notes=reason)
            except: pass
            
            self.active_position = False
            self.ce = None
            self.pe = None
            self.current_trade_id = None
            self.highest_profit_pct = 0.0
            self.trailing_active = False
            
        self._run_async(close())

    # --------------------------------------------------------
    # Tick Handling
    # --------------------------------------------------------
    def on_underlying_tick(self, tick):
        ltp = tick.get('ltp')
        ts_ms = tick.get('timestamp')
        if ltp is None: return

        dt = datetime.fromtimestamp(ts_ms/1000.0, pytz.utc).astimezone(IST)
        
        # CHANGED: Use 5-minute candles for volatility expansion
        # ATR compression/expansion is more meaningful on 5-min vs 1-min noise
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
                
                # Update detector with 5-min data
                self.detector.update_ohlc(self.ohlc_df)
                self.detector.latest_atr_short = self.atr_short
                self.detector.latest_atr_long = self.atr_long
                
                self.check_intraday_squareoff()
                
                if self.active_position:
                    self.manage_trade()
                else:
                    # Check for Entry - 5-min volatility expansion signals
                    signal = self.detector.evaluate_expansion_signals()
                    if signal:
                        logger.info(f"[5-MIN] VolExp Signal: {signal}")
                        lots = compute_lots_from_config(self.cfg)
                        self.place_trade(signal, lots)
                
                # Reset for next 5-min candle
                self.curr_min = candle_5min
                self.open = self.high = self.low = self.close = ltp
            else:
                # Update current 5-min candle
                self.close = ltp
                self.high = max(self.high, ltp)
                self.low = min(self.low, ltp)

    def on_option_tick(self, symbol, ltp):
        if self.ce and symbol == self.ce['symbol']:
            self.ce['current_price'] = ltp
        if self.pe and symbol == self.pe['symbol']:
            self.pe['current_price'] = ltp

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
        self.client.subscribe_ltp(subs_sym)
        self.client.subscribe_depth(subs_sym)
        self.client.subscribe_orderbook()
        logger.info("VolExp Runner Listening...")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            self.client.disconnect()

if __name__ == '__main__':
    runner = VolExpRun()
    runner.start()
