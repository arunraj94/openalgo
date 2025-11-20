# run_full.py
# Fully-populated runner for the Unified Short Straddle Bot (Modular Version)
# Updated for volume-free OHLC + dynamic ATR-based RSI + no open-range logic.

import time
from datetime import datetime
import pytz
import threading
import asyncio

# modules
from config import default_config as cfg
from entries import EntryDetectors
from exit_engine import ExitEngine
from exit_engine import ExitEngine
from db_logger import init_db, log_entry, log_leg_mod, log_exit
from openalgo_client import OpenAlgoClientWrapper
from position_sizing import compute_lots_from_config
from position_sizing import compute_lots_from_config
from logger import setup_logger
from ml_engine import MLEngine

# Setup Logger
logger = setup_logger("STRADDLE")

# ATR helper
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

logger.info("🔁 OpenAlgo Python Bot Running…")


class UnifiedRun:
    def __init__(self, config=cfg):
        self.cfg = config

        # components
        self.detectors = EntryDetectors()
        self.ml_engine = MLEngine(model_path=self.cfg.ML_MODEL_PATH) if getattr(self.cfg, 'ENABLE_ML_FILTER', False) else None

        self.db_session = init_db(self.cfg.DB_PATH)

        self.client = OpenAlgoClientWrapper(
            api_key=self.cfg.OPENALGO_API_KEY,
            host=self.cfg.OPENALGO_HOST,
            ws_url=self.cfg.OPENALGO_WS,
            on_ltp_callback=self.on_ltp
        )

        self.exit_engine = ExitEngine(
            target_pct_of_credit=self.cfg.TARGET_PCT_OF_CREDIT,
            trail_factor=self.cfg.TRAIL_FACTOR,
            trail_min_buffer=self.cfg.TRAIL_MIN_BUFFER,
            no_move_wait_minutes=self.cfg.NO_MOVE_WAIT_MINUTES,
            min_decay_pct=self.cfg.MIN_DECAY_PCT,
            move_threshold_mult=self.cfg.MOVE_THRESHOLD_MULT,
            client=self.client
        )

        # OHLC
        self.ohlc = []     # list of dict candles
        self.ohlc_df = None
        self.curr_min = None
        self.open = self.high = self.low = self.close = None

        # ATRs
        self.atr_short = None
        self.atr_long = None

        # Active straddle
        self.active_straddle = False
        self.ce = None
        self.pe = None
        self.straddle_credit = 0.0
        self.current_trade_id = None
        self.current_trade_entry_time = None
        self.current_underlying_at_entry = None
        
        # Trade management state
        self.breakeven_stage1 = False
        self.breakeven_stage2 = False
        self.enable_trailing = False

        # lock
        self.lock = threading.Lock()

    # sync-run wrapper
    def _run_async(self, coro):
        return asyncio.run(coro)

    # --------------------------------------------------------
    # Close legs
    # --------------------------------------------------------
    def _close_leg(self, leg):
        if not leg or leg.get('status') == 'CLOSED':
            return None
        sl_order = leg.get('sl_order')
        if not sl_order:
            return None

        order_info = self._run_async(
            self.client.async_modify_orders_to_exit(
                sl_order['symbol'],
                'BUY',
                sl_order['quantity'],
                sl_order['orderid'],
                strategy_tag="STRADDLE"
            )
        )
        if order_info:
            leg['buy_price'] = order_info.get('price')

        leg['status'] = 'CLOSED'
        return order_info

    def _record_exit(self, reason=""):
        if not self.current_trade_id:
            return
        try:
            ce_buy = self.ce.get('buy_price') if self.ce else None
            pe_buy = self.pe.get('buy_price') if self.pe else None
            realized = self.compute_total_pnl()['realized']
            log_exit(
                self.db_session,
                self.current_trade_id,
                ce_buy=ce_buy,
                pe_buy=pe_buy,
                realized_pnl=realized,
                notes=reason
            )
        except Exception as err:
            logger.error(f"[DB] Failed exit log: {err}")

    def _close_all_legs(self, reason=""):
        if self.ce and self.ce.get('status') != 'CLOSED':
            self._close_leg(self.ce)
        if self.pe and self.pe.get('status') != 'CLOSED':
            self._close_leg(self.pe)
        self._record_exit(reason)
        self.clear_active()

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------
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

    def compute_atrs(self):
        if self.ohlc_df is None or len(self.ohlc_df) < max(
                self.cfg.ATR_LONG_PERIOD, self.cfg.ATR_SHORT_PERIOD
        ) + 1:
            self.atr_short = None
            self.atr_long = None
            return

        highs = self.ohlc_df['high']
        lows = self.ohlc_df['low']
        closes = self.ohlc_df['close']
        try:
            self.atr_short = compute_atr(highs, lows, closes, period=self.cfg.ATR_SHORT_PERIOD)
            self.atr_long = compute_atr(highs, lows, closes, period=self.cfg.ATR_LONG_PERIOD)
        except Exception as e:
            logger.error(f"ATR compute error: {e}")
            self.atr_short = self.atr_long = None

        # feed ATRs into detectors
        self.detectors.latest_atr_short = self.atr_short
        self.detectors.latest_atr_long = self.atr_long

    # --------------------------------------------------------
    # Liquidity Check
    # --------------------------------------------------------
    def check_liquidity(self, symbol):
        """
        Checks if the option has sufficient liquidity.
        Criteria:
        1. Bid and Ask exist (qty > 0)
        2. Spread <= 5 ticks (approx 0.25 for Crude) OR Spread <= 2% of LTP
        """
        try:
            depth = self.client.get_market_depth(symbol)
            if not depth:
                logger.warning(f"Liquidity Warning: No depth for {symbol}")
                return False
                
            # OpenAlgo depth structure: {'bids': [{'price': p, 'quantity': q}, ...], 'asks': ...}
            bids = depth.get('bids', [])
            asks = depth.get('asks', [])
            
            if not bids or not asks:
                logger.warning(f"Liquidity Warning: Missing bids/asks for {symbol}")
                return False
                
            best_bid = bids[0]['price']
            best_ask = asks[0]['price']
            
            if best_bid <= 0 or best_ask <= 0:
                return False
                
            spread = best_ask - best_bid
            
            # 5 ticks (assuming 0.05 tick size) = 0.25
            # Or 2% of premium
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
    # Place straddle
    # --------------------------------------------------------
    def place_straddle(self, lots, entry_signal=None):
        atm = self.client.get_ATM_strike(self.client.get_exchange_ltp())
        ce_symbol = self.client.build_option_symbol(
            self.cfg.UNDERLYING_SYMBOL, self.client.nearest_expiry, atm, "CE"
        )['symbol']
        pe_symbol = self.client.build_option_symbol(
            self.cfg.UNDERLYING_SYMBOL, self.client.nearest_expiry, atm, "PE"
        )['symbol']

        # Global Risk Check
        import os
        if os.path.exists("STOP_TRADING"):
            logger.warning("⛔ Global Risk Stop Triggered (STOP_TRADING file found). Skipping Trade.")
            return

        # Liquidity Check
        if not self.check_liquidity(ce_symbol) or not self.check_liquidity(pe_symbol):
            logger.warning("Skipping Straddle: Poor Liquidity")
            return

        # # ML Filter Check
        # if self.ml_engine:
        #     confidence = self.ml_engine.predict()
        #     if confidence < self.cfg.ML_CONFIDENCE_THRESHOLD:
        #         logger.warning(f"⛔ ML Filter Blocked Trade. Confidence: {confidence:.2f} < {self.cfg.ML_CONFIDENCE_THRESHOLD}")
        #         return
        #     logger.info(f"✅ ML Filter Passed. Confidence: {confidence:.2f}")

        async def place():
            quantity = lots * self.cfg.LOTSIZE

            base_pct = self.get_base_sl_percent_for_now()
            mult = self.get_atr_multiplier()
            sl_pct = base_pct * mult

            # 3. Place Orders (Async)
            # Use NRML (Normal) product type via updated client
            ce_order = await self.client.async_place_orders(ce_symbol, 'SELL', quantity, strategy_tag="STRADDLE")
            pe_order = await self.client.async_place_orders(pe_symbol, 'SELL', quantity, strategy_tag="STRADDLE")

            # 4. Check Order Status
            # if not ce_order or ce_order.get('order_status') != 'complete':
            #     logger.error(f"CE Order Failed/Rejected: {ce_order}")
            #     # Attempt to cancel PE if CE failed (cleanup)
            #     if pe_order and pe_order.get('order_status') == 'open':
            #         await self.client.async_cancel_order(pe_order['orderid'])
            #     return
            #
            # if not pe_order or pe_order.get('order_status') != 'complete':
            #     logger.error(f"PE Order Failed/Rejected: {pe_order}")
            #     # Attempt to close CE if PE failed (cleanup)
            #     if ce_order and ce_order.get('order_status') == 'complete':
            #         await self.client.async_place_orders(ce_symbol, 'BUY', quantity, strategy_tag="STRADDLE_CLEANUP")
            #     return

            # 6. Place Stop Loss Orders FIRST (with proper checks)
            ce_sl_order = None
            pe_sl_order = None
            ce_price = 0
            pe_price = 0
            
            if ce_order and ce_order.get('order_status') == 'complete':
                ce_price = float(ce_order.get('average_price', ce_order.get('price', 0.0)))
                ce_sl_order = await self.client.async_sl_order(ce_order, sl_pct, strategy_tag="STRADDLE")
                self.ce = {
                    'symbol': ce_symbol,
                    'status': 'OPEN',
                    'sell_price': ce_price,
                    'current_price': ce_price,
                    'qty': quantity,
                    'sl_order': ce_sl_order
                }
            else:
                logger.warning(f"CE order not complete, skipping SL: {ce_order}")
                
            if pe_order and pe_order.get('order_status') == 'complete':
                pe_price = float(pe_order.get('average_price', pe_order.get('price', 0.0)))
                pe_sl_order = await self.client.async_sl_order(pe_order, sl_pct, strategy_tag="STRADDLE")
                self.pe = {
                    'symbol': pe_symbol,
                    'status': 'OPEN',
                    'sell_price': pe_price,
                    'current_price': pe_price,
                    'qty': quantity,
                    'sl_order': pe_sl_order
                }
            else:
                logger.warning(f"PE order not complete, skipping SL: {pe_order}")

            # 7. Mark position as active and initialize tracking
            credit = ce_price + pe_price
            self.active_straddle = True
            self.ce_symbol = ce_symbol
            self.pe_symbol = pe_symbol
            self.entry_price = credit  # Combined credit
            self.entry_time = self.now_ist()
            self.sl_percentage = sl_pct
            
            self.straddle_credit = credit
            self.current_trade_entry_time = self.entry_time
            self.current_underlying_at_entry = self.client.get_exchange_ltp()

            logger.info(f"Straddle Active. Credit: {credit:.2f}, SL: {sl_pct}%")
            
            # 8. Log to DB
            try:
                self.current_trade_id = log_entry(
                    self.db_session, ce_symbol, pe_symbol, 
                    ce_price, pe_price, lots, entry_signal
                )
            except Exception as e:
                logger.error(f"DB Log Error: {e}")

        asyncio.run(place())

    # SL% based on time bucket
    def get_base_sl_percent_for_now(self):
        now = self.now_ist().time()
        for s,e,p in self.cfg.TIME_BASE_SL_BUCKETS:
            if s <= now < e:
                return p
        return 18.0

    # ATR multiplier
    def get_atr_multiplier(self):
        if self.atr_short is None or self.atr_long is None:
            return 1.0
        if self.atr_short > self.atr_long * 1.0:
            return self.cfg.ATR_MULTIPLIER_HIGH
        if self.atr_short < self.atr_long * 0.5:
            return self.cfg.ATR_MULTIPLIER_LOW
        return 1.0

    # --------------------------------------------------------
    # Tick Handling
    # --------------------------------------------------------
    def on_underlying_tick(self, tick):
        ltp = tick.get('ltp')
        ts_ms = tick.get('timestamp')
        if ltp is None or ts_ms is None:
            return

        dt = datetime.fromtimestamp(ts_ms/1000.0, pytz.utc).astimezone(IST)
        minute = dt.replace(second=0, microsecond=0)

        with self.lock:
            # New candle
            if self.curr_min is None:
                self.curr_min = minute
                self.open = self.high = self.low = self.close = ltp
                return

            if minute != self.curr_min:
                # finalize previous minute
                row = {
                    'timestamp': self.curr_min,
                    'open': self.open,
                    'high': self.high,
                    'low': self.low,
                    'close': self.close
                }
                logger.info(f"Candle: {row}")

                self.ohlc.append(row)
                self.update_ohlc_df()

                # ATR calc
                self.compute_atrs()

                # feed OHLC to detectors
                self.detectors.update_ohlc(self.ohlc_df)

                # ENTRY LOGIC
                if self.in_allowed_window() and not self.active_straddle:
                    sig = self.detectors.evaluate_priority_signals()
                    # sig = self.entry.evaluate_()priority_signals()
                    logger.info(f"Signal: {sig}")
                    if sig:
                        lots = compute_lots_from_config(self.atr_short, self.cfg)
                        logger.info(f"Lots: {lots}")
                        # lots = 1  # REMOVED hardcoded override
                        if lots > 0:
                            self.place_straddle(lots, entry_signal=sig)

                # EXIT LOGIC
                self.check_intraday_squareoff()
                self.check_and_book_survivor()

                # reset for next minute
                self.curr_min = minute
                self.open = self.high = self.low = self.close = ltp

            else:
                # update running candle
                self.close = ltp
                self.high = max(self.high, ltp)
                self.low = min(self.low, ltp)
                if self.active_straddle:
                    self.manage_active()


    def on_option_tick(self, symbol, ltp):
        if self.ce and symbol == self.ce['symbol']:
            self.ce['current_price'] = ltp

        if self.pe and symbol == self.pe['symbol']:
            self.pe['current_price'] = ltp

        self.detectors.set_option_prices(
            ce_price=ltp if self.ce and symbol == self.ce['symbol'] else None,
            pe_price=ltp if self.pe and symbol == self.pe['symbol'] else None
        )

    # LTP callback
    def on_ltp(self, data):
        if data.get('type') != 'market_data':
            return
        symbol = data.get('symbol')
        exch = data.get('exchange')
        tick = data.get('data', {})

        if exch != self.cfg.UNDERLYING_EXCHANGE:
            return

        if symbol == self.client.future_symbol:
            self.on_underlying_tick(tick)
        else:
            self.on_option_tick(symbol, tick.get('ltp'))



    # --------------------------------------------------------
    # Exit Management
    # --------------------------------------------------------
    def handle_leg_sl_hit(self, hit_leg):
        logger.info(f"LEG SL HIT: {hit_leg['symbol']}")
        hit_leg['status'] = 'CLOSED'

        # Smart Survivor Decision
        decision, reason = self.exit_engine.smart_survivor_decision(self, hit_leg)
        
        if decision == "EXIT_ALL":
            logger.info(f"Smart Exit Triggered: {reason}")
            self._close_all_legs(f"Smart Exit: {reason}")
            return

        # If keeping survivor
        survivor = self.ce if hit_leg is self.pe else self.pe
        if survivor and survivor.get('status') != 'CLOSED':
            self.exit_engine.move_to_breakeven(survivor)
            self.exit_engine.update_trailing_sl(survivor)
            self.check_and_book_survivor()
        else:
            self._record_exit("Both legs closed via SL")
            self.clear_active()

    def check_and_book_survivor(self):
        pnl = self.compute_total_pnl()
        target = self.straddle_credit * self.exit_engine.target_pct_of_credit

        if pnl['total'] >= target:
            self._close_all_legs("Target hit")
            return

        if self.exit_engine.time_exit_due():
            self._close_all_legs("Time exit")
            return

        # Progressive Breakeven Check
        be_action = self.exit_engine.progressive_breakeven_check(self)
        if be_action == "STAGE1_25PCT":
            logger.info("Profit > 25%: Moving SL to reduce risk (10%)")
            # Logic to tighten SL would go here - for now we just mark stage
            self.breakeven_stage1 = True
            # Example: tighten both legs by moving SL closer
            # self.tighten_sl_both_legs(pct=10.0) 

        elif be_action == "STAGE2_40PCT":
            logger.info("Profit > 40%: Moving to Breakeven")
            if self.ce and self.ce['status'] == 'OPEN':
                self.exit_engine.move_to_breakeven(self.ce)
            if self.pe and self.pe['status'] == 'OPEN':
                self.exit_engine.move_to_breakeven(self.pe)
            self.breakeven_stage2 = True
            self.enable_trailing = True  # Enable trailing from here

        # Dual Leg Trailing (if enabled)
        if self.enable_trailing or self.exit_engine.check_dual_leg_trail(self):
            self.enable_trailing = True
            if self.ce and self.ce['status'] == 'OPEN':
                self.exit_engine.update_trailing_sl(self.ce)
            if self.pe and self.pe['status'] == 'OPEN':
                self.exit_engine.update_trailing_sl(self.pe)

        exit_flag, reason = self.exit_engine.should_exit_no_move(self)
        if exit_flag:
            logger.info(f"Early Exit: {reason}")
            self._close_all_legs(f"Early exit: {reason}")
            return

        # Trailing for single survivor (legacy logic, kept as fallback)
        survivor = None
        if self.ce and self.ce['status'] == 'OPEN' and (not self.pe or self.pe['status'] == 'CLOSED'):
            survivor = self.ce
        if self.pe and self.pe['status'] == 'OPEN' and (not self.ce or self.ce['status'] == 'CLOSED'):
            survivor = self.pe

        if survivor:
            self.exit_engine.update_trailing_sl(survivor)

    def compute_total_pnl(self):
        realized = unreal = 0.0
        if self.ce:
            if self.ce['status'] == 'CLOSED':
                realized += (self.ce['sell_price'] - self.ce.get('buy_price', self.ce['sell_price'])) * self.ce['qty']
            else:
                unreal += (self.ce['sell_price'] - self.ce['current_price']) * self.ce['qty']

        if self.pe:
            if self.pe['status'] == 'CLOSED':
                realized += (self.pe['sell_price'] - self.pe.get('buy_price', self.pe['sell_price'])) * self.pe['qty']
            else:
                unreal += (self.pe['sell_price'] - self.pe['current_price']) * self.pe['qty']

        return {'realized': realized, 'unrealized': unreal, 'total': realized + unreal}

    # reset trade
    def clear_active(self):
        self.active_straddle = False
        self.ce = None
        self.pe = None
        self.straddle_credit = 0.0
        self.current_trade_id = None
        self.current_trade_entry_time = None
        self.current_underlying_at_entry = None

    def manage_active(self):
        # check SL hits via order status
        # logger.debug(f"CE: {self.ce}") # Too verbose
        # logger.debug(f"PE: {self.pe}")
        if self.ce and self.ce.get('status') == 'OPEN':
            info = self.client.get_order_info_of_order(self.ce['sl_order']['orderid'])
            # print(info)
            if info and info.get('order_status','').lower() != 'open':
                self.ce['buy_price'] = info.get('price')
                self.handle_leg_sl_hit(self.ce)
                return

        if self.pe and self.pe.get('status') == 'OPEN':
            info = self.client.get_order_info_of_order(self.pe['sl_order']['orderid'])
            # logger.debug(f"PE Order Info: {info}")
            if info and info.get('order_status','').lower() != 'open':
                self.pe['buy_price'] = info.get('price')
                self.handle_leg_sl_hit(self.pe)
                return

        # no SL hit
        
    def check_intraday_squareoff(self):
        now = self.now_ist().time()
        if now >= self.cfg.INTRADAY_SQUAREOFF_TIME:
            if self.active_straddle:
                logger.warning(f"⏰ Intraday Square-off Time Reached ({now}). Closing all positions.")
                self._close_all_legs("Intraday Square-off")
       
    # --------------------------------------------------------
    # Start runner
    # --------------------------------------------------------
    def start(self):
        self.client.connect()

        subs_sym = self.client.get_option_symbols(self.cfg.UNDERLYING_SYMBOL, self.cfg.UNDERLYING_EXCHANGE)

        self.client.subscribe_ltp(subs_sym)
        self.client.subscribe_depth(subs_sym)
        self.client.subscribe_orderbook()

        logger.info("Runner started. Listening…")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping runner…")
            self.client.disconnect()


if __name__ == '__main__':
    runner = UnifiedRun()
    runner.start()