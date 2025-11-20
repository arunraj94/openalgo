# EntryDetectors.py
# Expanded entry detectors for Unified Short Straddle Bot (MCX:CRUDEOILM)
# Includes original detectors plus merged new signals per user request:
# 1) atr_squeeze_breakout
# 2) micro_trend_exhaustion
# 3) push_fail_reversal
# 4) atr_pinch_bounce
# 5) option_premium_compression
# 6) mid_session_vol_reset

from datetime import datetime, time as dt_time
import pytz
import pandas as pd
from typing import Optional

IST = pytz.timezone("Asia/Kolkata")


class EntryDetectors:
    """
    Encapsulates entry detectors used by the unified straddle bot.

    Usage:
      - call update_ohlc(df) with a DataFrame containing ['timestamp','open','high','low','close']
      - runner should set latest_atr_short and latest_atr_long on this object
      - runner should call set_option_prices(ce_price, pe_price) each option tick (optional)
    """

    def __init__(self):
        # compression params
        self.n_body_avg = 10
        self.body_pct_threshold = 0.40
        self.min_consec = 2

        # Bollinger params
        self.bb_len = 20
        self.bbwidth_lookback = 60
        # BBWidth percentile (Relaxed from 25 to 30)
        self.bbwidth_percentile = 30

        # Price anchor (VWAP / TWAP proxy)
        self.vwap_dev_pct = 0.25

        # Anchor reversion params (Renamed from VWAP)
        self.anchor_reversion_lookback = 10
        self.anchor_reversion_dev_pct = 0.5
        self.anchor_reversion_close_pct = 0.25

        # RSI base parameters
        self.rsi_length = 14

        # Range-break reversal params
        self.range_break_lookback = 15
        self.range_break_reversion_pct = 0.15  # percent

        # US proximity params
        self.us_session_start = dt_time(20, 0)
        self.us_proximity_minutes = 20

        # ATR squeeze / extra detectors tuning
        # (these can be tweaked from runner by setting attributes)
        self.atr_squeeze_ratio = 0.7
        self.atr_pinch_mult = 0.7

        # Trend Filter (Relaxed from 25 to 30)
        self.adx_period = 14
        self.adx_threshold = 30

        # RSI Divergence
        self.rsi_div_lookback = 10

        # internal OHLC storage
        self.ohlc: Optional[pd.DataFrame] = None

        # ATR values pushed by runner
        self.latest_atr_short = None
        self.latest_atr_long = None

        # option prices (runner should push these)
        self.ce_price = None
        self.pe_price = None
        self.ce_prev = None
        self.pe_prev = None

    # -------------------------------------------------------------
    # Updating OHLC (volume completely removed)
    # -------------------------------------------------------------
    def update_ohlc(self, ohlc: pd.DataFrame):
        """
        Provide latest OHLC DataFrame. Expect: ['timestamp','open','high','low','close']
        Volume not required. Timestamps will be localized to IST if naive.
        """
        if ohlc is None or len(ohlc) == 0:
            self.ohlc = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])
            return

        df = ohlc.copy()

        # Remove volume if present
        if 'volume' in df.columns:
            df = df.drop(columns=['volume'])

        # Ensure tz-aware timestamps
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(IST)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(IST)

        self.ohlc = df.reset_index(drop=True)

    # -------------------------------------------------------------
    # Option price helper
    # -------------------------------------------------------------
    def set_option_prices(self, ce_price: Optional[float], pe_price: Optional[float]):
        """
        Runner should call this each option tick. Maintains previous values for compression checks.
        """
        try:
            if ce_price is not None:
                self.ce_prev = self.ce_price
                self.ce_price = float(ce_price)
            if pe_price is not None:
                self.pe_prev = self.pe_price
                self.pe_price = float(pe_price)
        except Exception:
            # be defensive — ignore bad ticks
            pass

    # -------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------
    def _compute_price_anchor(self) -> Optional[float]:
        df = self.ohlc
        if df is None or len(df) == 0:
            return None
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        if len(tp) >= self.bb_len:
            return float(tp.iloc[-self.bb_len:].mean())
        return float(tp.mean())

    def _bbwidth_series(self) -> pd.Series:
        df = self.ohlc
        if df is None or len(df) == 0:
            return pd.Series(dtype=float)
        closes = df['close']
        if len(closes) < self.bb_len:
            return pd.Series([0.0] * len(closes))
        ma = closes.rolling(self.bb_len).mean()
        std = closes.rolling(self.bb_len).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        # avoid division by zero
        denom = ma.replace(0, 1e-9)
        return (upper - lower) / denom

    def _compute_rsi(self, period: int = None) -> Optional[float]:
        df = self.ohlc
        if df is None or len(df) == 0:
            return None
        period = period or self.rsi_length
        closes = df['close']
        if len(closes) <= period:
            return None
        delta = closes.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.rolling(window=period, min_periods=period).mean()
        avg_loss = losses.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        last = rsi.iloc[-1]
        return None if pd.isna(last) else float(last)

    def _compute_adx(self, period: int = 14) -> Optional[float]:
        df = self.ohlc
        if df is None or len(df) < period * 2:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='outer').max(axis=1)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (minus_dm.abs().ewm(alpha=1/period).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        
        if len(adx) == 0: return None
        return float(adx.iloc[-1])

    def trend_filter(self) -> bool:
        """
        Returns True if market is NOT trending (ADX < Threshold).
        """
        adx = self._compute_adx(self.adx_period)
        if adx is None:
            return True # Default to allow if no data
        
        if adx > self.adx_threshold:
            return self._debug_return('trend_filter', False, f'ADX {adx:.1f} > {self.adx_threshold}')
        
        return True

    def rsi_divergence(self) -> bool:
        """
        Detects RSI Divergence (Bearish or Bullish).
        """
        df = self.ohlc
        if df is None or len(df) < 20:
            return False

        rsi_series = []
        # We need a series of RSI, not just the last value.
        # Re-implementing RSI series calc locally for efficiency
        period = self.rsi_length
        closes = df['close']
        delta = closes.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        if len(rsi) < self.rsi_div_lookback + 5:
            return False

        # Look for peaks/troughs in the last N candles
        lookback = self.rsi_div_lookback
        
        # Current price/rsi
        curr_price = closes.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        
        # Find recent peak/trough
        # Simple logic: compare current vs max/min of lookback window
        
        # Bearish Divergence: Price Higher High, RSI Lower High
        recent_price_high = closes.iloc[-lookback:-1].max()
        recent_rsi_high = rsi.iloc[-lookback:-1].max()
        
        if curr_price > recent_price_high and curr_rsi < recent_rsi_high:
             # Check if RSI is in overbought territory to strengthen signal
             if recent_rsi_high > 60:
                 return self._debug_return('rsi_divergence', True, 'Bearish Divergence')

        # Bullish Divergence: Price Lower Low, RSI Higher Low
        recent_price_low = closes.iloc[-lookback:-1].min()
        recent_rsi_low = rsi.iloc[-lookback:-1].min()
        
        if curr_price < recent_price_low and curr_rsi > recent_rsi_low:
            if recent_rsi_low < 40:
                return self._debug_return('rsi_divergence', True, 'Bullish Divergence')
                
        return False

    def _log_return_false(self, detector: str, reason: str):
        # lightweight logger for debugging — runner can mute
        try:
            print(f"[EntryDetectors:{detector}] {reason}")
        except Exception:
            pass

    def _debug_return(self, detector: str, value, reason: Optional[str] = None):
        """
        Centralized debug printer to trace every indicator return path.
        """
        try:
            ts = datetime.now(IST).strftime("%H:%M:%S")
            msg = f"[EntryDetectors:{detector}] return={value}"
            if reason:
                msg += f" | {reason}"
            print(f"{ts} {msg}")
        except Exception:
            pass
        return value

    # -------------------------------------------------------------
    # Dynamic RSI band based on ATR regime
    # -------------------------------------------------------------
    def _get_dynamic_rsi_band(self, atr_short: float, atr_long: float):
        """
        High vol     (atr_short > atr_long * 1.1): 40–60
        Normal       (≈):                           35–65
        Tight range  (atr_short < atr_long * 0.8):  45–55
        """
        if atr_short is None or atr_long is None:
            return (35, 65)

        if atr_short > atr_long * 1.1:
            return (40, 60)

        if atr_short < atr_long * 0.8:
            return (45, 55)

        return (35, 65)

    # -------------------------------------------------------------
    # 1) Compression Detector (kept from original logic)
    # -------------------------------------------------------------
    def compression_detected(self) -> bool:
        df = self.ohlc
        if df is None or len(df) == 0:
            self._log_return_false('compression', 'No OHLC data')
            return self._debug_return('compression', False, 'No OHLC data')

        min_needed = max(self.n_body_avg, self.bbwidth_lookback) + self.min_consec + 2
        if len(df) < min_needed:
            self._log_return_false('compression', f'Need {min_needed} candles')
            return self._debug_return('compression', False, f'Need {min_needed} candles')

        bodies = (df['close'] - df['open']).abs()
        avg_body = bodies.iloc[-(self.n_body_avg + self.min_consec):-self.min_consec].mean()
        if avg_body <= 0:
            return self._debug_return('compression', False, 'Average body <= 0')

        last_bodies = bodies.iloc[-self.min_consec:]
        if (last_bodies > self.body_pct_threshold * avg_body).any():
            self._log_return_false('compression', 'Bodies not compressed')
            return self._debug_return('compression', False, 'Bodies not compressed')

        # BBWidth threshold
        bbw = self._bbwidth_series()
        if len(bbw) < self.bbwidth_lookback:
            return self._debug_return('compression', False, 'Insufficient BBWidth history')
        recent_bb = bbw.iloc[-self.bbwidth_lookback:]
        threshold = recent_bb.quantile(self.bbwidth_percentile / 100.0)
        if bbw.iloc[-1] > threshold:
            self._log_return_false('compression', 'BBWidth too high')
            return self._debug_return('compression', False, 'BBWidth too high')

        # Price anchor check
        anchor = self._compute_price_anchor()
        if anchor is None or anchor == 0:
            return self._debug_return('compression', False, 'No anchor')
        price = float(df['close'].iloc[-1])
        dev = abs((price - anchor) / anchor) * 100.0
        if dev > self.vwap_dev_pct:
            return self._debug_return('compression', False, f'Price dev {dev:.2f}% > {self.vwap_dev_pct}%')

        # ---- Dynamic RSI band ----
        rsi = self._compute_rsi()
        if rsi is None:
            return self._debug_return('compression', False, 'RSI unavailable')

        band_lo, band_hi = self._get_dynamic_rsi_band(
            self.latest_atr_short, self.latest_atr_long
        )

        if not (band_lo <= rsi <= band_hi):
            self._log_return_false(
                'compression',
                f'RSI {rsi:.1f} outside band [{band_lo}, {band_hi}]'
            )
            return self._debug_return('compression', False, f'RSI {rsi:.1f} outside band [{band_lo}, {band_hi}]')

        return self._debug_return('compression', True, 'Compression + RSI band satisfied')

    # -------------------------------------------------------------
    # 2) Anchor Reversion Detector (formerly VWAP Reversion)
    # -------------------------------------------------------------
    def anchor_reversion(self, lookback_minutes=None, dev_pct=None) -> bool:
        """
        Checks if price deviated from Price Anchor (SMA of Typical Price) and is reverting.
        Note: This is a proxy for VWAP since volume data is unavailable.
        """
        df = self.ohlc
        if df is None or len(df) < 3:
            return self._debug_return('anchor_reversion', False, 'Insufficient OHLC data')

        lookback_minutes = lookback_minutes or self.anchor_reversion_lookback
        dev_pct = dev_pct or self.anchor_reversion_dev_pct

        anchor = self._compute_price_anchor()
        if anchor is None:
            return self._debug_return('anchor_reversion', False, 'No anchor computed')

        lookback = min(len(df), lookback_minutes)
        past = df.iloc[-lookback:]
        devs = abs((past['close'] - anchor) / anchor) * 100.0
        if (devs >= dev_pct).any():
            last_dev = abs((df['close'].iloc[-1] - anchor) / anchor) * 100.0
            if last_dev <= self.anchor_reversion_close_pct:
                return self._debug_return('anchor_reversion', True, f'Reverted to anchor within {self.anchor_reversion_close_pct}%')

        return self._debug_return('anchor_reversion', False, 'No qualifying Anchor reversion')

    # -------------------------------------------------------------
    # 3) Range Break Reversal Detector
    # -------------------------------------------------------------
    def range_break_reversal(self, lookback=None, revert_pct=None) -> bool:
        df = self.ohlc
        if df is None or len(df) < 4:
            return self._debug_return('range_break_reversal', False, 'Insufficient OHLC data')

        lookback = lookback or self.range_break_lookback
        revert_pct = revert_pct or self.range_break_reversion_pct

        if len(df) < lookback + 2:
            return self._debug_return('range_break_reversal', False, f'Need {lookback + 2} candles')

        recent = df.iloc[-(lookback + 1):-1]
        high = recent['high'].max()
        low = recent['low'].min()

        prev = float(df.iloc[-2]['close'])
        last = float(df.iloc[-1]['close'])

        broke_high = prev > high
        broke_low = prev < low

        if broke_high:
            if last < high and ((high - last) / high) * 100 >= revert_pct:
                return self._debug_return('range_break_reversal', True, 'Failed breakout above range')

        if broke_low:
            if last > low and ((last - low) / low) * 100 >= revert_pct:
                return self._debug_return('range_break_reversal', True, 'Failed breakdown below range')

        return self._debug_return('range_break_reversal', False, 'No range break reversal detected')

    # -------------------------------------------------------------
    # 4) US Market Proximity Detector
    # -------------------------------------------------------------
    def us_market_proximity(self) -> bool:
        now = datetime.now(pytz.utc).astimezone(IST)
        us_dt = now.replace(hour=self.us_session_start.hour,
                            minute=self.us_session_start.minute,
                            second=0, microsecond=0)

        if now > us_dt:
            return self._debug_return('us_market_proximity', False, 'US session already started')

        delta = (us_dt - now).total_seconds() / 60.0
        result = 1 <= delta <= self.us_proximity_minutes
        reason = f'US opens in {delta:.1f}m'
        return self._debug_return('us_market_proximity', result, reason)

    # -------------------------------------------------------------
    # NEW DETECTORS (requested)
    # -------------------------------------------------------------

    def atr_squeeze_breakout(self) -> bool:
        """
        Detects squeeze → breakout context suitable for straddle.
        Conditions:
          - ATR short significantly below ATR long
          - BBWidth uptick vs recent mean
          - Last candle body expands vs recent average
        """
        df = self.ohlc
        if df is None or len(df) < 30:
            return self._debug_return('atr_squeeze_breakout', False, 'Need >=30 candles')

        if self.latest_atr_short is None or self.latest_atr_long is None:
            return self._debug_return('atr_squeeze_breakout', False, 'ATR values missing')

        # tight squeeze
        if not (self.latest_atr_short < self.latest_atr_long * self.atr_squeeze_ratio):
            return self._debug_return('atr_squeeze_breakout', False, 'ATR short not below squeeze ratio')

        # BBWidth uptick
        bbw = self._bbwidth_series()
        if len(bbw) < 25:
            return self._debug_return('atr_squeeze_breakout', False, 'BBWidth history <25')

        recent = bbw.iloc[-10:]
        if recent.iloc[-1] <= recent.mean():
            return self._debug_return('atr_squeeze_breakout', False, 'BBWidth not expanding')

        # body expansion
        bodies = (df['close'] - df['open']).abs()
        if len(bodies) < 10:
            return self._debug_return('atr_squeeze_breakout', False, 'Need >=10 bodies')

        avg_body = bodies.iloc[-6:-1].mean()
        last_body = bodies.iloc[-1]

        result = last_body > avg_body * 1.2
        if result and avg_body:
            reason = f'Body expansion {last_body/avg_body:.2f}x'
        else:
            reason = 'Body expansion < 1.2x'
        return self._debug_return('atr_squeeze_breakout', result, reason)

    def micro_trend_exhaustion(self) -> bool:
        """
        3-bar fade / exhaustion pattern:
          - 3 directional candles with decreasing body size
        """
        df = self.ohlc
        if df is None or len(df) < 4:
            return self._debug_return('micro_trend_exhaustion', False, 'Need >=4 candles')

        last3 = df.iloc[-3:]
        bodies = (last3['close'] - last3['open']).abs()

        # decreasing body size
        if not (bodies.iloc[0] > bodies.iloc[1] > bodies.iloc[2]):
            return self._debug_return('micro_trend_exhaustion', False, 'Bodies not strictly decreasing')

        # directional candles
        c0 = last3.iloc[0]
        c1 = last3.iloc[1]
        c2 = last3.iloc[2]

        bull = (c0['close'] > c0['open']) and (c1['close'] > c1['open']) and (c2['close'] > c2['open'])
        bear = (c0['close'] < c0['open']) and (c1['close'] < c1['open']) and (c2['close'] < c2['open'])

        result = bull or bear
        reason = 'Uniform direction for 3 bars' if result else 'Mixed candle directions'
        return self._debug_return('micro_trend_exhaustion', result, reason)

    def push_fail_reversal(self) -> bool:
        """
        Push-fail: attempted breakout (small distance) that closes back inside + long wick.
        """
        df = self.ohlc
        if df is None or len(df) < 6:
            return self._debug_return('push_fail_reversal', False, 'Need >=6 candles')

        prev = df.iloc[-2]
        last = df.iloc[-1]

        lookN = 6
        highN = df['high'].iloc[-lookN:-1].max()
        lowN = df['low'].iloc[-lookN:-1].min()

        # fake breakout of high
        if prev['high'] > highN and ((prev['high'] - highN) / highN) * 100 < 0.10:
            # but closes inside
            if last['close'] < prev['high']:
                upper_wick = prev['high'] - max(prev['close'], prev['open'])
                if upper_wick > (prev['high'] - prev['low']) * 0.3:
                    return self._debug_return('push_fail_reversal', True, 'Failed breakout with long upper wick')

        # fake breakdown of low
        if prev['low'] < lowN and ((lowN - prev['low']) / lowN) * 100 < 0.10:
            if last['close'] > prev['low']:
                lower_wick = min(prev['close'], prev['open']) - prev['low']
                if lower_wick > (prev['high'] - prev['low']) * 0.3:
                    return self._debug_return('push_fail_reversal', True, 'Failed breakdown with long lower wick')

        return self._debug_return('push_fail_reversal', False, 'No push-fail reversal detected')

    def atr_pinch_bounce(self) -> bool:
        """
        Price deviates from short MA by > factor*ATR and then bounces back toward MA.
        """
        df = self.ohlc
        if df is None or len(df) < 10:
            return self._debug_return('atr_pinch_bounce', False, 'Need >=10 candles')

        if self.latest_atr_short is None:
            return self._debug_return('atr_pinch_bounce', False, 'ATR short missing')

        closes = df['close']
        if len(closes) < 6:
            return self._debug_return('atr_pinch_bounce', False, 'Need >=6 closes')

        ma5 = closes.iloc[-6:-1].mean()

        last = closes.iloc[-1]
        prev = closes.iloc[-2]
        dev_last = abs(last - ma5)

        # deviation > multiplier * ATR
        if dev_last <= self.latest_atr_short * self.atr_pinch_mult:
            return self._debug_return('atr_pinch_bounce', False, 'Deviation below ATR threshold')

        # bounce: previous was further from MA than last (returning toward MA)
        result = abs(prev - ma5) > abs(last - ma5)
        reason = 'Price moving back toward MA' if result else 'No bounce toward MA'
        return self._debug_return('atr_pinch_bounce', result, reason)

    def option_premium_compression(self) -> bool:
        """
        CE/PE premium compression detector (runner must push option prices into detector).
        Conditions:
          - both CE and PE dropped by threshold from their previous values
          - underlying not moved much
        """
        # require at least previous prices (runner must supply)
        if self.ce_price is None or self.pe_price is None:
            return self._debug_return('option_premium_compression', False, 'Current option prices missing')
        if self.ce_prev is None or self.pe_prev is None:
            return self._debug_return('option_premium_compression', False, 'Previous option prices missing')

        try:
            if self.ce_prev == 0 or self.pe_prev == 0:
                return self._debug_return('option_premium_compression', False, 'Previous option price zero')
        except Exception:
            return self._debug_return('option_premium_compression', False, 'Error validating previous prices')

        ce_drop = (self.ce_prev - self.ce_price) / self.ce_prev * 100
        pe_drop = (self.pe_prev - self.pe_price) / self.pe_prev * 100

        # both compressed
        if ce_drop < 1.5 or pe_drop < 1.5:
            return self._debug_return('option_premium_compression', False, 'Premium drop <1.5%')

        df = self.ohlc
        if df is None or len(df) < 4:
            return self._debug_return('option_premium_compression', False, 'Insufficient OHLC data')

        # underlying not moved
        last = float(df.iloc[-1]['close'])
        prev = float(df.iloc[-4:-1]['close'].mean())
        dev = abs(last - prev) / prev * 100

        result = dev < 0.25
        reason = f'Underlying dev {dev:.3f}%'
        return self._debug_return('option_premium_compression', result, reason)

    def mid_session_vol_reset(self) -> bool:
        """
        Mid-session volatility reset (17:00–19:30 IST): ATR contraction + VWAP close + BBWidth rising
        """
        df = self.ohlc
        if df is None or len(df) < 30:
            return self._debug_return('mid_session_vol_reset', False, 'Need >=30 candles')

        now = datetime.now(pytz.utc).astimezone(IST).time()

        # between 17:00–19:30
        if not (now >= dt_time(17, 0) and now <= dt_time(19, 30)):
            return self._debug_return('mid_session_vol_reset', False, 'Outside 17:00-19:30 IST')

        if self.latest_atr_short is None or self.latest_atr_long is None:
            return self._debug_return('mid_session_vol_reset', False, 'ATR values missing')

        # ATR contraction (Relaxed from 0.65 to 0.75)
        if not (self.latest_atr_short < self.latest_atr_long * 0.75):
            return self._debug_return('mid_session_vol_reset', False, 'ATR short not contracted enough')

        # anchor / vwap deviation
        anchor = self._compute_price_anchor()
        if anchor is None:
            return self._debug_return('mid_session_vol_reset', False, 'No anchor computed')

        last = float(df.iloc[-1]['close'])
        dev = abs((last - anchor) / anchor) * 100
        if dev > 0.12:
            return self._debug_return('mid_session_vol_reset', False, f'Price dev {dev:.3f}% > 0.12%')

        # BBWidth is rising from a low percentile
        bbw = self._bbwidth_series()
        if len(bbw) < 40:
            return self._debug_return('mid_session_vol_reset', False, 'BBWidth history <40')

        last20 = bbw.iloc[-20:]
        try:
            result = last20.iloc[-1] > last20.quantile(0.3)
            reason = 'BBWidth rising from low percentile' if result else 'BBWidth not rising'
            return self._debug_return('mid_session_vol_reset', result, reason)
        except Exception:
            return self._debug_return('mid_session_vol_reset', False, 'Error evaluating BBWidth percentile')

    # -------------------------------------------------------------
    # Priority evaluation wrapper used by runner
    # -------------------------------------------------------------
    def evaluate_priority_signals(self) -> Optional[str]:
        """
        Time-phased priority signals for 15:00 - 23:30 IST session.
        Includes ADX Trend Filter and RSI Divergence.
        """
        now = datetime.now(IST).time()
        
        # Global Trend Filter (Skip if Strong Trend, except for Reversals)
        is_ranging = self.trend_filter()

        # High Priority: RSI Divergence (Works in all phases)
        if self.rsi_divergence():
             return self._debug_return('evaluate_priority_signals', 'RSI_DIV', 'RSI Divergence detected')

        # Phase 1: 15:00 - 18:00 (Range/Compression)
        if dt_time(15, 0) <= now < dt_time(18, 0):
            if is_ranging:
                if self.compression_detected():
                    return self._debug_return('evaluate_priority_signals', 'COMPRESSION', 'Phase 1: compression')
                if self.anchor_reversion():
                    return self._debug_return('evaluate_priority_signals', 'ANCHOR_REV', 'Phase 1: anchor_reversion')
            
            if self.range_break_reversal():
                return self._debug_return('evaluate_priority_signals', 'RANGE_REV', 'Phase 1: range_break_reversal')

        # Phase 2: 18:00 - 20:00 (Pre-US Volatility Reset)
        elif dt_time(18, 0) <= now < dt_time(20, 0):
            if is_ranging:
                if self.mid_session_vol_reset():
                    return self._debug_return('evaluate_priority_signals', 'MSVR', 'Phase 2: mid_session_vol_reset')
                if self.compression_detected():
                    return self._debug_return('evaluate_priority_signals', 'COMPRESSION', 'Phase 2: compression')
                # Expanded: Allow Anchor Reversion & Squeeze Breakout
                if self.anchor_reversion():
                    return self._debug_return('evaluate_priority_signals', 'ANCHOR_REV', 'Phase 2: anchor_reversion')
                if self.atr_squeeze_breakout():
                    return self._debug_return('evaluate_priority_signals', 'SQUEEZE_BREAK', 'Phase 2: atr_squeeze_breakout')

        # Phase 3: 20:00 - 20:30 (US Open - High Risk)
        elif dt_time(20, 0) <= now < dt_time(20, 30):
            # Strict filtering: only high-quality reversals
            if self.push_fail_reversal():
                return self._debug_return('evaluate_priority_signals', 'PUSH_FAIL', 'Phase 3: push_fail_reversal')

        # Phase 4: 20:30 - 23:30 (Post-Open Trends/Fades)
        elif dt_time(20, 30) <= now <= dt_time(23, 30):
            if self.micro_trend_exhaustion():
                return self._debug_return('evaluate_priority_signals', '3BAR_FADE', 'Phase 4: micro_trend_exhaustion')
            if self.push_fail_reversal():
                return self._debug_return('evaluate_priority_signals', 'PUSH_FAIL', 'Phase 4: push_fail_reversal')
            if self.atr_pinch_bounce():
                return self._debug_return('evaluate_priority_signals', 'ATR_PINCH', 'Phase 4: atr_pinch_bounce')
            # Expanded: Allow Squeeze Breakout & Compression
            if self.atr_squeeze_breakout():
                return self._debug_return('evaluate_priority_signals', 'SQUEEZE_BREAK', 'Phase 4: atr_squeeze_breakout')
            if self.compression_detected():
                return self._debug_return('evaluate_priority_signals', 'COMPRESSION', 'Phase 4: compression')

        return self._debug_return('evaluate_priority_signals', None, 'No signal matched for current phase')
