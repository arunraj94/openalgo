# VolatilityExpansion.py
# Volatility Expansion entry signals for Crude Oil MCX (Option BUYING strategy)
# Detects extreme compression and buys ATM straddles expecting expansion

from datetime import datetime, time as dt_time
import pytz
import pandas as pd
from typing import Optional, Literal

IST = pytz.timezone("Asia/Kolkata")

ExpansionSignal = Literal["EXTREME_COMPRESSION", "PRE_EVENT", "CONSOLIDATION_BREAKOUT", None]


class VolatilityExpansion:
    """
    Volatility Expansion strategy - BUYS options during extreme compression.
    
    This is the OPPOSITE of the short straddle strategy:
    - Short Straddle: SELLS during normal compression
    - This Strategy: BUYS during EXTREME compression expecting expansion
    
    Usage:
      - Call update_ohlc(df) with DataFrame containing ['timestamp','open','high','low','close']
      - Call evaluate_expansion_signals() to get entry signal
      - This strategy should NOT run simultaneously with short straddle
    """

    def __init__(self, config=None):
        """Initialize with optional config object"""
        # Apply config if provided
        if config:
            self.extreme_bbwidth_percentile = config.VOLEXP_EXTREME_BBWIDTH_PERCENTILE
            self.extreme_atr_ratio = config.VOLEXP_EXTREME_ATR_RATIO
            self.extreme_vwap_dev = config.VOLEXP_EXTREME_VWAP_DEV
            self.consolidation_candles = config.VOLEXP_CONSOLIDATION_CANDLES
            self.body_threshold = config.VOLEXP_BODY_THRESHOLD
            self.n_body_avg = config.VOLEXP_N_BODY_AVG
            self.pre_event_minutes_before = config.VOLEXP_PRE_EVENT_MINUTES_BEFORE
            self.pre_event_minutes_after = config.VOLEXP_PRE_EVENT_MINUTES_AFTER
            self.us_session_start = config.US_SESSION_START
            self.bb_len = config.BB_LEN
            self.bbwidth_lookback = config.BBWIDTH_LOOKBACK
        else:
            # Default values (same as before)
            self.extreme_bbwidth_percentile = 10
            self.extreme_atr_ratio = 0.55
            self.extreme_vwap_dev = 0.15
            self.consolidation_candles = 8
            self.body_threshold = 0.30
            self.n_body_avg = 10
            self.pre_event_minutes_before = 30
            self.pre_event_minutes_after = 45
            self.us_session_start = dt_time(20, 0)
            self.bb_len = 20
            self.bbwidth_lookback = 60
        
        # OHLC storage
        self.ohlc: Optional[pd.DataFrame] = None
        
        # ATR values (pushed by runner)
        self.latest_atr_short = None
        self.latest_atr_long = None

    def update_ohlc(self, ohlc: pd.DataFrame):
        """Update OHLC data. Expects: ['timestamp','open','high','low','close']"""
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

    # --------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------
    def _compute_price_anchor(self) -> Optional[float]:
        """Compute VWAP/TWAP anchor price"""
        df = self.ohlc
        if df is None or len(df) == 0:
            return None
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        if len(tp) >= self.bb_len:
            return float(tp.iloc[-self.bb_len:].mean())
        return float(tp.mean())

    def _bbwidth_series(self) -> pd.Series:
        """Calculate Bollinger Band Width series"""
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

    def _debug_log(self, message: str, signal_type: str = "VolatilityExpansion"):
        """Debug logging"""
        try:
            ts = datetime.now(IST).strftime("%H:%M:%S")
            print(f"{ts} [{signal_type}] {message}")
        except Exception:
            pass

    # --------------------------------------------------------
    # Detection Methods
    # --------------------------------------------------------
    
    def extreme_volatility_compression(self) -> bool:
        """
        Detects EXTREME volatility compression - tighter thresholds than selling strategy.
        
        Conditions:
          - BBWidth at bottom 10th percentile (vs 25-40% for selling)
          - ATR short < ATR long * 0.55 (vs 0.7-0.8 for selling)
          - Price within ±0.15% of VWAP (vs ±0.25% for selling)
        
        Returns: True if extreme compression detected
        """
        df = self.ohlc
        if df is None or len(df) < self.bbwidth_lookback + 5:
            self._debug_log("Insufficient data for extreme compression check")
            return False

        # Check ATR compression
        if self.latest_atr_short is None or self.latest_atr_long is None:
            self._debug_log("ATR values missing")
            return False

        if not (self.latest_atr_short < self.latest_atr_long * self.extreme_atr_ratio):
            self._debug_log(
                f"ATR not compressed enough: {self.latest_atr_short:.2f} >= {self.latest_atr_long * self.extreme_atr_ratio:.2f}"
            )
            return False

        # Check BBWidth at extreme low
        bbw = self._bbwidth_series()
        if len(bbw) < self.bbwidth_lookback:
            self._debug_log("Insufficient BBWidth history")
            return False

        recent_bb = bbw.iloc[-self.bbwidth_lookback:]
        threshold = recent_bb.quantile(self.extreme_bbwidth_percentile / 100.0)
        
        if bbw.iloc[-1] > threshold:
            self._debug_log(
                f"BBWidth not extreme: {bbw.iloc[-1]:.6f} > {threshold:.6f} ({self.extreme_bbwidth_percentile}%)"
            )
            return False

        # Check price near anchor
        anchor = self._compute_price_anchor()
        if anchor is None or anchor == 0:
            self._debug_log("No anchor price")
            return False

        price = float(df['close'].iloc[-1])
        dev = abs((price - anchor) / anchor) * 100.0
        
        if dev > self.extreme_vwap_dev:
            self._debug_log(f"Price deviation too high: {dev:.2f}% > {self.extreme_vwap_dev}%")
            return False

        self._debug_log(
            f"✅ EXTREME COMPRESSION: ATR={self.latest_atr_short:.2f}/{self.latest_atr_long:.2f}, "
            f"BBWidth={bbw.iloc[-1]:.6f}, Dev={dev:.2f}%"
        )
        return True

    def pre_event_setup(self) -> bool:
        """
        Detects compression before known volatile periods (US open).
        
        Timing:
          - 30-45 minutes before US session (19:15-19:30 IST)
          - Combined with compression signals
        
        Returns: True if in pre-event window with compression
        """
        now = datetime.now(IST).time()
        
        # Calculate pre-event window
        us_open_minutes = self.us_session_start.hour * 60 + self.us_session_start.minute
        pre_start_minutes = us_open_minutes - self.pre_event_minutes_after
        pre_end_minutes = us_open_minutes - self.pre_event_minutes_before
        
        pre_start = dt_time(pre_start_minutes // 60, pre_start_minutes % 60)
        pre_end = dt_time(pre_end_minutes // 60, pre_end_minutes % 60)
        
        if not (pre_start <= now <= pre_end):
            return False

        # Must also have compression
        if not self.extreme_volatility_compression():
            self._debug_log("Pre-event window but no compression")
            return False

        self._debug_log(f"✅ PRE-EVENT SETUP: {self.pre_event_minutes_before}-{self.pre_event_minutes_after}min before US open")
        return True

    def consolidation_breakout_imminent(self) -> bool:
        """
        Detects tight consolidation that often precedes breakouts.
        
        Conditions:
          - 8+ candles with small bodies (< 30% of average)
          - BBWidth contracting (current < 5-candle average)
          - Price range tightening
        
        Returns: True if consolidation detected
        """
        df = self.ohlc
        if df is None or len(df) < self.consolidation_candles + self.n_body_avg:
            return False

        # Check small bodies for N consecutive candles
        bodies = (df['close'] - df['open']).abs()
        avg_body = bodies.iloc[-(self.n_body_avg + self.consolidation_candles):-self.consolidation_candles].mean()
        
        if avg_body <= 0:
            return False

        last_bodies = bodies.iloc[-self.consolidation_candles:]
        small_bodies = (last_bodies < avg_body * self.body_threshold).sum()
        
        if small_bodies < self.consolidation_candles * 0.75:  # At least 75% must be small
            self._debug_log(f"Not enough small bodies: {small_bodies}/{self.consolidation_candles}")
            return False

        # Check BBWidth contraction
        bbw = self._bbwidth_series()
        if len(bbw) < self.consolidation_candles + 5:
            return False

        recent_bbw = bbw.iloc[-5:]
        if bbw.iloc[-1] >= recent_bbw.mean():
            self._debug_log("BBWidth not contracting")
            return False

        # Check price range tightening
        last_n = df.iloc[-self.consolidation_candles:]
        range_size = last_n['high'].max() - last_n['low'].min()
        avg_range = (df['high'] - df['low']).iloc[-20:-self.consolidation_candles].mean()
        
        if range_size >= avg_range * 0.6:  # Range must be < 60% of average
            self._debug_log(f"Range not tight enough: {range_size:.2f} vs {avg_range:.2f}")
            return False

        self._debug_log(
            f"✅ CONSOLIDATION: {small_bodies}/{self.consolidation_candles} small bodies, "
            f"Range={range_size:.2f} ({range_size/avg_range*100:.1f}% of avg)"
        )
        return True

    # --------------------------------------------------------
    # Main Evaluation Method
    # --------------------------------------------------------
    
    def evaluate_expansion_signals(self) -> ExpansionSignal:
        """
        Main method to evaluate volatility expansion signals.
        
        Priority:
          1. Pre-Event Setup (highest probability)
          2. Extreme Compression
          3. Consolidation Breakout
        
        Returns: Signal name or None
        """
        df = self.ohlc
        if df is None or len(df) < 30:
            self._debug_log("Insufficient OHLC data")
            return None

        # Priority 1: Pre-event setup (compression before known volatile period)
        if self.pre_event_setup():
            return "PRE_EVENT"

        # Priority 2: Extreme compression (tighter than selling thresholds)
        if self.extreme_volatility_compression():
            return "EXTREME_COMPRESSION"

        # Priority 3: Consolidation breakout imminent
        if self.consolidation_breakout_imminent():
            return "CONSOLIDATION_BREAKOUT"

        return None


# Example usage and testing
if __name__ == "__main__":
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range(start='2025-01-01 15:00', end='2025-01-01 20:00', freq='1min')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': 6000 + pd.Series(range(len(dates))) * 0.1,
        'high': 6000 + pd.Series(range(len(dates))) * 0.1 + 5,
        'low': 6000 + pd.Series(range(len(dates))) * 0.1 - 5,
        'close': 6000 + pd.Series(range(len(dates))) * 0.1
    })
    
    detector = VolatilityExpansion()
    detector.update_ohlc(sample_data)
    detector.latest_atr_short = 10.0
    detector.latest_atr_long = 20.0
    
    signal = detector.evaluate_expansion_signals()
    print(f"Signal: {signal}")
