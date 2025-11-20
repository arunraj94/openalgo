# OpeningRangeBreakout.py
# Opening Range Breakout (ORB) entry signals for Crude Oil MCX
# Time-based strategy that captures opening momentum

from datetime import datetime, time as dt_time
import pytz
import pandas as pd
from typing import Optional, Literal

IST = pytz.timezone("Asia/Kolkata")

ORBSignal = Literal["BULLISH_BREAKOUT", "BEARISH_BREAKOUT", None]


class OpeningRangeBreakout:
    """
    Opening Range Breakout (ORB) strategy for intraday trading.
    Captures momentum from the opening session range.
    
    Usage:
      - Call update_ohlc(df) with DataFrame containing ['timestamp','open','high','low','close']
      - Call evaluate_orb_signal() to get breakout signal
      - Call reset_session() at the end of trading day
    """

    def __init__(self):
        # Configuration (will be overridden by config)
        self.orb_sessions = [(dt_time(15, 0), dt_time(15, 30))]
        self.range_duration_minutes = 15
        self.breakout_buffer_pct = 0.1
        self.min_range_size = 10.0
        self.max_range_size = 100.0
        self.body_strength_pct = 0.6
        
        # State tracking
        self.ohlc: Optional[pd.DataFrame] = None
        self.range_high: Optional[float] = None
        self.range_low: Optional[float] = None
        self.range_open: Optional[float] = None
        self.range_defined = False
        self.range_valid = False
        self.traded_today = False
        self.current_session_start: Optional[datetime] = None

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

    def reset_session(self):
        """Reset state for a new trading session"""
        self.range_high = None
        self.range_low = None
        self.range_open = None
        self.range_defined = False
        self.range_valid = False
        self.traded_today = False
        self.current_session_start = None
        self._debug_log("Session reset")

    # --------------------------------------------------------
    # Range Definition
    # --------------------------------------------------------
    def _define_range(self):
        """Define the opening range based on configured duration"""
        df = self.ohlc
        if df is None or len(df) == 0:
            return

        now = datetime.now(IST)
        current_time = now.time()

        # Check if we're in a range definition window
        for session_start, session_end in self.orb_sessions:
            if session_start <= current_time <= session_end:
                # Check if this is a new session
                session_datetime = now.replace(
                    hour=session_start.hour,
                    minute=session_start.minute,
                    second=0,
                    microsecond=0
                )
                
                if self.current_session_start != session_datetime:
                    # New session detected
                    self._debug_log(f"New ORB session started at {session_start}")
                    self.current_session_start = session_datetime
                    self.range_defined = False
                    self.range_valid = False
                    self.traded_today = False

                # Calculate range end time
                range_end = session_datetime.replace(
                    minute=session_start.minute + self.range_duration_minutes
                )

                if now >= range_end and not self.range_defined:
                    # Range period is over, define the range
                    range_candles = df[
                        (df['timestamp'] >= session_datetime) &
                        (df['timestamp'] < range_end)
                    ]

                    if len(range_candles) == 0:
                        self._debug_log("No candles in range period")
                        return

                    self.range_high = float(range_candles['high'].max())
                    self.range_low = float(range_candles['low'].min())
                    self.range_open = float(range_candles.iloc[0]['open'])
                    self.range_defined = True

                    # Validate range
                    range_size = self.range_high - self.range_low
                    
                    if range_size < self.min_range_size:
                        self._debug_log(f"Range too small: {range_size:.2f} < {self.min_range_size}")
                        self.range_valid = False
                        return
                    
                    if range_size > self.max_range_size:
                        self._debug_log(f"Range too large: {range_size:.2f} > {self.max_range_size}")
                        self.range_valid = False
                        return

                    self.range_valid = True
                    self._debug_log(
                        f"Range defined: High={self.range_high:.2f}, "
                        f"Low={self.range_low:.2f}, Size={range_size:.2f}"
                    )
                break

    # --------------------------------------------------------
    # Breakout Detection
    # --------------------------------------------------------
    def _detect_breakout(self) -> ORBSignal:
        """Detect if price has broken out of the opening range"""
        if not self.range_defined or not self.range_valid:
            return None

        if self.traded_today:
            return None

        df = self.ohlc
        if df is None or len(df) == 0:
            return None

        # Get latest candle
        latest = df.iloc[-1]
        close = float(latest['close'])
        high = float(latest['high'])
        low = float(latest['low'])
        open_price = float(latest['open'])

        # Calculate breakout thresholds
        range_size = self.range_high - self.range_low
        buffer = range_size * (self.breakout_buffer_pct / 100.0)

        bullish_threshold = self.range_high + buffer
        bearish_threshold = self.range_low - buffer

        # Bullish Breakout
        if close > bullish_threshold:
            # Confirm with body strength
            candle_range = high - low
            if candle_range == 0:
                return None
            
            body = abs(close - open_price)
            body_strength = body / candle_range

            if body_strength >= self.body_strength_pct and close > open_price:
                self._debug_log(
                    f"BULLISH BREAKOUT: Close={close:.2f} > Threshold={bullish_threshold:.2f}, "
                    f"Body Strength={body_strength:.2%}"
                )
                self.traded_today = True
                return "BULLISH_BREAKOUT"

        # Bearish Breakout
        elif close < bearish_threshold:
            # Confirm with body strength
            candle_range = high - low
            if candle_range == 0:
                return None
            
            body = abs(close - open_price)
            body_strength = body / candle_range

            if body_strength >= self.body_strength_pct and close < open_price:
                self._debug_log(
                    f"BEARISH BREAKOUT: Close={close:.2f} < Threshold={bearish_threshold:.2f}, "
                    f"Body Strength={body_strength:.2%}"
                )
                self.traded_today = True
                return "BEARISH_BREAKOUT"

        return None

    # --------------------------------------------------------
    # Main Evaluation Method
    # --------------------------------------------------------
    def evaluate_orb_signal(self) -> ORBSignal:
        """
        Main method to evaluate ORB signals.
        Returns: "BULLISH_BREAKOUT", "BEARISH_BREAKOUT", or None
        """
        # First, ensure range is defined if we're in range period
        self._define_range()

        # Then check for breakout
        return self._detect_breakout()

    def _debug_log(self, message: str):
        """Debug logging"""
        try:
            ts = datetime.now(IST).strftime("%H:%M:%S")
            print(f"{ts} [ORB] {message}")
        except Exception:
            pass
