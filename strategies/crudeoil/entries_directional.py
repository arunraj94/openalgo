# DirectionalEntries.py
# Directional entry signals for Crude Oil MCX (Complementary to Short Straddle)
# Identifies strong trends and provides BULLISH/BEARISH direction for spreads

from datetime import datetime
import pytz
import pandas as pd
from typing import Optional, Literal

IST = pytz.timezone("Asia/Kolkata")

DirectionSignal = Literal["BULLISH", "BEARISH", None]


class DirectionalEntries:
    """
    Provides directional signals (BULLISH/BEARISH) during trending markets.
    Designed to complement the short straddle strategy by profiting during trends.
    
    Usage:
      - Call update_ohlc(df) with DataFrame containing ['timestamp','open','high','low','close']
      - Call evaluate_trend_direction() to get signal
    """

    def __init__(self):
        # EMA settings
        self.ema_fast = 9
        self.ema_slow = 21
        
        # ADX settings (opposite filter from straddle)
        self.adx_period = 14
        self.adx_threshold = 25
        
        # Supertrend settings
        self.supertrend_period = 10
        self.supertrend_multiplier = 3
        
        # Price action lookback
        self.trend_lookback = 5
        
        # OHLC storage
        self.ohlc: Optional[pd.DataFrame] = None

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
    def _compute_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA for a given series and period"""
        return series.ewm(span=period, adjust=False).mean()

    def _compute_adx(self, period: int = 14) -> Optional[float]:
        """Calculate ADX (Average Directional Index)"""
        df = self.ohlc
        if df is None or len(df) < period * 2:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='outer').max(axis=1)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        
        if len(adx) == 0:
            return None
        return float(adx.iloc[-1])

    def _compute_supertrend(self) -> Optional[tuple]:
        """
        Calculate Supertrend indicator.
        Returns: (supertrend_value, trend_direction) where trend_direction is 1 (bullish) or -1 (bearish)
        """
        df = self.ohlc
        if df is None or len(df) < self.supertrend_period + 5:
            return None

        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.supertrend_period).mean()
        
        # Basic Bands
        hl_avg = (high + low) / 2
        upper_band = hl_avg + (self.supertrend_multiplier * atr)
        lower_band = hl_avg - (self.supertrend_multiplier * atr)
        
        # Initialize supertrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(len(df)):
            if i == 0:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
                continue
                
            # Update bands
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]
            prev_supertrend = supertrend.iloc[i-1]
            prev_direction = direction.iloc[i-1]
            
            # Bullish trend
            if close.iloc[i] > prev_supertrend:
                direction.iloc[i] = 1
                supertrend.iloc[i] = max(curr_lower, prev_supertrend if prev_direction == 1 else curr_lower)
            # Bearish trend
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = min(curr_upper, prev_supertrend if prev_direction == -1 else curr_upper)
        
        return (float(supertrend.iloc[-1]), int(direction.iloc[-1]))

    def _price_action_trend(self) -> Optional[str]:
        """
        Determine trend based on Higher Highs/Lows or Lower Highs/Lows
        Returns: 'BULLISH', 'BEARISH', or None
        """
        df = self.ohlc
        if df is None or len(df) < self.trend_lookback + 1:
            return None
        
        recent = df.iloc[-self.trend_lookback:]
        
        # Check for Higher Highs and Higher Lows (Bullish)
        highs = recent['high']
        lows = recent['low']
        
        higher_highs = all(highs.iloc[i] >= highs.iloc[i-1] for i in range(1, len(highs)))
        higher_lows = all(lows.iloc[i] >= lows.iloc[i-1] for i in range(1, len(lows)))
        
        if higher_highs and higher_lows:
            return 'BULLISH'
        
        # Check for Lower Highs and Lower Lows (Bearish)
        lower_highs = all(highs.iloc[i] <= highs.iloc[i-1] for i in range(1, len(highs)))
        lower_lows = all(lows.iloc[i] <= lows.iloc[i-1] for i in range(1, len(lows)))
        
        if lower_highs and lower_lows:
            return 'BEARISH'
        
        return None

    # --------------------------------------------------------
    # Main Evaluation Method
    # --------------------------------------------------------
    def evaluate_trend_direction(self) -> DirectionSignal:
        """
        Main method to evaluate trend direction.
        Returns: "BULLISH", "BEARISH", or None
        """
        df = self.ohlc
        if df is None or len(df) < max(self.ema_slow, self.adx_period * 2):
            self._debug_log("Insufficient data for directional analysis")
            return None

        # 1. Check ADX - Must be trending
        adx = self._compute_adx(self.adx_period)
        if adx is None or adx < self.adx_threshold:
            self._debug_log(f"ADX {adx} below threshold {self.adx_threshold} - No trend")
            return None

        # 2. EMA Crossover
        close = df['close']
        ema_fast = self._compute_ema(close, self.ema_fast)
        ema_slow = self._compute_ema(close, self.ema_slow)
        
        ema_bullish = ema_fast.iloc[-1] > ema_slow.iloc[-1]
        ema_bearish = ema_fast.iloc[-1] < ema_slow.iloc[-1]

        # 3. Supertrend
        supertrend_result = self._compute_supertrend()
        if supertrend_result is None:
            self._debug_log("Supertrend calculation failed")
            return None
        
        _, st_direction = supertrend_result
        st_bullish = st_direction == 1
        st_bearish = st_direction == -1

        # 4. Price Action
        pa_trend = self._price_action_trend()
        pa_bullish = pa_trend == 'BULLISH'
        pa_bearish = pa_trend == 'BEARISH'

        # Confluence Logic: At least 2 out of 3 must agree
        bullish_votes = sum([ema_bullish, st_bullish, pa_bullish])
        bearish_votes = sum([ema_bearish, st_bearish, pa_bearish])

        if bullish_votes >= 2:
            self._debug_log(f"BULLISH: ADX={adx:.1f}, EMA={ema_bullish}, ST={st_bullish}, PA={pa_bullish}")
            return "BULLISH"
        
        if bearish_votes >= 2:
            self._debug_log(f"BEARISH: ADX={adx:.1f}, EMA={ema_bearish}, ST={st_bearish}, PA={pa_bearish}")
            return "BEARISH"

        self._debug_log(f"No consensus: ADX={adx:.1f}")
        return None

    def _debug_log(self, message: str):
        """Debug logging"""
        try:
            ts = datetime.now(IST).strftime("%H:%M:%S")
            print(f"{ts} [DirectionalEntries] {message}")
        except Exception:
            pass
