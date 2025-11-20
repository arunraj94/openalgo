import pandas as pd
import numpy as np
from config import default_config as cfg

class MeanReversionEntries:
    """
    Detects Mean Reversion opportunities based on:
    1. RSI Extremes (>80 or <20)
    2. Bollinger Band Blasts (Price > Upper or Price < Lower)
    3. Reversal Candles (Shooting Star / Hammer)
    """
    def __init__(self, config=cfg):
        self.cfg = config
        self.ohlc_df = None
        self.rsi_series = None
        self.upper_bb = None
        self.lower_bb = None

    def update_ohlc(self, ohlc_df):
        self.ohlc_df = ohlc_df
        if len(self.ohlc_df) > self.cfg.RSI_PERIOD:
            self._compute_indicators()

    def _compute_indicators(self):
        close = self.ohlc_df['close']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.cfg.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.cfg.RSI_PERIOD).mean()
        rs = gain / loss
        self.rsi_series = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma = close.rolling(window=self.cfg.BB_LEN).mean()
        std = close.rolling(window=self.cfg.BB_LEN).std()
        self.upper_bb = sma + (std * self.cfg.MEANREV_BB_DEV)
        self.lower_bb = sma - (std * self.cfg.MEANREV_BB_DEV)

    def _is_reversal_candle(self, direction):
        """
        Checks for reversal candle patterns.
        Bearish Reversal (Shooting Star): Long upper wick, small body at bottom.
        Bullish Reversal (Hammer): Long lower wick, small body at top.
        """
        if len(self.ohlc_df) < 1:
            return False
            
        last_candle = self.ohlc_df.iloc[-1]
        open_p = last_candle['open']
        close_p = last_candle['close']
        high_p = last_candle['high']
        low_p = last_candle['low']
        
        body = abs(close_p - open_p)
        total_range = high_p - low_p
        
        if total_range == 0:
            return False
            
        if direction == "BEARISH":
            # Shooting Star: Upper wick > 2x body, Close near Low
            upper_wick = high_p - max(open_p, close_p)
            return upper_wick > (2 * body) and (close_p < (low_p + 0.3 * total_range))
            
        elif direction == "BULLISH":
            # Hammer: Lower wick > 2x body, Close near High
            lower_wick = min(open_p, close_p) - low_p
            return lower_wick > (2 * body) and (close_p > (high_p - 0.3 * total_range))
            
        return False

    def evaluate_signal(self):
        """
        Returns 'SELL' (Short) or 'BUY' (Long) or None
        """
        if self.rsi_series is None or len(self.rsi_series) < 1:
            return None
            
        current_rsi = self.rsi_series.iloc[-1]
        current_close = self.ohlc_df['close'].iloc[-1]
        current_upper = self.upper_bb.iloc[-1]
        current_lower = self.lower_bb.iloc[-1]
        
        # Short Signal (Fade the Pump)
        if current_rsi > self.cfg.MEANREV_RSI_HIGH:
            if current_close > current_upper:
                if self._is_reversal_candle("BEARISH"):
                    return "SELL"
                    
        # Long Signal (Fade the Dump)
        if current_rsi < self.cfg.MEANREV_RSI_LOW:
            if current_close < current_lower:
                if self._is_reversal_candle("BULLISH"):
                    return "BUY"
                    
        return None
