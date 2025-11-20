# position_sizing.py
# Hybrid position sizing module for Unified Short Straddle Bot
# Option C (Hybrid): min( ATR-based sizing, Margin-based sizing )

from math import floor
from typing import Optional
from dataclasses import dataclass
import sqlite3
from config import default_config as cfg

@dataclass
class SizingParams:
    account_capital: float = 200000.0
    margin_per_leg: float = 20000.0
    legs_per_straddle: int = 2
    max_alloc_per_trade_pct: float = 0.08
    max_total_alloc_pct: float = 0.30
    k_atr_scaler: float = 500.0
    used_margin: float = 0.0  # margin already used by other trades
    correlation_reduction_factor: float = 0.5

def get_portfolio_bias(db_path):
    """
    Determines the current portfolio directional bias based on open positions.
    Returns: 'LONG', 'SHORT', or 'NEUTRAL'
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Count open BUY and SELL signals
        # Assuming 'signal' column contains 'BUY'/'SELL' or similar
        # And we only care about OPEN trades
        cursor.execute("SELECT signal FROM trades WHERE status = 'OPEN'")
        rows = cursor.fetchall()
        conn.close()
        
        longs = 0
        shorts = 0
        for r in rows:
            sig = r[0].upper() if r[0] else ""
            if 'BUY' in sig or 'LONG' in sig or 'BULL' in sig:
                longs += 1
            elif 'SELL' in sig or 'SHORT' in sig or 'BEAR' in sig:
                shorts += 1
                
        if longs > shorts:
            return 'LONG'
        elif shorts > longs:
            return 'SHORT'
        else:
            return 'NEUTRAL'
    except Exception as e:
        print(f"Error checking portfolio bias: {e}")
        return 'NEUTRAL'

def compute_lots_hybrid(atr_short: Optional[float], params: SizingParams, signal_direction: Optional[str] = None) -> int:
    # 1. ATR Sizing
    if atr_short is None or atr_short <= 0:
        lot_by_atr = 9999  # No limit if ATR invalid
    else:
        # e.g. K / ATR
        lot_by_atr = int(floor(params.k_atr_scaler / atr_short))

    # 2. Margin Sizing
    total_capital = params.account_capital
    max_trade_alloc = total_capital * params.max_alloc_per_trade_pct
    
    # Check global limit
    available_cap_global = (total_capital * params.max_total_alloc_pct) - params.used_margin
    if available_cap_global < 0:
        available_cap_global = 0
        
    trade_margin_budget = min(max_trade_alloc, available_cap_global)
    
    margin_per_straddle = params.margin_per_leg * params.legs_per_straddle
    
    if margin_per_straddle <= 0:
         lot_by_margin = 0
    else:
         lot_by_margin = int(floor(trade_margin_budget // margin_per_straddle))

    final_lots = min(lot_by_atr, lot_by_margin)
    
    # 3. Correlation Sizing
    # If we are adding to an existing bias, reduce size
    if signal_direction:
        bias = get_portfolio_bias(cfg.DB_PATH)
        is_correlated = False
        
        if bias == 'LONG' and signal_direction in ['BUY', 'LONG', 'BULLISH']:
            is_correlated = True
        elif bias == 'SHORT' and signal_direction in ['SELL', 'SHORT', 'BEARISH']:
            is_correlated = True
            
        if is_correlated:
            print(f"Portfolio Correlation Detected ({bias} + {signal_direction}). Reducing size.")
            final_lots = int(final_lots * params.correlation_reduction_factor)

    return max(0, int(final_lots))


# Convenience wrapper to accept config-like dict or object
def compute_lots_from_config(atr_short: Optional[float], cfg, legs_count: Optional[int] = None, signal_direction: Optional[str] = None) -> int:
    """
    Helper that builds SizingParams from a config-like object (e.g., config.default_config)
    Expected attributes on cfg:
      - ACCOUNT_CAPITAL, MARGIN_PER_LEG, LEGS_PER_STRADDLE
      - MAX_ALLOC_PER_TRADE_PCT, MAX_TOTAL_ALLOC_PCT, K_ATR_SCALER
      - used_margin (optional)
      - CORRELATION_REDUCTION_FACTOR

    Returns integer lots.
    """
    used_margin = getattr(cfg, 'used_margin', 0.0)
    
    # Allow override of legs count (e.g. 1 for Directional, 2 for Straddle)
    legs = legs_count if legs_count is not None else getattr(cfg, 'LEGS_PER_STRADDLE', 2)

    params = SizingParams(
        account_capital=getattr(cfg, 'ACCOUNT_CAPITAL', 8000000.0),
        margin_per_leg=getattr(cfg, 'MARGIN_PER_LEG', 100000.0),
        legs_per_straddle=legs,
        max_alloc_per_trade_pct=getattr(cfg, 'MAX_ALLOC_PER_TRADE_PCT', 0.2),
        max_total_alloc_pct=getattr(cfg, 'MAX_TOTAL_ALLOC_PCT', 0.8),
        k_atr_scaler=getattr(cfg, 'K_ATR_SCALER', 500.0),
        used_margin=used_margin,
        correlation_reduction_factor=getattr(cfg, 'CORRELATION_REDUCTION_FACTOR', 0.5)
    )
    return compute_lots_hybrid(atr_short, params, signal_direction)


if __name__ == '__main__':
    # quick smoke test
    p = SizingParams(account_capital=1000000.0, margin_per_leg=30000.0, legs_per_straddle=2, k_atr_scaler=500.0)
    print('ATR=2 -> lots', compute_lots_hybrid(2.0, p))
    print('ATR=5 -> lots', compute_lots_hybrid(50.0, p))
    print('ATR=None -> lots', compute_lots_hybrid(None, p))
