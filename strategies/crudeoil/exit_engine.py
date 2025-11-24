# exit_engine.py
# Exit engine for Unified Short Straddle Bot (modular version)
# Handles: SL-hit logic, survivor breakeven, trailing SL, early-exit on no movement / no decay.

from datetime import datetime, time as dt_time
import pytz
import pandas as pd
import asyncio

IST = pytz.timezone("Asia/Kolkata")

class ExitEngine:
    """
    Handles exit logic for the unified short straddle bot.
    This module is stateless—caller must provide current leg states.
    """

    def __init__(self,
                 target_pct_of_credit=0.5,
                 trail_factor=1.2,
                 trail_min_buffer=0.5,
                 no_move_wait_minutes=15,
                 min_decay_pct=3.0,
                 move_threshold_mult=0.5,
                 client=None,
                 use_dynamic_trailing=True,
                 dynamic_trail_levels=None):

        self.target_pct_of_credit = target_pct_of_credit
        self.trail_factor = trail_factor
        self.trail_min_buffer = trail_min_buffer
        self.no_move_wait_minutes = no_move_wait_minutes
        self.min_decay_pct = min_decay_pct
        self.move_threshold_mult = move_threshold_mult
        self.client = client
        self.use_dynamic_trailing = use_dynamic_trailing
        self.dynamic_trail_levels = dynamic_trail_levels or [
            (0,   3.0, 2.0),   # 0-10% profit: Wide buffer
            (10,  2.5, 1.5),   # 10-25% profit: Medium buffer
            (25,  2.0, 1.2),   # 25-40% profit: Tighter buffer
            (40,  1.5, 0.75),  # >40% profit: Tightest buffer
        ]

    def should_exit_no_move(self, bot):
        """
        Check early-exit based on:
        - No decay in total premium
        - No movement in underlying
        """
        if not bot.active_straddle or not bot.ce or not bot.pe:
            return False, "no_active"

        if not bot.current_trade_entry_time:
            return False, "no_entry_timestamp"

        now = datetime.now(pytz.utc).astimezone(IST)
        elapsed = (now - bot.current_trade_entry_time).total_seconds() / 60

        if elapsed < self.no_move_wait_minutes:
            return False, f"wait_more({elapsed:.1f}m)"

        ce_curr = bot.ce.get("current_price", bot.ce["sell_price"])
        pe_curr = bot.pe.get("current_price", bot.pe["sell_price"])
        qty = bot.ce.get("qty", 1)
        current_total = (ce_curr + pe_curr) * qty

        initial_credit = bot.straddle_credit if bot.straddle_credit else (bot.ce["sell_price"] + bot.pe["sell_price"]) * qty
        if initial_credit == 0:
            return False, "zero_initial_credit"

        decay_pct = (initial_credit - current_total) / initial_credit * 100

        # underlying move
        underlying_moved = False
        if bot.current_underlying_at_entry and bot.ohlc is not None:
            ohlc_df = bot.ohlc_df
            if not ohlc_df.empty:
                recent = ohlc_df[ohlc_df['timestamp'] >= bot.current_trade_entry_time]
                if not recent.empty:
                    max_p = recent['high'].max()
                    min_p = recent['low'].min()
                    entry = bot.current_underlying_at_entry
                    max_move = max(abs(max_p - entry), abs(entry - min_p))

                    atr_short = bot.atr_short if hasattr(bot, 'atr_short') else None
                    if atr_short:
                        threshold = atr_short * self.move_threshold_mult
                    else:
                        threshold = 2.0

                    underlying_moved = max_move >= threshold

        if decay_pct < self.min_decay_pct and not underlying_moved:
            return True, f"no_move_no_decay({decay_pct:.2f}%)"

        return False, f"decay={decay_pct:.2f}, moved={underlying_moved}"

    def move_to_breakeven(self, leg):
        """Move SL of survivor to breakeven."""
        asyncio.run(self.client.modify_sl_to_buycost(leg))

    def update_trailing_sl(self, leg):
        """Apply ATR-based trailing SL logic with dynamic buffer based on profit."""
        if not leg or leg.get("status") == "CLOSED":
            return None

        curr = leg.get("current_price")
        if curr is None:
            return None

        # Get dynamic buffer based on profit level
        buffer_amt = self._calculate_dynamic_buffer(leg)

        candidate = curr + buffer_amt
        prev = leg.get("stop_price")

        if prev is None or candidate < prev:
            leg["stop_price"] = candidate
            asyncio.run(self.client.modify_sl_to_cost(leg, leg["stop_price"]))
        return None

    def _calculate_dynamic_buffer(self, leg):
        """
        Calculate trailing buffer dynamically based on profit level.
        Returns wider buffers for early profits (let winners run),
        tighter buffers for large profits (protect gains).
        """
        sell_price = leg.get("sell_price")
        curr_price = leg.get("current_price")
        
        if not sell_price or not curr_price:
            # Fallback to static buffer
            opt_atr = leg.get("atr", 2.0)
            return max(self.trail_min_buffer, opt_atr * self.trail_factor)
        
        # Calculate profit percentage
        profit_pct = ((sell_price - curr_price) / sell_price) * 100.0
        
        # Use dynamic trailing if enabled
        if self.use_dynamic_trailing:
            # Find appropriate level based on profit percentage
            trail_factor = self.trail_factor
            min_buffer = self.trail_min_buffer
            
            for min_profit, factor, min_buf in sorted(self.dynamic_trail_levels, reverse=True):
                if profit_pct >= min_profit:
                    trail_factor = factor
                    min_buffer = min_buf
                    break
            
            # Calculate buffer with dynamic factor
            opt_atr = leg.get("atr", 2.0)
            buffer_amt = max(min_buffer, opt_atr * trail_factor)
            
            return buffer_amt
        else:
            # Static trailing (original logic)
            opt_atr = leg.get("atr", 2.0)
            return max(self.trail_min_buffer, opt_atr * self.trail_factor)

    def profit_target_hit(self, bot):
        pnl = bot.compute_total_pnl()
        target = bot.straddle_credit * self.target_pct_of_credit
        return pnl['total'] >= target

    def time_exit_due(self):
        now = datetime.now(pytz.utc).astimezone(IST)
        return now.time() >= dt_time(22, 0)

    # --------------------------------------------------------
    # Enhanced Trade Management
    # --------------------------------------------------------
    def progressive_breakeven_check(self, bot):
        """
        Check and apply progressive breakeven logic:
        1. At 25% profit -> Move SL to 10% risk (reduce risk)
        2. At 40% profit -> Move SL to Breakeven (risk free)
        Returns: True if an adjustment was made
        """
        if not bot.active_straddle:
            return False
            
        pnl = bot.compute_total_pnl()
        if bot.straddle_credit == 0:
            return False
            
        profit_pct = (pnl['total'] / bot.straddle_credit) * 100.0
        
        # Stage 1: 25% Profit -> Tighten SL
        if profit_pct >= 25.0 and not getattr(bot, 'breakeven_stage1', False):
            # Move SL to reduce risk (e.g., 10% of credit max loss)
            # This requires calculating new SL price for both legs
            # For simplicity in this modular design, we'll signal the bot to adjust
            return "STAGE1_25PCT"
            
        # Stage 2: 40% Profit -> Breakeven
        if profit_pct >= 40.0 and not getattr(bot, 'breakeven_stage2', False):
            return "STAGE2_40PCT"
            
        return None

    def check_dual_leg_trail(self, bot):
        """
        Check if we should trail BOTH legs (when straddle is profitable > 50%).
        """
        if not bot.active_straddle:
            return False
            
        pnl = bot.compute_total_pnl()
        if bot.straddle_credit == 0:
            return False
            
        profit_pct = (pnl['total'] / bot.straddle_credit) * 100.0
        
        # Start trailing both legs after 50% profit
        if profit_pct >= 50.0:
            return True
            
        return False

    def smart_survivor_decision(self, bot, hit_leg):
        """
        Decide what to do when one leg hits SL:
        - If net loss > threshold -> Exit survivor too
        - Else -> Keep survivor and trail aggressively
        """
        survivor = bot.ce if hit_leg == bot.pe else bot.pe
        
        if not survivor or survivor.get('status') == 'CLOSED':
            return "EXIT_ALL", "No survivor"
            
        # Calculate net position state
        hit_loss = (hit_leg['buy_price'] - hit_leg['sell_price']) * hit_leg['qty']
        survivor_profit = (survivor['sell_price'] - survivor['current_price']) * survivor['qty']
        net_pnl = survivor_profit - hit_loss
        
        # If net loss is already > 20% of initial credit, cut losses
        if net_pnl < -(bot.straddle_credit * 0.20):
            return "EXIT_ALL", f"Net loss {net_pnl:.2f} exceeds 20% threshold"
            
        return "KEEP_SURVIVOR", "Net PnL acceptable"
