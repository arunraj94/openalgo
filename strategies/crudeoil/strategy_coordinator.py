# strategy_coordinator.py
# Cross-Strategy Coordination to Avoid Overlapping Positions
# Prevents Directional + Ratio Spread from running simultaneously

import os
import json
from datetime import datetime
import pytz
from typing import Optional, Dict
import threading

IST = pytz.timezone("Asia/Kolkata")

class StrategyCoordinator:
    """
    Coordinates multiple strategies to avoid conflicts.
    
    Use Cases:
    1. Prevent Directional + Ratio Spread on same day (both trend-following)
    2. Track global position limits
    3. Share market regime information
    """
    
    def __init__(self, state_file="./strategy_state.json"):
        self.state_file = state_file
        self.lock = threading.Lock()
        self._load_state()
    
    def _load_state(self):
        """Load current strategy states from file"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {
                'directional': {'active': False, 'entry_time': None, 'direction': None},
                'ratio_spread': {'active': False, 'entry_time': None, 'direction': None},
                'straddle': {'active': False, 'entry_time': None},
                'mean_reversion': {'active': False, 'entry_time': None},
                'vol_expansion': {'active': False, 'entry_time': None},
                'last_update': None
            }
    
    def _save_state(self):
        """Persist state to file"""
        with self.lock:
            self.state['last_update'] = datetime.now(IST).isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
    
    # ========================================
    # Strategy Entry Permission
    # ========================================
    
    def can_enter_directional(self) -> tuple[bool, str]:
        """
        Check if Directional strategy can enter.
        
        Rules:
        1. If Ratio Spread active → BLOCK (avoid overlap)
        2. Otherwise → ALLOW
        """
        with self.lock:
            if self.state['ratio_spread']['active']:
                return False, "Ratio Spread already active (avoid trend overlap)"
            return True, "OK"
    
    def can_enter_ratio_spread(self) -> tuple[bool, str]:
        """
        Check if Ratio Spread can enter.
        
        Rules:
        1. If Directional active → BLOCK (avoid overlap)
        2. Otherwise → ALLOW
        
        Priority: Directional > Ratio (safer, defined risk)
        """
        with self.lock:
            if self.state['directional']['active']:
                return False, "Directional Spread already active (avoid trend overlap)"
            return True, "OK"
    
    def can_enter_any_strategy(self, strategy_name: str) -> tuple[bool, str]:
        """
        Generic entry check for any strategy.
        
        Args:
            strategy_name: 'directional', 'ratio_spread', 'straddle', etc.
        """
        # Special handling for correlated strategies
        if strategy_name == 'directional':
            return self.can_enter_directional()
        elif strategy_name == 'ratio_spread':
            return self.can_enter_ratio_spread()
        else:
            # Other strategies have no conflicts
            return True, "OK"
    
    # ========================================
    # Strategy State Management
    # ========================================
    
    def mark_entry(self, strategy_name: str, direction: Optional[str] = None):
        """
        Mark strategy as active (position entered).
        
        Args:
            strategy_name: 'directional', 'ratio_spread', etc.
            direction: 'BULLISH', 'BEARISH', or None (for non-directional)
        """
        with self.lock:
            self.state[strategy_name]['active'] = True
            self.state[strategy_name]['entry_time'] = datetime.now(IST).isoformat()
            if direction:
                self.state[strategy_name]['direction'] = direction
            self._save_state()
    
    def mark_exit(self, strategy_name: str):
        """Mark strategy as inactive (position exited)."""
        with self.lock:
            self.state[strategy_name]['active'] = False
            self.state[strategy_name]['entry_time'] = None
            self.state[strategy_name]['direction'] = None
            self._save_state()
    
    def is_active(self, strategy_name: str) -> bool:
        """Check if a strategy has an active position."""
        with self.lock:
            return self.state[strategy_name]['active']
    
    def get_active_strategies(self) -> list[str]:
        """Get list of all currently active strategies."""
        with self.lock:
            return [name for name, data in self.state.items() 
                   if isinstance(data, dict) and data.get('active', False)]
    
    # ========================================
    # Alternative: Time-Based Priority
    # ========================================
    
    def get_trend_strategy_priority(self) -> str:
        """
        Decide which trend strategy gets priority today.
        
        Logic:
        1. Before 3:30 PM → Directional (all-day strategy)
        2. After 3:30 PM → Ratio Spread (evening session)
        3. If one already active → Other blocked
        
        Returns: 'directional' or 'ratio_spread'
        """
        now = datetime.now(IST).time()
        
        # Check if either is active
        if self.state['directional']['active']:
            return 'directional'
        if self.state['ratio_spread']['active']:
            return 'ratio_spread'
        
        # Time-based priority
        from datetime import time as dt_time
        ratio_start_time = dt_time(15, 30)  # 3:30 PM
        
        if now >= ratio_start_time:
            return 'ratio_spread'  # Evening session
        else:
            return 'directional'   # All-day session
    
    # ========================================
    # Advanced: Same-Direction Check
    # ========================================
    
    def can_enter_same_direction(self, strategy_name: str, direction: str) -> tuple[bool, str]:
        """
        Advanced check: Allow both Directional + Ratio if SAME direction.
        
        This is safer than blocking entirely - if both are bullish,
        they reinforce each other instead of conflicting.
        
        Args:
            strategy_name: Strategy trying to enter
            direction: 'BULLISH' or 'BEARISH'
        """
        with self.lock:
            if strategy_name == 'directional':
                if self.state['ratio_spread']['active']:
                    ratio_dir = self.state['ratio_spread']['direction']
                    if ratio_dir and ratio_dir != direction:
                        return False, f"Ratio Spread active in {ratio_dir} (opposite direction)"
                    # Same direction is OK
                    return True, f"Ratio Spread active but same direction ({direction})"
                return True, "OK"
            
            elif strategy_name == 'ratio_spread':
                if self.state['directional']['active']:
                    dir_dir = self.state['directional']['direction']
                    if dir_dir and dir_dir != direction:
                        return False, f"Directional active in {dir_dir} (opposite direction)"
                    return True, f"Directional active but same direction ({direction})"
                return True, "OK"
            
            return True, "OK"
    
    # ========================================
    # Daily Reset
    # ========================================
    
    def reset_daily(self):
        """Reset all strategy states (call at start of day)."""
        with self.lock:
            for strategy in self.state:
                if isinstance(self.state[strategy], dict):
                    self.state[strategy]['active'] = False
                    self.state[strategy]['entry_time'] = None
                    self.state[strategy]['direction'] = None
            self._save_state()


# Singleton instance
_coordinator = None

def get_coordinator() -> StrategyCoordinator:
    """Get global coordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = StrategyCoordinator()
    return _coordinator


# ========================================
# Integration Examples
# ========================================

"""
USAGE IN run_directional.py:

from strategy_coordinator import get_coordinator

class DirectionalRun:
    def __init__(self):
        self.coordinator = get_coordinator()
        # ... rest of init
    
    def enter_trade(self, signal, lots):
        # CHECK BEFORE ENTRY
        can_enter, reason = self.coordinator.can_enter_directional()
        if not can_enter:
            logger.warning(f"⛔ Skipping Directional Entry: {reason}")
            return
        
        # ... place orders ...
        
        # MARK AS ACTIVE
        self.coordinator.mark_entry('directional', direction=signal)
    
    def exit_trade(self, reason):
        # ... close positions ...
        
        # MARK AS INACTIVE
        self.coordinator.mark_exit('directional')


USAGE IN run_crude_ratio_spread.py:

from strategy_coordinator import get_coordinator

class ProtectedRatioSpreadRunner:
    def __init__(self):
        self.coordinator = get_coordinator()
        # ... rest of init
    
    def enter_phase1(self, direction):
        # CHECK BEFORE ENTRY
        can_enter, reason = self.coordinator.can_enter_ratio_spread()
        if not can_enter:
            logger.warning(f"⛔ Skipping Ratio Spread: {reason}")
            return
        
        # ... place orders ...
        
        # MARK AS ACTIVE
        self.coordinator.mark_entry('ratio_spread', direction=direction)
    
    def exit_all_positions(self, reason):
        # ... close positions ...
        
        # MARK AS INACTIVE
        self.coordinator.mark_exit('ratio_spread')


ALTERNATIVE: Same-Direction Check (More Flexible)

    def enter_trade(self, signal, lots):
        # Allow both if same direction
        can_enter, reason = self.coordinator.can_enter_same_direction('directional', signal)
        if not can_enter:
            logger.warning(f"⛔ Skipping Entry: {reason}")
            return
        # ... proceed with entry


DAILY RESET (Call at session start):

    def start(self):
        # Reset coordinator at start of day
        now = datetime.now(IST).time()
        if now.hour == 9 and now.minute < 5:  # 9:00-9:05 AM
            self.coordinator.reset_daily()
            logger.info("✅ Strategy coordinator reset for new day")
        
        # ... rest of startup
"""

if __name__ == '__main__':
    # Test the coordinator
    coord = get_coordinator()
    
    print("Testing Strategy Coordination:\n")
    
    # Test 1: Try to enter Directional
    can_enter, reason = coord.can_enter_directional()
    print(f"1. Can Directional enter? {can_enter} - {reason}")
    
    # Enter Directional
    coord.mark_entry('directional', 'BULLISH')
    print("   → Directional entered (BULLISH)\n")
    
    # Test 2: Try to enter Ratio Spread (should be blocked)
    can_enter, reason = coord.can_enter_ratio_spread()
    print(f"2. Can Ratio Spread enter? {can_enter} - {reason}")
    print()
    
    # Test 3: Same direction check
    can_enter, reason = coord.can_enter_same_direction('ratio_spread', 'BULLISH')
    print(f"3. Can Ratio enter if same direction? {can_enter} - {reason}")
    print()
    
    # Exit Directional
    coord.mark_exit('directional')
    print("   → Directional exited\n")
    
    # Test 4: Now Ratio can enter
    can_enter, reason = coord.can_enter_ratio_spread()
    print(f"4. Can Ratio Spread enter now? {can_enter} - {reason}")
    
    # Show active strategies
    print(f"\nActive strategies: {coord.get_active_strategies()}")
