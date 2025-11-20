
import unittest
import sys
import os
import asyncio
import shutil
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules to test
# We need to mock setup_logger BEFORE importing run to avoid file creation
with patch('logger.setup_logger') as mock_setup:
    mock_setup.return_value = MagicMock()
    from run import UnifiedRun
    
from supervisor import check_risk
from db_logger import init_db, log_entry, Trade
import config
import sqlite3
import logging

# Mock Config
class MockConfig:
    DB_PATH = "test_trades.db"
    OPENALGO_API_KEY = "test_key"
    OPENALGO_HOST = "test_host"
    OPENALGO_WS = "test_ws"
    UNDERLYING_SYMBOL = "CRUDEOIL"
    UNDERLYING_EXCHANGE = "MCX"
    LOTSIZE = 100
    ALLOWED_WINDOWS = [(datetime.min.time(), datetime.max.time())] # Always open
    ATR_SHORT_PERIOD = 3
    ATR_LONG_PERIOD = 10
    ATR_MULTIPLIER_HIGH = 2.0
    ATR_MULTIPLIER_LOW = 0.5
    TIME_BASE_SL_BUCKETS = [(datetime.min.time(), datetime.max.time(), 20.0)]
    TARGET_PCT_OF_CREDIT = 0.5
    TRAIL_FACTOR = 0.1
    TRAIL_MIN_BUFFER = 5.0
    NO_MOVE_WAIT_MINUTES = 15
    MIN_DECAY_PCT = 0.05
    MOVE_THRESHOLD_MULT = 1.0
    MAX_DAILY_LOSS = 1000.0
    DAILY_PROFIT_LOCK = 5000.0
    CORRELATION_REDUCTION_FACTOR = 0.5

class TestCrudeOilWorkflow(unittest.TestCase):

    def setUp(self):
        # Clean up artifacts
        if os.path.exists("test_trades.db"):
            try:
                os.remove("test_trades.db")
            except OSError:
                pass
        if os.path.exists("STOP_TRADING"):
            os.remove("STOP_TRADING")
            
        # Initialize DB using actual db_logger logic
        self.session = init_db("test_trades.db")

    def tearDown(self):
        self.session.close()
        if os.path.exists("STOP_TRADING"):
            os.remove("STOP_TRADING")
        # We don't delete DB here to avoid locking issues on Windows if close() isn't immediate
        # setUp will clean it.

    @patch('run.init_db') # Patch init_db to prevent UnifiedRun from opening its own connection
    @patch('run.logger')
    @patch('run.OpenAlgoClientWrapper')
    def test_full_workflow(self, MockClient, mock_logger, MockInitDB):
        print("\n--- Starting Workflow Simulation ---")
        
        # Setup Mock InitDB to return a mock, but we will overwrite runner.db_session anyway
        MockInitDB.return_value = MagicMock()

        # Setup Mock Client
        client_instance = MockClient.return_value
        client_instance.get_ATM_strike.return_value = 6000
        client_instance.nearest_expiry = "19NOV24"
        client_instance.build_option_symbol.side_effect = lambda base, exp, strike, type: {'symbol': f"{base}{strike}{type}"}
        client_instance.get_exchange_ltp.return_value = 6005
        
        # Mock Market Depth for Liquidity Check
        client_instance.get_market_depth.return_value = {
            'bids': [{'price': 100, 'quantity': 10}],
            'asks': [{'price': 100.1, 'quantity': 10}]
        }
        
        # Mock Async Order Placement
        async def mock_place_order(*args, **kwargs):
            return {'orderid': '123', 'price': 100.0, 'status': 'complete'}
        client_instance.async_place_orders.side_effect = mock_place_order
        
        async def mock_sl_order(*args, **kwargs):
            return {'orderid': '124', 'status': 'open'}
        client_instance.async_sl_order.side_effect = mock_sl_order

        # Initialize Runner with Mock Config
        runner = UnifiedRun(config=MockConfig)
        # Inject the session we created
        runner.db_session = self.session
        
        # 1. Test Entry Logic
        print("Step 1: Testing Entry Logic...")
        runner.place_straddle(lots=1, entry_signal="TEST_SIGNAL")
        
        # Check if logger.error was called
        if mock_logger.error.called:
            print(f"  -> Logger Error Called: {mock_logger.error.call_args}")
        
        # Verify Active State
        self.assertTrue(runner.active_straddle)
        self.assertIsNotNone(runner.ce)
        self.assertIsNotNone(runner.pe)
        print("  -> Entry Successful. Active Straddle Set.")
        
        # Verify DB Entry
        trades = self.session.query(Trade).all()
        self.assertEqual(len(trades), 1)
        print("  -> DB Record Created.")

        # 2. Test Risk Supervisor (Stop Trading)
        print("Step 2: Testing Global Risk Supervisor...")
        # Manually insert a loss to trigger stop
        loss_trade = Trade(
            entry_time=datetime.now(),
            ce_symbol="CE", pe_symbol="PE",
            ce_sell_price=100, pe_sell_price=100,
            lots=1, credit_received=200,
            realized_pnl=-2000.0,
            exit_time=datetime.now() 
        )
        self.session.add(loss_trade)
        self.session.commit()
        
        # Run Supervisor Check
        with patch('supervisor.cfg', MockConfig):
             triggered = check_risk()
             self.assertTrue(triggered)
             self.assertTrue(os.path.exists("STOP_TRADING"))
             print("  -> Risk Check Triggered. STOP_TRADING file created.")

        # 3. Test Entry Blocked by Stop File
        print("Step 3: Testing Entry Blocked by Risk Stop...")
        runner.active_straddle = False # Reset
        runner.place_straddle(lots=1, entry_signal="TEST_SIGNAL_2")
        self.assertFalse(runner.active_straddle)
        print("  -> Entry Blocked as expected.")

        # 4. Test Liquidity Check Failure
        print("Step 4: Testing Liquidity Check...")
        os.remove("STOP_TRADING") # Clear stop
        client_instance.get_market_depth.return_value = None # Simulate no depth
        runner.place_straddle(lots=1, entry_signal="TEST_SIGNAL_3")
        self.assertFalse(runner.active_straddle)
        print("  -> Entry Blocked due to poor liquidity.")

        print("--- Simulation Complete: SUCCESS ---")

if __name__ == '__main__':
    unittest.main(verbosity=2)
