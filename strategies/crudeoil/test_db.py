
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db_logger import init_db, log_entry, Trade

def test_db():
    print("Initializing DB...")
    if os.path.exists("debug_trades.db"):
        os.remove("debug_trades.db")
        
    session = init_db("debug_trades.db")
    
    print("Logging Entry...")
    try:
        tid = log_entry(session, "CE", "PE", 100.0, 100.0, 1, "TEST")
        print(f"Entry Logged. ID: {tid}")
    except Exception as e:
        print(f"Log Entry Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Querying Entry...")
    trades = session.query(Trade).all()
    print(f"Found {len(trades)} trades.")
    for t in trades:
        print(f"Trade: {t.id}, {t.ce_symbol}, {t.realized_pnl}")

    session.close()
    print("Done.")

if __name__ == "__main__":
    test_db()
