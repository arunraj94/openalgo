# supervisor.py
import time
import os
import sqlite3
from datetime import datetime
import pytz
from config import default_config as cfg
from logger import setup_logger

# Setup Logger
logger = setup_logger("SUPERVISOR")
IST = pytz.timezone("Asia/Kolkata")

STOP_FILE = "STOP_TRADING"

def get_daily_pnl(db_path):
    """
    Calculates the total Realized PnL for the current day from trades.db.
    Note: This currently only tracks CLOSED trades (Realized PnL).
    For a perfect supervisor, we would also need to fetch live MTM of open positions,
    but that requires API access. For now, we stick to Realized PnL from DB.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        today_str = datetime.now(IST).strftime("%Y-%m-%d")
        
        # Query for closed trades today
        query = """
            SELECT realized_pnl 
            FROM trades 
            WHERE date(entry_time) = ? 
            AND exit_time IS NOT NULL
        """
        cursor.execute(query, (today_str,))
        rows = cursor.fetchall()
        
        total_pnl = sum([row[0] for row in rows if row[0] is not None])
        
        conn.close()
        return total_pnl
    except Exception as e:
        logger.error(f"Error calculating PnL: {e}")
        return 0.0

def check_risk():
    pnl = get_daily_pnl(cfg.DB_PATH)
    logger.info(f"Current Daily Realized PnL: {pnl:.2f}")
    
    # Check Max Loss
    if pnl < -cfg.MAX_DAILY_LOSS:
        logger.critical(f"🚨 MAX DAILY LOSS HIT ({pnl:.2f} < -{cfg.MAX_DAILY_LOSS})! Stopping all trading.")
        with open(STOP_FILE, "w") as f:
            f.write(f"MAX LOSS HIT: {pnl}")
        return True

    # Check Profit Lock
    if pnl > cfg.DAILY_PROFIT_LOCK:
        logger.info(f"💰 DAILY PROFIT TARGET HIT ({pnl:.2f} > {cfg.DAILY_PROFIT_LOCK})! Locking profits.")
        with open(STOP_FILE, "w") as f:
            f.write(f"PROFIT LOCK: {pnl}")
        return True
        
    # If we are back within limits (e.g. manual override), remove stop file?
    # Usually, once a stop is hit, it should stay for the day. 
    # We will NOT auto-remove the file. User must delete it manually to resume.
    
    return False

def main():
    logger.info("🛡️ Global Risk Supervisor Started")
    logger.info(f"Limits: Max Loss = -{cfg.MAX_DAILY_LOSS}, Profit Lock = {cfg.DAILY_PROFIT_LOCK}")
    
    # Remove old stop file on startup if it's a new day? 
    # For safety, we don't auto-delete. User must manage.
    
    try:
        while True:
            check_risk()
            time.sleep(30) # Check every 30 seconds
    except KeyboardInterrupt:
        logger.info("Supervisor Stopped")

if __name__ == "__main__":
    main()
