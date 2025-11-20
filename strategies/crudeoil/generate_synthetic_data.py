
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from db_logger import init_db, log_entry, log_exit, Trade
import config

def generate_data(num_trades=500):
    print(f"Generating {num_trades} synthetic trades...")
    session = init_db(config.default_config.DB_PATH)
    
    # Start date: 6 months ago
    start_date = datetime.now() - timedelta(days=180)
    
    count = 0
    for i in range(num_trades):
        # Random time
        days_offset = random.randint(0, 180)
        # Bias towards market hours (9 AM - 11 PM)
        hour = random.randint(9, 23) 
        minute = random.randint(0, 59)
        
        entry_time = start_date + timedelta(days=days_offset)
        entry_time = entry_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Skip weekends
        if entry_time.weekday() > 4:
            continue

        # Logic: Morning trades (9-11) and Evening trades (18-20) are profitable
        # Afternoon (12-16) is choppy/loss
        minute_of_day = hour * 60 + minute
        
        is_good_time = (9*60 <= minute_of_day <= 11*60) or (18*60 <= minute_of_day <= 20*60)
        
        if is_good_time:
            # 70% win rate
            win = random.random() < 0.7
        else:
            # 30% win rate
            win = random.random() < 0.3
            
        if win:
            pnl = random.uniform(500, 2000)
        else:
            pnl = random.uniform(-1000, -200)
            
        # Log Entry
        t = Trade(
            entry_time=entry_time,
            ce_symbol="SYNTH_CE",
            pe_symbol="SYNTH_PE",
            ce_sell_price=100,
            pe_sell_price=100,
            lots=1,
            credit_received=200,
            realized_pnl=pnl,
            entry_signal="SYNTHETIC",
            exit_time=entry_time + timedelta(minutes=30),
            notes="Synthetic Data"
        )
        session.add(t)
        count += 1

    session.commit()
    session.close()
    print(f"Successfully added {count} synthetic trades to DB.")

if __name__ == "__main__":
    generate_data()
