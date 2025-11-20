# db_logger.py
# SQLAlchemy ORM models + helper functions for Unified Short Straddle Bot

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pytz
import os

IST = pytz.timezone("Asia/Kolkata")
Base = declarative_base()

# ----------------------------------------------------------------------
# ORM Models
# ----------------------------------------------------------------------

class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)

    ce_symbol = Column(String, nullable=False)
    pe_symbol = Column(String, nullable=False)

    ce_sell_price = Column(Float, nullable=False)
    pe_sell_price = Column(Float, nullable=False)

    ce_buy_price = Column(Float)
    pe_buy_price = Column(Float)

    lots = Column(Integer, nullable=False)
    credit_received = Column(Float, nullable=False)

    realized_pnl = Column(Float)
    entry_signal = Column(String)
    notes = Column(Text)


class LegModification(Base):
    __tablename__ = 'leg_modifications'

    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    leg = Column(String, nullable=False)  # CE or PE
    event = Column(String, nullable=False)
    value = Column(Float)
    notes = Column(Text)


# ----------------------------------------------------------------------
# DB Helper Functions
# ----------------------------------------------------------------------

def init_db(path='./trades.db'):
    """Create SQLite DB and return a session."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    engine = create_engine(f'sqlite:///{path}', connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def now_ist():
    return datetime.now(pytz.utc).astimezone(IST)


def log_entry(session, ce_symbol, pe_symbol, ce_sell, pe_sell, lots, signal, notes=""):
    """Create a new Trade row and return trade_id."""
    t = Trade(
        entry_time=now_ist(),
        ce_symbol=ce_symbol,
        pe_symbol=pe_symbol,
        ce_sell_price=ce_sell,
        pe_sell_price=pe_sell,
        lots=lots,
        credit_received=(ce_sell + pe_sell) * lots,
        entry_signal=signal,
        notes=notes
    )
    session.add(t)
    session.commit()
    return t.id


def log_leg_mod(session, trade_id, leg, event, value=None, notes=""):
    m = LegModification(
        trade_id=trade_id,
        timestamp=now_ist(),
        leg=leg,
        event=event,
        value=value,
        notes=notes
    )
    session.add(m)
    session.commit()
    return m.id


def log_exit(session, trade_id, ce_buy=None, pe_buy=None, realized_pnl=None, notes=""):
    t = session.query(Trade).filter(Trade.id == trade_id).one_or_none()
    if not t:
        return None

    t.exit_time = now_ist()
    if ce_buy is not None:
        t.ce_buy_price = ce_buy
    if pe_buy is not None:
        t.pe_buy_price = pe_buy
    if realized_pnl is not None:
        t.realized_pnl = realized_pnl

    t.notes = (t.notes or '') + '\n' + notes
    session.commit()
    return t.id
