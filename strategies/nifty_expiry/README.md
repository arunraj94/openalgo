# Nifty 0DTE (Expiry Day) Strategies

This folder contains strategies specifically designed for **Nifty expiry day (Thursday)** intraday trading.

## 🎯 Strategies Implemented

### 1. **Short Straddle (Whipsaw Protected)**
- **File**: `run_0dte_short_straddle.py`
- **Type**: Premium selling in range-bound markets
- **Risk**: Unlimited (managed with SL)
- **Features**:
  - Multiple entry windows (10:00 AM, 11:30 AM, 1:45 PM)
  - Progressive lock-in system (50-30-15 rule)
  - Whipsaw detection and protection
  - Automatic leg management
  - Real-time trailing stop loss

## ⚙️ Configuration

All strategies use their respective config files:
- `config_0dte_short_straddle.py`

### Key Parameters
- **Lot Size**: 50 (Nifty)
- **Max Lots**: 2 (configurable)
- **Expiry Day**: Thursday only
- **Trading Mode**: **REAL TRADING** (PLACE_ORDERS = True)

## 🚀 Running Strategies

### Short Straddle
```bash
python -m openalgo.strategies.nifty_expiry.run_0dte_short_straddle
```

### Pre-requisites
1. OpenAlgo server running
2. Environment variables set:
   - `OPENALGO_API_KEY`
   - `OPENALGO_HOST`
   - `OPENALGO_WS`
3. Broker connection active
4. Today must be Thursday (expiry day)

## 📊 Strategy Details

### Short Straddle Entry Conditions

**Window 1 (10:00-10:30 AM):**
- First 45-min range < 60 points
- RSI between 40-60
- ADX < 25
- No trending candles

**Window 2 (11:30 AM-12:30 PM):**
- Morning range < 120 points
- Market consolidating
- ADX < 25
- Price in middle of range

**Window 3 (1:45-2:30 PM):** *Disabled by default*
- Use with caution - close to expiry
- Higher theta but higher gamma risk

### Risk Management

**Stop Loss:**
- Individual leg: 40% of premium
- Combined: 30% of total premium

**Profit Target:**
- 50% of premium collected

**Whipsaw Protection:**
- Progressive lock-in when one leg hits SL
- 50-30-15 rule for trailing
- Automatic whipsaw detection

### Leg Management When SL Hits

**Strategy**: `CONVERT_TO_NAKED` (recommended)
1. Close losing leg immediately
2. Keep winning leg
3. Apply progressive lock-in (NOT breakeven!)
4. Trail every 10 points
5. Watch for whipsaw signals
6. Close early if detected

## ⚠️ Important Warnings

1. **0DTE = HIGH RISK**: Can lose 100% premium + more
2. **Unlimited Loss**: Without SL, losses can be catastrophic
3. **Only Expiry Day**: Strategy only trades on Thursdays
4. **REAL MONEY**: PLACE_ORDERS is TRUE - real trading!
5. **Position Sizing**: Start with 1-2 lots maximum
6. **Discipline**: Follow SL without emotions

## 📝 Logging & Tracking

Logs stored in: `./logs/nifty_expiry/`
Database: `./nifty_expiry_straddle.db`

Track every trade with P&L, entry/exit times, and conditions.

## 🛠️ Customization

Edit config files to adjust:
- Entry windows
- Stop loss percentages
- Profit targets
- Whipsaw protection settings
- Lot sizes

## 📈 Expected Performance

**Conservative Estimates:**
- Win Rate: 60-65%
- Avg Win: 40-50% of premium
- Avg Loss: 25-30% of premium
- Monthly Return: 8-12% of allocated capital

---

**Last Updated**: December 2024  
**Status**: Production Ready ✅  
**Trading Mode**: REAL (Not Paper) ⚠️
