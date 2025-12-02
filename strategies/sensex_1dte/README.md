# Sensex 1DTE Directional Credit Spread Strategy

## Overview

**Strategy Type:** Intraday, Risk-Defined, Trend-Following Credit Spread  
**Instrument:** Sensex 1 DTE Options  
**Exchange:** BFO (BSE Futures & Options)

This strategy trades directional credit spreads (Bull Put Spreads and Bear Call Spreads) based on trend confirmation from multiple technical indicators.

---

## Data Sources

### Previous Day OHLC
- **Source:** Yahoo Finance (`^BSESN`)
- **Usage:** Calculate Pivot Points for strike selection
- **Fetch Command:**
  ```python
  df = yf.download("^BSESN", period="2d", interval="1d")
  prev_high  = df.iloc[-2]["High"]
  prev_low   = df.iloc[-2]["Low"]
  prev_close = df.iloc[-2]["Close"]
  ```

### Intraday Data
- **Timeframe:** 5-minute candles
- **Indicators:** Supertrend, ATR, ADX

---

## Technical Indicators

All indicators calculated on **5-minute timeframe**:

### 1. Supertrend (10, 3)
- **Purpose:** Primary trend direction
- **Signals:**
  - GREEN → Bullish
  - RED → Bearish

### 2. ATR (14)
- **Purpose:** Volatility filter
- **Condition:** ATR must be stable or decreasing for valid trade

### 3. ADX (14)
- **Purpose:** Trend strength filter
- **Thresholds:**
  - ADX < 18 → Weak trend → No entry
  - ADX 18-25 → Moderate → OK
  - ADX > 25 → Strong trend → Best zone

---

## Pivot Point Calculations

Using previous day OHLC from Yahoo Finance:

```
Pivot Point (P):  P = (H + L + C) / 3
Support 1 (S1):   S1 = (2 * P) - H
Resistance 1 (R1): R1 = (2 * P) - L
```

**Usage:**
- **S1** → Anchor for Bull Put Spreads
- **R1** → Anchor for Bear Call Spreads

---

## Entry Conditions

### Entry Time Window
✅ **Valid Entry:** 9:35 AM - 2:45 PM

### Bullish Entry (Bull Put Spread - BPS)

Enter **ONLY IF ALL** conditions are TRUE:

1. ✅ Supertrend = GREEN
2. ✅ ATR = stable or decreasing
3. ✅ ADX > 18
4. ✅ No Supertrend flip in last 3 candles
5. ✅ Price stays above Pivot Point (P)

**Strike Selection:**
```
- Use S1 as anchor (strong support)
- Sell PE near S1
- Buy lower PE 300-500 points below

Example (Sensex):
  S1 = 79,500
  → Sell 79,500 PE
  → Buy 79,000 PE
```

### Bearish Entry (Bear Call Spread - BCS)

Enter **ONLY IF ALL** conditions are TRUE:

1. ✅ Supertrend = RED
2. ✅ ATR = stable or decreasing
3. ✅ ADX > 18
4. ✅ No Supertrend flip in last 3 candles
5. ✅ Price stays below Pivot Point (P)

**Strike Selection:**
```
- Use R1 as resistance
- Sell CE near R1
- Buy higher CE 300-500 points above

Example (Sensex):
  R1 = 80,500
  → Sell 80,500 CE
  → Buy 81,000 CE
```

---

## Multiple Entries (Scaling)

**Allowed:** Yes, but only if trend continues

### Scaling Rules

✅ **Allowed ONLY IF:**
- Existing spread is in PROFIT
- Supertrend unchanged
- ATR decreasing
- ADX ≥ 20 (trend strengthening)
- Price not violating Pivot (P)
- Minimum gap of 10-15 minutes between entries
- Total risk stays inside daily risk budget

### Recommended Scaling Size
- **Entry 1:** 3 lots
- **Entry 2:** 2 lots
- **Entry 3:** 1 lot (only in very strong ADX > 25 trend)

❌ **Not Allowed When:**
- Current position is in loss
- Supertrend flips
- ADX falls by > 20%
- ATR increases sharply
- After 12:00 PM (avoid adding)

---

## Stop Loss Rules

Use **EITHER** one (whichever hits first):

### Option A: 50% Spread Loss
```
If spread premium received = ₹50
→ SL at ₹75-₹80
```

### Option B: Spread Value Doubles
```
Credit = ₹50
→ SL = ₹100
```

### SL Exit Trigger
- Immediate exit
- Do not re-enter same side for at least **30 minutes**

---

## Target Rules

### Profit Target
✅ Book target at **40-60% credit capture**

**Example:**
```
Credit = ₹50
→ Close at ₹20-30
```

### Time Exit
✅ **Mandatory Exit:** 3:00 PM (all positions, win or lose)

---

## Re-Entry After Stop Loss

If SL hits:

1. ✅ Wait **30 minutes**
2. ✅ Enter only if new Supertrend direction re-establishes
3. ✅ ADX > 18
4. ✅ ATR stable

❌ **Never** reverse trade immediately after SL

---

## Configuration

Edit `config.py` to customize parameters:

```python
# Entry window
ENTRY_START_TIME = dt_time(9, 35)
ENTRY_END_TIME = dt_time(14, 45)

# Indicators
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
ATR_PERIOD = 14
ADX_PERIOD = 14
ADX_MIN = 18.0

# Spread width
SPREAD_WIDTH_MIN = 300
SPREAD_WIDTH_MAX = 500

# Position sizing
ENTRY_LOTS = [3, 2, 1]

# Stop loss
SL_OPTION_A_PCT = 50.0
SL_OPTION_B_MULTIPLIER = 2.0

# Target
TARGET_MIN_PCT = 40.0
TARGET_MAX_PCT = 60.0
```

---

## Running the Strategy

### 1. Set Environment Variables
```bash
export OPENALGO_API_KEY="your_api_key"
export OPENALGO_HOST="http://localhost:5000/"
export OPENALGO_WS="ws://localhost:5000/ws"
```

### 2. Enable Live Trading
Edit `config.py`:
```python
PLACE_ORDERS = True  # Set to False for simulation
```

### 3. Run Strategy
```bash
cd sensex_1dte
python run_credit_spread.py
```

---

## Strategy Flow

```
1. Fetch Previous Day OHLC from Yahoo Finance
2. Calculate Pivot Points (P, S1, R1)
3. Connect to OpenAlgo WebSocket
4. Subscribe to Sensex 5-min data
5. Calculate Indicators (Supertrend, ATR, ADX)
6. Check Entry Signals
   ├─ Bullish? → Place Bull Put Spread (S1 anchor)
   └─ Bearish? → Place Bear Call Spread (R1 anchor)
7. Monitor Positions
   ├─ Check Stop Loss (50% loss or value doubles)
   ├─ Check Target (40-60% profit)
   └─ Check Time Exit (3:00 PM)
8. Scale positions if conditions met
9. Exit at 3:00 PM or earlier on SL/Target
```

---

## Risk Management

- **Max Loss per Spread:** ₹(Spread Width - Credit) × Lot Size
- **Position Sizing:** 3 + 2 + 1 = 6 total lots max
- **Daily Risk:** Controlled via stop loss and max entries
- **Time Risk:** Mandatory exit at 3:00 PM

---

## Files

- `config.py` - Strategy configuration
- `run_credit_spread.py` - Main strategy runner
- `README.md` - This file

---

## Notes

- Strategy trades **1 DTE** options only
- Theta decay works in your favor (credit spread)
- Uses Yahoo Finance for reliable previous day data
- All indicators run on 5-minute timeframe
- Pivot points recalculated daily

---

## Disclaimer

This strategy is for educational purposes. Test thoroughly in simulation mode before live trading.
