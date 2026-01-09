import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sqlite3
import os
import time
import threading
from datetime import datetime

# ==============================================================================
# 1. CONFIGURATION DU MANDAT PRO
# ==============================================================================
CONFIG = {
    "ASSET": "SPY", "CASH": "BIL", "VIX": "^VIX", "Y10": "^TNX", "Y3M": "^IRX",
    "DB_NAME": "trading_live.db",
    "INITIAL_CAPITAL": 100000.0,
    "VOL_TARGET": 0.12,
    "ES_GUARD": 0.10,
    "SMA_PERIOD": 200,
    "FEES_TOTAL": 0.0010, # Commissions + Slippage estimé
    "LOOP_INTERVAL": 60   # Vérification chaque minute
}

class TradingDatabase:
    def __init__(self, db_name):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_name) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS state 
                (id INTEGER PRIMARY KEY, equity REAL, position REAL, fees REAL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS logs 
                (timestamp TEXT, equity REAL, position REAL, score REAL)""")
            
            # Initialisation si vide
            if not conn.execute("SELECT * FROM state").fetchone():
                conn.execute("INSERT INTO state VALUES (1, ?, 0, 0)", (CONFIG["INITIAL_CAPITAL"],))

    def get_state(self):
        with sqlite3.connect(self.db_name) as conn:
            return conn.execute("SELECT equity, position, fees FROM state WHERE id=1").fetchone()

    def update_state(self, equity, position, fees, score):
        with sqlite3.connect(self.db_name) as conn:
            conn.execute("UPDATE state SET equity=?, position=?, fees=? WHERE id=1", (equity, position, fees))
            conn.execute("INSERT INTO logs VALUES (?, ?, ?, ?)", 
                         (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), equity, position, score))

    def get_history(self, limit=150):
        with sqlite3.connect(self.db_name) as conn:
            return pd.read_sql_query(f"SELECT * FROM logs ORDER BY timestamp DESC LIMIT {limit}", conn).iloc[::-1]

class MacroQuantPro:
    def __init__(self):
        self.db = TradingDatabase(CONFIG["DB_NAME"])
        self.last_price = None
        self.signals_state = {"Trend": 0, "Mom": 0, "Curve": 0, "VIX": 0}

    def get_signals(self):
        symbols = [CONFIG["ASSET"], CONFIG["CASH"], CONFIG["VIX"], CONFIG["Y10"], CONFIG["Y3M"]]
        df = yf.download(symbols, period="2y", progress=False, auto_adjust=True)['Close']
        
        if df.empty: return None, None, None

        # 1. Signaux scalaires (item())
        curr_p = df[CONFIG["ASSET"]].iloc[-1].item()
        sma = df[CONFIG["ASSET"]].rolling(CONFIG["SMA_PERIOD"]).mean().iloc[-1].item()
        spy_mom = df[CONFIG["ASSET"]].pct_change(126).iloc[-1].item()
        bil_mom = df[CONFIG["CASH"]].pct_change(126).iloc[-1].item()
        spread = df[CONFIG["Y10"]].iloc[-1].item() - df[CONFIG["Y3M"]].iloc[-1].item()
        vix_val = df[CONFIG["VIX"]].iloc[-1].item()
        vix_sma = df[CONFIG["VIX"]].rolling(20).mean().iloc[-1].item()

        # 2. Logic Graduelle
        s1 = 1 if curr_p > sma else 0
        s2 = 1 if spy_mom > bil_mom else 0
        s3 = 1 if spread > 0 else 0
        s4 = 1 if vix_val < 35 and vix_val < (vix_sma * 1.25) else 0
        
        self.signals_state = {"Trend": s1, "Mom": s2, "Curve": s3, "VIX": s4}
        score = (s1 + s2 + s3 + s4) / 4.0

        # 3. Risk Scaling
        rets = df[CONFIG["ASSET"]].pct_change().dropna()
        vol_ann = (rets.tail(20).std() * np.sqrt(252)).item()
        scaling = (CONFIG["VOL_TARGET"] / vol_ann) if vol_ann > 0 else 0
        if (vol_ann / np.sqrt(12)) * scaling * 2.06 > CONFIG["ES_GUARD"]:
            scaling = CONFIG["ES_GUARD"] / ((vol_ann / np.sqrt(12)) * 2.06)
        
        target_w = float(min(scaling, 1.5) * score)
        return target_w, float(curr_p), float(df[CONFIG["CASH"]].pct_change().iloc[-1].item())

    def trading_loop(self):
        while True:
            try:
                equity, pos, fees = self.db.get_state()
                target_w, curr_p, cash_ret = self.get_signals()

                if target_w is not None:
                    # PnL Calculation
                    if self.last_price:
                        asset_pnl = (curr_p / self.last_price - 1) * pos
                        cash_pnl = (1 - pos) * cash_ret
                        equity *= (1 + asset_pnl + cash_pnl)

                    # Execution Simulator
                    diff = abs(target_w - pos)
                    if diff > 0.01:
                        cost = (equity * diff * CONFIG["FEES_TOTAL"])
                        equity -= cost
                        fees += cost
                        pos = target_w
                        print(f"!!! [TRADE] {datetime.now().strftime('%H:%M')} | New Exposure: {pos*100:.1f}%")

                    self.last_price = curr_p
                    self.db.update_state(equity, pos, fees, (sum(self.signals_state.values())/4))

            except Exception as e:
                print(f"Erreur Loop: {e}")
            
            time.sleep(CONFIG["LOOP_INTERVAL"])

# ==============================================================================
# 2. DASHBOARD DE MONITORING LIVE
# ==============================================================================
bot = MacroQuantPro()
threading.Thread(target=bot.trading_loop, daemon=True).start()

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

def animate(i):
    history = bot.db.get_history()
    if history.empty: return

    # Panel 1: Equity
    ax1.clear()
    ax1.plot(history['timestamp'], history['equity'], color='#1e3799', lw=2)
    ax1.set_title(f"LIVE AUDIT | Equity: {history['equity'].iloc[-1]:.2f}$", fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Panel 2: Exposure level
    ax2.clear()
    ax2.fill_between(range(len(history)), history['position']*100, color='teal', alpha=0.3)
    ax2.set_ylabel("Levier %")
    ax2.set_ylim(0, 160)

    # Panel 3: Health Check Signals
    ax3.clear()
    names = list(bot.signals_state.keys())
    values = list(bot.signals_state.values())
    colors = ['#27ae60' if v else '#e74c3c' for v in values]
    ax3.barh(names, [1]*4, color=colors, alpha=0.6)
    ax3.set_xlim(0, 1)
    ax3.set_title("System Health Check (Trend | Mom | Curve | VIX)")

    plt.tight_layout()

ani = animation.FuncAnimation(fig, animate, interval=10000)
plt.show()