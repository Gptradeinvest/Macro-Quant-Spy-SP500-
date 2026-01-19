import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
import sqlite3
import logging
import threading
import time
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Tuple, Dict, Optional, Any

# ==============================================================================
# 1. ENTERPRISE CONFIGURATION
# ==============================================================================
CONFIG = {
    "ASSETS": {
        "RISK": "SPY",
        "SAFE": "BIL",
        "VOL": "^VIX",
        "YIELD_10Y": "^TNX",
        "YIELD_3M": "^IRX"
    },
    "SYSTEM": {
        "DB_PATH": "production_trading.db",
        "LOG_PATH": "production_trading.log",
        "UPDATE_INTERVAL": 60,  # Seconds
        "HISTORY_WINDOW": "2y"
    },
    "RISK_MODEL": {
        "INITIAL_CAPITAL": 100_000.0,
        "VOL_TARGET": 0.12,
        "ES_LIMIT": 0.10,
        "SMA_PERIOD": 200,
        "TRANSACTION_COST_BPS": 0.0010
    }
}

# ==============================================================================
# 2. LOGGING INFRASTRUCTURE
# ==============================================================================
def setup_logging():
    logger = logging.getLogger("MacroQuantEngine")
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(threadName)s | %(message)s')
    
    # File Handler (Rotating)
    file_handler = RotatingFileHandler(CONFIG["SYSTEM"]["LOG_PATH"], maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

LOGGER = setup_logging()

# ==============================================================================
# 3. PERSISTENCE LAYER (DAO)
# ==============================================================================
class PersistenceManager:
    """Thread-safe SQLite Data Access Object."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_schema()

    def _get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _initialize_schema(self):
        with self.lock, self._get_connection() as conn:
            cursor = conn.cursor()
            # Portfolio State Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    equity REAL,
                    position_exposure REAL,
                    accumulated_fees REAL
                )
            """)
            # Signal Logs Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_logs (
                    timestamp TEXT,
                    score REAL,
                    trend INTEGER,
                    momentum INTEGER,
                    yield_curve INTEGER,
                    vol_regime INTEGER
                )
            """)
            # Initialize if empty
            cursor.execute("SELECT count(*) FROM portfolio_state")
            if cursor.fetchone()[0] == 0:
                init_equity = CONFIG["RISK_MODEL"]["INITIAL_CAPITAL"]
                cursor.execute(
                    "INSERT INTO portfolio_state (timestamp, equity, position_exposure, accumulated_fees) VALUES (?, ?, ?, ?)",
                    (datetime.now().isoformat(), init_equity, 0.0, 0.0)
                )
                LOGGER.info(f"Database initialized with capital: ${init_equity:,.2f}")

    def get_latest_state(self) -> Dict[str, Any]:
        with self.lock, self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM portfolio_state ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else {}

    def get_history(self, limit: int = 500) -> pd.DataFrame:
        with self.lock, self._get_connection() as conn:
            query = f"""
                SELECT p.timestamp, p.equity, p.position_exposure, s.score, 
                       s.trend, s.momentum, s.yield_curve, s.vol_regime
                FROM portfolio_state p
                LEFT JOIN signal_logs s ON p.timestamp = s.timestamp
                ORDER BY p.timestamp DESC LIMIT {limit}
            """
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            return df

    def update_state(self, equity: float, position: float, fees: float, signals: Dict[str, int]):
        ts = datetime.now().isoformat()
        score = sum(signals.values()) / 4.0
        
        with self.lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO portfolio_state (timestamp, equity, position_exposure, accumulated_fees) VALUES (?, ?, ?, ?)",
                (ts, equity, position, fees)
            )
            cursor.execute(
                "INSERT INTO signal_logs (timestamp, score, trend, momentum, yield_curve, vol_regime) VALUES (?, ?, ?, ?, ?, ?)",
                (ts, score, signals['Trend'], signals['Mom'], signals['Curve'], signals['VIX'])
            )

# ==============================================================================
# 4. BUSINESS LOGIC ENGINE
# ==============================================================================
class TradingEngine:
    def __init__(self, persistence: PersistenceManager):
        self.db = persistence
        self.last_price: Optional[float] = None
        self.running = False
        self._thread = None
        
    def _fetch_market_data(self) -> pd.DataFrame:
        tickers = list(CONFIG["ASSETS"].values())
        try:
            data = yf.download(tickers, period=CONFIG["SYSTEM"]["HISTORY_WINDOW"], progress=False, auto_adjust=True)
            
            # Robust MultiIndex handling
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.get_level_values(0):
                    df = data['Close']
                else:
                    df = data.iloc[:, :len(tickers)] # Fallback
            else:
                df = data
                
            # Rename columns to standardized keys
            inv_map = {v: k for k, v in CONFIG["ASSETS"].items()}
            df = df.rename(columns=inv_map)
            
            return df.ffill().dropna()
        except Exception as e:
            LOGGER.error(f"Market data fetch failed: {str(e)}")
            return pd.DataFrame()

    def _calculate_target(self, df: pd.DataFrame) -> Tuple[float, Dict[str, int]]:
        try:
            # Extract scalars securely
            risk_asset = df["RISK"]
            safe_asset = df["SAFE"]
            vol_index = df["VOL"]
            
            curr_price = float(risk_asset.iloc[-1])
            
            # Indicators
            sma_200 = float(risk_asset.rolling(CONFIG["RISK_MODEL"]["SMA_PERIOD"]).mean().iloc[-1])
            mom_risk = float(risk_asset.pct_change(126).iloc[-1])
            mom_safe = float(safe_asset.pct_change(126).iloc[-1])
            
            spread = float(df["YIELD_10Y"].iloc[-1] - df["YIELD_3M"].iloc[-1])
            
            vix_val = float(vol_index.iloc[-1])
            vix_ma = float(vol_index.rolling(20).mean().iloc[-1])
            
            # Signal Logic
            signals = {
                "Trend": 1 if curr_price > sma_200 else 0,
                "Mom": 1 if mom_risk > mom_safe else 0,
                "Curve": 1 if spread > 0 else 0,
                "VIX": 1 if (vix_val < 35) and (vix_val < vix_ma * 1.25) else 0
            }
            
            score = sum(signals.values()) / 4.0
            
            # Volatility Targeting
            returns = risk_asset.pct_change()
            vol_ann = float(returns.tail(60).std() * np.sqrt(252))
            
            if vol_ann < 0.001: return 0.0, signals
            
            base_lev = CONFIG["RISK_MODEL"]["VOL_TARGET"] / vol_ann
            
            # Expected Shortfall Guardrail
            monthly_vol = vol_ann / np.sqrt(12)
            projected_es = monthly_vol * base_lev * 2.06
            
            risk_scaler = 1.0
            if projected_es > CONFIG["RISK_MODEL"]["ES_LIMIT"]:
                risk_scaler = CONFIG["RISK_MODEL"]["ES_LIMIT"] / projected_es
                
            final_weight = min(base_lev * risk_scaler * score, 1.5)
            
            return final_weight, signals
            
        except Exception as e:
            LOGGER.error(f"Signal calculation error: {str(e)}")
            return 0.0, {}

    def _execution_cycle(self):
        LOGGER.info("Execution thread started.")
        while self.running:
            try:
                start_time = time.time()
                
                # 1. Data Acquisition
                df = self._fetch_market_data()
                if df.empty:
                    time.sleep(10)
                    continue
                    
                curr_price = float(df["RISK"].iloc[-1])
                
                # 2. Logic Processing
                target_w, signals = self._calculate_target(df)
                
                # 3. Portfolio State Management
                state = self.db.get_latest_state()
                equity = state.get('equity', CONFIG["RISK_MODEL"]["INITIAL_CAPITAL"])
                current_pos = state.get('position_exposure', 0.0)
                fees_acc = state.get('accumulated_fees', 0.0)
                
                # 4. PnL Attribution (Mark-to-Market)
                if self.last_price is not None:
                    ret_asset = (curr_price - self.last_price) / self.last_price
                    # Safe asset return approx (daily / 1440 mins * interval)
                    ret_safe = (float(df["SAFE"].pct_change().iloc[-1]) / 24 / 60) * (CONFIG["SYSTEM"]["UPDATE_INTERVAL"] / 60)
                    
                    pnl = equity * (current_pos * ret_asset + (1 - current_pos) * ret_safe)
                    equity += pnl
                
                # Initialize reference price on first run
                else:
                    self.last_price = curr_price
                    LOGGER.info(f"Reference price initialized: ${curr_price:.2f}")

                # 5. Execution Logic (Simulation)
                delta_w = abs(target_w - current_pos)
                if delta_w > 0.005: # 0.5% threshold to avoid noise
                    txn_cost = equity * delta_w * CONFIG["RISK_MODEL"]["TRANSACTION_COST_BPS"]
                    equity -= txn_cost
                    fees_acc += txn_cost
                    LOGGER.info(f"REBALANCE | Old: {current_pos:.1%} -> New: {target_w:.1%} | Equity: ${equity:,.2f}")
                    current_pos = target_w
                
                self.last_price = curr_price
                
                # 6. Persist State
                self.db.update_state(equity, current_pos, fees_acc, signals)
                
                # Interval Management
                elapsed = time.time() - start_time
                sleep_time = max(0, CONFIG["SYSTEM"]["UPDATE_INTERVAL"] - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                LOGGER.critical(f"Cycle failure: {str(e)}", exc_info=True)
                time.sleep(30) # Cool down on error

    def start(self):
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._execution_cycle, name="ExecEngine")
            self._thread.daemon = True
            self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()
        LOGGER.info("Engine stopped.")

# ==============================================================================
# 5. VISUALIZATION DASHBOARD (VIEW)
# ==============================================================================
class Dashboard:
    def __init__(self, persistence: PersistenceManager):
        self.db = persistence
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(12, 8))
        self.gs = self.fig.add_gridspec(3, 1, height_ratios=[2, 1, 0.5])
        
        self.ax_eq = self.fig.add_subplot(self.gs[0])
        self.ax_pos = self.fig.add_subplot(self.gs[1])
        self.ax_sig = self.fig.add_subplot(self.gs[2])

    def _update(self, frame):
        df = self.db.get_history()
        if df.empty: return

        # 1. Equity Curve
        self.ax_eq.clear()
        latest_eq = df['equity'].iloc[-1]
        self.ax_eq.plot(df['timestamp'], df['equity'], color='#2ecc71', lw=1.5)
        self.ax_eq.set_title(f"NAV: ${latest_eq:,.2f}", fontsize=12, fontweight='bold', color='white')
        self.ax_eq.grid(True, alpha=0.15)
        self.ax_eq.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # 2. Exposure Area
        self.ax_pos.clear()
        self.ax_pos.fill_between(df['timestamp'], df['position_exposure'], color='#3498db', alpha=0.3)
        self.ax_pos.plot(df['timestamp'], df['position_exposure'], color='#3498db', lw=1)
        self.ax_pos.axhline(1.0, color='#e74c3c', linestyle=':', alpha=0.7)
        self.ax_pos.set_ylabel("Exposure", fontsize=9)
        self.ax_pos.set_ylim(0, 1.6)
        
        # 3. Signals Heatmap (Latest State)
        self.ax_sig.clear()
        if 'trend' in df.columns:
            latest = df.iloc[-1]
            metrics = ['Trend', 'Momentum', 'Yield Curve', 'Vol Regime']
            values = [latest['trend'], latest['momentum'], latest['yield_curve'], latest['vol_regime']]
            colors = ['#27ae60' if v==1 else '#c0392b' for v in values]
            
            self.ax_sig.barh(metrics, [1]*4, color=colors, edgecolor='#2c3e50')
            self.ax_sig.set_xlim(0, 1)
            self.ax_sig.set_xticks([])
            self.ax_sig.invert_yaxis() # Top down
            self.ax_sig.set_title(f"Signal Score: {latest['score']:.2f}", fontsize=10)

    def launch(self):
        anim = animation.FuncAnimation(self.fig, self._update, interval=5000, cache_frame_data=False)
        plt.show()

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    LOGGER.info("Starting Institutional Macro-Quant System...")
    
    # Initialize components
    persistence = PersistenceManager(CONFIG["SYSTEM"]["DB_PATH"])
    engine = TradingEngine(persistence)
    dashboard = Dashboard(persistence)
    
    try:
        # Start Backend
        engine.start()
        
        # Start Frontend (Main Thread)
        dashboard.launch()
        
    except KeyboardInterrupt:
        LOGGER.info("Shutdown signal received.")
    finally:
        engine.stop()
