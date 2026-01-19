import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import logging
import sys
from typing import Tuple, Dict, Optional

# ==============================================================================
# SYSTEM CONFIGURATION
# ==============================================================================
# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('strategy_execution.log')
    ]
)
logger = logging.getLogger(__name__)

CONFIG = {
    "ASSETS": {
        "RISK": "SPY",
        "SAFE": "BIL",
        "VOL": "^VIX",
        "YIELD_10Y": "^TNX",
        "YIELD_3M": "^IRX"
    },
    "PARAMS": {
        "START_DATE": "2007-01-01",
        "LOOKBACK_BUFFER": "2000-01-01",
        "INITIAL_CAPITAL": 100_000,
        "TC_BPS": 0.0005,          # 5 basis points transaction cost
        "BENCHMARK_FEE": 0.0003    # 3 basis points annual fee for SPY
    },
    "RISK_MGMT": {
        "VOL_TARGET": 0.12,        # Annualized volatility target
        "VOL_CAP": 0.40,           # Volatility cap for sizing calculations
        "ES_CONFIDENCE": 0.95,     # Confidence level for Expected Shortfall
        "ES_LIMIT": 0.15,          # Maximum allowable ES
        "STUDENT_T_FACTOR": 2.5    # Multiplier for fat-tail adjustment approx
    },
    "VALIDATION": {
        "N_TRIALS": 256,           # Estimated number of independent trials
        "BOOTSTRAP_SAMPLES": 1000,
        "SEED": 42
    }
}

# ==============================================================================
# 1. STATISTICAL ENGINE (AFML STANDARDS)
# ==============================================================================
def get_forensic_stats(returns: pd.Series, n_trials: int) -> Tuple[float, float, float, float]:
    """
    Calculates Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR)
    using Newey-West adjustment for autocorrelation.
    """
    rets = returns.dropna()
    n_obs = len(rets)
    if n_obs < 252:
        return 0.0, 0.0, 0.0, float(n_obs)

    # Moments
    mean, std = rets.mean(), rets.std()
    skew, kurt = rets.skew(), rets.kurtosis()
    sharpe = (mean * 252) / (std * np.sqrt(252))

    # Newey-West Autocorrelation Adjustment
    max_lag = 10
    acf_sum = 0.0
    for lag in range(1, min(max_lag + 1, n_obs // 2)):
        acf_val = rets.autocorr(lag)
        if not np.isnan(acf_val):
            # Linear decay weight
            weight = 1 - (lag / (max_lag + 1))
            acf_sum += weight * acf_val
            
    correction_factor = max(1 + 2 * acf_sum, 1.0)
    n_effective = max(n_obs / correction_factor, 252)

    # PSR Calculation
    sigma_sharpe = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / (n_effective - 1))
    psr = stats.norm.cdf(sharpe / sigma_sharpe)

    # DSR Calculation
    expected_max_sharpe = sigma_sharpe * np.sqrt(2 * np.log(n_trials))
    dsr = stats.norm.cdf((sharpe - expected_max_sharpe) / sigma_sharpe)

    return psr, dsr, sharpe, n_effective

def bootstrap_sharpe_ci(returns: pd.Series, n_samples: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Generates 95% Confidence Interval for Sharpe Ratio via Bootstrapping."""
    rng = np.random.default_rng(seed)
    values = returns.values
    sharpes = []
    
    for _ in range(n_samples):
        sample = rng.choice(values, size=len(values), replace=True)
        sample_std = np.std(sample)
        if sample_std > 1e-6:
            sharpes.append(np.mean(sample) / sample_std * np.sqrt(252))
            
    return np.percentile(sharpes, 2.5), np.percentile(sharpes, 97.5)

# ==============================================================================
# 2. DATA INGESTION LAYER
# ==============================================================================
def fetch_market_data() -> pd.DataFrame:
    """Fetches and normalizes data from yfinance."""
    logger.info("Initializing Data Ingestion Pipeline...")
    tickers = list(CONFIG["ASSETS"].values())
    
    try:
        raw_data = yf.download(
            tickers, 
            start=CONFIG["PARAMS"]["LOOKBACK_BUFFER"], 
            auto_adjust=True, 
            progress=False
        )
    except Exception as e:
        logger.critical(f"Data download failed: {e}")
        sys.exit(1)

    # Handle MultiIndex Logic Robustly
    if isinstance(raw_data.columns, pd.MultiIndex):
        level_0 = raw_data.columns.get_level_values(0)
        if 'Close' in level_0:
            df = raw_data['Close'].copy()
        elif 'Adj Close' in level_0:
            df = raw_data['Adj Close'].copy()
        else:
            logger.warning("Unknown MultiIndex structure. Attempting flat extraction.")
            try:
                df = raw_data.xs(raw_data.columns.levels[0][0], axis=1, level=0)
            except Exception:
                # Fallback: assume column order matches ticker request
                df = raw_data.iloc[:, :len(tickers)]
                df.columns = tickers
    else:
        df = raw_data.copy()

    # Rename columns to internal standard
    inv_map = {v: k for k, v in CONFIG["ASSETS"].items()}
    # yfinance sometimes returns columns not in the exact order or missing
    # We map based on intersection
    rename_dict = {}
    for col in df.columns:
        if col in inv_map:
            rename_dict[col] = inv_map[col]
            
    df = df.rename(columns=rename_dict)
    
    # Forward fill and drop NaN
    df = df.ffill().dropna()
    logger.info(f"Data Pipeline loaded successfully. Rows: {len(df)}")
    return df

# ==============================================================================
# 3. STRATEGY ENGINE
# ==============================================================================
def run_strategy() -> pd.DataFrame:
    df = fetch_market_data()
    
    # 3.1 Feature Engineering (Vectorized)
    # CRITICAL: All inputs must be lagged by 1 day to prevent look-ahead bias
    logger.info("Computing Signals and Indicators...")
    
    risk_asset = df["RISK"]
    vix = df["VOL"]
    
    # Base Returns
    risk_rets = risk_asset.pct_change()
    safe_rets = df["SAFE"].pct_change().fillna(0)
    
    # Lagged Indicators (T-1 Close)
    sma_200 = risk_asset.rolling(200).mean().shift(1)
    vix_ma_20 = vix.rolling(20).mean().shift(1)
    
    # Momentum (126d ~ 6 months)
    mom_risk = risk_asset.pct_change(126).shift(1)
    mom_safe = df["SAFE"].pct_change(126).shift(1)
    
    # Yield Curve
    yield_10y_lag = df["YIELD_10Y"].shift(1)
    yield_3m_lag = df["YIELD_3M"].shift(1)
    vix_lag = vix.shift(1)

    # 3.2 Signal Generation
    signals = pd.DataFrame(index=df.index)
    
    # Rule 1: Trend (Price > SMA200)
    # Note: risk_asset.shift(1) compares yesterday's price to yesterday's SMA (which is correct)
    signals['Trend'] = (risk_asset.shift(1) > sma_200).astype(int)
    
    # Rule 2: Relative Momentum
    signals['Mom'] = (mom_risk > mom_safe).astype(int)
    
    # Rule 3: Yield Curve (Inversion check)
    signals['Curve'] = (yield_10y_lag > yield_3m_lag).astype(int)
    
    # Rule 4: Volatility Regime
    signals['VIX'] = ((vix_lag < 35) & (vix_lag < vix_ma_20 * 1.25)).astype(int)
    
    # Composite Score
    signals['Score'] = signals.mean(axis=1)
    
    # 3.3 Risk Management & Execution
    logger.info("Applying Risk Management Rules...")
    bt = pd.DataFrame(index=df.index)
    bt['Signal'] = signals['Score']
    
    # Volatility Calculation (Shifted for T-1 availability)
    vol_60d = risk_rets.rolling(60).std() * np.sqrt(252)
    
    # Volatility Dissociation Logic
    # 1. Risk Volatility: Uncapped, used for Guardrails
    bt['Vol_Risk'] = vol_60d.shift(1).ffill()
    
    # 2. Sizing Volatility: Capped to prevent excessive deleveraging during high vol events
    bt['Vol_Size'] = bt['Vol_Risk'].clip(upper=CONFIG["RISK_MGMT"]["VOL_CAP"])
    
    # Leverage Calculation (Inverse Volatility)
    # Avoid division by zero
    bt['Lev_Base'] = np.where(
        bt['Vol_Size'] > 0.001, 
        CONFIG["RISK_MGMT"]["VOL_TARGET"] / bt['Vol_Size'], 
        0.0
    )
    
    # Expected Shortfall (ES) Guardrail
    # Uses Student-t approximation factor (2.5) instead of Gaussian (2.06)
    projected_es = bt['Vol_Risk'] * bt['Lev_Base'] * CONFIG["RISK_MGMT"]["STUDENT_T_FACTOR"]
    
    es_scaler = np.where(
        (projected_es > CONFIG["RISK_MGMT"]["ES_LIMIT"]) & (projected_es > 0),
        CONFIG["RISK_MGMT"]["ES_LIMIT"] / projected_es,
        1.0
    )
    
    # Final Weight Calculation
    # Hard cap at 1.5x leverage
    bt['Target_W'] = (bt['Lev_Base'] * es_scaler * bt['Signal']).clip(0, 1.5)
    
    # Transaction Cost Analysis
    prev_w = bt['Target_W'].shift(1).fillna(0)
    avg_exposure = (bt['Target_W'] + prev_w) / 2
    turnover = bt['Target_W'].diff().abs().fillna(0)
    
    bt['Costs'] = turnover * avg_exposure * CONFIG["PARAMS"]["TC_BPS"]
    
    # Net PnL Calculation
    bt['Asset_Ret'] = risk_rets
    bt['Safe_Ret'] = safe_rets
    bt['Strat_Net'] = (bt['Target_W'] * bt['Asset_Ret']) + \
                      ((1 - bt['Target_W']) * bt['Safe_Ret']) - \
                      bt['Costs']
                      
    return bt, signals

# ==============================================================================
# 4. VALIDATION & REPORTING
# ==============================================================================
def perform_audit(bt_data: pd.DataFrame, signals: pd.DataFrame):
    logger.info("Starting Forensic Audit...")
    
    # Filter for Walk-Forward Analysis period
    wfa_start = CONFIG["PARAMS"]["START_DATE"]
    wfa = bt_data.loc[wfa_start:].dropna().copy()
    
    # Equity Curves
    initial_cap = CONFIG["PARAMS"]["INITIAL_CAPITAL"]
    wfa['Equity'] = (1 + wfa['Strat_Net']).cumprod() * initial_cap
    
    # Benchmark with fees
    bench_daily_fee = CONFIG["PARAMS"]["BENCHMARK_FEE"] / 252
    wfa['Benchmark'] = (1 + wfa['Asset_Ret'] - bench_daily_fee).cumprod() * initial_cap
    
    # 4.1 Statistics
    years = len(wfa) / 252
    cagr = (wfa['Equity'].iloc[-1] / wfa['Equity'].iloc[0]) ** (1/years) - 1
    max_dd = (wfa['Equity'] / wfa['Equity'].cummax() - 1).min()
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0.0
    
    psr, dsr, sr, n_eff = get_forensic_stats(wfa['Strat_Net'], CONFIG["VALIDATION"]["N_TRIALS"])
    ci_low, ci_high = bootstrap_sharpe_ci(
        wfa['Strat_Net'], 
        CONFIG["VALIDATION"]["BOOTSTRAP_SAMPLES"], 
        CONFIG["VALIDATION"]["SEED"]
    )
    
    # 4.2 Kill Switch Check (Degradation)
    rolling_sr = wfa['Strat_Net'].rolling(252).apply(
        lambda x: x.mean()/x.std()*np.sqrt(252) if x.std() > 1e-6 else 0
    )
    degraded_days = (rolling_sr < 0).rolling(504).sum()
    kill_switch_active = (degraded_days > 403).any() # >80% of days in 2 years
    
    # 4.3 Stress Tests
    crises = {
        'GFC': ('2008-01-01', '2009-06-30'),
        'COVID': ('2020-02-15', '2020-04-15'),
        'INFLATION': ('2022-01-01', '2022-12-31')
    }
    
    stress_results = []
    gfc_passed = False
    
    for name, (s, e) in crises.items():
        if s < wfa.index[0].strftime('%Y-%m-%d'): continue
        sub = wfa.loc[s:e]
        if sub.empty: continue
        
        dd_strat = (sub['Equity'] / sub['Equity'].cummax() - 1).min()
        dd_bench = (sub['Benchmark'] / sub['Benchmark'].cummax() - 1).min()
        
        # Avoid division by zero if benchmark flat
        if dd_bench > -0.01: rel_dd = 1.0
        else: rel_dd = dd_strat / dd_bench
            
        passed = (rel_dd < 0.70) and (dd_strat > -0.25)
        if name == 'GFC': gfc_passed = passed
        
        stress_results.append({
            "Period": name,
            "Strat_DD": dd_strat,
            "Bench_DD": dd_bench,
            "Rel_DD": rel_dd,
            "Passed": passed
        })

    # 4.4 Console Report
    print("\n" + "="*80)
    print(f"STRATEGY AUDIT REPORT | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print(f"{'METRIC':<25} | {'VALUE':<15} | {'THRESHOLD/STATUS':<20}")
    print("-" * 65)
    print(f"{'CAGR':<25} | {cagr:<15.2%} | -")
    print(f"{'Sharpe Ratio':<25} | {sr:<15.2f} | 95% CI [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"{'Max Drawdown':<25} | {max_dd:<15.2%} | -")
    print(f"{'Calmar Ratio':<25} | {calmar:<15.2f} | -")
    print(f"{'PSR (Prob. Sharpe)':<25} | {psr:<15.2%} | Target > 95%")
    print(f"{'DSR (Deflated Sharpe)':<25} | {dsr:<15.2%} | Target > 50%")
    print(f"{'Effective N (Days)':<25} | {int(n_eff):<15} | -")
    print("-" * 65)
    
    print("\nSTRESS TEST RESULTS")
    print(f"{'EVENT':<12} | {'STRAT DD':<10} | {'BENCH DD':<10} | {'REL RISK':<10} | {'STATUS'}")
    print("-" * 65)
    for res in stress_results:
        status = "PASS" if res["Passed"] else "FAIL"
        print(f"{res['Period']:<12} | {res['Strat_DD']:<10.2%} | {res['Bench_DD']:<10.2%} | {res['Rel_DD']:<10.2f} | {status}")
        
    print("\nOPERATIONAL METRICS")
    print(f"Annual Turnover: {wfa['Target_W'].diff().abs().sum() / years:.2f}x")
    print(f"Kill Switch Status: {'ALERT (TRIGGERED)' if kill_switch_active else 'NOMINAL'}")
    print("="*80 + "\n")

    # 4.5 Visualization
    plot_dashboard(wfa, signals, rolling_sr, cagr, sr, max_dd, psr, dsr, n_eff, kill_switch_active)

def plot_dashboard(wfa, signals, rolling_sr, cagr, sr, max_dd, psr, dsr, n_eff, ks_active):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(4, 2, height_ratios=[2, 1, 1, 1])
    
    fig.suptitle("INSTITUTIONAL MACRO-QUANT STRATEGY | PERFORMANCE AUDIT", fontsize=14, weight='bold')

    # 1. Equity Curve (Log)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(wfa.index, wfa['Equity'], color='#003049', lw=1.5, label='Strategy (Net)')
    ax1.plot(wfa.index, wfa['Benchmark'], color='gray', lw=1, ls='--', alpha=0.6, label='Benchmark (Net)')
    ax1.set_yscale('log')
    ax1.set_ylabel("Capital (Log Scale)")
    ax1.legend(loc='upper left')
    ax1.set_title(f"Cumulative Performance | CAGR: {cagr:.2%} | Sharpe: {sr:.2f}")

    # 2. Drawdowns
    ax2 = fig.add_subplot(gs[1, 0])
    dd_series = wfa['Equity'] / wfa['Equity'].cummax() - 1
    ax2.fill_between(dd_series.index, dd_series, 0, color='#d62828', alpha=0.3)
    ax2.set_title(f"Drawdown Profile | Max DD: {max_dd:.2%}")

    # 3. Leverage / Exposure
    ax3 = fig.add_subplot(gs[1, 1])
    wfa['Target_W'].plot(ax=ax3, color='#f77f00', lw=1, alpha=0.8)
    ax3.fill_between(wfa.index, 0, wfa['Target_W'], alpha=0.1, color='#f77f00')
    ax3.axhline(1.0, color='black', ls=':', alpha=0.5)
    ax3.set_ylim(0, 1.6)
    ax3.set_title("Dynamic Exposure (Gross Leverage)")

    # 4. Regime Stackplot
    ax4 = fig.add_subplot(gs[2, :])
    sig_data = signals[['Trend', 'Mom', 'Curve', 'VIX']].loc[wfa.index]
    ax4.stackplot(sig_data.index, sig_data.T, labels=sig_data.columns, 
                  colors=['#264653', '#2a9d8f', '#e9c46a', '#e76f51'], alpha=0.8)
    ax4.legend(loc='upper left', ncol=4)
    ax4.set_title("Signal Regime Allocation")
    ax4.set_ylim(0, 4.2)
    
    # 5. Statistical Summary Box
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    
    report_text = (
        f"STATISTICAL VALIDATION REPORT\n"
        f"-----------------------------\n"
        f"Probabilistic Sharpe (PSR): {psr:.2%} (Pass > 95%)\n"
        f"Deflated Sharpe (DSR):      {dsr:.2%} (Pass > 50%)\n"
        f"Effective N (Indep. Obs):   {int(n_eff)}\n\n"
        f"RISK CONTROL STATUS\n"
        f"-----------------------------\n"
        f"Structure Health:           {'DEGRADED' if ks_active else 'STABLE'}\n"
        f"Volatility Cap (Sizing):    {CONFIG['RISK_MGMT']['VOL_CAP']:.0%}\n"
        f"ES Guardrail (95% Conf):    Active (Student-t Factor {CONFIG['RISK_MGMT']['STUDENT_T_FACTOR']})"
    )
    
    ax5.text(0.5, 0.5, report_text, fontsize=10, family='monospace', 
             va='center', ha='center', bbox=dict(boxstyle="round,pad=1", facecolor="#f8f9fa", edgecolor="#ced4da"))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        bt_data, signals = run_strategy()
        perform_audit(bt_data, signals)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.critical(f"Fatal Error: {e}", exc_info=True)
