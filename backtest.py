import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats

# ==============================================================================
# 1. CONFIGURATION DU MANDAT
# ==============================================================================
CONFIG = {
    "ASSET": "SPY", "CASH": "BIL", "VIX": "^VIX", "Y10": "^TNX", "Y3M": "^IRX",
    "START_DATE": "2011-01-01",
    "VOL_TARGET": 0.12,
    "ES_GUARD": 0.10,
    "INITIAL_CAPITAL": 100000,
    "FEES_TOTAL": 0.0010, # 0.1% (Commissions + Slippage)
    "N_TRIALS": 500       # Pour le calcul du DSR
}

def calculate_afml_metrics(returns, benchmark_sr=0, n_trials=500):
    """Calcul du PSR et DSR selon López de Prado"""
    n = len(returns)
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    skew = returns.skew()
    kurt = returns.kurtosis()
    
    # PSR
    sigma_sr = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / (n - 1))
    psr = stats.norm.cdf((sharpe - benchmark_sr) / sigma_sr)
    
    # DSR (Ajusté pour le nombre de tests)
    exp_max_sr = np.sqrt(2 * np.log(n_trials)) * np.sqrt(0.07) # 0.07 = variance estimée des Sharpes
    dsr = stats.norm.cdf((sharpe - exp_max_sr) / sigma_sr)
    
    return psr, dsr, sharpe

# ==============================================================================
# 2. PRÉPARATION DES DONNÉES
# ==============================================================================
print("--> Téléchargement des flux 15 ans...")
symbols = [CONFIG["ASSET"], CONFIG["CASH"], CONFIG["VIX"], CONFIG["Y10"], CONFIG["Y3M"]]
df = yf.download(symbols, start="2010-01-01", auto_adjust=True)['Close'].dropna()

spy, bil = df[CONFIG["ASSET"]], df[CONFIG["CASH"]]
vix, y10, y3 = df[CONFIG["VIX"]], df[CONFIG["Y10"]], df[CONFIG["Y3M"]]

spy_rets = spy.pct_change().dropna()
bil_rets = bil.pct_change().dropna()
sma200 = spy.rolling(200).mean()
spy_mom = spy.pct_change(126) # Momentum 6 mois
bil_mom = bil.pct_change(126)
spy_vol = spy_rets.rolling(20).std() * np.sqrt(252)

# ==============================================================================
# 3. MOTEUR DE BACKTEST (LOGIQUE GRADUELLE)
# ==============================================================================
dates = spy_rets.loc[CONFIG["START_DATE"]:].index
equity, cur_w = [CONFIG["INITIAL_CAPITAL"]], 0.0
rets_list = []

print(f"--> Simulation sur {len(dates)} séances...")

for d in dates:
    # --- CALCUL DES SIGNAUX BINAIRES ---
    s1 = 1 if spy.loc[d] > sma200.loc[d] else 0                           # Trend
    s2 = 1 if spy_mom.loc[d] > bil_mom.loc[d] else 0                     # Momentum
    s3 = 1 if (y10.loc[d] - y3.loc[d]) > 0 else 0                        # Macro (Yield Curve)
    s4 = 1 if vix.loc[d] < 35 and vix.loc[d] < (vix.rolling(20).mean().loc[d] * 1.25) else 0 # VIX
    
    # Score de conviction (0.0, 0.25, 0.50, 0.75, 1.0)
    conviction_score = (s1 + s2 + s3 + s4) / 4.0
    
    # --- RISK SCALING ---
    vol = spy_vol.loc[d]
    scaling = CONFIG["VOL_TARGET"] / vol if vol > 0 else 0
    # ES Guard 10%
    if (vol / np.sqrt(12)) * scaling * 2.06 > CONFIG["ES_GUARD"]:
        scaling = CONFIG["ES_GUARD"] / ((vol / np.sqrt(12)) * 2.06)
    
    # Exposition Cible
    target_w = min(scaling, 1.5) * conviction_score
    
    # --- PNL ET FRAIS ---
    diff = abs(target_w - cur_w)
    trade_cost = (equity[-1] * diff * CONFIG["FEES_TOTAL"]) if diff > 0.01 else 0
    
    day_ret = (spy_rets.loc[d] * target_w) + (1 - target_w) * bil_rets.loc[d]
    equity.append((equity[-1] * (1 + day_ret)) - trade_cost)
    rets_list.append(day_ret)
    cur_w = target_w

# ==============================================================================
# 4. ANALYSE ET DASHBOARD
# ==============================================================================
s_eq = pd.Series(equity[1:], index=dates)
s_ret = pd.Series(rets_list, index=dates)
dd = (s_eq / s_eq.cummax() - 1)
psr, dsr, sharpe = calculate_afml_metrics(s_ret, n_trials=CONFIG["N_TRIALS"])

# Graphiques
plt.style.use('default')
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(3, 3, figure=fig)

# Equity Curve
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(s_eq, color='#1e3799', lw=2)
ax1.set_title("STRATEGY PERFORMANCE : GRADUAL MACRO-QUANT SPY", fontsize=15, fontweight='bold', loc='left')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Metrics Table
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
stats_data = [
    ["Ann. Return", f"{(s_eq.iloc[-1]/s_eq.iloc[0])**(252/len(s_eq))-1:.2%}"],
    ["Sharpe Ratio", f"{sharpe:.2f}"],
    ["PSR", f"{psr*100:.2f}%"],
    ["DSR (Robustness)", f"{dsr*100:.2f}%"],
    ["Max Drawdown", f"{dd.min():.2%}"],
    ["Avg Exposure", f"{cur_w*100:.1f}%"]
]
ax2.table(cellText=stats_data, colLabels=["Metric", "Value"], loc='center', cellLoc='left').scale(1, 3)

# Monthly Attribution
ax3 = fig.add_subplot(gs[1, :2])
m_rets = s_ret.resample('ME').apply(lambda x: (1+x).prod()-1)
matrix = m_rets.groupby([m_rets.index.year, m_rets.index.month]).first().unstack() * 100
sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlGnBu", center=0, ax=ax3, cbar=False)

# Drawdown Analysis
ax4 = fig.add_subplot(gs[2, :2])
ax4.fill_between(dd.index, dd.values*100, 0, color='#e74c3c', alpha=0.3)
ax4.set_title("UNDERWATER DRAWDOWN PROFILE (%)")

plt.tight_layout()
plt.show()