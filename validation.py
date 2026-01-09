import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats

# ==============================================================================
# CONFIGURATION : AUDIT DE VALIDATION INSTITUTIONNEL (15 ANS)
# ==============================================================================
CONFIG = {
    "ASSET": "SPY", "CASH": "BIL", "VIX": "^VIX", "Y10": "^TNX", "Y3M": "^IRX",
    "START_WFA": "2011-01-01",
    "VOL_TARGET": 0.12, "ES_GUARD": 0.10,
    "N_TRIALS": 500,  # Calibré pour une stratégie multi-facteurs
    "MC_SIMS": 100000,
    "INITIAL_CAPITAL": 100000
}

# --- MÉTRIQUES AFML (LÓPEZ DE PRADO) ---
def get_afml_stats(returns, n_trials=500):
    n = len(returns)
    sr = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    skew, kurt = returns.skew(), returns.kurtosis()
    # PSR
    sigma_sr = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
    psr = stats.norm.cdf(sr / sigma_sr)
    # DSR
    exp_max_sr = np.sqrt(2 * np.log(n_trials)) * np.sqrt(0.07)
    dsr = stats.norm.cdf((sr - exp_max_sr) / sigma_sr)
    return psr, dsr, sr

# 1. ACQUISITION DATA
print("--> Validation : Chargement de l'historique 2010-2026...")
symbols = [CONFIG["ASSET"], CONFIG["CASH"], CONFIG["VIX"], CONFIG["Y10"], CONFIG["Y3M"]]
df = yf.download(symbols, start="2010-01-01", auto_adjust=True)['Close'].dropna()

spy, bil = df[CONFIG["ASSET"]], df[CONFIG["CASH"]]
vix, y10, y3 = df[CONFIG["VIX"]], df[CONFIG["Y10"]], df[CONFIG["Y3M"]]
spy_rets, bil_rets = spy.pct_change().dropna(), bil.pct_change().dropna()
sma200, spy_vol = spy.rolling(200).mean(), spy_rets.rolling(20).std() * np.sqrt(252)
spy_mom, bil_mom = spy.pct_change(126), bil.pct_change(126)

# Benchmark Buy & Hold pour comparaison
spy_bh_equity = (1 + spy_rets.loc[CONFIG["START_WFA"]:]).cumprod() * CONFIG["INITIAL_CAPITAL"]

# 2. MOTEUR WALK-FORWARD (GRADUEL)
dates = spy_rets.loc[CONFIG["START_WFA"]:].index
equity, rets_list, pos_list, score_list = [CONFIG["INITIAL_CAPITAL"]], [], [], []

for d in dates:
    # Signaux Macro Graduels
    s1 = 1 if spy.loc[d] > sma200.loc[d] else 0
    s2 = 1 if spy_mom.loc[d] > bil_mom.loc[d] else 0
    s3 = 1 if (y10.loc[d] - y3.loc[d]) > 0 else 0
    s4 = 1 if vix.loc[d] < 35 and vix.loc[d] < (vix.rolling(20).mean().loc[d] * 1.25) else 0
    
    score = (s1 + s2 + s3 + s4) / 4.0
    
    # Risk Scaling
    vol = spy_vol.loc[d]
    scaling = (CONFIG["VOL_TARGET"] / vol) if vol > 0 else 0
    if (vol/np.sqrt(12)) * scaling * 2.06 > CONFIG["ES_GUARD"]:
        scaling = CONFIG["ES_GUARD"] / ((vol/np.sqrt(12)) * 2.06)
    
    target_w = min(scaling, 1.5) * score
    day_ret = (spy_rets.loc[d] * target_w) + (1 - target_w)*bil_rets.loc[d]
    
    rets_list.append(day_ret)
    pos_list.append(target_w)
    score_list.append(score)
    equity.append(equity[-1] * (1 + day_ret))

s_eq = pd.Series(equity[1:], index=dates)
s_ret = pd.Series(rets_list, index=dates)
s_pos = pd.Series(pos_list, index=dates)
s_score = pd.Series(score_list, index=dates)

# 3. CALCUL DES MÉTRIQUES
psr, dsr, sr = get_afml_stats(s_ret, CONFIG["N_TRIALS"])
ann_ret = (s_eq.iloc[-1]/s_eq.iloc[0])**(252/len(s_eq)) - 1
max_dd = (s_eq/s_eq.cummax()-1).min()
calmar = abs(ann_ret / max_dd)

# ==============================================================================
# 4. DASHBOARD DE VALIDATION ULTIME (VISUALISATIONS MISES À JOUR)
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'strat': '#005b96', 'bench': '#b3cde0', 'pos': '#03396c', 'green': '#28a745', 'red': '#dc3545'}
fig = plt.figure(figsize=(24, 18))
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.5, 1, 1])
fig.suptitle("AUDIT DE VALIDATION INSTITUTIONNEL : MACRO-QUANT SPY (15 ANS WFA)", fontsize=18, fontweight='bold', y=0.95)

# --- ROW 1 : PERFORMANCE & SCORECARD ---

# A. Walk-Forward Equity vs Benchmark
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(s_eq, color=colors['strat'], lw=2.5, label='Macro-Quant Strategy (Net)')
ax1.plot(spy_bh_equity, color='gray', lw=1.5, ls='--', alpha=0.6, label='SPY Buy & Hold')
ax1.set_yscale('log')
ax1.set_title("PERFORMANCE COMPARATIVE (LOG SCALE)", fontsize=14, fontweight='bold', loc='left')
ax1.legend(loc='upper left', frameon=True)
ax1.set_ylabel("Capital ($)")

# Annotations des crises majeures
crises = [('2015-08-01', 'Flash Crash'), ('2018-10-01', 'Taux/Trade War'), ('2020-02-01', 'COVID-19'), ('2022-01-01', 'Inflation/Inversion')]
for d_str, label in crises:
    if d_str in s_eq.index:
        ax1.annotate(label, xy=(pd.to_datetime(d_str), s_eq.loc[d_str]), xytext=(10, 30), 
                     textcoords='offset points', arrowprops=dict(arrowstyle='->', color='red'), fontsize=9)

# B. Scorecard AFML & Risque
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')
verdict_color = colors['green'] if dsr > 0.95 else colors['red']
report = [
    ["MÉTRIQUE AFML (López de Prado)", "RÉSULTAT"],
    ["PSR (Probabilistic Sharpe)", f"{psr*100:.2f}% (Cible > 95%)"],
    ["DSR (Deflated Sharpe)", f"{dsr*100:.2f}% (Cible > 95%)"],
    ["VERDICT ROBUSTESSE", "VALIDÉ" if dsr > 0.95 else "ÉCHEC"],
    ["", ""], # Spacer
    ["MÉTRIQUES DE PERFORMANCE", ""],
    ["Rendement Annuel (CAGR)", f"{ann_ret:.2%}"],
    ["Sharpe Ratio Annuel", f"{sr:.2f}"],
    ["Max Drawdown", f"{max_dd:.2%}"],
    ["Ratio de Calmar", f"{calmar:.2f}"],
    ["Exposition Moyenne", f"{s_pos.mean()*100:.1f}%"]
]
table = ax2.table(cellText=report, loc='center', cellLoc='left', colWidths=[0.5, 0.4])
table.scale(1, 2.5)
table.auto_set_font_size(False)
table.set_fontsize(12)
# Coloration du verdict
table[(3, 1)].set_facecolor(verdict_color)
table[(3, 1)].set_text_props(color='white', fontweight='bold')
for i in range(len(report)): table[(i, 0)].set_text_props(fontweight='bold')

# --- ROW 2 : MÉCANIQUE DE LA STRATÉGIE ---

# C. Profil d'Exposition Historique (Le "Cerveau" Macro)
ax3 = fig.add_subplot(gs[1, 0])
ax3.fill_between(s_pos.index, s_pos.values*100, 0, color=colors['pos'], alpha=0.4, label='Exposition Nette (%)')
ax3.axhline(100, color='black', ls=':', alpha=0.5)
ax3.set_ylabel("Levier (%)")
ax3.set_title("HISTORIQUE D'EXPOSITION DYNAMIQUE (RISK SCALING + CONVICTION)", fontsize=12, fontweight='bold', loc='left')
ax3.legend(loc='upper right')
ax3.set_ylim(0, 155)

# D. CPCV : Performance par Score de Conviction
ax4 = fig.add_subplot(gs[1, 1])
conviction_buckets = np.round(s_score * 4) / 4 # Buckets 0.0, 0.25, 0.5, 0.75, 1.0
ann_ret_by_score = s_ret.groupby(conviction_buckets).apply(lambda x: (1 + x).prod()**(252/len(x)) - 1)
bars = ax4.bar(ann_ret_by_score.index.astype(str), ann_ret_by_score.values*100, color=colors['strat'], alpha=0.7)
ax4.bar_label(bars, fmt='%.1f%%', padding=3)
ax4.set_xlabel("Score de Conviction Macro (0.0 à 1.0)")
ax4.set_ylabel("Rendement Annuel (%)")
ax4.set_title("CPCV : RENDEMENT ANNUEL PAR NIVEAU DE CONVICTION", fontsize=12, fontweight='bold', loc='left')

# --- ROW 3 : ROBUSTESSE & FUTUR ---

# E. CPCV : Stabilité par Régime Macro (Yield Curve)
ax5 = fig.add_subplot(gs[2, 0])
spread = (y10 - y3).loc[dates]
regimes = pd.cut(spread, bins=[-np.inf, 0, np.inf], labels=["COURBE INVERSÉE\n(Risque Récession)", "COURBE NORMALE\n(Expansion)"])
sharpe_regime = s_ret.groupby(regimes, observed=False).apply(lambda x: (x.mean()*252)/(x.std()*np.sqrt(252)))
bars_reg = ax5.bar(sharpe_regime.index, sharpe_regime.values, color=[colors['red'], colors['green']], alpha=0.6, width=0.5)
ax5.bar_label(bars_reg, fmt='Sharpe: %.2f', padding=3)
ax5.axhline(0, color='black', ls='-', lw=1)
ax5.set_title("CPCV : ROBUSTESSE PAR RÉGIME DE TAUX (STRESS TEST MACRO)", fontsize=12, fontweight='bold', loc='left')

# F. Monte Carlo Projection (5 Ans)
ax6 = fig.add_subplot(gs[2, 1])
mc_paths = np.cumprod(1 + np.random.choice(s_ret, size=(252*5, CONFIG["MC_SIMS"]), replace=True), axis=0) * s_eq.iloc[-1]
ax6.plot(np.percentile(mc_paths, 50, axis=1), color=colors['green'], lw=2, label='Projection Médiane')
ax6.fill_between(range(252*5), np.percentile(mc_paths, 5, axis=1), np.percentile(mc_paths, 95, axis=1), color=colors['green'], alpha=0.15, label='Intervalle de Confiance 90%')
ax6.set_title(f"PROJECTION MONTE CARLO 5 ANS ({CONFIG['MC_SIMS']//1000}K SIMS)", fontsize=12, fontweight='bold', loc='left')
ax6.legend(loc='upper left')
ax6.set_xlabel("Jours de Trading Futurs")
ax6.set_ylabel("Capital Projeté ($)")

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.15)
plt.show()

print("\n" + "="*80)
print(f"RAPPORT FINAL DE VALIDATION AFML")
print("="*80)
print(f"PSR: {psr*100:.2f}% | DSR (N={CONFIG['N_TRIALS']}): {dsr*100:.2f}%")
print(f"Ratio de Calmar: {calmar:.2f} | Max DD: {max_dd:.2%}")
print("="*80)