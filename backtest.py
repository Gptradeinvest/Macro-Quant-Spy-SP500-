import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats
import logging
from datetime import datetime

# ==============================================================================
# 0. CONFIGURATION & LOGGING
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    "ASSET": "SPY", "CASH": "BIL", "VIX": "^VIX", "Y10": "^TNX", "Y3M": "^IRX",
    "START_DATE": "2010-01-01",     # Buffer pour les MAs
    "START_BACKTEST": "2011-01-01", # Début réel de l'analyse
    "VOL_TARGET": 0.12,
    "ES_GUARD": 0.10,
    "INITIAL_CAPITAL": 100_000,
    "TC_BPS": 0.0010,              # 10 bps (Transaction Costs)
    "N_TRIALS": 100                # Ajusté pour l'estimation DSR
}

# ==============================================================================
# 1. MOTEUR DE DONNÉES (ROBUSTE)
# ==============================================================================
class DataEngine:
    @staticmethod
    def fetch_data(tickers, start_date):
        logger.info(f"Acquisition des données pour : {tickers}")
        try:
            raw = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)
            
            # Gestion robuste du MultiIndex yfinance (v0.2+)
            if isinstance(raw.columns, pd.MultiIndex):
                # On essaie de récupérer 'Close', sinon on prend le premier niveau
                if 'Close' in raw.columns.get_level_values(0):
                    df = raw['Close']
                elif 'Adj Close' in raw.columns.get_level_values(0):
                    df = raw['Adj Close']
                else:
                    df = raw.iloc[:, :len(tickers)] # Fallback
                    df.columns = tickers
            else:
                df = raw

            # Renommage pour standardiser
            inv_map = {v: k for k, v in zip(["ASSET", "CASH", "VIX", "Y10", "Y3M"], tickers)}
            # Mapping flexible selon ce que yfinance retourne
            rename_dict = {}
            for col in df.columns:
                if col in tickers:
                    # Trouver la clé correspondante dans CONFIG
                    key = [k for k, v in CONFIG.items() if v == col]
                    if key: rename_dict[col] = key[0] # Ex: SPY -> ASSET
            
            # Au cas où les noms ne matchent pas parfaitement, on force via l'ordre (Risqué mais nécessaire parfois)
            if len(rename_dict) < len(tickers):
                df.columns = ["ASSET", "CASH", "VIX", "Y10", "Y3M"]
            else:
                df = df.rename(columns=rename_dict)
                
            return df.ffill().dropna()
        except Exception as e:
            logger.critical(f"Erreur critique lors du téléchargement : {e}")
            raise

# ==============================================================================
# 2. MOTEUR DE STRATÉGIE (VECTORISÉ)
# ==============================================================================
class MacroQuantStrategy:
    def __init__(self, data):
        self.df = data.copy()
        self.results = None

    def generate_signals(self):
        logger.info("Génération des signaux vectorisés...")
        
        # 1. Préparation des Features (T)
        prices = self.df["ASSET"]
        # Indicateurs calculés sur les données disponibles à la clôture T
        sma200 = prices.rolling(200).mean()
        mom_spy = prices.pct_change(126)
        mom_bil = self.df["CASH"].pct_change(126)
        yield_spread = self.df["Y10"] - self.df["Y3M"]
        vix = self.df["VIX"]
        vix_ma = vix.rolling(20).mean()
        
        # 2. Logique de Signal (T)
        # Conditions booléennes converties en entiers (0/1)
        sig_trend = (prices > sma200).astype(int)
        sig_mom = (mom_spy > mom_bil).astype(int)
        sig_macro = (yield_spread > 0).astype(int)
        sig_vix = ((vix < 35) & (vix < vix_ma * 1.25)).astype(int)
        
        # Score Composite (0.0 à 1.0)
        # IMPORTANT : On applique .shift(1) ici. 
        # Le signal calculé à la clôture T est utilisé pour trader à T+1.
        raw_score = (sig_trend + sig_mom + sig_macro + sig_vix) / 4.0
        self.df["Signal"] = raw_score.shift(1)

    def apply_risk_management(self):
        logger.info("Application du Risk Management (Vol Target + ES)...")
        
        # Rendements (utilisés pour le calcul de vol, donc connus à T)
        rets = self.df["ASSET"].pct_change()
        
        # Volatilité réalisée (20 jours, annualisée) - Connue à T, utilisée pour T+1
        realized_vol = rets.rolling(20).std() * np.sqrt(252)
        self.df["Vol_Lag"] = realized_vol.shift(1)
        
        # 1. Volatility Targeting
        # Évite la division par zéro
        lev_base = np.where(self.df["Vol_Lag"] > 0.001, 
                           CONFIG["VOL_TARGET"] / self.df["Vol_Lag"], 
                           0.0)
        
        # 2. Expected Shortfall Guardrail (Proxy Gaussien pour vitesse)
        # ES 95% approx = Vol * 1.65 (ou 2.06 pour t-student léger) * Levier
        projected_es = self.df["Vol_Lag"] * lev_base * 2.06
        
        # Facteur de réduction si ES dépasse la limite
        risk_scaler = np.where(projected_es > CONFIG["ES_GUARD"], 
                              CONFIG["ES_GUARD"] / projected_es, 
                              1.0)
        
        # Poids Final = Levier * Scaler * Conviction
        # Cap à 1.5x de levier max pour sécurité
        self.df["Target_W"] = (lev_base * risk_scaler * self.df["Signal"]).clip(0, 1.5)

    def backtest(self):
        logger.info("Exécution du Backtest (Vectorisé)...")
        
        # Alignement des rendements (T) avec les Poids (décidés à T-1)
        # Target_W est déjà shifté dans generate_signals et apply_risk_management
        asset_ret = self.df["ASSET"].pct_change()
        cash_ret = self.df["CASH"].pct_change().fillna(0)
        
        # Calcul du Turnover pour les coûts
        # Différence absolue entre le poids d'aujourd'hui et d'hier
        turnover = self.df["Target_W"].diff().abs().fillna(0)
        costs = turnover * CONFIG["TC_BPS"]
        
        # Rendement Stratégie
        strat_gross = (self.df["Target_W"] * asset_ret) + ((1 - self.df["Target_W"]) * cash_ret)
        self.df["Strat_Net"] = strat_gross - costs
        
        # Filtrage sur la période de backtest réelle
        self.results = self.df.loc[CONFIG["START_BACKTEST"]:].copy()
        self.results["Equity"] = (1 + self.results["Strat_Net"]).cumprod() * CONFIG["INITIAL_CAPITAL"]
        self.results["Benchmark"] = (1 + asset_ret.loc[CONFIG["START_BACKTEST"]:]).cumprod() * CONFIG["INITIAL_CAPITAL"]
        
        return self.results

# ==============================================================================
# 3. ANALYTICS & REPORTING
# ==============================================================================
def get_afml_stats(returns, n_trials):
    """Calculs robustes PSR/DSR (López de Prado)"""
    rets = returns.dropna()
    if len(rets) < 100: return 0, 0, 0
    
    # Moments
    mean, std = rets.mean(), rets.std()
    skew, kurt = rets.skew(), rets.kurtosis()
    sr = (mean * 252) / (std * np.sqrt(252))
    
    # Ajustement N (si autocorrelation, N effectif diminue) - Simplifié ici
    n = len(rets)
    
    # Probabilistic Sharpe Ratio
    sigma_sr = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
    psr = stats.norm.cdf(sr / sigma_sr)
    
    # Deflated Sharpe Ratio
    # Euler-Mascheroni approximation pour l'espérance du max SR
    exp_max_sr = sigma_sr * np.sqrt(2 * np.log(n_trials))
    dsr = stats.norm.cdf((sr - exp_max_sr) / sigma_sr)
    
    return psr, dsr, sr

def create_dashboard(res):
    logger.info("Génération du Dashboard...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Métriques
    psr, dsr, sr = get_afml_stats(res["Strat_Net"], CONFIG["N_TRIALS"])
    cagr = (res["Equity"].iloc[-1] / res["Equity"].iloc[0]) ** (252 / len(res)) - 1
    dd = res["Equity"] / res["Equity"].cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd)
    
    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig)
    
    # 1. Equity Curve (Main)
    ax1 = fig.add_subplot(gs[0, :3])
    ax1.plot(res["Equity"], color='#004e66', lw=2, label='Macro-Quant (Net)')
    ax1.plot(res["Benchmark"], color='gray', lw=1, ls='--', alpha=0.5, label='SPY Buy & Hold')
    ax1.set_yscale('log')
    ax1.set_title(f"PERFORMANCE CUMULÉE (CAGR: {cagr:.2%} | Sharpe: {sr:.2f})", fontweight='bold')
    ax1.legend(loc="upper left")
    
    # 2. Stats Table
    ax2 = fig.add_subplot(gs[0, 3])
    ax2.axis('off')
    row_labels = ["Ann. Return", "Sharpe Ratio", "Max Drawdown", "Calmar Ratio", "PSR (>95%)", "DSR (>50%)"]
    cell_text = [[f"{cagr:.2%}"], [f"{sr:.2f}"], [f"{max_dd:.2%}"], [f"{calmar:.2f}"], [f"{psr:.2%}"], [f"{dsr:.2%}"]]
    colors = [['white']] * 4 + [[ '#d4edda' if psr > 0.95 else '#f8d7da' ]] + [[ '#d4edda' if dsr > 0.5 else '#f8d7da' ]]
    
    tbl = ax2.table(cellText=cell_text, rowLabels=row_labels, loc='center', cellColours=colors, colWidths=[0.5])
    tbl.scale(1, 2.5)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    
    # 3. Drawdown Area
    ax3 = fig.add_subplot(gs[1, :3])
    ax3.fill_between(dd.index, dd.values, 0, color='#e74c3c', alpha=0.3)
    ax3.set_title("PROFIL DE DRAWDOWN", fontsize=10, fontweight='bold')
    ax3.set_ylabel("DD %")
    
    # 4. Exposure / Leverage
    ax4 = fig.add_subplot(gs[2, :3])
    ax4.fill_between(res.index, res["Target_W"], 0, color='#f1c40f', alpha=0.4)
    ax4.axhline(1.0, color='gray', ls=':')
    ax4.set_title("EXPOSITION DYNAMIQUE (LEVIER)", fontsize=10, fontweight='bold')
    ax4.set_ylabel("Poids")
    
    # 5. Monthly Heatmap (Simplifiée)
    ax5 = fig.add_subplot(gs[1:, 3])
    m_rets = res["Strat_Net"].resample('ME').apply(lambda x: (1+x).prod()-1)
    # Pivot table pour heatmap
    pivot = pd.DataFrame({'Year': m_rets.index.year, 'Month': m_rets.index.month, 'Ret': m_rets.values})
    pivot = pivot.pivot(index='Year', columns='Month', values='Ret')
    sns.heatmap(pivot, annot=False, cmap="RdYlGn", center=0, cbar=False, ax=ax5)
    ax5.set_title("HEATMAP MENSUELLE", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 4. EXÉCUTION
# ==============================================================================
if __name__ == "__main__":
    try:
        # 1. Load
        tickers = [CONFIG["ASSET"], CONFIG["CASH"], CONFIG["VIX"], CONFIG["Y10"], CONFIG["Y3M"]]
        df = DataEngine.fetch_data(tickers, CONFIG["START_DATE"])
        
        # 2. Compute
        strat = MacroQuantStrategy(df)
        strat.generate_signals()
        strat.apply_risk_management()
        results = strat.backtest()
        
        # 3. Visualize
        create_dashboard(results)
        
    except Exception as e:
        logger.error(f"Arrêt du programme suite à une erreur : {e}")
