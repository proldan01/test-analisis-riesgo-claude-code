# ============================================================
# Financial Intelligence Platform
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

# --- Optional imports with graceful fallbacks ---
try:
    from streamlit_echarts import st_echarts
    ECHARTS_OK = True
except ImportError:
    ECHARTS_OK = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_OK = True
except ImportError:
    ARIMA_OK = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Financial Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# WALL STREET DARK THEME
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #080810 !important;
    color: #d0d0d0 !important;
    font-family: 'Inter', sans-serif;
}
.main .block-container { padding: 1rem 1.5rem 2rem 1.5rem; max-width: 100%; }
section[data-testid="stSidebar"] { background-color: #0d0d1a !important; border-right: 1px solid #1e1e3a; }
section[data-testid="stSidebar"] .block-container { padding: 1rem; }

h1,h2,h3,h4 { color: #e8e8e8 !important; }
p, li, span { color: #b0b0c8; }

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg,#0e0e20 0%,#141430 100%);
    border: 1px solid #1e1e3a; border-radius: 10px;
    padding: 14px 16px; margin: 6px 0;
    transition: border-color .25s, box-shadow .25s;
}
.kpi-card:hover { border-color: #00e87a; box-shadow: 0 0 18px rgba(0,232,122,.12); }
.kpi-label  { color: #5a5a7a; font-size: .72em; text-transform: uppercase; letter-spacing: .08em; }
.kpi-ticker { color: #ffd700; font-size: 1em; font-weight: 700; font-family: 'Share Tech Mono', monospace; }
.kpi-price  { font-size: 1.5em; font-weight: 700; color: #e8e8e8; }
.kpi-change-pos { color: #00e87a; font-size: .9em; }
.kpi-change-neg { color: #ff4545; font-size: .9em; }
.kpi-sub    { color: #3a3a5a; font-size: .72em; margin-top: 4px; }

/* Signal badges */
.sig-buy  { background:linear-gradient(135deg,#00331a,#004d28); border:1px solid #00e87a; color:#00e87a; padding:7px 18px; border-radius:20px; font-weight:700; text-align:center; display:inline-block; font-family:'Share Tech Mono',monospace; letter-spacing:.1em; }
.sig-sell { background:linear-gradient(135deg,#330000,#4d0000); border:1px solid #ff4545; color:#ff4545; padding:7px 18px; border-radius:20px; font-weight:700; text-align:center; display:inline-block; font-family:'Share Tech Mono',monospace; letter-spacing:.1em; }
.sig-hold { background:linear-gradient(135deg,#332800,#4d3c00); border:1px solid #ffd700; color:#ffd700; padding:7px 18px; border-radius:20px; font-weight:700; text-align:center; display:inline-block; font-family:'Share Tech Mono',monospace; letter-spacing:.1em; }

/* Section headers */
.sec-hdr { border-left:3px solid #00e87a; padding-left:10px; color:#e0e0e0; font-size:1em; font-weight:600; margin:18px 0 10px; letter-spacing:.04em; }

/* Sidebar section headers */
.sb-sec {
    color: #c0c0d8; font-size: .78em; font-weight: 700;
    text-transform: uppercase; letter-spacing: .12em;
    margin: 14px 0 8px 0; padding: 6px 0 4px 0;
    border-bottom: 1px solid #1e1e3a;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:#0d0d1a; border-bottom:1px solid #1e1e3a; gap:4px; }
.stTabs [data-baseweb="tab"] { color:#5a5a7a; font-weight:500; padding:8px 18px; border-radius:6px 6px 0 0; }
.stTabs [aria-selected="true"] { color:#00e87a !important; border-bottom:2px solid #00e87a !important; background:#0e0e20 !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border:1px solid #1e1e3a; border-radius:8px; }

/* Metric component */
[data-testid="metric-container"] { background:#0e0e20; border:1px solid #1e1e3a; border-radius:8px; padding:10px 14px; }
[data-testid="metric-container"] label { color:#5a5a7a !important; font-size:.75em !important; text-transform:uppercase; letter-spacing:.05em; }
[data-testid="stMetricValue"] { color:#e8e8e8 !important; font-family:'Share Tech Mono',monospace; }

/* News card */
.news-card { background:#0e0e20; border:1px solid #1e1e3a; border-radius:8px; padding:10px 14px; margin:6px 0; }
.news-link { color:#4da6ff; text-decoration:none; font-size:.88em; }
.news-pub  { color:#3a3a5a; font-size:.72em; margin-top:3px; }

/* VIX bar */
.vix-bar { background:linear-gradient(135deg,#0e0e20,#141430); border:1px solid #1e1e3a; border-radius:10px; padding:12px 20px; display:flex; justify-content:space-between; align-items:center; }

/* MC insight box */
.mc-box { background:#0e0e20; border:1px solid #1e1e3a; border-left:3px solid #4da6ff; border-radius:8px; padding:14px 16px; margin:10px 0; font-size:.85em; line-height:1.6; }

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a52c0 0%, #1e5edb 100%) !important;
    border: 1px solid #2a72f0 !important; color: #fff !important;
    font-weight: 700 !important; letter-spacing: .06em !important;
    border-radius: 8px !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1e5edb 0%, #2266f0 100%) !important;
    box-shadow: 0 0 22px rgba(30,94,219,.45) !important;
}

/* Secondary buttons (add/remove) */
.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, #12122a 0%, #1a1a38 100%) !important;
    border: 1px solid #2a2a4a !important; color: #c0c0d8 !important;
    border-radius: 8px !important;
}

/* Hide Streamlit chrome */
#MainMenu{visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden;}
div[data-testid="stDecoration"]{display:none;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
TRADING_DAYS = 252

BENCHMARK_INDICES = {
    "S&P 500 (^GSPC)":          "^GSPC",
    "Dow Jones (^DJI)":         "^DJI",
    "NASDAQ Composite (^IXIC)": "^IXIC",
    "Russell 2000 (^RUT)":      "^RUT",
    "FTSE 100 (^FTSE)":         "^FTSE",
    "DAX (^GDAXI)":             "^GDAXI",
    "Nikkei 225 (^N225)":       "^N225",
    "Hang Seng (^HSI)":         "^HSI",
    "MSCI World ETF (URTH)":    "URTH",
    "MSCI Emerging (EEM)":      "EEM",
    "Total Market (VTI)":       "VTI",
}

POPULAR = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B",
    "JPM","V","JNJ","UNH","PG","HD","MA","BAC","XOM","CVX",
    "PFE","KO","PEP","WMT","MCD","DIS","NFLX","INTC","AMD","PYPL",
    "SPY","QQQ","IWM","GLD","SLV","BTC-USD","ETH-USD",
    "^MXX","^BVSP","^MERV","EWW","EWZ",
]

PLOTLY_COLORS = [
    "#00e87a","#ffd700","#4da6ff","#ff6b35","#da70d6",
    "#ff4545","#40e0d0","#ff69b4","#c0c020","#20c0c0",
]

_ASSET_DEFAULTS = ["AAPL", "MSFT", "NVDA"]

# ============================================================
# UTILITY
# ============================================================

def hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Convert #rrggbb to rgba(r,g,b,a) safely."""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def _date_str(d) -> str:
    """Return ISO date string from various date-like objects."""
    if hasattr(d, "date"):
        return str(d.date())
    return str(d)

# ============================================================
# FINANCIAL MATH
# ============================================================

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = [(df["Open"].iloc[0] + df["Close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha["HA_Close"].iloc[i - 1]) / 2)
    ha["HA_Open"] = ha_open
    ha["HA_High"] = pd.concat([df["High"], ha["HA_Open"], ha["HA_Close"]], axis=1).max(axis=1)
    ha["HA_Low"]  = pd.concat([df["Low"],  ha["HA_Open"], ha["HA_Close"]], axis=1).min(axis=1)
    ha["Volume"]  = df["Volume"]
    return ha

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    sma = series.rolling(period).mean()
    sd  = series.rolling(period).std()
    return sma + std_mult * sd, sma, sma - std_mult * sd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    m = ema(series, fast) - ema(series, slow)
    s = ema(m, signal)
    return m, s, m - s

def ann_return(ret: pd.Series) -> float:
    if len(ret) < 2:
        return np.nan
    return float((1 + ret.mean()) ** TRADING_DAYS - 1)

def ann_vol(ret: pd.Series) -> float:
    if len(ret) < 2:
        return np.nan
    return float(ret.std() * np.sqrt(TRADING_DAYS))

def sharpe(ret: pd.Series, rfr: float = 0.05) -> float:
    excess = ret - rfr / TRADING_DAYS
    sd = excess.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))

def sortino(ret: pd.Series, rfr: float = 0.05) -> float:
    down = ret[ret < 0]
    if len(down) < 2:
        return np.nan
    dsd = down.std() * np.sqrt(TRADING_DAYS)
    return float((ann_return(ret) - rfr) / dsd) if dsd != 0 else np.nan

def max_dd(prices: pd.Series) -> float:
    roll_max = prices.cummax()
    return float(((prices - roll_max) / roll_max).min())

def beta_calc(asset_ret: pd.Series, bench_ret: pd.Series) -> float:
    aligned = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else np.nan

def alpha_calc(asset_ret: pd.Series, bench_ret: pd.Series, rfr: float = 0.05) -> float:
    b = beta_calc(asset_ret, bench_ret)
    if np.isnan(b):
        return np.nan
    return float(ann_return(asset_ret) - (rfr + b * (ann_return(bench_ret) - rfr)))

def var_hist(ret: pd.Series, conf: float = 0.95) -> float:
    return float(np.percentile(ret.dropna(), (1 - conf) * 100))

def cvar_hist(ret: pd.Series, conf: float = 0.95) -> float:
    v = var_hist(ret, conf)
    tail = ret[ret <= v]
    return float(tail.mean()) if len(tail) > 0 else v

def treynor(asset_ret: pd.Series, bench_ret: pd.Series, rfr: float = 0.05) -> float:
    b = beta_calc(asset_ret, bench_ret)
    if np.isnan(b) or b == 0:
        return np.nan
    return float((ann_return(asset_ret) - rfr) / b)

def calmar(ret: pd.Series, prices: pd.Series) -> float:
    mdd = abs(max_dd(prices))
    return float(ann_return(ret) / mdd) if mdd != 0 else np.nan

def info_ratio(asset_ret: pd.Series, bench_ret: pd.Series) -> float:
    active = (asset_ret - bench_ret).dropna()
    if active.std() == 0 or len(active) < 2:
        return np.nan
    return float(active.mean() / active.std() * np.sqrt(TRADING_DAYS))

# ============================================================
# ML SIGNALS
# ============================================================

def ml_signals(df: pd.DataFrame) -> pd.Series:
    if not SKLEARN_OK or len(df) < 100:
        return pd.Series(0, index=df.index)
    try:
        p = df["Close"].copy()
        feat = pd.DataFrame(index=df.index)
        feat["rsi14"]   = rsi(p, 14)
        feat["rsi7"]    = rsi(p, 7)
        feat["ema7r"]   = ema(p, 7)  / p - 1
        feat["ema30r"]  = ema(p, 30) / p - 1
        feat["ema50r"]  = ema(p, 50) / p - 1
        feat["ema200r"] = ema(p, 200)/ p - 1
        feat["ret1"]    = p.pct_change(1)
        feat["ret5"]    = p.pct_change(5)
        feat["ret20"]   = p.pct_change(20)
        feat["vol20"]   = feat["ret1"].rolling(20).std()
        bb_u, _, bb_l   = bollinger(p)
        feat["bb_pct"]  = (p - bb_l) / (bb_u - bb_l + 1e-9)
        ml_line, ms, _  = macd(p)
        feat["macd_h"]  = ml_line - ms
        future_ret      = p.pct_change(10).shift(-10)
        labels = pd.cut(future_ret, bins=[-np.inf, -0.02, 0.02, np.inf], labels=[-1, 0, 1])
        data   = feat.join(labels.rename("y")).dropna()
        if len(data) < 60:
            return pd.Series(0, index=df.index)
        X = data.drop("y", axis=1).values
        y = data["y"].astype(int).values
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)
        split  = int(len(X_sc) * 0.8)
        clf    = RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1)
        clf.fit(X_sc[:split], y[:split])
        preds  = clf.predict(X_sc)
        return pd.Series(preds, index=data.index).reindex(df.index).fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

# ============================================================
# FORECASTING
# ============================================================

def forecast_3m(prices: pd.Series):
    N = 63  # ~3 months trading days
    last_date   = prices.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=N)
    result       = {}
    last_price   = float(prices.iloc[-1])

    if ARIMA_OK and len(prices) >= 80:
        try:
            lp  = np.log(prices.dropna())
            mdl = ARIMA(lp, order=(5, 1, 0)).fit()
            fc  = mdl.get_forecast(steps=N)
            fv  = np.exp(fc.predicted_mean.values)
            ci  = fc.conf_int()
            result["arima"] = {
                "dates":  future_dates,
                "values": fv,
                "lower":  np.exp(ci.iloc[:, 0].values),
                "upper":  np.exp(ci.iloc[:, 1].values),
                "last_price": last_price,
            }
        except Exception:
            pass

    if SKLEARN_OK and len(prices) >= 30:
        try:
            y_arr = prices.dropna().values
            X_arr = np.arange(len(y_arr)).reshape(-1, 1)
            lr    = LinearRegression().fit(X_arr, y_arr)
            Xf    = np.arange(len(y_arr), len(y_arr) + N).reshape(-1, 1)
            result["linear"] = {"dates": future_dates, "values": lr.predict(Xf),
                                 "last_price": last_price}
        except Exception:
            pass

    return result

# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def recommend(ticker_df: pd.DataFrame, info: dict, bench_ret: pd.Series, rfr: float):
    prices = ticker_df["Close"].dropna()
    ret    = prices.pct_change().dropna()
    score  = 0
    pros, cons, risks = [], [], []

    r14      = rsi(prices, 14)
    cur_rsi  = float(r14.iloc[-1]) if not r14.empty else 50
    cp       = float(prices.iloc[-1])
    e7_v     = float(ema(prices, 7).iloc[-1])
    e30_v    = float(ema(prices, 30).iloc[-1])
    e50_v    = float(ema(prices, 50).iloc[-1])
    e200_v   = float(ema(prices, 200).iloc[-1]) if len(prices) >= 200 else np.nan

    if cur_rsi < 30:
        score += 2; pros.append(f"RSI {cur_rsi:.1f} — oversold, potential reversal")
    elif cur_rsi > 70:
        score -= 2; cons.append(f"RSI {cur_rsi:.1f} — overbought, correction risk")

    if not np.isnan(e200_v):
        if cp > e200_v:
            score += 1; pros.append("Price above 200-day EMA — long-term uptrend intact")
        else:
            score -= 1; cons.append("Price below 200-day EMA — long-term trend broken")
            risks.append("Structural downtrend (price < EMA 200)")

    if e7_v > e30_v > e50_v:
        score += 2; pros.append("Bullish EMA alignment (7 > 30 > 50) — positive momentum")
    elif e7_v < e30_v < e50_v:
        score -= 2; cons.append("Bearish EMA alignment (7 < 30 < 50) — negative momentum")

    av = ann_vol(ret)
    if av > 0.45:
        risks.append(f"High annualized volatility: {av:.1%}")

    mdd_v = max_dd(prices)
    if mdd_v < -0.30:
        risks.append(f"Significant historical drawdown: {mdd_v:.1%}")

    sh = sharpe(ret, rfr)
    if not np.isnan(sh):
        if sh > 1.0:
            score += 1; pros.append(f"Strong risk-adjusted return — Sharpe {sh:.2f}")
        elif sh < 0:
            score -= 1; cons.append(f"Negative Sharpe ratio ({sh:.2f})")

    if len(prices) >= 126:
        mom = prices.iloc[-1] / prices.iloc[-126] - 1
        if mom > 0.10:
            score += 1; pros.append(f"6-month momentum: +{mom:.1%}")
        elif mom < -0.15:
            score -= 1; cons.append(f"Weak 6-month momentum: {mom:.1%}")

    valuation = "fairly valued"
    pe  = info.get("trailingPE")
    d2e = info.get("debtToEquity")
    roe_v = info.get("returnOnEquity")

    if pe:
        if pe < 15:
            score += 1; pros.append(f"P/E {pe:.1f} — potentially undervalued"); valuation = "undervalued"
        elif pe > 40:
            score -= 1; cons.append(f"P/E {pe:.1f} — premium valuation"); valuation = "overvalued"

    if d2e and d2e > 200:
        risks.append(f"High leverage: Debt/Equity {d2e:.0f}%")

    if roe_v and roe_v > 0.20:
        score += 1; pros.append(f"Strong ROE {roe_v:.1%}")

    def to_sig(s):
        if s >= 3:  return "BUY",  "buy"
        if s <= -3: return "SELL", "sell"
        return "HOLD", "hold"

    r1m, c1m = to_sig(score)
    r3m, c3m = to_sig(int(score * 1.05))
    r6m, c6m = to_sig(int(score * 0.9))

    return dict(
        score=score,
        rec_1m=r1m, cls_1m=c1m,
        rec_3m=r3m, cls_3m=c3m,
        rec_6m=r6m, cls_6m=c6m,
        pros=pros, cons=cons, risks=risks, valuation=valuation,
    )

# ============================================================
# DATA FETCHING
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_hist(tickers: tuple, start: str, end: str, interval: str) -> dict:
    out = {}
    for t in tickers:
        try:
            df = yf.Ticker(t).history(start=start, end=end, interval=interval, auto_adjust=True)
            if not df.empty:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                out[t] = df
        except Exception:
            pass
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(ticker: str) -> list:
    try:
        n = yf.Ticker(ticker).news
        return n[:6] if n else []
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def fetch_vix() -> pd.DataFrame:
    try:
        df = yf.Ticker("^VIX").history(period="1y", auto_adjust=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

# ============================================================
# ECHARTS HELPERS
# ============================================================

def echarts_heatmap(matrix: pd.DataFrame, title: str,
                    color_min: str = "#ff4545", color_max: str = "#00e87a") -> dict:
    cols = matrix.columns.tolist()
    data = [[j, i, round(float(matrix.iloc[i, j]), 4)]
            for i in range(len(cols)) for j in range(len(cols))]
    return {
        "backgroundColor": "#080810",
        "title": {"text": title, "textStyle": {"color": "#e0e0e0", "fontSize": 13}},
        "tooltip": {"position": "top"},
        "grid": {"height": "72%", "top": "12%", "left": "12%", "right": "4%"},
        "xAxis": {"type": "category", "data": cols,
                  "axisLabel": {"color": "#888", "fontSize": 11},
                  "splitArea": {"show": True, "areaStyle": {"color": ["#0d0d1a", "#111125"]}}},
        "yAxis": {"type": "category", "data": cols,
                  "axisLabel": {"color": "#888", "fontSize": 11},
                  "splitArea": {"show": True, "areaStyle": {"color": ["#0d0d1a", "#111125"]}}},
        "visualMap": {
            "min": -1, "max": 1,
            "calculable": True, "orient": "horizontal",
            "left": "center", "bottom": "3%",
            "inRange": {"color": [color_min, "#111125", color_max]},
            "textStyle": {"color": "#888"},
        },
        "series": [{"name": title, "type": "heatmap", "data": data,
                    "label": {"show": True, "color": "#e0e0e0", "fontSize": 10},
                    "emphasis": {"itemStyle": {"shadowBlur": 10,
                                               "shadowColor": "rgba(0,232,122,.4)"}}}],
    }

def echarts_area_animated(dates_list: list, series_dict: dict, title: str = "") -> dict:
    colors = PLOTLY_COLORS
    series = []
    for i, (name, vals) in enumerate(series_dict.items()):
        series.append({
            "name": name, "type": "line",
            "data": [round(v, 2) for v in vals],
            "smooth": True, "showSymbol": False,
            "areaStyle": {"opacity": 0.08},
            "lineStyle": {"width": 2, "color": colors[i % len(colors)]},
            "itemStyle": {"color": colors[i % len(colors)]},
            "animationDuration": 1800,
            "animationEasing": "cubicOut",
        })
    return {
        "backgroundColor": "#080810",
        "animation": True,
        "title": {"text": title, "textStyle": {"color": "#e0e0e0", "fontSize": 12}},
        "tooltip": {
            "trigger": "axis",
            "backgroundColor": "rgba(8,8,20,.95)",
            "borderColor": "#1e1e3a",
            "textStyle": {"color": "#d0d0d0", "fontSize": 11},
        },
        "legend": {"textStyle": {"color": "#888"}, "top": "5%"},
        "grid": {"left": "6%", "right": "3%", "bottom": "12%", "containLabel": True},
        "xAxis": {
            "type": "category", "data": dates_list,
            "axisLabel": {"color": "#555", "fontSize": 10},
            "axisLine": {"lineStyle": {"color": "#1e1e3a"}},
            "boundaryGap": False,
        },
        "yAxis": {
            "type": "value",
            "splitLine": {"lineStyle": {"color": "#141428"}},
            "axisLabel": {"color": "#555", "fontSize": 10},
        },
        "dataZoom": [
            {"type": "inside", "start": 60, "end": 100},
            {"type": "slider", "start": 60, "end": 100,
             "borderColor": "#1e1e3a", "fillerColor": "rgba(0,232,122,.08)",
             "textStyle": {"color": "#555"}, "bottom": "1%"},
        ],
        "series": series,
    }

# ============================================================
# CANDLESTICK CHART  (ECharts — Heikin Ashi)
# ============================================================

def echarts_candle(df: pd.DataFrame, ticker: str, ema_cfg: dict,
                   custom_emas: list, show_bb: bool,
                   show_sigs: bool, sigs: pd.Series = None,
                   fc: dict = None) -> dict:

    ha     = heikin_ashi(df)
    dates  = [_date_str(d) for d in ha.index]
    candle = [[round(r.HA_Open,2), round(r.HA_Close,2),
               round(r.HA_Low,2),  round(r.HA_High,2)]
              for r in ha.itertuples()]
    vol    = [int(v) if not np.isnan(v) else 0 for v in df["Volume"].fillna(0)]
    vol_colors = ["#00e87a" if ha.HA_Close.iloc[i] >= ha.HA_Open.iloc[i] else "#ff4545"
                  for i in range(len(ha))]

    series = [{
        "name": ticker,
        "type": "candlestick",
        "data": candle,
        "gridIndex": 0, "xAxisIndex": 0, "yAxisIndex": 0,
        "itemStyle": {
            "color": "#00e87a", "color0": "#ff4545",
            "borderColor": "#00e87a", "borderColor0": "#ff4545",
        },
    }]

    ema_palette = {"7":"#ff6b35","30":"#ffd700","50":"#4da6ff","200":"#da70d6"}
    legend_items = [ticker]

    # Default EMAs
    for period, show in ema_cfg.items():
        if show:
            vals = ema(df["Close"], int(period)).round(2).tolist()
            color = ema_palette.get(period, "#aaa")
            series.append({
                "name": f"EMA {period}", "type": "line", "data": vals,
                "smooth": True,
                "lineStyle": {"width": 1.5, "color": color},
                "itemStyle": {"color": color},
                "showSymbol": False,
                "gridIndex": 0, "xAxisIndex": 0, "yAxisIndex": 0,
            })
            legend_items.append(f"EMA {period}")

    # Custom EMAs
    extra_colors = ["#40e0d0","#ff69b4","#c0c020","#20c0c0"]
    for idx, p_val in enumerate(custom_emas):
        vals  = ema(df["Close"], p_val).round(2).tolist()
        color = extra_colors[idx % len(extra_colors)]
        series.append({
            "name": f"EMA {p_val}", "type": "line", "data": vals,
            "smooth": True,
            "lineStyle": {"width": 1.5, "color": color, "type": "dashed"},
            "itemStyle": {"color": color},
            "showSymbol": False,
            "gridIndex": 0, "xAxisIndex": 0, "yAxisIndex": 0,
        })
        legend_items.append(f"EMA {p_val}")

    # Bollinger Bands
    if show_bb:
        bb_u, bb_m, bb_l = bollinger(df["Close"])
        for name, vals, dash in [("BB Up", bb_u, "dashed"),
                                   ("BB Mid", bb_m, "solid"),
                                   ("BB Low", bb_l, "dashed")]:
            series.append({
                "name": name, "type": "line",
                "data": [round(v,2) if not np.isnan(v) else None for v in vals],
                "smooth": True,
                "lineStyle": {"width": 1, "color": "rgba(90,90,200,.6)", "type": dash},
                "showSymbol": False,
                "gridIndex": 0, "xAxisIndex": 0, "yAxisIndex": 0,
            })
        legend_items += ["BB Up", "BB Mid", "BB Low"]

    # Buy/Sell markers
    if show_sigs and sigs is not None:
        buy_idx  = [i for i, idx in enumerate(df.index)
                    if idx in sigs.index and sigs.loc[idx] == 1]
        sell_idx = [i for i, idx in enumerate(df.index)
                    if idx in sigs.index and sigs.loc[idx] == -1]
        if buy_idx:
            series.append({
                "name": "BUY", "type": "scatter",
                "data": [[dates[i], round(float(df["Low"].iloc[i]) * 0.98, 2)]
                         for i in buy_idx],
                "symbol": "triangle", "symbolSize": 10,
                "itemStyle": {"color": "#00e87a"},
                "gridIndex": 0, "xAxisIndex": 0, "yAxisIndex": 0,
            })
            legend_items.append("BUY")
        if sell_idx:
            series.append({
                "name": "SELL", "type": "scatter",
                "data": [[dates[i], round(float(df["High"].iloc[i]) * 1.02, 2)]
                         for i in sell_idx],
                "symbol": "triangle", "symbolSize": 10, "symbolRotate": 180,
                "itemStyle": {"color": "#ff4545"},
                "gridIndex": 0, "xAxisIndex": 0, "yAxisIndex": 0,
            })
            legend_items.append("SELL")

    # Forecast  — connect to last known price
    all_dates = dates
    if fc and "arima" in fc:
        fd  = [_date_str(d) for d in fc["arima"]["dates"]]
        fv  = [round(float(v), 2) for v in fc["arima"]["values"]]
        last_price = round(fc["arima"]["last_price"], 2)
        # Connect forecast band and line to the last historical candle
        fc_vals_conn  = [last_price] + fv
        fc_upper_conn = [last_price] + [round(float(v),2) for v in fc["arima"]["upper"]]
        fc_lower_conn = [last_price] + [round(float(v),2) for v in fc["arima"]["lower"]]
        all_dates = dates + fd
        pad = len(dates) - 1  # Nones for all history except last point

        for name, conn_vals, style in [
            ("FC Upper", fc_upper_conn, {"width":1,"color":"rgba(255,215,0,.25)","type":"dashed"}),
            ("FC Lower", fc_lower_conn, {"width":1,"color":"rgba(255,215,0,.25)","type":"dashed"}),
        ]:
            series.append({
                "name": name, "type": "line",
                "data": [None]*pad + conn_vals,
                "lineStyle": style, "showSymbol": False,
                "gridIndex": 0, "xAxisIndex": 0, "yAxisIndex": 0,
            })
        series.append({
            "name": "3M Forecast", "type": "line",
            "data": [None]*pad + fc_vals_conn,
            "lineStyle": {"width": 2, "color": "#ffd700", "type": "dotted"},
            "itemStyle": {"color": "#ffd700"},
            "showSymbol": False,
            "gridIndex": 0, "xAxisIndex": 0, "yAxisIndex": 0,
        })
        legend_items.append("3M Forecast")

    # Volume
    series.append({
        "name": "Volume", "type": "bar",
        "data": [{"value": vol[i], "itemStyle": {"color": vol_colors[i]}}
                 for i in range(len(vol))],
        "gridIndex": 1, "xAxisIndex": 1, "yAxisIndex": 1, "barMaxWidth": 6,
    })

    # RSI
    rsi_vals = [round(v, 2) if not np.isnan(v) else None for v in rsi(df["Close"], 14)]
    series.append({
        "name": "RSI 14", "type": "line", "data": rsi_vals,
        "lineStyle": {"width": 1.5, "color": "#9b59b6"},
        "showSymbol": False,
        "gridIndex": 2, "xAxisIndex": 2, "yAxisIndex": 2,
    })

    return {
        "backgroundColor": "#080810",
        "animation": True, "animationDuration": 800,
        "tooltip": {
            "trigger": "axis", "axisPointer": {"type": "cross"},
            "backgroundColor": "rgba(8,8,20,.95)",
            "borderColor": "#1e1e3a",
            "textStyle": {"color": "#d0d0d0", "fontSize": 11},
        },
        # Scrollable legend – prevents overlap
        "legend": {
            "type": "scroll",
            "data": legend_items,
            "top": "0%", "left": "center", "width": "96%",
            "pageIconColor": "#888", "pageTextStyle": {"color": "#888"},
            "textStyle": {"color": "#888", "fontSize": 10},
            "icon": "roundRect", "itemHeight": 8, "itemGap": 14,
        },
        "axisPointer": {"link": [{"xAxisIndex": "all"}]},
        "grid": [
            {"left": "7%", "right": "3%", "top": "10%", "height": "51%"},
            {"left": "7%", "right": "3%", "top": "65%", "height": "12%"},
            {"left": "7%", "right": "3%", "top": "80%", "height": "12%"},
        ],
        "xAxis": [
            {"type": "category", "data": all_dates, "gridIndex": 0,
             "axisLabel": {"show": False},
             "axisLine": {"lineStyle": {"color": "#1e1e3a"}}, "splitLine": {"show": False}},
            {"type": "category", "data": dates, "gridIndex": 1,
             "axisLabel": {"show": False},
             "axisLine": {"lineStyle": {"color": "#1e1e3a"}}},
            {"type": "category", "data": dates, "gridIndex": 2,
             "axisLabel": {"color": "#555", "fontSize": 9},
             "axisLine": {"lineStyle": {"color": "#1e1e3a"}}},
        ],
        "yAxis": [
            {"scale": True, "gridIndex": 0,
             "splitLine": {"lineStyle": {"color": "#111125"}},
             "axisLabel": {"color": "#555", "fontSize": 10}},
            {"scale": True, "gridIndex": 1, "splitNumber": 2,
             "axisLabel": {"color": "#555", "fontSize": 9},
             "splitLine": {"lineStyle": {"color": "#111125"}}},
            {"scale": True, "gridIndex": 2, "min": 0, "max": 100, "splitNumber": 2,
             "axisLabel": {"color": "#555", "fontSize": 9},
             "splitLine": {"lineStyle": {"color": "#111125"}}},
        ],
        "dataZoom": [
            {"type": "inside", "xAxisIndex": [0, 1, 2], "start": 70, "end": 100},
            {"type": "slider", "xAxisIndex": [0, 1, 2], "start": 70, "end": 100,
             "bottom": "1%", "height": 18, "borderColor": "#1e1e3a",
             "fillerColor": "rgba(0,232,122,.08)",
             "textStyle": {"color": "#555"}},
        ],
        "series": series,
    }

# ============================================================
# PLOTLY HELPERS
# ============================================================

def _dark_layout(**kw):
    base = dict(
        template="plotly_dark",
        paper_bgcolor="#080810",
        plot_bgcolor="#0d0d1a",
        font=dict(color="#888", size=11),
        margin=dict(l=50, r=20, t=45, b=40),
        hovermode="x unified",
    )
    base.update(kw)
    return base

def plotly_perf(prices_df: pd.DataFrame, bench: pd.Series, bench_name: str):
    fig = go.Figure()
    for i, col in enumerate(prices_df.columns):
        norm = prices_df[col] / prices_df[col].dropna().iloc[0] * 100
        fig.add_trace(go.Scatter(x=norm.index, y=norm, name=col,
                                 line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], width=2)))
    if bench is not None and not bench.empty:
        nb = bench / bench.dropna().iloc[0] * 100
        fig.add_trace(go.Scatter(x=nb.index, y=nb, name=bench_name,
                                 line=dict(color="#555", width=2, dash="dash")))
    fig.update_layout(title="Normalized Performance (Base 100)",
                      yaxis_title="Indexed Price", **_dark_layout(height=380))
    return fig

def plotly_rolling_beta(returns_df: pd.DataFrame, bench_ret: pd.Series):
    fig = go.Figure()
    for i, col in enumerate(returns_df.columns):
        rb = returns_df[col].rolling(60).cov(bench_ret) / bench_ret.rolling(60).var()
        fig.add_trace(go.Scatter(x=rb.index, y=rb, name=col,
                                 line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], width=1.5)))
    fig.add_hline(y=1, line_dash="dash", line_color="#333",
                  annotation_text="β = 1", annotation_font_color="#555")
    fig.update_layout(title="Rolling 60-Day Beta", yaxis_title="Beta", **_dark_layout(height=320))
    return fig

def plotly_drawdown(prices_df: pd.DataFrame):
    """Draw drawdown chart — uses hex_to_rgba to build valid fillcolor strings."""
    fig = go.Figure()
    for i, col in enumerate(prices_df.columns):
        p   = prices_df[col].dropna()
        dd  = (p - p.cummax()) / p.cummax() * 100
        col_hex  = PLOTLY_COLORS[i % len(PLOTLY_COLORS)]
        col_fill = hex_to_rgba(col_hex, 0.07)          # ← fix: proper rgba conversion
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd, name=col,
            line=dict(color=col_hex, width=1.5),
            fill="tozeroy", fillcolor=col_fill,
        ))
    fig.update_layout(title="Drawdown from Peak (%)", yaxis_title="Drawdown (%)",
                      **_dark_layout(height=320))
    return fig

def plotly_rr_scatter(risk_df: pd.DataFrame, rfr: float, bench_ret: pd.Series):
    fig = go.Figure()
    for i, ticker in enumerate(risk_df.index):
        row = risk_df.loc[ticker]
        rv, rr = row.get("Ann. Volatility", np.nan), row.get("Ann. Return", np.nan)
        sh = row.get("Sharpe Ratio", 0) or 0
        if np.isnan(rv) or np.isnan(rr):
            continue
        c = PLOTLY_COLORS[i % len(PLOTLY_COLORS)]
        fig.add_trace(go.Scatter(
            x=[rv * 100], y=[rr * 100], mode="markers+text",
            text=[ticker], textposition="top center",
            marker=dict(size=max(8, min(28, abs(sh) * 8)), color=c, opacity=.85,
                        line=dict(width=1, color="white")),
            name=ticker,
            hovertemplate=f"<b>{ticker}</b><br>Vol: %{{x:.1f}}%<br>Ret: %{{y:.1f}}%<br>Sharpe: {sh:.2f}<extra></extra>",
        ))
    if not bench_ret.empty:
        bv    = ann_vol(bench_ret) * 100
        br    = ann_return(bench_ret) * 100
        slope = (br - rfr * 100) / (bv + 1e-9)
        xs    = np.linspace(0, max(35, bv * 1.6), 50)
        fig.add_trace(go.Scatter(x=xs, y=rfr * 100 + slope * xs, mode="lines",
                                 name="CML", line=dict(color="#333", dash="dash", width=1)))
    fig.update_layout(title="Risk-Return Map  (bubble size ∝ |Sharpe|)",
                      xaxis_title="Annualized Volatility (%)",
                      yaxis_title="Annualized Return (%)",
                      **_dark_layout(height=440))
    return fig

def plotly_scatter_matrix(returns_df: pd.DataFrame):
    fig = px.scatter_matrix(returns_df, dimensions=returns_df.columns.tolist(),
                            color_discrete_sequence=PLOTLY_COLORS)
    fig.update_traces(marker=dict(size=2, opacity=.4))
    fig.update_layout(title="Returns Scatter Matrix", **_dark_layout(height=600))
    return fig

def plotly_vol_bar(returns_df: pd.DataFrame):
    vols   = returns_df.std() * np.sqrt(TRADING_DAYS) * 100
    colors = ["#00e87a" if v < 25 else "#ffd700" if v < 40 else "#ff4545" for v in vols]
    fig    = go.Figure(go.Bar(
        x=vols.index.tolist(), y=vols.values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in vols], textposition="outside",
    ))
    fig.add_hline(y=20, line_dash="dash", line_color="#333", annotation_text="20% ref.")
    fig.update_layout(title="Annualized Volatility by Asset (%)",
                      yaxis_title="Volatility (%)", **_dark_layout(height=320))
    return fig

def plotly_vix(vix_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vix_df.index, y=vix_df["Close"], name="VIX",
                             line=dict(color="#ff6b35", width=2),
                             fill="tozeroy", fillcolor="rgba(255,107,53,.07)"))
    fig.add_hrect(y0=0,  y1=15,  fillcolor="rgba(0,232,122,.04)",  line_width=0)
    fig.add_hrect(y0=15, y1=25,  fillcolor="rgba(255,215,0,.03)",  line_width=0)
    fig.add_hrect(y0=25, y1=40,  fillcolor="rgba(255,107,53,.03)", line_width=0)
    fig.add_hrect(y0=40, y1=100, fillcolor="rgba(255,69,69,.04)",  line_width=0)
    for y, lbl, col in [(15,"15 — Normal","rgba(0,232,122,.5)"),
                        (25,"25 — Elevated","rgba(255,215,0,.5)"),
                        (40,"40 — Extreme","rgba(255,69,69,.5)")]:
        fig.add_hline(y=y, line_dash="dash", line_color=col,
                      annotation_text=lbl, annotation_font_color=col)
    fig.update_layout(title="CBOE Volatility Index — VIX (1Y)",
                      yaxis_title="VIX", **_dark_layout(height=280))
    return fig

# ============================================================
# PAIR RETURNS SCATTER  (Correlation tab)
# ============================================================

def plotly_pair_scatter(returns_df: pd.DataFrame, t1: str, t2: str):
    if t1 not in returns_df.columns or t2 not in returns_df.columns:
        return None
    aligned = pd.concat([returns_df[t1], returns_df[t2]], axis=1).dropna()
    x_vals  = aligned.iloc[:, 0].values * 100
    y_vals  = aligned.iloc[:, 1].values * 100
    corr_val = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="markers",
        marker=dict(size=4, color="#4da6ff", opacity=0.45),
        name=f"{t1} vs {t2}",
        hovertemplate=f"{t1}: %{{x:.2f}}%<br>{t2}: %{{y:.2f}}%<extra></extra>",
    ))

    # OLS trend line
    if SKLEARN_OK and len(x_vals) > 5:
        lr  = LinearRegression().fit(x_vals.reshape(-1, 1), y_vals)
        xs  = np.linspace(x_vals.min(), x_vals.max(), 100)
        ys  = lr.predict(xs.reshape(-1, 1))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                 line=dict(color="#4da6ff", width=2),
                                 name="OLS Trend"))
    elif len(x_vals) > 1:
        m, b = np.polyfit(x_vals, y_vals, 1)
        xs   = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines",
                                 line=dict(color="#4da6ff", width=2),
                                 name="Trend"))

    fig.update_layout(
        title=f"Pair Returns Scatter: {t1} vs {t2}   (ρ = {corr_val:.3f})",
        xaxis_title=f"{t1} Daily Return (%)",
        yaxis_title=f"{t2} Daily Return (%)",
        xaxis=dict(ticksuffix="%"),
        yaxis=dict(ticksuffix="%"),
        **_dark_layout(height=420),
    )
    return fig

# ============================================================
# MONTE CARLO EFFICIENT FRONTIER  (Covariance tab)
# ============================================================

def monte_carlo_frontier(returns_df: pd.DataFrame, n_portfolios: int = 5000,
                         rfr: float = 0.05) -> dict | None:
    n = len(returns_df.columns)
    if n < 2 or len(returns_df) < 30:
        return None
    tickers  = returns_df.columns.tolist()
    mean_ret = returns_df.mean() * TRADING_DAYS
    cov_mat  = returns_df.cov() * TRADING_DAYS
    np.random.seed(42)

    vols, rets, sharpes, weights_list = [], [], [], []
    for _ in range(n_portfolios):
        w  = np.random.dirichlet(np.ones(n))
        pr = float(np.dot(w, mean_ret))
        pv = float(np.sqrt(w @ cov_mat.values @ w))
        ps = (pr - rfr) / pv if pv > 0 else 0
        vols.append(pv * 100)
        rets.append(pr * 100)
        sharpes.append(ps)
        weights_list.append(w)

    max_sh_idx = int(np.argmax(sharpes))
    min_v_idx  = int(np.argmin(vols))

    return dict(
        vols=vols, rets=rets, sharpes=sharpes,
        weights=weights_list, tickers=tickers,
        max_sh_idx=max_sh_idx, min_v_idx=min_v_idx,
    )

def plotly_mc_frontier(mc: dict) -> go.Figure:
    fig = go.Figure()

    # All portfolios coloured by Sharpe
    fig.add_trace(go.Scatter(
        x=mc["vols"], y=mc["rets"], mode="markers",
        marker=dict(
            color=mc["sharpes"],
            colorscale=[[0,"#ff4545"],[0.5,"#ffd700"],[1,"#00e87a"]],
            size=3, opacity=0.55,
            colorbar=dict(title="Sharpe", tickfont=dict(color="#888"),
                          titlefont=dict(color="#888")),
        ),
        text=[f"Sharpe: {s:.2f}" for s in mc["sharpes"]],
        hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<br>%{text}<extra></extra>",
        name="Portfolios",
    ))

    # Max Sharpe
    msi = mc["max_sh_idx"]
    fig.add_trace(go.Scatter(
        x=[mc["vols"][msi]], y=[mc["rets"][msi]],
        mode="markers+text",
        marker=dict(size=16, color="#00e87a", symbol="star",
                    line=dict(width=1, color="white")),
        text=["Max Sharpe"], textposition="top right",
        textfont=dict(color="#00e87a"),
        name="Max Sharpe",
    ))

    # Min Volatility
    mvi = mc["min_v_idx"]
    fig.add_trace(go.Scatter(
        x=[mc["vols"][mvi]], y=[mc["rets"][mvi]],
        mode="markers+text",
        marker=dict(size=16, color="#4da6ff", symbol="diamond",
                    line=dict(width=1, color="white")),
        text=["Min Risk"], textposition="top right",
        textfont=dict(color="#4da6ff"),
        name="Min Volatility",
    ))

    fig.update_layout(
        title=f"Monte Carlo Efficient Frontier  ({len(mc['vols']):,} portfolios)",
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
        xaxis=dict(ticksuffix="%"),
        yaxis=dict(ticksuffix="%"),
        **_dark_layout(height=520),
    )
    return fig

# ============================================================
# DCF SIMULATOR
# ============================================================

def simple_dcf(info: dict, g: float, wacc: float, tg: float, years: int):
    fcf    = info.get("freeCashflow")
    shares = info.get("sharesOutstanding")
    price  = info.get("currentPrice") or info.get("regularMarketPrice")
    if not fcf or not shares or fcf <= 0:
        return None
    pv_flows = sum(fcf * (1 + g) ** y / (1 + wacc) ** y for y in range(1, years + 1))
    terminal = fcf * (1 + g) ** years * (1 + tg) / (wacc - tg) / (1 + wacc) ** years
    total    = pv_flows + terminal
    iv       = total / shares
    return dict(iv=iv, price=price, upside=(iv / price - 1) if price else None,
                pv_flows=pv_flows, terminal=terminal, total=total)

# ============================================================
# EXCEL EXPORT
# ============================================================

def build_excel(hist_data: dict, risk_df: pd.DataFrame, returns_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for t, df in hist_data.items():
            if df.empty:
                continue
            d = df.copy()
            if hasattr(d.index, "tz") and d.index.tz:
                d.index = d.index.tz_localize(None)
            d.to_excel(w, sheet_name=t[:28])
        if not returns_df.empty:
            rd = returns_df.copy()
            if hasattr(rd.index, "tz") and rd.index.tz:
                rd.index = rd.index.tz_localize(None)
            rd.to_excel(w, sheet_name="Returns")
        if not risk_df.empty:
            risk_df.to_excel(w, sheet_name="Risk_Metrics")
        if not returns_df.empty:
            returns_df.corr().to_excel(w, sheet_name="Correlation")
            (returns_df.cov() * TRADING_DAYS).to_excel(w, sheet_name="Covariance_Ann")
    buf.seek(0)
    return buf.read()

# ============================================================
# VIX WIDGET
# ============================================================

def vix_widget(vix_df: pd.DataFrame):
    if vix_df.empty:
        return
    cur  = float(vix_df["Close"].iloc[-1])
    prev = float(vix_df["Close"].iloc[-2]) if len(vix_df) > 1 else cur
    chg  = cur - prev
    if cur < 15:
        regime, col = "Low Volatility — Risk-On", "#00e87a"
    elif cur < 20:
        regime, col = "Normal",                   "#00e87a"
    elif cur < 25:
        regime, col = "Elevated — Caution",       "#ffd700"
    elif cur < 35:
        regime, col = "High — Risk-Off",          "#ff6b35"
    else:
        regime, col = "Extreme Fear",             "#ff4545"
    arrow  = "▲" if chg >= 0 else "▼"
    ch_col = "#ff4545" if chg >= 0 else "#00e87a"
    st.markdown(f"""
    <div class="vix-bar">
      <div>
        <span style="color:#3a3a5a;font-size:.72em;text-transform:uppercase;letter-spacing:.08em;">CBOE VIX</span><br>
        <span style="color:{col};font-size:1.9em;font-weight:700;font-family:'Share Tech Mono',monospace;">{cur:.2f}</span>
        <span style="color:{ch_col};font-size:.9em;margin-left:8px;">{arrow} {abs(chg):.2f}</span>
      </div>
      <div style="text-align:right;">
        <span style="color:#3a3a5a;font-size:.72em;">MARKET REGIME</span><br>
        <span style="color:{col};font-weight:600;font-size:.9em;">{regime}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

def sidebar() -> dict:
    # ── Session state defaults (only set once) ────────────────
    if "n_assets" not in st.session_state:
        st.session_state.n_assets = len(_ASSET_DEFAULTS)
    for i, v in enumerate(_ASSET_DEFAULTS):
        if f"asset_{i}" not in st.session_state:
            st.session_state[f"asset_{i}"] = v

    with st.sidebar:
        st.markdown("### ⚙️ Parameters")
        st.caption("Configure analysis settings")
        st.markdown("---")

        # ─── ASSET SELECTION ───────────────────────────────────
        st.markdown('<div class="sb-sec">📌 ASSET SELECTION</div>', unsafe_allow_html=True)
        st.caption("Type any Yahoo Finance ticker (AAPL, BTC-USD, GC=F, ^MXX …) or pick from suggestions.")

        assets_raw = []
        for i in range(st.session_state.n_assets):
            c1, c2 = st.columns([5, 4])
            typed  = c1.text_input(f"Asset {i+1}", key=f"asset_{i}")
            picked = c2.selectbox("or pick", [""] + POPULAR,
                                   key=f"pick_{i}", label_visibility="visible")
            val = (picked.strip() if picked else typed.strip().upper())
            assets_raw.append(val)

        btn1, btn2 = st.columns(2)
        if btn1.button("＋ Add asset", use_container_width=True):
            if st.session_state.n_assets < 10:
                st.session_state.n_assets += 1
                st.rerun()
        if btn2.button("－ Remove last", use_container_width=True):
            if st.session_state.n_assets > 1:
                st.session_state.n_assets -= 1
                st.rerun()

        st.markdown("---")

        # ─── PERIOD & FREQUENCY ────────────────────────────────
        st.markdown('<div class="sb-sec">📅 PERIOD & FREQUENCY</div>', unsafe_allow_html=True)
        today = datetime.today()
        period_map = {
            "1 Month":  today - timedelta(30),
            "3 Months": today - timedelta(90),
            "6 Months": today - timedelta(180),
            "1 Year":   today - timedelta(365),
            "2 Years":  today - timedelta(730),
            "3 Years":  today - timedelta(1095),
            "5 Years":  today - timedelta(1825),
            "10 Years": today - timedelta(3650),
            "Custom":   None,
        }
        period = st.selectbox("Analysis Period", list(period_map.keys()), index=5)
        if period == "Custom":
            start = st.date_input("Start:", today - timedelta(1095))
            end   = st.date_input("End:",   today)
        else:
            start = period_map[period]
            end   = today

        freq_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
        freq = st.selectbox("Data Frequency", list(freq_map.keys()), index=0)
        st.selectbox("Price Type", ["Close", "Open", "High", "Low"], index=0)

        st.markdown("---")

        # ─── BENCHMARK ─────────────────────────────────────────
        st.markdown('<div class="sb-sec">🏁 BENCHMARK</div>', unsafe_allow_html=True)
        bench_name   = st.selectbox("Reference Index", list(BENCHMARK_INDICES.keys()), index=0)
        bench_ticker = BENCHMARK_INDICES[bench_name]

        st.markdown("---")

        # ─── RISK PARAMETERS ───────────────────────────────────
        st.markdown('<div class="sb-sec">⚠️ RISK PARAMETERS</div>', unsafe_allow_html=True)
        rfr_val = st.number_input(
            "Risk-Free Rate (annual)", min_value=0.0, max_value=0.30,
            value=0.0457, step=0.0001, format="%.4f",
        )
        capital = st.number_input(
            "Capital ($)", min_value=0.0, value=10000.0, step=100.0, format="%.2f",
        )
        conf = st.slider("Confidence Interval", min_value=0.80,
                         max_value=0.99, value=0.95, step=0.01)

        st.markdown("---")

        # ─── CHART STUDIES ─────────────────────────────────────
        st.markdown('<div class="sb-sec">📊 CHART STUDIES</div>', unsafe_allow_html=True)
        e7   = st.checkbox("EMA 7",               value=True)
        e30  = st.checkbox("EMA 30",              value=True)
        e50  = st.checkbox("EMA 50",              value=True)
        e200 = st.checkbox("EMA 200",             value=True)
        show_bb   = st.checkbox("Bollinger Bands (20,2)", value=True)
        show_sigs = st.checkbox("Buy / Sell Signals",     value=True)
        custom_raw = st.text_input(
            "Custom EMA periods (comma-sep)", "",
            placeholder="e.g. 10, 100, 150",
            help="Extra EMA periods to overlay on the chart",
        )
        custom_emas = [int(x.strip()) for x in custom_raw.split(",")
                       if x.strip().isdigit()]

        st.markdown("---")

        # ─── DCF PARAMS ────────────────────────────────────────
        st.markdown('<div class="sb-sec">💹 DCF SIMULATOR</div>', unsafe_allow_html=True)
        dcf_g    = st.slider("FCF growth (%)",         1.0, 30.0, 10.0, 0.5) / 100
        dcf_wacc = st.slider("WACC / Discount (%)",    5.0, 20.0, 10.0, 0.5) / 100
        dcf_tg   = st.slider("Terminal growth (%)",    1.0,  5.0,  3.0, 0.25) / 100
        dcf_yrs  = st.slider("Projection years",       3,   10,    5)

        st.markdown("---")
        run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)
        st.caption("Data: Yahoo Finance · Not financial advice.")

    tickers = [a for a in assets_raw if a]
    return dict(
        tickers=tickers, start=start, end=end,
        bench_ticker=bench_ticker, bench_name=bench_name,
        freq=freq_map[freq], rfr=rfr_val, conf=conf, capital=capital,
        ema_cfg={"7": e7, "30": e30, "50": e50, "200": e200},
        custom_emas=custom_emas,
        show_bb=show_bb, show_sigs=show_sigs,
        dcf_g=dcf_g, dcf_wacc=dcf_wacc, dcf_tg=dcf_tg, dcf_yrs=dcf_yrs,
        run=run_btn,
    )

# ============================================================
# MAIN
# ============================================================

def main():
    # ── Run-gate via session state ────────────────────────────
    if "running" not in st.session_state:
        st.session_state.running = False

    st.markdown("""
    <div style="border-bottom:1px solid #1e1e3a;padding-bottom:12px;margin-bottom:18px;">
      <span style="color:#e0e0e0;font-size:1.5em;font-weight:700;
                   font-family:'Share Tech Mono',monospace;">
        📈 Financial Intelligence Platform
      </span><br>
      <span style="color:#3a3a5a;font-size:.8em;">
        Risk &amp; Return · Technical Analysis · ML Signals · Valuation · Forecasting
      </span>
    </div>
    """, unsafe_allow_html=True)

    p = sidebar()

    if p["run"]:
        st.session_state.running = True

    if not st.session_state.running:
        st.info("👈  Configure parameters in the panel and click **Run Analysis** to begin.")
        return

    if not p["tickers"]:
        st.warning("Enter at least one ticker symbol in the control panel.")
        return

    start_str = (p["start"].strftime("%Y-%m-%d")
                 if hasattr(p["start"], "strftime") else str(p["start"]))
    end_str   = (p["end"].strftime("%Y-%m-%d")
                 if hasattr(p["end"], "strftime") else str(p["end"]))

    with st.spinner("Fetching market data…"):
        hist      = fetch_hist(tuple(p["tickers"]), start_str, end_str, p["freq"])
        bench_raw = fetch_hist((p["bench_ticker"],), start_str, end_str, p["freq"])
        vix_df    = fetch_vix()

    valid = [t for t in p["tickers"] if t in hist]
    if not valid:
        st.error("No data retrieved. Verify ticker symbols and date range.")
        return

    prices_df = pd.DataFrame({t: hist[t]["Close"] for t in valid}).dropna(how="all")
    bench_ser = (bench_raw[p["bench_ticker"]]["Close"]
                 if p["bench_ticker"] in bench_raw else pd.Series(dtype=float))
    returns_df = prices_df.pct_change().dropna()
    bench_ret  = bench_ser.pct_change().dropna()

    # ── Risk metrics table ────────────────────────────────────
    rows = []
    for t in valid:
        if t not in returns_df.columns:
            continue
        r  = returns_df[t].dropna()
        pr = prices_df[t].dropna()
        rows.append({
            "Ticker":               t,
            "Ann. Return":          ann_return(r),
            "Ann. Volatility":      ann_vol(r),
            "Sharpe Ratio":         sharpe(r, p["rfr"]),
            "Sortino Ratio":        sortino(r, p["rfr"]),
            "Max Drawdown":         max_dd(pr),
            "Beta":                 beta_calc(r, bench_ret),
            "Alpha (Jensen)":       alpha_calc(r, bench_ret, p["rfr"]),
            f"VaR {int(p['conf']*100)}%":  var_hist(r, p["conf"]),
            f"CVaR {int(p['conf']*100)}%": cvar_hist(r, p["conf"]),
            "Treynor":              treynor(r, bench_ret, p["rfr"]),
            "Calmar":               calmar(r, pr),
            "Info. Ratio":          info_ratio(r, bench_ret.reindex(r.index).dropna()),
        })
    risk_df = pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()

    # ── VIX compact strip at top ──────────────────────────────
    vix_col, _ = st.columns([1, 3])
    with vix_col:
        vix_widget(vix_df)

    # ============================================================
    # TABS
    # ============================================================
    (t_overview, t_bench, t_corr, t_cov,
     t_risk, t_sim, t_export) = st.tabs([
        "Overview", "Benchmark", "Correlation",
        "Covariance", "Risk Metrics", "Simulator", "Export",
    ])

    # ─────────────────────────────────────────────────────────
    # OVERVIEW
    # ─────────────────────────────────────────────────────────
    with t_overview:
        st.markdown(f"<div class='sec-hdr'>Selected Assets — {len(valid)}</div>",
                    unsafe_allow_html=True)

        # KPI cards
        kpi_cols = st.columns(min(len(valid), 4))
        for i, t in enumerate(valid):
            info_d = fetch_info(t)
            pr     = prices_df[t].dropna()
            rt     = returns_df[t].dropna() if t in returns_df else pd.Series()
            cp     = float(pr.iloc[-1]) if not pr.empty else 0
            pp     = float(pr.iloc[-2]) if len(pr) > 1 else cp
            dc     = (cp / pp - 1) * 100 if pp else 0
            dc_cls = "kpi-change-pos" if dc >= 0 else "kpi-change-neg"
            nm     = info_d.get("longName", t)[:22]
            bt     = beta_calc(rt, bench_ret) if len(rt) > 10 else np.nan
            with kpi_cols[i % 4]:
                st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-label">{nm}</div>
                  <div class="kpi-ticker">{t}</div>
                  <div class="kpi-price">${cp:,.2f}</div>
                  <div class="{dc_cls}">{"+" if dc>=0 else ""}{dc:.2f}%</div>
                  <div class="kpi-sub">
                    Vol: {ann_vol(rt):.1%} &nbsp;|&nbsp;
                    β: {"N/A" if np.isnan(bt) else f"{bt:.2f}"}
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Animated cumulative return (ECharts)
        st.markdown("<div class='sec-hdr'>Cumulative Return — Animated</div>",
                    unsafe_allow_html=True)
        norm_dict   = {}
        for t in valid:
            s = prices_df[t].dropna()
            norm_dict[t] = (s / s.iloc[0] * 100).round(2).tolist()
        date_labels = [_date_str(d) for d in prices_df.index]
        if ECHARTS_OK:
            st_echarts(options=echarts_area_animated(date_labels, norm_dict,
                                                     "Normalized Price (Base 100)"),
                       height="320px")
        elif PLOTLY_OK:
            st.plotly_chart(plotly_perf(prices_df, bench_ser, p["bench_name"]),
                            use_container_width=True)

        st.markdown("---")

        # Heikin Ashi chart
        st.markdown("<div class='sec-hdr'>Price Chart</div>", unsafe_allow_html=True)
        sel = st.selectbox("Asset:", valid, key="chart_sel")
        chart_df = hist[sel].copy()

        with st.spinner("Computing ML signals & forecast…"):
            sigs = ml_signals(chart_df) if p["show_sigs"] else None
            fc   = forecast_3m(chart_df["Close"])

        if ECHARTS_OK:
            candle_opt = echarts_candle(chart_df, sel, p["ema_cfg"],
                                        p["custom_emas"], p["show_bb"],
                                        p["show_sigs"], sigs, fc)
            st_echarts(options=candle_opt, height="620px")
        elif PLOTLY_OK:
            ha    = heikin_ashi(chart_df)
            fig_c = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  row_heights=[0.75, 0.25], vertical_spacing=0.02)
            fig_c.add_trace(go.Candlestick(
                x=ha.index, open=ha.HA_Open, high=ha.HA_High,
                low=ha.HA_Low, close=ha.HA_Close, name="HA",
                increasing_line_color="#00e87a", decreasing_line_color="#ff4545"), row=1, col=1)
            fig_c.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"],
                                   name="Vol", marker_color="#2a2a4a"), row=2, col=1)
            fig_c.update_layout(**_dark_layout(height=500),
                                xaxis_rangeslider_visible=False,
                                title=f"{sel} — Heikin Ashi")
            st.plotly_chart(fig_c, use_container_width=True)

        # Signal cards & insights
        st.markdown("<div class='sec-hdr'>Signal Analysis &amp; Insights</div>",
                    unsafe_allow_html=True)
        for t in valid:
            info_d = fetch_info(t)
            rec    = recommend(hist[t], info_d, bench_ret, p["rfr"])
            with st.expander(f"**{t}** — {info_d.get('longName', t)}", expanded=(t == valid[0])):
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.markdown("**1 Month**")
                    st.markdown(f"<div class='sig-{rec['cls_1m']}'>{rec['rec_1m']}</div>",
                                unsafe_allow_html=True)
                with rc2:
                    st.markdown("**3 Months**")
                    st.markdown(f"<div class='sig-{rec['cls_3m']}'>{rec['rec_3m']}</div>",
                                unsafe_allow_html=True)
                with rc3:
                    st.markdown("**6 Months**")
                    st.markdown(f"<div class='sig-{rec['cls_6m']}'>{rec['rec_6m']}</div>",
                                unsafe_allow_html=True)

                ic1, ic2 = st.columns(2)
                with ic1:
                    if rec["pros"]:
                        st.markdown("**Factors in favor:**")
                        for r in rec["pros"]:
                            st.markdown(f"✅ {r}")
                with ic2:
                    if rec["cons"]:
                        st.markdown("**Factors against:**")
                        for r in rec["cons"]:
                            st.markdown(f"⚠️ {r}")
                if rec["risks"]:
                    st.markdown("**Key risks:**")
                    for r in rec["risks"]:
                        st.markdown(f"🔴 {r}")

                val_cls = ("buy" if rec["valuation"] == "undervalued"
                           else "sell" if rec["valuation"] == "overvalued" else "hold")
                st.markdown(
                    f"<div class='sig-{val_cls}' style='margin-top:8px;'>"
                    f"Valuation: {rec['valuation'].upper()}</div>",
                    unsafe_allow_html=True)

                news = fetch_news(t)
                if news:
                    st.markdown("**Recent News:**")
                    for article in news:
                        title_n  = article.get("title", "")
                        link_n   = article.get("link", "#")
                        pub_n    = article.get("publisher", "")
                        st.markdown(
                            f'<div class="news-card">'
                            f'<a class="news-link" href="{link_n}" target="_blank">{title_n}</a>'
                            f'<div class="news-pub">{pub_n}</div></div>',
                            unsafe_allow_html=True)

        with st.expander("Valuation Methodology & Formulas"):
            st.markdown(f"""
**Annualization basis: {TRADING_DAYS} trading days / year.**

| Metric | Formula |
|--------|---------|
| Annualized Return | $(1+\\bar{{r}})^{{{TRADING_DAYS}}}-1$ |
| Annualized Volatility | $\\sigma\\cdot\\sqrt{{{TRADING_DAYS}}}$ |
| Sharpe Ratio | $(R_p-R_f)/\\sigma_p\\cdot\\sqrt{{{TRADING_DAYS}}}$ |
| Sortino Ratio | $(R_p-R_f)/\\sigma_{{down}}\\cdot\\sqrt{{{TRADING_DAYS}}}$ |
| Beta | $\\text{{Cov}}(R_i,R_m)/\\text{{Var}}(R_m)$ |
| Jensen's Alpha | $R_i-[R_f+\\beta(R_m-R_f)]$ |
| VaR (Historical) | $\\alpha$-th percentile of loss distribution |
| CVaR | $E[R\\mid R\\leq\\text{{VaR}}]$ |
| Treynor | $(R_p-R_f)/\\beta$ |
| Calmar | $R_p/|\\text{{Max Drawdown}}|$ |
| Information Ratio | $\\overline{{R_{{active}}}}/\\sigma_{{active}}\\cdot\\sqrt{{{TRADING_DAYS}}}$ |

> Signals combine RSI, EMA crossovers, Bollinger Band position, MACD, momentum and fundamentals.
> ML signals use Random Forest. The 3-month forecast uses ARIMA(5,1,0) on log-prices.
> **Not financial advice.**
""")

    # ─────────────────────────────────────────────────────────
    # BENCHMARK
    # ─────────────────────────────────────────────────────────
    with t_bench:
        st.markdown(f"<div class='sec-hdr'>Performance vs {p['bench_name']}</div>",
                    unsafe_allow_html=True)
        if bench_ser.empty:
            st.warning("Benchmark data unavailable.")
        else:
            if PLOTLY_OK:
                st.plotly_chart(plotly_perf(prices_df, bench_ser, p["bench_name"]),
                                use_container_width=True)
                if len(returns_df) > 60:
                    st.plotly_chart(plotly_rolling_beta(returns_df, bench_ret),
                                    use_container_width=True)
            if not risk_df.empty:
                st.markdown("<div class='sec-hdr'>Summary Table</div>", unsafe_allow_html=True)
                disp = risk_df[["Ann. Return","Ann. Volatility","Sharpe Ratio",
                                "Beta","Alpha (Jensen)","Max Drawdown"]].copy()
                fmt = {"Ann. Return":"{:.2%}","Ann. Volatility":"{:.2%}",
                       "Max Drawdown":"{:.2%}","Sharpe Ratio":"{:.3f}",
                       "Beta":"{:.3f}","Alpha (Jensen)":"{:.3f}"}
                st.dataframe(disp.style.format(fmt), use_container_width=True)

    # ─────────────────────────────────────────────────────────
    # CORRELATION
    # ─────────────────────────────────────────────────────────
    with t_corr:
        st.markdown("<div class='sec-hdr'>Returns Correlation Matrix</div>",
                    unsafe_allow_html=True)
        if len(valid) < 2:
            st.info("Add at least 2 tickers to view correlation.")
        else:
            corr = returns_df.corr()

            # ECharts heatmap
            if ECHARTS_OK:
                st_echarts(options=echarts_heatmap(corr, "Correlation Matrix"),
                           height="480px")
            elif PLOTLY_OK:
                fig_cr = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
                    colorscale=[[0,"#ff4545"],[.5,"#0d0d1a"],[1,"#00e87a"]],
                    zmid=0, text=corr.values.round(3), texttemplate="%{text}",
                    textfont=dict(size=11)))
                fig_cr.update_layout(title="Correlation Matrix", **_dark_layout(height=460))
                st.plotly_chart(fig_cr, use_container_width=True)

            st.dataframe(corr.style.format("{:.3f}"), use_container_width=True)

            st.markdown("---")

            # ── Rolling 30-day correlation vs benchmark ────────
            if PLOTLY_OK and not bench_ret.empty:
                st.markdown("<div class='sec-hdr'>Rolling 30-Day Correlation vs Benchmark</div>",
                            unsafe_allow_html=True)
                fig_rc = go.Figure()
                for i, t in enumerate(valid):
                    rc = returns_df[t].rolling(30).corr(bench_ret)
                    fig_rc.add_trace(go.Scatter(
                        x=rc.index, y=rc, name=t,
                        line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], width=1.5)))
                fig_rc.add_hline(y=0, line_color="#333", line_dash="dash")
                fig_rc.update_layout(
                    title=f"Rolling 30-Day Correlation vs {p['bench_name']}",
                    yaxis_range=[-1, 1], **_dark_layout(height=280))
                st.plotly_chart(fig_rc, use_container_width=True)

            st.markdown("---")

            # ── Pair Returns Scatter ───────────────────────────
            st.markdown("<div class='sec-hdr'>Pair Returns Scatter</div>",
                        unsafe_allow_html=True)
            pair_c1, pair_c2 = st.columns(2)
            t1_sel = pair_c1.selectbox("Asset X", valid, index=0, key="pair_x")
            t2_sel = pair_c2.selectbox("Asset Y", valid,
                                        index=min(1, len(valid)-1), key="pair_y")
            if t1_sel != t2_sel and PLOTLY_OK:
                fig_pair = plotly_pair_scatter(returns_df, t1_sel, t2_sel)
                if fig_pair:
                    st.plotly_chart(fig_pair, use_container_width=True)
            elif t1_sel == t2_sel:
                st.info("Select two different assets to compare.")

            # Scatter matrix (≤6 assets)
            if PLOTLY_OK and len(valid) <= 6:
                st.markdown("---")
                st.plotly_chart(plotly_scatter_matrix(returns_df), use_container_width=True)

    # ─────────────────────────────────────────────────────────
    # COVARIANCE
    # ─────────────────────────────────────────────────────────
    with t_cov:
        st.markdown("<div class='sec-hdr'>Annualized Covariance Matrix</div>",
                    unsafe_allow_html=True)
        if len(valid) < 2:
            st.info("Add at least 2 tickers to view covariance.")
        else:
            cov = returns_df.cov() * TRADING_DAYS
            if ECHARTS_OK:
                cov_opt = echarts_heatmap(cov, "Annualized Covariance",
                                          color_min="#ffd700", color_max="#ff4545")
                cov_opt["visualMap"]["min"] = float(cov.values.min())
                cov_opt["visualMap"]["max"] = float(cov.values.max())
                st_echarts(options=cov_opt, height="480px")
            elif PLOTLY_OK:
                fig_cv = go.Figure(go.Heatmap(
                    z=cov.values, x=cov.columns.tolist(), y=cov.columns.tolist(),
                    colorscale="YlOrRd",
                    text=cov.values.round(5), texttemplate="%{text}"))
                fig_cv.update_layout(title="Annualized Covariance", **_dark_layout(height=460))
                st.plotly_chart(fig_cv, use_container_width=True)

            st.dataframe(cov.style.format("{:.6f}"), use_container_width=True)

            if PLOTLY_OK:
                st.plotly_chart(plotly_vol_bar(returns_df), use_container_width=True)

            st.markdown("---")

            # ── Monte Carlo Efficient Frontier ─────────────────
            st.markdown("<div class='sec-hdr'>Monte Carlo Efficient Frontier</div>",
                        unsafe_allow_html=True)

            mc_n = st.select_slider("Number of simulated portfolios:",
                                    options=[1000, 2000, 5000, 10000], value=5000)

            with st.spinner(f"Simulating {mc_n:,} random portfolios…"):
                mc = monte_carlo_frontier(returns_df, n_portfolios=mc_n, rfr=p["rfr"])

            if mc and PLOTLY_OK:
                st.plotly_chart(plotly_mc_frontier(mc), use_container_width=True)

                # Optimal portfolio weights
                msi = mc["max_sh_idx"]
                mvi = mc["min_v_idx"]
                w_sh = mc["weights"][msi]
                w_mv = mc["weights"][mvi]

                opt_c1, opt_c2 = st.columns(2)
                with opt_c1:
                    st.markdown("**⭐ Max Sharpe Portfolio**")
                    for t, w in zip(mc["tickers"], w_sh):
                        st.markdown(f"- `{t}` — {w:.1%}")
                    st.caption(f"Vol: {mc['vols'][msi]:.1f}%  |  "
                               f"Ret: {mc['rets'][msi]:.1f}%  |  "
                               f"Sharpe: {mc['sharpes'][msi]:.3f}")
                with opt_c2:
                    st.markdown("**◆ Min Volatility Portfolio**")
                    for t, w in zip(mc["tickers"], w_mv):
                        st.markdown(f"- `{t}` — {w:.1%}")
                    st.caption(f"Vol: {mc['vols'][mvi]:.1f}%  |  "
                               f"Ret: {mc['rets'][mvi]:.1f}%  |  "
                               f"Sharpe: {mc['sharpes'][mvi]:.3f}")

                # Interpretation
                st.markdown(f"""
<div class="mc-box">
<strong>How to read this chart</strong><br><br>
Each point represents a randomly generated portfolio allocating capital among the
<strong>{len(mc['tickers'])} selected assets</strong>.
The <strong>color gradient</strong> shows the Sharpe ratio — <span style="color:#00e87a">green = high
risk-adjusted return</span>, <span style="color:#ff4545">red = low</span>.<br><br>
• The <span style="color:#00e87a">★ star</span> marks the <strong>Maximum Sharpe portfolio</strong>
— the allocation with the best return per unit of risk taken. This is generally the theoretically
optimal portfolio under Modern Portfolio Theory.<br>
• The <span style="color:#4da6ff">◆ diamond</span> marks the <strong>Minimum Volatility portfolio</strong>
— the most conservative allocation, minimizing total risk regardless of return.<br><br>
The curved upper boundary of the cloud is the <strong>Efficient Frontier</strong> — portfolios
<em>above and to the left</em> are mathematically impossible; portfolios <em>below and to the right</em>
are suboptimal (you could achieve the same return with less risk). Any rational investor should aim
to hold a portfolio on or near the frontier.
</div>
""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # RISK METRICS
    # ─────────────────────────────────────────────────────────
    with t_risk:
        st.markdown("<div class='sec-hdr'>Risk &amp; Return Metrics</div>",
                    unsafe_allow_html=True)
        if not risk_df.empty:
            pct_cols = ["Ann. Return","Ann. Volatility","Max Drawdown",
                        f"VaR {int(p['conf']*100)}%", f"CVaR {int(p['conf']*100)}%"]
            fmt = {c: "{:.2%}" for c in pct_cols if c in risk_df.columns}
            for c in ["Sharpe Ratio","Sortino Ratio","Beta","Alpha (Jensen)",
                      "Treynor","Calmar","Info. Ratio"]:
                if c in risk_df.columns:
                    fmt[c] = "{:.3f}"
            st.dataframe(risk_df.style.format(fmt), use_container_width=True, height=320)

            if PLOTLY_OK:
                st.plotly_chart(plotly_rr_scatter(risk_df, p["rfr"], bench_ret),
                                use_container_width=True)
                st.plotly_chart(plotly_drawdown(prices_df), use_container_width=True)

        # VIX — most fitting here as a volatility/risk indicator
        if not vix_df.empty and PLOTLY_OK:
            st.markdown("<div class='sec-hdr'>CBOE Volatility Index (VIX)</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(plotly_vix(vix_df), use_container_width=True)
            st.caption(
                "VIX < 15: low fear · 15–25: normal · 25–35: elevated risk-off · "
                "> 35: extreme stress. Peaks in VIX historically coincide with "
                "market bottoms and potential entry points."
            )

    # ─────────────────────────────────────────────────────────
    # SIMULATOR
    # ─────────────────────────────────────────────────────────
    with t_sim:
        st.markdown("<div class='sec-hdr'>Fundamental Analysis &amp; Valuation Simulator</div>",
                    unsafe_allow_html=True)
        sel_sim = st.selectbox("Select asset:", valid, key="sim_sel")
        info_s  = fetch_info(sel_sim)

        if not info_s:
            st.warning("Fundamental data not available.")
        else:
            nm  = info_s.get("longName", sel_sim)
            sec = info_s.get("sector", "N/A")
            ind = info_s.get("industry", "N/A")
            ctr = info_s.get("country", "N/A")
            st.markdown(f"""
            <div class="kpi-card">
              <h3 style="color:#ffd700;margin:0;">{nm}</h3>
              <div style="color:#3a3a5a;font-size:.82em;">{sec} · {ind} · {ctr}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("#### Key Fundamentals")
            fund_map = [
                ("Market Cap",    info_s.get("marketCap"),         "cap"),
                ("P/E (TTM)",     info_s.get("trailingPE"),        "ratio"),
                ("Forward P/E",   info_s.get("forwardPE"),         "ratio"),
                ("P/B",           info_s.get("priceToBook"),       "ratio"),
                ("EV/EBITDA",     info_s.get("enterpriseToEbitda"),"ratio"),
                ("EPS (TTM)",     info_s.get("trailingEps"),       "dollar"),
                ("Forward EPS",   info_s.get("forwardEps"),        "dollar"),
                ("Revenue TTM",   info_s.get("totalRevenue"),      "cap"),
                ("Gross Margin",  info_s.get("grossMargins"),      "pct"),
                ("Op. Margin",    info_s.get("operatingMargins"),  "pct"),
                ("Net Margin",    info_s.get("profitMargins"),     "pct"),
                ("ROE",           info_s.get("returnOnEquity"),    "pct"),
                ("ROA",           info_s.get("returnOnAssets"),    "pct"),
                ("Debt/Equity",   info_s.get("debtToEquity"),      "ratio"),
                ("Current Ratio", info_s.get("currentRatio"),      "ratio"),
                ("Free Cash Flow",info_s.get("freeCashflow"),      "cap"),
                ("Div. Yield",    info_s.get("dividendYield"),     "pct"),
                ("Beta",          info_s.get("beta"),              "ratio"),
                ("52W High",      info_s.get("fiftyTwoWeekHigh"),  "dollar"),
                ("52W Low",       info_s.get("fiftyTwoWeekLow"),   "dollar"),
            ]
            f_cols = st.columns(4)
            for i, (lbl, val, typ) in enumerate(fund_map):
                with f_cols[i % 4]:
                    if val is None:
                        st.metric(lbl, "N/A")
                    elif typ == "cap":
                        v = abs(val)
                        s = (f"${v/1e12:.2f}T" if v >= 1e12
                             else f"${v/1e9:.2f}B" if v >= 1e9
                             else f"${v/1e6:.2f}M")
                        st.metric(lbl, s)
                    elif typ == "pct":
                        st.metric(lbl, f"{val:.2%}")
                    elif typ == "dollar":
                        st.metric(lbl, f"${val:.2f}")
                    else:
                        st.metric(lbl, f"{val:.2f}")

            st.markdown("---")
            st.markdown("#### Discounted Cash Flow Valuation")
            dcf = simple_dcf(info_s, p["dcf_g"], p["dcf_wacc"], p["dcf_tg"], p["dcf_yrs"])
            if dcf:
                dc1, dc2, dc3 = st.columns(3)
                with dc1:
                    st.metric("DCF Intrinsic Value", f"${dcf['iv']:.2f}")
                with dc2:
                    cp_val = dcf["price"] or 0
                    st.metric("Current Price", f"${cp_val:.2f}")
                with dc3:
                    up = dcf["upside"] or 0
                    st.metric("Upside / Downside", f"{up:.1%}", delta=f"{up:.1%}")

                up_v = dcf["upside"] or 0
                if up_v > 0.15:
                    tag, cls2 = "UNDERVALUED — DCF signals meaningful upside", "buy"
                elif up_v < -0.15:
                    tag, cls2 = "OVERVALUED — DCF signals downside vs intrinsic value", "sell"
                else:
                    tag, cls2 = "FAIRLY VALUED — Priced near DCF intrinsic value", "hold"
                st.markdown(
                    f"<div class='sig-{cls2}' style='margin:10px 0;'>{tag}</div>",
                    unsafe_allow_html=True)

                if PLOTLY_OK:
                    fig_dcf = go.Figure(go.Waterfall(
                        orientation="v",
                        measure=["relative", "relative", "total"],
                        x=["PV of FCF", "PV of Terminal Value", "Total Equity Value"],
                        y=[dcf["pv_flows"], dcf["terminal"], dcf["total"]],
                        text=[f"${dcf['pv_flows']/1e9:.1f}B",
                              f"${dcf['terminal']/1e9:.1f}B",
                              f"${dcf['total']/1e9:.1f}B"],
                        increasing=dict(marker=dict(color="#00e87a")),
                        totals=dict(marker=dict(color="#ffd700")),
                        connector=dict(line=dict(color="#1e1e3a")),
                    ))
                    fig_dcf.update_layout(
                        title=f"DCF Value Bridge — {sel_sim}", **_dark_layout(height=340))
                    st.plotly_chart(fig_dcf, use_container_width=True)
            else:
                st.info("DCF requires positive Free Cash Flow — not available for this ticker.")

            # EPS history
            try:
                ed = yf.Ticker(sel_sim).earnings_dates
                if ed is not None and not ed.empty and "Reported EPS" in ed.columns and PLOTLY_OK:
                    clean = ed.dropna(subset=["Reported EPS"]).head(8)
                    if not clean.empty:
                        c_eps = ["#00e87a" if v >= 0 else "#ff4545"
                                 for v in clean["Reported EPS"]]
                        fig_eps = go.Figure(go.Bar(
                            x=[str(d.date()) for d in clean.index],
                            y=clean["Reported EPS"],
                            marker_color=c_eps,
                            text=[f"${v:.2f}" for v in clean["Reported EPS"]],
                            textposition="outside",
                        ))
                        fig_eps.update_layout(
                            title="Reported EPS by Quarter", **_dark_layout(height=300))
                        st.markdown("#### EPS History")
                        st.plotly_chart(fig_eps, use_container_width=True)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────
    with t_export:
        st.markdown("<div class='sec-hdr'>Export Analysis to Excel</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "Generates a multi-sheet workbook with price data, returns, risk metrics, "
            "correlation and annualized covariance matrices.")
        if st.button("📥 Build & Download Excel Report", type="primary"):
            with st.spinner("Generating report…"):
                xls   = build_excel(hist, risk_df, returns_df)
                fname = (f"analysis_{'_'.join(valid[:5])}_"
                         f"{datetime.today().strftime('%Y%m%d')}.xlsx")
                st.download_button(
                    label="Download Excel Report",
                    data=xls,
                    file_name=fname,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        st.markdown("""
**Report sheets:**
- One sheet per ticker — full OHLCV history
- `Returns` — daily return matrix
- `Risk_Metrics` — Sharpe, Sortino, Beta, Alpha, VaR, CVaR, Calmar, Treynor, Info. Ratio
- `Correlation` — pairwise return correlations
- `Covariance_Ann` — annualized covariance matrix
""")


if __name__ == "__main__":
    main()
