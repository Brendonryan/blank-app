import streamlit as st
import time, json, pathlib, threading, queue, pickle
from datetime import datetime
from typing import List, Dict, Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ===== Config =====
BITVAVO_HTTP = "https://api.bitvavo.com/v2"
BITVAVO_WS   = "wss://ws.bitvavo.com/v2/"
LOCAL_TZ = ZoneInfo("Europe/Amsterdam")

REQUEST_TIMEOUT = 6
MAX_RETRIES     = 2
BACKOFF_START   = 0.4

# ===== Page =====
st.set_page_config(page_title="Bitvavo Pro Volume Cockpit", layout="wide")
st.title("📈 Bitvavo Pro Volume Cockpit — Monitor-First (Pro Metrics)")

# ===== State =====
ss = st.session_state
ss.setdefault("watch_latency", [])
ss.setdefault("watch_errors", 0)
ss.setdefault("last_good_raw", None)
ss.setdefault("history", {})
ss.setdefault("ws_queue", queue.Queue())

# ===== Helpers =====
def to_float(x) -> float:
    try: return float(x)
    except: return 0.0

def human_amount(x: float) -> str:
    try: x = float(x)
    except: return "—"
    ax = abs(x)
    if ax >= 1_000_000_000: return f"{x/1_000_000_000:.2f}B"
    if ax >= 1_000_000:     return f"{x/1_000_000:.2f}M"
    if ax >= 1_000:         return f"{x/1_000:.2f}K"
    return f"{x:.2f}"

def human_million(x: float) -> str:
    try: x = float(x)
    except: return "—"
    return f"{x/1_000_000:.2f}M"

def fetch_json(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    headers = {"User-Agent": "BitvavoVolumeCockpit/pro"}
    err = None
    for i in range(MAX_RETRIES):
        t0 = time.time()
        try:
            r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            ss.watch_latency.append(time.time() - t0)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            err = e
            time.sleep(BACKOFF_START*(2**i))
    ss.watch_errors += 1
    raise RuntimeError(f"HTTP fout na {MAX_RETRIES} pogingen ({url}) · {err}")

@st.cache_data(ttl=600)
def fetch_candles(market: str, interval: str = "1h", limit: int = 200) -> pd.DataFrame:
    try:
        data = fetch_json(f"{BITVAVO_HTTP}/{market}/candles", {"interval": interval, "limit": limit})
        if not isinstance(data, list) or not data:
            return pd.DataFrame()
        c = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"]).apply(pd.to_numeric, errors="ignore")
        c["time"] = pd.to_datetime(c["ts"], unit="ms", utc=True).dt.tz_convert(LOCAL_TZ)
        return c.sort_values("time").reset_index(drop=True)
    except: return pd.DataFrame()

# ===== Metrics helpers =====
def realized_vol_24h_annualized(c: pd.DataFrame) -> float:
    if c.empty or len(c) < 26: return np.nan
    r = np.log(c["close"]).diff().dropna().tail(24)
    if r.empty: return np.nan
    return float(r.std(ddof=0) * np.sqrt(24*365) * 100.0)

def trend_apr_annualized(c: pd.DataFrame) -> float:
    if c.empty or len(c) < 26: return np.nan
    r = np.log(c["close"]).diff().dropna().tail(24)
    if r.empty: return np.nan
    mu_hour = r.mean()
    return float((np.exp(mu_hour * 24 * 365) - 1.0) * 100.0)

# ===== Sidebar =====
with st.sidebar:
    st.subheader("⚙️ Instellingen")
    interval = st.number_input("Auto-refresh (s)", 3, 120, 10, 1)
    eur_only = st.checkbox("Alleen EUR-paren", True)
    pro_metrics = st.toggle("Pro metrics in Top-tabel (Trend APR, Vol, VAF, Liq)", True)

# ===== Snapshot =====
raw = fetch_json(f"{BITVAVO_HTTP}/ticker/24h")
rows = []
for t in raw if isinstance(raw, list) else []:
    m = (t.get("market") or t.get("symbol") or "").strip()
    if not m: continue
    if eur_only and not m.endswith("-EUR"): continue
    last = to_float(t.get("last") or t.get("price"))
    base_vol = to_float(t.get("volume"))
    qv = to_float(t.get("quoteVolume")) if t.get("quoteVolume") else base_vol * last
    openp = to_float(t.get("open"))
    chg = ((last-openp)/openp*100.0) if openp else 0.0
    rows.append({"market":m,"last":last,"change_pct":chg,"quote_vol":qv,"base_vol":base_vol})

df = pd.DataFrame(rows)
if df.empty:
    st.error("Geen markten.")
    st.stop()

df["quote_share_pct"] = df["quote_vol"]/max(df["quote_vol"].sum(), 1e-12)*100.0

# ===== Pro metrics =====
vol_real_24h_map, trend_apr_map, liq_q_05_map = {}, {}, {}
if pro_metrics:
    for m in df.sort_values("quote_vol", ascending=False).head(15)["market"]:
        c = fetch_candles(m, "1h", 200)
        vol_real_24h_map[m] = realized_vol_24h_annualized(c)
        trend_apr_map[m]    = trend_apr_annualized(c)
        # liq ±0.5%
        try:
            ob = fetch_json(f"{BITVAVO_HTTP}/{m}/book", {"depth":50})
            bids = [(to_float(p), to_float(a)) for p,a,*_ in ob.get("bids", [])]
            asks = [(to_float(p), to_float(a)) for p,a,*_ in ob.get("asks", [])]
            if bids and asks:
                mid = (bids[0][0]+asks[0][0])/2
                lo, hi = mid*0.995, mid*1.005
                q_bids = sum(p*a for p,a in bids if p>=lo)
                q_asks = sum(p*a for p,a in asks if p<=hi)
                liq_q_05_map[m] = q_bids+q_asks
        except: liq_q_05_map[m] = np.nan

df["vol_real_24h_ann_pct"] = df["market"].map(lambda m: vol_real_24h_map.get(m))
df["trend_apr_pct"]        = df["market"].map(lambda m: trend_apr_map.get(m))
df["liq_q05"]              = df["market"].map(lambda m: liq_q_05_map.get(m))

if pro_metrics:
    df["vaf"] = df.apply(lambda r: (r["quote_share_pct"] / r["vol_real_24h_ann_pct"]) if r.get("vol_real_24h_ann_pct") and r["vol_real_24h_ann_pct"]>0 else np.nan, axis=1)
    # Rank score
    def zscore(s):
        s = pd.to_numeric(s, errors="coerce")
        mu, sd = s.mean(), s.std(ddof=0) or 1.0
        return (s - mu)/sd
    _flow_z = zscore(df["quote_share_pct"])
    _apr_z  = zscore(df["trend_apr_pct"])
    _vol_z  = zscore(df["vol_real_24h_ann_pct"])
    _rank   = 0.5*_flow_z + 0.3*_apr_z - 0.2*_vol_z
    _rank_min, _rank_ptp = float(np.nanmin(_rank)), float(np.nanmax(_rank)-np.nanmin(_rank) or 1.0)
    df["rank_score"] = ((_rank - _rank_min)/_rank_ptp*100.0).round(1)

# ===== Table view =====
view = df.sort_values("rank_score" if pro_metrics else "quote_vol", ascending=False).copy()
view["chg_d"] = view["change_pct"].map(lambda x: f"{x:+.2f}%")
view["flow_d"] = view["quote_share_pct"].map(lambda x: f"{x:.2f}%")
view["vol_real_d"] = view["vol_real_24h_ann_pct"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
view["trend_apr_d"] = view["trend_apr_pct"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
view["liq_d"] = view["liq_q05"].map(lambda x: human_amount(x) if pd.notna(x) else "—")
view["vaf_d"] = view["vaf"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
view["rank_d"] = view["rank_score"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

cols = ["rank_d","market","last","chg_d","trend_apr_d","vol_real_d","flow_d","liq_d","vaf_d"]

st.subheader("🏆 Top markten — beslissingsgericht overzicht")
st.dataframe(view[cols].rename(columns={
    "rank_d":"Rank","chg_d":"24h Δ%","trend_apr_d":"Trend APR","vol_real_d":"Vol (ann.)","flow_d":"Flow %","liq_d":"Liq ±0.5% (quote)","vaf_d":"VAF"
}), use_container_width=True, hide_index=True)

# ===== Footer =====
st.caption("Bron: Bitvavo publieke API. Geen financieel advies.")
ms = max(3000, int(float(interval)*1000))
st_autorefresh(interval=ms, key="datarefresh")
