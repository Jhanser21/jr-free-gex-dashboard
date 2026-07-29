from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="JR GEX / Flow Read", page_icon="⚡", layout="wide")

CONTRACT_MULTIPLIER = 100


# -----------------------------
# Core calculations
# -----------------------------
def safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def normal_pdf(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / math.sqrt(2.0 * math.pi)


def bs_gamma(spot: float, strike: np.ndarray, t: float, rate: float, iv: np.ndarray) -> np.ndarray:
    strike = np.asarray(strike, dtype=float)
    iv = np.asarray(iv, dtype=float)
    valid = (spot > 0) & (strike > 0) & (iv > 0) & (t > 0)
    output = np.zeros_like(strike, dtype=float)
    if not np.any(valid):
        return output

    sqrt_t = math.sqrt(t)
    d1 = (np.log(spot / strike[valid]) + (rate + 0.5 * iv[valid] ** 2) * t) / (
        iv[valid] * sqrt_t
    )
    output[valid] = normal_pdf(d1) / (spot * iv[valid] * sqrt_t)
    return output


@st.cache_data(ttl=300, show_spinner=False)
def load_market_data(symbol: str):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="5d", interval="1m", prepost=True, auto_adjust=False)
    if hist.empty:
        hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
    if hist.empty or hist["Close"].dropna().empty:
        raise RuntimeError(f"No price data returned for {symbol}.")

    closes = hist["Close"].dropna()
    spot = safe_float(closes.iloc[-1])
    previous = safe_float(closes.iloc[-2], spot) if len(closes) > 1 else spot
    change = spot - previous
    change_pct = (change / previous * 100.0) if previous else 0.0
    expirations = list(ticker.options)
    if not expirations:
        raise RuntimeError(f"No option expirations returned for {symbol}.")
    return spot, change, change_pct, expirations


@st.cache_data(ttl=300, show_spinner=False)
def load_chain(symbol: str, expirations: tuple[str, ...]) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    frames = []
    now = datetime.now(timezone.utc)

    for expiration in expirations:
        chain = ticker.option_chain(expiration)
        expiry_dt = datetime.strptime(expiration, "%Y-%m-%d").replace(hour=21, tzinfo=timezone.utc)
        days = max((expiry_dt - now).total_seconds() / 86400.0, 1.0 / 24.0)

        for option_type, frame in (("call", chain.calls), ("put", chain.puts)):
            if frame is None or frame.empty:
                continue
            data = frame.copy()
            data["optionType"] = option_type
            data["expiration"] = expiration
            data["daysToExpiry"] = days
            frames.append(data)

    if not frames:
        raise RuntimeError("No option-chain rows were returned.")

    result = pd.concat(frames, ignore_index=True)
    for col in ["strike", "openInterest", "impliedVolatility", "volume", "bid", "ask", "lastPrice"]:
        if col not in result.columns:
            result[col] = 0.0
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
    return result


def calculate_gex(chain: pd.DataFrame, spot: float, rate: float) -> pd.DataFrame:
    data = chain.copy()
    times = np.maximum(data["daysToExpiry"].to_numpy(float) / 365.0, 1 / (365 * 24))
    strikes = data["strike"].to_numpy(float)
    iv = data["impliedVolatility"].to_numpy(float)

    gamma = np.zeros(len(data))
    for unique_t in np.unique(times):
        mask = times == unique_t
        gamma[mask] = bs_gamma(spot, strikes[mask], float(unique_t), rate, iv[mask])

    sign = np.where(data["optionType"].eq("call"), 1.0, -1.0)
    data["gamma"] = gamma
    data["gex"] = sign * gamma * data["openInterest"] * CONTRACT_MULTIPLIER * spot**2 * 0.01
    return data


def aggregate_by_strike(data: pd.DataFrame) -> pd.DataFrame:
    gex = data.pivot_table(index="strike", columns="optionType", values="gex", aggfunc="sum", fill_value=0)
    oi = data.pivot_table(index="strike", columns="optionType", values="openInterest", aggfunc="sum", fill_value=0)
    vol = data.pivot_table(index="strike", columns="optionType", values="volume", aggfunc="sum", fill_value=0)

    out = pd.DataFrame(index=sorted(data["strike"].unique()))
    out["callGex"] = gex.get("call", 0)
    out["putGex"] = gex.get("put", 0)
    out["netGex"] = out["callGex"] + out["putGex"]
    out["callOI"] = oi.get("call", 0)
    out["putOI"] = oi.get("put", 0)
    out["volume"] = vol.get("call", 0) + vol.get("put", 0)
    return out.reset_index(names="strike").fillna(0).sort_values("strike")


def nearest_zero_gamma(frame: pd.DataFrame, spot: float) -> float | None:
    frame = frame.sort_values("strike")
    x = frame["strike"].to_numpy(float)
    y = frame["netGex"].to_numpy(float)
    candidates = []
    for i in range(len(x) - 1):
        if y[i] == 0:
            candidates.append(x[i])
        elif y[i] * y[i + 1] < 0:
            crossing = x[i] + (0 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            candidates.append(crossing)
    return min(candidates, key=lambda v: abs(v - spot)) if candidates else None


def max_pain(chain: pd.DataFrame) -> float | None:
    nearest_expiry = sorted(chain["expiration"].unique())[0]
    near = chain[chain["expiration"].eq(nearest_expiry)]
    strikes = np.sort(near["strike"].unique())
    if len(strikes) == 0:
        return None
    calls = near[near["optionType"].eq("call")][["strike", "openInterest"]]
    puts = near[near["optionType"].eq("put")][["strike", "openInterest"]]
    payouts = []
    for settlement in strikes:
        call_loss = ((settlement - calls["strike"]).clip(lower=0) * calls["openInterest"]).sum()
        put_loss = ((puts["strike"] - settlement).clip(lower=0) * puts["openInterest"]).sum()
        payouts.append(call_loss + put_loss)
    return float(strikes[int(np.argmin(payouts))])


def expected_move(chain: pd.DataFrame, spot: float):
    expiry = sorted(chain["expiration"].unique())[0]
    near = chain[chain["expiration"].eq(expiry)].copy()
    near["distance"] = (near["strike"] - spot).abs()
    atm_strike = float(near.loc[near["distance"].idxmin(), "strike"])
    atm = near[np.isclose(near["strike"], atm_strike)]

    def midpoint(option_type: str):
        row = atm[atm["optionType"].eq(option_type)]
        if row.empty:
            return 0.0
        bid, ask, last = [safe_float(row.iloc[0].get(c)) for c in ["bid", "ask", "lastPrice"]]
        return (bid + ask) / 2 if bid > 0 and ask > 0 else last

    move = midpoint("call") + midpoint("put")
    return (move if move > 0 else None), expiry


def compact(value: float) -> str:
    sign = "-" if value < 0 else ""
    n = abs(value)
    if n >= 1_000_000_000:
        return f"{sign}{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{sign}{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{sign}{n / 1_000:.0f}K"
    return f"{sign}{n:.0f}"


def price(value: float | None) -> str:
    return "N/A" if value is None else f"{value:,.2f}"


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&family=Inter:wght@600;700;800&display=swap');
:root {
  --bg:#070908; --panel:#111312; --line:#242824; --muted:#9b9f9c;
  --green:#00ff86; --red:#ff4d5c; --gold:#e2bd3d; --white:#f6f7f6;
}
html, body, [class*="css"] { font-family:'JetBrains Mono', monospace; }
.stApp {
  background-color:var(--bg);
  background-image:linear-gradient(rgba(35,40,36,.32) 1px,transparent 1px),linear-gradient(90deg,rgba(35,40,36,.32) 1px,transparent 1px);
  background-size:54px 54px;
  color:var(--white);
}
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#0b0d0c; border-right:1px solid #252925; }
.block-container { max-width:1500px; padding-top:1rem; padding-bottom:2rem; }
#MainMenu, footer { visibility:hidden; }
.header-shell,.panel,.analysis-card,.map-card,.mini-card { background:rgba(15,17,16,.96); border:1px solid var(--line); border-radius:11px; }
.header-shell { padding:17px 20px; margin-bottom:15px; box-shadow:0 0 0 1px rgba(255,255,255,.02) inset; }
.header-grid { display:grid; grid-template-columns:1.2fr 1.75fr .8fr; align-items:center; gap:18px; }
.symbol { font-family:Inter,sans-serif; font-size:42px; font-weight:800; letter-spacing:-2px; }
.live { color:var(--gold); font-size:22px; font-weight:800; margin-left:10px; }
.center-title { text-align:center; color:var(--gold); font-weight:800; font-size:16px; }
.control { border:2px solid #caa625; border-radius:9px; padding:9px 12px; text-align:center; }
.control small { color:#9c9a8e; font-size:8px; display:block; }
.control strong { font-family:Inter,sans-serif; font-size:29px; display:block; }
.control span { color:var(--red); font-size:13px; font-weight:800; }
.control em { display:block; color:var(--gold); font-size:7px; font-style:normal; font-weight:700; }
.panel { overflow:hidden; height:100%; }
.table-head,.table-row { display:grid; grid-template-columns:1fr 1.25fr 1.25fr 1.25fr; padding:9px 13px; align-items:center; }
.table-head { color:#777c78; border-bottom:1px solid #505450; font-size:12px; letter-spacing:.5px; }
.table-row { border-bottom:1px solid #4a4d4a; font-size:14px; font-weight:700; }
.table-row:last-child { border-bottom:0; }
.table-row span:not(:first-child) { text-align:right; }
.green { color:#70ff79; }.red { color:#ff5865; }.gold { color:#f2d71e; }.muted { color:#a9ada9; }
.row-call { background:rgba(0,255,90,.08); }.row-put { background:rgba(255,0,25,.12); }.row-zero { background:rgba(240,210,0,.10); }
.analysis-card { margin-bottom:13px; padding-bottom:9px; min-height:142px; }
.analysis-title { padding:5px 11px; border-radius:8px 8px 0 0; font-weight:800; font-size:13px; }
.analysis-body { color:#aaaead; padding:8px 11px 0; font-size:13px; line-height:1.55; }
.bull { border-color:#006d3b; background:rgba(0,39,20,.68); }.bull .analysis-title { color:var(--green); background:#003c20; }
.bear { border-color:#7c181c; background:rgba(45,2,4,.68); }.bear .analysis-title { color:var(--red); background:#450b0d; }
.chop { border-color:#6c5b00; background:rgba(37,31,0,.67); }.chop .analysis-title { color:#ffe000; background:#403600; }
.plan { border-color:#5e4d13; background:rgba(31,25,5,.78); min-height:132px; }.plan .analysis-title { color:#d4b54a; background:#3a300e; }
.section-title { font-family:Inter,sans-serif; font-weight:800; font-size:25px; margin:31px 0 12px 10px; }
.map-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:13px; }
.map-card { min-height:270px; padding:14px; }
.map-card h3 { margin:0 0 8px; font-size:14px; }.map-card b { font-size:12px; }.map-card p { color:#a7aaa8; font-size:12px; margin-top:7px; }
.map-bull { border-color:#007840; background:rgba(0,36,19,.72); }.map-bull h3{color:var(--green)}
.map-chop { border-color:#665500; background:rgba(35,29,0,.72); }.map-chop h3{color:#ffe000}
.map-bear { border-color:#74171b; background:rgba(41,2,4,.72); }.map-bear h3{color:var(--red)}
.footer-note { color:#353936; margin-top:70px; font-size:10px; }
[data-testid="stSelectbox"], [data-testid="stTextInput"], [data-testid="stNumberInput"] { color:white; }
.stButton button { background:#151915; color:#f3cf48; border:1px solid #79651f; font-weight:800; }
@media(max-width:900px){.header-grid{grid-template-columns:1fr}.center-title{text-align:left}.map-grid{grid-template-columns:1fr}.symbol{font-size:34px}}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Controls
# -----------------------------
with st.sidebar:
    st.markdown("## JR GEX SETTINGS")
    symbol = st.text_input("Ticker", "QQQ").strip().upper()
    expiration_count = st.slider("Expirations", 1, 10, 2)
    strike_count = st.slider("Rows around spot", 8, 30, 16)
    rate_pct = st.number_input("Risk-free rate (%)", 0.0, 15.0, 4.5, 0.1)
    manual_refresh = st.button("REFRESH DATA", use_container_width=True)
    st.caption("Free public chain data. Open interest may be delayed and dealer positioning is estimated.")

if manual_refresh:
    st.cache_data.clear()

if not symbol:
    st.stop()

try:
    with st.spinner(f"Loading {symbol} gamma structure..."):
        spot, change, change_pct, expirations = load_market_data(symbol)
        selected_expirations = tuple(expirations[:expiration_count])
        chain = load_chain(symbol, selected_expirations)
        gex_data = calculate_gex(chain, spot, rate_pct / 100)
        strikes = aggregate_by_strike(gex_data)
except Exception as exc:
    st.error(f"Unable to load {symbol}: {exc}")
    st.stop()

# Display nearest strikes around spot
strikes["distance"] = (strikes["strike"] - spot).abs()
visible = strikes.nsmallest(strike_count, "distance").sort_values("strike", ascending=False).copy()
analysis_frame = strikes[strikes["strike"].between(spot * 0.82, spot * 1.18)].copy()
if analysis_frame.empty:
    analysis_frame = strikes.copy()

call_wall_row = analysis_frame.loc[analysis_frame["callGex"].idxmax()]
put_wall_row = analysis_frame.loc[analysis_frame["putGex"].idxmin()]
control_row = analysis_frame.loc[analysis_frame["netGex"].abs().idxmax()]
call_wall = float(call_wall_row["strike"])
put_wall = float(put_wall_row["strike"])
control_node = float(control_row["strike"])
control_gex = float(control_row["netGex"])
zero_gamma = nearest_zero_gamma(analysis_frame, spot)
max_pain_level = max_pain(chain)
exp_move, nearest_expiry = expected_move(chain, spot)

# Structural thresholds and targets
above = sorted(analysis_frame.loc[analysis_frame["strike"] > spot, "strike"].unique())
below = sorted(analysis_frame.loc[analysis_frame["strike"] < spot, "strike"].unique(), reverse=True)
bull_trigger = zero_gamma if zero_gamma and zero_gamma > spot else call_wall
bear_trigger = zero_gamma if zero_gamma and zero_gamma < spot else put_wall
if bull_trigger <= spot:
    bull_trigger = above[0] if above else call_wall
if bear_trigger >= spot:
    bear_trigger = below[0] if below else put_wall
bull_targets = [x for x in above if x > bull_trigger][:3]
bear_targets = [x for x in below if x < bear_trigger][:3]
while len(bull_targets) < 3:
    bull_targets.append(call_wall)
while len(bear_targets) < 3:
    bear_targets.append(put_wall)

chop_low = min(bear_trigger, bull_trigger)
chop_high = max(bear_trigger, bull_trigger)
regime = "BULLISH" if spot > chop_high else "BEARISH" if spot < chop_low else "CHOP"

# Header
change_class = "green" if change >= 0 else "red"
header = f"""
<div class="header-shell">
  <div class="header-grid">
    <div><span class="symbol">${escape(symbol)}</span><span class="live">{spot:,.2f} LIVE</span></div>
    <div class="center-title">GEX / FLOW READ — Exp: {nearest_expiry}</div>
    <div class="control">
      <small>CONTROL NODE</small><strong>{control_node:,.0f}</strong>
      <span>{compact(control_gex)}</span>
      <em>MAJOR PIVOT — Largest Absolute Net GEX</em>
    </div>
  </div>
</div>
"""
st.markdown(header, unsafe_allow_html=True)

left, right = st.columns([0.42, 0.58], gap="medium")

# Strike table
with left:
    rows = []
    for _, row in visible.iterrows():
        strike = float(row["strike"])
        classes = ["table-row"]
        if math.isclose(strike, call_wall):
            classes.append("row-call")
        if math.isclose(strike, put_wall):
            classes.append("row-put")
        if zero_gamma and abs(strike - zero_gamma) == visible.assign(z=(visible["strike"] - zero_gamma).abs())["z"].min():
            classes.append("row-zero")

        strike_color = "green" if math.isclose(strike, call_wall) else "red" if math.isclose(strike, put_wall) else "gold" if zero_gamma and abs(strike-zero_gamma) < 0.51 else "muted"
        net_color = "green" if row["netGex"] >= 0 else "red"
        call_color = "green"
        put_color = "red"
        rows.append(
            f'<div class="{" ".join(classes)}"><span class="{strike_color}">{strike:,.0f}</span>'
            f'<span class="{net_color}">{compact(row["netGex"])}</span>'
            f'<span class="{call_color}">{compact(row["callGex"])}</span>'
            f'<span class="{put_color}">{compact(row["putGex"])}</span></div>'
        )

    table = (
        '<div class="panel"><div class="table-head"><span>STRIKE</span><span>NET GEX</span><span>CALL GEX</span><span>PUT GEX</span></div>'
        + "".join(rows)
        + "</div>"
    )
    st.markdown(table, unsafe_allow_html=True)

with right:
    bull_text = (
        f"Target 1: {bull_targets[0]:,.0f}<br>Target 2: {bull_targets[1]:,.0f}<br>Target 3: {bull_targets[2]:,.0f}<br>"
        f"Call Wall at {call_wall:,.0f} ({compact(call_wall_row['callGex'])}) — positive call gamma concentration.<br>"
        f"Acceptance above {bull_trigger:,.0f} favors a controlled push toward higher strikes."
    )
    bear_text = (
        f"Target 1: {bear_targets[0]:,.0f}<br>Target 2: {bear_targets[1]:,.0f}<br>Target 3: {bear_targets[2]:,.0f}<br>"
        f"Put Wall at {put_wall:,.0f} ({compact(put_wall_row['putGex'])}) — largest negative put-gamma concentration.<br>"
        f"Acceptance below {bear_trigger:,.0f} can increase directional volatility."
    )
    chop_text = (
        f"Range: {chop_low:,.0f} — {chop_high:,.0f}<br>"
        f"Spot at {spot:,.2f} is currently classified as <b>{regime}</b>.<br>"
        f"Expected move for {nearest_expiry}: ±{price(exp_move)}; Max Pain: {price(max_pain_level)}."
    )
    bull_path = " → ".join(f"{x:,.0f}" for x in [bull_trigger] + bull_targets)
    bear_path = " → ".join(f"{x:,.0f}" for x in [bear_trigger] + bear_targets)
    plan_text = (
        f"🐂 BULL: Reclaim & hold {bull_path}. Invalidation: rejection back below {bear_trigger:,.0f}.<br>"
        f"🐻 BEAR: Break & close below {bear_path}. Invalidation: reclaim above {bull_trigger:,.0f}.<br>"
        f"Use price action, VWAP, volume and momentum confirmation before entry."
    )

    st.markdown(f'<div class="analysis-card bull"><div class="analysis-title">↗ BULLISH STRUCTURE (Above {bull_trigger:,.0f})</div><div class="analysis-body">{bull_text}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="analysis-card bear"><div class="analysis-title">↙ BEARISH STRUCTURE (Below {bear_trigger:,.0f})</div><div class="analysis-body">{bear_text}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="analysis-card chop"><div class="analysis-title">〰 CHOP ZONE & HIDDEN LEVELS</div><div class="analysis-body">{chop_text}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="analysis-card plan"><div class="analysis-title">📋 TRADE PLAN</div><div class="analysis-body">{plan_text}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Positioning Map</div>', unsafe_allow_html=True)
map_html = f"""
<div class="map-grid">
  <div class="map-card map-bull"><h3>BULLISH</h3><b>LONG GAMMA</b><p>Above {bull_trigger:,.0f}<br><br>Price acceptance above this level may reduce realized volatility and favor movement toward {call_wall:,.0f} and higher targets.</p></div>
  <div class="map-card map-chop"><h3>CHOP</h3><b>LOW CONVICTION</b><p>{chop_low:,.0f} — {chop_high:,.0f}<br><br>Near spot. Expect pinning, failed breakouts, whipsaw and mean reversion until price leaves the range.</p></div>
  <div class="map-card map-bear"><h3>BEARISH</h3><b>SHORT GAMMA</b><p>Below {bear_trigger:,.0f}<br><br>Price acceptance below this level may increase hedging pressure and directional volatility toward {put_wall:,.0f} and lower targets.</p></div>
</div>
<div class="footer-note">Generated {datetime.now().strftime('%m/%d/%Y %I:%M %p')} · JR GEX / Flow Read · Estimated from public option-chain data</div>
"""
st.markdown(map_html, unsafe_allow_html=True)
