from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="JR Free GEX Dashboard", page_icon="📊", layout="wide")

CONTRACT_MULTIPLIER = 100


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def normal_pdf(x: np.ndarray | float) -> np.ndarray | float:
    return np.exp(-0.5 * np.asarray(x) ** 2) / math.sqrt(2.0 * math.pi)


def bs_gamma(spot: float, strike: np.ndarray, t: float, rate: float, iv: np.ndarray) -> np.ndarray:
    """Black-Scholes gamma for calls and puts (same gamma)."""
    strike = np.asarray(strike, dtype=float)
    iv = np.asarray(iv, dtype=float)
    valid = (strike > 0) & (iv > 0) & (t > 0) & (spot > 0)
    result = np.zeros_like(strike, dtype=float)
    if not np.any(valid):
        return result

    sqrt_t = math.sqrt(t)
    d1 = (
        np.log(spot / strike[valid])
        + (rate + 0.5 * iv[valid] ** 2) * t
    ) / (iv[valid] * sqrt_t)
    result[valid] = normal_pdf(d1) / (spot * iv[valid] * sqrt_t)
    return result


@st.cache_data(ttl=300, show_spinner=False)
def load_ticker_data(symbol: str) -> tuple[float, list[str]]:
    ticker = yf.Ticker(symbol)
    history = ticker.history(period="5d", interval="1m", auto_adjust=False, prepost=True)
    if history.empty:
        history = ticker.history(period="5d", interval="1d", auto_adjust=False)
    if history.empty:
        raise RuntimeError(f"No price data returned for {symbol}.")
    spot = _safe_float(history["Close"].dropna().iloc[-1])
    expirations = list(ticker.options)
    if not expirations:
        raise RuntimeError(f"No listed option expirations returned for {symbol}.")
    return spot, expirations


@st.cache_data(ttl=300, show_spinner=False)
def load_chain(symbol: str, expirations: tuple[str, ...]) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    frames: list[pd.DataFrame] = []
    now = datetime.now(timezone.utc)

    for expiration in expirations:
        chain = ticker.option_chain(expiration)
        expiry_dt = datetime.strptime(expiration, "%Y-%m-%d").replace(
            hour=21, minute=0, tzinfo=timezone.utc
        )
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
    required = ["strike", "openInterest", "impliedVolatility"]
    for column in required:
        if column not in result.columns:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result


def calculate_gex(chain: pd.DataFrame, spot: float, rate: float) -> pd.DataFrame:
    data = chain.copy()
    t = np.maximum(data["daysToExpiry"].to_numpy(float) / 365.0, 1.0 / (365.0 * 24.0))
    strike = data["strike"].to_numpy(float)
    iv = data["impliedVolatility"].to_numpy(float)

    gammas = np.zeros(len(data), dtype=float)
    for unique_t in np.unique(t):
        mask = t == unique_t
        gammas[mask] = bs_gamma(spot, strike[mask], float(unique_t), rate, iv[mask])

    data["gamma"] = gammas
    sign = np.where(data["optionType"].eq("call"), 1.0, -1.0)
    # Approximate dealer GEX convention: calls positive, puts negative.
    # Dollar gamma for a 1% move in the underlying.
    data["gex"] = (
        sign
        * data["gamma"]
        * data["openInterest"]
        * CONTRACT_MULTIPLIER
        * spot**2
        * 0.01
    )
    return data


def aggregate_by_strike(data: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        data.pivot_table(
            index="strike",
            columns="optionType",
            values="gex",
            aggfunc="sum",
            fill_value=0.0,
        )
        .rename(columns={"call": "callGex", "put": "putGex"})
        .reset_index()
    )
    if "callGex" not in pivot:
        pivot["callGex"] = 0.0
    if "putGex" not in pivot:
        pivot["putGex"] = 0.0
    pivot["netGex"] = pivot["callGex"] + pivot["putGex"]
    return pivot.sort_values("strike").reset_index(drop=True)


def estimate_zero_gamma(strikes: pd.DataFrame) -> float | None:
    x = strikes["strike"].to_numpy(float)
    y = strikes["netGex"].to_numpy(float)
    if len(x) < 2:
        return None
    for i in range(len(x) - 1):
        if y[i] == 0:
            return float(x[i])
        if y[i] * y[i + 1] < 0:
            return float(x[i] + (0 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return None


def max_pain(chain: pd.DataFrame) -> float | None:
    expiries = sorted(chain["expiration"].unique())
    if not expiries:
        return None
    nearest = chain[chain["expiration"].eq(expiries[0])]
    strikes = np.sort(nearest["strike"].unique())
    if len(strikes) == 0:
        return None

    calls = nearest[nearest["optionType"].eq("call")][["strike", "openInterest"]]
    puts = nearest[nearest["optionType"].eq("put")][["strike", "openInterest"]]
    payouts = []
    for settlement in strikes:
        call_loss = ((settlement - calls["strike"]).clip(lower=0) * calls["openInterest"]).sum()
        put_loss = ((puts["strike"] - settlement).clip(lower=0) * puts["openInterest"]).sum()
        payouts.append(call_loss + put_loss)
    return float(strikes[int(np.argmin(payouts))])


def expected_move(chain: pd.DataFrame, spot: float) -> tuple[float | None, str | None]:
    expiries = sorted(chain["expiration"].unique())
    if not expiries:
        return None, None
    expiry = expiries[0]
    near = chain[chain["expiration"].eq(expiry)].copy()
    if near.empty:
        return None, expiry
    near["distance"] = (near["strike"] - spot).abs()
    atm_strike = float(near.loc[near["distance"].idxmin(), "strike"])
    atm = near[np.isclose(near["strike"], atm_strike)]
    call_mid = atm[atm["optionType"].eq("call")]
    put_mid = atm[atm["optionType"].eq("put")]

    def quote_mid(row: pd.DataFrame) -> float:
        if row.empty:
            return 0.0
        bid = _safe_float(row.iloc[0].get("bid"))
        ask = _safe_float(row.iloc[0].get("ask"))
        last = _safe_float(row.iloc[0].get("lastPrice"))
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        return last

    move = quote_mid(call_mid) + quote_mid(put_mid)
    return (move if move > 0 else None), expiry


def money(value: float | None) -> str:
    return "N/A" if value is None else f"${value:,.2f}"


st.title("JR Free GEX Dashboard")
st.caption(
    "Approximate gamma-exposure levels from public option-chain data. "
    "Refreshes every five minutes and is not a proprietary real-time dealer feed."
)

with st.sidebar:
    symbol = st.text_input("Ticker", value="SPY").strip().upper()
    expiration_count = st.slider("Number of expirations", 1, 12, 4)
    strike_range_pct = st.slider("Strike range around spot", 5, 50, 20, step=5)
    rate_pct = st.number_input("Risk-free rate (%)", 0.0, 15.0, 4.5, step=0.1)
    refresh = st.button("Refresh data", use_container_width=True)
    st.markdown("---")
    st.write("**Suggested:** 1–4 expirations for intraday levels; 6–12 for broader positioning.")

if refresh:
    st.cache_data.clear()

if not symbol:
    st.stop()

try:
    with st.spinner(f"Loading {symbol} options..."):
        spot, available_expirations = load_ticker_data(symbol)
        selected_expirations = tuple(available_expirations[:expiration_count])
        chain = load_chain(symbol, selected_expirations)
        gex_data = calculate_gex(chain, spot, rate_pct / 100.0)
        strikes = aggregate_by_strike(gex_data)
except Exception as exc:
    st.error(f"Unable to load data: {exc}")
    st.info("Try another ticker, confirm your internet connection, or refresh in a few minutes.")
    st.stop()

lower = spot * (1.0 - strike_range_pct / 100.0)
upper = spot * (1.0 + strike_range_pct / 100.0)
visible = strikes[strikes["strike"].between(lower, upper)].copy()
if visible.empty:
    visible = strikes.copy()

call_wall_row = visible.loc[visible["callGex"].idxmax()] if not visible.empty else None
put_wall_row = visible.loc[visible["putGex"].idxmin()] if not visible.empty else None
call_wall = None if call_wall_row is None else float(call_wall_row["strike"])
put_wall = None if put_wall_row is None else float(put_wall_row["strike"])
zero_gamma = estimate_zero_gamma(visible)
max_pain_level = max_pain(chain)
exp_move, exp_move_expiry = expected_move(chain, spot)
net_gex = float(visible["netGex"].sum())

cols = st.columns(6)
cols[0].metric("Spot", money(spot))
cols[1].metric("Call Wall", money(call_wall))
cols[2].metric("Put Wall", money(put_wall))
cols[3].metric("Zero Gamma", money(zero_gamma))
cols[4].metric("Max Pain", money(max_pain_level))
cols[5].metric("Net GEX", f"${net_gex / 1_000_000:,.1f}M")

if exp_move is not None:
    st.info(
        f"Nearest-expiration expected move ({exp_move_expiry}): ±{money(exp_move)} — "
        f"approximately {money(spot - exp_move)} to {money(spot + exp_move)}."
    )

fig = go.Figure()
fig.add_bar(x=visible["strike"], y=visible["callGex"] / 1_000_000, name="Call GEX")
fig.add_bar(x=visible["strike"], y=visible["putGex"] / 1_000_000, name="Put GEX")
fig.add_scatter(
    x=visible["strike"],
    y=visible["netGex"] / 1_000_000,
    mode="lines+markers",
    name="Net GEX",
    yaxis="y2",
)
fig.add_vline(x=spot, line_dash="dash", annotation_text=f"Spot {spot:.2f}")
if call_wall is not None:
    fig.add_vline(x=call_wall, line_dash="dot", annotation_text="Call Wall")
if put_wall is not None:
    fig.add_vline(x=put_wall, line_dash="dot", annotation_text="Put Wall")
if zero_gamma is not None:
    fig.add_vline(x=zero_gamma, line_dash="dashdot", annotation_text="Zero Gamma")
fig.update_layout(
    title=f"{symbol} GEX by Strike — expirations through {selected_expirations[-1]}",
    xaxis_title="Strike",
    yaxis_title="Call/Put GEX ($ millions per 1% move)",
    yaxis2=dict(
        title="Net GEX ($ millions)",
        overlaying="y",
        side="right",
        showgrid=False,
    ),
    barmode="relative",
    hovermode="x unified",
    height=650,
    legend=dict(orientation="h"),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Key levels")
levels = pd.DataFrame(
    [
        ["Spot", spot, "Current underlying price"],
        ["Call Wall", call_wall, "Strike with the largest positive call gamma"],
        ["Put Wall", put_wall, "Strike with the largest negative put gamma"],
        ["Zero Gamma", zero_gamma, "Approximate strike where net GEX changes sign"],
        ["Max Pain", max_pain_level, "Nearest-expiration minimum aggregate option payout"],
    ],
    columns=["Level", "Price", "Meaning"],
)
st.dataframe(levels, use_container_width=True, hide_index=True)

with st.expander("Raw strike table"):
    table = visible.copy()
    for c in ["callGex", "putGex", "netGex"]:
        table[c] = table[c] / 1_000_000
    table = table.rename(
        columns={
            "callGex": "Call GEX ($M)",
            "putGex": "Put GEX ($M)",
            "netGex": "Net GEX ($M)",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

st.warning(
    "GEX is a model, not an exact view of dealer inventory. Open interest can be stale, "
    "intraday trades may not be reflected, and the calls-positive/puts-negative convention "
    "is an approximation. Use these levels as context—not automatic entries."
)
