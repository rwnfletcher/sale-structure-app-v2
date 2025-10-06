# app.py — Streamlit "Vendor Finance vs Lump Sum" Dashboard (v2: Monthly Cost Panel)
# -------------------------------------------------------------------------------------
# README — How to run
# 1) pip install streamlit pandas numpy matplotlib xlsxwriter
# 2) streamlit run app.py
#
# What’s new in v2:
# - Sidebar toggle: "Show monthly cost panel".
# - Monthly costs panel appears *under the charts*, side-by-side Vendor Finance vs Lump Sum.
# - Monthly costs are structure-aware:
#   * Amortizing: fixed monthly payment (rate/12, months = years*12).
#   * Interest-Only + Balloon: monthly interest only (principal * rate/12).
#   * Equal Principal: shows Month-1 payment (declines over time).
# - Equity roll (% of gross) reduces the amount financed for both sides before monthly calcs.
# - For Lump Sum monthly cost, assumes bank *amortizing* at the same rate & term (not IO/Equal Principal).
#   (You can change this assumption later if you want separate bank inputs.)
# -------------------------------------------------------------------------------------

import io
import json
import math
from dataclasses import dataclass
from typing import List, Literal, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

Structure = Literal["Amortizing", "Interest-Only + Balloon", "Equal Principal"]

def aud(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"A${x:,.0f}"

@dataclass
class CashFlow:
    year: int
    payment: float
    interest: float
    principal: float
    ending_balance: float

def amortizing_schedule(principal: float, rate: float, years: int) -> List[CashFlow]:
    n = int(years); r = rate
    if n <= 0: return []
    if r == 0: pmt = principal / n
    else: pmt = principal * r / (1 - (1 + r) ** (-n))
    bal = principal; rows = []
    for t in range(1, n + 1):
        interest = bal * r
        principal_part = pmt - interest
        bal = max(0.0, bal - principal_part)
        rows.append(CashFlow(t, pmt, interest, principal_part, bal))
    return rows

def interest_only_balloon_schedule(principal: float, rate: float, years: int) -> List[CashFlow]:
    n = int(years); r = rate
    if n <= 0: return []
    bal = principal; rows = []
    for t in range(1, n):
        interest = bal * r
        rows.append(CashFlow(t, interest, interest, 0.0, bal))
    interest = bal * r
    payment = interest + bal
    rows.append(CashFlow(n, payment, interest, bal, 0.0))
    return rows

def equal_principal_schedule(principal: float, rate: float, years: int) -> List[CashFlow]:
    n = int(years); r = rate
    if n <= 0: return []
    bal = principal; rows = []; principal_part = principal / n
    for t in range(1, n + 1):
        interest = bal * r
        payment = principal_part + interest
        bal = max(0.0, bal - principal_part)
        rows.append(CashFlow(t, payment, interest, principal_part, bal))
    return rows

def build_schedule(structure: Structure, principal: float, rate: float, years: int) -> List[CashFlow]:
    if structure == "Amortizing": return amortizing_schedule(principal, rate, years)
    if structure == "Interest-Only + Balloon": return interest_only_balloon_schedule(principal, rate, years)
    if structure == "Equal Principal": return equal_principal_schedule(principal, rate, years)
    raise ValueError("Unknown structure")

def draw_timebar(ax, seller_weeks: float, lump_min: float, lump_max: float):
    ax.set_title("Handover Timing (weeks)")
    labels = ["Seller Finance", "Lump Sum"]; y_pos = [1, 0]
    ax.barh([y_pos[0]], [seller_weeks], left=[0], height=0.35)
    ax.barh([y_pos[1]], [lump_max - lump_min], left=[lump_min], height=0.35)
    ax.set_yticks(y_pos, labels); ax.set_xlabel("Weeks"); ax.grid(True, axis="x", alpha=0.25)
    ax.text(seller_weeks + 0.2, y_pos[0], f"{seller_weeks:.0f} wk", va="center")
    ax.text(lump_max + 0.2, y_pos[1], f"{lump_min:.0f}–{lump_max:.0f} wks", va="center")

def monthly_amortizing_payment(principal: float, annual_rate: float, years: int) -> float:
    r_m = annual_rate / 12.0
    n_m = years * 12
    if n_m <= 0: return 0.0
    if r_m == 0: return principal / n_m
    return principal * r_m / (1 - (1 + r_m) ** (-n_m))

def vendor_monthly_cost(structure: str, financed_amt: float, annual_rate: float, years: int) -> dict:
    n_m = years * 12; r_m = annual_rate / 12.0
    if structure == "Amortizing":
        pmt = monthly_amortizing_payment(financed_amt, annual_rate, years)
        return {"label": "Amortizing", "monthly": pmt, "note": "Fixed monthly payment."}
    elif structure == "Interest-Only + Balloon":
        monthly_interest = financed_amt * r_m
        return {"label": "Interest-only", "monthly": monthly_interest, "note": "Interest-only monthly; balloon at end."}
    else:
        principal_month = financed_amt / n_m if n_m > 0 else 0.0
        month1_payment = principal_month + financed_amt * r_m
        return {"label": "Equal Principal", "monthly": month1_payment, "note": "Month-1 payment; declines over time."}

def lump_monthly_cost_amortizing(principal: float, annual_rate: float, years: int) -> float:
    return monthly_amortizing_payment(principal, annual_rate, years)

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Sale Structure Comparator — v2", layout="wide")

with st.sidebar:
    st.header("Inputs")
    headline = st.number_input("Headline value (A$)", min_value=0.0, value=5_000_000.0, step=100_000.0, format="%.0f")
    lump_disc = st.number_input("Lump Sum discount %", min_value=0.0, max_value=95.0, value=25.0, step=1.0) / 100.0
    vf_prem = st.number_input("Vendor Finance premium %", min_value=0.0, max_value=200.0, value=15.0, step=1.0) / 100.0
    rate = st.number_input("Interest rate %", min_value=0.0, max_value=100.0, value=5.0, step=0.5) / 100.0
    years = int(st.number_input("Term (years)", min_value=1, max_value=50, value=10, step=1))
    structure = st.selectbox("Structure", ["Amortizing", "Interest-Only + Balloon", "Equal Principal"])
    equity_roll_pct = st.number_input("Equity roll (% of gross kept as equity)", min_value=0.0, max_value=90.0, value=0.0, step=1.0)
    seller_weeks = st.number_input("Seller finance (weeks)", min_value=0.1, max_value=12.0, value=1.0, step=0.1)
    lump_min = st.number_input("Lump sum — min weeks", min_value=1.0, max_value=52.0, value=12.0, step=1.0)
    lump_max = st.number_input("Lump sum — max weeks", min_value=1.0, max_value=52.0, value=16.0, step=1.0)
    show_monthly = st.checkbox("Show monthly cost panel", value=True)

lump_gross = headline * (1 - lump_disc)
vf_principal = headline * (1 + vf_prem)
schedule = build_schedule(structure, vf_principal, rate, years)
total_interest = sum(r.interest for r in schedule)
vf_gross = vf_principal + total_interest

roll_mult = (1 - equity_roll_pct/100.0)
lump_cash = lump_gross * roll_mult
vf_cash = vf_gross * roll_mult

col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.markdown("### Lump Sum (Cash)")
    st.markdown(f"## {aud(lump_cash)}")
with col2:
    st.markdown("### Vendor Finance (Cash)")
    st.markdown(f"## {aud(vf_cash)}")
delta_amt = vf_cash - lump_cash
with col3:
    st.markdown("### Delta (Cash)")
    st.markdown(f"## +{aud(delta_amt)}")

left, right = st.columns([1,1])
with left:
    st.markdown("#### Cash Proceeds — Breakdown")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar([0], [lump_cash], label="Lump Cash")
    ax.bar([1], [vf_principal * roll_mult], label="VF Principal (cash)")
    ax.bar([1], [total_interest * roll_mult], bottom=[vf_principal * roll_mult], label="VF Interest (cash)")
    ax.set_xticks([0,1], ["Lump", "Vendor Finance"])
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
with right:
    st.markdown("#### Handover Timing (weeks)")
    fig2, ax2 = plt.subplots(figsize=(6,4))
    draw_timebar(ax2, seller_weeks, lump_min, lump_max)
    st.pyplot(fig2)

if show_monthly:
    st.markdown("### Monthly Cost (Buyer’s View)")
    vendor_financed = vf_principal * roll_mult
    lump_financed = lump_gross * roll_mult
    vendor_cost = vendor_monthly_cost(structure, vendor_financed, rate, years)
    lump_cost = lump_monthly_cost_amortizing(lump_financed, rate, years)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Vendor Finance — Monthly")
        st.markdown(f"### {aud(vendor_cost['monthly'])}")
        st.caption(vendor_cost['note'])
    with c2:
        st.subheader("Lump Sum (Bank) — Monthly")
        st.markdown(f"### {aud(lump_cost)}")
        st.caption("Amortizing monthly payment (bank).")
