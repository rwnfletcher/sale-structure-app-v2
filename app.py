# app.py — Streamlit "Vendor Finance vs Lump Sum" Dashboard
# v3.9 — Stable
# • Offer Price card
# • 0–16wk Lump bar
# • Vendor-only Monthly Cost (bold Alleviator beside it)
# • XLSX export
# • Yearly/Monthly amortisation toggle
# • Tax Burden Alleviator: can input as $ or % of Offer Price (auto-linked)
# • Comma-formatted inputs in box (Headline & Alleviator)

import io, math, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import dataclass
from typing import List, Literal

Structure = Literal["Amortizing", "Interest-Only + Balloon", "Equal Principal"]

# ---------- Helpers ----------
def aud(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"A${x:,.0f}"

def parse_money_to_float(s: str) -> float:
    """Parse commas or $ into float."""
    if not s: return 0.0
    cleaned = re.sub(r"[^\d\.\-]", "", str(s))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def money_input(label: str, default: float, key: str) -> float:
    """Single text input prefilled with commas; parses safely to float."""
    default_str = f"{default:,.0f}"
    raw = st.text_input(label, value=default_str, key=key)
    return parse_money_to_float(raw)

@dataclass
class CashFlow:
    year: int
    payment: float
    interest: float
    principal: float
    ending_balance: float

# ---------- Schedule Builders ----------
def amortizing_schedule(principal: float, rate: float, years: int) -> List[CashFlow]:
    n, r = int(years), rate
    if n <= 0: return []
    pmt = principal * r / (1 - (1 + r) ** (-n)) if r else principal / n
    bal, rows = principal, []
    for t in range(1, n + 1):
        interest = bal * r
        principal_part = pmt - interest
        bal = max(0.0, bal - principal_part)
        rows.append(CashFlow(t, pmt, interest, principal_part, bal))
    return rows

def interest_only_balloon_schedule(principal: float, rate: float, years: int) -> List[CashFlow]:
    n, r, bal, rows = int(years), rate, principal, []
    for t in range(1, n):
        interest = bal * r
        rows.append(CashFlow(t, interest, interest, 0.0, bal))
    interest, payment = bal * r, bal * r + bal
    rows.append(CashFlow(n, payment, interest, bal, 0.0))
    return rows

def equal_principal_schedule(principal: float, rate: float, years: int) -> List[CashFlow]:
    n, r, bal, rows, principal_part = int(years), rate, principal, [], principal / int(years)
    for t in range(1, n + 1):
        interest = bal * r
        payment  = principal_part + interest
        bal = max(0.0, bal - principal_part)
        rows.append(CashFlow(t, payment, interest, principal_part, bal))
    return rows

def build_schedule(structure: Structure, principal: float, rate: float, years: int):
    if structure == "Amortizing":
        return amortizing_schedule(principal, rate, years)
    elif structure == "Interest-Only + Balloon":
        return interest_only_balloon_schedule(principal, rate, years)
    elif structure == "Equal Principal":
        return equal_principal_schedule(principal, rate, years)
    else:
        raise ValueError("Unknown structure")

# ---------- Graphics ----------
def draw_timebar(ax, seller_weeks: float, lump_max: float):
    ax.set_title("Handover Timing (weeks)")
    labels, y = ["Seller Finance", "Lump Sum"], [1, 0]
    ax.barh([y[0]], [seller_weeks], left=[0], color="#1f77b4", height=0.35)
    ax.barh([y[1]], [lump_max], left=[0], color="#ff7f0e", height=0.35)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Weeks")
    ax.grid(True, axis="x", alpha=0.25)
    ax.text(seller_weeks + 0.3, y[0], f"{seller_weeks:.0f} wk", va="center")
    ax.text(lump_max + 0.3, y[1], f"0–{lump_max:.0f} wks", va="center")

def vendor_monthly_payment(principal, rate, years, structure):
    r_m, n_m = rate/12, years*12
    if n_m <= 0: return 0
    if structure == "Amortizing":
        return principal * r_m / (1 - (1 + r_m) ** (-n_m)) if r_m else principal / n_m
    if structure == "Interest-Only + Balloon":
        return principal * r_m
    principal_month = principal / n_m
    return principal_month + principal * r_m

# ---------- UI ----------
st.set_page_config(page_title="Sale Structure Comparator — v3.9", layout="wide")

with st.sidebar:
    st.header("Inputs")
    headline  = money_input("Headline value (A$)", 5_000_000.0, "headline")
    lump_disc = st.number_input("Lump Sum discount %", 0.0, 95.0, 25.0) / 100
    vf_prem   = st.number_input("Vendor Finance premium %", 0.0, 200.0, 15.0) / 100
    rate      = st.number_input("Interest rate %", 0.0, 100.0, 5.0, 0.5) / 100
    years     = int(st.number_input("Term (years)", 1, 50, 10))
    structure = st.selectbox("Structure", ["Amortizing","Interest-Only + Balloon","Equal Principal"])
    equity_roll_pct = st.number_input("Equity roll (% of gross kept as equity)", 0.0, 90.0, 0.0, 1.0)
    seller_weeks = st.number_input("Seller finance (weeks)", 0.1, 12.0, 1.0, 0.1)
    lump_max     = st.number_input("Lump sum duration (weeks)", 1.0, 52.0, 16.0, 1.0)
    show_monthly = st.checkbox("Show monthly cost panel", True)
    st.markdown("---")

    # ---- Tax Burden Alleviator inputs ----
    st.subheader("Tax Burden Alleviator (First Year extra principal)")
    tax_as_percent = st.checkbox("Input as % of Offer Price", value=False)
    if tax_as_percent:
        tax_alleviator_percent = st.number_input("Alleviator (% of Offer Price)", 0.0, 100.0, 0.0, step=0.5)
        tax_alleviator_amount = 0.0  # placeholder; will compute after offer price is known
    else:
        tax_alleviator_amount = money_input("Alleviator Amount (A$)", 0.0, "allev_amt")
        tax_alleviator_percent = 0.0  # computed after offer price

    tax_alleviator_month  = int(st.number_input("Month number for Alleviator (1–term months)",
                                                min_value=1, max_value=years*12,
                                                value=min(6, years*12), step=1))

# ---------- Calculations ----------
lump_gross   = headline * (1 - lump_disc)
vf_principal = headline * (1 + vf_prem)
offer_price  = vf_principal

# Link % ↔ A$ for the alleviator
if tax_as_percent:
    tax_alleviator_amount = offer_price * tax_alleviator_percent / 100
else:
    tax_alleviator_percent = (tax_alleviator_amount / offer_price * 100) if offer_price > 0 else 0

# Effective principal (offer − alleviator)
effective_principal = max(0.0, vf_principal - tax_alleviator_amount)
yearly_sched    = build_schedule(structure, effective_principal, rate, years)
yearly_interest = sum(r.interest for r in yearly_sched)
vf_gross_year   = effective_principal + yearly_interest
roll_mult       = 1 - equity_roll_pct/100
lump_cash       = lump_gross * roll_mult
vf_cash_year    = vf_gross_year * roll_mult

# ---------- Offer Price ----------
st.markdown(f"## Offer Price (Vendor Finance): **{aud(offer_price)}**")

# ---------- Summary Cards ----------
col1,col2,col3 = st.columns(3)
col1.metric("Lump Sum (Cash)", aud(lump_cash))
col2.metric("Vendor Finance (Cash)", aud(vf_cash_year))
col3.metric("Delta (Cash)", f"+{aud(vf_cash_year - lump_cash)}")

# ---------- Charts ----------
left,right = st.columns(2)
with left:
    st.markdown("#### Cash Proceeds — Breakdown (indicative)")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar([0],[lump_cash],color="#ff7f0e", label="Lump Cash")
    ax.bar([1],[effective_principal*roll_mult],color="#1f77b4", label="VF Principal (cash)")
    ax.bar([1],[yearly_interest*roll_mult],bottom=[effective_principal*roll_mult],color="#aec7e8", label="VF Interest (cash)")
    ax.set_xticks([0,1],["Lump","Vendor Finance"]); ax.set_ylabel("A$"); ax.grid(True,alpha=.2)
    ax.legend(); st.pyplot(fig)
with right:
    st.markdown("#### Handover Timing (weeks)")
    fig2, ax2 = plt.subplots(figsize=(6,4))
    draw_timebar(ax2, seller_weeks, lump_max)
    st.pyplot(fig2)

# ---------- Monthly Cost ----------
if show_monthly:
    vendor_pmt = vendor_monthly_payment(effective_principal, rate, years, structure)
    line = f"#### Estimated Monthly Payment: **{aud(vendor_pmt)}**"
    if tax_alleviator_amount > 0:
        line += f" | **Tax Burden Alleviator: {aud(tax_alleviator_amount)} ({tax_alleviator_percent:.1f}% of Offer, Month {tax_alleviator_month})**"
    st.markdown(line)
    st.caption(f"{structure} • {years} yrs @ {rate*100:.1f}%")

# ---------- Amortisation ----------
st.markdown("---")
st.markdown("## Vendor Finance Amortisation Schedule")
months = years * 12
r_m, bal, rows = rate/12, effective_principal, []
amort_pmt = (effective_principal * r_m / (1 - (1 + r_m) ** (-months))) if (structure=="Amortizing" and r_m) else (
            effective_principal / months if structure=="Amortizing" else None)

for m in range(1, months+1):
    allev = 0.0
    if structure == "Amortizing":
        pmt = amort_pmt or 0.0
        interest = bal * r_m
        principal = pmt - interest
        if m == tax_alleviator_month and tax_alleviator_amount > 0:
            allev = tax_alleviator_amount
        bal = max(0.0, bal - principal)
    elif structure == "Interest-Only + Balloon":
        interest = bal * r_m
        principal = 0.0
        pmt = interest
        if m == tax_alleviator_month and tax_alleviator_amount > 0:
            allev = tax_alleviator_amount
        if m == months and bal > 0:
            principal += bal; pmt += bal; bal = 0.0
    else:
        principal = effective_principal / months
        interest  = bal * r_m
        pmt = principal + interest
        if m == tax_alleviator_month and tax_alleviator_amount > 0:
            allev = tax_alleviator_amount
        bal = max(0.0, bal - principal)
    total_cash = pmt + allev
    rows.append([m, total_cash, interest, principal, bal, allev])

df = pd.DataFrame(rows, columns=["Month","Payment","Interest","Principal","Balance","Alleviator"])
df["Year"] = np.ceil(df["Month"]/12).astype(int)

# ---------- Chart Toggle ----------
chart_view = st.radio("Amortisation chart view:", ["Yearly", "Monthly"], horizontal=True)
if chart_view == "Yearly":
    dfy = df.groupby("Year")[["Payment","Interest","Principal","Alleviator"]].sum().reset_index()
    st.markdown("### Yearly Amortisation Chart")
    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.bar(dfy["Year"], dfy["Payment"], label="Total Cash Outflow")
    ax3.bar(dfy["Year"], dfy["Interest"], label="Interest Portion")
    ax3.set_xlabel("Year"); ax3.set_ylabel("A$"); ax3.legend()
    st.pyplot(fig3)
else:
    st.markdown("### Monthly Amortisation Chart")
    fig4, ax4 = plt.subplots(figsize=(10,4))
    ax4.plot(df["Month"], df["Payment"], linewidth=2, label="Total Cash")
    ax4.plot(df["Month"], df["Interest"], linewidth=1, label="Interest")
    ax4.plot(df["Month"], df["Principal"], linewidth=1, label="Principal")
    if tax_alleviator_amount > 0:
        mask = df["Month"] == tax_alleviator_month
        if mask.any():
            ax4.scatter([tax_alleviator_month],
                        [df.loc[mask,"Payment"].values[0]], s=40, color="red", label="Alleviator Month")
    ax4.set_xlabel("Month"); ax4.set_ylabel("A$"); ax4.grid(True, alpha=0.25); ax4.legend()
    st.pyplot(fig4)

# ---------- XLSX ----------
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
    df.to_excel(w, index=False, sheet_name="Amortisation")
buf.seek(0)
st.download_button("📥 Download Full Amortisation (XLSX)", data=buf,
                   file_name="vendor_finance_amortisation.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption(
    "Headline and Alleviator accept commas (e.g., 4,000,000). "
    "Alleviator can be input as a percentage of Offer Price or fixed amount; both auto-link. "
    "Amortizing keeps identical monthly payments; IO+Balloon and Equal-Principal are built on the reduced principal."
)
