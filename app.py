# app.py — Streamlit "Vendor Finance vs Lump Sum" Dashboard
# v4.0 — Full, stable build
# • Money inputs (Headline & Alleviator) accept commas; auto-format on commit
# • Alleviator can be entered as % of Offer or A$ (linked both ways)
# • Offer Price card, Handover Timing bar (0–N weeks), Monthly Cost with bold Alleviator
# • Amortisation charts (Yearly/Monthly) + XLSX export
# • PDF download of dashboard summary (graceful if reportlab missing)
# • "How to Use" + "Input Reference" sections at the bottom

import io, math, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import dataclass
from typing import List, Literal

# ---- optional PDF dependency
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

Structure = Literal["Amortizing", "Interest-Only + Balloon", "Equal Principal"]

# ---------- Helpers ----------
def aud(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"A${x:,.0f}"

def parse_money_to_float(s: str) -> float:
    """Parse input like '4,000,000' or '$4,000,000' into float safely."""
    if s is None or str(s).strip() == "":
        return 0.0
    cleaned = re.sub(r"[^\d\.\-]", "", str(s))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def money_input(label: str, default: float, key: str) -> float:
    """
    Text input that:
      - seeds a comma-formatted default,
      - re-formats back to comma style on commit (Enter or blur),
      - returns a float for calculations.
    Uses an internal state key "<key>__fmt" so it never collides with other widgets.
    """
    fmt_key = f"{key}__fmt"
    if fmt_key not in st.session_state:
        st.session_state[fmt_key] = f"{default:,.0f}"

    def _format_money():
        val = parse_money_to_float(st.session_state[fmt_key])
        st.session_state[fmt_key] = f"{val:,.0f}" if val != 0 else "0"

    st.text_input(label, key=fmt_key, on_change=_format_money)
    return parse_money_to_float(st.session_state[fmt_key])

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
    if n <= 0:
        return []
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
    ax.barh([y[1]], [lump_max],    left=[0], color="#ff7f0e", height=0.35)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Weeks")
    ax.grid(True, axis="x", alpha=0.25)
    ax.text(seller_weeks + 0.3, y[0], f"{seller_weeks:.0f} wk", va="center")
    ax.text(lump_max + 0.3, y[1], f"0–{lump_max:.0f} wks", va="center")

def vendor_monthly_payment(principal, rate, years, structure):
    r_m, n_m = rate/12, years*12
    if n_m <= 0:
        return 0
    if structure == "Amortizing":
        return principal * r_m / (1 - (1 + r_m) ** (-n_m)) if r_m else principal / n_m
    if structure == "Interest-Only + Balloon":
        return principal * r_m
    principal_month = principal / n_m
    return principal_month + principal * r_m

# ---------- UI ----------
st.set_page_config(page_title="Sale Structure Comparator — v4.0", layout="wide")

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

    # ---- Tax Burden Alleviator inputs (linked $ and %) ----
    st.subheader("Tax Burden Alleviator (First Year extra principal)")
    tax_as_percent = st.checkbox("Input as % of Offer Price", value=False)
    if tax_as_percent:
        tax_alleviator_percent = st.number_input("Alleviator (% of Offer Price)", 0.0, 100.0, 0.0, step=0.5)
        tax_alleviator_amount = 0.0  # computed after offer price is known
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

# Link alleviator % <-> $
if tax_as_percent:
    tax_alleviator_amount = offer_price * tax_alleviator_percent / 100
else:
    tax_alleviator_percent = (tax_alleviator_amount / offer_price * 100) if offer_price > 0 else 0.0

# Effective principal (priced as Day-1 reduction)
effective_principal = max(0.0, vf_principal - tax_alleviator_amount)

# Build indicative yearly schedule on effective principal
yearly_sched    = build_schedule(structure, effective_principal, rate, years)
yearly_interest = sum(r.interest for r in yearly_sched)
vf_gross_year   = effective_principal + yearly_interest

# Equity roll modifies *cash optics* only
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

# ---------- PRINT / DOWNLOAD DASHBOARD AS PDF ----------
st.markdown("---")
st.markdown("## 🖨️ Print / Download Dashboard as PDF")

def generate_dashboard_pdf() -> io.BytesIO:
    """Build a simple A4 PDF summary of the current dashboard state."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(25 * mm, height - 25 * mm, "Vendor Finance vs Lump Sum — Dashboard Summary")

    y = height - 40 * mm
    c.setFont("Helvetica", 11)
    # Offer & inputs
    c.drawString(25 * mm, y, f"Offer Price (Vendor Finance): {aud(offer_price)}"); y -= 7 * mm
    c.drawString(25 * mm, y, f"Headline Value: {aud(headline)}"); y -= 7 * mm
    c.drawString(25 * mm, y, f"Lump Sum Discount: {lump_disc * 100:.1f}%"); y -= 7 * mm
    c.drawString(25 * mm, y, f"Vendor Finance Premium: {vf_prem * 100:.1f}%"); y -= 7 * mm
    c.drawString(25 * mm, y, f"Interest Rate: {rate * 100:.2f}%  |  Term: {years} yrs")

    # Alleviator
    y -= 10 * mm
    c.setFont("Helvetica-Bold", 12); c.drawString(25 * mm, y, "Tax Burden Alleviator")
    c.setFont("Helvetica", 11); y -= 6 * mm
    c.drawString(25 * mm, y, f"Amount: {aud(tax_alleviator_amount)}  ({tax_alleviator_percent:.1f}% of Offer)"); y -= 7 * mm
    c.drawString(25 * mm, y, f"Month Due: {tax_alleviator_month}")

    # Metrics
    y -= 10 * mm
    c.setFont("Helvetica-Bold", 12); c.drawString(25 * mm, y, "Summary Metrics")
    c.setFont("Helvetica", 11); y -= 6 * mm
    c.drawString(25 * mm, y, f"Lump Sum (Cash): {aud(lump_cash)}"); y -= 6 * mm
    c.drawString(25 * mm, y, f"Vendor Finance (Cash): {aud(vf_cash_year)}"); y -= 6 * mm
    c.drawString(25 * mm, y, f"Delta (Cash): +{aud(vf_cash_year - lump_cash)}")

    # Structure details
    y -= 10 * mm
    c.setFont("Helvetica-Bold", 12); c.drawString(25 * mm, y, "Structure & Configuration")
    c.setFont("Helvetica", 11); y -= 6 * mm
    c.drawString(25 * mm, y, f"Structure: {structure}"); y -= 6 * mm
    c.drawString(25 * mm, y, f"Equity Roll: {equity_roll_pct * 100:.1f}%"); y -= 6 * mm
    c.drawString(25 * mm, y, f"Handover (Seller Finance): {seller_weeks} wks"); y -= 6 * mm
    c.drawString(25 * mm, y, f"Handover (Lump Sum): 0–{lump_max} wks")

    # Footer
    y -= 15 * mm
    c.setFont("Helvetica-Oblique", 9); c.setFillColor(colors.grey)
    c.drawString(25 * mm, y, "Generated from Streamlit Dashboard (Vendor Finance vs Lump Sum Comparison)")
    c.save(); buf.seek(0)
    return buf

if REPORTLAB_OK:
    pdf_buf = generate_dashboard_pdf()
    st.download_button(
        label="📄 Download Dashboard Summary (PDF)",
        data=pdf_buf,
        file_name="dashboard_summary.pdf",
        mime="application/pdf"
    )
else:
    st.info("To enable PDF download, add `reportlab` to requirements.txt and redeploy.")

# ---------- How to Use ----------
st.markdown("---")
st.markdown("## 🧭 How to Use This Dashboard")
st.markdown("""
1. Enter **Headline**, discounts/premiums, rate, term, and structure in the left sidebar.  
2. For **Tax Burden Alleviator**, either type a dollar amount (e.g., `4,000,000`) **or** toggle **% of Offer**.  
3. Set the **Month number** when the alleviator is paid (e.g., Month 6).  
4. Click outside an input or press **Enter** to snap values into **comma format**.  
5. Review:
   - **Offer Price** at the top,  
   - **Monthly Cost** (with bold alleviator if applicable),  
   - **Cash Proceeds** bar and **Handover Timing**,  
   - **Amortisation** (toggle Yearly/Monthly).  
6. Export the full **amortisation schedule (XLSX)** and **Dashboard Summary (PDF)** from the buttons below the charts.  
""")

# ---------- Input Reference ----------
st.markdown("## 📄 Input Reference")
st.markdown("""
**Headline value (A$)** — Purchase price before discounts/premiums.  
**Lump Sum discount (%)** — Price reduction for an upfront cash sale.  
**Vendor Finance premium (%)** — Price uplift offered when paid over time.  
**Interest rate (%)** — Annual interest rate applied to financed balance.  
**Term (years)** — Loan duration.  
**Structure** —  
• *Amortizing* — fixed total monthly payments.  
• *Interest-Only + Balloon* — interest-only until final month; principal due at end.  
• *Equal Principal* — same principal each month; total payment declines.  
**Equity roll (% of gross)** — Seller retains this share of equity (affects optics, not math in this view).  
**Seller finance (weeks) / Lump sum (weeks)** — Indicative handover timelines (visual only).  
**Tax Burden Alleviator (A$ or %)** — Extra principal paid in first year for tax planning.  
• Treated as **Day-1 principal reduction** for pricing/monthlies.  
• **Shown as a cash event** in the nominated month.  
**Month number for Alleviator** — Month in which the alleviator is paid (default 6).  
**Show monthly cost panel** — Toggle to display monthly repayment summary.  
""")
