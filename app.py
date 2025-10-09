# ---------- How to Use ----------
st.markdown("---")
st.markdown("## 🧭 How to Use This Dashboard")
st.markdown("""
1. **Enter the deal details** in the sidebar (left panel):  
   - Headline Value, Lump Sum Discount, Vendor Finance Premium, etc.  
2. **(Optional)** — specify a **Tax Burden Alleviator** either as:  
   - a fixed dollar amount (e.g., `4,000,000`), or  
   - a percentage of the Offer Price (toggle “Input as % of Offer Price”).  
3. Adjust the **Month number** if the alleviator is paid later (e.g., Month 6).  
4. Click anywhere outside an input or press **Enter** to apply comma formatting.  
5. Review results in the main panel:
   - **Offer Price** — shows Headline + Vendor Finance Premium.  
   - **Monthly Cost** — shows Vendor Finance monthly payment and Alleviator summary.  
   - **Charts** — view side-by-side Cash Proceeds, Handover Timing, and Amortisation (Yearly/Monthly toggle).  
6. Download the full amortisation schedule via the **📥 Download XLSX** button.  

**Example:**  
After entering your data (see below), the screen should look similar to:  
![Example Screenshot](https://docs.streamlit.io/logo.svg)
""")

# ---------- Input Reference ----------
st.markdown("## 📄 Input Reference")
st.markdown("""
**Headline value (A$)** — Base price for the transaction before discounts or premiums.  

**Lump Sum discount (%)** — Percentage reduction for an upfront cash purchase.  

**Vendor Finance premium (%)** — Additional premium offered to the seller when paid over time.  

**Interest rate (%)** — Annual interest rate applied to the financed portion.  

**Term (years)** — Loan duration in years.  

**Structure** — Type of repayment model:  
- *Amortizing* — fixed total monthly payments.  
- *Interest-Only + Balloon* — interest-only until the end, then full principal.  
- *Equal Principal* — equal principal payments; total payments decline over time.  

**Equity roll (% of gross)** — Portion of the business equity retained by the seller.  

**Seller finance (weeks)** — Estimated handover time for Vendor Finance deals.  

**Lump sum duration (weeks)** — Estimated handover time for Lump Sum deals.  

**Tax Burden Alleviator (A$ or %)** — Extra principal paid in the first year to offset taxable income (treated as a Day-1 principal reduction, paid in the nominated month).  

**Month number for Alleviator** — Month in which the alleviator payment occurs (default = 6).  

**Show monthly cost panel** — Toggles visibility of the monthly repayment summary.
""")
