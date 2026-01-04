import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# =========================================================
# App config
# =========================================================
st.set_page_config(
    page_title="Heat Pump vs Gas Boiler (V1)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Heat Pump vs Gas Boiler (V1)")
st.caption("Compares annual heating cost using a simple, defensible energy balance model.")

st.info(
    "Model logic (important):\n"
    "- Gas boiler cost = (annual useful heat / boiler efficiency) × gas price\n"
    "- HP+Booster: booster upgrades part of the heat to 70°C. The base HP must supply:\n"
    "  (1) low-temp heat + (2) the booster source heat.\n"
    "This avoids over-optimistic results."
)

# =========================================================
# PDF builder
# =========================================================
def build_pdf_report(payload: dict) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    left = 2.0 * cm
    y = h - 2.0 * cm

    def line(text, dy=0.7 * cm, font="Helvetica", size=11):
        nonlocal y
        if y < 2.0 * cm:
            c.showPage()
            y = h - 2.0 * cm
        c.setFont(font, size)
        c.drawString(left, y, text)
        y -= dy

    c.setTitle("Heat Pump vs Gas Boiler (V1)")

    line("Heat Pump vs Gas Boiler (V1) – Summary Report", dy=0.9 * cm, font="Helvetica-Bold", size=14)
    line(f"Generated: {payload['meta']['generated_at']}", dy=0.9 * cm, size=10)
    c.line(left, y, w - left, y)
    y -= 0.7 * cm

    line("Inputs", dy=0.8 * cm, font="Helvetica-Bold", size=12)
    for k, v in payload["inputs"].items():
        line(f"- {k}: {v}")

    y -= 0.3 * cm

    line("Results", dy=0.8 * cm, font="Helvetica-Bold", size=12)
    for k, v in payload["results"].items():
        line(f"- {k}: {v}")

    y -= 0.3 * cm

    line("Model note:", dy=0.7 * cm, font="Helvetica-Bold", size=11)
    line("Gas: (useful heat / boiler efficiency) × gas price.", size=10)
    line(
        "HP+Booster: base HP supplies low-temp heat + booster source heat; booster upgrades to 70°C.",
        size=10,
    )

    c.showPage()
    c.save()
    return buf.getvalue()


# =========================================================
# Sidebar inputs
# =========================================================
with st.sidebar:
    st.header("1) Heating Demand")
    q_annual_kwh_th = st.number_input(
        "Annual useful heat demand (kWh_th/year)",
        min_value=1.0,
        value=2_000_000.0,
        step=50_000.0,
    )

    st.header("2) Gas Boiler (Baseline)")
    eta_boiler = st.number_input(
        "Boiler seasonal efficiency (η)",
        min_value=0.50,
        max_value=1.00,
        value=0.93,
        step=0.01,
    )

    gas_price_gel_per_m3 = st.number_input(
        "Gas price (GEL per m³)",
        min_value=0.01,
        value=1.29,
        step=0.01,
    )

    kwh_per_m3 = st.number_input(
        "Gas energy content (kWh per m³)",
        min_value=5.0,
        max_value=15.0,
        value=10.0,
        step=0.1,
    )

    st.header("3) Electricity")
    el_price_gel_per_kwh = st.number_input(
        "Electricity price (GEL per kWh)",
        min_value=0.001,
        value=0.30,
        step=0.01,
        format="%.3f",
    )

    st.header("4) Heat Pump + Booster")
    cop_base = st.number_input(
        "Base heat pump COP (net, low/intermediate temp)",
        min_value=0.5,
        value=2.4,
        step=0.1,
    )

    cop_boost = st.number_input(
        "Booster COP (net, to 70°C)",
        min_value=0.5,
        value=6.3,
        step=0.1,
    )

    boost_share_pct = st.slider(
        "Share of annual heat that must be delivered at 70°C (%)",
        min_value=0,
        max_value=100,
        value=40,
        step=1,
    )

    with st.expander("Optional: CAPEX / Payback", expanded=False):
        enable_payback = st.checkbox("Calculate payback", value=False)
        capex_hp_booster_gel = st.number_input(
            "CAPEX: HP + Booster total (GEL)",
            min_value=0.0,
            value=0.0,
            step=10_000.0,
        )
        capex_boiler_gel = st.number_input(
            "CAPEX: Boiler / baseline (GEL)",
            min_value=0.0,
            value=0.0,
            step=10_000.0,
        )

# =========================================================
# Calculations
# =========================================================
f_boost = boost_share_pct / 100.0

Q70 = q_annual_kwh_th * f_boost
Qlow = q_annual_kwh_th * (1.0 - f_boost)

E_boost = Q70 / cop_boost if cop_boost > 0 else 0.0
Q_source = Q70 - E_boost

Q_base_out = Qlow + Q_source
E_base = Q_base_out / cop_base if cop_base > 0 else 0.0

E_total_hp = E_base + E_boost
cost_hp_gel = E_total_hp * el_price_gel_per_kwh

gas_input_kwh = q_annual_kwh_th / eta_boiler if eta_boiler > 0 else 0.0
gas_volume_m3 = gas_input_kwh / kwh_per_m3 if kwh_per_m3 > 0 else 0.0
cost_gas_gel = gas_volume_m3 * gas_price_gel_per_m3

annual_savings_gel = cost_gas_gel - cost_hp_gel
cop_effective = q_annual_kwh_th / E_total_hp if E_total_hp > 0 else 0.0

# =========================================================
# KPI row
# =========================================================
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Annual useful heat", f"{q_annual_kwh_th:,.0f} kWh_th")
k2.metric("70°C share", f"{boost_share_pct}%")
k3.metric("Electricity price", f"{el_price_gel_per_kwh:.3f} GEL/kWh")
k4.metric("Gas price", f"{gas_price_gel_per_m3:.2f} GEL/m³")

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs(["📊 Summary", "🧮 Details", "📄 Export"])

with tab1:
    r1, r2, r3 = st.columns(3)
    r1.metric("Annual cost (Boiler)", f"{cost_gas_gel:,.0f} GEL")
    r2.metric("Annual cost (HP+Booster)", f"{cost_hp_gel:,.0f} GEL")

    if annual_savings_gel >= 0:
        r3.metric("Annual savings", f"{annual_savings_gel:,.0f} GEL")
    else:
        r3.metric("Annual difference", f"{annual_savings_gel:,.0f} GEL")
        st.warning("HP+Booster is more expensive than gas under these inputs.")

    df_cost = pd.DataFrame(
        {"System": ["Gas Boiler", "HP + Booster"], "Annual cost (GEL)": [cost_gas_gel, cost_hp_gel]}
    ).set_index("System")

    st.subheader("Cost comparison")
    st.bar_chart(df_cost)

    if enable_payback:
        st.subheader("Payback (Optional)")
        delta_capex = capex_hp_booster_gel - capex_boiler_gel

        p1, p2, p3 = st.columns(3)
        p1.metric("Extra CAPEX (HP+Booster − Boiler)", f"{delta_capex:,.0f} GEL")

        if annual_savings_gel > 0 and delta_capex > 0:
            payback_years = delta_capex / annual_savings_gel
            p2.metric("Payback (years)", f"{payback_years:.2f}")
            p3.metric("Payback (months)", f"{payback_years * 12:.1f}")
        else:
            p2.metric("Payback (years)", "N/A")
            p3.metric("Payback (months)", "N/A")
            st.caption("Payback requires: annual savings > 0 and extra CAPEX > 0.")

with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Gas Boiler (Baseline)")
        st.write(f"Boiler efficiency (η): **{eta_boiler:.2f}**")
        st.write(f"Gas input energy: **{gas_input_kwh:,.0f} kWh_gas/year**")
        st.write(f"Gas volume: **{gas_volume_m3:,.0f} m³/year**")
        st.write(f"Annual cost: **{cost_gas_gel:,.0f} GEL/year**")

    with c2:
        st.subheader("Heat Pump + Booster to 70°C")
        st.write(f"Base COP: **{cop_base:.2f}**")
        st.write(f"Booster COP: **{cop_boost:.2f}**")
        st.write(f"Booster delivered heat (70°C): **{Q70:,.0f} kWh_th/year**")
        st.write(f"Base HP delivered heat: **{Q_base_out:,.0f} kWh_th/year**")
        st.write("---")
        st.write(f"Base electricity: **{E_base:,.0f} kWh_el/year**")
        st.write(f"Booster electricity: **{E_boost:,.0f} kWh_el/year**")
        st.write(f"Total electricity: **{E_total_hp:,.0f} kWh_el/year**")
        st.write(f"Effective system COP: **{cop_effective:.2f}**")
        st.write(f"Annual cost: **{cost_hp_gel:,.0f} GEL/year**")

    st.subheader("Energy balance sanity check")
    st.write(
        f"Q70 = Q_source + E_boost → {Q70:,.0f} = {Q_source:,.0f} + {E_boost:,.0f} kWh"
    )

with tab3:
    pdf_payload = {
        "meta": {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")},
        "inputs": {
            "Annual useful heat demand (kWh_th/year)": f"{q_annual_kwh_th:,.0f}",
            "Boiler efficiency (η)": f"{eta_boiler:.2f}",
            "Gas price (GEL/m³)": f"{gas_price_gel_per_m3:.2f}",
            "Gas energy content (kWh/m³)": f"{kwh_per_m3:.2f}",
            "Electricity price (GEL/kWh)": f"{el_price_gel_per_kwh:.3f}",
            "Base HP COP": f"{cop_base:.2f}",
            "Booster COP (to 70°C)": f"{cop_boost:.2f}",
            "70°C share (%)": f"{boost_share_pct}%",
        },
        "results": {
            "Gas boiler annual cost (GEL/year)": f"{cost_gas_gel:,.0f}",
            "HP+Booster annual cost (GEL/year)": f"{cost_hp_gel:,.0f}",
            "Annual savings (GEL/year)": f"{annual_savings_gel:,.0f}",
            "Effective system COP": f"{cop_effective:.2f}",
            "Total electricity (kWh_el/year)": f"{E_total_hp:,.0f}",
            "Gas volume (m³/year)": f"{gas_volume_m3:,.0f}",
        },
    }

    pdf_bytes = build_pdf_report(pdf_payload)

    st.download_button(
        label="📄 Download PDF report",
        data=pdf_bytes,
        file_name=f"hp_vs_boiler_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    df_export = pd.DataFrame(
        [{"Type": "Input", "Key": k, "Value": v} for k, v in pdf_payload["inputs"].items()]
        + [{"Type": "Result", "Key": k, "Value": v} for k, v in pdf_payload["results"].items()]
    )

    csv_bytes = df_export.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name=f"hp_vs_boiler_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
