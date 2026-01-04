import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# =========================================================
# Config
# =========================================================
st.set_page_config(
    page_title="Heat Pump vs Gas Boiler (V5 - Clean)",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Heat Pump vs Gas Boiler"
APP_VER = "V5 (Clean, HP≤50°C + Booster optional, SH + DHW)"
HP_MAX_SUPPLY_C = 50  # per your requirement


# =========================================================
# Presets
# =========================================================
BUILDING_TYPES = ["Office", "Hotel", "Hospital"]
INSULATION_LEVELS = ["Poor / Old", "Standard", "Good / New", "High-performance"]
CLIMATES = ["Tbilisi", "Batumi", "Gudauri"]

APPLICATIONS = ["Space heating only", "Space heating + DHW (year-round DHW)"]

# Annual useful heat intensity (kWh_th/m²·year) baseline (Tbilisi-ish) – total heat (SH+DHW)
BASE_KWH_PER_M2_YEAR = {
    "Office": {"Poor / Old": 140, "Standard": 100, "Good / New": 70, "High-performance": 45},
    "Hotel": {"Poor / Old": 230, "Standard": 170, "Good / New": 130, "High-performance": 90},
    "Hospital": {"Poor / Old": 320, "Standard": 250, "Good / New": 200, "High-performance": 150},
}
CLIMATE_INTENSITY_FACTOR = {"Tbilisi": 1.00, "Batumi": 0.90, "Gudauri": 1.25}

# Full-load hours presets for peak kW -> annual kWh (Tbilisi-ish)
BASE_FLH = {
    "Office": {"Poor / Old": 1800, "Standard": 1600, "Good / New": 1400, "High-performance": 1200},
    "Hotel": {"Poor / Old": 2400, "Standard": 2200, "Good / New": 2000, "High-performance": 1800},
    "Hospital": {"Poor / Old": 2600, "Standard": 2400, "Good / New": 2200, "High-performance": 2000},
}
CLIMATE_FLH_FACTOR = {"Tbilisi": 1.00, "Batumi": 0.92, "Gudauri": 1.18}

# DHW share presets (fraction of total annual useful heat)
DHW_SHARE_PRESET = {"Office": 0.10, "Hotel": 0.30, "Hospital": 0.35}

# Heating regimes (proxy for required SH supply temp)
HEATING_REGIMES = {
    "45/35 °C (underfloor/very low-temp)": (45, 35),
    "50/40 °C (low-temp radiators/FCU)": (50, 40),
    "60/40 °C (medium radiators/FCU)": (60, 40),
    "70/50 °C (high-temp radiators/AHU coils)": (70, 50),
    "80/60 °C (very high-temp radiators)": (80, 60),
}

# If you enable "Mixed systems", only the fraction assigned to "High-temp loop" will be boosted.
# Otherwise, SH is either fully ≤50 or fully >50 based on regime selection.


# COP winter 3-point weights by climate for optional "advanced bin" method
COP_TEMPS_3 = [-3, 2, 7]
SH_WEIGHTS_3 = {
    "Tbilisi": {-3: 0.25, 2: 0.45, 7: 0.30},
    "Batumi": {-3: 0.10, 2: 0.35, 7: 0.55},
    "Gudauri": {-3: 0.45, 2: 0.40, 7: 0.15},
}


# =========================================================
# Helpers
# =========================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def weighted_avg(points: dict, weights: dict) -> float:
    available = [t for t in weights.keys() if t in points and points[t] > 0]
    if not available:
        return 0.0
    wsum = sum(weights[t] for t in available)
    if wsum <= 0:
        return 0.0
    return sum(points[t] * (weights[t] / wsum) for t in available)


def hp_booster_chain(Q_low: float, Q_high: float, cop_base: float, cop_boost: float) -> dict:
    """
    Defensible energy balance chain:
      - Booster delivers Q_high. Electricity: E_boost = Q_high / cop_boost
      - Booster source heat from base: Q_source = Q_high - E_boost
      - Base must deliver: Q_base_out = Q_low + Q_source
      - Base electricity: E_base = Q_base_out / cop_base
    """
    Q_low = max(0.0, Q_low)
    Q_high = max(0.0, Q_high)
    cop_base = max(1e-9, cop_base)

    if Q_high <= 0 or cop_boost <= 0:
        E_boost = 0.0
        Q_source = 0.0
    else:
        E_boost = Q_high / cop_boost
        Q_source = Q_high - E_boost

    Q_base_out = Q_low + Q_source
    E_base = Q_base_out / cop_base

    return {"E_base": E_base, "E_boost": E_boost, "Q_source": Q_source, "Q_base_out": Q_base_out}


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

    c.setTitle(payload["meta"]["title"])
    line(payload["meta"]["title"], dy=0.9 * cm, font="Helvetica-Bold", size=14)
    line(f"Generated: {payload['meta']['generated_at']}", dy=0.9 * cm, size=10)
    line(f"Mode: {payload['meta']['mode']}", dy=0.8 * cm, size=10)
    c.line(left, y, w - left, y)
    y -= 0.7 * cm

    line("Inputs", dy=0.8 * cm, font="Helvetica-Bold", size=12)
    for k, v in payload["inputs"].items():
        line(f"- {k}: {v}", size=11)

    y -= 0.3 * cm
    line("Results", dy=0.8 * cm, font="Helvetica-Bold", size=12)
    for k, v in payload["results"].items():
        line(f"- {k}: {v}", size=11)

    y -= 0.3 * cm
    line("Model note:", dy=0.7 * cm, font="Helvetica-Bold", size=11)
    line("Gas: (useful heat / boiler efficiency) × gas price.", size=10)
    line("HP: Base HP supplies ≤50°C loop. Any demand >50°C is boosted.", size=10)
    line("Booster balance: Q_high = Q_source + E_boost (defensible, non-optimistic).", size=10)

    c.showPage()
    c.save()
    return buf.getvalue()


# =========================================================
# UI header
# =========================================================
st.title(f"{APP_TITLE} — {APP_VER}")
st.caption("Clean feasibility tool. Base HP is capped at 50°C supply; higher-temp demand goes through booster if installed.")

st.info(
    "Key assumptions:\n"
    f"- Base heat pump max supply temperature = {HP_MAX_SUPPLY_C}°C\n"
    "- If space-heating supply > 50°C, that portion must be boosted.\n"
    "- Booster COP is a single constant (no ambient dependence).\n"
    "- DHW is assumed feasible from base HP by default (50°C), unless you enable high-temp DHW."
)

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("1) Mode & Application")
    mode = st.radio("Project mode", ["Existing building (known demand)", "Scratch project (estimate demand)"], index=1)
    application = st.selectbox("Application", APPLICATIONS, index=1)

    st.header("2) Climate")
    climate = st.selectbox("Climate", CLIMATES, index=0)

    st.divider()
    st.header("3) Prices")
    el_price_gel_per_kwh = st.number_input("Electricity price (GEL/kWh)", min_value=0.001, value=0.30, step=0.01, format="%.3f")
    gas_price_gel_per_m3 = st.number_input("Gas price (GEL/m³)", min_value=0.01, value=1.29, step=0.01)

    with st.expander("Advanced: Gas conversion", expanded=False):
        kwh_per_m3 = st.number_input("Gas energy content (kWh/m³)", min_value=5.0, max_value=15.0, value=10.0, step=0.1)

    st.divider()
    st.header("4) Baseline boiler")
    eta_boiler = st.number_input("Boiler seasonal efficiency (η)", min_value=0.50, max_value=1.00, value=0.93, step=0.01)

    st.divider()
    st.header("5) Heat pump performance")
    cop_input_method = st.radio("Base HP COP input", ["Single seasonal COP", "Advanced: 3-point winter COP"], index=0)

    if cop_input_method == "Single seasonal COP":
        cop_base = st.number_input("Base HP seasonal COP (≤50°C supply)", min_value=0.5, value=2.8, step=0.1)
    else:
        st.caption("Enter COP points for the base HP delivering ≤50°C supply. App computes seasonal COP using climate weights.")
        base_points = {}
        for t in COP_TEMPS_3:
            base_points[t] = st.number_input(f"COP at {t}°C (base HP)", min_value=0.1, value=2.6 + (t+3)*0.03, step=0.05)
        cop_base = weighted_avg(base_points, SH_WEIGHTS_3[climate])
        st.success(f"Derived seasonal COP (base HP): {cop_base:.2f}")

    st.divider()
    st.header("6) Booster")
    booster_installed = st.checkbox("Booster installed (for >50°C loads)", value=True)

    if booster_installed:
        cop_boost = st.number_input("Booster COP (constant)", min_value=0.5, value=6.3, step=0.1)
    else:
        cop_boost = 0.0

    st.divider()
    st.header("Optional: CAPEX / Payback")
    enable_payback = st.checkbox("Calculate payback", value=False)
    capex_hp_booster_gel = st.number_input("CAPEX: HP system total (GEL)", min_value=0.0, value=0.0, step=10_000.0)
    capex_boiler_gel = st.number_input("CAPEX: Boiler / baseline (GEL)", min_value=0.0, value=0.0, step=10_000.0)


# =========================================================
# Demand definition
# =========================================================
scratch_meta = {"mode": mode, "application": application, "climate": climate}

if mode == "Existing building (known demand)":
    st.subheader("Demand")
    Q_total = st.number_input("Annual useful heat demand (kWh_th/year)", min_value=1.0, value=2_000_000.0, step=50_000.0)

    building_type = st.selectbox("Building type (for DHW preset only)", BUILDING_TYPES, index=1)
    scratch_meta["building_type"] = building_type

else:
    st.subheader("Scratch project — demand estimation")
    building_type = st.selectbox("Building type", BUILDING_TYPES, index=1)
    insulation = st.selectbox("Insulation level", INSULATION_LEVELS, index=1)
    scratch_meta.update({"building_type": building_type, "insulation": insulation})

    demand_method = st.radio(
        "Demand input method",
        ["From area (m²) + benchmarks", "From peak heat load (kW) + FLH", "Direct annual useful heat demand"],
        index=0,
    )

    if demand_method == "From area (m²) + benchmarks":
        area_m2 = st.number_input("Heated area (m²)", min_value=1.0, value=12000.0, step=100.0)
        base_intensity = float(BASE_KWH_PER_M2_YEAR[building_type][insulation])
        intensity = base_intensity * float(CLIMATE_INTENSITY_FACTOR[climate])
        Q_est = area_m2 * intensity
        st.caption(f"Benchmark intensity: {intensity:.0f} kWh/m²·year (includes climate factor)")
        scratch_meta.update({"demand_method": demand_method, "area_m2": area_m2, "kwh_m2_year": intensity})

    elif demand_method == "From peak heat load (kW) + FLH":
        p_peak_kw = st.number_input("Peak heating load (kW)", min_value=1.0, value=500.0, step=10.0)
        base_flh = float(BASE_FLH[building_type][insulation])
        flh = base_flh * float(CLIMATE_FLH_FACTOR[climate])
        override = st.checkbox("Override FLH", value=False)
        flh_used = st.number_input("Full-load hours (h/year)", min_value=200.0, value=float(round(flh)), step=100.0) if override else flh
        if not override:
            st.caption(f"FLH preset: {flh_used:.0f} h/year (includes climate factor)")
        Q_est = p_peak_kw * flh_used
        scratch_meta.update({"demand_method": demand_method, "p_peak_kw": p_peak_kw, "flh_used": flh_used})

    else:
        Q_est = st.number_input("Annual useful heat demand (kWh_th/year) — provided", min_value=1.0, value=2_000_000.0, step=50_000.0)
        scratch_meta.update({"demand_method": demand_method})

    Q_total = st.number_input("Annual useful heat demand used (kWh_th/year)", min_value=1.0, value=float(Q_est), step=50_000.0)


# =========================================================
# SH vs DHW split
# =========================================================
if application == "Space heating only":
    dhw_share_pct = 0
    sh_share_pct = 100
else:
    # default from building type preset + override
    preset_dhw = int(round(DHW_SHARE_PRESET[building_type] * 100))
    use_override = st.checkbox("Override DHW share", value=False)
    dhw_share_pct = st.slider("DHW share of annual heat (%)", 0, 100, preset_dhw, 1) if use_override else preset_dhw
    sh_share_pct = 100 - dhw_share_pct
    st.caption(f"Space heating share: {sh_share_pct}%")

scratch_meta.update({"sh_share_pct": sh_share_pct, "dhw_share_pct": dhw_share_pct})

Q_sh = Q_total * (sh_share_pct / 100.0)
Q_dhw = Q_total * (dhw_share_pct / 100.0)

# DHW target
if dhw_share_pct > 0:
    high_temp_dhw = st.checkbox("High-temp DHW required (≥60°C)", value=False)
    dhw_target_c = 60 if high_temp_dhw else 50
else:
    high_temp_dhw = False
    dhw_target_c = 50

scratch_meta.update({"dhw_target_c": dhw_target_c, "high_temp_dhw": high_temp_dhw})


# =========================================================
# Space heating temperature requirement and boosted fraction
# =========================================================
st.divider()
st.subheader("Space heating temperature requirement (Base HP max 50°C)")

if Q_sh > 0:
    regime_name = st.selectbox("Space heating regime (supply/return)", list(HEATING_REGIMES.keys()), index=2)
    supply_c, return_c = HEATING_REGIMES[regime_name]
    scratch_meta.update({"regime": regime_name, "supply_c": supply_c, "return_c": return_c})

    mixed = st.checkbox("Mixed space-heating systems (optional)", value=False)

    if not mixed:
        # All SH is either <=50 or >50 depending on supply temp
        sh_high_frac = 1.0 if supply_c > HP_MAX_SUPPLY_C else 0.0
        st.info(
            f"Selected supply {supply_c}°C. "
            f"Boosted fraction for SH = {'100%' if sh_high_frac == 1.0 else '0%'} (base HP cap {HP_MAX_SUPPLY_C}°C)."
        )
    else:
        st.caption("Specify fraction of space heating that requires >50°C supply (e.g., radiators/AHU coils loop).")
        sh_high_frac = st.slider("Fraction of SH needing >50°C (%)", 0, 100, 40, 1) / 100.0
        st.info(f"Boosted fraction for SH = {sh_high_frac*100:.0f}%")

else:
    regime_name = None
    supply_c = None
    sh_high_frac = 0.0

scratch_meta.update({"sh_high_frac": sh_high_frac})

# DHW boosted?
dhw_high_frac = 1.0 if (dhw_share_pct > 0 and dhw_target_c > HP_MAX_SUPPLY_C) else 0.0

# If booster not installed, force high-temp fractions to 0 and warn if needed
if not booster_installed:
    if sh_high_frac > 0 or dhw_high_frac > 0:
        st.warning("Booster is NOT installed, but some demand requires >50°C. For feasibility, boosted fractions are forced to 0% (system would need redesign).")
    sh_high_frac = 0.0
    dhw_high_frac = 0.0

Q_sh_high = Q_sh * sh_high_frac
Q_sh_low = Q_sh * (1.0 - sh_high_frac)

Q_dhw_high = Q_dhw * dhw_high_frac
Q_dhw_low = Q_dhw * (1.0 - dhw_high_frac)

boost_share_pct = int(round(0 if Q_total <= 0 else ((Q_sh_high + Q_dhw_high) / Q_total) * 100.0))

# =========================================================
# HP electricity (single base COP; booster COP constant)
# =========================================================
chain_sh = hp_booster_chain(Q_sh_low, Q_sh_high, cop_base, cop_boost)
chain_dhw = hp_booster_chain(Q_dhw_low, Q_dhw_high, cop_base, cop_boost)

E_total_hp = chain_sh["E_base"] + chain_sh["E_boost"] + chain_dhw["E_base"] + chain_dhw["E_boost"]
cost_hp_gel = E_total_hp * el_price_gel_per_kwh

# Gas baseline
gas_input_kwh = Q_total / eta_boiler if eta_boiler > 0 else 0.0
gas_volume_m3 = gas_input_kwh / kwh_per_m3 if kwh_per_m3 > 0 else 0.0
cost_gas_gel = gas_volume_m3 * gas_price_gel_per_m3

annual_savings_gel = cost_gas_gel - cost_hp_gel
cop_effective = (Q_total / E_total_hp) if E_total_hp > 0 else 0.0


# =========================================================
# KPI row
# =========================================================
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Annual useful heat", f"{Q_total:,.0f} kWh_th")
k2.metric("Boosted share (>50°C)", f"{boost_share_pct}%")
k3.metric("Base HP COP (used)", f"{cop_base:.2f}")
k4.metric("Booster installed", "Yes" if booster_installed else "No")


# =========================================================
# Tabs: Summary / Details / Export
# =========================================================
tab1, tab2, tab3 = st.tabs(["📊 Summary", "🧮 Details", "📄 Export"])

with tab1:
    r1, r2, r3 = st.columns(3)
    r1.metric("Annual cost (Boiler)", f"{cost_gas_gel:,.0f} GEL")
    r2.metric("Annual cost (HP system)", f"{cost_hp_gel:,.0f} GEL")
    if annual_savings_gel >= 0:
        r3.metric("Annual savings", f"{annual_savings_gel:,.0f} GEL")
    else:
        r3.metric("Annual difference", f"{annual_savings_gel:,.0f} GEL")
        st.warning("HP system is more expensive than gas under these inputs.")

    st.subheader("Cost comparison")
    df_cost = pd.DataFrame({"System": ["Gas Boiler", "HP System"], "Annual cost (GEL)": [cost_gas_gel, cost_hp_gel]})
    fig, ax = plt.subplots()
    ax.barh(df_cost["System"], df_cost["Annual cost (GEL)"])
    ax.set_xlabel("Annual cost (GEL)")
    ax.set_title("Annual cost comparison")
    for i, v in enumerate(df_cost["Annual cost (GEL)"]):
        ax.text(v, i, f"{v:,.0f} GEL", va="center", ha="left")
    st.pyplot(fig)

    if enable_payback:
        st.subheader("Payback (Optional)")
        delta_capex = capex_hp_booster_gel - capex_boiler_gel
        p1, p2, p3 = st.columns(3)
        p1.metric("Extra CAPEX (HP − Boiler)", f"{delta_capex:,.0f} GEL")
        if annual_savings_gel > 0 and delta_capex > 0:
            payback_years = delta_capex / annual_savings_gel
            p2.metric("Payback (years)", f"{payback_years:.2f}")
            p3.metric("Payback (months)", f"{payback_years * 12:.1f}")
        else:
            p2.metric("Payback (years)", "N/A")
            p3.metric("Payback (months)", "N/A")
            st.caption("Payback requires: annual savings > 0 and extra CAPEX > 0.")

with tab2:
    st.subheader("Transparency")
    st.write("### Split")
    st.write(f"- Space heating: **{sh_share_pct}%** → {Q_sh:,.0f} kWh_th")
    st.write(f"- DHW: **{dhw_share_pct}%** → {Q_dhw:,.0f} kWh_th (target {dhw_target_c}°C)")

    st.write("### Boosted fractions (>50°C)")
    st.write(f"- SH boosted fraction: **{sh_high_frac*100:.0f}%** → {Q_sh_high:,.0f} kWh_th")
    st.write(f"- DHW boosted fraction: **{dhw_high_frac*100:.0f}%** → {Q_dhw_high:,.0f} kWh_th")
    st.write(f"- Total boosted share: **{boost_share_pct}%**")

    st.divider()
    st.write("### HP energy balance")
    st.write("**Space heating chain**")
    st.write(f"- Base electricity: **{chain_sh['E_base']:,.0f} kWh_el**")
    st.write(f"- Booster electricity: **{chain_sh['E_boost']:,.0f} kWh_el**")
    st.write(f"- Booster source heat from base: **{chain_sh['Q_source']:,.0f} kWh_th**")
    st.write(f"- Base delivered heat: **{chain_sh['Q_base_out']:,.0f} kWh_th**")

    st.write("**DHW chain**")
    st.write(f"- Base electricity: **{chain_dhw['E_base']:,.0f} kWh_el**")
    st.write(f"- Booster electricity: **{chain_dhw['E_boost']:,.0f} kWh_el**")
    st.write(f"- Booster source heat from base: **{chain_dhw['Q_source']:,.0f} kWh_th**")
    st.write(f"- Base delivered heat: **{chain_dhw['Q_base_out']:,.0f} kWh_th**")

    st.divider()
    st.write("### Totals")
    st.write(f"- Total electricity: **{E_total_hp:,.0f} kWh_el/year**")
    st.write(f"- Effective system COP: **{cop_effective:.2f}**")
    st.write(f"- Gas volume: **{gas_volume_m3:,.0f} m³/year**")

with tab3:
    meta_mode = "Existing" if mode.startswith("Existing") else "Scratch"
    pdf_payload = {
        "meta": {
            "title": f"{APP_TITLE} — {APP_VER}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "mode": f"{meta_mode} | {application} | {climate}",
        },
        "inputs": {
            "Annual useful heat (kWh_th/year)": f"{Q_total:,.0f}",
            "Space heating share (%)": f"{sh_share_pct}%",
            "DHW share (%)": f"{dhw_share_pct}%",
            "DHW target (°C)": f"{dhw_target_c}",
            "Boosted share (>50°C) (%)": f"{boost_share_pct}%",
            "Base HP COP (≤50°C supply)": f"{cop_base:.2f}",
            "Booster installed": "Yes" if booster_installed else "No",
            "Booster COP (constant)": f"{cop_boost:.2f}" if booster_installed else "N/A",
            "Electricity price (GEL/kWh)": f"{el_price_gel_per_kwh:.3f}",
            "Gas price (GEL/m³)": f"{gas_price_gel_per_m3:.2f}",
            "Boiler efficiency (η)": f"{eta_boiler:.2f}",
        },
        "results": {
            "Gas annual cost (GEL/year)": f"{cost_gas_gel:,.0f}",
            "HP annual cost (GEL/year)": f"{cost_hp_gel:,.0f}",
            "Annual savings (GEL/year)": f"{annual_savings_gel:,.0f}",
            "Effective system COP": f"{cop_effective:.2f}",
            "Total electricity (kWh_el/year)": f"{E_total_hp:,.0f}",
            "Gas volume (m³/year)": f"{gas_volume_m3:,.0f}",
        },
    }

    # Add some scratch metadata lightly (keep PDF clean)
    for key in ["building_type", "insulation", "demand_method", "area_m2", "kwh_m2_year", "p_peak_kw", "flh_used", "regime", "supply_c"]:
        if key in scratch_meta and scratch_meta[key] is not None:
            pdf_payload["inputs"][f"Meta: {key}"] = str(scratch_meta[key])

    pdf_bytes = build_pdf_report(pdf_payload)

    st.download_button(
        "📄 Download PDF report",
        data=pdf_bytes,
        file_name=f"hp_vs_boiler_v5_{meta_mode.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    st.divider()
    st.subheader("CSV export (inputs + results)")
    df_export = pd.DataFrame(
        [{"Type": "Input", "Key": k, "Value": v} for k, v in pdf_payload["inputs"].items()]
        + [{"Type": "Result", "Key": k, "Value": v} for k, v in pdf_payload["results"].items()]
    )
    csv_bytes = df_export.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name=f"hp_vs_boiler_v5_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
