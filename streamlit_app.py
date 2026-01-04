import json
from io import BytesIO
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# PDF export (requires reportlab in requirements.txt)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# =========================================================
# App Config
# =========================================================
st.set_page_config(
    page_title="Treimax Energy Tools (HP vs Boiler + Chiller Payback)",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Treimax Energy Tools"
APP_VER = "V6 (Heating + Chiller + Save/Load JSON)"


# =========================================================
# Shared Helpers
# =========================================================
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
      Booster delivers Q_high. Electricity: E_boost = Q_high / cop_boost
      Booster source heat from base: Q_source = Q_high - E_boost
      Base must deliver: Q_base_out = Q_low + Q_source
      Base electricity: E_base = Q_base_out / cop_base
    """
    Q_low = max(0.0, float(Q_low))
    Q_high = max(0.0, float(Q_high))
    cop_base = max(1e-9, float(cop_base))

    if Q_high <= 0 or cop_boost <= 0:
        E_boost = 0.0
        Q_source = 0.0
    else:
        E_boost = Q_high / cop_boost
        Q_source = Q_high - E_boost

    Q_base_out = Q_low + Q_source
    E_base = Q_base_out / cop_base

    return {"E_base": E_base, "E_boost": E_boost, "Q_source": Q_source, "Q_base_out": Q_base_out}


def build_pdf_report(title: str, lines: list[str]) -> bytes:
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

    c.setTitle(title)
    line(title, dy=0.9 * cm, font="Helvetica-Bold", size=14)
    line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", dy=0.9 * cm, size=10)
    c.line(left, y, w - left, y)
    y -= 0.8 * cm

    for t in lines:
        line(t, size=11)

    c.showPage()
    c.save()
    return buf.getvalue()


# =========================================================
# Save/Load JSON Helpers (no DB)
# =========================================================
PROJECT_KEYS = [
    # Global
    "active_tool",

    # Heating tool
    "heat_mode",
    "heat_application",
    "heat_climate",
    "heat_el_price",
    "heat_gas_price",
    "heat_kwh_per_m3",
    "heat_eta_boiler",
    "heat_cop_source",
    "heat_scop",
    "heat_cop_method_manual",
    "heat_cop_method_scop",
    "heat_cop_base_manual",
    "heat_base_cop_m3",
    "heat_base_cop_p2",
    "heat_base_cop_p7",
    "heat_checkpoint_cop",
    "heat_booster_installed",
    "heat_cop_boost",
    "heat_enable_payback",
    "heat_capex_hp",
    "heat_capex_boiler",

    # Heating demand / scratch
    "heat_q_annual",
    "heat_building_type",
    "heat_insulation",
    "heat_demand_method",
    "heat_area_m2",
    "heat_peak_kw",
    "heat_override_flh",
    "heat_flh_used",
    "heat_dhw_override",
    "heat_dhw_share_pct",
    "heat_high_temp_dhw",

    # Heating temps
    "heat_regime_name",
    "heat_mixed_systems",
    "heat_sh_high_frac_pct",

    # Chiller tool
    "ch_mode",
    "ch_el_price",
    "ch_enable_payback",
    "ch_capex_new",
    "ch_capex_old",
    "ch_cool_demand_method",
    "ch_q_cool_annual",
    "ch_peak_cool_kw",
    "ch_override_cflh",
    "ch_cflh_used",
    "ch_eer_old",
    "ch_eer_new",
]


def collect_project_state() -> dict:
    payload = {"_meta": {"app": APP_TITLE, "ver": APP_VER, "saved_at": datetime.now().isoformat()}}
    for k in PROJECT_KEYS:
        if k in st.session_state:
            payload[k] = st.session_state[k]
    return payload


def apply_project_state(payload: dict) -> None:
    # Apply only known keys to avoid garbage
    for k in PROJECT_KEYS:
        if k in payload:
            st.session_state[k] = payload[k]


# =========================================================
# UI: Header + Save/Load
# =========================================================
st.title(f"{APP_TITLE} — {APP_VER}")

with st.sidebar:
    st.header("Project Save/Load (JSON)")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        if st.button("⬇️ Save project"):
            st.session_state["_download_project_json"] = json.dumps(
                collect_project_state(), ensure_ascii=False, indent=2
            )

    with col_s2:
        uploaded = st.file_uploader("⬆️ Load project", type=["json"], label_visibility="collapsed")
        if uploaded is not None:
            try:
                payload = json.load(uploaded)
                apply_project_state(payload)
                st.success("Loaded. Applying settings…")
                st.rerun()
            except Exception as e:
                st.error(f"Could not load JSON: {e}")

    if "_download_project_json" in st.session_state:
        st.download_button(
            "Download JSON",
            data=st.session_state["_download_project_json"].encode("utf-8"),
            file_name=f"treimax_project_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()


# =========================================================
# Tool selector (3 modes)
# =========================================================
tool = st.sidebar.radio(
    "Tool",
    ["Heat Pump vs Boiler", "Scratch project (estimate demand)", "Chiller EER payback"],
    key="active_tool",
)

# For Heating: we still keep an internal mode selector, but to match your request,
# we map the first two items to different default modes.
# - "Heat Pump vs Boiler" -> Existing building
# - "Scratch project" -> Scratch
# Both run the same heating engine.

# =========================================================
# Heating Tool Constants
# =========================================================
HP_MAX_SUPPLY_C = 50

BUILDING_TYPES = ["Office", "Hotel", "Hospital"]
INSULATION_LEVELS = ["Poor / Old", "Standard", "Good / New", "High-performance"]
CLIMATES = ["Tbilisi", "Batumi", "Gudauri"]
APPLICATIONS = ["Space heating only", "Space heating + DHW (year-round DHW)"]

BASE_KWH_PER_M2_YEAR = {
    "Office": {"Poor / Old": 140, "Standard": 100, "Good / New": 70, "High-performance": 45},
    "Hotel": {"Poor / Old": 230, "Standard": 170, "Good / New": 130, "High-performance": 90},
    "Hospital": {"Poor / Old": 320, "Standard": 250, "Good / New": 200, "High-performance": 150},
}
CLIMATE_INTENSITY_FACTOR = {"Tbilisi": 1.00, "Batumi": 0.90, "Gudauri": 1.25}

BASE_FLH = {
    "Office": {"Poor / Old": 1800, "Standard": 1600, "Good / New": 1400, "High-performance": 1200},
    "Hotel": {"Poor / Old": 2400, "Standard": 2200, "Good / New": 2000, "High-performance": 1800},
    "Hospital": {"Poor / Old": 2600, "Standard": 2400, "Good / New": 2200, "High-performance": 2000},
}
CLIMATE_FLH_FACTOR = {"Tbilisi": 1.00, "Batumi": 0.92, "Gudauri": 1.18}

DHW_SHARE_PRESET = {"Office": 0.10, "Hotel": 0.30, "Hospital": 0.35}

# Temperature-only regime labels (no equipment claims)
HEATING_REGIMES = {
    "45/35 °C": (45, 35),
    "50/40 °C": (50, 40),
    "55/45 °C": (55, 45),
    "60/40 °C": (60, 40),
    "70/50 °C": (70, 50),
    "80/60 °C (legacy/very high-temp loop)": (80, 60),
}

COP_TEMPS_3 = [-3, 2, 7]
SH_WEIGHTS_3 = {
    "Tbilisi": {-3: 0.25, 2: 0.45, 7: 0.30},
    "Batumi": {-3: 0.10, 2: 0.35, 7: 0.55},
    "Gudauri": {-3: 0.45, 2: 0.40, 7: 0.15},
}


# =========================================================
# Heating UI + Engine (for first two tools)
# =========================================================
def run_heating_ui(default_mode: str):
    st.subheader("Heat Pump vs Gas Boiler (HP ≤ 50°C + optional booster)")
    st.info(
        "Model:\n"
        "- Base HP supplies up to 50°C loop.\n"
        "- Any demand requiring >50°C must be boosted (if booster installed).\n"
        "- Booster COP is constant.\n"
        "- Boiler baseline uses seasonal efficiency and gas price."
    )

    # Sidebar inputs
    with st.sidebar:
        st.header("Heating Inputs")

        heat_mode = st.selectbox(
            "Project mode",
            ["Existing building (known demand)", "Scratch project (estimate demand)"],
            index=0 if default_mode.startswith("Existing") else 1,
            key="heat_mode",
        )

        heat_application = st.selectbox("Application", APPLICATIONS, index=1, key="heat_application")
        heat_climate = st.selectbox("Climate", CLIMATES, index=0, key="heat_climate")

        st.divider()
        st.subheader("Prices")
        heat_el_price = st.number_input("Electricity price (GEL/kWh)", min_value=0.001, value=0.30, step=0.01, format="%.3f", key="heat_el_price")
        heat_gas_price = st.number_input("Gas price (GEL/m³)", min_value=0.01, value=1.29, step=0.01, key="heat_gas_price")
        with st.expander("Advanced: Gas conversion", expanded=False):
            heat_kwh_per_m3 = st.number_input("Gas energy content (kWh/m³)", min_value=5.0, max_value=15.0, value=10.0, step=0.1, key="heat_kwh_per_m3")

        st.divider()
        st.subheader("Boiler baseline")
        heat_eta_boiler = st.number_input("Boiler seasonal efficiency (η)", min_value=0.50, max_value=1.00, value=0.93, step=0.01, key="heat_eta_boiler")

        st.divider()
        st.subheader("Base HP seasonal COP (≤50°C supply)")
        heat_cop_source = st.radio("COP source", ["Manual", "From datasheet SCOP"], index=0, key="heat_cop_source")

        if heat_cop_source == "From datasheet SCOP":
            heat_scop = st.number_input("SCOP (datasheet)", min_value=0.5, value=3.78, step=0.01, key="heat_scop")
            heat_cop_method_scop = st.radio("Use SCOP", ["Use SCOP directly", "Advanced: 3-point winter"], index=0, key="heat_cop_method_scop")
            if heat_cop_method_scop == "Use SCOP directly":
                cop_base = float(heat_scop)
            else:
                p_m3 = st.number_input("COP at −3°C (W≤50)", min_value=0.1, value=2.6, step=0.05, key="heat_base_cop_m3")
                p_2 = st.number_input("COP at +2°C (W≤50)", min_value=0.1, value=2.8, step=0.05, key="heat_base_cop_p2")
                p_7 = st.number_input("COP at +7°C (W≤50)", min_value=0.1, value=3.0, step=0.05, key="heat_base_cop_p7")
                cop_base = weighted_avg({-3: p_m3, 2: p_2, 7: p_7}, SH_WEIGHTS_3[heat_climate])
                st.caption(f"Derived seasonal COP: {cop_base:.2f}")
        else:
            heat_cop_method_manual = st.radio("Manual method", ["Single seasonal COP", "Advanced: 3-point winter"], index=0, key="heat_cop_method_manual")
            if heat_cop_method_manual == "Single seasonal COP":
                cop_base = st.number_input("Base HP seasonal COP (≤50°C)", min_value=0.5, value=2.8, step=0.1, key="heat_cop_base_manual")
            else:
                p_m3 = st.number_input("COP at −3°C (W≤50)", min_value=0.1, value=2.6, step=0.05, key="heat_base_cop_m3")
                p_2 = st.number_input("COP at +2°C (W≤50)", min_value=0.1, value=2.8, step=0.05, key="heat_base_cop_p2")
                p_7 = st.number_input("COP at +7°C (W≤50)", min_value=0.1, value=3.0, step=0.05, key="heat_base_cop_p7")
                cop_base = weighted_avg({-3: p_m3, 2: p_2, 7: p_7}, SH_WEIGHTS_3[heat_climate])
                st.caption(f"Derived seasonal COP: {cop_base:.2f}")

        with st.expander("Advanced: datasheet checkpoint warning", expanded=False):
            heat_checkpoint_cop = st.number_input("Checkpoint COP at cold/design (optional)", min_value=0.0, value=0.0, step=0.01, key="heat_checkpoint_cop")
        checkpoint_cop = float(st.session_state.get("heat_checkpoint_cop", 0.0) or 0.0)

        st.divider()
        st.subheader("Booster")
        heat_booster_installed = st.checkbox("Booster installed (for >50°C loads)", value=True, key="heat_booster_installed")
        if heat_booster_installed:
            cop_boost = st.number_input("Booster COP (constant)", min_value=0.5, value=6.3, step=0.1, key="heat_cop_boost")
        else:
            cop_boost = 0.0

        st.divider()
        st.subheader("Optional: CAPEX / Payback")
        heat_enable_payback = st.checkbox("Calculate payback", value=False, key="heat_enable_payback")
        heat_capex_hp = st.number_input("CAPEX: HP system (GEL)", min_value=0.0, value=0.0, step=10_000.0, key="heat_capex_hp")
        heat_capex_boiler = st.number_input("CAPEX: Boiler baseline (GEL)", min_value=0.0, value=0.0, step=10_000.0, key="heat_capex_boiler")

    # Demand block (main page)
    st.markdown("### Demand")
    heat_mode = st.session_state["heat_mode"]
    heat_application = st.session_state["heat_application"]
    heat_climate = st.session_state["heat_climate"]

    if heat_mode == "Existing building (known demand)":
        Q_total = st.number_input("Annual useful heat demand (kWh_th/year)", min_value=1.0, value=float(st.session_state.get("heat_q_annual", 2_000_000.0)), step=50_000.0, key="heat_q_annual")
        building_type = st.selectbox("Building type (DHW preset only)", BUILDING_TYPES, index=1, key="heat_building_type")
        insulation = st.session_state.get("heat_insulation", "Standard")
    else:
        c1, c2, c3 = st.columns([1.2, 1.0, 1.2])
        with c1:
            building_type = st.selectbox("Building type", BUILDING_TYPES, index=1, key="heat_building_type")
            insulation = st.selectbox("Insulation level", INSULATION_LEVELS, index=1, key="heat_insulation")

            demand_method = st.radio(
                "Demand method",
                ["From area (m²) + benchmarks", "From peak heat load (kW) + FLH", "Direct annual useful heat demand"],
                index=0,
                key="heat_demand_method",
            )

            if demand_method == "From area (m²) + benchmarks":
                area_m2 = st.number_input("Heated area (m²)", min_value=1.0, value=float(st.session_state.get("heat_area_m2", 12000.0)), step=100.0, key="heat_area_m2")
                base_intensity = float(BASE_KWH_PER_M2_YEAR[building_type][insulation])
                intensity = base_intensity * float(CLIMATE_INTENSITY_FACTOR[heat_climate])
                Q_est = area_m2 * intensity
                st.caption(f"Benchmark intensity: {intensity:.0f} kWh/m²·year (incl. climate factor)")

            elif demand_method == "From peak heat load (kW) + FLH":
                peak_kw = st.number_input("Peak heating load (kW)", min_value=1.0, value=float(st.session_state.get("heat_peak_kw", 500.0)), step=10.0, key="heat_peak_kw")
                base_flh = float(BASE_FLH[building_type][insulation])
                flh = base_flh * float(CLIMATE_FLH_FACTOR[heat_climate])
                override_flh = st.checkbox("Override FLH", value=bool(st.session_state.get("heat_override_flh", False)), key="heat_override_flh")
                flh_used = st.number_input("FLH used (h/year)", min_value=200.0, value=float(st.session_state.get("heat_flh_used", round(flh))), step=100.0, key="heat_flh_used") if override_flh else flh
                if not override_flh:
                    st.caption(f"FLH preset: {flh_used:.0f} h/year (incl. climate factor)")
                Q_est = peak_kw * flh_used
            else:
                Q_est = st.number_input("Annual useful heat demand (kWh_th/year) — provided", min_value=1.0, value=float(st.session_state.get("heat_q_annual", 2_000_000.0)), step=50_000.0)

        with c2:
            st.markdown("**Used in calculation**")
            Q_total = st.number_input(
                "Annual useful heat demand (kWh_th/year)",
                min_value=1.0,
                value=float(st.session_state.get("heat_q_annual", float(Q_est))),
                step=50_000.0,
                key="heat_q_annual",
            )

        with c3:
            st.markdown("**Notes**")
            st.write("- Scratch is feasibility-grade.")
            st.write("- You can overwrite the final annual heat demand anytime.")

    # DHW share
    if heat_application == "Space heating only":
        dhw_share_pct = 0
        sh_share_pct = 100
        high_temp_dhw = False
        dhw_target_c = 50
    else:
        preset = int(round(DHW_SHARE_PRESET[building_type] * 100))
        dhw_override = st.checkbox("Override DHW share", value=bool(st.session_state.get("heat_dhw_override", False)), key="heat_dhw_override")
        if dhw_override:
            dhw_share_pct = st.slider("DHW share (%)", 0, 100, int(st.session_state.get("heat_dhw_share_pct", preset)), 1, key="heat_dhw_share_pct")
        else:
            dhw_share_pct = preset
            st.session_state["heat_dhw_share_pct"] = dhw_share_pct
        sh_share_pct = 100 - dhw_share_pct

        high_temp_dhw = st.checkbox("High-temp DHW required (≥60°C)", value=bool(st.session_state.get("heat_high_temp_dhw", False)), key="heat_high_temp_dhw")
        dhw_target_c = 60 if high_temp_dhw else 50

    Q_sh = float(Q_total) * (sh_share_pct / 100.0)
    Q_dhw = float(Q_total) * (dhw_share_pct / 100.0)

    # Space heating regime and boosted fraction
    st.markdown("### Space heating temperatures (base HP capped at 50°C)")
    if Q_sh > 0:
        regime_name = st.selectbox("Space heating regime", list(HEATING_REGIMES.keys()), index=3, key="heat_regime_name")
        supply_c, _ = HEATING_REGIMES[regime_name]

        mixed = st.checkbox("Mixed systems (optional)", value=bool(st.session_state.get("heat_mixed_systems", False)), key="heat_mixed_systems")
        if not mixed:
            sh_high_frac = 1.0 if supply_c > HP_MAX_SUPPLY_C else 0.0
            st.info(
                f"Selected supply {supply_c}°C → boosted fraction for SH = "
                f"{'100%' if sh_high_frac == 1.0 else '0%'} (HP cap {HP_MAX_SUPPLY_C}°C)."
            )
        else:
            sh_high_frac_pct = st.slider("Fraction of SH needing >50°C (%)", 0, 100, int(st.session_state.get("heat_sh_high_frac_pct", 40)), 1, key="heat_sh_high_frac_pct")
            sh_high_frac = sh_high_frac_pct / 100.0
    else:
        sh_high_frac = 0.0
        supply_c = 0.0
        regime_name = None

    # DHW boosted fraction (only if DHW target >50)
    dhw_high_frac = 1.0 if (Q_dhw > 0 and dhw_target_c > HP_MAX_SUPPLY_C) else 0.0

    booster_installed = bool(st.session_state.get("heat_booster_installed", True))
    cop_boost = float(st.session_state.get("heat_cop_boost", 0.0)) if booster_installed else 0.0

    if not booster_installed:
        if sh_high_frac > 0 or dhw_high_frac > 0:
            st.warning("Booster is OFF, but some demand requires >50°C. Model forces boosted fractions to 0% (system would need redesign).")
        sh_high_frac = 0.0
        dhw_high_frac = 0.0

    # Split heat
    Q_sh_high = Q_sh * sh_high_frac
    Q_sh_low = Q_sh * (1.0 - sh_high_frac)
    Q_dhw_high = Q_dhw * dhw_high_frac
    Q_dhw_low = Q_dhw * (1.0 - dhw_high_frac)

    boosted_share = 0.0 if Q_total <= 0 else (Q_sh_high + Q_dhw_high) / float(Q_total)
    boost_share_pct = int(round(boosted_share * 100))

    # Compute energies/costs
    el_price = float(st.session_state["heat_el_price"])
    gas_price = float(st.session_state["heat_gas_price"])
    kwh_per_m3 = float(st.session_state["heat_kwh_per_m3"])
    eta_boiler = float(st.session_state["heat_eta_boiler"])

    chain_sh = hp_booster_chain(Q_sh_low, Q_sh_high, cop_base, cop_boost)
    chain_dhw = hp_booster_chain(Q_dhw_low, Q_dhw_high, cop_base, cop_boost)

    E_total_hp = chain_sh["E_base"] + chain_sh["E_boost"] + chain_dhw["E_base"] + chain_dhw["E_boost"]
    cost_hp = E_total_hp * el_price

    gas_input_kwh = float(Q_total) / eta_boiler if eta_boiler > 0 else 0.0
    gas_m3 = gas_input_kwh / kwh_per_m3 if kwh_per_m3 > 0 else 0.0
    cost_gas = gas_m3 * gas_price

    savings = cost_gas - cost_hp
    eff_cop = (float(Q_total) / E_total_hp) if E_total_hp > 0 else 0.0

    # Datasheet checkpoint warning (only warning)
    if checkpoint_cop and checkpoint_cop > 0:
        # Basic guard: seasonal COP shouldn't be wildly above cold-point COP.
        # We warn if seasonal COP exceeds checkpoint by > ~60% (tunable but sensible for feasibility).
        if cop_base > checkpoint_cop * 1.6:
            st.warning(
                f"Seasonal COP ({cop_base:.2f}) seems optimistic vs checkpoint COP ({checkpoint_cop:.2f}). "
                "Consider lowering seasonal COP or using conservative assumptions."
            )

    # Outputs
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annual useful heat", f"{float(Q_total):,.0f} kWh_th")
    k2.metric("Boosted share (>50°C)", f"{boost_share_pct}%")
    k3.metric("Base seasonal COP used", f"{cop_base:.2f}")
    k4.metric("Booster installed", "Yes" if booster_installed else "No")

    tab1, tab2, tab3 = st.tabs(["📊 Summary", "🧮 Details", "📄 Export"])

    with tab1:
        a, b, c = st.columns(3)
        a.metric("Annual cost (Boiler)", f"{cost_gas:,.0f} GEL")
        b.metric("Annual cost (HP system)", f"{cost_hp:,.0f} GEL")
        if savings >= 0:
            c.metric("Annual savings", f"{savings:,.0f} GEL")
        else:
            c.metric("Annual difference", f"{savings:,.0f} GEL")
            st.warning("HP system is more expensive than gas under these inputs.")

        # readable chart
        df_cost = pd.DataFrame({"System": ["Gas Boiler", "HP System"], "Annual cost (GEL)": [cost_gas, cost_hp]})
        fig, ax = plt.subplots()
        ax.barh(df_cost["System"], df_cost["Annual cost (GEL)"])
        ax.set_xlabel("Annual cost (GEL)")
        ax.set_title("Annual cost comparison")
        for i, v in enumerate(df_cost["Annual cost (GEL)"]):
            ax.text(v, i, f"{v:,.0f} GEL", va="center", ha="left")
        st.pyplot(fig)

        if bool(st.session_state.get("heat_enable_payback", False)):
            delta_capex = float(st.session_state.get("heat_capex_hp", 0.0)) - float(st.session_state.get("heat_capex_boiler", 0.0))
            st.subheader("Payback (optional)")
            p1, p2, p3 = st.columns(3)
            p1.metric("Extra CAPEX (HP − Boiler)", f"{delta_capex:,.0f} GEL")
            if savings > 0 and delta_capex > 0:
                pb_years = delta_capex / savings
                p2.metric("Payback (years)", f"{pb_years:.2f}")
                p3.metric("Payback (months)", f"{pb_years*12:.1f}")
            else:
                p2.metric("Payback (years)", "N/A")
                p3.metric("Payback (months)", "N/A")

    with tab2:
        st.write("### Split")
        st.write(f"- Space heating: **{sh_share_pct}%** → {Q_sh:,.0f} kWh_th")
        st.write(f"- DHW: **{dhw_share_pct}%** → {Q_dhw:,.0f} kWh_th (target {dhw_target_c}°C)")
        st.write("### Boosting")
        st.write(f"- SH boosted fraction: **{sh_high_frac*100:.0f}%** → {Q_sh_high:,.0f} kWh_th")
        st.write(f"- DHW boosted fraction: **{dhw_high_frac*100:.0f}%** → {Q_dhw_high:,.0f} kWh_th")
        st.write(f"- Total boosted share: **{boost_share_pct}%**")
        st.divider()
        st.write("### HP energy balance")
        st.write("**Space heating chain**")
        st.write(f"- Base electricity: **{chain_sh['E_base']:,.0f} kWh_el**")
        st.write(f"- Booster electricity: **{chain_sh['E_boost']:,.0f} kWh_el**")
        st.write(f"- Booster source heat from base: **{chain_sh['Q_source']:,.0f} kWh_th**")
        st.write("**DHW chain**")
        st.write(f"- Base electricity: **{chain_dhw['E_base']:,.0f} kWh_el**")
        st.write(f"- Booster electricity: **{chain_dhw['E_boost']:,.0f} kWh_el**")
        st.write(f"- Booster source heat from base: **{chain_dhw['Q_source']:,.0f} kWh_th**")
        st.divider()
        st.write("### Totals")
        st.write(f"- Total electricity: **{E_total_hp:,.0f} kWh_el/year**")
        st.write(f"- Effective system COP: **{eff_cop:.2f}**")
        st.write(f"- Gas volume: **{gas_m3:,.0f} m³/year**")

    with tab3:
        # PDF
        pdf_lines = [
            f"Mode: {heat_mode} | {heat_application} | {heat_climate}",
            "",
            "Inputs:",
            f"- Annual useful heat: {float(Q_total):,.0f} kWh_th/year",
            f"- SH share: {sh_share_pct}% | DHW share: {dhw_share_pct}%",
            f"- DHW target: {dhw_target_c}°C",
            f"- SH regime: {regime_name or 'N/A'} (supply {supply_c if supply_c else 'N/A'}°C)",
            f"- Boosted share (>50°C): {boost_share_pct}%",
            f"- Base seasonal COP used: {cop_base:.2f}",
            f"- Booster installed: {'Yes' if booster_installed else 'No'} | Booster COP: {cop_boost:.2f}" if booster_installed else "- Booster installed: No",
            f"- Electricity price: {el_price:.3f} GEL/kWh",
            f"- Gas price: {gas_price:.2f} GEL/m³ | Gas energy: {kwh_per_m3:.1f} kWh/m³",
            f"- Boiler efficiency: {eta_boiler:.2f}",
            "",
            "Results:",
            f"- Gas annual cost: {cost_gas:,.0f} GEL/year",
            f"- HP annual cost: {cost_hp:,.0f} GEL/year",
            f"- Annual savings: {savings:,.0f} GEL/year",
            f"- Effective system COP: {eff_cop:.2f}",
        ]
        pdf_bytes = build_pdf_report("HP vs Boiler Report", pdf_lines)
        st.download_button(
            "📄 Download PDF report",
            data=pdf_bytes,
            file_name=f"hp_vs_boiler_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        # CSV
        df_export = pd.DataFrame(
            [
                ("Input", "Annual useful heat (kWh_th/year)", float(Q_total)),
                ("Input", "SH share (%)", sh_share_pct),
                ("Input", "DHW share (%)", dhw_share_pct),
                ("Input", "DHW target (°C)", dhw_target_c),
                ("Input", "Boosted share (%)", boost_share_pct),
                ("Input", "Base COP used", cop_base),
                ("Input", "Booster installed", booster_installed),
                ("Input", "Booster COP", cop_boost if booster_installed else None),
                ("Result", "Cost gas (GEL/year)", cost_gas),
                ("Result", "Cost HP (GEL/year)", cost_hp),
                ("Result", "Savings (GEL/year)", savings),
            ],
            columns=["Type", "Key", "Value"],
        )
        st.download_button(
            "⬇️ Download CSV",
            data=df_export.to_csv(index=False).encode("utf-8"),
            file_name=f"hp_vs_boiler_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# Chiller Tool (EER Payback)
# =========================================================
def run_chiller_ui():
    st.subheader("Chiller EER Payback (simple feasibility)")
    st.info(
        "Model:\n"
        "- Annual cooling electricity = Annual cooling demand (kWh_cool) / EER\n"
        "- Annual cost = kWh_el × electricity price\n"
        "- Savings = old cost − new cost\n"
        "- Payback = extra CAPEX / savings (optional)"
    )

    with st.sidebar:
        st.header("Chiller Inputs")

        st.selectbox(
            "Project mode",
            ["Existing building (known demand)", "Scratch project (estimate demand)"],
            index=0,
            key="ch_mode",
        )

        st.divider()
        st.subheader("Electricity")
        ch_el_price = st.number_input("Electricity price (GEL/kWh)", min_value=0.001, value=0.30, step=0.01, format="%.3f", key="ch_el_price")

        st.divider()
        st.subheader("Efficiency")
        st.caption("EER here is net seasonal/average you want to assume. If you only have full-load EER, be conservative.")
        eer_old = st.number_input("Existing chiller EER (avg)", min_value=0.5, value=3.2, step=0.1, key="ch_eer_old")
        eer_new = st.number_input("New chiller EER (avg)", min_value=0.5, value=4.2, step=0.1, key="ch_eer_new")

        st.divider()
        st.subheader("Optional: CAPEX / Payback")
        ch_enable_payback = st.checkbox("Calculate payback", value=False, key="ch_enable_payback")
        ch_capex_new = st.number_input("CAPEX: New chiller system (GEL)", min_value=0.0, value=0.0, step=10_000.0, key="ch_capex_new")
        ch_capex_old = st.number_input("CAPEX: Baseline / keep old (GEL)", min_value=0.0, value=0.0, step=10_000.0, key="ch_capex_old")

    # Demand entry (main page)
    st.markdown("### Cooling demand")
    demand_method = st.radio(
        "Demand input method",
        ["Direct annual cooling demand (kWh_cool/year)", "From peak cooling load (kW) + cooling FLH"],
        index=0,
        key="ch_cool_demand_method",
    )

    if demand_method.startswith("Direct"):
        Q_cool = st.number_input(
            "Annual cooling demand (kWh_cool/year)",
            min_value=1.0,
            value=float(st.session_state.get("ch_q_cool_annual", 1_000_000.0)),
            step=50_000.0,
            key="ch_q_cool_annual",
        )
        cflh_used = None
        peak_kw = None
    else:
        peak_kw = st.number_input(
            "Peak cooling load (kW)",
            min_value=1.0,
            value=float(st.session_state.get("ch_peak_cool_kw", 500.0)),
            step=10.0,
            key="ch_peak_cool_kw",
        )
        st.caption("Cooling FLH = equivalent full-load cooling hours per year (use 1200–2500 as common feasibility range).")
        override = st.checkbox("Override cooling FLH", value=bool(st.session_state.get("ch_override_cflh", True)), key="ch_override_cflh")
        cflh_used = st.number_input(
            "Cooling FLH used (h/year)",
            min_value=200.0,
            value=float(st.session_state.get("ch_cflh_used", 1800.0)),
            step=100.0,
            key="ch_cflh_used",
        ) if override else 1800.0
        Q_cool = peak_kw * cflh_used

    # Calc
    el_price = float(st.session_state["ch_el_price"])
    eer_old = float(st.session_state["ch_eer_old"])
    eer_new = float(st.session_state["ch_eer_new"])

    E_old = float(Q_cool) / eer_old if eer_old > 0 else 0.0
    E_new = float(Q_cool) / eer_new if eer_new > 0 else 0.0

    cost_old = E_old * el_price
    cost_new = E_new * el_price
    savings = cost_old - cost_new

    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annual cooling demand", f"{float(Q_cool):,.0f} kWh_cool")
    k2.metric("Old EER", f"{eer_old:.2f}")
    k3.metric("New EER", f"{eer_new:.2f}")
    k4.metric("Annual savings", f"{savings:,.0f} GEL")

    tab1, tab2, tab3 = st.tabs(["📊 Summary", "🧮 Details", "📄 Export"])

    with tab1:
        a, b, c = st.columns(3)
        a.metric("Old annual cost", f"{cost_old:,.0f} GEL")
        b.metric("New annual cost", f"{cost_new:,.0f} GEL")
        if savings >= 0:
            c.metric("Savings", f"{savings:,.0f} GEL/year")
        else:
            c.metric("Difference", f"{savings:,.0f} GEL/year")
            st.warning("New chiller is worse than baseline under these assumptions.")

        df = pd.DataFrame({"Case": ["Old chiller", "New chiller"], "Annual cost (GEL)": [cost_old, cost_new]})
        fig, ax = plt.subplots()
        ax.barh(df["Case"], df["Annual cost (GEL)"])
        ax.set_xlabel("Annual cost (GEL)")
        ax.set_title("Chiller annual cost comparison")
        for i, v in enumerate(df["Annual cost (GEL)"]):
            ax.text(v, i, f"{v:,.0f} GEL", va="center", ha="left")
        st.pyplot(fig)

        if bool(st.session_state.get("ch_enable_payback", False)):
            delta_capex = float(st.session_state.get("ch_capex_new", 0.0)) - float(st.session_state.get("ch_capex_old", 0.0))
            st.subheader("Payback (optional)")
            p1, p2, p3 = st.columns(3)
            p1.metric("Extra CAPEX (New − Old)", f"{delta_capex:,.0f} GEL")
            if savings > 0 and delta_capex > 0:
                pb_years = delta_capex / savings
                p2.metric("Payback (years)", f"{pb_years:.2f}")
                p3.metric("Payback (months)", f"{pb_years*12:.1f}")
            else:
                p2.metric("Payback (years)", "N/A")
                p3.metric("Payback (months)", "N/A")

    with tab2:
        st.write("### Electricity")
        st.write(f"- Old electricity: **{E_old:,.0f} kWh_el/year**")
        st.write(f"- New electricity: **{E_new:,.0f} kWh_el/year**")
        st.write("### Notes")
        if cflh_used is not None:
            st.write(f"- Peak cooling load: **{peak_kw:,.0f} kW**")
            st.write(f"- Cooling FLH used: **{cflh_used:,.0f} h/year**")

    with tab3:
        pdf_lines = [
            "Chiller EER Payback Report",
            "",
            "Inputs:",
            f"- Annual cooling demand: {float(Q_cool):,.0f} kWh_cool/year",
            f"- Electricity price: {el_price:.3f} GEL/kWh",
            f"- Old EER: {eer_old:.2f}",
            f"- New EER: {eer_new:.2f}",
            "",
            "Results:",
            f"- Old annual cost: {cost_old:,.0f} GEL/year",
            f"- New annual cost: {cost_new:,.0f} GEL/year",
            f"- Annual savings: {savings:,.0f} GEL/year",
        ]
        pdf_bytes = build_pdf_report("Chiller EER Payback Report", pdf_lines)
        st.download_button(
            "📄 Download PDF report",
            data=pdf_bytes,
            file_name=f"chiller_eer_payback_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        df_export = pd.DataFrame(
            [
                ("Input", "Annual cooling demand (kWh_cool/year)", float(Q_cool)),
                ("Input", "Electricity price (GEL/kWh)", el_price),
                ("Input", "Old EER", eer_old),
                ("Input", "New EER", eer_new),
                ("Result", "Old cost (GEL/year)", cost_old),
                ("Result", "New cost (GEL/year)", cost_new),
                ("Result", "Savings (GEL/year)", savings),
            ],
            columns=["Type", "Key", "Value"],
        )
        st.download_button(
            "⬇️ Download CSV",
            data=df_export.to_csv(index=False).encode("utf-8"),
            file_name=f"chiller_eer_payback_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# Routing
# =========================================================
if tool in ["Heat Pump vs Boiler", "Scratch project (estimate demand)"]:
    default_mode = "Existing building (known demand)" if tool == "Heat Pump vs Boiler" else "Scratch project (estimate demand)"
    run_heating_ui(default_mode=default_mode)
else:
    run_chiller_ui()
