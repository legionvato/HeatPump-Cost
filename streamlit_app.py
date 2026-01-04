import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# =========================================================
# App Config
# =========================================================
st.set_page_config(
    page_title="Heat Pump vs Gas Boiler (V4)",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Heat Pump vs Gas Boiler"
APP_VER = "V4 (Application + COP bins + Climate + DHW + Emitters)"


# =========================================================
# Presets (feasibility-grade defaults; tune later)
# =========================================================
BUILDING_TYPES = ["Office", "Hotel", "Hospital"]
INSULATION_LEVELS = ["Poor / Old", "Standard", "Good / New", "High-performance"]
CLIMATES = ["Tbilisi", "Batumi", "Gudauri"]

APPLICATIONS = [
    "Space heating only",
    "DHW only (year-round)",
    "Space heating + DHW (year-round DHW)",
]

# Annual useful heat intensity (kWh_th/m²·year) baseline (Tbilisi-ish) – includes total heat (SH+DHW)
BASE_KWH_PER_M2_YEAR = {
    "Office": {"Poor / Old": 140, "Standard": 100, "Good / New": 70, "High-performance": 45},
    "Hotel": {"Poor / Old": 230, "Standard": 170, "Good / New": 130, "High-performance": 90},
    "Hospital": {"Poor / Old": 320, "Standard": 250, "Good / New": 200, "High-performance": 150},
}

# Climate multipliers for energy intensity
CLIMATE_INTENSITY_FACTOR = {"Tbilisi": 1.00, "Batumi": 0.90, "Gudauri": 1.25}

# Full-load hours presets for converting peak kW → annual kWh (Tbilisi-ish)
BASE_FLH = {
    "Office": {"Poor / Old": 1800, "Standard": 1600, "Good / New": 1400, "High-performance": 1200},
    "Hotel": {"Poor / Old": 2400, "Standard": 2200, "Good / New": 2000, "High-performance": 1800},
    "Hospital": {"Poor / Old": 2600, "Standard": 2400, "Good / New": 2200, "High-performance": 2000},
}
CLIMATE_FLH_FACTOR = {"Tbilisi": 1.00, "Batumi": 0.92, "Gudauri": 1.18}

# DHW share presets (fraction of total annual useful heat)
DHW_SHARE_PRESET = {"Office": 0.10, "Hotel": 0.30, "Hospital": 0.35}

# Heating regimes (proxy for required supply temp)
HEATING_REGIMES = {
    "80/60 °C (very high-temp radiators)": (80, 60),
    "70/50 °C (high-temp radiators/AHU coils)": (70, 50),
    "60/40 °C (medium radiators/FCU)": (60, 40),
    "50/40 °C (low-temp radiators/FCU)": (50, 40),
    "45/35 °C (underfloor/very low-temp)": (45, 35),
}

# Equipment mix is only for SPACE HEATING (DHW handled separately)
HIGH_TEMP_THRESHOLD_C = 65
BOOST_TARGET_C = 70

# COP bin temperatures
COP_TEMPS_3 = [-3, 2, 7]
COP_TEMPS_5 = [-3, 2, 7, 15, 30]

# Heating-season weights (space heating) by climate: weights for (-3, +2, +7)
SH_WEIGHTS_3 = {
    "Tbilisi": {-3: 0.25, 2: 0.45, 7: 0.30},
    "Batumi": {-3: 0.10, 2: 0.35, 7: 0.55},
    "Gudauri": {-3: 0.45, 2: 0.40, 7: 0.15},
}

# Year-round weights (for DHW year-round) by climate: weights for (-3, +2, +7, +15, +30)
DHW_WEIGHTS_5 = {
    "Tbilisi": {-3: 0.10, 2: 0.20, 7: 0.25, 15: 0.30, 30: 0.15},
    "Batumi": {-3: 0.05, 2: 0.15, 7: 0.25, 15: 0.30, 30: 0.25},
    "Gudauri": {-3: 0.20, 2: 0.25, 7: 0.25, 15: 0.20, 30: 0.10},
}


# =========================================================
# Helpers
# =========================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def weighted_avg(points: dict, weights: dict) -> float:
    """
    points: {tempC: value}, weights: {tempC: weight}, weight sum should be 1
    If some points missing, renormalize weights over available temps.
    """
    available = [t for t in weights.keys() if t in points and points[t] > 0]
    if not available:
        return 0.0
    wsum = sum(weights[t] for t in available)
    if wsum <= 0:
        return 0.0
    return sum(points[t] * (weights[t] / wsum) for t in available)


def derive_sh_high_temp_share(regime_supply_c: float, share_rad: float, share_fcu: float, share_ufh: float, share_ahu: float) -> float:
    """
    Space heating high-temp share fraction (0..1).
    Simple/defensible rule:
    - If regime supply >= HIGH_TEMP_THRESHOLD_C:
        Radiators + AHU coils counted as high-temp share.
      Else:
        no high-temp share (booster not needed for SH).
    """
    total = share_rad + share_fcu + share_ufh + share_ahu
    if total <= 0:
        return 0.0
    rad = share_rad / total
    ahu = share_ahu / total
    if regime_supply_c >= HIGH_TEMP_THRESHOLD_C:
        return clamp(rad + ahu, 0.0, 1.0)
    return 0.0


def dhw_is_high_temp(dhw_target_c: float) -> float:
    return 1.0 if dhw_target_c >= HIGH_TEMP_THRESHOLD_C else 0.0


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
    line("HP+Booster: booster upgrades high-temp share; base HP supplies low-temp heat + booster source heat.", size=10)

    c.showPage()
    c.save()
    return buf.getvalue()


# =========================================================
# Header
# =========================================================
st.title(f"{APP_TITLE} — {APP_VER}")
st.caption("V4 supports application selection (SH/DHW), climate bin COP averaging, scratch demand estimation, and automatic high-temp share derivation.")

st.info(
    "Core model logic:\n"
    "- Gas boiler cost = (annual useful heat / boiler efficiency) × gas price\n"
    "- HP+Booster: booster upgrades 'high-temp share' to 70°C.\n"
    "  Base HP must supply low-temp heat + booster source heat.\n"
    "  This avoids over-optimistic results."
)

# =========================================================
# Sidebar: global inputs
# =========================================================
with st.sidebar:
    st.header("Project Mode")
    mode = st.radio(
        "Choose mode",
        ["Existing building (known demand)", "Scratch project (estimate demand)"],
        index=1,
    )

    st.divider()
    st.header("Application")
    application = st.selectbox("What are you evaluating?", APPLICATIONS, index=2)

    st.divider()
    st.header("Climate")
    climate = st.selectbox("Select climate", CLIMATES, index=0)

    st.divider()
    st.header("Energy Prices")
    el_price_gel_per_kwh = st.number_input("Electricity price (GEL/kWh)", min_value=0.001, value=0.30, step=0.01, format="%.3f")
    gas_price_gel_per_m3 = st.number_input("Gas price (GEL/m³)", min_value=0.01, value=1.29, step=0.01)
    kwh_per_m3 = st.number_input("Gas energy content (kWh/m³)", min_value=5.0, max_value=15.0, value=10.0, step=0.1)

    st.divider()
    st.header("Gas Boiler (Baseline)")
    eta_boiler = st.number_input("Boiler seasonal efficiency (η)", min_value=0.50, max_value=1.00, value=0.93, step=0.01)

    st.divider()
    st.header("Heat Pump / Booster Presence")
    no_booster = st.checkbox("No booster installed", value=False)

    st.divider()
    st.header("COP input method")
    cop_method = st.radio(
        "How will you provide COPs?",
        ["Single seasonal COP", "From temperature points (bin-average)"],
        index=1,
    )

    if cop_method == "Single seasonal COP":
        cop_base_sh = st.number_input("Base HP COP (Space heating seasonal)", min_value=0.5, value=2.6, step=0.1)
        cop_base_dhw = st.number_input("Base HP COP (DHW year-round seasonal)", min_value=0.5, value=3.0, step=0.1)
        if no_booster:
            cop_boost_sh = 0.0
            cop_boost_dhw = 0.0
        else:
            cop_boost_sh = st.number_input("Booster COP (SH high-temp seasonal)", min_value=0.5, value=6.0, step=0.1)
            cop_boost_dhw = st.number_input("Booster COP (DHW high-temp seasonal)", min_value=0.5, value=6.3, step=0.1)

    else:
        # Choose 3-point or 5-point entry.
        # Clean rule:
        # - Space heating uses 3-point bins always
        # - DHW year-round uses 5-point bins
        st.caption("Space heating uses 3-point bins (−3/+2/+7). DHW year-round uses 5-point bins (−3/+2/+7/+15/+30).")

        with st.expander("Enter Base HP COP points", expanded=True):
            st.markdown("**Space Heating COP points** (same heating regime; ambient bins):")
            base_points_sh = {}
            for t in COP_TEMPS_3:
                base_points_sh[t] = st.number_input(f"Base COP at {t}°C (SH)", min_value=0.1, value=2.5 + (t+3)*0.03, step=0.05)

            st.markdown("**DHW Year-round COP points** (if DHW included):")
            base_points_dhw = {}
            for t in COP_TEMPS_5:
                base_points_dhw[t] = st.number_input(f"Base COP at {t}°C (DHW)", min_value=0.1, value=2.7 + (t+3)*0.02, step=0.05)

        if no_booster:
            boost_points_sh = {}
            boost_points_dhw = {}
            cop_boost_sh = 0.0
            cop_boost_dhw = 0.0
        else:
            with st.expander("Enter Booster COP points (optional, recommended for DHW)", expanded=False):
                st.markdown("**Booster COP points** (boost to 70°C):")
                boost_points_sh = {}
                for t in COP_TEMPS_3:
                    boost_points_sh[t] = st.number_input(f"Booster COP at {t}°C (SH)", min_value=0.1, value=5.8 + (t+3)*0.02, step=0.05)

                boost_points_dhw = {}
                for t in COP_TEMPS_5:
                    boost_points_dhw[t] = st.number_input(f"Booster COP at {t}°C (DHW)", min_value=0.1, value=6.2 + (t+3)*0.015, step=0.05)

        # Compute seasonal COPs
        cop_base_sh = weighted_avg(base_points_sh, SH_WEIGHTS_3[climate])
        cop_base_dhw = weighted_avg(base_points_dhw, DHW_WEIGHTS_5[climate]) if application != "Space heating only" else 0.0
        cop_boost_sh = weighted_avg(boost_points_sh, SH_WEIGHTS_3[climate]) if (not no_booster) else 0.0
        cop_boost_dhw = weighted_avg(boost_points_dhw, DHW_WEIGHTS_5[climate]) if (not no_booster and application != "Space heating only") else 0.0

        st.success(f"Derived COPs — Base: SH {cop_base_sh:.2f} | DHW {cop_base_dhw:.2f} ; Booster: SH {cop_boost_sh:.2f} | DHW {cop_boost_dhw:.2f}")

    st.divider()
    st.header("Optional: CAPEX / Payback")
    enable_payback = st.checkbox("Calculate payback", value=False)
    capex_hp_booster_gel = st.number_input("CAPEX: HP (+Booster) total (GEL)", min_value=0.0, value=0.0, step=10_000.0)
    capex_boiler_gel = st.number_input("CAPEX: Boiler / baseline (GEL)", min_value=0.0, value=0.0, step=10_000.0)


# =========================================================
# Demand inputs + splits + temps
# =========================================================
scratch_meta = {}

# ---- Shares for SH vs DHW based on application
if application == "Space heating only":
    sh_share_pct = 100
    dhw_share_pct = 0
elif application == "DHW only (year-round)":
    sh_share_pct = 0
    dhw_share_pct = 100
else:
    # SH + DHW, will be set from building type presets or manual override
    sh_share_pct = None
    dhw_share_pct = None

# ---- Mode-specific demand definition
if mode == "Existing building (known demand)":
    st.subheader("Existing building inputs")

    q_annual_kwh_th = st.number_input(
        "Annual useful heat demand (kWh_th/year)",
        min_value=1.0,
        value=2_000_000.0,
        step=50_000.0,
    )

    if application == "Space heating + DHW (year-round DHW)":
        col1, col2 = st.columns(2)
        with col1:
            sh_share_pct = st.slider("Space heating share of annual heat (%)", 0, 100, 70, 1)
        with col2:
            dhw_share_pct = 100 - sh_share_pct
            st.metric("DHW share (%)", f"{dhw_share_pct}%")

        scratch_meta["sh_share_pct"] = sh_share_pct
        scratch_meta["dhw_share_pct"] = dhw_share_pct

    dhw_target_c = st.selectbox("DHW target temperature (°C)", [55, 60, 65, 70], index=1) if dhw_share_pct > 0 else 55
    scratch_meta["dhw_target_c"] = dhw_target_c

    # Space heating emitters + regime only relevant if SH exists
    if sh_share_pct > 0:
        st.divider()
        st.subheader("Space heating emitters & regime (for high-temp share)")

        cA, cB = st.columns(2)
        with cA:
            regime_name = st.selectbox("Heating regime (supply/return)", list(HEATING_REGIMES.keys()), index=2)
            regime_supply_c, regime_return_c = HEATING_REGIMES[regime_name]

            share_rad = st.number_input("Radiators (%)", 0, 100, 40, 5)
            share_fcu = st.number_input("FCU (%)", 0, 100, 40, 5)
            share_ufh = st.number_input("Underfloor (%)", 0, 100, 10, 5)
            share_ahu = st.number_input("AHU coils (%)", 0, 100, 10, 5)

        with cB:
            auto_boost = st.checkbox("Auto-calculate high-temp share", value=True)
            if not auto_boost:
                sh_high_frac = st.slider("Space heating high-temp fraction (%)", 0, 100, 40, 1) / 100.0
            else:
                sh_high_frac = derive_sh_high_temp_share(regime_supply_c, share_rad, share_fcu, share_ufh, share_ahu)
                st.success(f"Derived SH high-temp fraction: {sh_high_frac*100:.0f}%")

        scratch_meta.update({
            "regime": regime_name,
            "regime_supply_c": regime_supply_c,
            "regime_return_c": regime_return_c,
            "equipment_mix_sh": {"Radiators": share_rad, "FCU": share_fcu, "Underfloor": share_ufh, "AHU": share_ahu},
            "sh_high_frac": sh_high_frac,
            "auto_boost": auto_boost,
        })
    else:
        sh_high_frac = 0.0
        scratch_meta["sh_high_frac"] = sh_high_frac

else:
    st.subheader("Scratch project inputs (early-stage)")

    colA, colB, colC = st.columns([1.1, 1.0, 1.0])

    with colA:
        building_type = st.selectbox("Building type", BUILDING_TYPES, index=1)
        insulation = st.selectbox("Insulation level", INSULATION_LEVELS, index=1)

        demand_method = st.radio(
            "Demand input method",
            [
                "From area (m²) + benchmarks",
                "From peak heat load (kW) + full-load hours (FLH)",
                "Direct annual useful heat demand (kWh_th/year)",
            ],
            index=0,
        )

        derived_kwh_per_m2 = None
        derived_flh = None

        if demand_method == "From area (m²) + benchmarks":
            area_m2 = st.number_input("Heated area (m²)", min_value=1.0, value=12000.0, step=100.0)
            base_intensity = float(BASE_KWH_PER_M2_YEAR[building_type][insulation])
            intensity = base_intensity * float(CLIMATE_INTENSITY_FACTOR[climate])
            derived_kwh_per_m2 = intensity
            q_annual_est = area_m2 * intensity
            st.caption(f"Benchmark intensity: {intensity:.0f} kWh/m²·year (incl. climate factor)")

        elif demand_method == "From peak heat load (kW) + full-load hours (FLH)":
            p_peak_kw = st.number_input("Peak heating load (kW)", min_value=1.0, value=500.0, step=10.0)
            base_flh = float(BASE_FLH[building_type][insulation])
            flh = base_flh * float(CLIMATE_FLH_FACTOR[climate])
            derived_flh = flh
            override = st.checkbox("Override FLH", value=False)
            flh_used = st.number_input("Full-load hours (h/year)", min_value=200.0, value=float(round(flh)), step=100.0) if override else flh
            if not override:
                st.caption(f"FLH preset: {flh_used:.0f} h/year (incl. climate factor)")
            q_annual_est = p_peak_kw * flh_used

        else:
            q_annual_est = st.number_input("Annual useful heat demand (kWh_th/year) — provided", min_value=1.0, value=2_000_000.0, step=50_000.0)

        q_annual_kwh_th = st.number_input(
            "Annual useful heat demand (kWh_th/year) — used in calculation",
            min_value=1.0,
            value=float(q_annual_est),
            step=50_000.0,
        )

        scratch_meta.update({
            "building_type": building_type,
            "insulation": insulation,
            "demand_method": demand_method,
            "kwh_per_m2_year": derived_kwh_per_m2,
            "flh": derived_flh,
        })

    with colB:
        st.markdown("### DHW")
        if application == "Space heating + DHW (year-round DHW)":
            preset_dhw_pct = int(round(DHW_SHARE_PRESET[building_type] * 100))
            use_override = st.checkbox("Override DHW share", value=False)
            dhw_share_pct = st.slider("DHW share (%)", 0, 100, preset_dhw_pct, 1) if use_override else preset_dhw_pct
            sh_share_pct = 100 - dhw_share_pct
            st.metric("Space heating share (%)", f"{sh_share_pct}%")
        elif application == "DHW only (year-round)":
            dhw_share_pct = 100
            sh_share_pct = 0
            st.info("Application is DHW only → DHW share = 100%")
        else:
            dhw_share_pct = 0
            sh_share_pct = 100
            st.info("Application is Space heating only → DHW share = 0%")

        dhw_target_c = st.selectbox("DHW target temperature (°C)", [55, 60, 65, 70], index=1) if dhw_share_pct > 0 else 55

        scratch_meta.update({
            "sh_share_pct": sh_share_pct,
            "dhw_share_pct": dhw_share_pct,
            "dhw_target_c": dhw_target_c,
        })

    with colC:
        st.markdown("### Space heating emitters & regime")
        if sh_share_pct > 0:
            regime_name = st.selectbox("Heating regime (supply/return)", list(HEATING_REGIMES.keys()), index=2)
            regime_supply_c, regime_return_c = HEATING_REGIMES[regime_name]

            share_rad = st.number_input("Radiators (%)", 0, 100, 40, 5, key="rad_s")
            share_fcu = st.number_input("FCU (%)", 0, 100, 40, 5, key="fcu_s")
            share_ufh = st.number_input("Underfloor (%)", 0, 100, 10, 5, key="ufh_s")
            share_ahu = st.number_input("AHU coils (%)", 0, 100, 10, 5, key="ahu_s")

            auto_boost = st.checkbox("Auto-calculate high-temp share", value=True, key="auto_boost_s")
            if not auto_boost:
                sh_high_frac = st.slider("Space heating high-temp fraction (%)", 0, 100, 40, 1, key="sh_high_manual") / 100.0
            else:
                sh_high_frac = derive_sh_high_temp_share(regime_supply_c, share_rad, share_fcu, share_ufh, share_ahu)
                st.success(f"Derived SH high-temp fraction: {sh_high_frac*100:.0f}%")

            scratch_meta.update({
                "regime": regime_name,
                "regime_supply_c": regime_supply_c,
                "regime_return_c": regime_return_c,
                "equipment_mix_sh": {"Radiators": share_rad, "FCU": share_fcu, "Underfloor": share_ufh, "AHU": share_ahu},
                "sh_high_frac": sh_high_frac,
                "auto_boost": auto_boost,
            })
        else:
            sh_high_frac = 0.0
            scratch_meta["sh_high_frac"] = sh_high_frac

# Ensure shares are set
if sh_share_pct is None:
    sh_share_pct = 70
if dhw_share_pct is None:
    dhw_share_pct = 30

# DHW high-temp fraction (0 or 1) based on target
dhw_high_frac = dhw_is_high_temp(float(dhw_target_c)) if dhw_share_pct > 0 else 0.0

# No booster forces high-temp fractions to 0
if no_booster:
    if sh_high_frac > 0 or dhw_high_frac > 0:
        st.warning("No booster is selected, but high-temp demand exists (SH regime/emitters or DHW target). Booster share will be forced to 0%.")
    sh_high_frac = 0.0
    dhw_high_frac = 0.0


# =========================================================
# Split energy into SH vs DHW and high-temp vs low-temp
# =========================================================
Q_total = float(q_annual_kwh_th)
Q_sh = Q_total * (sh_share_pct / 100.0)
Q_dhw = Q_total * (dhw_share_pct / 100.0)

Q_sh_high = Q_sh * sh_high_frac
Q_sh_low = Q_sh * (1.0 - sh_high_frac)

Q_dhw_high = Q_dhw * dhw_high_frac
Q_dhw_low = Q_dhw * (1.0 - dhw_high_frac)

# Booster share (energy-weighted)
boost_share_pct = int(round(0 if Q_total <= 0 else ((Q_sh_high + Q_dhw_high) / Q_total) * 100.0))

scratch_meta.update({
    "application": application,
    "climate": climate,
    "boost_share_pct": boost_share_pct,
    "dhw_high_frac": dhw_high_frac,
})

# =========================================================
# Electricity calculations with defensible energy balance, split by SH season vs DHW year-round
# For each component (SH high, DHW high):
#   E_boost = Q_high / COP_boost
#   Q_source = Q_high - E_boost
# Base HP must provide:
#   Q_base = Q_low + Q_source
# and electricity:
#   E_base = Q_base / COP_base
# =========================================================
def hp_chain(Q_low: float, Q_high: float, cop_base: float, cop_boost: float) -> dict:
    """
    Returns dict with E_base, E_boost, Q_source, Q_base_out.
    If Q_high = 0 => no booster used in this chain.
    """
    if Q_low < 0: Q_low = 0.0
    if Q_high < 0: Q_high = 0.0
    if cop_base <= 0: cop_base = 1e-9

    if Q_high <= 0 or cop_boost <= 0:
        # No boosting
        Q_source = 0.0
        E_boost = 0.0
    else:
        E_boost = Q_high / cop_boost
        Q_source = Q_high - E_boost

    Q_base_out = Q_low + Q_source
    E_base = Q_base_out / cop_base

    return {"E_base": E_base, "E_boost": E_boost, "Q_source": Q_source, "Q_base_out": Q_base_out}

# Determine which chains exist based on application:
# - Space heating energy occurs in heating season → use SH COPs
# - DHW energy occurs year-round → use DHW COPs
chain_sh = hp_chain(Q_sh_low, Q_sh_high, cop_base_sh if Q_sh > 0 else 0.0, cop_boost_sh if Q_sh > 0 else 0.0) if Q_sh > 0 else {"E_base": 0.0, "E_boost": 0.0, "Q_source": 0.0, "Q_base_out": 0.0}
chain_dhw = hp_chain(Q_dhw_low, Q_dhw_high, cop_base_dhw if Q_dhw > 0 else 0.0, cop_boost_dhw if Q_dhw > 0 else 0.0) if Q_dhw > 0 else {"E_base": 0.0, "E_boost": 0.0, "Q_source": 0.0, "Q_base_out": 0.0}

E_total_hp = chain_sh["E_base"] + chain_sh["E_boost"] + chain_dhw["E_base"] + chain_dhw["E_boost"]
cost_hp_gel = E_total_hp * el_price_gel_per_kwh

# Gas baseline (single boiler for total useful heat)
gas_input_kwh = Q_total / eta_boiler if eta_boiler > 0 else 0.0
gas_volume_m3 = gas_input_kwh / kwh_per_m3 if kwh_per_m3 > 0 else 0.0
cost_gas_gel = gas_volume_m3 * gas_price_gel_per_m3

annual_savings_gel = cost_gas_gel - cost_hp_gel
cop_effective = (Q_total / E_total_hp) if E_total_hp > 0 else 0.0


# =========================================================
# KPI Row
# =========================================================
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Annual useful heat", f"{Q_total:,.0f} kWh_th")
k2.metric("High-temp share (boosted)", f"{boost_share_pct}%")
k3.metric("Electricity price", f"{el_price_gel_per_kwh:.3f} GEL/kWh")
k4.metric("Gas price", f"{gas_price_gel_per_m3:.2f} GEL/m³")


# =========================================================
# Tabs
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

    st.subheader("Cost comparison (readable)")
    df_cost = pd.DataFrame(
        {"System": ["Gas Boiler", "Heat Pump System"], "Annual cost (GEL)": [cost_gas_gel, cost_hp_gel]}
    )

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
        p1.metric("Extra CAPEX (HP system − Boiler)", f"{delta_capex:,.0f} GEL")

        if annual_savings_gel > 0 and delta_capex > 0:
            payback_years = delta_capex / annual_savings_gel
            p2.metric("Payback (years)", f"{payback_years:.2f}")
            p3.metric("Payback (months)", f"{payback_years * 12:.1f}")
        else:
            p2.metric("Payback (years)", "N/A")
            p3.metric("Payback (months)", "N/A")
            st.caption("Payback requires: annual savings > 0 and extra CAPEX > 0.")

with tab2:
    st.subheader("Assumptions & intermediate results (transparent)")

    st.write("### Application split")
    st.write(f"- Application: **{application}**")
    st.write(f"- Space heating share: **{sh_share_pct}%** → Q_sh = **{Q_sh:,.0f} kWh_th**")
    st.write(f"- DHW share: **{dhw_share_pct}%** → Q_dhw = **{Q_dhw:,.0f} kWh_th**")

    st.write("### High-temp shares (booster)")
    st.write(f"- SH high-temp fraction: **{sh_high_frac*100:.0f}%** → Q_sh_high = **{Q_sh_high:,.0f}**")
    st.write(f"- DHW high-temp fraction: **{dhw_high_frac*100:.0f}%** (target {dhw_target_c}°C) → Q_dhw_high = **{Q_dhw_high:,.0f}**")
    st.write(f"- Total boosted share (energy-weighted): **{boost_share_pct}%**")

    st.divider()
    st.write("### COPs used")
    st.write(f"- Base COP (SH seasonal): **{cop_base_sh:.2f}**")
    st.write(f"- Base COP (DHW year-round): **{cop_base_dhw:.2f}**")
    if no_booster:
        st.write("- Booster: **disabled**")
    else:
        st.write(f"- Booster COP (SH seasonal): **{cop_boost_sh:.2f}**")
        st.write(f"- Booster COP (DHW year-round): **{cop_boost_dhw:.2f}**")

    st.divider()
    st.write("### HP energy balance by chain")
    st.write("**Space Heating chain**")
    st.write(f"- Booster electricity: **{chain_sh['E_boost']:,.0f} kWh_el**")
    st.write(f"- Booster source heat from base: **{chain_sh['Q_source']:,.0f} kWh_th**")
    st.write(f"- Base HP delivered heat: **{chain_sh['Q_base_out']:,.0f} kWh_th**")
    st.write(f"- Base electricity: **{chain_sh['E_base']:,.0f} kWh_el**")

    st.write("**DHW chain**")
    st.write(f"- Booster electricity: **{chain_dhw['E_boost']:,.0f} kWh_el**")
    st.write(f"- Booster source heat from base: **{chain_dhw['Q_source']:,.0f} kWh_th**")
    st.write(f"- Base HP delivered heat: **{chain_dhw['Q_base_out']:,.0f} kWh_th**")
    st.write(f"- Base electricity: **{chain_dhw['E_base']:,.0f} kWh_el**")

    st.divider()
    st.write("### Totals")
    st.write(f"- Total HP electricity: **{E_total_hp:,.0f} kWh_el/year**")
    st.write(f"- Effective system COP (useful/elec): **{cop_effective:.2f}**")
    st.write(f"- Gas volume: **{gas_volume_m3:,.0f} m³/year**")

with tab3:
    meta_mode = "Existing building" if mode.startswith("Existing") else "Scratch project"

    pdf_payload = {
        "meta": {
            "title": f"{APP_TITLE} — {APP_VER}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "mode": f"{meta_mode} | {application} | {climate}",
        },
        "inputs": {
            "Annual useful heat demand (kWh_th/year)": f"{Q_total:,.0f}",
            "Space heating share (%)": f"{sh_share_pct}%",
            "DHW share (%)": f"{dhw_share_pct}%",
            "DHW target (°C)": f"{dhw_target_c}",
            "Boosted (high-temp) share (%)": f"{boost_share_pct}%",
            "Electricity price (GEL/kWh)": f"{el_price_gel_per_kwh:.3f}",
            "Gas price (GEL/m³)": f"{gas_price_gel_per_m3:.2f}",
            "Gas energy content (kWh/m³)": f"{kwh_per_m3:.2f}",
            "Boiler efficiency (η)": f"{eta_boiler:.2f}",
            "Base COP (SH seasonal)": f"{cop_base_sh:.2f}",
            "Base COP (DHW year-round)": f"{cop_base_dhw:.2f}",
            "Booster installed": "No" if no_booster else "Yes",
        },
        "results": {
            "Gas boiler annual cost (GEL/year)": f"{cost_gas_gel:,.0f}",
            "HP system annual cost (GEL/year)": f"{cost_hp_gel:,.0f}",
            "Annual savings (GEL/year)": f"{annual_savings_gel:,.0f}",
            "Effective system COP": f"{cop_effective:.2f}",
            "Total HP electricity (kWh_el/year)": f"{E_total_hp:,.0f}",
            "Gas volume (m³/year)": f"{gas_volume_m3:,.0f}",
        },
    }

    if not no_booster:
        pdf_payload["inputs"]["Booster COP (SH seasonal)"] = f"{cop_boost_sh:.2f}"
        pdf_payload["inputs"]["Booster COP (DHW year-round)"] = f"{cop_boost_dhw:.2f}"

    pdf_bytes = build_pdf_report(pdf_payload)

    st.download_button(
        label="📄 Download PDF report",
        data=pdf_bytes,
        file_name=f"hp_vs_boiler_v4_{meta_mode.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
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
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name=f"hp_vs_boiler_v4_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
