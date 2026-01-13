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
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors


# =========================================================
# App Config
# =========================================================
st.set_page_config(
    page_title="Treimax Energy Tools (HP vs Boiler + Chiller)",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Treimax Energy Tools"
APP_VER = "V11.2 (Richer PDF presets + Charts — NO Logo)"


# =========================================================
# Helpers
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


def fig_to_imagereader(fig) -> ImageReader:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return ImageReader(buf)


def barh_chart(labels, values, title, xlabel, value_fmt="{:,.0f}"):
    df = pd.DataFrame({"Label": labels, "Value": values})
    fig, ax = plt.subplots()
    ax.barh(df["Label"], df["Value"])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, v in enumerate(df["Value"]):
        ax.text(v, i, value_fmt.format(v), va="center", ha="left")
    return fig


def stacked_bar_cost_hp(gas_cost, hp_cost_base, hp_cost_boost, title="Annual cost comparison (HP breakdown)"):
    fig, ax = plt.subplots()
    labels = ["Gas Boiler", "HP System"]
    ax.bar(labels, [gas_cost, hp_cost_base], label="Base HP")
    ax.bar(labels, [0, hp_cost_boost], bottom=[0, hp_cost_base], label="Booster")
    ax.set_title(title)
    ax.set_ylabel("Annual cost (GEL)")
    ax.legend()
    return fig


def pie_chart(labels, values, title):
    fig, ax = plt.subplots()
    vals = [max(0.0, float(v)) for v in values]
    if sum(vals) <= 0:
        vals = [1.0 for _ in vals]
    ax.pie(vals, labels=labels, autopct="%1.0f%%")
    ax.set_title(title)
    return fig


def harmonic_mean_weighted_by_capacity(cap_list, eff_list):
    """
    Weighted harmonic mean by capacity:
      eff_weighted = sum(Cap) / sum(Cap/eff)
    """
    pairs = [(c, e) for c, e in zip(cap_list, eff_list) if c > 0 and e > 0]
    if not pairs:
        return 0.0
    num = sum(c for c, _ in pairs)
    den = sum(c / e for c, e in pairs)
    return num / den if den > 0 else 0.0


# =========================================================
# Rich PDF Builder (Treimax Georgia, no logo)
# FIXED: nonlocal y binding + KPI overlap + indentation
# =========================================================
def build_rich_pdf_report(
    report_title: str,
    subtitle: str,
    sections: list[dict],
    footer_note: str = "",
) -> bytes:
    """
    sections: list of dicts like:
      {"type":"kpi", "title":"Executive Summary", "items":[("Annual savings","12,000 GEL"), ...]}
      {"type":"table", "title":"Inputs", "columns":["Key","Value"], "rows":[["Gas price","1.29"], ...]}
      {"type":"text", "title":"Notes", "lines":[...]}
      {"type":"image", "title":"Chart title", "image": ImageReader, "w_cm": 16, "h_cm": 7}
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    margin_l = 1.7 * cm
    margin_r = 1.7 * cm
    margin_t = 1.6 * cm
    margin_b = 1.4 * cm
    usable_w = W - margin_l - margin_r

    page_num = 0

    # IMPORTANT FIX: define y in outer scope so nonlocal works
    y = H - margin_t

    def draw_footer():
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.grey)
        left = margin_l
        right = W - margin_r
        c.drawString(left, margin_b - 0.6 * cm, f"{APP_TITLE} — {APP_VER}")
        c.drawRightString(right, margin_b - 0.6 * cm, f"Page {page_num}")
        if footer_note:
            c.drawString(left, margin_b - 1.0 * cm, footer_note)
        c.setFillColor(colors.black)

    def draw_header():
        nonlocal y
        # Brand header (text-only)
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin_l, y, "TREIMAX GEORGIA")
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.grey)
        c.drawString(margin_l, y - 0.45 * cm, "Cooling, Heating and Ventilation Systems")
        c.setFillColor(colors.black)

        # Title block
        y -= 1.05 * cm
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin_l, y, report_title)

        y -= 0.55 * cm
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.grey)
        c.drawString(margin_l, y, subtitle)
        c.setFillColor(colors.black)

        y -= 0.35 * cm
        c.setStrokeColor(colors.lightgrey)
        c.line(margin_l, y, W - margin_r, y)
        c.setStrokeColor(colors.black)
        y -= 0.6 * cm

    def new_page():
        nonlocal page_num, y
        if page_num > 0:
            draw_footer()
            c.showPage()
        page_num += 1
        y = H - margin_t
        draw_header()

    def ensure_space(h_needed):
        nonlocal y
        if y - h_needed < margin_b:
            new_page()

    def draw_section_title(title):
        nonlocal y
        ensure_space(1.1 * cm)
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(colors.black)
        c.drawString(margin_l, y, title)
        y -= 0.35 * cm
        c.setStrokeColor(colors.lightgrey)
        c.line(margin_l, y, W - margin_r, y)
        c.setStrokeColor(colors.black)
        y -= 0.45 * cm

    # FIXED KPI: dynamic height (no overlap)
    def draw_kpi(items):
        nonlocal y
        import math

        cols = 2
        n = len(items)
        rows = max(1, math.ceil(n / cols))

        row_h = 1.05 * cm
        pad_top = 0.25 * cm
        pad_bottom = 0.25 * cm
        box_h = rows * row_h + pad_top + pad_bottom

        ensure_space(box_h + 0.35 * cm)

        x0 = margin_l
        y_top = y
        y0 = y_top - box_h

        c.setStrokeColor(colors.lightgrey)
        c.setFillColor(colors.whitesmoke)
        c.rect(x0, y0, usable_w, box_h, fill=1, stroke=1)

        col_w = usable_w / cols

        for idx, (lab, val) in enumerate(items):
            r = idx // cols
            col = idx % cols
            xx = x0 + col * col_w + 0.35 * cm

            row_top = y_top - pad_top - r * row_h
            label_y = row_top - 0.35 * cm
            value_y = row_top - 0.80 * cm

            c.setFont("Helvetica", 8)
            c.setFillColor(colors.grey)
            c.drawString(xx, label_y, str(lab))

            c.setFont("Helvetica-Bold", 11)
            c.setFillColor(colors.black)
            c.drawString(xx, value_y, str(val))

        c.setFillColor(colors.black)
        c.setStrokeColor(colors.black)
        y = y0 - 0.55 * cm

    def draw_table(columns, rows, col_widths=None):
        nonlocal y
        if not columns:
            return
        if col_widths is None:
            col_widths = [usable_w / len(columns)] * len(columns)

        row_h = 0.55 * cm
        header_h = 0.65 * cm
        table_h = header_h + row_h * max(1, len(rows))
        ensure_space(table_h + 0.4 * cm)

        x = margin_l
        y_top = y

        c.setFillColor(colors.lightgrey)
        c.rect(x, y_top - header_h, usable_w, header_h, fill=1, stroke=0)
        c.setFillColor(colors.black)

        c.setFont("Helvetica-Bold", 9)
        xx = x
        for i, col in enumerate(columns):
            c.drawString(xx + 0.15 * cm, y_top - 0.45 * cm, str(col))
            xx += col_widths[i]

        c.setStrokeColor(colors.lightgrey)
        y_cursor = y_top - header_h
        c.line(x, y_cursor, x + usable_w, y_cursor)

        c.setFont("Helvetica", 9)
        for r in rows:
            y_cursor -= row_h
            xx = x
            for i, cell in enumerate(r):
                c.drawString(xx + 0.15 * cm, y_cursor + 0.18 * cm, str(cell))
                xx += col_widths[i]
            c.line(x, y_cursor, x + usable_w, y_cursor)

        xx = x
        c.line(xx, y_top, xx, y_cursor)
        for w in col_widths:
            xx += w
            c.line(xx, y_top, xx, y_cursor)

        c.setStrokeColor(colors.lightgrey)
        c.rect(x, y_cursor, usable_w, (y_top - y_cursor), fill=0, stroke=1)
        c.setStrokeColor(colors.black)

        y = y_cursor - 0.55 * cm

    def draw_text(lines):
        nonlocal y
        c.setFont("Helvetica", 9.5)
        for line in lines:
            ensure_space(0.55 * cm)
            safe = (line or "").replace("CO₂", "CO2").replace("tCO₂", "tCO2")
            c.drawString(margin_l, y, safe)
            y -= 0.48 * cm
        y -= 0.2 * cm

    def draw_image(img: ImageReader, w_cm=16, h_cm=7):
        nonlocal y
        w = w_cm * cm
        h = h_cm * cm
        ensure_space(h + 0.5 * cm)
        x = margin_l
        c.setStrokeColor(colors.lightgrey)
        c.rect(x, y - h, min(w, usable_w), h, fill=0, stroke=1)
        c.drawImage(img, x, y - h, width=min(w, usable_w), height=h, preserveAspectRatio=True, anchor="sw")
        c.setStrokeColor(colors.black)
        y -= (h + 0.55 * cm)

    # Start first page
    new_page()

    for sec in sections:
        stype = sec.get("type")
        title = sec.get("title", "")
        if title:
            draw_section_title(title)

        if stype == "kpi":
            draw_kpi(sec.get("items", []))
        elif stype == "table":
            draw_table(sec.get("columns", []), sec.get("rows", []), sec.get("col_widths"))
        elif stype == "text":
            draw_text(sec.get("lines", []))
        elif stype == "image":
            draw_image(sec.get("image"), sec.get("w_cm", 16), sec.get("h_cm", 7))

    draw_footer()
    c.showPage()
    c.save()
    return buf.getvalue()


# =========================================================
# Save/Load JSON (no DB)
# =========================================================
PROJECT_KEYS = [
    "project_name",
    "active_tool",

    # HEATING
    "heat_view_mode",  # Client-friendly vs Expert
    "heat_mode",
    "heat_demand_input_method",
    "heat_gas_m3_year",
    "heat_gas_includes_dhw",
    "heat_application",
    "heat_climate",
    "heat_el_price",
    "heat_gas_price",
    "heat_kwh_per_m3",
    "heat_boiler_preset",
    "heat_eta_override",
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
    "heat_regime_name",
    "heat_mixed_systems",
    "heat_sh_high_frac_pct",

    # CHILLER
    "ch_el_price",
    "ch_demand_method",
    "ch_q_cool_annual",
    "ch_peak_cool_kw",
    "ch_cflh_used",
    "ch_months",
    "ch_hours_per_day",
    "ch_load_factor",
    "ch_label_a",
    "ch_label_b",
    "ch_enable_payback",
    "ch_capex_a",
    "ch_capex_b",
]

for setup in ["a", "b"]:
    for i in range(1, 5):
        PROJECT_KEYS += [
            f"ch_{setup}_use_{i}",
            f"ch_{setup}_qty_{i}",
            f"ch_{setup}_kw_{i}",
            f"ch_{setup}_metric_{i}",
            f"ch_{setup}_eff_{i}",
        ]


def collect_project_state() -> dict:
    payload = {"_meta": {"app": APP_TITLE, "ver": APP_VER, "saved_at": datetime.now().isoformat()}}
    for k in PROJECT_KEYS:
        if k in st.session_state:
            payload[k] = st.session_state[k]
    return payload


def apply_project_state(payload: dict) -> None:
    for k in PROJECT_KEYS:
        if k in payload:
            st.session_state[k] = payload[k]


# =========================================================
# Header + Save/Load UI
# =========================================================
st.title(f"{APP_TITLE} — {APP_VER}")

with st.sidebar:
    st.header("Project")
    st.text_input("Project name", value=st.session_state.get("project_name", ""), key="project_name")

    st.divider()
    st.header("Project Save/Load (JSON)")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("⬇️ Save project"):
            st.session_state["_download_project_json"] = json.dumps(
                collect_project_state(), ensure_ascii=False, indent=2
            )

    with c2:
        uploaded = st.file_uploader("⬆️ Load project", type=["json"], label_visibility="collapsed")
        if uploaded is not None:
            try:
                payload = json.load(uploaded)
                apply_project_state(payload)
                st.success("Loaded project. Applying…")
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
    tool = st.radio(
        "Tool",
        ["Heat Pump vs Boiler", "Chiller comparison (payback)"],
        key="active_tool",
    )


# =========================================================
# HEATING MODULE CONSTANTS
# =========================================================
HP_MAX_SUPPLY_C = 50
GAS_CO2_FACTOR_KG_PER_KWH = 0.202  # kgCO2 per kWh_gas (combustion only)

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

HEATING_REGIMES = {
    "45/35 C (low-temp emitters)": (45, 35),
    "50/40 C (typical)": (50, 40),
    "55/45 C (higher-temp loop)": (55, 45),
    "60/40 C (mixed / legacy)": (60, 40),
    "70/50 C (radiators)": (70, 50),
    "80/60 C (legacy/very high-temp loop)": (80, 60),
}

SH_WEIGHTS_3 = {
    "Tbilisi": {-3: 0.25, 2: 0.45, 7: 0.30},
    "Batumi": {-3: 0.10, 2: 0.35, 7: 0.55},
    "Gudauri": {-3: 0.45, 2: 0.40, 7: 0.15},
}

BOILER_PRESETS = {
    "Modern condensing boiler (typical)": 0.95,
    "Average boiler (common)": 0.90,
    "Older boiler (likely)": 0.85,
    "Very old / poor (conservative)": 0.75,
}


# =========================================================
# HEATING MODULE
# =========================================================
def run_heating():
    st.subheader("Heat Pump vs Gas Boiler")

    with st.expander("📘 Glossary (click to expand)", expanded=False):
        st.markdown(
            """
**FLH (Full Load Hours):** Equivalent hours per year at peak load that deliver the same annual energy.  
**COP (Coefficient of Performance):** Heat output / electricity input (higher is better).  
**Seasonal COP:** Annual average COP used for yearly energy/cost estimates.  
**SCOP:** Seasonal COP from manufacturer datasheet (test standard).  
**DHW (Domestic Hot Water):** Hot water used for taps/showers (often year-round in hotels/hospitals).  
**SH (Space Heating):** Heating used for indoor temperature comfort.  
**Boosted share (>50°C):** Portion of annual heat that needs supply temperature above 50°C (requires booster if base HP is capped).  
**Booster:** High-temperature stage used only for loads needing >50°C supply (assumed constant COP in this tool).  
**η (Boiler efficiency):** Useful heat delivered / gas energy input (seasonal average).
"""
        )

    st.info(
        "Assumptions:\n"
        f"- Base heat pump max supply temperature = {HP_MAX_SUPPLY_C}°C\n"
        "- Loads requiring >50°C are covered by a booster only if installed\n"
        "- Gas baseline uses seasonal boiler efficiency and gas price\n"
        "- CO2 shown is GAS BOILER CO2 only (combustion only, baseline)"
    )

    # Sidebar inputs
    with st.sidebar:
        st.header("Heating Inputs")

        st.radio(
            "View",
            ["Client-friendly (simple)", "Expert (advanced)"],
            index=0,
            key="heat_view_mode",
        )
        client_view = st.session_state.get("heat_view_mode", "Client-friendly (simple)").startswith("Client")

        st.selectbox(
            "Project mode",
            ["Existing building (comparison)", "Scratch project (estimate demand)"],
            index=0,
            key="heat_mode",
        )

        st.selectbox("Climate", CLIMATES, index=0, key="heat_climate")
        st.selectbox("Application", APPLICATIONS, index=0, key="heat_application")

        st.divider()
        st.subheader("Prices")
        st.number_input(
            "Electricity price (GEL/kWh)",
            min_value=0.001, value=0.30, step=0.01, format="%.3f",
            key="heat_el_price"
        )
        st.number_input("Gas price (GEL/m³)", min_value=0.01, value=1.29, step=0.01, key="heat_gas_price")

        if not client_view:
            with st.expander("Advanced: Gas energy content", expanded=False):
                st.number_input(
                    "Gas energy content (kWh/m³)",
                    min_value=5.0, max_value=15.0, value=10.0, step=0.1,
                    key="heat_kwh_per_m3"
                )
        else:
            if "heat_kwh_per_m3" not in st.session_state:
                st.session_state["heat_kwh_per_m3"] = 10.0

        st.divider()
        st.subheader("Boiler baseline")

        st.selectbox(
            "Boiler efficiency preset (η)",
            list(BOILER_PRESETS.keys()),
            index=0,
            key="heat_boiler_preset",
        )
        st.checkbox("Override boiler efficiency (η)", value=False, key="heat_eta_override")

        preset_eta = float(BOILER_PRESETS.get(st.session_state.get("heat_boiler_preset"), 0.95))
        if st.session_state.get("heat_eta_override", False) or (not client_view):
            st.number_input(
                "Boiler seasonal efficiency η (useful heat / gas input)",
                min_value=0.50, max_value=1.00,
                value=float(st.session_state.get("heat_eta_boiler", preset_eta)),
                step=0.01,
                key="heat_eta_boiler",
            )
        else:
            st.session_state["heat_eta_boiler"] = preset_eta

        st.divider()
        st.subheader("Heat pump efficiency")

        if client_view:
            st.number_input(
                "Seasonal COP (annual average)",
                min_value=0.5, value=float(st.session_state.get("heat_cop_base_manual", 2.8)),
                step=0.1,
                key="heat_cop_base_manual",
            )
            st.session_state["heat_cop_source"] = "Manual"
            st.session_state["heat_cop_method_manual"] = "Single seasonal COP"
        else:
            st.radio("COP source", ["Manual", "From datasheet SCOP"], index=0, key="heat_cop_source")

            heat_climate = st.session_state.get("heat_climate", "Tbilisi")
            cop_source = st.session_state.get("heat_cop_source", "Manual")
            cop_base = 2.8

            if cop_source == "From datasheet SCOP":
                st.number_input("SCOP (datasheet, seasonal)", min_value=0.5, value=3.78, step=0.01, key="heat_scop")
                st.radio("Use SCOP", ["Use SCOP directly", "Advanced: 3-point winter"], index=0, key="heat_cop_method_scop")
                if st.session_state.get("heat_cop_method_scop") == "Use SCOP directly":
                    cop_base = float(st.session_state.get("heat_scop", 3.78))
                else:
                    st.number_input("COP at -3°C (≤50°C supply)", min_value=0.1, value=2.6, step=0.05, key="heat_base_cop_m3")
                    st.number_input("COP at +2°C (≤50°C supply)", min_value=0.1, value=2.8, step=0.05, key="heat_base_cop_p2")
                    st.number_input("COP at +7°C (≤50°C supply)", min_value=0.1, value=3.0, step=0.05, key="heat_base_cop_p7")
                    pts = {
                        -3: st.session_state["heat_base_cop_m3"],
                        2: st.session_state["heat_base_cop_p2"],
                        7: st.session_state["heat_base_cop_p7"],
                    }
                    cop_base = weighted_avg(pts, SH_WEIGHTS_3[heat_climate])
                    st.caption(f"Derived seasonal COP (weighted by climate): {cop_base:.2f}")
            else:
                st.radio("Manual method", ["Single seasonal COP", "Advanced: 3-point winter"], index=0, key="heat_cop_method_manual")
                if st.session_state.get("heat_cop_method_manual") == "Single seasonal COP":
                    st.number_input("Seasonal COP (annual average)", min_value=0.5, value=2.8, step=0.1, key="heat_cop_base_manual")
                else:
                    st.number_input("COP at -3°C (≤50°C supply)", min_value=0.1, value=2.6, step=0.05, key="heat_base_cop_m3")
                    st.number_input("COP at +2°C (≤50°C supply)", min_value=0.1, value=2.8, step=0.05, key="heat_base_cop_p2")
                    st.number_input("COP at +7°C (≤50°C supply)", min_value=0.1, value=3.0, step=0.05, key="heat_base_cop_p7")
                    pts = {
                        -3: st.session_state["heat_base_cop_m3"],
                        2: st.session_state["heat_base_cop_p2"],
                        7: st.session_state["heat_base_cop_p7"],
                    }
                    cop_base = weighted_avg(pts, SH_WEIGHTS_3[heat_climate])
                    st.caption(f"Derived seasonal COP (weighted by climate): {cop_base:.2f}")

            with st.expander("Advanced: checkpoint COP (sanity check)", expanded=False):
                st.number_input("Checkpoint COP at cold/design (optional)", min_value=0.0, value=0.0, step=0.01, key="heat_checkpoint_cop")

        st.divider()
        st.subheader("High-temperature loads (>50°C)")

        st.checkbox("Booster installed (required if any load >50°C)", value=False, key="heat_booster_installed")
        if st.session_state.get("heat_booster_installed", False):
            st.number_input("Booster COP (constant)", min_value=0.5, value=6.3, step=0.1, key="heat_cop_boost")
        else:
            st.session_state["heat_cop_boost"] = 0.0

        st.divider()
        st.subheader("CAPEX / Payback (optional)")
        st.checkbox("Calculate payback", value=False, key="heat_enable_payback")
        if st.session_state.get("heat_enable_payback", False):
            st.number_input("CAPEX: HP system (GEL)", min_value=0.0, value=0.0, step=10_000.0, key="heat_capex_hp")
            st.number_input("CAPEX: Boiler baseline (GEL)", min_value=0.0, value=0.0, step=10_000.0, key="heat_capex_boiler")
        else:
            if "heat_capex_hp" not in st.session_state:
                st.session_state["heat_capex_hp"] = 0.0
            if "heat_capex_boiler" not in st.session_state:
                st.session_state["heat_capex_boiler"] = 0.0

    # =====================================================
    # Demand (main)
    # =====================================================
    st.markdown("### Demand")

    heat_mode = st.session_state.get("heat_mode", "Existing building (comparison)")
    heat_application = st.session_state.get("heat_application", APPLICATIONS[0])
    heat_climate = st.session_state.get("heat_climate", CLIMATES[0])

    if heat_mode == "Existing building (comparison)":
        colA, colB = st.columns([1.2, 1.0])

        with colA:
            st.selectbox("Building type (for DHW presets only)", BUILDING_TYPES, index=1, key="heat_building_type")
            building_type = st.session_state.get("heat_building_type", "Hotel")

            st.radio(
                "How do you know the building demand?",
                [
                    "I know annual useful heat demand (kWh_th/year)",
                    "I know annual gas consumption (m³/year)",
                ],
                index=1,
                key="heat_demand_input_method",
            )

            method = st.session_state.get("heat_demand_input_method")
            eta_boiler = float(st.session_state.get("heat_eta_boiler", 0.93))
            kwh_per_m3 = float(st.session_state.get("heat_kwh_per_m3", 10.0))

            if method == "I know annual useful heat demand (kWh_th/year)":
                st.number_input(
                    "Annual useful heat demand (kWh_th/year)",
                    min_value=1.0, value=float(st.session_state.get("heat_q_annual", 344_100.0)),
                    step=50_000.0, key="heat_q_annual"
                )
            else:
                st.number_input(
                    "Annual gas consumption (m³/year)",
                    min_value=1.0, value=float(st.session_state.get("heat_gas_m3_year", 50_000.0)),
                    step=1_000.0,
                    key="heat_gas_m3_year"
                )
                st.selectbox(
                    "Does this gas include DHW (Domestic Hot Water)?",
                    ["No (space heating only)", "Yes (includes DHW)", "Not sure"],
                    index=0,
                    key="heat_gas_includes_dhw",
                )
                gas_m3_in = float(st.session_state.get("heat_gas_m3_year", 0.0))
                gas_input_kwh_in = gas_m3_in * kwh_per_m3
                Q_from_gas = gas_input_kwh_in * eta_boiler
                st.session_state["heat_q_annual"] = float(Q_from_gas)

                if st.session_state.get("heat_gas_includes_dhw") == "Yes (includes DHW)" and heat_application == "Space heating only":
                    st.warning("Your gas includes DHW, but Application is set to 'Space heating only'. Consider switching to 'Space heating + DHW'.")

                if st.session_state.get("heat_gas_includes_dhw") == "Not sure":
                    st.info("If gas includes kitchen/process loads, results may be distorted. Use best estimate for heating-related gas where possible.")

        with colB:
            st.markdown("**Used in calculation**")
            Q_total = float(st.session_state.get("heat_q_annual", 344_100.0))
            st.number_input(
                "Annual useful heat demand (kWh_th/year)",
                min_value=1.0,
                value=float(Q_total),
                step=50_000.0,
                disabled=True,
            )
            method = st.session_state.get("heat_demand_input_method")
            if method == "I know annual gas consumption (m³/year)":
                st.caption("Derived from gas: Q = (m³/year × kWh/m³) × η")
            else:
                st.caption("Direct input (annual useful heat).")

        Q_total = float(st.session_state.get("heat_q_annual", 344_100.0))
        insulation = st.session_state.get("heat_insulation", "Standard")

    else:
        colA, colB = st.columns([1.2, 1.0])

        with colA:
            st.selectbox("Building type", BUILDING_TYPES, index=1, key="heat_building_type")
            st.selectbox("Insulation level", INSULATION_LEVELS, index=1, key="heat_insulation")
            building_type = st.session_state.get("heat_building_type", "Hotel")
            insulation = st.session_state.get("heat_insulation", "Standard")

            st.radio(
                "Demand method",
                [
                    "From area (m²) + benchmarks",
                    "From peak heat load (kW) + FLH (Full Load Hours)",
                    "Direct annual useful heat demand (kWh_th/year)",
                ],
                index=0,
                key="heat_demand_method",
            )
            dm = st.session_state.get("heat_demand_method")

            if dm == "From area (m²) + benchmarks":
                st.number_input("Heated area (m²)", min_value=1.0, value=12000.0, step=100.0, key="heat_area_m2")
                area_m2 = float(st.session_state.get("heat_area_m2", 12000.0))
                intensity = float(BASE_KWH_PER_M2_YEAR[building_type][insulation]) * float(CLIMATE_INTENSITY_FACTOR[heat_climate])
                Q_est = area_m2 * intensity
                st.caption(f"Benchmark intensity used: {intensity:.0f} kWh/m²·year (incl. climate factor)")
                st.session_state["heat_q_annual"] = float(Q_est)

            elif dm == "From peak heat load (kW) + FLH (Full Load Hours)":
                st.number_input("Peak heating load (kW)", min_value=1.0, value=500.0, step=10.0, key="heat_peak_kw")
                peak_kw = float(st.session_state.get("heat_peak_kw", 500.0))
                flh = float(BASE_FLH[building_type][insulation]) * float(CLIMATE_FLH_FACTOR[heat_climate])
                st.checkbox("Override FLH (Full Load Hours)", value=False, key="heat_override_flh")
                if st.session_state.get("heat_override_flh", False):
                    st.number_input("FLH used (hours/year)", min_value=200.0, value=float(round(flh)), step=100.0, key="heat_flh_used")
                    flh_used = float(st.session_state.get("heat_flh_used", flh))
                else:
                    flh_used = flh
                    st.caption(f"FLH preset used: {flh_used:.0f} hours/year (incl. climate factor)")
                Q_est = peak_kw * flh_used
                st.session_state["heat_q_annual"] = float(Q_est)

            else:
                st.number_input(
                    "Annual useful heat demand (kWh_th/year)",
                    min_value=1.0, value=float(st.session_state.get("heat_q_annual", 344_100.0)),
                    step=50_000.0, key="heat_q_annual"
                )

        with colB:
            st.markdown("**Used in calculation**")
            Q_total = float(st.session_state.get("heat_q_annual", 344_100.0))
            st.number_input(
                "Annual useful heat demand (kWh_th/year)",
                min_value=1.0,
                value=float(Q_total),
                step=50_000.0,
                disabled=True,
            )
            st.caption("This is the value used in calculation.")

        Q_total = float(st.session_state.get("heat_q_annual", 344_100.0))

    # =====================================================
    # SH/DHW split
    # =====================================================
    if heat_application == "Space heating only":
        dhw_share_pct = 0
        sh_share_pct = 100
        dhw_target_c = 50
    else:
        building_type = st.session_state.get("heat_building_type", "Hotel")
        preset = int(round(DHW_SHARE_PRESET.get(building_type, 0.10) * 100))
        st.checkbox("Override DHW share (%)", value=False, key="heat_dhw_override")
        if st.session_state.get("heat_dhw_override", False):
            st.slider("DHW share (%)", 0, 100, preset, 1, key="heat_dhw_share_pct")
            dhw_share_pct = int(st.session_state.get("heat_dhw_share_pct", preset))
        else:
            dhw_share_pct = preset
            st.session_state["heat_dhw_share_pct"] = dhw_share_pct

        sh_share_pct = 100 - dhw_share_pct

        st.checkbox("High-temp DHW required (>=60°C)", value=False, key="heat_high_temp_dhw")
        dhw_target_c = 60 if st.session_state.get("heat_high_temp_dhw", False) else 50

    Q_sh = Q_total * (sh_share_pct / 100.0)
    Q_dhw = Q_total * (dhw_share_pct / 100.0)

    # =====================================================
    # Temperature regime and boosted fraction
    # =====================================================
    st.divider()
    st.markdown("### Space heating temperatures (base HP capped at 50°C)")

    if Q_sh > 0:
        st.selectbox("Space heating regime (supply/return)", list(HEATING_REGIMES.keys()), index=1, key="heat_regime_name")
        regime_name = st.session_state.get("heat_regime_name")
        supply_c, _ = HEATING_REGIMES[regime_name]

        st.checkbox("Mixed systems (some emitters need >50°C)", value=False, key="heat_mixed_systems")
        if not st.session_state.get("heat_mixed_systems", False):
            sh_high_frac = 1.0 if supply_c > HP_MAX_SUPPLY_C else 0.0
            st.info(f"Selected supply {supply_c}°C → boosted share for space heating = {'100%' if sh_high_frac==1 else '0%'}")
        else:
            st.slider("Fraction of space heating needing >50°C (%)", 0, 100, 40, 1, key="heat_sh_high_frac_pct")
            sh_high_frac = float(st.session_state.get("heat_sh_high_frac_pct", 40)) / 100.0
    else:
        sh_high_frac = 0.0

    dhw_high_frac = 1.0 if (Q_dhw > 0 and dhw_target_c > HP_MAX_SUPPLY_C) else 0.0

    booster_installed = bool(st.session_state.get("heat_booster_installed", False))
    cop_boost = float(st.session_state.get("heat_cop_boost", 0.0)) if booster_installed else 0.0

    if (not booster_installed) and (sh_high_frac > 0 or dhw_high_frac > 0):
        st.warning("Booster is OFF but some demand needs >50°C. Model forces boosted fractions to 0% (system would need redesign).")
        sh_high_frac = 0.0
        dhw_high_frac = 0.0

    Q_sh_high = Q_sh * sh_high_frac
    Q_sh_low = Q_sh * (1.0 - sh_high_frac)
    Q_dhw_high = Q_dhw * dhw_high_frac
    Q_dhw_low = Q_dhw * (1.0 - dhw_high_frac)

    boosted_share = 0.0 if Q_total <= 0 else (Q_sh_high + Q_dhw_high) / Q_total
    boosted_share_pct = int(round(boosted_share * 100))

    # =====================================================
    # COP base (recompute from sidebar selections)
    # =====================================================
    heat_climate = st.session_state.get("heat_climate", "Tbilisi")
    cop_source = st.session_state.get("heat_cop_source", "Manual")

    if cop_source == "From datasheet SCOP":
        if st.session_state.get("heat_cop_method_scop") == "Use SCOP directly":
            cop_base = float(st.session_state.get("heat_scop", 3.78))
        else:
            pts = {
                -3: float(st.session_state.get("heat_base_cop_m3", 2.6)),
                2: float(st.session_state.get("heat_base_cop_p2", 2.8)),
                7: float(st.session_state.get("heat_base_cop_p7", 3.0)),
            }
            cop_base = weighted_avg(pts, SH_WEIGHTS_3[heat_climate])
    else:
        if st.session_state.get("heat_cop_method_manual", "Single seasonal COP") == "Single seasonal COP":
            cop_base = float(st.session_state.get("heat_cop_base_manual", 2.8))
        else:
            pts = {
                -3: float(st.session_state.get("heat_base_cop_m3", 2.6)),
                2: float(st.session_state.get("heat_base_cop_p2", 2.8)),
                7: float(st.session_state.get("heat_base_cop_p7", 3.0)),
            }
            cop_base = weighted_avg(pts, SH_WEIGHTS_3[heat_climate])

    # =====================================================
    # Prices and baseline
    # =====================================================
    el_price = float(st.session_state.get("heat_el_price", 0.30))
    gas_price = float(st.session_state.get("heat_gas_price", 1.29))
    kwh_per_m3 = float(st.session_state.get("heat_kwh_per_m3", 10.0))
    eta_boiler = float(st.session_state.get("heat_eta_boiler", 0.93))

    chain_sh = hp_booster_chain(Q_sh_low, Q_sh_high, cop_base, cop_boost)
    chain_dhw = hp_booster_chain(Q_dhw_low, Q_dhw_high, cop_base, cop_boost)

    E_total_hp = chain_sh["E_base"] + chain_sh["E_boost"] + chain_dhw["E_base"] + chain_dhw["E_boost"]
    cost_hp = E_total_hp * el_price

    gas_method_active = (
        st.session_state.get("heat_mode") == "Existing building (comparison)"
        and st.session_state.get("heat_demand_input_method") == "I know annual gas consumption (m³/year)"
        and float(st.session_state.get("heat_gas_m3_year", 0.0)) > 0
    )

    if gas_method_active:
        gas_m3 = float(st.session_state.get("heat_gas_m3_year", 0.0))
        gas_input_kwh = gas_m3 * kwh_per_m3
        cost_gas = gas_m3 * gas_price
    else:
        gas_input_kwh = Q_total / eta_boiler if eta_boiler > 0 else 0.0
        gas_m3 = gas_input_kwh / kwh_per_m3 if kwh_per_m3 > 0 else 0.0
        cost_gas = gas_m3 * gas_price

    gas_co2_kg_per_year = gas_input_kwh * GAS_CO2_FACTOR_KG_PER_KWH
    gas_co2_tonnes_per_year = gas_co2_kg_per_year / 1000.0

    savings = cost_gas - cost_hp
    eff_cop = (Q_total / E_total_hp) if E_total_hp > 0 else 0.0

    checkpoint = float(st.session_state.get("heat_checkpoint_cop", 0.0) or 0.0)
    if (not st.session_state.get("heat_view_mode", "").startswith("Client")) and checkpoint > 0 and cop_base > checkpoint * 1.6:
        st.warning(
            f"Seasonal COP ({cop_base:.2f}) seems optimistic vs checkpoint COP ({checkpoint:.2f}). "
            "Consider lowering seasonal COP or using conservative assumptions."
        )

    project_name = (st.session_state.get("project_name") or "").strip()

    # =====================================================
    # Headline metrics
    # =====================================================
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annual useful heat demand", f"{Q_total:,.0f} kWh_th/yr")
    k2.metric("Boosted share (>50°C)", f"{boosted_share_pct}%")
    k3.metric("Seasonal COP used", f"{cop_base:.2f}")
    k4.metric("Boiler CO2 (baseline)", f"{gas_co2_tonnes_per_year:,.2f} tCO2/yr")

    t1, t2, t3 = st.tabs(["📊 Summary", "🧮 Details", "📄 Export"])

    # =====================================================
    # Summary tab
    # =====================================================
    with t1:
        a, b, c, d = st.columns(4)
        a.metric("Annual cost (Boiler)", f"{cost_gas:,.0f} GEL")
        b.metric("Annual cost (HP system)", f"{cost_hp:,.0f} GEL")
        if savings >= 0:
            c.metric("Annual savings", f"{savings:,.0f} GEL")
        else:
            c.metric("Annual difference", f"{savings:,.0f} GEL")
            st.warning("HP system is more expensive than gas under these inputs.")
        d.metric("Boiler CO2 (baseline)", f"{gas_co2_tonnes_per_year:,.2f} tCO2/yr")

        if cost_gas > 0:
            pct = (savings / cost_gas) * 100.0
            if savings >= 0:
                st.success(
                    f"Under these assumptions, the heat pump system reduces annual energy cost by **{pct:.1f}%** "
                    f"(**{savings:,.0f} GEL/year**) compared to a gas boiler baseline."
                )
            else:
                st.info(
                    f"Under these assumptions, the heat pump system increases annual energy cost by **{abs(pct):.1f}%** "
                    f"(**{abs(savings):,.0f} GEL/year**) compared to a gas boiler baseline."
                )

        with st.expander("Key assumptions used", expanded=False):
            st.write(f"- Electricity price: **{el_price:.3f} GEL/kWh**")
            st.write(f"- Gas price: **{gas_price:.2f} GEL/m³**")
            st.write(f"- Boiler efficiency η: **{eta_boiler:.2f}**")
            st.write(f"- Seasonal COP (≤50°C supply): **{cop_base:.2f}**")
            st.write(f"- Booster installed: **{'Yes' if booster_installed else 'No'}**" + (f" (COP {cop_boost:.2f})" if booster_installed else ""))
            st.write(f"- Boosted share (>50°C): **{boosted_share_pct}%**")
            if gas_method_active:
                st.write(f"- Gas baseline from bills: **{gas_m3:,.0f} m³/year**")
            else:
                st.write(f"- Gas baseline derived from demand: **{gas_m3:,.0f} m³/year**")

        fig = barh_chart(
            ["Gas Boiler", "HP System"],
            [cost_gas, cost_hp],
            "Annual cost comparison",
            "Annual cost (GEL)",
            value_fmt="{:,.0f} GEL",
        )
        st.pyplot(fig)

        if st.session_state.get("heat_enable_payback", False):
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

    # =====================================================
    # Details tab
    # =====================================================
    with t2:
        st.write("### Split")
        st.write(f"- SH (Space heating): **{sh_share_pct}%** → {Q_sh:,.0f} kWh_th/year")
        st.write(f"- DHW (Domestic hot water): **{dhw_share_pct}%** → {Q_dhw:,.0f} kWh_th/year (target {dhw_target_c}°C)")
        st.write("### Boosting (>50°C supply)")
        st.write(f"- SH boosted fraction: **{sh_high_frac*100:.0f}%** → {Q_sh_high:,.0f} kWh_th/year")
        st.write(f"- DHW boosted fraction: **{dhw_high_frac*100:.0f}%** → {Q_dhw_high:,.0f} kWh_th/year")
        st.write(f"- Total boosted share: **{boosted_share_pct}%**")
        st.divider()
        st.write("### Totals")
        st.write(f"- Total electricity (HP system): **{E_total_hp:,.0f} kWh_el/year**")
        st.write(f"- Effective system COP (useful heat / electricity): **{eff_cop:.2f}**")
        st.write(f"- Gas baseline: **{gas_m3:,.0f} m³/year**")
        st.divider()
        st.write("### CO2 (boiler baseline only)")
        st.write(f"- Gas boiler CO2: **{gas_co2_tonnes_per_year:,.2f} tCO2/year**")

    # =====================================================
    # Export tab
    # =====================================================
    with t3:
        st.markdown("### Report preset (select what you need)")
        preset = st.radio(
            "PDF report type",
            ["Client One-Pager", "Sales Proposal", "Technical (with Appendix)", "Financial Focus"],
            index=0,
            horizontal=True,
        )

        report_title = "Heat Pump vs Gas Boiler — Report"
        if project_name:
            report_title = f"{project_name} — Heat Pump vs Gas Boiler"

        subtitle = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Climate: {heat_climate} | Application: {heat_application}"
        if st.session_state.get("heat_mode") == "Existing building (comparison)":
            subtitle += " | Mode: Existing building"
        else:
            subtitle += " | Mode: Scratch estimate"

        pct_savings = (savings / cost_gas * 100.0) if cost_gas > 0 else 0.0
        if cost_gas > 0:
            if savings >= 0:
                narrative = (
                    f"Under current inputs, the heat pump system reduces annual energy cost by "
                    f"{pct_savings:.1f}% ({savings:,.0f} GEL/year) versus the gas boiler baseline."
                )
            else:
                narrative = (
                    f"Under current inputs, the heat pump system increases annual energy cost by "
                    f"{abs(pct_savings):.1f}% ({abs(savings):,.0f} GEL/year) versus the gas boiler baseline."
                )
        else:
            narrative = "Could not compute % savings (gas baseline cost is zero or undefined)."

        img_cost = fig_to_imagereader(
            barh_chart(
                ["Gas Boiler", "HP System"],
                [cost_gas, cost_hp],
                "Annual cost comparison",
                "Annual cost (GEL)",
                value_fmt="{:,.0f} GEL",
            )
        )

        img_pie_split = fig_to_imagereader(
            pie_chart(
                ["Space heating", "DHW"],
                [Q_sh, Q_dhw],
                "Heat demand split (annual useful heat)",
            )
        )

        img_pie_boost = fig_to_imagereader(
            pie_chart(
                ["≤50°C (base HP)", ">50°C (boosted)"],
                [Q_total - (Q_sh_high + Q_dhw_high), (Q_sh_high + Q_dhw_high)],
                "Boosted share (>50°C)",
            )
        )

        E_base_total = chain_sh["E_base"] + chain_dhw["E_base"]
        E_boost_total = chain_sh["E_boost"] + chain_dhw["E_boost"]
        cost_hp_base = E_base_total * el_price
        cost_hp_boost = E_boost_total * el_price
        img_cost_stacked = fig_to_imagereader(
            stacked_bar_cost_hp(cost_gas, cost_hp_base, cost_hp_boost)
        )

        sens_rows = []
        if preset == "Financial Focus":
            el_up = el_price * 1.20
            gas_up = gas_price * 1.20
            base_sav = cost_gas - cost_hp
            sav_el_up = cost_gas - (E_total_hp * el_up)
            cost_gas_gasup = gas_m3 * gas_up
            sav_gas_up = cost_gas_gasup - cost_hp

            sens_rows = [
                ["Base case", f"{el_price:.3f}", f"{gas_price:.2f}", f"{base_sav:,.0f}"],
                ["Electricity +20%", f"{el_up:.3f}", f"{gas_price:.2f}", f"{sav_el_up:,.0f}"],
                ["Gas +20%", f"{el_price:.3f}", f"{gas_up:.2f}", f"{sav_gas_up:,.0f}"],
            ]

        sections = []

        kpis = [
            ("Annual useful heat", f"{Q_total:,.0f} kWh_th/yr"),
            ("Annual cost (Gas)", f"{cost_gas:,.0f} GEL"),
            ("Annual cost (HP)", f"{cost_hp:,.0f} GEL"),
            ("Savings (Gas − HP)", f"{savings:,.0f} GEL"),
            ("Savings (%)", f"{pct_savings:.1f}%" if cost_gas > 0 else "N/A"),
            ("Boosted share", f"{boosted_share_pct}%"),
            ("Seasonal COP used", f"{cop_base:.2f}"),
            ("CO2 baseline (gas)", f"{gas_co2_tonnes_per_year:,.2f} tCO2/yr"),
        ]
        sections.append({"type": "kpi", "title": "Executive Summary", "items": kpis})
        sections.append({"type": "text", "title": "Summary", "lines": [narrative]})

        inputs_rows = [
            ["Electricity price (GEL/kWh)", f"{el_price:.3f}"],
            ["Gas price (GEL/m³)", f"{gas_price:.2f}"],
            ["Boiler efficiency η", f"{eta_boiler:.2f}"],
            ["Gas energy content (kWh/m³)", f"{kwh_per_m3:.1f}"],
            ["Seasonal COP (≤50°C)", f"{cop_base:.2f}"],
            ["Booster installed", "Yes" if booster_installed else "No"],
            ["Booster COP", f"{cop_boost:.2f}" if booster_installed else "N/A"],
            ["Heat mode", heat_mode],
            ["Climate", heat_climate],
            ["Application", heat_application],
        ]
        if gas_method_active:
            inputs_rows.append(["Gas baseline source", "From bills (m³/year input)"])
            inputs_rows.append(["Gas bills (m³/year)", f"{gas_m3:,.0f}"])
        else:
            inputs_rows.append(["Gas baseline source", "Derived from heat demand"])
            inputs_rows.append(["Derived gas (m³/year)", f"{gas_m3:,.0f}"])

        sections.append(
            {"type": "table", "title": "Inputs (snapshot)", "columns": ["Item", "Value"], "rows": inputs_rows}
        )

        if preset in ["Client One-Pager", "Sales Proposal", "Financial Focus"]:
            sections.append({"type": "image", "title": "Annual cost comparison", "image": img_cost, "w_cm": 16, "h_cm": 6.5})

        if preset in ["Client One-Pager", "Sales Proposal"]:
            sections.append({"type": "image", "title": "Heat demand split", "image": img_pie_split, "w_cm": 15, "h_cm": 7})

        if preset == "Sales Proposal":
            if boosted_share_pct > 0:
                sections.append({"type": "image", "title": "Boosted share", "image": img_pie_boost, "w_cm": 15, "h_cm": 7})

            sections.append(
                {
                    "type": "text",
                    "title": "Next steps",
                    "lines": [
                        "• Confirm space-heating temperature regime and whether any emitters require >50°C supply.",
                        "• Confirm DHW share and DHW target temperature (50°C vs 60°C).",
                        "• Validate whether gas bills include DHW / kitchen / process loads.",
                        "• Obtain HP datasheet and confirm SCOP / winter COP points for final proposal.",
                    ],
                }
            )

        if preset == "Financial Focus":
            if sens_rows:
                sections.append(
                    {
                        "type": "table",
                        "title": "Price sensitivity (stress test)",
                        "columns": ["Scenario", "Electricity (GEL/kWh)", "Gas (GEL/m³)", "Savings (GEL/yr)"],
                        "rows": sens_rows,
                    }
                )
            sections.append(
                {
                    "type": "text",
                    "title": "Interpretation",
                    "lines": [
                        "Sensitivity shows how savings change if energy prices move, keeping demand and system performance unchanged.",
                        "This is not a forecast; it is a robustness check for decision-making.",
                    ],
                }
            )

        if preset == "Technical (with Appendix)":
            sections.append({"type": "image", "title": "Annual cost comparison (HP breakdown)", "image": img_cost_stacked, "w_cm": 16, "h_cm": 7})

            tech_rows = [
                ["Total useful heat (kWh_th/yr)", f"{Q_total:,.0f}"],
                ["SH useful heat (kWh_th/yr)", f"{Q_sh:,.0f}"],
                ["DHW useful heat (kWh_th/yr)", f"{Q_dhw:,.0f}"],
                ["Boosted useful heat (kWh_th/yr)", f"{(Q_sh_high + Q_dhw_high):,.0f}"],
                ["Non-boosted useful heat (kWh_th/yr)", f"{(Q_total - (Q_sh_high + Q_dhw_high)):,.0f}"],
                ["HP electricity base (kWh_el/yr)", f"{E_base_total:,.0f}"],
                ["HP electricity booster (kWh_el/yr)", f"{E_boost_total:,.0f}"],
                ["HP total electricity (kWh_el/yr)", f"{E_total_hp:,.0f}"],
                ["Effective system COP", f"{eff_cop:.2f}"],
                ["Gas input energy (kWh/yr)", f"{gas_input_kwh:,.0f}"],
                ["Gas volume (m³/yr)", f"{gas_m3:,.0f}"],
                ["CO2 baseline (tCO2/yr)", f"{gas_co2_tonnes_per_year:,.2f}"],
            ]
            sections.append(
                {"type": "table", "title": "Appendix — Calculation breakdown", "columns": ["Metric", "Value"], "rows": tech_rows}
            )

            sections.append(
                {
                    "type": "text",
                    "title": "Assumptions & limitations",
                    "lines": [
                        "• Base HP supply is capped at 50°C; demand above 50°C is treated as boosted share only if booster is installed.",
                        "• Seasonal COP is an annual average (manual or derived). Actual performance depends on design and operating conditions.",
                        "• CO2 shown is gas combustion baseline only (does not include electricity emissions).",
                    ],
                }
            )

        sections.append(
            {
                "type": "text",
                "title": "Notes",
                "lines": [
                    "CO2 values represent gas combustion baseline only. Electricity emissions are not included in this tool.",
                    "This report is a screening-level comparison and should be validated with detailed design inputs for final sizing.",
                ],
            }
        )

        pdf_bytes = build_rich_pdf_report(
            report_title=report_title,
            subtitle=subtitle,
            sections=sections,
            footer_note="Treimax Georgia — Internal tool output for feasibility screening.",
        )

        pdf_filename = f"hp_vs_boiler_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        if project_name:
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in project_name).strip("_")
            if safe_name:
                pdf_filename = f"{safe_name}_hp_vs_boiler_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

        st.download_button(
            "📄 Download PDF report",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True,
        )

        df_export = pd.DataFrame(
            [
                ("Project", "Project name", project_name),
                ("Input", "Annual useful heat (kWh_th/year)", Q_total),
                ("Input", "Boosted share (%)", boosted_share_pct),
                ("Input", "Seasonal COP used", cop_base),
                ("Input", "Booster installed", booster_installed),
                ("Input", "Electricity price (GEL/kWh)", el_price),
                ("Input", "Gas price (GEL/m3)", gas_price),
                ("Input", "Boiler efficiency eta", eta_boiler),
                ("Baseline", "Gas baseline (m3/year)", gas_m3),
                ("Result", "Cost gas (GEL/year)", cost_gas),
                ("Result", "Cost HP (GEL/year)", cost_hp),
                ("Result", "Savings (GEL/year)", savings),
                ("Result", "Gas CO2 (tCO2/year)", gas_co2_tonnes_per_year),
                ("Report", "Preset", preset),
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
# CHILLER MODULE
# =========================================================
def compute_setup_eff_and_capacity(setup: str) -> dict:
    rows = []
    cap_list = []
    eff_list = []

    for i in range(1, 5):
        use = bool(st.session_state.get(f"ch_{setup}_use_{i}", i == 1))
        if not use:
            continue
        qty = int(st.session_state.get(f"ch_{setup}_qty_{i}", 1) or 1)
        kw = float(st.session_state.get(f"ch_{setup}_kw_{i}", 500.0) or 0.0)
        eff = float(st.session_state.get(f"ch_{setup}_eff_{i}", 3.5) or 0.0)
        metric = st.session_state.get(f"ch_{setup}_metric_{i}", "SEER (recommended)")

        cap = qty * kw
        rows.append({"Line": i, "Qty": qty, "kW_each": kw, "Capacity_kW": cap, "Metric": metric, "Efficiency": eff})

        cap_list.append(cap)
        eff_list.append(eff)

    total_cap = sum(cap_list)
    eff_weighted = harmonic_mean_weighted_by_capacity(cap_list, eff_list)
    return {"rows": rows, "total_cap_kw": total_cap, "eff_weighted": eff_weighted}


def run_chiller():
    st.subheader("Chiller comparison (payback)")
    st.info(
        "Model:\n"
        "- Build Setup A and Setup B (each can have 1–4 chillers).\n"
        "- App computes a capacity-weighted seasonal efficiency (harmonic mean).\n"
        "- Annual electricity = Annual cooling demand (kWh_cool) / efficiency\n"
        "- Annual cost = kWh_el × electricity price\n"
        "- Payback (optional) = extra CAPEX / savings"
    )

    with st.sidebar:
        st.header("Chiller Inputs")
        st.number_input("Electricity price (GEL/kWh)", min_value=0.001, value=0.30, step=0.01, format="%.3f", key="ch_el_price")

        st.divider()
        st.subheader("Cooling demand (choose method)")
        st.radio(
            "Demand method",
            [
                "Direct annual cooling demand (kWh_cool/year)",
                "Peak cooling load (kW) + EFLH (Equivalent Full Load Hours)",
                "Months + hours/day + load factor (estimator)",
            ],
            index=1,
            key="ch_demand_method",
        )

        dm = st.session_state.get("ch_demand_method")

        if dm.startswith("Direct"):
            st.number_input("Annual cooling demand (kWh_cool/year)", min_value=1.0, value=1_000_000.0, step=50_000.0, key="ch_q_cool_annual")
        elif dm.startswith("Peak"):
            st.number_input("Peak cooling load (kW)", min_value=1.0, value=800.0, step=10.0, key="ch_peak_cool_kw")
            st.number_input("EFLH used (hours/year)", min_value=200.0, value=1800.0, step=100.0, key="ch_cflh_used")
        else:
            st.number_input("Peak cooling load (kW)", min_value=1.0, value=800.0, step=10.0, key="ch_peak_cool_kw")
            st.number_input("Cooling season (months)", min_value=1, max_value=12, value=6, step=1, key="ch_months")
            st.number_input("Avg operating hours per day", min_value=1.0, max_value=24.0, value=12.0, step=0.5, key="ch_hours_per_day")
            st.slider("Avg load factor (0–100%)", 5, 100, 55, 1, key="ch_load_factor")

        st.divider()
        st.subheader("Setup names")
        st.text_input("Setup A name", value="Setup A", key="ch_label_a")
        st.text_input("Setup B name", value="Setup B", key="ch_label_b")

        st.divider()
        st.subheader("Setup A chillers (up to 4 lines)")
        for i in range(1, 5):
            st.checkbox(f"Use line {i}", value=(i == 1), key=f"ch_a_use_{i}")
            if st.session_state.get(f"ch_a_use_{i}", i == 1):
                cols = st.columns([1.0, 1.2, 1.8, 1.2])
                with cols[0]:
                    st.number_input("Qty", min_value=1, value=1, step=1, key=f"ch_a_qty_{i}")
                with cols[1]:
                    st.number_input("kW each", min_value=1.0, value=500.0, step=10.0, key=f"ch_a_kw_{i}")
                with cols[2]:
                    st.selectbox("Metric", ["SEER (recommended)", "EER avg (fallback)"], index=0, key=f"ch_a_metric_{i}")
                with cols[3]:
                    st.number_input("Value", min_value=0.5, value=4.0, step=0.1, key=f"ch_a_eff_{i}")

        st.divider()
        st.subheader("Setup B chillers (up to 4 lines)")
        for i in range(1, 5):
            st.checkbox(f"Use line {i}", value=(i == 1), key=f"ch_b_use_{i}")
            if st.session_state.get(f"ch_b_use_{i}", i == 1):
                cols = st.columns([1.0, 1.2, 1.8, 1.2])
                with cols[0]:
                    st.number_input("Qty", min_value=1, value=1, step=1, key=f"ch_b_qty_{i}")
                with cols[1]:
                    st.number_input("kW each", min_value=1.0, value=500.0, step=10.0, key=f"ch_b_kw_{i}")
                with cols[2]:
                    st.selectbox("Metric", ["SEER (recommended)", "EER avg (fallback)"], index=0, key=f"ch_b_metric_{i}")
                with cols[3]:
                    st.number_input("Value", min_value=0.5, value=4.5, step=0.1, key=f"ch_b_eff_{i}")

        st.divider()
        st.subheader("Payback (optional)")
        st.checkbox("Calculate payback", value=False, key="ch_enable_payback")
        st.number_input("CAPEX Setup A (GEL)", min_value=0.0, value=0.0, step=10_000.0, key="ch_capex_a")
        st.number_input("CAPEX Setup B (GEL)", min_value=0.0, value=0.0, step=10_000.0, key="ch_capex_b")

    dm = st.session_state.get("ch_demand_method", "Peak cooling load (kW) + EFLH (Equivalent Full Load Hours)")
    if dm.startswith("Direct"):
        Q_cool = float(st.session_state.get("ch_q_cool_annual", 1_000_000.0))
        demand_note = "Direct annual cooling demand"
    elif dm.startswith("Peak"):
        peak_kw = float(st.session_state.get("ch_peak_cool_kw", 800.0))
        eflh = float(st.session_state.get("ch_cflh_used", 1800.0))
        Q_cool = peak_kw * eflh
        demand_note = f"Peak×EFLH: {peak_kw:,.0f} kW × {eflh:,.0f} h"
    else:
        peak_kw = float(st.session_state.get("ch_peak_cool_kw", 800.0))
        months = int(st.session_state.get("ch_months", 6))
        hours_day = float(st.session_state.get("ch_hours_per_day", 12.0))
        load_factor = float(st.session_state.get("ch_load_factor", 55)) / 100.0
        eflh_equiv = months * 30.4 * hours_day * load_factor
        Q_cool = peak_kw * eflh_equiv
        demand_note = f"Estimator: {months} mo, {hours_day} h/day, {load_factor*100:.0f}% load -> EFLH~{eflh_equiv:,.0f} h"

    setup_a = compute_setup_eff_and_capacity("a")
    setup_b = compute_setup_eff_and_capacity("b")

    label_a = st.session_state.get("ch_label_a", "Setup A")
    label_b = st.session_state.get("ch_label_b", "Setup B")

    eff_a = setup_a["eff_weighted"]
    eff_b = setup_b["eff_weighted"]

    el_price = float(st.session_state.get("ch_el_price", 0.30))

    E_a = (Q_cool / eff_a) if eff_a > 0 else 0.0
    E_b = (Q_cool / eff_b) if eff_b > 0 else 0.0

    cost_a = E_a * el_price
    cost_b = E_b * el_price
    savings = cost_a - cost_b

    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annual cooling demand", f"{Q_cool:,.0f} kWh_cool")
    k2.metric(f"{label_a} weighted eff.", f"{eff_a:.2f}" if eff_a > 0 else "N/A")
    k3.metric(f"{label_b} weighted eff.", f"{eff_b:.2f}" if eff_b > 0 else "N/A")
    k4.metric("Savings (A − B)", f"{savings:,.0f} GEL/yr")

    st.caption(f"Demand method: {demand_note}")

    t1, t2, t3 = st.tabs(["📊 Summary", "🧮 Details", "📄 Export"])

    with t1:
        a, b, c = st.columns(3)
        a.metric(f"{label_a} annual cost", f"{cost_a:,.0f} GEL")
        b.metric(f"{label_b} annual cost", f"{cost_b:,.0f} GEL")
        if savings >= 0:
            c.metric("Annual savings", f"{savings:,.0f} GEL")
        else:
            c.metric("Annual difference", f"{savings:,.0f} GEL")
            st.warning("Setup B is more expensive than Setup A under these assumptions.")

        fig = barh_chart(
            [label_a, label_b],
            [cost_a, cost_b],
            "Chiller annual cost comparison",
            "Annual cost (GEL)",
            value_fmt="{:,.0f} GEL",
        )
        st.pyplot(fig)

        if st.session_state.get("ch_enable_payback", False):
            capex_a = float(st.session_state.get("ch_capex_a", 0.0))
            capex_b = float(st.session_state.get("ch_capex_b", 0.0))
            delta_capex = capex_b - capex_a
            st.subheader("Payback (optional)")
            p1, p2, p3 = st.columns(3)
            p1.metric("Extra CAPEX (B − A)", f"{delta_capex:,.0f} GEL")
            if savings > 0 and delta_capex > 0:
                pb_years = delta_capex / savings
                p2.metric("Payback (years)", f"{pb_years:.2f}")
                p3.metric("Payback (months)", f"{pb_years*12:.1f}")
            else:
                p2.metric("Payback (years)", "N/A")
                p3.metric("Payback (months)", "N/A")

    with t2:
        st.write("### Setups")
        st.write(f"**{label_a}** total capacity: **{setup_a['total_cap_kw']:,.0f} kW** | weighted efficiency: **{eff_a:.2f}**")
        st.dataframe(pd.DataFrame(setup_a["rows"]))
        st.write(f"**{label_b}** total capacity: **{setup_b['total_cap_kw']:,.0f} kW** | weighted efficiency: **{eff_b:.2f}**")
        st.dataframe(pd.DataFrame(setup_b["rows"]))

        st.divider()
        st.write("### Electricity & costs")
        st.write(f"- {label_a} electricity: **{E_a:,.0f} kWh_el/year**")
        st.write(f"- {label_b} electricity: **{E_b:,.0f} kWh_el/year**")
        st.write(f"- Electricity price: **{el_price:.3f} GEL/kWh**")

    with t3:
        project_name = (st.session_state.get("project_name") or "").strip()
        report_title = "Chiller Comparison Report"
        if project_name:
            report_title = f"{project_name} — Chiller Comparison"

        sections = [
            {"type": "kpi", "title": "Executive Summary", "items": [
                ("Annual cooling demand", f"{Q_cool:,.0f} kWh_cool/yr"),
                (f"{label_a} annual cost", f"{cost_a:,.0f} GEL"),
                (f"{label_b} annual cost", f"{cost_b:,.0f} GEL"),
                ("Savings (A − B)", f"{savings:,.0f} GEL/yr"),
            ]},
            {"type": "text", "title": "Inputs", "lines": [
                f"Demand method: {demand_note}",
                f"Electricity price: {el_price:.3f} GEL/kWh",
                f"{label_a} weighted efficiency: {eff_a:.2f}",
                f"{label_b} weighted efficiency: {eff_b:.2f}",
            ]},
            {"type": "image", "title": "Annual cost comparison", "image": fig_to_imagereader(
                barh_chart([label_a, label_b], [cost_a, cost_b], "Chiller annual cost comparison", "Annual cost (GEL)", "{:,.0f} GEL")
            ), "w_cm": 16, "h_cm": 6.5},
        ]

        pdf_bytes = build_rich_pdf_report(
            report_title=report_title,
            subtitle=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            sections=sections,
            footer_note="Treimax Georgia — Internal tool output for feasibility screening.",
        )

        pdf_filename = f"chiller_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        if project_name:
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in project_name).strip("_")
            if safe_name:
                pdf_filename = f"{safe_name}_chiller_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

        st.download_button(
            "📄 Download PDF report",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True,
        )

        export_rows = [
            ("Project", "Project name", project_name),
            ("Input", "Annual cooling demand (kWh_cool/year)", Q_cool),
            ("Input", "Electricity price (GEL/kWh)", el_price),
            ("SetupA", "Weighted efficiency", eff_a),
            ("SetupA", "Annual cost (GEL/year)", cost_a),
            ("SetupB", "Weighted efficiency", eff_b),
            ("SetupB", "Annual cost (GEL/year)", cost_b),
            ("Result", "Savings (A − B) (GEL/year)", savings),
        ]
        df_export = pd.DataFrame(export_rows, columns=["Type", "Key", "Value"])
        st.download_button(
            "⬇️ Download CSV",
            data=df_export.to_csv(index=False).encode("utf-8"),
            file_name=f"chiller_comparison_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# ROUTING
# =========================================================
if tool == "Heat Pump vs Boiler":
    run_heating()
else:
    run_chiller()
