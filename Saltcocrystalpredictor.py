import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
from xml.sax.saxutils import escape
import math

# ── Matplotlib (server-side rendering) ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── ReportLab ──
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether, Image as RLImage
)
from reportlab.platypus.flowables import Flowable

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="PharmaCrystal Pro v6.0", layout="wide", page_icon="🔬")

st.title("🔬 PharmaCrystal Pro v6.0 — Advanced Solid-State Screening Engine")
st.markdown(
    "Enhanced scientific models: **Cruz-Cabeza ΔpKa classification** · "
    "**Yalkowsky GSE intrinsic solubility** · **Dose Number & MAD estimation** · "
    "**Ksp-based salt solubility** · **Common ion correction** · "
    "**Kauzmann temperature for ASD** · **Spring-and-Parachute Index** · "
    "**Lattice energy proxy** · **Gordon-Taylor Tg** · **10+ embedded PDF charts**"
)

# ─────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────
R_GAS       = 8.314          # J/(mol·K)
TEMP_K      = 298.15         # K (25°C)
V_REF       = 100.0          # cm³/mol  (reference molar volume for χ)
V_GI        = 250.0          # mL  (volume of GI fluid, FDA guidance)
PEFF_HIGH   = 2.0e-4         # cm/s  (high permeability threshold BCS)
SA_INTESTINE= 2.0e4          # cm²  (effective absorptive surface area)
T_RESIDENCE = 3.5 * 3600     # s    (small intestine transit ~3.5 h)

PH_GASTRIC  = 1.2
PH_FASSIF   = 6.5   # updated: FDA bioequivalence guidance uses 6.5
PH_FESSIF   = 5.0
PH_COLONIC  = 7.4

# ── Brand colours ──
BRAND_DARK   = colors.HexColor("#0D2B45")
BRAND_MID    = colors.HexColor("#1565C0")
BRAND_ACCENT = colors.HexColor("#00ACC1")
BRAND_LIGHT  = colors.HexColor("#E3F2FD")
BRAND_GREEN  = colors.HexColor("#2E7D32")
BRAND_AMBER  = colors.HexColor("#E65100")
BRAND_RED    = colors.HexColor("#B71C1C")
BRAND_GREY   = colors.HexColor("#F5F7FA")
TABLE_HEADER = colors.HexColor("#1565C0")
TABLE_ALT    = colors.HexColor("#EEF4FB")

MC = {
    'primary': '#1565C0', 'accent': '#00ACC1', 'green': '#2E7D32',
    'amber': '#E65100', 'red': '#B71C1C', 'purple': '#6A1B9A',
    'grey': '#546E7A', 'light': '#E3F2FD',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 11, 'axes.labelsize': 9, 'axes.titleweight': 'bold',
    'figure.facecolor': 'white', 'axes.facecolor': '#FAFAFA',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
})

# ─────────────────────────────────────────────
# COFORMER DATABASE  (expanded with mp, aqueous solubility)
# ─────────────────────────────────────────────
coformers_db = [
    # ── Inorganic Anions ──
    {"name": "Hydrochloride (Cl⁻)", "pKa": -7.0, "dd": 16.5, "dp": 13.0, "dh": 10.0,
     "type": "Salt (Anion)", "synthon": "Inorganic Ion", "mw": 36.5, "logP": -3.0,
     "rot_bonds": 0, "mp_celsius": None, "aq_sol_mg_ml": None,
     "hygro_risk": "High", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "Ph. Eur. / USP",
     "note": "Most common counterion (~29-43% marketed salts). Cl⁻ common-ion effect in gastric HCl "
             "can suppress dissolution — monitor with in-situ fibre optic UV. Risk of deliquescence "
             "for basic APIs with pKa > 8. Consider mesylate or besylate alternatives."},
    {"name": "Sulfate (SO₄²⁻)", "pKa": -3.0, "dd": 16.0, "dp": 16.0, "dh": 15.0,
     "type": "Salt (Anion)", "synthon": "Inorganic Ion", "mw": 96.1, "logP": -4.0,
     "rot_bonds": 0, "mp_celsius": None, "aq_sol_mg_ml": None,
     "hygro_risk": "Medium", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "Ph. Eur. / USP",
     "note": "~7-8% of marketed salts. Divalent — 1:2 stoichiometry possible. Generally lower "
             "hygroscopicity than HCl. Excellent crystallinity."},
    {"name": "Hydrobromide (Br⁻)", "pKa": -9.0, "dd": 16.5, "dp": 12.0, "dh": 9.0,
     "type": "Salt (Anion)", "synthon": "Inorganic Ion", "mw": 80.9, "logP": -3.0,
     "rot_bonds": 0, "mp_celsius": None, "aq_sol_mg_ml": None,
     "hygro_risk": "Medium", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "Ph. Eur. / USP",
     "note": "~1.9% marketed. Stronger acid than HCl. Monitor bromide daily intake (FDA TDI ~8 mg/kg)."},
    {"name": "Phosphate (H₂PO₄⁻)", "pKa": 2.1, "dd": 17.0, "dp": 15.0, "dh": 18.0,
     "type": "Salt (Anion)", "synthon": "Inorganic Ion", "mw": 95.0, "logP": -4.5,
     "rot_bonds": 0, "mp_celsius": None, "aq_sol_mg_ml": None,
     "hygro_risk": "Medium", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "Ph. Eur. / USP / GRAS",
     "note": "~3.1% marketed. Buffer capacity aids dissolution. Triprotic — specify 1:1, 1:2, 1:3."},
    {"name": "Tartrate (L-tartaric)", "pKa": 2.9, "dd": 17.2, "dp": 12.5, "dh": 21.0,
     "type": "Salt (Anion)", "synthon": "Carboxylic Acid", "mw": 150.1, "logP": -1.5,
     "rot_bonds": 1, "mp_celsius": 171, "aq_sol_mg_ml": 1390,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Medium", "tg_polymer": None, "regulatory_status": "GRAS / Ph. Eur.",
     "note": "~3.5% marketed. Natural origin (GRAS). Chiral — prefer L-form for regulatory simplicity."},
    {"name": "Mesylate (CH₃SO₃⁻)", "pKa": -1.2, "dd": 16.0, "dp": 15.0, "dh": 12.0,
     "type": "Salt (Anion)", "synthon": "Sulfonate", "mw": 96.1, "logP": -1.0,
     "rot_bonds": 0, "mp_celsius": 20, "aq_sol_mg_ml": None,
     "hygro_risk": "Low", "ich_flag": True,
     "ich_note": "ICH M7 Class 2: Alkyl mesylate genotoxic impurity risk in presence of MeOH. "
                 "Mandatory: GC-HS limit test for methyl methanesulfonate (MMS) < 1.5 ppm (TTC).",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "Ph. Eur. (ICH M7 control needed)",
     "note": "~2% marketed. Excellent physicochemical properties. No common-ion effect in GI. "
             "ICH M7 control strategy is mandatory — avoid MeOH in final synthesis steps."},
    {"name": "Esylate (C₂H₅SO₃⁻)", "pKa": -1.8, "dd": 16.0, "dp": 14.0, "dh": 11.5,
     "type": "Salt (Anion)", "synthon": "Sulfonate", "mw": 110.1, "logP": -0.5,
     "rot_bonds": 1, "mp_celsius": None, "aq_sol_mg_ml": None,
     "hygro_risk": "Low", "ich_flag": True,
     "ich_note": "ICH M7: Ethyl sulfonate ester risk. Lower concern than mesylate but formal "
                 "risk assessment per ICH M7 Annex required.",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "ICH M7 assessment required",
     "note": "Useful mesylate alternative; lower genotoxic risk but still needs ICH M7 dossier."},
    {"name": "Tosylate (C₇H₇SO₃⁻)", "pKa": -1.3, "dd": 18.0, "dp": 14.5, "dh": 11.0,
     "type": "Salt (Anion)", "synthon": "Sulfonate", "mw": 172.2, "logP": 0.8,
     "rot_bonds": 1, "mp_celsius": 106, "aq_sol_mg_ml": 670,
     "hygro_risk": "Low", "ich_flag": True,
     "ich_note": "ICH M7: Benzyl-type sulfonate ester impurity risk. Control alkyl tosylate formation.",
     "poly_risk": "Medium", "tg_polymer": None, "regulatory_status": "ICH M7 control needed",
     "note": "Good physicochemical profile. Genotoxic impurity control needed."},
    {"name": "Acetate (CH₃COO⁻)", "pKa": 4.76, "dd": 16.2, "dp": 11.0, "dh": 13.0,
     "type": "Salt (Anion)", "synthon": "Carboxylic Acid", "mw": 59.0, "logP": -0.2,
     "rot_bonds": 0, "mp_celsius": 17, "aq_sol_mg_ml": None,
     "hygro_risk": "Medium", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "GRAS / Ph. Eur.",
     "note": "~1.3% marketed. Volatile — acetic acid loss during spray drying/HME. Monitor by TGA."},
    {"name": "Maleate", "pKa": 1.9, "dd": 17.5, "dp": 11.5, "dh": 20.2,
     "type": "Salt (Anion)", "synthon": "Carboxylic Acid", "mw": 116.1, "logP": -0.5,
     "rot_bonds": 1, "mp_celsius": 131, "aq_sol_mg_ml": 788,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Medium", "tg_polymer": None, "regulatory_status": "Ph. Eur.",
     "note": "Strong acid. Monitor GI tolerability at high dose. Cis-isomer may isomerise to fumarate."},
    {"name": "Fumarate", "pKa": 3.0, "dd": 17.5, "dp": 12.0, "dh": 19.0,
     "type": "Salt (Anion)", "synthon": "Carboxylic Acid", "mw": 116.1, "logP": -0.5,
     "rot_bonds": 1, "mp_celsius": 287, "aq_sol_mg_ml": 4.9,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Medium", "tg_polymer": None, "regulatory_status": "GRAS / Ph. Eur.",
     "note": "Trans isomer of maleic acid. Excellent chemical stability. High mp = high lattice energy. GRAS."},
    {"name": "Citrate", "pKa": 3.1, "dd": 17.5, "dp": 12.5, "dh": 22.0,
     "type": "Salt (Anion)", "synthon": "Carboxylic Acid", "mw": 192.1, "logP": -1.7,
     "rot_bonds": 3, "mp_celsius": 153, "aq_sol_mg_ml": 592,
     "hygro_risk": "Medium", "ich_flag": False, "ich_note": "-",
     "poly_risk": "High", "tg_polymer": None, "regulatory_status": "GRAS / Ph. Eur.",
     "note": "Triprotic — can form 1:1, 1:2, or 1:3 stoichiometries. GRAS. High MW = tablet burden."},
    {"name": "Oxalate", "pKa": 1.2, "dd": 17.5, "dp": 12.8, "dh": 22.5,
     "type": "Salt (Anion)", "synthon": "Carboxylic Acid", "mw": 90.0, "logP": -1.4,
     "rot_bonds": 0, "mp_celsius": 190, "aq_sol_mg_ml": 143,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "Ph. Eur.",
     "note": "Very strong diacid. Nephrotoxic — limit to low-dose APIs (<50 mg oxalate/day)."},
    {"name": "Succinate", "pKa": 4.2, "dd": 17.0, "dp": 12.2, "dh": 18.5,
     "type": "Salt (Anion)", "synthon": "Carboxylic Acid", "mw": 118.1, "logP": -0.6,
     "rot_bonds": 2, "mp_celsius": 185, "aq_sol_mg_ml": 58,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Medium", "tg_polymer": None, "regulatory_status": "GRAS / Ph. Eur.",
     "note": "Krebs cycle intermediate. GRAS. Excellent cocrystal former for amines."},
    {"name": "Besylate (C₆H₅SO₃⁻)", "pKa": -1.0, "dd": 18.5, "dp": 14.0, "dh": 11.0,
     "type": "Salt (Anion)", "synthon": "Sulfonate", "mw": 157.2, "logP": 0.5,
     "rot_bonds": 1, "mp_celsius": None, "aq_sol_mg_ml": None,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "Ph. Eur.",
     "note": "Aromatic sulfonate — no ICH M7 concern (no alkyl sulfonate ester risk). Low hygroscopicity."},
    # ── Inorganic Cations ──
    {"name": "Sodium (Na⁺)", "pKa": 14.0, "dd": 15.5, "dp": 10.0, "dh": 5.0,
     "type": "Salt (Cation)", "synthon": "Inorganic Ion", "mw": 23.0, "logP": -5.0,
     "rot_bonds": 0, "mp_celsius": None, "aq_sol_mg_ml": None,
     "hygro_risk": "Medium", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "Ph. Eur. / USP",
     "note": "~9% marketed. Often dramatically improves dissolution rate. Na load relevant for cardiac/hypertensive patients (FDA labelling guidance)."},
    {"name": "Potassium (K⁺)", "pKa": 14.0, "dd": 15.0, "dp": 9.0, "dh": 4.0,
     "type": "Salt (Cation)", "synthon": "Inorganic Ion", "mw": 39.1, "logP": -5.0,
     "rot_bonds": 0, "mp_celsius": None, "aq_sol_mg_ml": None,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "Ph. Eur. / USP",
     "note": "Lower hygroscopicity than Na. Potassium load relevant for renal patients."},
    {"name": "Calcium (Ca²⁺)", "pKa": 12.0, "dd": 16.0, "dp": 8.0, "dh": 6.0,
     "type": "Salt (Cation)", "synthon": "Inorganic Ion", "mw": 40.1, "logP": -5.0,
     "rot_bonds": 0, "mp_celsius": None, "aq_sol_mg_ml": None,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "Ph. Eur. / USP",
     "note": "Divalent — forms 1:2 salts. Tends to form hydrates. Useful for poorly soluble acids."},
    {"name": "Magnesium (Mg²⁺)", "pKa": 11.5, "dd": 16.0, "dp": 8.0, "dh": 6.0,
     "type": "Salt (Cation)", "synthon": "Inorganic Ion", "mw": 24.3, "logP": -5.0,
     "rot_bonds": 0, "mp_celsius": None, "aq_sol_mg_ml": None,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "Ph. Eur. / USP / GRAS",
     "note": "Low toxicity. Commonly forms crystalline hydrates. GRAS."},
    {"name": "Meglumine", "pKa": 9.5, "dd": 17.5, "dp": 14.0, "dh": 25.0,
     "type": "Salt (Cation)", "synthon": "Amine", "mw": 195.2, "logP": -3.0,
     "rot_bonds": 5, "mp_celsius": 129, "aq_sol_mg_ml": None,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Medium", "tg_polymer": None, "regulatory_status": "Ph. Eur. / GRAS (parenteral)",
     "note": "Preferred for injectables. GRAS. Often gives amorphous or metastable forms."},
    {"name": "Tromethamine (TRIS)", "pKa": 8.1, "dd": 16.5, "dp": 12.0, "dh": 22.0,
     "type": "Salt (Cation)", "synthon": "Amine", "mw": 121.1, "logP": -3.3,
     "rot_bonds": 3, "mp_celsius": 172, "aq_sol_mg_ml": 800,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "Ph. Eur. / USP",
     "note": "Buffer agent. Used in parenterals. Growing popularity for oral acidic drug salts."},
    {"name": "L-Lysine", "pKa": 9.5, "dd": 17.0, "dp": 13.0, "dh": 20.0,
     "type": "Salt (Cation)", "synthon": "Amine", "mw": 146.2, "logP": -3.1,
     "rot_bonds": 4, "mp_celsius": 224, "aq_sol_mg_ml": None,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "High", "tg_polymer": None, "regulatory_status": "GRAS / Ph. Eur.",
     "note": "Amino acid counterion. Intrinsically chiral. GRAS."},
    {"name": "L-Arginine", "pKa": 10.8, "dd": 17.5, "dp": 13.5, "dh": 21.0,
     "type": "Salt (Cation)", "synthon": "Amine", "mw": 174.2, "logP": -3.5,
     "rot_bonds": 5, "mp_celsius": 244, "aq_sol_mg_ml": 150,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "High", "tg_polymer": None, "regulatory_status": "GRAS / Ph. Eur.",
     "note": "Strong base (guanidinium pKa ~12.5). Very high aqueous solubility. GRAS."},
    # ── Cocrystal Coformers ──
    {"name": "Nicotinamide", "pKa": 3.3, "dd": 19.0, "dp": 12.0, "dh": 11.0,
     "type": "Coformer", "synthon": "Pyridine/Amide", "mw": 122.1, "logP": -0.4,
     "rot_bonds": 1, "mp_celsius": 129, "aq_sol_mg_ml": 1000,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Medium", "tg_polymer": None, "regulatory_status": "GRAS (vitamin B3)",
     "note": "GRAS vitamin. CSD has 200+ cocrystal entries. Dual synthon: pyridine N + amide."},
    {"name": "Saccharin", "pKa": 2.0, "dd": 21.0, "dp": 13.9, "dh": 11.5,
     "type": "Coformer", "synthon": "Imide", "mw": 183.2, "logP": 0.9,
     "rot_bonds": 1, "mp_celsius": 229, "aq_sol_mg_ml": 4.0,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Medium", "tg_polymer": None, "regulatory_status": "FDA approved sweetener",
     "note": "Sweetener — beneficial for ODT/sublingual. Strong NH donor (imide). Low aqueous solubility."},
    {"name": "Urea", "pKa": 0.1, "dd": 17.0, "dp": 13.0, "dh": 15.0,
     "type": "Coformer", "synthon": "Amide", "mw": 60.1, "logP": -2.1,
     "rot_bonds": 0, "mp_celsius": 133, "aq_sol_mg_ml": 1080,
     "hygro_risk": "High", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "Ph. Eur.",
     "note": "Classic amide coformer. Highly hygroscopic — avoid for moisture-sensitive APIs."},
    {"name": "Glutaric Acid", "pKa": 4.3, "dd": 17.0, "dp": 12.0, "dh": 19.0,
     "type": "Coformer", "synthon": "Carboxylic Acid", "mw": 132.1, "logP": -0.3,
     "rot_bonds": 3, "mp_celsius": 97, "aq_sol_mg_ml": 640,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Medium", "tg_polymer": None, "regulatory_status": "GRAS",
     "note": "Flexible C5 diacid. GRAS. Commonly used in cocrystal screens."},
    {"name": "Malonic Acid", "pKa": 2.8, "dd": 17.2, "dp": 12.5, "dh": 20.0,
     "type": "Coformer", "synthon": "Carboxylic Acid", "mw": 104.1, "logP": -1.0,
     "rot_bonds": 1, "mp_celsius": 135, "aq_sol_mg_ml": 763,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": None, "regulatory_status": "GRAS / Ph. Eur.",
     "note": "Short-chain C3 diacid. Strong H-bond donor. Decarboxylation risk above 140°C."},
    {"name": "Caffeine", "pKa": -0.1, "dd": 18.5, "dp": 12.5, "dh": 10.8,
     "type": "Coformer", "synthon": "Amide", "mw": 194.2, "logP": -0.1,
     "rot_bonds": 0, "mp_celsius": 236, "aq_sol_mg_ml": 21.6,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "High", "tg_polymer": None, "regulatory_status": "GRAS",
     "note": "CNS-active — limits use to compatible APIs. Notorious for polymorphism. CSD: 50+ cocrystals."},
    {"name": "Theophylline", "pKa": 8.8, "dd": 19.0, "dp": 12.0, "dh": 11.0,
     "type": "Coformer", "synthon": "Amide", "mw": 180.2, "logP": 0.0,
     "rot_bonds": 0, "mp_celsius": 270, "aq_sol_mg_ml": 8.3,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "High", "tg_polymer": None, "regulatory_status": "Ph. Eur. / USP",
     "note": "Pharmacologically active (bronchodilator). Narrow therapeutic index — regulatory complexity."},
    # ── ASD Polymers ──
    {"name": "PVP-K30", "pKa": 13.0, "dd": 17.4, "dp": 10.8, "dh": 9.5,
     "type": "Polymer (ASD)", "synthon": "Amide", "mw": 40000, "logP": -1.0,
     "rot_bonds": 0, "mp_celsius": None, "aq_sol_mg_ml": None,
     "hygro_risk": "High", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": 443.0,
     "regulatory_status": "Ph. Eur. / USP / NF",
     "note": "Most used ASD polymer. Tg ~170°C. Highly hygroscopic — water plasticisation lowers Tg in situ. "
             "Monitor at 40°C/75%RH."},
    {"name": "HPMCAS-LF", "pKa": 5.0, "dd": 16.5, "dp": 8.5, "dh": 14.5,
     "type": "Polymer (ASD)", "synthon": "Carboxylic Acid", "mw": 18000, "logP": 0.5,
     "rot_bonds": 0, "mp_celsius": None, "aq_sol_mg_ml": None,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": 393.0,
     "regulatory_status": "Ph. Eur. / USP / NF",
     "note": "pH-dependent dissolution (>pH 5.5). Preferred for oral ASD via spray drying. "
             "Superior precipitation inhibition vs PVP. Lower hygroscopicity."},
    {"name": "PVPVA 64", "pKa": 13.0, "dd": 16.8, "dp": 10.2, "dh": 8.0,
     "type": "Polymer (ASD)", "synthon": "Amide", "mw": 45000, "logP": -0.5,
     "rot_bonds": 0, "mp_celsius": None, "aq_sol_mg_ml": None,
     "hygro_risk": "Medium", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": 379.0,
     "regulatory_status": "Ph. Eur. / NF",
     "note": "6:4 VP:VA copolymer. Lower Tg (~106°C) and hygroscopicity vs PVP. "
             "Preferred for hot-melt extrusion (lower processing temperature)."},
    {"name": "Eudragit L100", "pKa": 4.8, "dd": 16.2, "dp": 9.5, "dh": 16.0,
     "type": "Polymer (ASD)", "synthon": "Carboxylic Acid", "mw": 135000, "logP": 0.3,
     "rot_bonds": 0, "mp_celsius": None, "aq_sol_mg_ml": None,
     "hygro_risk": "Low", "ich_flag": False, "ich_note": "-",
     "poly_risk": "Low", "tg_polymer": 433.0,
     "regulatory_status": "Ph. Eur.",
     "note": "Enteric polymer. Dissolves >pH 6. Excellent for intestinal-targeted release. "
             "Good precipitation inhibition."},
]

# ─────────────────────────────────────────────
# ETTER'S SYNTHON SCORING (with detailed rationale)
# ─────────────────────────────────────────────
etter_scores = {
    ("Carboxylic Acid", "Pyridine/Amide"): 2, ("Carboxylic Acid", "Amine"): 2,
    ("Carboxylic Acid", "Amide"): 1,          ("Carboxylic Acid", "Carboxylic Acid"): 1,
    ("Carboxylic Acid", "Inorganic Ion"): 1,  ("Carboxylic Acid", "Imide"): 1,
    ("Amine", "Carboxylic Acid"): 2,          ("Amine", "Sulfonate"): 2,
    ("Amine", "Inorganic Ion"): 2,            ("Amine", "Amide"): 1,
    ("Amine", "Imide"): 1,
    ("Pyridine", "Carboxylic Acid"): 2,       ("Pyridine", "Phenol"): 2,
    ("Pyridine", "Imide"): 2,                 ("Pyridine", "Amide"): 1,
    ("Amide", "Carboxylic Acid"): 2,          ("Amide", "Pyridine/Amide"): 2,
    ("Amide", "Amide"): 1,                    ("Amide", "Inorganic Ion"): 1,
    ("Phenol", "Pyridine/Amide"): 2,          ("Phenol", "Amine"): 2,
    ("Phenol", "Amide"): 1,
    ("Sulfonate", "Amine"): 2,                ("Sulfonate", "Inorganic Ion"): 1,
    ("Hydroxyl", "Carboxylic Acid"): 2,       ("Hydroxyl", "Amide"): 1,
    ("Hydroxyl", "Pyridine/Amide"): 1,
}

def etter_score(api_s, cf_s):
    return etter_scores.get((api_s, cf_s), 0)

# ─────────────────────────────────────────────
# ENHANCED SCIENTIFIC MODELS
# ─────────────────────────────────────────────

def cruz_cabeza_pka_probability(delta_pka):
    """
    Cruz-Cabeza (CrystEngComm 2012, 14, 6362) three-zone classification:
    - ΔpKa < -1:  Cocrystal zone   — P(salt) ≈ 0-5%
    - -1 ≤ ΔpKa ≤ 4:  Continuum zone — sigmoidal transition
    - ΔpKa > 4:   Salt zone — P(salt) > 95%
    
    Fitted to Fig. 5 of Cruz-Cabeza's 6465 structure analysis.
    Steeper sigmoid (k=1.8) centred at ΔpKa=2.0 fits the empirical distribution.
    """
    if delta_pka < -1:
        return round(max(0.02, 0.05 * np.exp(delta_pka + 1)), 3)
    elif delta_pka > 6:
        return 0.99
    else:
        # Sigmoid fitted to CSD data: inflection at ΔpKa ≈ 2, slope k ≈ 1.8
        return round(1.0 / (1.0 + np.exp(-1.8 * (delta_pka - 2.0))), 3)


def yalkowsky_gse(logP, mp_celsius):
    """
    Yalkowsky General Solubility Equation (GSE, 2001):
        log S₀ (mol/L) = 0.5 − 0.01·(mp − 25) − logP
    
    Estimates intrinsic aqueous solubility of the un-ionized free form.
    Valid for non-electrolytes and free-form APIs.
    Returns S₀ in mg/mL if MW provided, else mol/L.
    """
    if logP is None:
        return None
    mp = mp_celsius if mp_celsius is not None else 150  # default estimate
    log_s0 = 0.5 - 0.01 * (mp - 25) - logP
    return round(10 ** log_s0, 6)  # mol/L


def dose_number(dose_mg, solubility_mg_ml, v_gi_ml=250):
    """
    Dose Number (D₀) — Amidon et al., Pharm. Res. 1995:
        D₀ = Dose / (Cs × V_GI)
    D₀ > 1 ⟹ dose unlikely to dissolve completely ⟹ solubility-limited absorption.
    """
    if solubility_mg_ml is None or solubility_mg_ml <= 0:
        return None
    return round(dose_mg / (solubility_mg_ml * v_gi_ml), 2)


def maximum_absorbable_dose(solubility_mg_ml, peff_cm_s=None):
    """
    Löbenberg-Amidon MAD estimation:
        MAD (mg) = S × K_a × V_GI × T_si
    where K_a = 2 × P_eff / R_intestine (simplified)
    
    Crude estimate — useful for rank ordering, not for quantitative prediction.
    """
    if solubility_mg_ml is None:
        return None
    if peff_cm_s is None:
        peff_cm_s = 1.0e-4  # moderate permeability
    ka = 2 * peff_cm_s / 1.75  # R_intestine ≈ 1.75 cm
    mad = solubility_mg_ml * ka * V_GI * T_RESIDENCE
    return round(mad, 1)


def hh_solubility_at_pH(pka, mol_type, pH, s0_mol_l=None, mw=None):
    """
    Henderson-Hasselbalch equation for pH-dependent solubility:
        Bases: S_total = S₀ × (1 + 10^(pKa − pH))
        Acids: S_total = S₀ × (1 + 10^(pH − pKa))
    
    Returns fold-advantage (S_total/S₀) and absolute solubility if S₀ known.
    """
    if mol_type == "Base":
        ratio = 1.0 + 10 ** (pka - pH)
    else:
        ratio = 1.0 + 10 ** (pH - pka)
    ratio = max(ratio, 1.0)
    
    abs_sol = None
    if s0_mol_l is not None and mw is not None:
        abs_sol = round(s0_mol_l * ratio * mw, 4)  # mg/mL
    
    return round(ratio, 1), abs_sol


def common_ion_correction_hcl(base_pka, gastric_cl_M=0.05):
    """
    Common-ion effect for HCl salts in gastric fluid.
    
    In gastric HCl, Cl⁻ concentration ≈ 0.034-0.05 M, which can suppress
    dissolution of HCl salts via Ksp limitation.
    
    Returns a penalty factor (0-1) where 1 = no penalty, 0 = complete suppression.
    A real Ksp calculation requires experimental data; this is a heuristic flag.
    """
    # Higher base pKa = more ionized at gastric pH = more Cl⁻ generated = worse
    if base_pka > 8:
        return 0.65  # significant suppression risk
    elif base_pka > 5:
        return 0.80
    else:
        return 0.95  # minimal risk


def flory_huggins_chi(api_hsp, cf_hsp, temperature_k=298.15):
    """
    Flory-Huggins interaction parameter:
        χ = V_ref / (R·T) × Σ(δᵢ_API − δᵢ_CF)²
    
    Marsac et al., Pharm. Res. 2006.
    """
    dd_a, dp_a, dh_a = api_hsp
    dd_c, dp_c, dh_c = cf_hsp
    delta_sq = (dd_a - dd_c)**2 + (dp_a - dp_c)**2 + (dh_a - dh_c)**2
    chi = (V_REF / (R_GAS * temperature_k)) * delta_sq
    return round(chi, 3)


def frac_ionized(pka, mol_type, pH):
    """Fraction ionized at given pH (Henderson-Hasselbalch)."""
    if mol_type == "Base":
        return round(1.0 / (1.0 + 10 ** (pH - pka)), 4)
    else:
        return round(1.0 / (1.0 + 10 ** (pka - pH)), 4)


def gastric_survival(api_pka, api_type, cf_pka):
    """Predict whether the salt form will survive gastric pH without disproportionation."""
    if api_type == "Base":
        d = api_pka - cf_pka
        if d > 5: return "Stable"
        elif d > 2: return "Marginal"
        else: return "Risk"
    else:
        d = cf_pka - api_pka
        if d > 4: return "Stable"
        elif d > 1.5: return "Marginal"
        else: return "Risk"


def hygro_penalty(hygro_risk, api_pka, api_type):
    """Hygroscopicity risk penalty with HCl-base correction."""
    base = {"High": 0.80, "Medium": 0.40, "Low": 0.05}[hygro_risk]
    if hygro_risk == "High" and api_type == "Base" and api_pka > 8:
        base = min(base + 0.15, 1.0)
    return base


def gordon_taylor_tg(tg_api_c, tg_pol_k, w_api):
    """Gordon-Taylor Tg_mix prediction."""
    if tg_api_c is None or tg_pol_k is None or tg_pol_k <= 0:
        return None
    tg_api_k = tg_api_c + 273.15
    w_pol = 1.0 - w_api
    if w_pol <= 0:
        return round(tg_api_c, 1)
    K_gt = tg_api_k / tg_pol_k
    tg_mix_k = (w_api * tg_api_k + K_gt * w_pol * tg_pol_k) / (w_api + K_gt * w_pol)
    return round(tg_mix_k - 273.15, 1)


def kauzmann_temperature(tg_mix_c):
    """
    Kauzmann temperature Tₖ ≈ Tg − 50 K.
    Below Tₖ, the amorphous system has negligible molecular mobility and is considered
    kinetically trapped. Storage below Tₖ provides maximum stability.
    """
    if tg_mix_c is None:
        return None
    return round(tg_mix_c - 50, 1)


def asd_stability_class(tg_mix_c, storage_temp_c=25.0):
    """ASD stability classification: Tg_mix − T_storage rule."""
    if tg_mix_c is None:
        return "N/A"
    delta = tg_mix_c - storage_temp_c
    if delta >= 50:
        return "Stable (Tg−Ts ≥ 50°C)"
    elif delta >= 30:
        return "Borderline (30-50°C)"
    else:
        return "Unstable (< 30°C)"


def supersaturation_risk_index(delta_pka, inter_type, etter_val, api_logP=None):
    """
    Supersaturation Risk Index (SRI).
    Enhanced: high logP APIs have higher precipitation kinetics risk.
    """
    if inter_type == "Salt / Cocrystal Continuum":
        base = 3 if delta_pka < 1.0 else (2 if delta_pka < 2.0 else 1)
        base -= min(etter_val, 1)  # strong synthon reduces risk
        if api_logP and api_logP > 3:
            base += 1  # lipophilic APIs precipitate faster
        if base >= 3: return "High"
        elif base >= 2: return "Medium"
        else: return "Low"
    elif inter_type == "Unlikely":
        return "High"
    else:
        return "N/A"


def spring_parachute_index(sri_label, api_logP, cf_type):
    """
    Spring-and-Parachute Index: estimates need for precipitation inhibitor.
    
    Spring: salt/cocrystal provides initial supersaturation (spring)
    Parachute: polymer maintains supersaturation during absorption window
    
    Returns: recommendation level and suggested polymer.
    """
    if sri_label == "N/A" or sri_label == "Low":
        return "Not Required", "-"
    
    if "Polymer" in cf_type:
        return "Built-in (ASD)", cf_type
    
    if sri_label == "High":
        rec = "Critical — add precipitation inhibitor"
        polymer = "HPMCAS or PVP (0.5-2% w/v in dissolution medium)"
    else:
        rec = "Recommended — monitor dissolution"
        polymer = "HPMC or PVP (low concentration)"
    
    if api_logP and api_logP > 4:
        rec += "; high logP = fast nucleation"
    
    return rec, polymer


def lattice_energy_proxy(mp_celsius):
    """
    Melting point as a proxy for lattice energy.
    Higher mp ⟹ stronger crystal packing ⟹ lower solubility but higher physical stability.
    Pudipeddi & Serajuddin, J. Pharm. Sci. 2005.
    """
    if mp_celsius is None:
        return "Unknown"
    if mp_celsius > 250:
        return "Very High (mp>250°C)"
    elif mp_celsius > 180:
        return "High (mp 180-250°C)"
    elif mp_celsius > 120:
        return "Moderate (mp 120-180°C)"
    else:
        return "Low (mp<120°C)"


def composite_lead_score(delta_pka, etter_val, ra, chi, hygro_risk, ich_flag,
                         api_pka, api_type, cf_type, sri_label):
    """
    Composite Lead Score (0-100). Scientifically weighted:
    
    Component         Weight  Rationale
    ─────────────────────────────────────────────
    ΔpKa probability    30    Primary thermodynamic driver (Cruz-Cabeza 2012)
    Synthon match       20    Supramolecular complementarity (Etter 1990)
    Miscibility (Ra/χ)  20    Solubility parameter compatibility (Hansen/Marsac)
    Hygroscopicity      15    Critical for stability and manufacturing
    ICH M7 safety       10    Regulatory risk (binary)
    Supersaturation      5    Dissolution robustness (NEW in v6)
    ─────────────────────────────────────────────
    Total              100
    """
    p_salt = cruz_cabeza_pka_probability(delta_pka)
    score_pka = p_salt * 30
    
    score_synthon = (etter_val / 2) * 20
    
    if "Polymer" in cf_type:
        if chi is not None:
            score_misc = 20 if chi < 0.5 else (12 if chi < 1.0 else 2)
        else:
            score_misc = 10
    else:
        score_misc = 20 if ra < 5 else (12 if ra < 7 else (5 if ra < 10 else 0))
    
    pen = hygro_penalty(hygro_risk, api_pka, api_type)
    score_hygro = (1 - pen) * 15
    
    score_ich = 0 if ich_flag else 10
    
    sri_score = {"N/A": 5, "Low": 5, "Medium": 2.5, "High": 0}.get(sri_label, 2.5)
    
    total = score_pka + score_synthon + score_misc + score_hygro + score_ich + sri_score
    return int(round(min(total, 100)))


# ─────────────────────────────────────────────
# CHART GENERATORS
# ─────────────────────────────────────────────

def fig_to_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf


def chart_lead_scores(df_full):
    """Chart 1: Horizontal bar — Top 20 Lead Scores."""
    df = df_full.nlargest(20, 'Lead Score').sort_values('Lead Score')
    fig, ax = plt.subplots(figsize=(11, 7.5))
    bar_colors = []
    for s in df['Lead Score']:
        if s >= 70:   bar_colors.append(MC['green'])
        elif s >= 50: bar_colors.append(MC['primary'])
        elif s >= 30: bar_colors.append(MC['amber'])
        else:         bar_colors.append(MC['red'])
    bars = ax.barh(range(len(df)), df['Lead Score'], color=bar_colors,
                   edgecolor='white', linewidth=0.5, height=0.72)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Partner'], fontsize=8.5)
    for i, (bar, score, (_, row)) in enumerate(zip(bars, df['Lead Score'], df.iterrows())):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{score}', va='center', ha='left', fontsize=8.5, fontweight='bold', color='#333')
        if row.get('ICH M7 Flag') == 'Yes':
            ax.text(1, bar.get_y() + bar.get_height()/2,
                    'ICH!', va='center', ha='left', fontsize=7, color=MC['amber'], fontweight='bold')
    ax.axvline(x=60, color=MC['green'], linestyle='--', lw=1.5, alpha=0.8, label='Lead threshold (60)')
    ax.axvline(x=70, color='#1B5E20', linestyle=':', lw=1.5, alpha=0.8, label='Strong lead (70)')
    patches = [
        mpatches.Patch(color=MC['green'], label='Score >= 70 (Strong)'),
        mpatches.Patch(color=MC['primary'], label='Score 50-69 (Lead)'),
        mpatches.Patch(color=MC['amber'], label='Score 30-49 (Weak)'),
        mpatches.Patch(color=MC['red'], label='Score < 30 (Unlikely)'),
    ]
    ax.legend(handles=patches, fontsize=8, loc='lower right', framealpha=0.9)
    ax.set_xlabel('Composite Lead Score (0-100)', fontweight='bold')
    ax.set_title('Ranked Partner Lead Scores (Top 20)', pad=14)
    ax.set_xlim(0, 112)
    fig.tight_layout(pad=1.5)
    return fig_to_bytes(fig)


def chart_score_components(df_full):
    """Chart 2: Stacked bar showing score breakdown (Top 15)."""
    df = df_full.nlargest(15, 'Lead Score').sort_values('Lead Score', ascending=True)
    pka_s, syn_s, misc_s, hyg_s, ich_s, sri_s = [], [], [], [], [], []
    for _, row in df.iterrows():
        p = cruz_cabeza_pka_probability(row['Delta pKa'])
        pka_s.append(p * 30)
        syn_s.append((row['Etter Score'] / 2) * 20)
        chi_val = row.get('chi (F-H)', '-')
        ra_val = pd.to_numeric(row['Ra'], errors='coerce') if str(row['Ra']) != '-' else None
        if 'Polymer' in str(row['Type']) and str(chi_val) != '-':
            c = float(chi_val)
            misc_s.append(20 if c < 0.5 else (12 if c < 1.0 else 2))
        elif ra_val is not None and not np.isnan(ra_val):
            misc_s.append(20 if ra_val < 5 else (12 if ra_val < 7 else (5 if ra_val < 10 else 0)))
        else:
            misc_s.append(10)
        hyg_s.append({'Low': 14.25, 'Medium': 9, 'High': 3}[row['Hygro Risk']])
        ich_s.append(0 if row.get('ICH M7 Flag') == 'Yes' else 10)
        sri_s.append({"N/A": 5, "Low": 5, "Medium": 2.5, "High": 0}.get(row.get('SRI', 'N/A'), 2.5))

    fig, ax = plt.subplots(figsize=(11, 7.5))
    partners = df['Partner'].tolist()
    y = np.arange(len(partners))
    ax.barh(y, pka_s, color=MC['primary'], label='pKa (max 30)', height=0.65)
    l1 = pka_s
    ax.barh(y, syn_s, left=l1, color=MC['accent'], label='Synthon (max 20)', height=0.65)
    l2 = [a+b for a, b in zip(l1, syn_s)]
    ax.barh(y, misc_s, left=l2, color='#4CAF50', label='Miscibility (max 20)', height=0.65)
    l3 = [a+b for a, b in zip(l2, misc_s)]
    ax.barh(y, hyg_s, left=l3, color=MC['amber'], label='Hygro (max 15)', height=0.65)
    l4 = [a+b for a, b in zip(l3, hyg_s)]
    ax.barh(y, ich_s, left=l4, color=MC['purple'], label='ICH (max 10)', height=0.65)
    l5 = [a+b for a, b in zip(l4, ich_s)]
    ax.barh(y, sri_s, left=l5, color=MC['grey'], label='SRI (max 5)', height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(partners, fontsize=8.5)
    ax.axvline(x=60, color=MC['red'], linestyle='--', lw=1.5, alpha=0.8, label='Lead threshold (60)')
    ax.set_xlabel('Composite Score (0-100)', fontweight='bold')
    ax.set_title('Lead Score Component Breakdown (Top 15)', pad=14)
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
    ax.set_xlim(0, 108)
    fig.tight_layout(pad=1.5)
    return fig_to_bytes(fig)


def chart_landscape(df_full):
    """Chart 3: Scatter — DeltapKa vs Ra coloured by Interaction Type."""
    itype_colors = {
        'Salt': MC['primary'], 'Salt / Cocrystal Continuum': MC['amber'],
        'Cocrystal': MC['green'], 'Unlikely': MC['red'],
    }
    fig, ax = plt.subplots(figsize=(11, 7))
    for itype, grp in df_full.groupby('Interaction Type'):
        color = itype_colors.get(itype, MC['grey'])
        ra_vals = pd.to_numeric(grp['Ra'], errors='coerce')
        sizes = (grp['Lead Score'] / 100 * 280 + 50).values
        ax.scatter(grp['Delta pKa'], ra_vals, c=color, s=sizes,
                   alpha=0.75, label=itype, edgecolors='white', linewidth=0.6)
    ax.axhline(y=7, color='red', linestyle='--', lw=1.5, alpha=0.7, label='Ra = 7')
    ax.axvline(x=4, color='blue', linestyle='--', lw=1.5, alpha=0.7, label='pKa = 4 (Cruz-Cabeza salt)')
    ax.axvline(x=-1, color='grey', linestyle=':', lw=0.8, alpha=0.5, label='pKa = -1 (cocrystal)')
    ax.axvspan(-1, 4, alpha=0.05, color=MC['amber'], label='Continuum zone')
    top_leads = df_full.nlargest(8, 'Lead Score')
    for _, row in top_leads.iterrows():
        ra_v = pd.to_numeric(row['Ra'], errors='coerce')
        if not pd.isna(ra_v):
            ax.annotate(row['Partner'], xy=(row['Delta pKa'], ra_v),
                        xytext=(5, 4), textcoords='offset points', fontsize=7,
                        color='#333', arrowprops=dict(arrowstyle='-', color='#CCC', lw=0.5))
    ax.set_xlabel('Delta-pKa (Cruz-Cabeza classification)', fontweight='bold')
    ax.set_ylabel('Hansen Distance Ra (MPa^0.5)', fontweight='bold')
    ax.set_title('Physicochemical Landscape: Delta-pKa vs Hansen Distance\n(Bubble size ~ Lead Score; shaded = continuum zone)', pad=14)
    ax.legend(fontsize=8, framealpha=0.9, loc='upper right')
    fig.tight_layout(pad=1.5)
    return fig_to_bytes(fig)


def chart_interaction_distribution(df_full):
    """Chart 4: Interaction type + Etter Score distribution."""
    itype_colors = {
        'Salt': MC['primary'], 'Salt / Cocrystal Continuum': MC['amber'],
        'Cocrystal': MC['green'], 'Unlikely': MC['red'],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ax1 = axes[0]
    ic = df_full['Interaction Type'].value_counts()
    lbls, vals = ic.index.tolist(), ic.values.tolist()
    clrs = [itype_colors.get(l, MC['grey']) for l in lbls]
    bars = ax1.bar(range(len(lbls)), vals, color=clrs, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
                 str(v), ha='center', fontsize=11, fontweight='bold', color='#333')
    ax1.set_xticks(range(len(lbls)))
    ax1.set_xticklabels([l.replace(' / ', '\n/ ') for l in lbls], fontsize=8.5)
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Predicted Interaction Types\n(Cruz-Cabeza 2012 classification)', pad=10)

    ax2 = axes[1]
    ec = df_full['Etter Score'].value_counts().sort_index()
    e_colors = {0: MC['red'], 1: MC['amber'], 2: MC['green']}
    e_labels = {0: '0 — None', 1: '1 — Moderate', 2: '2 — Strong'}
    bars2 = ax2.bar([e_labels.get(k, '') for k in ec.index], ec.values,
                    color=[e_colors.get(k, MC['grey']) for k in ec.index],
                    edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars2, ec.values):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
                 str(v), ha='center', fontsize=11, fontweight='bold', color='#333')
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title("Etter's Synthon Score Distribution", pad=10)
    fig.tight_layout(pad=1.5)
    return fig_to_bytes(fig)


def chart_radar_top3(df_full):
    """Chart 5: Radar chart for Top 3 leads."""
    top3 = df_full.head(3)
    if top3.empty:
        return None
    categories = ['pKa\nProb', 'Synthon', 'Miscibility', 'Hygro\nSafety', 'ICH\nSafety', 'SRI\nScore']
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
    r_colors = [MC['primary'], MC['green'], MC['amber']]
    for i, (_, row) in enumerate(top3.iterrows()):
        p_s = cruz_cabeza_pka_probability(row['Delta pKa']) * 100
        syn_s = (row['Etter Score'] / 2) * 100
        misc_lbl = str(row['Miscibility'])
        if 'Miscible' in misc_lbl or misc_lbl == 'High': misc_v = 100
        elif 'Borderline' in misc_lbl or misc_lbl == 'Low': misc_v = 40
        else: misc_v = 10
        hyg_v = {'Low': 100, 'Medium': 55, 'High': 15}[row['Hygro Risk']]
        ich_v = 0 if row.get('ICH M7 Flag') == 'Yes' else 100
        sri_v = {"N/A": 100, "Low": 100, "Medium": 50, "High": 0}.get(row.get('SRI', 'N/A'), 50)
        vals = [p_s, syn_s, misc_v, hyg_v, ich_v, sri_v]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', lw=2.2, color=r_colors[i % 3],
                label=f"{row['Partner']} ({row['Lead Score']})", markersize=6)
        ax.fill(angles, vals, alpha=0.12, color=r_colors[i % 3])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=7.5, color='grey')
    ax.set_title('Score Component Radar — Top 3 Partners', pad=22, fontweight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), fontsize=8.5, ncol=1, framealpha=0.92)
    fig.tight_layout(pad=1.5)
    return fig_to_bytes(fig)


def chart_chi_bars(df_poly):
    """Chart 6: Flory-Huggins chi for polymers."""
    if df_poly.empty:
        return None
    df_p = df_poly.copy()
    df_p['chi_num'] = pd.to_numeric(df_p['chi (F-H)'], errors='coerce')
    df_p = df_p.dropna(subset=['chi_num']).sort_values('chi_num')
    if df_p.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bar_colors = [MC['green'] if c < 0.5 else (MC['amber'] if c < 1.0 else MC['red'])
                  for c in df_p['chi_num']]
    bars = ax.bar(df_p['Partner'], df_p['chi_num'], color=bar_colors,
                  edgecolor='white', linewidth=0.5, width=0.6)
    for bar, v in zip(bars, df_p['chi_num']):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                f'{v:.3f}', ha='center', fontsize=10, fontweight='bold', color='#333')
    ax.axhline(y=0.5, color=MC['amber'], linestyle='--', lw=1.8, alpha=0.9, label='chi=0.5 (miscibility)')
    ax.axhline(y=1.0, color=MC['red'], linestyle='--', lw=1.8, alpha=0.9, label='chi=1.0 (immiscible)')
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_ylabel('Flory-Huggins chi', fontweight='bold')
    ax.set_title('Polymer-API Miscibility (chi)', pad=12)
    ax.set_ylim(bottom=0)
    fig.tight_layout(pad=1.5)
    return fig_to_bytes(fig)


def chart_hh_curve(api_pka, api_type, api_name, s0_mol_l, api_mw):
    """Chart 7: Henderson-Hasselbalch solubility vs pH."""
    pH_sweep = np.linspace(0.5, 10.8, 350)
    sol_ratio = [hh_solubility_at_pH(api_pka, api_type, ph)[0] for ph in pH_sweep]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.semilogy(pH_sweep, sol_ratio, color=MC['primary'], lw=2.5, label='S_total / S_0')
    ax.fill_between(pH_sweep, 1, sol_ratio, alpha=0.10, color=MC['primary'])
    ax.axhline(y=10, color=MC['green'], linestyle='--', lw=1.5, alpha=0.85, label='10x threshold')
    ax.axhline(y=100, color=MC['amber'], linestyle='--', lw=1.2, alpha=0.75, label='100x')
    ax.axhline(y=1000, color=MC['red'], linestyle=':', lw=1.0, alpha=0.55, label='1000x')
    bio_phs = [(1.2, MC['amber'], 'Gastric'), (5.0, MC['purple'], 'FeSSIF'),
               (6.5, MC['primary'], 'FaSSIF'), (7.4, MC['green'], 'Colonic')]
    for ph_v, col, lbl in bio_phs:
        ratio = hh_solubility_at_pH(api_pka, api_type, ph_v)[0]
        ax.axvline(x=ph_v, color=col, linestyle=':', lw=1.8, alpha=0.7)
        ax.text(ph_v + 0.07, ratio * 1.5, f'{lbl}\npH {ph_v}\n{ratio:.0f}x',
                ha='left', fontsize=7.5, color=col, fontweight='bold')
    if s0_mol_l:
        ax2 = ax.twinx()
        abs_sol = [s0_mol_l * r * (api_mw if api_mw else 300) for r in sol_ratio]
        ax2.semilogy(pH_sweep, abs_sol, color=MC['accent'], lw=1.5, alpha=0.5, linestyle='--')
        ax2.set_ylabel('Estimated Absolute Solubility (mg/mL)', color=MC['accent'], fontweight='bold')
    ax.set_xlabel('pH', fontweight='bold')
    ax.set_ylabel('S_total / S_0 (fold advantage)', fontweight='bold')
    title_extra = f'  |  GSE S_0 = {s0_mol_l:.2e} mol/L' if s0_mol_l else ''
    ax.set_title(f'Henderson-Hasselbalch Solubility Advantage\n{api_name}  |  pKa={api_pka}  |  {api_type}{title_extra}', pad=14)
    ax.legend(fontsize=8.5, loc='upper right' if api_type=='Base' else 'upper left', framealpha=0.9)
    ax.set_xlim(0.5, 10.8)
    fig.tight_layout(pad=1.5)
    return fig_to_bytes(fig)


def chart_ionization_profile(api_pka, api_type, api_name):
    """Chart 8: Ionization profile."""
    pH_vals = np.linspace(0, 12.5, 350)
    if api_type == 'Base':
        fi = [1.0 / (1.0 + 10 ** (ph - api_pka)) * 100 for ph in pH_vals]
    else:
        fi = [1.0 / (1.0 + 10 ** (api_pka - ph)) * 100 for ph in pH_vals]
    fu = [100 - v for v in fi]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(pH_vals, fi, color=MC['primary'], lw=2.5, label='Ionized (charged, soluble)')
    ax.plot(pH_vals, fu, color=MC['amber'], lw=2.5, ls='--', label='Unionized (neutral, permeable)')
    ax.fill_between(pH_vals, fi, alpha=0.10, color=MC['primary'])
    ax.axhline(y=50, color='grey', ls=':', lw=0.8, alpha=0.5)
    ax.axvline(x=api_pka, color='black', ls='-', lw=1.8, alpha=0.7, label=f'pKa = {api_pka}')
    bio_phs = [(1.2, MC['amber'], 'Gastric'), (5.0, MC['purple'], 'FeSSIF'),
               (6.5, MC['primary'], 'FaSSIF'), (7.4, MC['green'], 'Colonic')]
    for ph_v, col, lbl in bio_phs:
        if api_type == 'Base':
            fi_pt = 1.0/(1.0+10**(ph_v-api_pka))*100
        else:
            fi_pt = 1.0/(1.0+10**(api_pka-ph_v))*100
        ax.axvline(x=ph_v, color=col, ls=':', lw=1.5, alpha=0.65)
        offset = 8 if fi_pt < 85 else -15
        ax.annotate(f'{lbl}\n{fi_pt:.0f}% ion.', xy=(ph_v, fi_pt),
                    xytext=(ph_v+0.18, fi_pt+offset), fontsize=7.5, color=col, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color=col, lw=0.6))
    ax.set_xlabel('pH', fontweight='bold')
    ax.set_ylabel('Species Fraction (%)', fontweight='bold')
    ax.set_title(f'Ionization Profile — {api_name} (pKa={api_pka}, {api_type})', pad=14)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(-2, 105)
    fig.tight_layout(pad=1.5)
    return fig_to_bytes(fig)


def chart_hygro_donut(df_full):
    """Chart 9: Hygroscopicity risk donut."""
    hygro_c = df_full['Hygro Risk'].value_counts()
    lbls = [k for k in ['High', 'Medium', 'Low'] if k in hygro_c.index]
    sizes = [hygro_c[k] for k in lbls]
    clrs = {'High': MC['red'], 'Medium': MC['amber'], 'Low': MC['green']}
    fig, ax = plt.subplots(figsize=(7, 5.5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=lbls, colors=[clrs[k] for k in lbls],
        autopct='%1.0f%%', startangle=90,
        wedgeprops=dict(width=0.58, edgecolor='white', linewidth=2.5),
        pctdistance=0.75, textprops={'fontsize': 10, 'fontweight': 'bold'})
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight('bold'); at.set_color('white')
    ax.add_artist(plt.Circle((0,0), 0.35, fc='white'))
    ax.text(0, 0.05, str(len(df_full)), ha='center', va='center',
            fontsize=20, fontweight='bold', color='#333')
    ax.text(0, -0.2, 'Partners', ha='center', va='center', fontsize=8.5, color='#666')
    ax.set_title('Hygroscopicity Risk Distribution', pad=12, fontweight='bold')
    fig.tight_layout(pad=1.5)
    return fig_to_bytes(fig)


def chart_polymorph_risk(df_full):
    """Chart 10: Polymorphism risk by category."""
    risk_order = ['Low', 'Medium', 'High']
    by_type = df_full.groupby(['Type', 'Polymorphism Risk']).size().unstack(fill_value=0)
    for r in risk_order:
        if r not in by_type.columns:
            by_type[r] = 0
    by_type = by_type[risk_order]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(by_type))
    w = 0.25
    clrs = [MC['green'], MC['amber'], MC['red']]
    for i, (risk, col) in enumerate(zip(risk_order, clrs)):
        bars = ax.bar(x + (i-1)*w, by_type[risk], w, label=f'{risk}', color=col,
                      edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x()+bar.get_width()/2, h+0.08,
                        str(int(h)), ha='center', fontsize=8.5, fontweight='bold', color='#333')
    ax.set_xticks(x)
    ax.set_xticklabels(by_type.index, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Polymorphism Risk by Partner Category', pad=14)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout(pad=1.5)
    return fig_to_bytes(fig)


def chart_dose_solubility(api_name, dose_mg, s0_mol_l, api_mw, api_pka, api_type):
    """Chart 11 (NEW): Dose number vs pH showing BCS boundary."""
    if s0_mol_l is None or api_mw is None:
        return None
    pH_sweep = np.linspace(0.5, 10, 250)
    d0_vals = []
    for ph in pH_sweep:
        ratio, _ = hh_solubility_at_pH(api_pka, api_type, ph)
        s_ph_mg_ml = s0_mol_l * ratio * api_mw
        d0 = dose_mg / (s_ph_mg_ml * V_GI) if s_ph_mg_ml > 0 else 999
        d0_vals.append(min(d0, 1000))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogy(pH_sweep, d0_vals, color=MC['primary'], lw=2.5)
    ax.fill_between(pH_sweep, 1, d0_vals, where=[d > 1 for d in d0_vals],
                    alpha=0.1, color=MC['red'])
    ax.fill_between(pH_sweep, [min(d, 1) for d in d0_vals], d0_vals,
                    where=[d <= 1 for d in d0_vals], alpha=0.1, color=MC['green'])
    ax.axhline(y=1, color=MC['red'], linestyle='--', lw=2, alpha=0.8, label='D0=1 (BCS solubility limit)')
    bio_phs = [(1.2, 'Gastric'), (5.0, 'FeSSIF'), (6.5, 'FaSSIF'), (7.4, 'Colonic')]
    for ph_v, lbl in bio_phs:
        ratio, _ = hh_solubility_at_pH(api_pka, api_type, ph_v)
        s_ph = s0_mol_l * ratio * api_mw
        d0_pt = dose_mg / (s_ph * V_GI) if s_ph > 0 else 999
        ax.plot(ph_v, min(d0_pt, 1000), 'o', color=MC['amber'], markersize=8, zorder=5)
        ax.annotate(f'{lbl}\nD0={d0_pt:.1f}', xy=(ph_v, min(d0_pt, 1000)),
                    xytext=(5, 10), textcoords='offset points', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('pH', fontweight='bold')
    ax.set_ylabel('Dose Number D0 (log scale)', fontweight='bold')
    ax.set_title(f'Dose Number vs pH — {api_name} ({dose_mg} mg)\nD0 > 1 = solubility-limited; salt/cocrystal needed', pad=14)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim(0.5, 10)
    fig.tight_layout(pad=1.5)
    return fig_to_bytes(fig)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("1. API Identity & Ionization")
api_name = st.sidebar.text_input("API Name", value="Target_Molecule")
api_pka = st.sidebar.number_input("API pKa", value=7.0, step=0.1)
api_type = st.sidebar.selectbox("API Nature", ["Base", "Acid"])
api_synthon = st.sidebar.selectbox("Primary Functional Group",
    ["Carboxylic Acid", "Amine", "Pyridine", "Amide", "Phenol", "Sulfonate", "Hydroxyl"])

st.sidebar.header("2. Hansen Solubility Parameters (MPa^0.5)")
api_dd = st.sidebar.number_input("Dispersion (dd)", value=18.50)
api_dp = st.sidebar.number_input("Polar (dp)", value=10.50)
api_dh = st.sidebar.number_input("H-Bonding (dh)", value=7.50)

st.sidebar.header("3. Physicochemical Properties")
api_logP = st.sidebar.number_input("API logP", value=2.0, step=0.1)
api_mw = st.sidebar.number_input("API MW (g/mol)", value=350.0, step=10.0,
    help="Molecular weight — required for absolute solubility and dose number calculations")
api_mp = st.sidebar.number_input("API Melting Point (°C)", value=180.0, step=5.0,
    help="Used in Yalkowsky GSE for intrinsic solubility estimation")
api_tg = st.sidebar.number_input("API Glass Transition Tg (°C)", value=100.0, step=1.0,
    help="For Gordon-Taylor ASD Tg_mix prediction")
api_loading = st.sidebar.slider("ASD Drug Loading (%)", 5, 60, 30) / 100.0
dose_mg = st.sidebar.number_input("Dose (mg)", value=100, step=10,
    help="For Dose Number (D0) and MAD calculations")

st.sidebar.header("4. Biorelevant pH Targets")
target_pH_str = st.sidebar.multiselect(
    "Report solubility advantage at:",
    ["Gastric (pH 1.2)", "FaSSIF (pH 6.5)", "FeSSIF (pH 5.0)", "Colonic (pH 7.4)"],
    default=["FaSSIF (pH 6.5)"])
pH_map = {"Gastric (pH 1.2)": PH_GASTRIC, "FaSSIF (pH 6.5)": PH_FASSIF,
           "FeSSIF (pH 5.0)": PH_FESSIF, "Colonic (pH 7.4)": PH_COLONIC}

st.sidebar.header("5. Display Filters")
min_score = st.sidebar.slider("Minimum Lead Score", 0, 100, 30)
show_ich = st.sidebar.checkbox("Hide ICH-flagged counterions", value=False)
show_all_ra = st.sidebar.checkbox("Include low-miscibility partners (Ra > 7)", value=False)

# ─────────────────────────────────────────────
# CORE COMPUTATION ENGINE
# ─────────────────────────────────────────────
api_hsp = (api_dd, api_dp, api_dh)

# Yalkowsky GSE intrinsic solubility
s0_mol_l = yalkowsky_gse(api_logP, api_mp)
s0_mg_ml = round(s0_mol_l * api_mw, 4) if s0_mol_l and api_mw else None

# HH results at target pH
hh_results = {}
for lbl in target_pH_str:
    ratio, abs_sol = hh_solubility_at_pH(api_pka, api_type, pH_map[lbl], s0_mol_l, api_mw)
    hh_results[lbl] = {"ratio": ratio, "abs_sol_mg_ml": abs_sol}

# Dose number at FaSSIF
d0_fassif = None
if s0_mg_ml:
    ratio_fassif, _ = hh_solubility_at_pH(api_pka, api_type, PH_FASSIF)
    s_fassif = s0_mg_ml * ratio_fassif
    d0_fassif = dose_number(dose_mg, s_fassif)

# BCS classification
bcs_hint = "Unknown"
if d0_fassif is not None:
    if d0_fassif <= 1:
        bcs_hint = f"BCS I or III (D0={d0_fassif:.2f} — high solubility)"
    else:
        bcs_hint = f"BCS II or IV (D0={d0_fassif:.1f} — low solubility, salt/cocrystal needed)"
is_sol_limited = d0_fassif is not None and d0_fassif > 1

# MAD estimate
mad_est = maximum_absorbable_dose(s0_mg_ml)

# ── Screen all coformers ──
results = []
for cf in coformers_db:
    delta_pka = (api_pka - cf["pKa"]) if api_type == "Base" else (cf["pKa"] - api_pka)
    p_form = cruz_cabeza_pka_probability(delta_pka)
    e_score = etter_score(api_synthon, cf["synthon"])
    cf_hsp = (cf["dd"], cf["dp"], cf["dh"])
    ra = np.sqrt(4*(api_dd-cf["dd"])**2 + (api_dp-cf["dp"])**2 + (api_dh-cf["dh"])**2)
    chi = flory_huggins_chi(api_hsp, cf_hsp) if "Polymer" in cf["type"] else None

    # Cruz-Cabeza classification (updated boundaries)
    if delta_pka > 4:
        inter_type = "Salt"
    elif -1 <= delta_pka <= 4:
        inter_type = "Salt / Cocrystal Continuum"
    elif e_score > 0:
        inter_type = "Cocrystal"
    else:
        inter_type = "Unlikely"

    if "Polymer" in cf["type"]:
        misc_label = ("Miscible (chi<0.5)" if chi and chi < 0.5 else
                      ("Borderline (0.5-1)" if chi and chi < 1.0 else
                       "Immiscible (chi>1)")) if chi is not None else "N/A"
    else:
        misc_label = "High" if ra < 7 else "Low"

    gastric = gastric_survival(api_pka, api_type, cf["pKa"])
    
    sri = supersaturation_risk_index(delta_pka, inter_type, e_score, api_logP)
    sp_rec, sp_polymer = spring_parachute_index(sri, api_logP, cf["type"])
    
    comp = composite_lead_score(delta_pka, e_score, ra, chi, cf["hygro_risk"],
                                cf["ich_flag"], api_pka, api_type, cf["type"], sri)
    
    fi_gastric = frac_ionized(api_pka, api_type, PH_GASTRIC)
    fi_fassif = frac_ionized(api_pka, api_type, PH_FASSIF)
    
    # Common ion correction for HCl
    ci_factor = 1.0
    if "Hydrochloride" in cf["name"] and api_type == "Base":
        ci_factor = common_ion_correction_hcl(api_pka)
    
    # Gordon-Taylor for polymers
    gt_tg_mix = None
    gt_stability = "N/A"
    t_kauz = None
    if "Polymer" in cf["type"] and cf.get("tg_polymer") and api_tg is not None:
        gt_tg_mix = gordon_taylor_tg(api_tg, cf["tg_polymer"], api_loading)
        gt_stability = asd_stability_class(gt_tg_mix)
        t_kauz = kauzmann_temperature(gt_tg_mix)
    
    # Lattice energy proxy
    le_proxy = lattice_energy_proxy(cf.get("mp_celsius"))

    row = {
        "Partner": cf["name"],
        "Type": cf["type"],
        "Regulatory": cf.get("regulatory_status", "-"),
        "Delta pKa": round(delta_pka, 2),
        "P(Formation)": p_form,
        "Etter Score": e_score,
        "Ra": round(ra, 2),
        "chi (F-H)": chi if chi is not None else "-",
        "Miscibility": misc_label,
        "Interaction Type": inter_type,
        "Gastric Survival": gastric,
        "% Ion Gastric": f"{fi_gastric*100:.0f}%",
        "% Ion FaSSIF": f"{fi_fassif*100:.0f}%",
        "SRI": sri,
        "S&P Rec": sp_rec,
        "CI Factor": round(ci_factor, 2) if ci_factor < 1.0 else "-",
        "Hygro Risk": cf["hygro_risk"],
        "ICH M7 Flag": "Yes" if cf["ich_flag"] else "No",
        "Polymorphism Risk": cf.get("poly_risk", "Medium"),
        "Lattice E Proxy": le_proxy,
        "GT Tg_mix (C)": gt_tg_mix if gt_tg_mix is not None else "-",
        "ASD Stability": gt_stability,
        "Kauzmann T (C)": t_kauz if t_kauz is not None else "-",
        "Lead Score": comp,
        "_ich_flag": cf["ich_flag"],
        "_note": cf.get("note", ""),
        "_ich_note": cf.get("ich_note", ""),
    }
    for lbl in target_pH_str:
        ratio_val = hh_results.get(lbl, {}).get("ratio", 1.0)
        row[f"HH x ({lbl})"] = ratio_val
    results.append(row)

df_full = pd.DataFrame(results).sort_values("Lead Score", ascending=False)

# Filtered view
df_display = df_full.copy()
if show_ich:
    df_display = df_display[df_display["_ich_flag"] == False]
if not show_all_ra:
    df_display = df_display[df_display["Miscibility"].str.contains("High|Miscible|Borderline", na=False)]
df_display = df_display[df_display["Lead Score"] >= min_score]
display_cols = [c for c in df_display.columns if not c.startswith("_")]
df_show = df_display[display_cols]

# ─────────────────────────────────────────────
# PDF REPORT STYLES
# ─────────────────────────────────────────────
def make_pdf_styles():
    base = getSampleStyleSheet()
    S = {
        "cover_title": ParagraphStyle("cover_title", parent=base["Title"],
            fontSize=30, textColor=colors.white, fontName="Helvetica-Bold",
            alignment=TA_LEFT, leading=38, spaceAfter=6),
        "cover_sub": ParagraphStyle("cover_sub", parent=base["Normal"],
            fontSize=13, textColor=colors.HexColor("#B0D4F1"),
            fontName="Helvetica", alignment=TA_LEFT, leading=18, spaceAfter=4),
        "cover_meta": ParagraphStyle("cover_meta", parent=base["Normal"],
            fontSize=10, textColor=colors.HexColor("#CFE8FF"),
            fontName="Helvetica", alignment=TA_LEFT, leading=14),
        "section_h1": ParagraphStyle("section_h1", parent=base["Heading1"],
            fontSize=15, textColor=BRAND_DARK, fontName="Helvetica-Bold",
            spaceBefore=16, spaceAfter=6),
        "section_h2": ParagraphStyle("section_h2", parent=base["Heading2"],
            fontSize=11, textColor=BRAND_MID, fontName="Helvetica-Bold",
            spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"],
            fontSize=9, textColor=colors.HexColor("#2C3E50"),
            fontName="Helvetica", leading=13, spaceAfter=4, alignment=TA_JUSTIFY),
        "body_small": ParagraphStyle("body_small", parent=base["Normal"],
            fontSize=8, textColor=colors.HexColor("#555555"),
            fontName="Helvetica", leading=11, spaceAfter=3),
        "caption": ParagraphStyle("caption", parent=base["Normal"],
            fontSize=7.5, textColor=colors.HexColor("#7F8C8D"),
            fontName="Helvetica-Oblique", leading=11, spaceAfter=2, alignment=TA_CENTER),
        "kpi_label": ParagraphStyle("kpi_label", parent=base["Normal"],
            fontSize=8, textColor=colors.white, fontName="Helvetica", alignment=TA_CENTER),
        "kpi_value": ParagraphStyle("kpi_value", parent=base["Normal"],
            fontSize=18, textColor=colors.white, fontName="Helvetica-Bold", alignment=TA_CENTER),
    }
    return S


def page_template(canv, doc):
    w, h = A4
    canv.setFillColor(BRAND_DARK)
    canv.rect(0, h - 1.2*cm, w, 1.2*cm, fill=1, stroke=0)
    canv.setFillColor(colors.white)
    canv.setFont("Helvetica-Bold", 9)
    canv.drawString(2*cm, h - 0.8*cm, "PharmaCrystal Pro v6.0")
    canv.setFont("Helvetica", 8)
    canv.drawRightString(w - 2*cm, h - 0.8*cm, f"CONFIDENTIAL  |  {datetime.now().strftime('%B %Y')}")
    canv.setFillColor(BRAND_ACCENT)
    canv.rect(0, h - 1.4*cm, w, 0.2*cm, fill=1, stroke=0)
    canv.setFillColor(BRAND_DARK)
    canv.rect(0, 0, w, 0.9*cm, fill=1, stroke=0)
    canv.setFillColor(colors.white)
    canv.setFont("Helvetica", 7)
    canv.drawString(2*cm, 0.3*cm, "PharmaCrystal Pro v6.0  |  For research use only.")
    canv.drawRightString(w - 2*cm, 0.3*cm, f"Page {doc.page}")


def cover_page_template(canv, doc):
    w, h = A4
    canv.setFillColor(BRAND_DARK)
    canv.rect(0, h/2, w, h/2, fill=1, stroke=0)
    canv.setFillColor(BRAND_ACCENT)
    canv.rect(0, h/2 - 0.4*cm, w, 0.4*cm, fill=1, stroke=0)
    canv.setFillColor(BRAND_DARK)
    canv.rect(0, 0, w, 0.9*cm, fill=1, stroke=0)
    canv.setFillColor(colors.white)
    canv.setFont("Helvetica", 7)
    canv.drawCentredString(w/2, 0.3*cm, "PharmaCrystal Pro v6.0  |  CONFIDENTIAL")


def build_kpi_table(data_rows, S):
    bg_colors = [BRAND_MID, BRAND_ACCENT, colors.HexColor("#1B5E20"), colors.HexColor("#4A148C")]
    cells = []
    for i, (label, value) in enumerate(data_rows):
        bg = bg_colors[i % len(bg_colors)]
        inner = Table(
            [[Paragraph(str(value), S["kpi_value"])],
             [Paragraph(label, S["kpi_label"])]],
            colWidths=[4*cm])
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), bg),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ]))
        cells.append(inner)
    tbl = Table([cells], colWidths=[4.5*cm]*len(cells))
    tbl.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    return tbl


def df_to_rl_table(df, col_widths, S, max_rows=None, font_size=7.2):
    if max_rows:
        df = df.head(max_rows)
    header = [Paragraph(f"<b>{escape(str(c))}</b>", ParagraphStyle(
        "th", fontSize=font_size, textColor=colors.white,
        fontName="Helvetica-Bold", alignment=TA_CENTER, leading=font_size+2
    )) for c in df.columns]
    data = [header]
    for _, row in df.iterrows():
        r_cells = []
        for val in row:
            txt = str(val)
            safe = escape(txt)
            if txt in ("High", "Risk", "Yes", "Unstable (< 30" + chr(176) + "C)"):
                style = ParagraphStyle("td_r", fontSize=font_size, textColor=BRAND_RED,
                    fontName="Helvetica-Bold", alignment=TA_CENTER, leading=font_size+2)
            elif "Medium" in txt or "Marginal" in txt or "Borderline" in txt:
                style = ParagraphStyle("td_a", fontSize=font_size, textColor=BRAND_AMBER,
                    fontName="Helvetica-Bold", alignment=TA_CENTER, leading=font_size+2)
            elif txt in ("Low", "Stable", "No") or "Stable" in txt:
                style = ParagraphStyle("td_g", fontSize=font_size, textColor=BRAND_GREEN,
                    fontName="Helvetica-Bold", alignment=TA_CENTER, leading=font_size+2)
            else:
                style = ParagraphStyle("td", fontSize=font_size, textColor=colors.HexColor("#2C3E50"),
                    fontName="Helvetica", alignment=TA_LEFT, leading=font_size+2)
            r_cells.append(Paragraph(safe, style))
        data.append(r_cells)
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), TABLE_HEADER),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, TABLE_ALT]),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#CCC")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
    ]))
    return tbl


def embed_chart(buf, width_cm=16, height_cm=9):
    if buf is None:
        return Spacer(1, 0.1*cm)
    return RLImage(buf, width=width_cm*cm, height=height_cm*cm)


# ─────────────────────────────────────────────
# PDF REPORT GENERATOR (v6.0)
# ─────────────────────────────────────────────
def generate_pdf_report(df_full, api_name, api_pka, api_type, api_synthon,
                         api_dd, api_dp, api_dh, hh_results, target_pH_str,
                         api_logP, api_mw, api_mp, api_tg, api_loading,
                         bcs_hint, is_sol_limited, s0_mol_l, s0_mg_ml,
                         d0_fassif, mad_est, dose_mg):
    buf = io.BytesIO()
    PAGE_W, PAGE_H = A4
    MARGIN = 1.5*cm
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.8*cm, bottomMargin=1.4*cm,
        title=f"PharmaCrystal Pro Report - {api_name}",
        author="PharmaCrystal Pro v6.0")
    S = make_pdf_styles()
    story = []

    # Pre-compute charts
    st.info("Generating charts for PDF report...")
    c1 = chart_lead_scores(df_full)
    c2 = chart_score_components(df_full)
    c3 = chart_landscape(df_full)
    c4 = chart_interaction_distribution(df_full)
    c5 = chart_radar_top3(df_full)
    df_poly = df_full[df_full["Type"] == "Polymer (ASD)"].copy()
    c6 = chart_chi_bars(df_poly)
    c7 = chart_hh_curve(api_pka, api_type, api_name, s0_mol_l, api_mw)
    c8 = chart_ionization_profile(api_pka, api_type, api_name)
    c9 = chart_hygro_donut(df_full)
    c10 = chart_polymorph_risk(df_full)
    c11 = chart_dose_solubility(api_name, dose_mg, s0_mol_l, api_mw, api_pka, api_type)

    # ── COVER ──
    story.append(Spacer(1, PAGE_H * 0.18))
    story.append(Paragraph("PharmaCrystal Pro v6.0", S["cover_sub"]))
    story.append(Paragraph("Solid-State<br/>Screening Report", S["cover_title"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"API: <b>{escape(api_name)}</b>  |  pKa = {api_pka}  |  {api_type}  |  "
        f"logP = {api_logP}  |  MW = {api_mw} g/mol  |  mp = {api_mp}" + chr(176) + "C", S["cover_meta"]))
    story.append(Paragraph(
        f"HSP: dd={api_dd} / dp={api_dp} / dh={api_dh}  |  Tg = {api_tg}" + chr(176) + "C  |  "
        f"Dose = {dose_mg} mg", S["cover_meta"]))
    s0_str = f"{s0_mg_ml:.4f} mg/mL" if s0_mg_ml else "N/A"
    story.append(Paragraph(
        f"Yalkowsky GSE S<sub>0</sub> = {s0_str}  |  D<sub>0</sub> at FaSSIF = "
        f"{d0_fassif if d0_fassif else 'N/A'}  |  BCS: {escape(bcs_hint)}", S["cover_meta"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y at %H:%M')}", S["cover_meta"]))
    story.append(Spacer(1, PAGE_H * 0.05))

    n_leads = len(df_full[df_full["Lead Score"] >= 60])
    n_ich = len(df_full[df_full["_ich_flag"] == True])
    best_hh_lbl = target_pH_str[0] if target_pH_str else "FaSSIF (pH 6.5)"
    best_hh_val = hh_results.get(best_hh_lbl, {}).get("ratio", 1.0)

    kpi_tbl = build_kpi_table([
        ("Partners\nScreened", len(df_full)),
        ("Leads\n(Score>=60)", n_leads),
        ("ICH M7\nFlagged", n_ich),
        (f"HH Fold\n{best_hh_lbl[:7]}", f"{best_hh_val:.0f}x"),
    ], S)
    story.append(kpi_tbl)
    top3 = df_full.head(3)
    top3_names = " / ".join(top3["Partner"].tolist())
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"Top Recommended: <b>{escape(top3_names)}</b>", S["body"]))
    story.append(PageBreak())

    # ── TABLE OF CONTENTS ──
    story.append(Paragraph("Table of Contents", S["section_h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_MID))
    story.append(Spacer(1, 0.15*cm))
    toc_items = [
        "1. Executive Summary",
        "2. Scientific Methodology",
        "3. Complete Screening Results",
        "4. Lead Candidate Analysis (Score >= 60)",
        "5. Biorelevant Solubility & Dose Number Analysis",
        "6. Polymer Miscibility & ASD Stability",
        "7. Safety: ICH M7, Hygroscopicity, Polymorphism",
        "8. Supersaturation Risk & Spring-and-Parachute",
        "9. Partner Notes & Formulation Guidance",
        "10. Regulatory & Scientific References",
    ]
    for item in toc_items:
        story.append(Paragraph(item, S["body"]))
        story.append(Spacer(1, 0.04*cm))
    story.append(PageBreak())

    # ── SECTION 1 — EXECUTIVE SUMMARY ──
    story.append(Paragraph("1. Executive Summary", S["section_h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_MID))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        f"This report presents the results of a comprehensive solid-state screening campaign for "
        f"<b>{escape(api_name)}</b> (pKa = {api_pka}, {api_type}). "
        f"A total of <b>{len(df_full)} pharmaceutical coformers</b> and counterions were evaluated "
        f"using a multi-dimensional physicochemical model integrating Cruz-Cabeza acid-base ionisation "
        f"thermodynamics (CrystEngComm 2012), Hansen solubility parameters, Flory-Huggins polymer miscibility "
        f"theory, Etter's supramolecular synthon rules, Yalkowsky General Solubility Equation, dose number "
        f"analysis, ICH M7 genotoxic impurity assessment, hygroscopicity risk, and supersaturation risk "
        f"modelling with Spring-and-Parachute formulation guidance.",
        S["body"]))

    # Solubility context
    if s0_mg_ml:
        story.append(Paragraph(
            f"<b>Intrinsic Solubility (Yalkowsky GSE):</b> S<sub>0</sub> = {s0_mg_ml:.4f} mg/mL "
            f"({s0_mol_l:.2e} mol/L). "
            f"At {dose_mg} mg dose in {int(V_GI)} mL GI fluid, "
            f"D<sub>0</sub> at FaSSIF = <b>{d0_fassif if d0_fassif else 'N/A'}</b>. "
            + ("D<sub>0</sub> > 1 indicates solubility-limited absorption — "
               "salt or cocrystal strategy is indicated." if is_sol_limited else
               "D<sub>0</sub> <= 1 — solubility is adequate at intestinal pH. "
               "Crystal engineering for stability, bioavailability, or IP may still be valuable."),
            S["body"]))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        f"Of the {len(df_full)} candidates screened, <b>{n_leads} partners achieved a composite lead score "
        f"of 60 or above</b>, making them primary candidates for experimental follow-up. "
        f"The Henderson-Hasselbalch theoretical solubility advantage at {best_hh_lbl} is "
        f"<b>{best_hh_val:.0f}-fold</b> relative to the free form"
        + (", indicating a significant formulation benefit from salt or cocrystal formation." if best_hh_val >= 10
           else ".") , S["body"]))
    story.append(Spacer(1, 0.15*cm))
    kpi_tbl2 = build_kpi_table([
        ("Total Partners\nScreened", len(df_full)),
        ("Leads\n(Score >= 60)", n_leads),
        ("ICH M7\nFlagged", n_ich),
        (f"HH Advantage\n({best_hh_lbl[:12]})", f"{best_hh_val:.0f}x"),
    ], S)
    story.append(kpi_tbl2)
    story.append(Spacer(1, 0.2*cm))

    # Top 5 table
    story.append(Paragraph("Top 5 Ranked Partners", S["section_h2"]))
    top5 = df_full[["Partner","Type","Delta pKa","P(Formation)","Etter Score",
                     "Lead Score","Hygro Risk","ICH M7 Flag","Interaction Type"]].head(5)
    story.append(df_to_rl_table(top5,
        [3.5*cm, 2.2*cm, 1.3*cm, 1.3*cm, 1.2*cm, 1.3*cm, 1.3*cm, 1.2*cm, 2.8*cm], S))
    story.append(Spacer(1, 0.08*cm))
    story.append(Paragraph(
        "Lead Score = weighted sum: DeltapKa probability (30 pts) + Etter synthon (20 pts) + "
        "Miscibility (20 pts) + Hygroscopicity (15 pts) + ICH Safety (10 pts) + "
        "Supersaturation Risk (5 pts). Max = 100.", S["caption"]))
    story.append(Spacer(1, 0.3*cm))

    # Charts 1 & 2
    story.append(Paragraph("Chart 1: Ranked Lead Scores (Top 20)", S["section_h2"]))
    story.append(embed_chart(c1, 16, 9.5))
    story.append(Paragraph(
        "Colour bands: Green >=70, Blue 50-69, Amber 30-49, Red <30. ICH! = ICH M7 flag.", S["caption"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Chart 2: Score Component Breakdown (Top 15)", S["section_h2"]))
    story.append(embed_chart(c2, 16, 9.5))
    story.append(Paragraph(
        "Score = pKa(30) + Synthon(20) + Miscibility(20) + Hygro(15) + ICH(10) + SRI(5) = 100 max.",
        S["caption"]))
    story.append(PageBreak())

    # ── SECTION 2 — METHODOLOGY ──
    story.append(Paragraph("2. Scientific Methodology", S["section_h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_MID))
    story.append(Spacer(1, 0.15*cm))

    methods = [
        ("2.1 Cruz-Cabeza Delta-pKa Classification",
         "Salt formation probability uses Cruz-Cabeza's 2012 three-zone model (CrystEngComm 14, 6362): "
         "delta-pKa &lt; -1 = cocrystal zone; -1 to 4 = continuum (sigmoid P); &gt; 4 = salt zone. "
         "This is fitted to 6465 CSD structures and supersedes the classical binary threshold (Stahl &amp; "
         "Wermuth 2002, Berry et al. 2008) of delta-pKa = 2-3. The sigmoid inflection is at delta-pKa = 2.0 "
         "with slope k = 1.8, correctly reflecting the continuum zone where both salt and cocrystal outcomes are observed."),
        ("2.2 Yalkowsky General Solubility Equation (GSE)",
         "log S<sub>0</sub> (mol/L) = 0.5 - 0.01*(mp - 25) - logP. "
         "Estimates intrinsic aqueous solubility of the unionized free form from melting point and logP alone. "
         "This enables absolute solubility calculation at any pH via Henderson-Hasselbalch, and is critical for "
         "downstream BCS classification, dose number, and MAD calculations. Yalkowsky &amp; Valvani, J. Pharm. Sci. 1980, 69, 912."),
        ("2.3 Dose Number (D<sub>0</sub>) &amp; MAD",
         "D<sub>0</sub> = Dose / (C<sub>s</sub> x V<sub>GI</sub>), where V<sub>GI</sub> = 250 mL (FDA guidance). "
         "D<sub>0</sub> &gt; 1 indicates solubility-limited absorption — the dose cannot fully dissolve in "
         "available GI fluid. MAD = S x K<sub>a</sub> x V x T<sub>si</sub> estimates maximum absorbable dose. "
         "Amidon et al., Pharm. Res. 1995; Lobenberg &amp; Amidon, Eur. J. Pharm. Biopharm. 2000."),
        ("2.4 Hansen Solubility Parameters (Ra)",
         "The Hansen distance Ra = sqrt[4(dd<sub>A</sub> - dd<sub>B</sub>)<sup>2</sup> + "
         "(dp<sub>A</sub> - dp<sub>B</sub>)<sup>2</sup> + (dh<sub>A</sub> - dh<sub>B</sub>)<sup>2</sup>] "
         "is used as a miscibility proxy for small-molecule partners. Partners with Ra &lt; 7 MPa<sup>0.5</sup> "
         "are classified as miscible (Hansen 1967; Greenhalgh et al., J. Pharm. Sci. 1999, 88(11), 1182-1190)."),
        ("2.5 Flory-Huggins Interaction Parameter (chi)",
         "For polymer-API systems, chi = (V<sub>ref</sub> / RT) * (delta<sub>API</sub> - delta<sub>polymer</sub>)<sup>2</sup>. "
         "chi &lt; 0.5 predicts miscibility; 0.5 &lt; chi &lt; 1 is borderline; chi &gt; 1 predicts phase separation. "
         "This is the industry standard for ASD screening (Marsac et al., Pharm. Res. 2006, 23(10), 2417-2426)."),
        ("2.6 Etter's Supramolecular Synthon Rules",
         "Synthon compatibility is scored 0-2 based on Etter's hydrogen bond hierarchy (Etter, Acc. Chem. Res. "
         "1990, 23(4), 120-126): Score 2 = strong complementary donor-acceptor pair (e.g., Acid-Pyridine); "
         "Score 1 = moderate secondary interaction; Score 0 = no productive synthon recognised."),
        ("2.7 Henderson-Hasselbalch Solubility Advantage",
         "The theoretical ionised-form solubility ratio S<sub>total</sub>/S<sub>0</sub> is computed at each "
         "biorelevant pH using the H-H equation. For bases: ratio = 1 + 10<sup>(pKa - pH)</sup>. For acids: "
         "ratio = 1 + 10<sup>(pH - pKa)</sup>. Values &gt; 10x are considered pharmaceutically significant. "
         "Combined with GSE S<sub>0</sub> for absolute solubility estimation. Avdeef, Absorption and Drug "
         "Development, 2nd ed., Wiley, 2012."),
        ("2.8 Common Ion Effect (HCl Salts)",
         "Gastric HCl provides ~0.034-0.05 M Cl<sup>-</sup> which can suppress HCl salt dissolution via "
         "K<sub>sp</sub> limitation. Particularly problematic for highly basic APIs (pKa &gt; 8). A correction "
         "factor is applied to the composite score for HCl salts when appropriate. "
         "Serajuddin, Adv. Drug Deliv. Rev. 2007."),
        ("2.9 Gordon-Taylor Tg &amp; Kauzmann Temperature",
         "Tg<sub>mix</sub> = (w<sub>API</sub>*Tg<sub>API</sub> + K*w<sub>pol</sub>*Tg<sub>pol</sub>) / "
         "(w<sub>API</sub> + K*w<sub>pol</sub>). The Kauzmann temperature T<sub>K</sub> = Tg<sub>mix</sub> - 50K "
         "defines the kinetic trap below which molecular mobility is negligible. Store ASD below T<sub>K</sub> "
         "for maximum stability. Shamblin et al., J. Phys. Chem. B 1999; Gordon &amp; Taylor, J. Appl. Chem. 1952."),
        ("2.10 Supersaturation Risk &amp; Spring-and-Parachute",
         "SRI identifies forms in the continuum zone (delta-pKa -1 to 4) at risk of solution-phase "
         "disproportionation. The Spring-and-Parachute Index evaluates the need for a polymer precipitation "
         "inhibitor (HPMCAS, PVP, HPMC) to maintain supersaturation during the absorption window. "
         "Serajuddin 1992; Guzman et al., Eur. J. Pharm. Sci. 2007."),
        ("2.11 ICH M7 Genotoxic Impurity Assessment",
         "Sulfonate counterions (mesylate, esylate, tosylate) are flagged per ICH M7(R2) (2014, revised 2023) "
         "due to the risk of forming alkyl sulfonate ester genotoxins in the presence of residual alcoholic "
         "solvents. A specific control strategy must be implemented. Mandatory control below TTC "
         "(1.5 ppm for chronic exposure)."),
        ("2.12 Composite Lead Score",
         "A 0-100 composite score weights six dimensions: DeltapKa probability (30), Etter synthon (20), "
         "miscibility (20), hygroscopicity (15), ICH safety (10), supersaturation risk (5). Partners scoring "
         "&gt;= 60 are recommended for experimental follow-up."),
    ]
    for title, body in methods:
        story.append(KeepTogether([
            Paragraph(title, S["section_h2"]),
            Paragraph(body, S["body"]),
            Spacer(1, 0.08*cm)
        ]))
    story.append(PageBreak())

    # ── SECTION 3 — COMPLETE RESULTS ──
    story.append(Paragraph("3. Complete Screening Results", S["section_h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_MID))
    story.append(Spacer(1, 0.1*cm))
    cols_full = ["Partner","Type","Delta pKa","P(Formation)","Etter Score",
                 "Ra","Miscibility","Interaction Type","Hygro Risk","ICH M7 Flag",
                 "SRI","Lead Score"]
    w_full = [2.6*cm, 1.7*cm, 1.1*cm, 1.2*cm, 1.0*cm,
              1.0*cm, 2.0*cm, 2.4*cm, 1.2*cm, 1.1*cm, 1.2*cm, 1.0*cm]
    story.append(df_to_rl_table(df_full[cols_full], w_full, S, font_size=6.5))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Chart 3: Physicochemical Landscape", S["section_h2"]))
    story.append(embed_chart(c3, 16, 9))
    story.append(Paragraph(
        "Bubble ~ Lead Score. Shaded zone = Cruz-Cabeza continuum (-1 < delta-pKa < 4).", S["caption"]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Chart 4: Interaction Type &amp; Etter Distributions", S["section_h2"]))
    story.append(embed_chart(c4, 16, 7))
    story.append(PageBreak())

    # ── SECTION 4 — LEADS ──
    story.append(Paragraph("4. Lead Candidate Analysis (Score >= 60)", S["section_h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_MID))
    story.append(Spacer(1, 0.1*cm))
    df_leads = df_full[df_full["Lead Score"] >= 60].copy()
    if df_leads.empty:
        story.append(Paragraph("No candidates reached the 60-point threshold.", S["body"]))
    else:
        story.append(Paragraph(
            f"<b>{len(df_leads)}</b> partners were classified as lead candidates. Detailed parameter "
            f"profiles are provided across two tables below.", S["body"]))
        story.append(Spacer(1, 0.1*cm))

        # Table 4A — Physicochemical Profile
        story.append(Paragraph("Table 4A — Physicochemical Profile", S["section_h2"]))
        cols_la = ["Partner","Type","Delta pKa","P(Formation)","Etter Score",
                   "Ra","Miscibility","Interaction Type","SRI","Lead Score"]
        w_la = [2.8*cm, 1.8*cm, 1.2*cm, 1.2*cm, 1.0*cm, 1.0*cm, 2.0*cm, 2.4*cm, 1.4*cm, 1.0*cm]
        story.append(df_to_rl_table(df_leads[cols_la], w_la, S, font_size=6.8))
        story.append(Spacer(1, 0.2*cm))

        # Table 4B — Risk & Safety Profile
        story.append(Paragraph("Table 4B — Risk &amp; Safety Profile", S["section_h2"]))
        cols_lb = ["Partner","Gastric Survival","Hygro Risk","ICH M7 Flag","Polymorphism Risk","Lead Score"]
        w_lb = [3.5*cm, 2.8*cm, 2.5*cm, 2.0*cm, 2.8*cm, 2.0*cm]
        story.append(df_to_rl_table(df_leads[cols_lb], w_lb, S, font_size=6.8))
        story.append(Spacer(1, 0.2*cm))

        story.append(Paragraph("Chart 5: Radar — Top 3 Leads", S["section_h2"]))
        story.append(embed_chart(c5, 13, 9.5))
        story.append(Spacer(1, 0.2*cm))

        # Lead commentary — detailed v4.0 style
        story.append(Paragraph("Lead Partner Commentary", S["section_h2"]))
        story.append(Spacer(1, 0.08*cm))
        for _, row in df_leads.head(10).iterrows():
            tag = "Excellent" if row["Lead Score"]>=80 else ("Strong" if row["Lead Score"]>=70 else "Lead")
            # Build detailed commentary paragraph
            paras = []
            paras.append(f"Predicted interaction: <b>{row['Interaction Type']}</b>. ")
            paras.append(f"DeltapKa = {row['Delta pKa']}, P(Formation) = {row['P(Formation)']}. ")
            paras.append(f"Etter synthon score: {row['Etter Score']}/2. ")
            paras.append(f"Hansen distance Ra = {row['Ra']} MPa^0.5 (miscibility: {row['Miscibility']}). ")
            paras.append(f"Gastric survival: {row['Gastric Survival']}. ")
            # Risk flags
            risk_notes = []
            if row["Hygro Risk"] == "High":
                risk_notes.append(f"Hygroscopicity risk: <b>High</b> — desiccant packaging mandatory; review alternative counterion.")
            elif row["Hygro Risk"] == "Medium":
                risk_notes.append(f"Hygroscopicity risk: <b>Medium</b> — review packaging.")
            if row["ICH M7 Flag"] == "Yes":
                risk_notes.append(f"ICH M7: Genotoxic impurity control required.")
            if row["Polymorphism Risk"] in ("High","Medium"):
                risk_notes.append(f"Polymorphism risk: {row['Polymorphism Risk']} — thorough polymorph screen recommended.")
            if row["SRI"] in ("High","Medium"):
                risk_notes.append(f"Supersaturation risk: {row['SRI']}.")
            if row.get("CI Factor","-") != "-":
                risk_notes.append("HCl common-ion correction applied — dissolution may be suppressed in gastric fluid.")
            if risk_notes:
                paras.append(" ".join(risk_notes))

            story.append(KeepTogether([
                Paragraph(f"<b>{escape(row['Partner'])}</b> — Lead Score: {row['Lead Score']} ({tag})", S["section_h2"]),
                Paragraph("".join(paras), S["body"]),
                Spacer(1, 0.1*cm)
            ]))
    story.append(PageBreak())

    # ── SECTION 5 — SOLUBILITY ──
    story.append(Paragraph("5. Biorelevant Solubility &amp; Dose Number Analysis", S["section_h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_MID))
    story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(
        "Henderson-Hasselbalch theoretical solubility advantage (S<sub>total</sub>/S<sub>0</sub>) quantifies "
        "the maximum fold-improvement achievable through salt or cocrystal formation at each biorelevant pH. "
        "This is an upper-bound estimate; actual advantage depends on crystal lattice energy, supersaturation "
        "behaviour, and GI transit conditions.", S["body"]))
    story.append(Spacer(1, 0.1*cm))
    if s0_mg_ml:
        story.append(Paragraph(
            f"Intrinsic solubility estimated by Yalkowsky GSE: S<sub>0</sub> = {s0_mg_ml:.4f} mg/mL. "
            f"At {dose_mg} mg dose, D<sub>0</sub> at FaSSIF pH 6.5 = <b>{d0_fassif}</b>. "
            + (f"MAD estimate = <b>{mad_est:.0f} mg</b>." if mad_est else ""), S["body"]))
    story.append(Spacer(1, 0.15*cm))

    # Per-partner H-H dissolution table for leads
    if not df_leads.empty and hh_results:
        story.append(Paragraph("Biorelevant Dissolution — Lead Partners", S["section_h2"]))
        story.append(Spacer(1, 0.05*cm))
        hh_ph_labels = list(hh_results.keys())[:4]
        hh_header = ["Partner","Type","Interaction Type","Gastric\nSurvival"] + [f"HH x\n({lbl})" for lbl in hh_ph_labels]
        hh_data = [hh_header]
        for _, row in df_leads.head(20).iterrows():
            hh_row = [row["Partner"], row["Type"], row["Interaction Type"], row["Gastric Survival"]]
            for lbl in hh_ph_labels:
                hh_val = hh_results.get(lbl, {}).get("ratio", 1.0)
                hh_row.append(f"{hh_val:.1f}")
            hh_data.append(hh_row)
        n_hh_cols = len(hh_header)
        hh_col_w = [2.6*cm, 1.5*cm, 2.3*cm, 1.5*cm] + [1.8*cm]*len(hh_ph_labels)
        hh_total_w = sum(hh_col_w)
        if hh_total_w > 17.4*cm:
            scale = 17.4*cm / hh_total_w
            hh_col_w = [w*scale for w in hh_col_w]
        rl_hh_data = []
        for ri, r_row in enumerate(hh_data):
            styled_row = []
            for ci, cell in enumerate(r_row):
                fs = 5.5 if ri == 0 else 6.0
                fn = "Helvetica-Bold" if ri == 0 else "Helvetica"
                styled_row.append(Paragraph(f'<font name="{fn}" size="{fs}">{escape(str(cell))}</font>', S["body_small"]))
            rl_hh_data.append(styled_row)
        hh_tbl = Table(rl_hh_data, colWidths=hh_col_w, repeatRows=1)
        hh_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCC")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(hh_tbl)
        story.append(Spacer(1, 0.15*cm))

    # Solubility Advantage at Key pH Values summary table
    story.append(Paragraph("Solubility Advantage at Key Physiological pH Values", S["section_h2"]))
    ph_summary = []
    biorelevant_phs = [
        (1.2, "Gastric (fasted)"), (3.0, "Stomach (fed)"), (4.5, "Duodenum"),
        (5.0, "FeSSIF / Jejunum"), (6.0, "Jejunum (distal)"), (6.5, "FaSSIF / Ileum"),
        (7.4, "Colonic / Plasma"),
    ]
    ph_header = ["pH", "Biorelevant Compartment", f"S_total/S_0 ({escape(api_name)})"]
    ph_data_rows = [ph_header]
    for ph_val, compartment in biorelevant_phs:
        if api_type == "Base":
            ratio = 1 + 10**(api_pka - ph_val)
        else:
            ratio = 1 + 10**(ph_val - api_pka)
        significance = "high" if ratio >= 10 else ("moderate" if ratio >= 3 else "limited")
        ph_data_rows.append([str(ph_val), compartment, f"{ratio:.1f}x"])
    ph_col_w = [2.0*cm, 5.0*cm, 5.0*cm]
    rl_ph_data = []
    for ri, r_row in enumerate(ph_data_rows):
        styled_row = []
        for cell in r_row:
            fs = 6.5 if ri == 0 else 7.0
            fn = "Helvetica-Bold" if ri == 0 else "Helvetica"
            styled_row.append(Paragraph(f'<font name="{fn}" size="{fs}">{escape(str(cell))}</font>', S["body_small"]))
        rl_ph_data.append(styled_row)
    ph_tbl = Table(rl_ph_data, colWidths=ph_col_w, repeatRows=1)
    ph_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(ph_tbl)
    story.append(Spacer(1, 0.08*cm))
    story.append(Paragraph(
        "S<sub>total</sub>/S<sub>0</sub> > 10x = high pharmaceutical significance (green). "
        "3-10x = moderate (amber). &lt; 3x = limited benefit (red). Avdeef, 2012.", S["caption"]))
    story.append(Spacer(1, 0.15*cm))

    story.append(Paragraph("Chart 7: H-H Solubility Advantage vs pH", S["section_h2"]))
    story.append(embed_chart(c7, 16, 7))
    story.append(Spacer(1, 0.15*cm))

    if c11:
        story.append(Paragraph("Chart 11: Dose Number (D<sub>0</sub>) vs pH (NEW)", S["section_h2"]))
        story.append(embed_chart(c11, 16, 6.5))
        story.append(Paragraph(
            "D0 > 1 = dose cannot fully dissolve at that pH. Red zone = solubility-limited. "
            "Salt/cocrystal needed to shift curve below D0=1 at intestinal pH.", S["caption"]))
    story.append(Spacer(1, 0.15*cm))

    story.append(Paragraph("Chart 8: Ionization Profile", S["section_h2"]))
    story.append(embed_chart(c8, 16, 6.5))
    story.append(PageBreak())

    # ── SECTION 6 — POLYMER / ASD ──
    story.append(Paragraph("6. Polymer Miscibility &amp; ASD Stability", S["section_h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_MID))
    story.append(Spacer(1, 0.1*cm))
    if not df_poly.empty:
        cols_p = ["Partner","chi (F-H)","Miscibility","GT Tg_mix (C)","ASD Stability","Kauzmann T (C)","Lead Score"]
        w_p = [3.5*cm, 1.8*cm, 2.8*cm, 2.2*cm, 3.0*cm, 2.2*cm, 1.5*cm]
        story.append(df_to_rl_table(df_poly[cols_p], w_p, S))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Chart 6: Flory-Huggins chi", S["section_h2"]))
        story.append(embed_chart(c6, 14, 5.5))
        story.append(Paragraph(
            "Green: chi<0.5 (miscible). Amber: 0.5-1 (borderline). Red: >1 (immiscible).", S["caption"]))
        story.append(Spacer(1, 0.15*cm))
        story.append(Paragraph(
            f"Kauzmann temperature (T<sub>K</sub> = Tg<sub>mix</sub> - 50K) defines the kinetic trap "
            f"below which molecular mobility is negligible. Store ASD below T<sub>K</sub> for maximum "
            f"stability. Drug loading = {api_loading*100:.0f}% w/w.", S["body"]))
    else:
        story.append(Paragraph("No polymer partners in dataset.", S["body"]))
    story.append(PageBreak())

    # ── SECTION 7 — SAFETY ──
    story.append(Paragraph("7. Safety: ICH M7, Hygroscopicity, Polymorphism", S["section_h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_MID))
    story.append(Spacer(1, 0.1*cm))

    # ICH M7
    df_ich = df_full[df_full["_ich_flag"] == True].copy()
    if not df_ich.empty:
        story.append(Paragraph(
            f"<b>WARNING:</b> {len(df_ich)} counterion(s) carry ICH M7 concerns. A formal risk assessment "
            f"and control strategy must be established before these can be progressed to IND-enabling studies.",
            ParagraphStyle("w", parent=S["body"], textColor=BRAND_RED, fontName="Helvetica-Bold")))
        story.append(Spacer(1, 0.1*cm))
        for _, row in df_ich.iterrows():
            story.append(KeepTogether([
                Paragraph(f"<b>{escape(row['Partner'])}</b> — Lead Score: {row['Lead Score']}", S["section_h2"]),
                Paragraph(f"ICH M7: {escape(row['_ich_note'])}", S["body"]),
                Paragraph(
                    "Recommended action: Conduct Ames test on potential degradants; implement residual solvent "
                    "controls; consult ICH M7 Annex I for acceptable intakes.", S["body_small"]),
                Spacer(1, 0.08*cm)
            ]))
    else:
        story.append(Paragraph("No ICH M7 flags in current set.", S["body"]))
    story.append(Spacer(1, 0.2*cm))

    # Hygroscopicity — grouped by risk level with recommendations (v4.0 style)
    story.append(Paragraph("Hygroscopicity Risk Analysis", S["section_h2"]))
    story.append(Spacer(1, 0.05*cm))
    story.append(Paragraph(
        "Hygroscopicity is a critical solid-state property affecting chemical stability, physical form stability, "
        "powder flow, tabletting, and shelf life. Risk assignment is based on counterion identity and a "
        "supplementary penalty when HCl is paired with an API of pKa > 8 (known to increase deliquescence "
        "propensity).", S["body"]))
    story.append(Spacer(1, 0.1*cm))

    for risk_level, recommendation in [
        ("High", "Desiccant packaging mandatory; controlled humidity manufacturing; consider alternative counterion."),
        ("Medium", "Desiccant packaging recommended; monitor RH during processing."),
        ("Low", "Standard packaging conditions typically sufficient.")
    ]:
        group = df_full[df_full["Hygro Risk"] == risk_level]
        if not group.empty:
            names = ", ".join(group["Partner"].tolist())
            story.append(KeepTogether([
                Paragraph(f"<b>{risk_level} Hygroscopicity Risk — {len(group)} partners</b>", S["body"]),
                Paragraph(names, S["body_small"]),
                Paragraph(f"<i>Recommendation: {recommendation}</i>", S["body_small"]),
                Spacer(1, 0.06*cm)
            ]))
    story.append(Spacer(1, 0.1*cm))

    # Hygro chart
    story.append(Paragraph("Chart 9: Hygroscopicity Distribution", S["section_h2"]))
    donut_tbl = Table([[embed_chart(c9, 9, 7)]], colWidths=[16*cm])
    donut_tbl.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER")]))
    story.append(donut_tbl)
    story.append(Spacer(1, 0.2*cm))

    # Polymorphism — full table + chart (v4.0 style)
    story.append(Paragraph("Polymorphism Risk Summary", S["section_h2"]))
    story.append(Spacer(1, 0.05*cm))
    story.append(Paragraph(
        "Polymorphism risk is estimated based on coformer structural flexibility (rotatable bonds), molecular "
        "weight, aromatic content, and CSD literature precedent. High-risk coformers should be subjected to a "
        "comprehensive polymorph screen (temperature, solvent, humidity, and grinding methods) before advancing "
        "to formulation development.", S["body"]))
    story.append(Spacer(1, 0.1*cm))
    cols_poly_risk = ["Partner","Type","Polymorphism Risk","Etter Score","Lead Score"]
    w_poly_risk = [3.8*cm, 2.5*cm, 2.8*cm, 2.0*cm, 2.0*cm]
    # Sort by polymorphism risk for clarity
    poly_order = {"High": 0, "Medium": 1, "Low": 2}
    df_poly_sorted = df_full.copy()
    df_poly_sorted["_poly_sort"] = df_poly_sorted["Polymorphism Risk"].map(poly_order)
    df_poly_sorted = df_poly_sorted.sort_values(["_poly_sort","Lead Score"], ascending=[True,False])
    story.append(df_to_rl_table(df_poly_sorted[cols_poly_risk], w_poly_risk, S, font_size=6.5))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph("Chart 10: Polymorphism Risk by Category", S["section_h2"]))
    story.append(embed_chart(c10, 16, 7))
    story.append(PageBreak())

    # ── SECTION 8 — SUPERSATURATION ──
    story.append(Paragraph("8. Supersaturation Risk &amp; Spring-and-Parachute", S["section_h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_MID))
    story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(
        "Partners in the continuum zone (-1 &lt; delta-pKa &lt; 4) risk solution-phase "
        "disproportionation. The Spring-and-Parachute Index recommends whether a polymer "
        "precipitation inhibitor should be included in the formulation to maintain supersaturation "
        "during the absorption window.", S["body"]))
    df_sri = df_full[df_full["SRI"] != "N/A"].copy()
    if not df_sri.empty:
        cols_s = ["Partner","Delta pKa","Etter Score","Interaction Type","SRI","S&P Rec","Lead Score"]
        w_s = [3.0*cm, 1.3*cm, 1.2*cm, 2.8*cm, 1.5*cm, 5.0*cm, 1.2*cm]
        story.append(df_to_rl_table(df_sri[cols_s], w_s, S, font_size=6.8))
    story.append(PageBreak())

    # ── SECTION 9 — NOTES ──
    story.append(Paragraph("9. Partner Notes &amp; Formulation Guidance", S["section_h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_MID))
    story.append(Spacer(1, 0.1*cm))
    for _, row in df_full.iterrows():
        note = row.get("_note","")
        if not note: continue
        story.append(KeepTogether([
            Paragraph(
                f"<b>{escape(row['Partner'])}</b> [Score {row['Lead Score']} | {row['Interaction Type']}]",
                S["section_h2"]),
            Paragraph(escape(note), S["body_small"]),
            Spacer(1, 0.05*cm)
        ]))
    story.append(PageBreak())

    # ── SECTION 10 — REFERENCES ──
    story.append(Paragraph("10. References", S["section_h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_MID))
    refs = [
        "Cruz-Cabeza, A.J. Acid-base crystalline complexes and the pKa rule. CrystEngComm 2012, 14, 6362.",
        "Yalkowsky, S.H. &amp; Valvani, S.C. Solubility and partitioning I: Estimation of aqueous solubility. J. Pharm. Sci. 1980, 69, 912-922.",
        "Amidon, G.L. et al. A theoretical basis for a biopharmaceutic drug classification: the correlation of in vitro drug product dissolution and in vivo bioavailability. Pharm. Res. 1995, 12(3), 413-420.",
        "Stahl, P.H. &amp; Wermuth, C.G. (Eds.). Handbook of Pharmaceutical Salts: Properties, Selection, and Use. Wiley-VCH, 2002.",
        "Berry, D.J. et al. Applying Hot-Stage Microscopy to Co-Crystal Screening: A Study of Nicotinamide with Seven APIs. Cryst. Growth Des. 2008.",
        "Hansen, C.M. Hansen Solubility Parameters: A User's Handbook. CRC Press, 2007.",
        "Greenhalgh, D.J. et al. Solubility Parameters as Predictors of Miscibility in Solid Dispersions. J. Pharm. Sci. 1999, 88(11), 1182-1190.",
        "Marsac, P.J. et al. Estimation of Drug-Polymer Miscibility and Solubility in ASD Using Experimentally Determined Interaction Parameters. Pharm. Res. 2006, 23(10), 2417-2426.",
        "Etter, M.C. Encoding and Decoding Hydrogen-Bond Patterns of Organic Compounds. Acc. Chem. Res. 1990, 23(4), 120-126.",
        "Avdeef, A. Absorption and Drug Development: Solubility, Permeability, and Charge State. 2nd ed. Wiley, 2012.",
        "Serajuddin, A.T.M. Salt Formation to Improve Drug Solubility. Adv. Drug Deliv. Rev. 2007, 59(7), 603-616.",
        "Shamblin, S.L. et al. Characterization of the Time Scales of Molecular Motion in Pharmaceutically Important Glasses. J. Phys. Chem. B 1999.",
        "Gordon, M. &amp; Taylor, J.S. Ideal Copolymers and the Second-Order Transitions of Synthetic Rubbers. J. Appl. Chem. 1952.",
        "ICH M7(R2). Assessment and Control of DNA Reactive (Mutagenic) Impurities in Pharmaceuticals to Limit Potential Carcinogenic Risk. EMA/CHMP/ICH/83812/2013, 2023.",
        "Pudipeddi, M. &amp; Serajuddin, A.T.M. Trends in Solubility of Polymorphs. J. Pharm. Sci. 2005, 94(5), 929-939.",
        "Lobenberg, R. &amp; Amidon, G.L. Modern bioavailability, bioequivalence and biopharmaceutics classification system. Eur. J. Pharm. Biopharm. 2000, 50(1), 3-12.",
        "Guzman, H.R. et al. Combined use of crystalline salt forms and precipitation inhibitors to improve oral absorption of celecoxib. Eur. J. Pharm. Sci. 2007, 30(3-4), 218-230.",
        "FDA Guidance for Industry: Regulatory Classification of Pharmaceutical Co-Crystals. U.S. Department of Health and Human Services, 2018.",
        "EMA Reflection Paper on the Use of Cocrystals and Other Solid State Forms of Active Substances in Medicinal Products. EMA/CHMP/CVMP/QWP/284008/2018, 2022.",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {ref}", S["body_small"]))
        story.append(Spacer(1, 0.05*cm))

    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCC")))
    story.append(Paragraph(
        "DISCLAIMER: PharmaCrystal Pro v6.0 is a computational screening tool for research use only. "
        "Predictions must be validated experimentally. Not a substitute for professional pharmaceutical advice.",
        ParagraphStyle("disc", parent=S["body_small"], textColor=colors.HexColor("#999"), alignment=TA_JUSTIFY)))

    doc.build(story, onFirstPage=cover_page_template, onLaterPages=page_template)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# MAIN DISPLAY
# ─────────────────────────────────────────────
st.subheader(f"Screening Report: {api_name}  (pKa={api_pka}, {api_type})")
s0_disp = f"{s0_mg_ml:.4f} mg/mL" if s0_mg_ml else "N/A"
st.caption(f"**BCS:** {bcs_hint}  |  **GSE S₀:** {s0_disp}  |  **D₀ (FaSSIF):** {d0_fassif if d0_fassif else 'N/A'}  |  **MAD:** {mad_est if mad_est else 'N/A'} mg")

col_k1, col_k2, col_k3, col_k4 = st.columns(4)
col_k1.metric("Partners Screened", len(df_full))
col_k2.metric("Leads (>=60)", len(df_full[df_full["Lead Score"] >= 60]))
col_k3.metric("ICH-Flagged", len(df_full[df_full["_ich_flag"] == True]))
best_hh_lbl = target_pH_str[0] if target_pH_str else "FaSSIF (pH 6.5)"
best_hh_val = hh_results.get(best_hh_lbl, {}).get("ratio", 1.0)
col_k4.metric(f"HH ({best_hh_lbl})", f"{best_hh_val:.0f}x")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Screening Table", "Landscape", "Solubility & Ionization",
    "Lead Scorecard", "Safety", "ASD & Supersaturation"])

with tab1:
    def highlight_leads(row):
        score = row.get("Lead Score", 0)
        ich = row.get("ICH M7 Flag", "")
        if ich == "Yes": return ["background-color: #fff3e0"] * len(row)
        if score >= 70: return ["background-color: #c8e6c9"] * len(row)
        if score >= 50: return ["background-color: #e8f5e9"] * len(row)
        return [""] * len(row)
    st.dataframe(df_show.style.apply(highlight_leads, axis=1),
                 use_container_width=True, height=500)
    with st.expander("Partner Notes"):
        for _, row in df_display.iterrows():
            st.markdown(f"**{row['Partner']}** — {row['_note']}")
            if row["_ich_flag"]:
                st.error(row["_ich_note"])

with tab2:
    import plotly.express as px
    import plotly.graph_objects as go
    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig_s = px.scatter(df_full, x="Delta pKa", y="Ra", text="Partner",
            color="Interaction Type", size="Lead Score",
            hover_data=["Etter Score","Hygro Risk","SRI"],
            title="Delta-pKa vs Hansen Distance (Ra)",
            color_discrete_sequence=px.colors.qualitative.Set2)
        fig_s.add_hline(y=7, line_dash="dash", line_color="red", annotation_text="Ra=7")
        fig_s.add_vline(x=4, line_dash="dot", line_color="blue", annotation_text="pKa=4 (salt)")
        fig_s.add_vline(x=-1, line_dash="dot", line_color="grey", annotation_text="pKa=-1 (cocrystal)")
        fig_s.update_traces(textposition="top center", textfont_size=8)
        st.plotly_chart(fig_s, use_container_width=True)
    with col_b:
        df_et = df_full[["Partner","Etter Score","Type"]].sort_values("Etter Score").tail(15)
        fig_e = px.bar(df_et, x="Etter Score", y="Partner", orientation="h",
            color="Etter Score", color_continuous_scale="Greens", title="Etter Synthon Score (Top 15)")
        st.plotly_chart(fig_e, use_container_width=True)

    df_pt = df_full[df_full["Type"] == "Polymer (ASD)"].copy()
    if not df_pt.empty:
        st.divider()
        st.markdown("#### Flory-Huggins chi")
        df_pt["chi_num"] = pd.to_numeric(df_pt["chi (F-H)"], errors="coerce")
        fig_c = px.bar(df_pt.dropna(subset=["chi_num"]), x="Partner", y="chi_num",
            color="chi_num", color_continuous_scale="RdYlGn_r",
            title="Flory-Huggins chi (< 0.5 = Miscible)", labels={"chi_num":"chi"})
        fig_c.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="chi=0.5")
        fig_c.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="chi=1.0")
        st.plotly_chart(fig_c, use_container_width=True)

with tab3:
    import plotly.graph_objects as go
    st.markdown("#### Henderson-Hasselbalch Solubility Advantage")
    pH_sw = np.linspace(0.5, 10.5, 300)
    sol_r = [hh_solubility_at_pH(api_pka, api_type, ph)[0] for ph in pH_sw]
    fig_hh = go.Figure()
    fig_hh.add_trace(go.Scatter(x=pH_sw, y=sol_r, mode="lines",
        line=dict(color="#1976D2", width=2.5), name="S_total/S0"))
    for label, pH_val in pH_map.items():
        if label in target_pH_str:
            r_at = hh_solubility_at_pH(api_pka, api_type, pH_val)[0]
            fig_hh.add_vline(x=pH_val, line_dash="dot", annotation_text=f"{label} ({r_at}x)")
    fig_hh.add_hline(y=10, line_dash="dash", line_color="green", annotation_text="10x")
    fig_hh.update_layout(title=f"H-H Solubility — {api_name}", xaxis_title="pH",
                         yaxis_title="S_total/S0", yaxis_type="log")
    st.plotly_chart(fig_hh, use_container_width=True)

    if s0_mg_ml and d0_fassif:
        st.divider()
        st.markdown("#### Dose Number (D₀) vs pH (NEW)")
        d0_vals = []
        for ph in pH_sw:
            r, _ = hh_solubility_at_pH(api_pka, api_type, ph)
            s_ph = s0_mg_ml * r
            d0_vals.append(min(dose_mg / (s_ph * V_GI) if s_ph > 0 else 999, 1000))
        fig_d0 = go.Figure()
        fig_d0.add_trace(go.Scatter(x=pH_sw, y=d0_vals, mode="lines",
            line=dict(color="#E65100", width=2.5), name="D0"))
        fig_d0.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="D0=1 (BCS limit)")
        fig_d0.update_layout(title=f"Dose Number vs pH — {api_name} ({dose_mg} mg)",
                             xaxis_title="pH", yaxis_title="D0", yaxis_type="log")
        st.plotly_chart(fig_d0, use_container_width=True)

    st.divider()
    st.markdown("#### Ionization Profile")
    pH_ip = np.linspace(0, 12.5, 300)
    if api_type == "Base":
        fi_arr = [1/(1+10**(ph-api_pka))*100 for ph in pH_ip]
    else:
        fi_arr = [1/(1+10**(api_pka-ph))*100 for ph in pH_ip]
    fu_arr = [100-v for v in fi_arr]
    fig_ip = go.Figure()
    fig_ip.add_trace(go.Scatter(x=pH_ip, y=fi_arr, mode="lines", name="Ionized", line=dict(color="#1565C0", width=2.5)))
    fig_ip.add_trace(go.Scatter(x=pH_ip, y=fu_arr, mode="lines", name="Unionized", line=dict(color="#E65100", width=2.5, dash="dash")))
    fig_ip.add_vline(x=api_pka, line_dash="solid", line_color="black", annotation_text=f"pKa={api_pka}")
    fig_ip.update_layout(title=f"Ionization — {api_name}", xaxis_title="pH", yaxis_title="Fraction (%)")
    st.plotly_chart(fig_ip, use_container_width=True)

with tab4:
    import plotly.graph_objects as go, plotly.express as px
    df_sc = df_full[["Partner","Type","Lead Score","Interaction Type","Etter Score",
                     "Hygro Risk","ICH M7 Flag","SRI"]].sort_values("Lead Score", ascending=False)
    fig_sc = px.bar(df_sc.head(20), x="Lead Score", y="Partner", orientation="h",
        color="Lead Score", color_continuous_scale="RdYlGn", text="Lead Score",
        hover_data=["Interaction Type","Hygro Risk","SRI"],
        title="Top 20 Partners — Lead Score")
    fig_sc.update_traces(textposition="outside")
    fig_sc.update_layout(height=650, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_sc, use_container_width=True)

    top3 = df_full.head(3)
    categories = ["pKa","Synthon","Miscibility","Hygro","ICH","SRI"]
    fig_rd = go.Figure()
    r_cols = ["#2196F3","#4CAF50","#FF9800"]
    def rv(row):
        p = cruz_cabeza_pka_probability(row["Delta pKa"])
        pka_ = p * 30; eth_ = (row["Etter Score"]/2)*20
        ra_ = pd.to_numeric(row["Ra"], errors='coerce')
        ra_s = max(0, 20 - float(ra_)*2) if not pd.isna(ra_) else 10
        hyg_ = {"Low":14.25,"Medium":9,"High":3}[row["Hygro Risk"]]
        ich_ = 0 if row.get("ICH M7 Flag")=="Yes" else 10
        sri_ = {"N/A":5,"Low":5,"Medium":2.5,"High":0}.get(row.get("SRI","N/A"),2.5)
        vals = [pka_, eth_, ra_s, hyg_, ich_, sri_]
        maxv = [30, 20, 20, 15, 10, 5]
        return [round(v/m*100,1) for v,m in zip(vals, maxv)]
    for i, (_, row) in enumerate(top3.iterrows()):
        vals = rv(row)
        fig_rd.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=categories+[categories[0]],
            fill="toself", name=row["Partner"], line=dict(color=r_cols[i%3])))
    fig_rd.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                         title="Top 3 — Score Radar")
    st.plotly_chart(fig_rd, use_container_width=True)

with tab5:
    import plotly.express as px
    st.markdown("#### ICH M7 Genotoxic Impurity Risk")
    df_ich_t = df_full[df_full["_ich_flag"]==True][["Partner","Type","_ich_note","Lead Score"]]
    if df_ich_t.empty:
        st.success("No ICH M7 flags.")
    else:
        for _, row in df_ich_t.iterrows():
            st.error(f"**{row['Partner']}**: {row['_ich_note']}")
    st.divider()
    st.markdown("#### Hygroscopicity Risk")
    for risk in ["High","Medium","Low"]:
        partners = df_full[df_full["Hygro Risk"]==risk]["Partner"].tolist()
        if partners:
            emoji = {"High":"🔴","Medium":"🟡","Low":"🟢"}[risk]
            with st.expander(f"{emoji} {risk} — {len(partners)} partners"):
                st.write(", ".join(partners))
    st.divider()
    st.markdown("#### Polymorphism Risk")
    st.dataframe(df_full[["Partner","Type","Polymorphism Risk","Lattice E Proxy","Lead Score"]]
        .sort_values(["Polymorphism Risk","Lead Score"], ascending=[True,False]),
        use_container_width=True)

with tab6:
    import plotly.express as px, plotly.graph_objects as go
    st.markdown("#### Gordon-Taylor ASD Tg & Kauzmann Temperature")
    df_gt = df_full[df_full["Type"]=="Polymer (ASD)"].copy()
    if not df_gt.empty and api_tg is not None:
        st.dataframe(df_gt[["Partner","chi (F-H)","Miscibility","GT Tg_mix (C)",
                            "ASD Stability","Kauzmann T (C)","Lead Score"]], use_container_width=True)
        loadings = np.linspace(0.05, 0.60, 100)
        fig_gt = go.Figure()
        for _, row in df_gt.iterrows():
            cf_e = next((c for c in coformers_db if c["name"]==row["Partner"]), None)
            tg_p = cf_e.get("tg_polymer") if cf_e else None
            if tg_p:
                tg_mix_v = [gordon_taylor_tg(api_tg, tg_p, w) for w in loadings]
                fig_gt.add_trace(go.Scatter(x=loadings*100, y=tg_mix_v, mode="lines", name=row["Partner"]))
        fig_gt.add_hline(y=75, line_dash="dash", line_color="red",
                         annotation_text="Tg_mix=75C (25C+50 stability)")
        fig_gt.add_hline(y=25, line_dash="dot", line_color="grey",
                         annotation_text="Kauzmann T (25C storage)")
        fig_gt.update_layout(title="Gordon-Taylor Tg_mix vs Drug Loading",
                             xaxis_title="Drug Loading (% w/w)", yaxis_title="Predicted Tg_mix (C)")
        st.plotly_chart(fig_gt, use_container_width=True)
    else:
        st.info("No polymer data or API Tg not set.")

    st.divider()
    st.markdown("#### Supersaturation Risk & Spring-and-Parachute")
    df_sri_t = df_full[["Partner","Type","Delta pKa","Etter Score","Interaction Type",
                        "SRI","S&P Rec","% Ion FaSSIF","Lead Score"]]
    def colour_sri(row):
        sri = row.get("SRI","N/A")
        if sri == "High": return ["background-color: #ffcdd2"]*len(row)
        if sri == "Medium": return ["background-color: #fff9c4"]*len(row)
        if sri == "Low": return ["background-color: #c8e6c9"]*len(row)
        return [""]*len(row)
    st.dataframe(df_sri_t.style.apply(colour_sri, axis=1), use_container_width=True)
    st.caption("Spring-and-Parachute: recommends polymer precipitation inhibitor for high SRI partners.")

# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
st.divider()
st.markdown("### Export Results")
export_df = df_full[[c for c in df_full.columns if not c.startswith("_")]]

col_e1, col_e2, col_e3 = st.columns(3)
with col_e1:
    st.download_button("CSV (Full)", export_df.to_csv(index=False),
        f"{api_name}_pharmacrystal_v6.csv", "text/csv", use_container_width=True)

with col_e2:
    with st.spinner("Generating PDF with 11 charts..."):
        pdf_bytes = generate_pdf_report(
            df_full, api_name, api_pka, api_type, api_synthon,
            api_dd, api_dp, api_dh, hh_results, target_pH_str,
            api_logP, api_mw, api_mp, api_tg, api_loading,
            bcs_hint, is_sol_limited, s0_mol_l, s0_mg_ml,
            d0_fassif, mad_est, dose_mg)
    st.download_button("PDF Report (11 Charts)", pdf_bytes,
        f"{api_name}_PharmaCrystal_v6_{datetime.now().strftime('%Y%m%d')}.pdf",
        "application/pdf", use_container_width=True, type="primary")

with col_e3:
    top_leads = export_df[export_df["Lead Score"] >= 60]
    st.download_button("CSV (Leads Only)", top_leads.to_csv(index=False),
        f"{api_name}_leads_v6.csv", "text/csv", use_container_width=True)

st.caption(
    "PharmaCrystal Pro v6.0 | Scientific models: Cruz-Cabeza 2012, Yalkowsky GSE, "
    "Dose Number (Amidon 1995), Flory-Huggins (Marsac 2006), Etter 1990, "
    "Gordon-Taylor 1952, Kauzmann T, Spring-and-Parachute, ICH M7(R2) 2023, "
    "Common Ion Effect, Lattice Energy Proxy."
)
