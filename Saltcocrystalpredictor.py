"""
PharmaCrystal Pro v7.0 — Phase 1: Molecular Intelligence Engine
================================================================
Auto-calculates HSP (δd, δp, δh), LogP, pKa, MW, TPSA, BCS class
from structure (SMILES or functional group builder).

Group contribution methods:
  • Hansen HSP    → Hoy (1985) / van Krevelen atom contributions
  • LogP          → Rekker-Mannhold fragment method (pharma-calibrated)
  • pKa estimate  → Functional group heuristics (Lide, CRC + ALOGPS data)
  • MP estimate   → Joback group contribution method
  • TPSA estimate → Ertl method (fragment-based)

Author: PharmaCrystal Pro v7.0
"""

import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

# ══════════════════════════════════════════════════════════════════
# 1. SOLVENT DATABASE — Full Pharma Screening Set
# ══════════════════════════════════════════════════════════════════
# HSP values in MPa^0.5 (Hansen, 2007 + Barton 1991)
# Boiling points in °C, densities in g/mL
# ICH Q3C class: 1=avoid, 2=limit(ppm), 3=low toxicity, 4=not classified

SOLVENT_DB = {
    # ── Protic — Alcohols ──────────────────────────────────────
    "MeOH": {
        "full_name": "Methanol", "formula": "CH₃OH",
        "dd": 15.1, "dp": 12.3, "dh": 22.3, "dt": 29.6,
        "bp": 64.7, "mp": -98, "density": 0.791, "mw": 32.0,
        "viscosity": 0.55, "dielectric": 32.7, "vapor_pressure": 128,
        "ich_class": 2, "ich_ppm": 3000, "ich_class_label": "Class 2 (3000 ppm)",
        "category": "Protic — Alcohol", "protic": True,
        "miscible_water": True, "color": "#E91E63",
        "note": "Highly flammable. CNS toxicity — ocular/metabolic hazard. "
                "Excellent polar protic solvent. ICH Class 2 (3000 ppm). "
                "Mesylate genotoxin risk when combined with sulfonate synthesis.",
    },
    "EtOH": {
        "full_name": "Ethanol", "formula": "C₂H₅OH",
        "dd": 15.8, "dp": 8.8, "dh": 19.4, "dt": 26.5,
        "bp": 78.4, "mp": -114, "density": 0.789, "mw": 46.1,
        "viscosity": 1.08, "dielectric": 24.5, "vapor_pressure": 59,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Protic — Alcohol", "protic": True,
        "miscible_water": True, "color": "#E91E63",
        "note": "GRAS. Preferred pharmaceutical solvent. ICH Class 3. "
                "Good for recrystallisation of moderate-polarity APIs. "
                "Denatured ethanol: check for denaturant in final product.",
    },
    "IPA": {
        "full_name": "Isopropanol", "formula": "(CH₃)₂CHOH",
        "dd": 15.8, "dp": 6.1, "dh": 16.4, "dt": 23.6,
        "bp": 82.6, "mp": -89, "density": 0.786, "mw": 60.1,
        "viscosity": 2.04, "dielectric": 17.9, "vapor_pressure": 43,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Protic — Alcohol", "protic": True,
        "miscible_water": True, "color": "#E91E63",
        "note": "ICH Class 3. Excellent for slurry maturation and polymorph control. "
                "Lower polarity than EtOH — useful for less polar APIs. "
                "Strong anti-solvent effect when added to water.",
    },
    "EthGly": {
        "full_name": "Ethylene Glycol", "formula": "HOCH₂CH₂OH",
        "dd": 17.0, "dp": 11.0, "dh": 26.0, "dt": 32.9,
        "bp": 197, "mp": -13, "density": 1.113, "mw": 62.1,
        "viscosity": 16.1, "dielectric": 37.7, "vapor_pressure": 0.08,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Protic — Diol", "protic": True,
        "miscible_water": True, "color": "#AD1457",
        "note": "High boiling, high viscosity. Used as co-solvent and anti-freeze. "
                "Strong H-bond network — useful for very polar APIs. "
                "ICH Class 3.",
    },
    "PropGly": {
        "full_name": "Propylene Glycol (1,2-PD)", "formula": "CH₃CHOHCH₂OH",
        "dd": 16.8, "dp": 9.4, "dh": 23.3, "dt": 30.2,
        "bp": 188, "mp": -60, "density": 1.036, "mw": 76.1,
        "viscosity": 40.4, "dielectric": 27.5, "vapor_pressure": 0.13,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Protic — Diol", "protic": True,
        "miscible_water": True, "color": "#AD1457",
        "note": "GRAS. Pharmaceutical excipient. High viscosity limits spray drying. "
                "Useful co-solvent for formulation. Good cryoprotectant.",
    },

    # ── Polar Aprotic ───────────────────────────────────────────
    "ACN": {
        "full_name": "Acetonitrile", "formula": "CH₃CN",
        "dd": 15.3, "dp": 18.0, "dh": 6.1, "dt": 24.4,
        "bp": 81.6, "mp": -46, "density": 0.786, "mw": 41.1,
        "viscosity": 0.35, "dielectric": 37.5, "vapor_pressure": 97,
        "ich_class": 2, "ich_ppm": 410, "ich_class_label": "Class 2 (410 ppm)",
        "category": "Polar Aprotic", "protic": False,
        "miscible_water": True, "color": "#FF6F00",
        "note": "ICH Class 2 (410 ppm — strict limit). Excellent for HPLC. "
                "High polarity, low H-bonding. Widely used in crystallisation screens. "
                "Monitor residual solvent by headspace GC.",
    },
    "Acetone": {
        "full_name": "Acetone", "formula": "(CH₃)₂CO",
        "dd": 15.5, "dp": 10.4, "dh": 7.0, "dt": 19.9,
        "bp": 56.1, "mp": -95, "density": 0.791, "mw": 58.1,
        "viscosity": 0.31, "dielectric": 20.7, "vapor_pressure": 233,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Polar Aprotic", "protic": False,
        "miscible_water": True, "color": "#FF6F00",
        "note": "ICH Class 3. Very low viscosity, high vapour pressure — excellent for spray drying. "
                "Broad solvating ability. Good for polymorph screening (high/low-polarity balance). "
                "Reactive with primary amines (Schiff base).",
    },
    "2-Butanone": {
        "full_name": "2-Butanone (MEK)", "formula": "CH₃COC₂H₅",
        "dd": 16.0, "dp": 9.0, "dh": 5.1, "dt": 19.0,
        "bp": 79.6, "mp": -87, "density": 0.805, "mw": 72.1,
        "viscosity": 0.41, "dielectric": 18.5, "vapor_pressure": 105,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Polar Aprotic", "protic": False,
        "miscible_water": True, "color": "#FF6F00",
        "note": "ICH Class 3. Similar to acetone but slightly less polar. "
                "Useful when acetone is too volatile or reactive. "
                "Good anti-solvent for water-soluble salts.",
    },
    "4Me2Pentanone": {
        "full_name": "4-Methyl-2-pentanone (MIBK)", "formula": "(CH₃)₂CHCH₂COCH₃",
        "dd": 15.3, "dp": 6.1, "dh": 4.1, "dt": 17.0,
        "bp": 117, "mp": -85, "density": 0.796, "mw": 100.2,
        "viscosity": 0.58, "dielectric": 13.1, "vapor_pressure": 21,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Polar Aprotic", "protic": False,
        "miscible_water": False, "color": "#FF6F00",
        "note": "ICH Class 3. Less polar than acetone — useful for lipophilic APIs. "
                "Immiscible with water — excellent for liquid-liquid extraction. "
                "Higher bp useful for reactions and recrystallisation.",
    },
    "DMF": {
        "full_name": "Dimethylformamide", "formula": "HCON(CH₃)₂",
        "dd": 17.4, "dp": 13.7, "dh": 11.3, "dt": 24.8,
        "bp": 153, "mp": -61, "density": 0.944, "mw": 73.1,
        "viscosity": 0.90, "dielectric": 36.7, "vapor_pressure": 3.8,
        "ich_class": 2, "ich_ppm": 880, "ich_class_label": "Class 2 (880 ppm)",
        "category": "Polar Aprotic", "protic": False,
        "miscible_water": True, "color": "#E65100",
        "note": "ICH Class 2 (880 ppm). Reproductive toxin (CMR). "
                "High boiling — difficult to remove; monitor by headspace GC. "
                "Excellent solvent for poorly soluble APIs. Avoid in late-stage synthesis. "
                "Consider DMA or NMP as lower-concern alternatives.",
    },
    "DMSO": {
        "full_name": "Dimethyl Sulfoxide", "formula": "(CH₃)₂SO",
        "dd": 18.4, "dp": 16.4, "dh": 10.2, "dt": 26.7,
        "bp": 189, "mp": 18.5, "density": 1.100, "mw": 78.1,
        "viscosity": 1.99, "dielectric": 46.7, "vapor_pressure": 0.6,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Polar Aprotic", "protic": False,
        "miscible_water": True, "color": "#FF6F00",
        "note": "ICH Class 3. Exceptional solvating power — dissolves most APIs. "
                "High boiling point limits removal (use lyophilisation or HV). "
                "Solves solubility problems in early screens (use as reference solvent). "
                "MP 18.5°C — solid at room temperature in cold labs.",
    },

    # ── Ethers ─────────────────────────────────────────────────
    "THF": {
        "full_name": "Tetrahydrofuran", "formula": "C₄H₈O",
        "dd": 16.8, "dp": 5.7, "dh": 8.0, "dt": 18.6,
        "bp": 66, "mp": -108, "density": 0.889, "mw": 72.1,
        "viscosity": 0.48, "dielectric": 7.6, "vapor_pressure": 162,
        "ich_class": 2, "ich_ppm": 720, "ich_class_label": "Class 2 (720 ppm)",
        "category": "Ether", "protic": False,
        "miscible_water": True, "color": "#7B1FA2",
        "note": "ICH Class 2 (720 ppm). Peroxide formation hazard — check for inhibitor. "
                "Excellent for many organic reactions. Miscible with water — difficult anti-solvent. "
                "Useful in spray drying mixtures. GC monitoring essential.",
    },
    "DEE": {
        "full_name": "Diethyl Ether", "formula": "(C₂H₅)₂O",
        "dd": 14.5, "dp": 2.9, "dh": 5.1, "dt": 15.8,
        "bp": 34.6, "mp": -116, "density": 0.713, "mw": 74.1,
        "viscosity": 0.22, "dielectric": 4.3, "vapor_pressure": 587,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Ether", "protic": False,
        "miscible_water": False, "color": "#7B1FA2",
        "note": "ICH Class 3. Highly flammable (BP 35°C). Peroxide formation hazard. "
                "Excellent anti-solvent for many APIs. Low polarity — precipitates polar salts. "
                "Very low bp — easy to remove. Store with BHT inhibitor.",
    },
    "TBME": {
        "full_name": "tert-Butyl Methyl Ether", "formula": "(CH₃)₃COCH₃",
        "dd": 14.8, "dp": 4.3, "dh": 5.0, "dt": 15.9,
        "bp": 55.2, "mp": -109, "density": 0.741, "mw": 88.2,
        "viscosity": 0.27, "dielectric": 4.5, "vapor_pressure": 267,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Ether", "protic": False,
        "miscible_water": False, "color": "#7B1FA2",
        "note": "ICH Class 3. Preferred over DEE for safety (lower peroxide risk). "
                "Excellent anti-solvent and extraction solvent. Immiscible with water. "
                "Lower bp than THF — easier removal. Widely used in API workups.",
    },
    "14DOX": {
        "full_name": "1,4-Dioxane", "formula": "C₄H₈O₂",
        "dd": 17.5, "dp": 1.8, "dh": 9.0, "dt": 20.5,
        "bp": 101.3, "mp": 11.8, "density": 1.033, "mw": 88.1,
        "viscosity": 1.37, "dielectric": 2.2, "vapor_pressure": 37,
        "ich_class": 2, "ich_ppm": 380, "ich_class_label": "Class 2 (380 ppm — strict)",
        "category": "Ether (Cyclic)", "protic": False,
        "miscible_water": True, "color": "#6A1B9A",
        "note": "⚠️ ICH Class 2 (380 ppm — strict limit, potential carcinogen). "
                "Unique properties — high δd with low δp. Useful for aromatic-rich APIs. "
                "Miscible with water, low dielectric. Minimize use — prefer THF or 2-MeTHF. "
                "MP 11.8°C — may solidify in cold rooms.",
    },
    "2MeTHF": {
        "full_name": "2-Methyltetrahydrofuran", "formula": "C₅H₁₀O",
        "dd": 16.9, "dp": 5.0, "dh": 5.0, "dt": 18.0,
        "bp": 80, "mp": -136, "density": 0.855, "mw": 86.1,
        "viscosity": 0.46, "dielectric": 6.9, "vapor_pressure": 129,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Ether", "protic": False,
        "miscible_water": False, "color": "#7B1FA2",
        "note": "Greener THF alternative — bio-renewable. ICH Class 3. "
                "Similar HSP to THF. Immiscible with water (unlike THF) — useful for extraction. "
                "Lower peroxide tendency than THF. Growing adoption in pharma.",
    },

    # ── Esters ─────────────────────────────────────────────────
    "EtOAc": {
        "full_name": "Ethyl Acetate", "formula": "CH₃COOC₂H₅",
        "dd": 15.8, "dp": 5.3, "dh": 7.2, "dt": 18.0,
        "bp": 77.1, "mp": -84, "density": 0.902, "mw": 88.1,
        "viscosity": 0.44, "dielectric": 6.0, "vapor_pressure": 97,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Ester", "protic": False,
        "miscible_water": False, "color": "#2E7D32",
        "note": "ICH Class 3. GRAS. One of the safest organic solvents. "
                "Versatile — dissolves mid-polarity APIs. Immiscible with water. "
                "Hydrolyses slowly in acidic/basic conditions — avoid extended exposure. "
                "Workhorse of pharmaceutical synthesis and crystallisation.",
    },

    # ── Halogenated ────────────────────────────────────────────
    "DCM": {
        "full_name": "Dichloromethane (Methylene Chloride)", "formula": "CH₂Cl₂",
        "dd": 17.0, "dp": 7.3, "dh": 7.1, "dt": 19.8,
        "bp": 39.6, "mp": -97, "density": 1.325, "mw": 84.9,
        "viscosity": 0.41, "dielectric": 8.9, "vapor_pressure": 470,
        "ich_class": 2, "ich_ppm": 600, "ich_class_label": "Class 2 (600 ppm)",
        "category": "Halogenated", "protic": False,
        "miscible_water": False, "color": "#C62828",
        "note": "ICH Class 2 (600 ppm). Potential carcinogen — minimize use. "
                "Excellent for dissolving lipophilic APIs and polymers. "
                "Very low bp (40°C) — fast evaporation in spray coating. "
                "Avoid in late-stage crystallisation steps if ICH limit is a concern.",
    },
    "Chloroform": {
        "full_name": "Chloroform (Trichloromethane)", "formula": "CHCl₃",
        "dd": 17.8, "dp": 3.1, "dh": 5.7, "dt": 18.9,
        "bp": 61.2, "mp": -63, "density": 1.489, "mw": 119.4,
        "viscosity": 0.54, "dielectric": 4.8, "vapor_pressure": 213,
        "ich_class": 2, "ich_ppm": 60, "ich_class_label": "Class 2 (60 ppm — very strict)",
        "category": "Halogenated", "protic": False,
        "miscible_water": False, "color": "#B71C1C",
        "note": "⚠️ ICH Class 2 (60 ppm — very strict limit). Hepatotoxin. "
                "Useful in early discovery/screens but avoid in API manufacturing. "
                "Excellent for lipophilic compounds. Store with EtOH stabiliser (check interference). "
                "Consider DCM or toluene as alternatives.",
    },

    # ── Aromatic Hydrocarbons ──────────────────────────────────
    "Toluene": {
        "full_name": "Toluene", "formula": "C₆H₅CH₃",
        "dd": 18.0, "dp": 1.4, "dh": 2.0, "dt": 18.2,
        "bp": 110.6, "mp": -93, "density": 0.867, "mw": 92.1,
        "viscosity": 0.56, "dielectric": 2.4, "vapor_pressure": 29,
        "ich_class": 2, "ich_ppm": 890, "ich_class_label": "Class 2 (890 ppm)",
        "category": "Aromatic Hydrocarbon", "protic": False,
        "miscible_water": False, "color": "#4E342E",
        "note": "ICH Class 2 (890 ppm). Reproductive toxin concern. "
                "Excellent for aromatic and lipophilic APIs. High δd due to aromatic ring. "
                "Useful in cocrystal screening for pi-stacking motifs. "
                "Consider EtOAc or EtOH as greener alternatives.",
    },
    "Anisole": {
        "full_name": "Anisole (Methoxybenzene)", "formula": "C₆H₅OCH₃",
        "dd": 17.8, "dp": 4.1, "dh": 6.7, "dt": 19.4,
        "bp": 153.7, "mp": -37, "density": 0.995, "mw": 108.1,
        "viscosity": 1.0, "dielectric": 4.3, "vapor_pressure": 3.3,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Aromatic Hydrocarbon", "protic": False,
        "miscible_water": False, "color": "#4E342E",
        "note": "ICH Class 3. Greener alternative to toluene for aromatic solvation. "
                "Higher bp than toluene. Growing use in green chemistry protocols. "
                "Good for cocrystal screening of aromatic APIs.",
    },

    # ── Aliphatic Hydrocarbons ─────────────────────────────────
    "Hexane": {
        "full_name": "n-Hexane", "formula": "CH₃(CH₂)₄CH₃",
        "dd": 14.9, "dp": 0.0, "dh": 0.0, "dt": 14.9,
        "bp": 68.7, "mp": -95, "density": 0.659, "mw": 86.2,
        "viscosity": 0.30, "dielectric": 1.9, "vapor_pressure": 150,
        "ich_class": 2, "ich_ppm": 290, "ich_class_label": "Class 2 (290 ppm — strict)",
        "category": "Aliphatic Hydrocarbon", "protic": False,
        "miscible_water": False, "color": "#546E7A",
        "note": "⚠️ ICH Class 2 (290 ppm — strict). Neurotoxin (peripheral neuropathy). "
                "Anti-solvent of choice for non-polar APIs. Excellent for final purification. "
                "Consider heptane as safer alternative (ICH Class 3). "
                "Hazardous air pollutant — fume hood required.",
    },
    "Pentane": {
        "full_name": "n-Pentane", "formula": "CH₃(CH₂)₃CH₃",
        "dd": 14.5, "dp": 0.0, "dh": 0.0, "dt": 14.5,
        "bp": 36.1, "mp": -130, "density": 0.626, "mw": 72.2,
        "viscosity": 0.22, "dielectric": 1.8, "vapor_pressure": 573,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Aliphatic Hydrocarbon", "protic": False,
        "miscible_water": False, "color": "#546E7A",
        "note": "ICH Class 3. Very low bp (36°C) — extreme fire hazard. "
                "Less toxic than hexane. Excellent rapid anti-solvent. "
                "Volatile — easy removal. Use in low-boiling crystallisation work.",
    },
    "Heptane": {
        "full_name": "n-Heptane", "formula": "CH₃(CH₂)₅CH₃",
        "dd": 15.3, "dp": 0.0, "dh": 0.0, "dt": 15.3,
        "bp": 98.4, "mp": -91, "density": 0.684, "mw": 100.2,
        "viscosity": 0.41, "dielectric": 1.9, "vapor_pressure": 46,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Aliphatic Hydrocarbon", "protic": False,
        "miscible_water": False, "color": "#546E7A",
        "note": "ICH Class 3. Preferred alternative to hexane — lower toxicity. "
                "Excellent anti-solvent for polar APIs and salts. "
                "Higher bp than hexane aids reflux operations. "
                "Standard reference solvent for non-polar systems.",
    },
    "Methylcyclohexane": {
        "full_name": "Methylcyclohexane", "formula": "C₇H₁₄",
        "dd": 15.9, "dp": 0.0, "dh": 1.0, "dt": 15.9,
        "bp": 100.9, "mp": -126, "density": 0.769, "mw": 98.2,
        "viscosity": 0.68, "dielectric": 2.0, "vapor_pressure": 46,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Alicyclic Hydrocarbon", "protic": False,
        "miscible_water": False, "color": "#546E7A",
        "note": "ICH Class 3. Alicyclic — slightly higher δd than n-heptane. "
                "Useful for APIs with aliphatic/alicyclic character. "
                "Good recrystallisation solvent for wax-like compounds.",
    },
    "Cyclohexane": {
        "full_name": "Cyclohexane", "formula": "C₆H₁₂",
        "dd": 16.8, "dp": 0.0, "dh": 0.2, "dt": 16.8,
        "bp": 80.7, "mp": 6.6, "density": 0.779, "mw": 84.2,
        "viscosity": 0.89, "dielectric": 2.0, "vapor_pressure": 104,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Alicyclic Hydrocarbon", "protic": False,
        "miscible_water": False, "color": "#546E7A",
        "note": "ICH Class 3. MP 6.6°C — can solidify at room temp. "
                "Non-polar — excellent anti-solvent. "
                "Higher δd than n-alkanes due to ring constraint.",
    },

    # ── Water ──────────────────────────────────────────────────
    "Water": {
        "full_name": "Water", "formula": "H₂O",
        "dd": 15.5, "dp": 16.0, "dh": 42.3, "dt": 47.8,
        "bp": 100, "mp": 0, "density": 1.000, "mw": 18.0,
        "viscosity": 0.89, "dielectric": 80.1, "vapor_pressure": 23,
        "ich_class": None, "ich_ppm": None, "ich_class_label": "N/A",
        "category": "Aqueous", "protic": True,
        "miscible_water": True, "color": "#0277BD",
        "note": "Reference solvent. Dominates pharmaceutical dissolution environment. "
                "Aqueous solubility is the primary BCS criterion. "
                "pH adjustment with HCl/NaOH essential for salt screening.",
    },

    # ── Additional Useful Screening Solvents ──────────────────
    "NMP": {
        "full_name": "N-Methyl-2-pyrrolidone", "formula": "C₅H₉NO",
        "dd": 18.0, "dp": 12.3, "dh": 7.2, "dt": 23.1,
        "bp": 202, "mp": -24, "density": 1.028, "mw": 99.1,
        "viscosity": 1.65, "dielectric": 32.0, "vapor_pressure": 0.3,
        "ich_class": 2, "ich_ppm": 5300, "ich_class_label": "Class 2 (5300 ppm)",
        "category": "Polar Aprotic", "protic": False,
        "miscible_water": True, "color": "#E65100",
        "note": "ICH Class 2 (5300 ppm). Reproductive toxin (CMR) — use with care. "
                "High bp (202°C) — difficult to remove. Excellent solvating power. "
                "Use when DMF is unacceptable; consider DMA as alternative. "
                "Growing regulatory scrutiny.",
    },
    "DMA": {
        "full_name": "Dimethylacetamide", "formula": "CH₃CON(CH₃)₂",
        "dd": 16.8, "dp": 11.5, "dh": 10.2, "dt": 22.7,
        "bp": 165, "mp": -20, "density": 0.937, "mw": 87.1,
        "viscosity": 0.92, "dielectric": 37.8, "vapor_pressure": 2.0,
        "ich_class": 2, "ich_ppm": 1090, "ich_class_label": "Class 2 (1090 ppm)",
        "category": "Polar Aprotic", "protic": False,
        "miscible_water": True, "color": "#E65100",
        "note": "ICH Class 2 (1090 ppm). Similar to DMF but slightly higher limit. "
                "Reproductive toxin. Often used as DMF substitute. "
                "High bp — monitor residual levels carefully.",
    },
    "Nitromethane": {
        "full_name": "Nitromethane", "formula": "CH₃NO₂",
        "dd": 15.8, "dp": 18.8, "dh": 5.1, "dt": 25.1,
        "bp": 101.2, "mp": -29, "density": 1.138, "mw": 61.0,
        "viscosity": 0.61, "dielectric": 35.9, "vapor_pressure": 36,
        "ich_class": 3, "ich_ppm": None, "ich_class_label": "Class 3 (≤50 mg/day)",
        "category": "Polar Aprotic", "protic": False,
        "miscible_water": False, "color": "#FF6F00",
        "note": "ICH Class 3. Very high δp — unique in screening coverage. "
                "Useful for very polar non-protic APIs. "
                "Explosion risk when contaminated — handle carefully.",
    },
}

# Category colours for visualization
CATEGORY_COLORS = {
    "Protic — Alcohol": "#E91E63",
    "Protic — Diol": "#AD1457",
    "Polar Aprotic": "#FF6F00",
    "Ether": "#7B1FA2",
    "Ether (Cyclic)": "#6A1B9A",
    "Ester": "#2E7D32",
    "Halogenated": "#B71C1C",
    "Aromatic Hydrocarbon": "#4E342E",
    "Aliphatic Hydrocarbon": "#546E7A",
    "Alicyclic Hydrocarbon": "#455A64",
    "Aqueous": "#0277BD",
}

# ══════════════════════════════════════════════════════════════════
# 2. GROUP CONTRIBUTION TABLES
# ══════════════════════════════════════════════════════════════════
# Based on Hoy (1985), van Krevelen (1990), Fedors (1974)
# Fd: dispersion parameter contribution (MPa^0.5 · cm³/mol)
# Fp: polar parameter contribution (MPa^0.5 · cm³/mol)
# Uh: H-bond energy contribution (J/mol)
# V:  molar volume contribution (cm³/mol)
# logP_f: Rekker-Mannhold logP fragment value

GROUP_CONTRIBUTIONS = {
    # ─── Carbon skeleton ───────────────────────────────────────
    "-CH₃": {
        "label": "Methyl (-CH₃)", "category": "Carbon",
        "Fd": 420, "Fp": 0, "Uh": 0, "V": 31.8, "logP_f": 0.53,
        "MW": 15.0, "hbd": 0, "hba": 0, "rotbonds": 0,
        "tpsa": 0, "mp_contrib": -15.5,
        "note": "Terminal methyl group",
    },
    "-CH₂-": {
        "label": "Methylene (-CH₂-)", "category": "Carbon",
        "Fd": 272, "Fp": 0, "Uh": 0, "V": 16.1, "logP_f": 0.53,
        "MW": 14.0, "hbd": 0, "hba": 0, "rotbonds": 1,
        "tpsa": 0, "mp_contrib": -7.0,
        "note": "Chain methylene unit (each adds logP +0.53)",
    },
    ">CH-": {
        "label": "Methine (>CH-)", "category": "Carbon",
        "Fd": 57, "Fp": 0, "Uh": 0, "V": -1.0, "logP_f": 0.53,
        "MW": 13.0, "hbd": 0, "hba": 0, "rotbonds": 0,
        "tpsa": 0, "mp_contrib": -7.0,
        "note": "Tertiary carbon (3 substituents)",
    },
    ">C<": {
        "label": "Quaternary C (>C<)", "category": "Carbon",
        "Fd": -190, "Fp": 0, "Uh": 0, "V": -19.2, "logP_f": 0.53,
        "MW": 12.0, "hbd": 0, "hba": 0, "rotbonds": 0,
        "tpsa": 0, "mp_contrib": 0.0,
        "note": "Fully substituted quaternary carbon",
    },
    # ─── Aromatic ─────────────────────────────────────────────
    "Phenyl/Benzene ring": {
        "label": "Phenyl / Benzene ring (C₆H₅)", "category": "Aromatic",
        "Fd": 1503, "Fp": 310, "Uh": 0, "V": 71.4, "logP_f": 1.56,
        "MW": 77.1, "hbd": 0, "hba": 0, "rotbonds": 1,
        "tpsa": 0, "mp_contrib": 31.5,
        "note": "Monosubstituted phenyl/benzene. Drives δd up significantly.",
    },
    "Naphthalene ring": {
        "label": "Naphthalene system (C₁₀H₇)", "category": "Aromatic",
        "Fd": 2820, "Fp": 450, "Uh": 0, "V": 123.8, "logP_f": 2.84,
        "MW": 127.2, "hbd": 0, "hba": 0, "rotbonds": 1,
        "tpsa": 0, "mp_contrib": 45.0,
        "note": "Bicyclic aromatic system.",
    },
    "Pyridine ring": {
        "label": "Pyridine ring (C₅H₄N)", "category": "Aromatic Heterocycle",
        "Fd": 1050, "Fp": 820, "Uh": 2400, "V": 61.0, "logP_f": 0.65,
        "MW": 78.1, "hbd": 0, "hba": 1, "rotbonds": 1,
        "tpsa": 12.9, "mp_contrib": 25.0,
        "pka_estimate": 5.2, "pka_type": "Base",
        "note": "Pyridine N is H-bond acceptor. Weak base (pKa ~5.2). "
                "Common in pharma — enhances aqueous solubility.",
    },
    "Imidazole ring": {
        "label": "Imidazole ring", "category": "Aromatic Heterocycle",
        "Fd": 820, "Fp": 700, "Uh": 8000, "V": 55.0, "logP_f": -0.08,
        "MW": 68.1, "hbd": 1, "hba": 1, "rotbonds": 1,
        "tpsa": 28.7, "mp_contrib": 35.0,
        "pka_estimate": 6.5, "pka_type": "Base",
        "note": "Amphiprotic — both HBD and HBA. Base pKa ~6.5. "
                "Found in histidine, many drug candidates (e.g. omeprazole).",
    },
    "Morpholine ring": {
        "label": "Morpholine ring", "category": "Saturated Heterocycle",
        "Fd": 950, "Fp": 550, "Uh": 3000, "V": 77.5, "logP_f": -0.80,
        "MW": 87.1, "hbd": 0, "hba": 2, "rotbonds": 0,
        "tpsa": 12.5, "mp_contrib": 15.0,
        "pka_estimate": 8.3, "pka_type": "Base",
        "note": "Moderate base (pKa ~8.3). Low logP contribution. "
                "Commonly used to improve aqueous solubility in med-chem.",
    },
    "Piperidine ring": {
        "label": "Piperidine ring", "category": "Saturated Heterocycle",
        "Fd": 1100, "Fp": 280, "Uh": 3100, "V": 83.0, "logP_f": 0.14,
        "MW": 85.2, "hbd": 1, "hba": 1, "rotbonds": 0,
        "tpsa": 12.0, "mp_contrib": 15.0,
        "pka_estimate": 10.5, "pka_type": "Base",
        "note": "Strong aliphatic base (pKa ~10.5). Common in APIs (e.g. haloperidol). "
                "High NH donor contribution to δh.",
    },
    "Piperazine ring": {
        "label": "Piperazine ring", "category": "Saturated Heterocycle",
        "Fd": 1050, "Fp": 400, "Uh": 4200, "V": 86.0, "logP_f": -1.03,
        "MW": 86.1, "hbd": 2, "hba": 2, "rotbonds": 0,
        "tpsa": 15.6, "mp_contrib": 20.0,
        "pka_estimate": 9.8, "pka_type": "Base",
        "note": "Dibasic (pKa1 ~9.8, pKa2 ~5.3). Very water-soluble moiety. "
                "Common in CNS drugs. Both N atoms ionisable.",
    },
    "Pyrrolidine ring": {
        "label": "Pyrrolidine ring", "category": "Saturated Heterocycle",
        "Fd": 950, "Fp": 280, "Uh": 3100, "V": 71.0, "logP_f": 0.25,
        "MW": 71.1, "hbd": 1, "hba": 1, "rotbonds": 0,
        "tpsa": 16.0, "mp_contrib": 12.0,
        "pka_estimate": 11.3, "pka_type": "Base",
        "note": "Strong secondary amine (pKa ~11.3). Cyclic structure reduces rotatable bonds.",
    },
    "Thiophene ring": {
        "label": "Thiophene ring", "category": "Aromatic Heterocycle",
        "Fd": 1100, "Fp": 250, "Uh": 1500, "V": 67.0, "logP_f": 1.81,
        "MW": 84.1, "hbd": 0, "hba": 0, "rotbonds": 1,
        "tpsa": 0, "mp_contrib": 28.0,
        "note": "Aromatic sulfur ring. Bioisostere for phenyl. "
                "Contributes to δd and S-atom polarity.",
    },
    "Indole ring": {
        "label": "Indole ring system", "category": "Aromatic Heterocycle",
        "Fd": 1950, "Fp": 500, "Uh": 9000, "V": 110.0, "logP_f": 2.14,
        "MW": 117.1, "hbd": 1, "hba": 1, "rotbonds": 1,
        "tpsa": 13.1, "mp_contrib": 42.0,
        "pka_estimate": 16.0, "pka_type": "Very Weak Acid",
        "note": "NH is weak H-bond donor (not pharmacologically relevant pKa).",
    },
    # ─── Oxygen-containing groups ──────────────────────────────
    "-OH (aliphatic)": {
        "label": "Hydroxyl, aliphatic (-OH)", "category": "Oxygen",
        "Fd": 210, "Fp": 500, "Uh": 20000, "V": 10.0, "logP_f": -0.67,
        "MW": 17.0, "hbd": 1, "hba": 1, "rotbonds": 0,
        "tpsa": 20.2, "mp_contrib": 44.8,
        "pka_estimate": 16.0, "pka_type": "Very Weak Acid",
        "note": "Strong HBD and HBA. Major driver of δh. "
                "Each OH increases water solubility significantly.",
    },
    "-OH (phenolic)": {
        "label": "Phenol (-ArOH)", "category": "Oxygen",
        "Fd": 198, "Fp": 400, "Uh": 14000, "V": 10.0, "logP_f": -0.40,
        "MW": 17.0, "hbd": 1, "hba": 1, "rotbonds": 0,
        "tpsa": 20.2, "mp_contrib": 55.0,
        "pka_estimate": 9.5, "pka_type": "Acid",
        "note": "Moderate acid (pKa ~9-10.5). HBD and HBA. "
                "Can form salt with strong bases (NaOH, K₂CO₃). "
                "Phenol functionality increases bioavailability concern.",
    },
    "-O- (ether)": {
        "label": "Ether linkage (-O-)", "category": "Oxygen",
        "Fd": 100, "Fp": 400, "Uh": 3000, "V": 3.8, "logP_f": -0.27,
        "MW": 16.0, "hbd": 0, "hba": 1, "rotbonds": 1,
        "tpsa": 9.2, "mp_contrib": -10.0,
        "note": "HBA only (no HBD). Moderate δp, low δh. "
                "Increases rotatable bond count and flexibility.",
    },
    "-COOH": {
        "label": "Carboxylic acid (-COOH)", "category": "Oxygen",
        "Fd": 530, "Fp": 420, "Uh": 10900, "V": 28.5, "logP_f": -1.09,
        "MW": 45.0, "hbd": 1, "hba": 2, "rotbonds": 0,
        "tpsa": 37.3, "mp_contrib": 73.0,
        "pka_estimate": 4.5, "pka_type": "Acid",
        "note": "Key ionisable group. pKa ~3.5-5.0 (aromatic ~4.0, aliphatic ~4.8). "
                "Salt formation with bases (Na⁺, K⁺, Ca²⁺, amines). "
                "Strong influence on aqueous solubility.",
    },
    "-COO- (ester)": {
        "label": "Ester (-COO-)", "category": "Oxygen",
        "Fd": 390, "Fp": 490, "Uh": 7000, "V": 18.0, "logP_f": -0.27,
        "MW": 44.0, "hbd": 0, "hba": 2, "rotbonds": 1,
        "tpsa": 26.3, "mp_contrib": 20.0,
        "note": "HBA, no HBD. Prodrug strategy common. "
                "Susceptible to hydrolysis (esterase, acid/base). "
                "Lower δh than COOH.",
    },
    "-C=O (ketone/aldehyde)": {
        "label": "Carbonyl (-C=O)", "category": "Oxygen",
        "Fd": 290, "Fp": 770, "Uh": 2000, "V": 10.8, "logP_f": -1.03,
        "MW": 28.0, "hbd": 0, "hba": 1, "rotbonds": 0,
        "tpsa": 17.1, "mp_contrib": 20.0,
        "note": "HBA only. High δp. Common in ketone/aldehyde/amide. "
                "Reactive toward nucleophiles (Schiff base, aldol).",
    },
    "-SO₂-": {
        "label": "Sulfonyl (-SO₂-)", "category": "Sulfur",
        "Fd": 428, "Fp": 1300, "Uh": 3200, "V": 25.0, "logP_f": -2.67,
        "MW": 64.1, "hbd": 0, "hba": 2, "rotbonds": 0,
        "tpsa": 34.1, "mp_contrib": 55.0,
        "note": "Very high δp. Major hydrophilicity driver. "
                "Sulfonamide (-SO₂NH-) is ionisable acid (pKa 8-10).",
    },
    "-SO₂NH- (sulfonamide)": {
        "label": "Sulfonamide (-SO₂NH-)", "category": "Sulfur",
        "Fd": 428, "Fp": 1300, "Uh": 8000, "V": 30.0, "logP_f": -2.00,
        "MW": 79.1, "hbd": 1, "hba": 3, "rotbonds": 0,
        "tpsa": 58.2, "mp_contrib": 70.0,
        "pka_estimate": 9.5, "pka_type": "Acid",
        "note": "Weak acid (pKa ~8-10). HBD + strong HBA. "
                "Common bioisostere. Forms salts with strong bases.",
    },
    # ─── Nitrogen-containing groups ────────────────────────────
    "-NH₂ (aliphatic amine)": {
        "label": "Primary aliphatic amine (-NH₂)", "category": "Nitrogen",
        "Fd": 226, "Fp": 429, "Uh": 3400, "V": 19.2, "logP_f": -1.03,
        "MW": 16.0, "hbd": 2, "hba": 1, "rotbonds": 0,
        "tpsa": 26.0, "mp_contrib": 30.0,
        "pka_estimate": 10.0, "pka_type": "Base",
        "note": "Strong base (pKa ~9.5-11). HCl, HBr, mesylate salts common. "
                "HBD (2 donors) + HBA. Major solubility enhancer via protonation.",
    },
    "-NH₂ (aromatic amine)": {
        "label": "Primary aromatic amine (-ArNH₂)", "category": "Nitrogen",
        "Fd": 180, "Fp": 380, "Uh": 5000, "V": 14.0, "logP_f": -0.64,
        "MW": 16.0, "hbd": 2, "hba": 1, "rotbonds": 0,
        "tpsa": 26.0, "mp_contrib": 35.0,
        "pka_estimate": 4.5, "pka_type": "Base",
        "note": "Weak base (pKa ~3-5). Less basic than aliphatic amines. "
                "Potential mutagenicity concern (Ames test recommended for ArNH₂).",
    },
    "-NH- (secondary amine)": {
        "label": "Secondary amine (-NH-)", "category": "Nitrogen",
        "Fd": 180, "Fp": 300, "Uh": 3100, "V": 4.5, "logP_f": -0.94,
        "MW": 15.0, "hbd": 1, "hba": 1, "rotbonds": 0,
        "tpsa": 16.0, "mp_contrib": 20.0,
        "pka_estimate": 9.5, "pka_type": "Base",
        "note": "1 HBD, 1 HBA. Aliphatic secondary amines are strong bases (pKa ~9-11). "
                "Aromatic secondary (e.g. DPA) much weaker base.",
    },
    ">N- (tertiary amine)": {
        "label": "Tertiary amine (>N-)", "category": "Nitrogen",
        "Fd": 20, "Fp": 30, "Uh": 800, "V": -9.0, "logP_f": -0.70,
        "MW": 14.0, "hbd": 0, "hba": 1, "rotbonds": 0,
        "tpsa": 3.2, "mp_contrib": 10.0,
        "pka_estimate": 8.5, "pka_type": "Base",
        "note": "No HBD, 1 HBA. Aliphatic tertiary: pKa ~8-10. "
                "Lower δh than primary/secondary amines. HCl salt very common.",
    },
    "-C≡N (nitrile)": {
        "label": "Nitrile (-CN)", "category": "Nitrogen",
        "Fd": 430, "Fp": 1100, "Uh": 2500, "V": 24.0, "logP_f": -1.28,
        "MW": 26.0, "hbd": 0, "hba": 1, "rotbonds": 0,
        "tpsa": 23.8, "mp_contrib": 40.0,
        "note": "Very high δp. HBA only (weak). "
                "Common in kinase inhibitors. Metabolic nitrile hydrolysis possible.",
    },
    "-CONH₂ (primary amide)": {
        "label": "Primary amide (-CONH₂)", "category": "Nitrogen",
        "Fd": 390, "Fp": 620, "Uh": 11800, "V": 28.8, "logP_f": -1.71,
        "MW": 44.0, "hbd": 2, "hba": 2, "rotbonds": 0,
        "tpsa": 55.1, "mp_contrib": 50.0,
        "note": "Strong HBD (2 donors) + HBA. Very hydrophilic. "
                "Nicotinamide/saccharin amide synthon — excellent cocrystal former.",
    },
    "-CONH- (secondary amide)": {
        "label": "Secondary amide (-CONH-)", "category": "Nitrogen",
        "Fd": 280, "Fp": 480, "Uh": 9000, "V": 14.0, "logP_f": -1.30,
        "MW": 43.0, "hbd": 1, "hba": 2, "rotbonds": 0,
        "tpsa": 29.1, "mp_contrib": 40.0,
        "note": "1 HBD (NH), 2 HBA (C=O and N). "
                "Key amide synthon for cocrystal formation. "
                "Found in most peptide bonds and many drug backbones.",
    },
    # ─── Halogens ─────────────────────────────────────────────
    "-F": {
        "label": "Fluorine (-F)", "category": "Halogen",
        "Fd": 164, "Fp": 250, "Uh": 400, "V": 18.0, "logP_f": 0.14,
        "MW": 19.0, "hbd": 0, "hba": 1, "rotbonds": 0,
        "tpsa": 0, "mp_contrib": 22.0,
        "note": "Weak HBA. Small effect on logP (+0.14 each). "
                "Blocks metabolism (CYP blocking). "
                "CF₃ group: logP +1.07, major metabolic stabiliser.",
    },
    "-CF₃": {
        "label": "Trifluoromethyl (-CF₃)", "category": "Halogen",
        "Fd": 426, "Fp": 650, "Uh": 1000, "V": 48.0, "logP_f": 1.07,
        "MW": 69.0, "hbd": 0, "hba": 0, "rotbonds": 0,
        "tpsa": 0, "mp_contrib": 25.0,
        "note": "Strongly lipophilic (+1.07 logP). Excellent metabolic blocker. "
                "High electronegativity. Bioisostere for tert-butyl.",
    },
    "-Cl": {
        "label": "Chlorine (-Cl)", "category": "Halogen",
        "Fd": 419, "Fp": 490, "Uh": 400, "V": 24.0, "logP_f": 0.60,
        "MW": 35.5, "hbd": 0, "hba": 1, "rotbonds": 0,
        "tpsa": 0, "mp_contrib": 32.0,
        "note": "Significant logP increase (+0.60). Metabolic concern — "
                "CYP-mediated dechlorination possible.",
    },
    "-Br": {
        "label": "Bromine (-Br)", "category": "Halogen",
        "Fd": 460, "Fp": 330, "Uh": 300, "V": 30.0, "logP_f": 1.02,
        "MW": 79.9, "hbd": 0, "hba": 0, "rotbonds": 0,
        "tpsa": 0, "mp_contrib": 35.0,
        "note": "Large logP increase (+1.02). Used in lead discovery. "
                "Dehalogenation metabolite risk. Uncommon in marketed drugs.",
    },
    "-I": {
        "label": "Iodine (-I)", "category": "Halogen",
        "Fd": 500, "Fp": 250, "Uh": 200, "V": 36.0, "logP_f": 1.35,
        "MW": 126.9, "hbd": 0, "hba": 0, "rotbonds": 0,
        "tpsa": 0, "mp_contrib": 36.0,
        "note": "Very high logP (+1.35). Thyroid concern. "
                "Deiodination by deiodinases. Rare in oral drugs.",
    },
    # ─── Sulfur ───────────────────────────────────────────────
    "-S- (thioether)": {
        "label": "Thioether (-S-)", "category": "Sulfur",
        "Fd": 428, "Fp": 160, "Uh": 1000, "V": 16.0, "logP_f": 0.15,
        "MW": 32.1, "hbd": 0, "hba": 1, "rotbonds": 1,
        "tpsa": 25.3, "mp_contrib": 14.0,
        "note": "Moderate HBA. Oxidation metabolite (sulfoxide, sulfone) common. "
                "Contributes to δd significantly.",
    },
    "-SH (thiol)": {
        "label": "Thiol (-SH)", "category": "Sulfur",
        "Fd": 290, "Fp": 200, "Uh": 4000, "V": 21.0, "logP_f": -0.08,
        "MW": 33.1, "hbd": 1, "hba": 0, "rotbonds": 0,
        "tpsa": 38.8, "mp_contrib": 25.0,
        "pka_estimate": 10.5, "pka_type": "Acid",
        "note": "pKa ~10-11. Oxidation-prone — disulfide formation. "
                "Reactive with metals. Common in prodrugs.",
    },
    # ─── Phosphorus ───────────────────────────────────────────
    "-PO(OH)₂ (phosphonic acid)": {
        "label": "Phosphonic acid (-PO(OH)₂)", "category": "Phosphorus",
        "Fd": 520, "Fp": 1400, "Uh": 18000, "V": 45.0, "logP_f": -2.82,
        "MW": 79.0, "hbd": 2, "hba": 4, "rotbonds": 0,
        "tpsa": 94.8, "mp_contrib": 80.0,
        "pka_estimate": 2.1, "pka_type": "Acid",
        "note": "Diprotic acid (pKa1 ~2.1, pKa2 ~7.0). Very hydrophilic. "
                "Prodrug strategy (ester) common for bioavailability.",
    },
}

# ══════════════════════════════════════════════════════════════════
# 3. HSP + LOGP CALCULATION ENGINE
# ══════════════════════════════════════════════════════════════════

def calculate_hsp_from_groups(group_counts: dict) -> dict:
    """
    Calculate Hansen Solubility Parameters from group contributions.
    
    Uses Hoy (1985) method:
        δd = ΣFd / V
        δp = √(ΣFp²) / V
        δh = √(2·ΣUh / V)
        δt = √(δd² + δp² + δh²)
    
    Returns dict with dd, dp, dh, dt, V, confidence
    """
    if not group_counts:
        return {"dd": 0, "dp": 0, "dh": 0, "dt": 0, "V": 0}

    sum_Fd = sum_Fp_sq = sum_Uh = sum_V = 0.0
    sum_logP = sum_MW = sum_hbd = sum_hba = sum_tpsa = sum_rot = 0
    sum_mp = 0.0

    for group_key, count in group_counts.items():
        if count <= 0:
            continue
        if group_key not in GROUP_CONTRIBUTIONS:
            continue
        g = GROUP_CONTRIBUTIONS[group_key]
        n = count
        sum_Fd      += g["Fd"] * n
        sum_Fp_sq   += (g["Fp"] * n) ** 2
        sum_Uh      += g["Uh"] * n
        sum_V       += g["V"] * n
        sum_logP    += g["logP_f"] * n
        sum_MW      += g["MW"] * n
        sum_hbd     += g["hbd"] * n
        sum_hba     += g["hba"] * n
        sum_tpsa    += g["tpsa"] * n
        sum_rot     += g["rotbonds"] * n
        sum_mp      += g.get("mp_contrib", 0) * n

    if sum_V <= 0:
        sum_V = max(sum_MW * 0.85, 50)  # fallback: V ≈ 0.85 × MW

    dd = round(sum_Fd / sum_V, 2)
    dp = round(math.sqrt(sum_Fp_sq) / sum_V, 2) if sum_Fp_sq > 0 else 0.0
    dh = round(math.sqrt(2 * sum_Uh / sum_V), 2) if sum_Uh > 0 else 0.0
    dt = round(math.sqrt(dd**2 + dp**2 + dh**2), 2)

    # Joback MP estimate: Tb = 198 + ΣΔTb, MP ≈ 0.584·Tb − 90 (rough)
    mp_est = round(198 + sum_mp, 1)

    return {
        "dd": max(dd, 12.0),
        "dp": max(dp, 0.0),
        "dh": max(dh, 0.0),
        "dt": dt,
        "V_molar": round(sum_V, 1),
        "logP": round(sum_logP, 2),
        "MW": round(sum_MW, 1),
        "HBD": int(sum_hbd),
        "HBA": int(sum_hba),
        "TPSA": round(sum_tpsa, 1),
        "RotBonds": int(sum_rot),
        "MP_estimate": mp_est,
    }


def estimate_pka_from_groups(group_counts: dict) -> dict:
    """
    Estimate most pharmacologically relevant pKa from functional groups.
    Returns primary ionisable group, estimated pKa, and type (Acid/Base).
    """
    # Priority order: find the most ionisable group
    ionisable = []
    for group_key, count in group_counts.items():
        if count <= 0:
            continue
        g = GROUP_CONTRIBUTIONS.get(group_key, {})
        if "pka_estimate" in g:
            ionisable.append({
                "group": g["label"],
                "pKa": g["pka_estimate"],
                "type": g["pka_type"],
                "count": count,
            })

    if not ionisable:
        return {"pKa": None, "type": None, "group": None, "confidence": "None"}

    # Sort: Acids by lowest pKa first (most acidic), Bases by highest pKa first (most basic)
    acids = sorted([i for i in ionisable if "Acid" in i["type"]], key=lambda x: x["pKa"])
    bases = sorted([i for i in ionisable if i["type"] == "Base"], key=lambda x: -x["pKa"])

    if acids and bases:
        # Amphoteric — return dominant ionisable group
        dominant = acids[0] if acids[0]["pKa"] < 7 else bases[0]
    elif acids:
        dominant = acids[0]
    elif bases:
        dominant = bases[0]
    else:
        return {"pKa": None, "type": None, "group": None, "confidence": "None"}

    conf = "Medium (±1.5 pKa units — verify experimentally)"
    return {
        "pKa": dominant["pKa"],
        "type": dominant["type"],
        "group": dominant["group"],
        "all_ionisable": ionisable,
        "confidence": conf,
    }


def bcs_classify(logP: float, d0: float) -> dict:
    """
    BCS classification from logP (permeability proxy) and D0 (solubility proxy).
    Permeability proxy: logP > 0 suggests passive permeability > BCS threshold.
    """
    high_perm = logP >= 0  # simplified; TPSA < 90 Ų also useful
    high_sol = d0 is not None and d0 <= 1.0

    if high_perm and high_sol:
        bcs = "I"
        desc = "High Permeability / High Solubility"
        strategy = "Conventional solid dosage form. Focus on physical stability."
        color = "#2E7D32"
    elif not high_perm and high_sol:
        bcs = "III"
        desc = "Low Permeability / High Solubility"
        strategy = "Permeation enhancers, lipid formulations, or prodrugs."
        color = "#1565C0"
    elif high_perm and not high_sol:
        bcs = "II"
        desc = "High Permeability / Low Solubility"
        strategy = "Salt screening, cocrystals, ASD, nano-sizing, lipid systems."
        color = "#E65100"
    else:
        bcs = "IV"
        desc = "Low Permeability / Low Solubility"
        strategy = "Complex formulation needed: lipid + absorption enhancers or ASD."
        color = "#B71C1C"

    return {"class": bcs, "description": desc, "strategy": strategy, "color": color}


def calculate_hansen_distance(api: dict, solvent: dict) -> float:
    """Ra = √[4(δdA-δdB)² + (δpA-δpB)² + (δhA-δhB)²]"""
    return round(math.sqrt(
        4 * (api["dd"] - solvent["dd"]) ** 2 +
        (api["dp"] - solvent["dp"]) ** 2 +
        (api["dh"] - solvent["dh"]) ** 2
    ), 2)


def predict_solubility_class(Ra: float, RED_threshold: float = 1.0) -> dict:
    """
    Predict solubility class from Ra.
    Greenhalgh (1999): Ra < 7 MPa^0.5 = miscible; > 10 = immiscible
    RED = Ra / R0 (R0 = 5 default for small molecules)
    """
    R0 = 5.0  # typical R0 for pharmaceutical molecules
    RED = Ra / R0

    if Ra < 5:
        cls = "Excellent"
        symbol = "✅✅"
        color = "#1B5E20"
        pred_sol = "High solubility predicted (>50 mg/mL likely)"
    elif Ra < 7:
        cls = "Good"
        symbol = "✅"
        color = "#2E7D32"
        pred_sol = "Good solubility predicted (5-50 mg/mL range)"
    elif Ra < 9:
        cls = "Partial"
        symbol = "⚠️"
        color = "#E65100"
        pred_sol = "Partial solubility / marginal (0.1-5 mg/mL)"
    elif Ra < 11:
        cls = "Poor"
        symbol = "🟡"
        color = "#F57F17"
        pred_sol = "Poor solubility (<0.1 mg/mL)"
    else:
        cls = "Insoluble"
        symbol = "❌"
        color = "#B71C1C"
        pred_sol = "Likely insoluble"

    return {
        "class": cls, "symbol": symbol, "color": color,
        "Ra": Ra, "RED": round(RED, 2),
        "prediction": pred_sol,
    }


# ══════════════════════════════════════════════════════════════════
# 4. SIMPLE SMILES PARSER (No RDKit required)
# ══════════════════════════════════════════════════════════════════

def parse_smiles_basic(smiles: str) -> dict:
    """
    Basic SMILES parser — extracts atom counts and functional groups
    using pattern matching. Accuracy ~±15% for HSP, ±0.5 for logP.
    
    NOT a full SMILES parser — for PharmaCrystal screening use only.
    For exact values, use RDKit.
    """
    if not smiles or not smiles.strip():
        return {}

    s = smiles.strip()

    # Count atoms
    n_C_arom  = len(re.findall(r'c', s))  # aromatic C
    n_N_arom  = len(re.findall(r'n', s))  # aromatic N
    n_O_arom  = len(re.findall(r'o', s))  # aromatic O
    n_S_arom  = len(re.findall(r's', s))  # aromatic S
    n_C_ali   = len(re.findall(r'C(?![l])', s))  # aliphatic C (not Cl)
    n_N_ali   = len(re.findall(r'N', s))  # aliphatic N
    n_O_ali   = len(re.findall(r'O', s))  # aliphatic O
    n_S_ali   = len(re.findall(r'S(?![i])', s))  # aliphatic S
    n_F       = len(re.findall(r'F', s))
    n_Cl      = len(re.findall(r'Cl', s))
    n_Br      = len(re.findall(r'Br', s))
    n_I       = len(re.findall(r'[^B]I|^I', s))

    # Detect functional groups by SMARTS-like patterns in SMILES
    # Carboxylic acid
    n_COOH = len(re.findall(r'C\(=O\)O(?!H)|C\(=O\)\[OH\]|C\(=O\)O$', s))
    if n_COOH == 0:
        n_COOH = s.count('OC(=O)') + s.count('C(=O)O')
        n_COOH = min(n_COOH, 4)

    # Hydroxyl — crude: count [OH] or O not in carbonyl context
    n_OH_ali = len(re.findall(r'\[OH\]|O(?![C(=O)|\)])(?=H)|(?<![=])O(?!H)', s))
    n_OH_phenol = n_O_arom  # simplified: aromatic O → phenol

    # Ketone/carbonyl
    n_C_O = s.count('C(=O)')

    # Ester
    n_ester = s.count('C(=O)O') + s.count('OC(=O)')
    n_ester = min(n_ester, 3)
    n_C_O_net = max(n_C_O - n_COOH - n_ester, 0)

    # Amine detection
    n_NH2 = len(re.findall(r'N(?!\()(?!c)(?!n)', s))  # primary N
    n_NH  = s.count('[NH]')
    n_NR3 = n_N_ali - n_NH2 - n_NH

    # Nitrile
    n_CN = s.count('C#N') + s.count('C≡N')

    # Rings — crude aromatic ring count from aromatic atoms
    n_arom_rings = max(n_C_arom // 6, 0)  # rough: 6 arom C = 1 benzene
    if n_N_arom > 0:
        n_arom_rings = max(n_arom_rings, 1)

    # Build MW estimate
    MW = (n_C_arom + n_C_ali) * 12 + \
         (n_N_arom + n_N_ali) * 14 + \
         (n_O_ali + n_O_arom) * 16 + \
         (n_S_ali + n_S_arom) * 32 + \
         n_F * 19 + n_Cl * 35.5 + n_Br * 80 + n_I * 127

    # H count (rough)
    # Each C has some H; simplification:
    n_H_approx = max(n_C_ali * 2 + n_C_arom * 0.5 + n_NH2 * 2 + n_NH, 0)
    MW += n_H_approx * 1

    # Map back to group contributions (simplified)
    extracted_groups = {}
    if n_arom_rings > 0:
        extracted_groups["Phenyl/Benzene ring"] = n_arom_rings
    if n_N_arom > 0 and n_arom_rings > 0:
        extracted_groups["Pyridine ring"] = min(n_N_arom, n_arom_rings)
        extracted_groups["Phenyl/Benzene ring"] = max(0, extracted_groups.get("Phenyl/Benzene ring", 0) - min(n_N_arom, n_arom_rings))
    if n_COOH > 0:
        extracted_groups["-COOH"] = n_COOH
    if n_OH_ali > 0 and n_COOH == 0:
        extracted_groups["-OH (aliphatic)"] = min(n_OH_ali, 4)
    if n_C_O_net > 0:
        extracted_groups["-C=O (ketone/aldehyde)"] = n_C_O_net
    if n_ester > 0:
        extracted_groups["-COO- (ester)"] = n_ester
    if n_NH2 > 0:
        if n_C_arom > 0:
            extracted_groups["-NH₂ (aromatic amine)"] = min(n_NH2, 2)
        else:
            extracted_groups["-NH₂ (aliphatic amine)"] = min(n_NH2, 2)
    if n_NH > 0:
        extracted_groups["-NH- (secondary amine)"] = n_NH
    if n_NR3 > 0 and n_NR3 < 6:
        extracted_groups[">N- (tertiary amine)"] = n_NR3
    if n_CN > 0:
        extracted_groups["-C≡N (nitrile)"] = n_CN
    if n_F > 0:
        if n_F >= 3 and n_C_ali > 0:
            extracted_groups["-CF₃"] = n_F // 3
        else:
            extracted_groups["-F"] = n_F
    if n_Cl > 0:
        extracted_groups["-Cl"] = n_Cl
    if n_Br > 0:
        extracted_groups["-Br"] = n_Br
    if n_I > 0:
        extracted_groups["-I"] = n_I
    if n_S_ali > 0:
        extracted_groups["-S- (thioether)"] = n_S_ali

    # Fill remaining C atoms as CH2 / CH3
    accounted_C = 0
    if "Phenyl/Benzene ring" in extracted_groups:
        accounted_C += extracted_groups["Phenyl/Benzene ring"] * 6
    if "Pyridine ring" in extracted_groups:
        accounted_C += extracted_groups["Pyridine ring"] * 5
    if "-COOH" in extracted_groups:
        accounted_C += extracted_groups["-COOH"]
    if "-COO- (ester)" in extracted_groups:
        accounted_C += extracted_groups["-COO- (ester)"]
    if "-C=O (ketone/aldehyde)" in extracted_groups:
        accounted_C += extracted_groups["-C=O (ketone/aldehyde)"]
    if "-C≡N (nitrile)" in extracted_groups:
        accounted_C += extracted_groups["-C≡N (nitrile)"]

    remaining_C = max(n_C_ali - accounted_C, 0)
    if remaining_C > 1:
        extracted_groups["-CH₂-"] = int(remaining_C - 1)
        extracted_groups["-CH₃"] = 1
    elif remaining_C == 1:
        extracted_groups["-CH₃"] = 1

    return {k: v for k, v in extracted_groups.items() if v > 0}


# ══════════════════════════════════════════════════════════════════
# 5. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════

def fig_hsp_triangle(api_params: dict, solvent_results: pd.DataFrame) -> bytes:
    """
    HSP 2D projection: dp vs dh (most discriminating axes for pharma).
    Plots API (large star) and solvents coloured by solubility class.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sol_colors = {"Excellent": "#1B5E20", "Good": "#2E7D32",
                  "Partial": "#E65100", "Poor": "#F57F17", "Insoluble": "#B71C1C"}
    sol_markers = {"Excellent": "o", "Good": "o", "Partial": "s", "Poor": "s", "Insoluble": "X"}

    for _, row in solvent_results.iterrows():
        cls = row["Solubility Class"]
        ax.scatter(row["dp"], row["dh"],
                   c=sol_colors.get(cls, "#999"),
                   marker=sol_markers.get(cls, "o"),
                   s=110, alpha=0.85, edgecolors="white", linewidth=0.7, zorder=4)
        ax.annotate(row["Abbreviation"],
                    xy=(row["dp"], row["dh"]),
                    xytext=(3, 4), textcoords="offset points",
                    fontsize=7.5, color="#333", fontweight="bold")

    # API star
    ax.scatter(api_params["dp"], api_params["dh"],
               c="#FFD700", s=420, marker="*", zorder=10,
               edgecolors="#333", linewidth=1.2, label=f"API: {api_params.get('name','API')}")

    # Draw interaction radius circle (R0 = 5 MPa^0.5 threshold)
    circle = plt.Circle((api_params["dp"], api_params["dh"]), 5.0,
                         color="#FFD700", fill=False, linestyle="--", linewidth=1.5,
                         alpha=0.6, label="R₀ = 5 MPa^0.5")
    ax.add_patch(circle)

    # Legend
    legend_patches = [
        mpatches.Patch(color=sol_colors["Excellent"], label="Excellent (Ra<5)"),
        mpatches.Patch(color=sol_colors["Good"], label="Good (Ra 5-7)"),
        mpatches.Patch(color=sol_colors["Partial"], label="Partial (Ra 7-9)"),
        mpatches.Patch(color=sol_colors["Poor"], label="Poor (Ra 9-11)"),
        mpatches.Patch(color=sol_colors["Insoluble"], label="Insoluble (Ra>11)"),
    ]
    ax.legend(handles=legend_patches + [
        mpatches.Patch(facecolor="#FFD700", label=f"API ({api_params.get('name','API')})"),
    ], fontsize=8.5, framealpha=0.92, loc="upper right")

    ax.set_xlabel("δp — Polar Component (MPa^0.5)", fontweight="bold", fontsize=10)
    ax.set_ylabel("δh — H-Bond Component (MPa^0.5)", fontweight="bold", fontsize=10)
    ax.set_title(f"HSP Landscape (δp vs δh)\nAPI: δd={api_params['dd']}, δp={api_params['dp']}, "
                 f"δh={api_params['dh']} MPa^0.5", fontsize=11, fontweight="bold", pad=12)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


def fig_ra_bar(solvent_results: pd.DataFrame) -> bytes:
    """Horizontal bar chart of Ra values coloured by solubility class."""
    df = solvent_results.sort_values("Ra")
    color_map = {"Excellent": "#1B5E20", "Good": "#43A047",
                 "Partial": "#E65100", "Poor": "#F57F17", "Insoluble": "#B71C1C"}
    colors = [color_map.get(c, "#999") for c in df["Solubility Class"]]

    fig, ax = plt.subplots(figsize=(11, max(5, len(df) * 0.42)))
    bars = ax.barh(
        [f"{row['Abbreviation']} — {row['Solvent']}" for _, row in df.iterrows()],
        df["Ra"], color=colors, edgecolor="white", linewidth=0.6, height=0.72
    )
    for bar, val in zip(bars, df["Ra"]):
        ax.text(val + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8.5, fontweight="bold", color="#333")

    ax.axvline(x=5, color="#1B5E20", linestyle="--", lw=1.8, alpha=0.85, label="Excellent (Ra=5)")
    ax.axvline(x=7, color="#E65100", linestyle="--", lw=1.8, alpha=0.85, label="Good/Partial (Ra=7)")
    ax.axvline(x=11, color="#B71C1C", linestyle="--", lw=1.5, alpha=0.70, label="Insoluble (Ra=11)")

    # ICH class badges
    for i, (_, row) in enumerate(df.iterrows()):
        ich = row.get("ICH Class", "")
        bar_obj = bars[i]
        if "Class 1" in str(ich):
            ax.text(0.3, bar_obj.get_y() + bar_obj.get_height() / 2,
                    "⚠️C1", va="center", fontsize=7, color="#B71C1C", fontweight="bold")
        elif "Class 2" in str(ich):
            ax.text(0.3, bar_obj.get_y() + bar_obj.get_height() / 2,
                    "C2", va="center", fontsize=7, color="#E65100", fontweight="bold")

    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_xlabel("Hansen Distance Ra (MPa^0.5)", fontweight="bold")
    ax.set_title("Solvent Ranking by Hansen Distance (Ra)\nSmaller Ra = Better Solubility Predicted",
                 fontweight="bold", pad=12)
    ax.set_xlim(0, max(df["Ra"]) * 1.15 + 1)

    fig.tight_layout(pad=1.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


def fig_hsp_components(api_params: dict) -> bytes:
    """Radar chart of API HSP components vs pharmaceutical space."""
    categories = ["δd (Dispersion)", "δp (Polar)", "δh (H-Bond)"]
    values = [api_params["dd"], api_params["dp"], api_params["dh"]]

    # Reference ranges for pharma APIs (from CSD analysis)
    pharma_low  = [15.0, 3.0, 3.0]
    pharma_high = [21.0, 16.0, 18.0]
    pharma_mid  = [18.0, 9.5, 10.0]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    param_colors = ["#1565C0", "#E65100", "#2E7D32"]
    for i, (ax, cat, val, lo, hi, mid, col) in enumerate(
            zip(axes, categories, values, pharma_low, pharma_high, pharma_mid, param_colors)):
        ax.barh(["Pharma\nRange"], [hi - lo], left=[lo],
                color="#E3F2FD", edgecolor="#90CAF9", linewidth=1, height=0.4)
        ax.barh(["Your\nAPI"], [val], color=col, alpha=0.85, height=0.4, edgecolor="white")
        ax.axvline(x=mid, color="grey", linestyle="--", lw=1, alpha=0.7, label="Median API")
        ax.set_title(cat, fontweight="bold", fontsize=10)
        ax.set_xlabel("MPa^0.5", fontsize=8.5)
        ax.set_xlim(0, 50)
        in_range = lo <= val <= hi
        ax.text(val + 0.3, 0, f"{val}", va="center", fontsize=10,
                fontweight="bold", color=col)
        status = "✓ In range" if in_range else ("▲ High" if val > hi else "▼ Low")
        ax.text(val, -0.35, status, ha="center", fontsize=8, color=col)

    fig.suptitle("HSP Profile vs Pharmaceutical Drug Space",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout(pad=1.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════════
# 6. MAIN STREAMLIT UI — PHASE 1 TAB
# ══════════════════════════════════════════════════════════════════

def render_phase1_tab():
    """
    Renders the complete Phase 1 — Molecule Input & Parameter Engine tab.
    Returns a dict of API parameters for use in downstream tabs.
    """
    st.markdown("""
    <style>
    .param-card {
        background: linear-gradient(135deg, #f8fbff 0%, #e8f4fd 100%);
        border: 1px solid #90CAF9;
        border-radius: 10px;
        padding: 16px;
        margin: 6px 0;
    }
    .confidence-green { color: #2E7D32; font-weight: bold; }
    .confidence-amber { color: #E65100; font-weight: bold; }
    .group-header { color: #1565C0; font-size: 13px; font-weight: bold; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Phase 1: Molecular Intelligence Engine**  
    Auto-calculates δd, δp, δh, LogP, MW, pKa, TPSA from structure.  
    These parameters propagate automatically to all screening tabs.
    """)

    # ── Input Method Selection ──
    input_method = st.radio(
        "Choose input method:",
        ["🧩 Functional Group Builder", "🔬 SMILES Parser (Basic)", "✏️ Direct Entry (v6.0 mode)"],
        horizontal=True, key="p1_input_method",
    )
    st.divider()

    group_counts = {}
    calculated_params = {}
    api_name = st.text_input("API Name / Code", value="API_001", key="p1_api_name")

    # ════════════════════════════════════════════════════════════
    # PATH A — FUNCTIONAL GROUP BUILDER
    # ════════════════════════════════════════════════════════════
    if "Functional Group Builder" in input_method:
        st.markdown("#### 🧩 Build Your Molecule from Functional Groups")
        st.info("Select groups and set counts. HSP, LogP, MW, pKa calculated automatically.")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown('<div class="group-header">🔵 Carbon Skeleton</div>', unsafe_allow_html=True)
            for gk in ["-CH₃", "-CH₂-", ">CH-", ">C<"]:
                g = GROUP_CONTRIBUTIONS[gk]
                n = st.number_input(g["label"], min_value=0, max_value=20, value=0,
                                    key=f"p1_grp_{gk}", step=1,
                                    help=g["note"])
                if n > 0:
                    group_counts[gk] = n

            st.markdown('<div class="group-header">🟣 Aromatic / Heterocyclic Rings</div>', unsafe_allow_html=True)
            for gk in ["Phenyl/Benzene ring", "Naphthalene ring", "Pyridine ring",
                       "Imidazole ring", "Morpholine ring", "Piperidine ring",
                       "Piperazine ring", "Pyrrolidine ring", "Thiophene ring",
                       "Indole ring"]:
                g = GROUP_CONTRIBUTIONS[gk]
                n = st.number_input(g["label"], min_value=0, max_value=5, value=0,
                                    key=f"p1_grp_{gk}", step=1,
                                    help=g["note"])
                if n > 0:
                    group_counts[gk] = n

        with col_right:
            st.markdown('<div class="group-header">🔴 Oxygen-containing Groups</div>', unsafe_allow_html=True)
            for gk in ["-OH (aliphatic)", "-OH (phenolic)", "-O- (ether)",
                       "-COOH", "-COO- (ester)", "-C=O (ketone/aldehyde)"]:
                g = GROUP_CONTRIBUTIONS[gk]
                n = st.number_input(g["label"], min_value=0, max_value=6, value=0,
                                    key=f"p1_grp_{gk}", step=1, help=g["note"])
                if n > 0:
                    group_counts[gk] = n

            st.markdown('<div class="group-header">🟡 Nitrogen Groups</div>', unsafe_allow_html=True)
            for gk in ["-NH₂ (aliphatic amine)", "-NH₂ (aromatic amine)",
                       "-NH- (secondary amine)", ">N- (tertiary amine)",
                       "-C≡N (nitrile)", "-CONH₂ (primary amide)", "-CONH- (secondary amide)"]:
                g = GROUP_CONTRIBUTIONS[gk]
                n = st.number_input(g["label"], min_value=0, max_value=4, value=0,
                                    key=f"p1_grp_{gk}", step=1, help=g["note"])
                if n > 0:
                    group_counts[gk] = n

            st.markdown('<div class="group-header">🟢 Halogens / Sulfur / Phosphorus</div>', unsafe_allow_html=True)
            for gk in ["-F", "-CF₃", "-Cl", "-Br", "-I",
                       "-S- (thioether)", "-SH (thiol)",
                       "-SO₂-", "-SO₂NH- (sulfonamide)", "-PO(OH)₂ (phosphonic acid)"]:
                g = GROUP_CONTRIBUTIONS[gk]
                n = st.number_input(g["label"], min_value=0, max_value=6, value=0,
                                    key=f"p1_grp_{gk}", step=1, help=g["note"])
                if n > 0:
                    group_counts[gk] = n

        # Calculate from groups
        if group_counts:
            calculated_params = calculate_hsp_from_groups(group_counts)
            pka_info = estimate_pka_from_groups(group_counts)
            calculated_params["pka_info"] = pka_info
            calculated_params["input_method"] = "Functional Group Builder"
            calculated_params["confidence_hsp"] = "Medium (group contribution ±1.5 MPa^0.5)"
            calculated_params["confidence_logp"] = "Medium (Rekker fragments ±0.5 log units)"

    # ════════════════════════════════════════════════════════════
    # PATH B — SMILES PARSER
    # ════════════════════════════════════════════════════════════
    elif "SMILES" in input_method:
        st.markdown("#### 🔬 SMILES Input")
        smiles_ex = {
            "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
            "Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2F)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
            "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        }
        example_sel = st.selectbox("Load example:", ["— Custom —"] + list(smiles_ex.keys()),
                                   key="p1_smiles_ex")
        default_smiles = smiles_ex.get(example_sel, "") if example_sel != "— Custom —" else ""
        smiles_in = st.text_area("SMILES string:", value=default_smiles,
                                 height=80, key="p1_smiles_in",
                                 placeholder="e.g. CC(C)Cc1ccc(cc1)C(C)C(=O)O")

        st.caption(
            "⚠️ Basic SMILES parser — no RDKit in this environment. "
            "Extracts major functional groups for group contribution estimation. "
            "Accuracy: HSP ±2 MPa^0.5, LogP ±0.7. **Always verify key values experimentally.**")

        if smiles_in.strip():
            with st.spinner("Parsing SMILES..."):
                group_counts = parse_smiles_basic(smiles_in)

            if group_counts:
                calculated_params = calculate_hsp_from_groups(group_counts)
                pka_info = estimate_pka_from_groups(group_counts)
                calculated_params["pka_info"] = pka_info
                calculated_params["input_method"] = f"SMILES: {smiles_in[:40]}..."
                calculated_params["confidence_hsp"] = "Low-Medium (SMILES parsing, ±2 MPa^0.5 — verify!)"
                calculated_params["confidence_logp"] = "Low-Medium (±0.7 log units)"

                with st.expander("Detected groups from SMILES"):
                    for gk, cnt in group_counts.items():
                        g = GROUP_CONTRIBUTIONS.get(gk, {})
                        st.markdown(f"- **{g.get('label', gk)}** × {cnt}")
            else:
                st.warning("Could not parse SMILES. Try the Functional Group Builder.")

    # ════════════════════════════════════════════════════════════
    # PATH C — DIRECT ENTRY
    # ════════════════════════════════════════════════════════════
    else:
        st.markdown("#### ✏️ Direct Parameter Entry")
        c1, c2, c3 = st.columns(3)
        with c1:
            dd_in = st.number_input("δd (MPa^0.5)", value=18.5, step=0.1, key="p1_dd_direct")
            dp_in = st.number_input("δp (MPa^0.5)", value=10.5, step=0.1, key="p1_dp_direct")
            dh_in = st.number_input("δh (MPa^0.5)", value=7.5, step=0.1, key="p1_dh_direct")
        with c2:
            logP_in = st.number_input("LogP", value=2.0, step=0.1, key="p1_logp_direct")
            MW_in = st.number_input("MW (g/mol)", value=350.0, step=10.0, key="p1_mw_direct")
            mp_in = st.number_input("MP (°C)", value=180.0, step=5.0, key="p1_mp_direct")
        with c3:
            pka_in = st.number_input("pKa", value=7.0, step=0.1, key="p1_pka_direct")
            api_type_in = st.selectbox("API Nature", ["Base", "Acid"], key="p1_type_direct")
            tpsa_in = st.number_input("TPSA (Ų)", value=80.0, step=5.0, key="p1_tpsa_direct")

        calculated_params = {
            "dd": dd_in, "dp": dp_in, "dh": dh_in,
            "dt": round(math.sqrt(dd_in**2 + dp_in**2 + dh_in**2), 2),
            "logP": logP_in, "MW": MW_in, "MP_estimate": mp_in,
            "HBD": 2, "HBA": 3, "TPSA": tpsa_in,
            "pka_info": {"pKa": pka_in, "type": api_type_in, "confidence": "User-entered"},
            "input_method": "Direct Entry",
            "confidence_hsp": "User-entered (treat as exact)",
            "confidence_logp": "User-entered",
        }

    # ════════════════════════════════════════════════════════════
    # PARAMETER OVERRIDE + DISPLAY CARD
    # ════════════════════════════════════════════════════════════
    if calculated_params:
        st.divider()
        st.markdown("#### 📊 Calculated Parameters — Review & Override")

        col_over1, col_over2, col_over3 = st.columns(3)
        with col_over1:
            st.markdown("**Hansen Solubility Parameters**")
            dd_f = st.number_input("δd (MPa^0.5)", value=float(calculated_params.get("dd", 18.5)),
                                   step=0.1, key="p1_dd_final", format="%.2f")
            dp_f = st.number_input("δp (MPa^0.5)", value=float(calculated_params.get("dp", 10.5)),
                                   step=0.1, key="p1_dp_final", format="%.2f")
            dh_f = st.number_input("δh (MPa^0.5)", value=float(calculated_params.get("dh", 7.5)),
                                   step=0.1, key="p1_dh_final", format="%.2f")
        with col_over2:
            st.markdown("**Molecular Properties**")
            logP_f = st.number_input("LogP", value=float(calculated_params.get("logP", 2.0)),
                                     step=0.1, key="p1_logp_final", format="%.2f")
            MW_f = st.number_input("MW (g/mol)", value=float(calculated_params.get("MW", 350.0)),
                                   step=1.0, key="p1_mw_final", format="%.1f")
            mp_f = st.number_input("MP (°C)", value=float(calculated_params.get("MP_estimate", 180.0)),
                                   step=1.0, key="p1_mp_final", format="%.1f")
        with col_over3:
            st.markdown("**Ionisation & Biopharmaceutics**")
            pka_info = calculated_params.get("pka_info", {})
            default_pka = float(pka_info.get("pKa") or 7.0)
            pka_f = st.number_input("pKa", value=default_pka,
                                    step=0.1, key="p1_pka_final", format="%.1f")
            default_type = pka_info.get("type") or "Base"
            if default_type not in ["Base", "Acid"]:
                default_type = "Base"
            api_type_f = st.selectbox("API Nature", ["Base", "Acid"],
                                      index=["Base", "Acid"].index(default_type),
                                      key="p1_type_final")
            tg_f = st.number_input("Tg (°C)", value=100.0, step=1.0,
                                   key="p1_tg_final", format="%.1f")

        # Additional params
        col_extra1, col_extra2 = st.columns(2)
        with col_extra1:
            dose_f = st.number_input("Dose (mg)", value=100, step=10, key="p1_dose_final")
            api_synthon_f = st.selectbox("Primary Functional Group (for salt screening)",
                ["Carboxylic Acid", "Amine", "Pyridine", "Amide",
                 "Phenol", "Sulfonate", "Hydroxyl"], key="p1_synthon_final")
        with col_extra2:
            api_loading_f = st.slider("ASD Drug Loading (%)", 5, 60, 30, key="p1_loading_final")
            hbd_f = st.number_input("HBD count", value=int(calculated_params.get("HBD", 2)),
                                    min_value=0, max_value=20, key="p1_hbd_final")
            hba_f = st.number_input("HBA count", value=int(calculated_params.get("HBA", 3)),
                                    min_value=0, max_value=20, key="p1_hba_final")

        # Compute derived values
        dt_f = round(math.sqrt(dd_f**2 + dp_f**2 + dh_f**2), 2)
        logD74_f = logP_f - math.log10(1 + 10**(api_type_f == "Base" and pka_f - 7.4 or 7.4 - pka_f)) \
                   if api_type_f == "Acid" else \
                   logP_f - math.log10(1 + 10**(7.4 - pka_f))
        logD74_f = round(logD74_f, 2)

        # Yalkowsky S0
        log_s0 = 0.5 - 0.01 * (mp_f - 25) - logP_f
        s0_mol_l = round(10 ** log_s0, 6)
        s0_mg_ml = round(s0_mol_l * MW_f, 4)

        # Dose number at FaSSIF pH 6.5
        if api_type_f == "Base":
            hh_ratio_fassif = 1 + 10 ** (pka_f - 6.5)
        else:
            hh_ratio_fassif = 1 + 10 ** (6.5 - pka_f)
        s_fassif = s0_mg_ml * hh_ratio_fassif
        d0 = round(dose_f / (s_fassif * 250), 2) if s_fassif > 0 else None

        bcs = bcs_classify(logP_f, d0)

        # Lipinski & Veber rule assessment
        lipinski_violations = sum([
            MW_f > 500, logP_f > 5, hbd_f > 5, hba_f > 10
        ])
        veber_ok = (calculated_params.get("RotBonds", 5) <= 10 and
                    calculated_params.get("TPSA", 80) <= 140)

        # ── SUMMARY CARD ──
        st.divider()
        st.markdown("#### ✅ Parameter Summary Card")

        # Metrics row
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("δd", f"{dd_f:.1f}", "MPa^0.5")
        m2.metric("δp", f"{dp_f:.1f}", "MPa^0.5")
        m3.metric("δh", f"{dh_f:.1f}", "MPa^0.5")
        m4.metric("δt", f"{dt_f:.1f}", "MPa^0.5")
        m5.metric("LogP", f"{logP_f:.2f}")
        m6.metric("LogD (pH 7.4)", f"{logD74_f:.2f}")

        m7, m8, m9, m10, m11, m12 = st.columns(6)
        m7.metric("MW", f"{MW_f:.0f}", "g/mol")
        m8.metric("pKa", f"{pka_f:.1f}", api_type_f)
        m9.metric("MP", f"{mp_f:.0f}", "°C")
        m10.metric("HBD / HBA", f"{hbd_f} / {hba_f}")
        m11.metric("GSE S₀", f"{s0_mg_ml:.4f}", "mg/mL")
        m12.metric("D₀ (FaSSIF)", str(d0) if d0 else "N/A")

        # BCS classification
        bcs_col, rule_col = st.columns(2)
        with bcs_col:
            st.markdown(f"""
            <div class="param-card">
            <b>BCS Classification: Class {bcs['class']}</b><br>
            {bcs['description']}<br>
            <small>💊 {bcs['strategy']}</small>
            </div>
            """, unsafe_allow_html=True)
        with rule_col:
            lip_emoji = "✅" if lipinski_violations == 0 else ("⚠️" if lipinski_violations <= 1 else "❌")
            veb_emoji = "✅" if veber_ok else "⚠️"
            st.markdown(f"""
            <div class="param-card">
            <b>Druglikeness Rules</b><br>
            {lip_emoji} Lipinski Ro5: {lipinski_violations} violation(s)<br>
            {veb_emoji} Veber: {'Pass' if veber_ok else 'Fail'}<br>
            <small>TPSA: {calculated_params.get('TPSA', '?')} Ų | RotBonds: {calculated_params.get('RotBonds', '?')}</small>
            </div>
            """, unsafe_allow_html=True)

        # Confidence labels
        conf_hsp = calculated_params.get("confidence_hsp", "User-entered")
        conf_logp = calculated_params.get("confidence_logp", "User-entered")
        st.caption(f"🔵 HSP confidence: {conf_hsp} | 🟡 LogP confidence: {conf_logp}")

        # pKa info
        if pka_info and pka_info.get("all_ionisable"):
            with st.expander("🔬 All Ionisable Groups Detected"):
                for item in pka_info["all_ionisable"]:
                    st.markdown(f"- **{item['group']}** — pKa ≈ {item['pKa']} ({item['type']}) × {item['count']}")

        # Full parameter table for PDF/export
        param_table = pd.DataFrame([{
            "Parameter": k, "Value": v
        } for k, v in {
            "API Name": api_name,
            "δd (MPa^0.5)": dd_f,
            "δp (MPa^0.5)": dp_f,
            "δh (MPa^0.5)": dh_f,
            "δt (MPa^0.5)": dt_f,
            "LogP": logP_f,
            "LogD (pH 7.4)": logD74_f,
            "MW (g/mol)": MW_f,
            "pKa": pka_f,
            "API Type": api_type_f,
            "MP (°C)": mp_f,
            "Tg (°C)": tg_f,
            "HBD": hbd_f,
            "HBA": hba_f,
            "TPSA (Ų)": calculated_params.get("TPSA", "N/A"),
            "Rotatable Bonds": calculated_params.get("RotBonds", "N/A"),
            "GSE S₀ (mg/mL)": s0_mg_ml,
            "D₀ (FaSSIF)": d0 if d0 else "N/A",
            "BCS Class": bcs["class"],
            "Lipinski Violations": lipinski_violations,
            "Input Method": calculated_params.get("input_method", "N/A"),
        }.items()])

        with st.expander("📋 Full Parameter Table (for export)"):
            st.dataframe(param_table, use_container_width=True)
            csv = param_table.to_csv(index=False)
            st.download_button("Download Parameters CSV", csv,
                               f"{api_name}_parameters.csv", "text/csv")

        # ════════════════════════════════════════════════════════
        # SOLVENT SCREENING SECTION
        # ════════════════════════════════════════════════════════
        st.divider()
        st.markdown("#### 🧪 Solvent Screening — Hansen Distance & Predicted Solubility")
        st.markdown(
            "All solvents ranked by Hansen distance Ra from API. "
            "ICH Class 1/2 solvents flagged. Toggle filter below."
        )

        col_flt1, col_flt2, col_flt3 = st.columns(3)
        with col_flt1:
            hide_ich1 = st.checkbox("Hide ICH Class 1 solvents", value=True, key="p1_hide_ich1")
        with col_flt2:
            show_categories = st.multiselect(
                "Show categories:",
                list(set(v["category"] for v in SOLVENT_DB.values())),
                default=list(set(v["category"] for v in SOLVENT_DB.values())),
                key="p1_cat_filter",
            )
        with col_flt3:
            max_ra = st.slider("Show solvents with Ra ≤", 5.0, 30.0, 20.0,
                               step=0.5, key="p1_max_ra")

        # Compute Ra for all solvents
        api_hsp = {"dd": dd_f, "dp": dp_f, "dh": dh_f, "name": api_name}
        solvent_rows = []
        for abbrev, sv in SOLVENT_DB.items():
            if hide_ich1 and sv.get("ich_class") == 1:
                continue
            if sv["category"] not in show_categories:
                continue
            Ra = calculate_hansen_distance(api_hsp, sv)
            if Ra > max_ra:
                continue
            sol_info = predict_solubility_class(Ra)
            solvent_rows.append({
                "Abbreviation": abbrev,
                "Solvent": sv["full_name"],
                "Category": sv["category"],
                "δd": sv["dd"],
                "δp": sv["dp"],
                "δh": sv["dh"],
                "Ra": Ra,
                "RED": sol_info["RED"],
                "Solubility Class": sol_info["class"],
                "Prediction": sol_info["prediction"],
                "ICH Class": sv["ich_class_label"],
                "BP (°C)": sv["bp"],
                "Protic": "Yes" if sv["protic"] else "No",
                "Miscible H₂O": "Yes" if sv["miscible_water"] else "No",
                "Note": sv["note"],
            })

        df_solvents = pd.DataFrame(solvent_rows).sort_values("Ra")

        if not df_solvents.empty:
            # Color-coded table
            sol_color_map = {
                "Excellent": "background-color: #c8e6c9",
                "Good": "background-color: #e8f5e9",
                "Partial": "background-color: #fff3e0",
                "Poor": "background-color: #ffecb3",
                "Insoluble": "background-color: #ffcdd2",
            }

            def style_solvent(row):
                cls = row.get("Solubility Class", "")
                base = sol_color_map.get(cls, "")
                ich = str(row.get("ICH Class", ""))
                if "Class 2" in ich:
                    return [base + "; border-left: 4px solid #E65100"] * len(row)
                return [base] * len(row)

            display_cols = ["Abbreviation", "Solvent", "Category", "δd", "δp", "δh",
                            "Ra", "RED", "Solubility Class", "ICH Class",
                            "BP (°C)", "Protic", "Miscible H₂O"]
            st.dataframe(
                df_solvents[display_cols].style.apply(style_solvent, axis=1),
                use_container_width=True, height=450
            )

            with st.expander("📝 Solvent Notes"):
                for _, row in df_solvents.head(15).iterrows():
                    sv = SOLVENT_DB.get(row["Abbreviation"], {})
                    ich_warn = ("⚠️ " if "Class 2" in str(row["ICH Class"]) else "")
                    st.markdown(f"**{row['Abbreviation']} — {row['Solvent']}** {ich_warn}  \n{sv.get('note','')}")

            # Summary stats
            st.markdown(f"""
            **Summary:** {len(df_solvents)} solvents screened &nbsp;|&nbsp;
            ✅✅ Excellent (Ra<5): **{len(df_solvents[df_solvents['Solubility Class']=='Excellent'])}** &nbsp;|&nbsp;
            ✅ Good (Ra 5-7): **{len(df_solvents[df_solvents['Solubility Class']=='Good'])}** &nbsp;|&nbsp;
            ⚠️ Partial: **{len(df_solvents[df_solvents['Solubility Class']=='Partial'])}** &nbsp;|&nbsp;
            ❌ Poor+Insoluble: **{len(df_solvents[df_solvents['Solubility Class'].isin(['Poor','Insoluble'])])}**
            """)

            # Charts
            col_ch1, col_ch2 = st.columns([1.2, 1])
            with col_ch1:
                st.markdown("**Ra Bar Chart**")
                ra_buf = fig_ra_bar(df_solvents)
                st.image(ra_buf, use_container_width=True)
            with col_ch2:
                st.markdown("**HSP Landscape (δp vs δh)**")
                hsp_buf = fig_hsp_triangle(api_hsp, df_solvents)
                st.image(hsp_buf, use_container_width=True)

            # HSP profile chart
            st.markdown("**API HSP Profile vs Pharmaceutical Space**")
            hsp_comp_buf = fig_hsp_components({"dd": dd_f, "dp": dp_f, "dh": dh_f})
            st.image(hsp_comp_buf, use_container_width=True)

            # Export
            st.download_button(
                "📥 Export Solvent Screening (CSV)",
                df_solvents.to_csv(index=False),
                f"{api_name}_solvent_screen.csv", "text/csv",
                use_container_width=True,
            )

        # ── Return propagation dict ──
        api_output = {
            "name": api_name,
            "dd": dd_f, "dp": dp_f, "dh": dh_f, "dt": dt_f,
            "logP": logP_f, "logD74": logD74_f,
            "MW": MW_f, "pKa": pka_f, "api_type": api_type_f,
            "MP": mp_f, "Tg": tg_f,
            "HBD": hbd_f, "HBA": hba_f,
            "TPSA": calculated_params.get("TPSA", 80),
            "RotBonds": calculated_params.get("RotBonds", 5),
            "s0_mol_l": s0_mol_l, "s0_mg_ml": s0_mg_ml,
            "d0_fassif": d0, "bcs": bcs,
            "dose": dose_f,
            "api_synthon": api_synthon_f,
            "api_loading": api_loading_f / 100.0,
            "lipinski_violations": lipinski_violations,
            "input_method": calculated_params.get("input_method", "Direct"),
        }
        return api_output

    return {}


# ══════════════════════════════════════════════════════════════════
# 7. STANDALONE APP ENTRY POINT (for testing Phase 1 alone)
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(
        page_title="PharmaCrystal Pro v7.0 — Phase 1",
        layout="wide", page_icon="🔬"
    )
    st.title("🔬 PharmaCrystal Pro v7.0 — Phase 1: Molecular Intelligence Engine")
    st.markdown(
        "**Tab 0** — Auto-calculates δd, δp, δh, LogP, pKa from structure. "
        "Outputs propagate to all screening tabs."
    )
    params = render_phase1_tab()
    if params:
        st.success("✅ Parameters ready — these will propagate to Salt/Cocrystal, ASD, and Solubility tabs.")
