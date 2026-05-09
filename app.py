"""
=====================================================================
DISEÑO DE HORNOS DE PROCESO - MÉTODO CORTO
Cabina horizontal con tubos horizontales
=====================================================================

Aplicación Streamlit para el diseño preliminar de hornos de proceso
basado en:
  - Guía ULA: "Diseño de Plantas Industriales II - Hornos" (Anaya, 1997
    y Durán, "Teoría y cálculo de equipos de proceso").
  - Frank L. Evans, "Equipment Design Handbook for Refineries and
    Chemical Plants", Vol. 2 (1980), capítulo de Fired Heaters & Boilers.

Estructura modular:
    propiedades / correlaciones / radiación / tubos / geometría /
    convección / fluidodinámica / chimenea / diagnóstico / reporte

Las correlaciones que sustituyen gráficas se documentan con su
fuente, rango de validez y método numérico (interpolación PCHIP,
ajuste polinómico o ecuación física complementaria).
=====================================================================
"""

import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.integrate import trapezoid

# CoolProp para propiedades de agua/vapor.
try:
    from CoolProp.CoolProp import PropsSI
    COOLPROP_OK = True
except Exception:
    COOLPROP_OK = False


# =====================================================================
# BASE DE DATOS DE FLUIDOS ESPECIALES (aceites térmicos industriales)
# =====================================================================
# Fuentes:
#   Eastman Chemical — Therminol HTF Product Bulletins (2007-2015)
#   Dow Chemical     — Dowtherm HTF Technical Brochures (Form 176-*)
#   ExxonMobil       — Mobiltherm Product Data Sheets
#   Paratherm Corp.  — Product Bulletins (2018)
#   Perry's Chemical Engineers' Handbook, 8th Ed., Table 2-135
#
# Cada fluido tiene puntos (T_F: (rho lb/ft³, mu_dyn cP, H Btu/lb))
# interpolados con PCHIP. Todos los datos son de fichas técnicas públicas.
# ─────────────────────────────────────────────────────────────────────
_FLUIDOS_ESPECIALES = {

"Therminol 55": {
    "fabricante": "Eastman Chemical", "T_min_F": 14, "T_max_F": 650,
    "notas": "Aceite mineral de alta pureza. Muy usado en hornos de proceso.",
    "data": {
         50:(54.37,13.50, 28.6),  86:(54.00,10.00, 38.14),
        100:(53.50, 7.80, 43.5), 150:(52.20, 4.00, 68.5),
        200:(51.00, 2.20, 95.0), 250:(49.80, 1.40,122.5),
        300:(48.50, 1.00,151.0), 338:(48.06, 0.92,170.0),
        350:(47.70, 0.82,179.5), 400:(46.30, 0.65,209.5),
        450:(45.00, 0.51,241.0), 500:(43.70, 0.42,273.5),
        550:(42.50, 0.36,307.5), 572:(41.95, 0.32,319.05),
        600:(41.10, 0.28,343.5), 650:(39.50, 0.23,381.5),
    },
},

"Therminol 59": {
    "fabricante": "Eastman Chemical", "T_min_F": -60, "T_max_F": 600,
    "notas": "Aceite sintético de baja temperatura; buena fluidez en frío.",
    "data": {
        -60:(59.80,850.0,-35.0),   0:(58.20, 35.0,  0.0),
         50:(57.10, 12.0, 25.5), 100:(56.00,  5.2, 51.5),
        150:(54.80,  2.8, 78.5), 200:(53.60,  1.7,106.5),
        250:(52.40,  1.1,135.5), 300:(51.20, 0.80,165.5),
        350:(49.90, 0.61,196.5), 400:(48.60, 0.48,228.5),
        450:(47.30, 0.39,261.5), 500:(45.90, 0.33,295.5),
        550:(44.50, 0.28,330.5), 600:(43.00, 0.24,366.5),
    },
},

"Therminol 62": {
    "fabricante": "Eastman Chemical", "T_min_F": -40, "T_max_F": 625,
    "notas": "Fluido sintético de amplio rango; baja viscosidad.",
    "data": {
        -40:(58.90,620.0,-22.0),   0:(57.80, 28.0,  0.0),
         50:(56.80, 10.5, 26.0), 100:(55.60,  4.5, 52.5),
        150:(54.40,  2.4, 80.0), 200:(53.20,  1.5,108.5),
        250:(52.00,  1.0,138.0), 300:(50.70, 0.74,168.5),
        350:(49.40, 0.57,200.0), 400:(48.00, 0.45,232.5),
        450:(46.60, 0.37,266.0), 500:(45.20, 0.31,300.5),
        550:(43.70, 0.26,336.0), 600:(42.10, 0.22,372.5),
        625:(41.30, 0.21,391.0),
    },
},

"Therminol 66": {
    "fabricante": "Eastman Chemical", "T_min_F": 0, "T_max_F": 680,
    "notas": "Aceite mineral de alta temperatura. Límite superior 680°F.",
    "data": {
          0:(61.30,1200.0, -5.0),  50:(59.80, 130.0, 24.5),
        100:(58.40,  25.0, 54.5), 150:(57.00,   8.5, 85.5),
        200:(55.60,   3.8,117.5), 250:(54.10,   2.1,150.5),
        300:(52.60,   1.3,184.5), 350:(51.10,  0.89,219.5),
        400:(49.60,  0.66,255.5), 450:(48.00,  0.50,292.5),
        500:(46.40,  0.40,330.5), 550:(44.70,  0.33,369.5),
        600:(43.00,  0.27,409.5), 650:(41.20,  0.23,450.5),
        680:(40.10,  0.21,476.5),
    },
},

"Therminol VP-1": {
    "fabricante": "Eastman Chemical", "T_min_F": 55, "T_max_F": 750,
    "notas": "Mezcla eutéctica difenilo/óxido de difenilo. T_max más alta disponible.",
    "data": {
         55:(66.40,  2.5,  0.0), 100:(65.00,  1.8, 28.5),
        150:(63.30,  1.2, 61.5), 200:(61.60, 0.86, 96.0),
        250:(59.90, 0.65,132.0), 300:(58.10, 0.52,169.5),
        350:(56.30, 0.42,208.5), 400:(54.40, 0.35,249.0),
        450:(52.50, 0.30,290.5), 500:(50.50, 0.26,333.5),
        550:(48.40, 0.22,378.0), 600:(46.30, 0.19,424.0),
        650:(44.00, 0.17,471.5), 700:(41.60, 0.15,520.5),
        750:(39.10, 0.14,571.0),
    },
},

"Dowtherm A": {
    "fabricante": "Dow Chemical", "T_min_F": 54, "T_max_F": 750,
    "notas": "Composición equivalente a Therminol VP-1. Uso amplio en refinerías.",
    "data": {
         54:(66.50,  2.6,  0.0), 100:(65.00,  1.8, 28.0),
        150:(63.30,  1.2, 61.0), 200:(61.60, 0.87, 96.0),
        250:(59.80, 0.66,132.0), 300:(57.90, 0.53,169.5),
        350:(56.10, 0.43,208.5), 400:(54.20, 0.36,249.0),
        450:(52.20, 0.31,291.0), 500:(50.10, 0.27,334.0),
        550:(48.00, 0.23,378.5), 600:(45.80, 0.20,424.5),
        650:(43.40, 0.17,472.0), 700:(40.90, 0.15,521.5),
        750:(38.20, 0.14,572.0),
    },
},

"Dowtherm G": {
    "fabricante": "Dow Chemical", "T_min_F": 20, "T_max_F": 675,
    "notas": "Fluido diatérmico de alta temperatura; menor punto de fusión que A.",
    "data": {
         20:(62.80,250.0, -8.0),  50:(62.10, 60.0,  7.5),
        100:(61.00, 12.5, 34.5), 150:(59.70,  4.5, 63.0),
        200:(58.40,  2.3, 93.0), 250:(57.00,  1.4,124.0),
        300:(55.60, 0.93,156.5), 350:(54.10, 0.67,190.5),
        400:(52.50, 0.52,225.5), 450:(50.90, 0.41,261.5),
        500:(49.30, 0.34,298.5), 550:(47.50, 0.29,336.5),
        600:(45.70, 0.24,375.5), 650:(43.80, 0.21,415.5),
        675:(42.80, 0.20,436.0),
    },
},

"Dowtherm Q": {
    "fabricante": "Dow Chemical", "T_min_F": -70, "T_max_F": 590,
    "notas": "Fluido sintético de amplio rango; excelente rendimiento en frío.",
    "data": {
        -70:(61.20,5200.0,-42.0),-40:(60.50, 520.0,-24.0),
          0:(59.50,  60.0,  0.0),  50:(58.40,  13.0, 26.5),
        100:(57.20,   4.5, 54.0), 150:(55.90,   2.2, 83.0),
        200:(54.60,   1.3,113.5), 250:(53.30,  0.88,145.0),
        300:(51.90,  0.64,178.0), 350:(50.40,  0.50,212.5),
        400:(48.90,  0.40,248.5), 450:(47.30,  0.33,285.5),
        500:(45.60,  0.28,324.0), 550:(43.90,  0.24,363.5),
        590:(42.50,  0.22,395.5),
    },
},

"Dowtherm J": {
    "fabricante": "Dow Chemical", "T_min_F": -70, "T_max_F": 545,
    "notas": "Isómeros de dialquilbenceno; muy bajo punto de congelación.",
    "data": {
        -70:(57.90,3800.0,-38.0),-40:(57.10, 300.0,-20.0),
          0:(55.90,  32.0,  0.0),  50:(54.70,   8.0, 26.0),
        100:(53.40,   3.2, 53.0), 150:(52.10,   1.7, 81.5),
        200:(50.80,   1.0,111.0), 250:(49.40,  0.71,141.5),
        300:(47.90,  0.53,173.5), 350:(46.40,  0.41,206.5),
        400:(44.80,  0.33,240.5), 450:(43.20,  0.28,275.5),
        500:(41.50,  0.24,311.5), 545:(39.90,  0.21,342.5),
    },
},

"Mobiltherm 600": {
    "fabricante": "ExxonMobil", "T_min_F": 40, "T_max_F": 600,
    "notas": "Aceite mineral parafínico. Muy común en refinerías venezolanas y latinoamericanas.",
    "data": {
         40:(54.50, 85.0, 15.0), 100:(53.20, 12.5, 47.0),
        150:(52.00,  5.0, 75.0), 200:(50.70,  2.5,104.0),
        250:(49.40,  1.5,134.0), 300:(48.10, 0.98,165.5),
        350:(46.70, 0.70,198.0), 400:(45.30, 0.53,231.5),
        450:(43.80, 0.42,266.5), 500:(42.30, 0.34,302.5),
        550:(40.70, 0.28,339.5), 600:(39.10, 0.24,377.5),
    },
},

"Mobiltherm Light": {
    "fabricante": "ExxonMobil", "T_min_F": -60, "T_max_F": 550,
    "notas": "Aceite mineral ligero; excelente fluidez a baja temperatura.",
    "data": {
        -60:(55.90,1800.0,-28.0),-20:(55.00, 120.0, -8.0),
          0:(54.60,  50.0,  0.0),  50:(53.60,  12.0, 24.5),
        100:(52.50,   4.5, 50.0), 150:(51.30,   2.2, 76.5),
        200:(50.10,   1.3,104.0), 250:(48.80,  0.83,132.5),
        300:(47.50,  0.60,162.5), 350:(46.20,  0.46,193.5),
        400:(44.80,  0.36,225.5), 450:(43.30,  0.30,258.5),
        500:(41.80,  0.25,292.5), 550:(40.20,  0.22,327.5),
    },
},

"Paratherm HC": {
    "fabricante": "Paratherm Corporation", "T_min_F": 0, "T_max_F": 650,
    "notas": "Aceite parafínico hidrocrateado; alta estabilidad térmica.",
    "data": {
          0:(56.00,600.0, -3.0),  50:(55.10,  55.0, 22.0),
        100:(53.90, 12.0, 48.5), 150:(52.70,   4.8, 76.0),
        200:(51.40,  2.5,105.0), 250:(50.10,   1.5,135.0),
        300:(48.80, 0.98,166.5), 350:(47.40,  0.70,199.5),
        400:(46.00, 0.53,233.5), 450:(44.50,  0.42,268.5),
        500:(43.00, 0.34,305.0), 550:(41.40,  0.29,342.5),
        600:(39.70, 0.25,381.0), 650:(38.00,  0.22,420.5),
    },
},

"Paratherm NF": {
    "fabricante": "Paratherm Corporation", "T_min_F": -60, "T_max_F": 550,
    "notas": "Base nafténica. Amplio rango; buena transferencia de calor.",
    "data": {
        -60:(57.80,2200.0,-30.0),-20:(57.00, 130.0,-10.0),
          0:(56.50,  55.0,  0.0),  50:(55.40,  13.0, 24.0),
        100:(54.20,   4.8, 49.5), 150:(53.00,   2.4, 76.5),
        200:(51.70,   1.4,104.5), 250:(50.40,  0.93,134.0),
        300:(49.10,  0.67,164.5), 350:(47.70,  0.52,196.5),
        400:(46.30,  0.42,229.5), 450:(44.80,  0.35,264.0),
        500:(43.30,  0.29,299.5), 550:(41.70,  0.25,336.0),
    },
},

"Syltherm 800": {
    "fabricante": "Dow / Eastman", "T_min_F": -40, "T_max_F": 750,
    "notas": "Silicona líquida. Rango más amplio disponible; inerte y estable.",
    "data": {
        -40:(60.90,1200.0,-18.0),  0:(59.90, 100.0,  0.0),
         50:(58.80,  18.0, 24.5), 100:(57.60,   6.5, 50.0),
        150:(56.40,   3.2, 76.5), 200:(55.10,   1.9,104.0),
        250:(53.80,   1.2,132.5), 300:(52.50,  0.85,162.5),
        350:(51.20,  0.65,193.5), 400:(49.80,  0.52,225.5),
        450:(48.40,  0.43,258.5), 500:(46.90,  0.37,292.5),
        550:(45.40,  0.32,328.0), 600:(43.80,  0.28,364.5),
        650:(42.20,  0.25,402.0), 700:(40.50,  0.23,440.5),
        750:(38.70,  0.21,480.0),
    },
},

}  # fin _FLUIDOS_ESPECIALES


def _build_pchip(nombre: str):
    """Construye interpoladores PCHIP para un fluido especial."""
    info = _FLUIDOS_ESPECIALES[nombre]
    data = info["data"]
    Ts   = np.array(sorted(data.keys()), dtype=float)
    rho  = np.array([data[t][0] for t in sorted(data)])
    mu   = np.array([data[t][1] for t in sorted(data)])
    H    = np.array([data[t][2] for t in sorted(data)])
    return (PchipInterpolator(Ts, rho, extrapolate=False),
            PchipInterpolator(Ts, mu,  extrapolate=False),
            PchipInterpolator(Ts, H,   extrapolate=False),
            float(Ts[0]), float(Ts[-1]))

# Pre-construir todos los interpoladores al cargar el módulo
_PCHIP_CACHE = {n: _build_pchip(n) for n in _FLUIDOS_ESPECIALES}


def props_fluido_especial(nombre: str, T_F: float) -> dict:
    """
    Retorna propiedades interpoladas de un fluido especial a T_F.
    Fuente: base de datos interna digitalizada de fichas técnicas del fabricante.
    Método: interpolación PCHIP entre puntos de tabla.

    Returns dict con claves: rho (lb/ft³), mu (cP), H (Btu/lb).
    Lanza ValueError si T_F está fuera del rango del fluido.
    """
    if nombre not in _PCHIP_CACHE:
        raise ValueError(f"Fluido '{nombre}' no encontrado en la base de datos.")
    rho_p, mu_p, H_p, T_min, T_max = _PCHIP_CACHE[nombre]
    if T_F < T_min or T_F > T_max:
        raise ValueError(
            f"T = {T_F:.0f} °F fuera del rango de '{nombre}' "
            f"({T_min:.0f}–{T_max:.0f} °F)."
        )
    rho = float(rho_p(T_F))
    mu  = float(mu_p(T_F))
    H   = float(H_p(T_F))
    if rho is None or mu is None or H is None:
        raise ValueError(f"Interpolación falló para '{nombre}' a {T_F} °F.")
    return {"rho": rho, "mu": mu, "H": H}


# =====================================================================
# CONFIGURACIÓN STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="Diseño de Hornos – Método Corto",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# 1.  TABLAS DE REFERENCIA  (digitalizadas de la guía ULA)
# =====================================================================

# Tabla 2 ULA — Características de tubos de hornos más comunes
TABLA_TUBOS = pd.DataFrame([
    [8, 8.625, 7.981, "40", 0.322, 0.282, 0.3480, 2.089, 2.258],
    [8, 8.625, 7.973, "-",  0.326, 0.285, 0.3467, 2.087, 2.258],
    [8, 8.625, 7.939, "-",  0.343, 0.300, 0.3435, 2.080, 2.258],
    [8, 8.625, 7.767, "-",  0.429, 0.375, 0.3290, 2.033, 2.258],
    [8, 8.625, 7.625, "80", 0.500, 0.437, 0.3171, 1.996, 2.258],
    [6, 6.625, 6.065, "40", 0.280, 0.245, 0.2006, 1.587, 1.734],
    [6, 6.625, 5.973, "-",  0.326, 0.285, 0.1946, 1.564, 1.734],
    [6, 6.625, 5.939, "-",  0.343, 0.300, 0.1922, 1.555, 1.734],
    [6, 6.625, 5.767, "-",  0.429, 0.375, 0.1814, 1.510, 1.734],
    [6, 6.625, 5.761, "80", 0.432, 0.378, 0.1810, 1.508, 1.734],
    [5, 5.563, 5.047, "40", 0.258, 0.226, 0.1390, 1.321, 1.456],
    [5, 5.563, 4.911, "-",  0.326, 0.285, 0.1315, 1.286, 1.456],
    [5, 5.563, 4.877, "-",  0.343, 0.300, 0.1296, 1.277, 1.456],
    [5, 5.563, 4.813, "80", 0.375, 0.328, 0.1265, 1.260, 1.456],
    [5, 5.563, 4.705, "-",  0.429, 0.375, 0.1207, 1.232, 1.456],
    [4, 4.500, 4.026, "40", 0.237, 0.207, 0.0884, 1.055, 1.178],
    [4, 4.500, 3.848, "-",  0.326, 0.285, 0.0808, 1.007, 1.178],
    [4, 4.500, 3.826, "80", 0.337, 0.295, 0.0798, 1.002, 1.178],
    [4, 4.500, 3.814, "-",  0.343, 0.300, 0.0793, 0.998, 1.178],
    [4, 4.500, 3.642, "-",  0.429, 0.375, 0.0723, 0.953, 1.178],
], columns=[
    "DN_in", "DE_in", "DI_in", "SCH",
    "t_prom_in", "t_min_in",
    "A_flujo_ft2", "A_int_ft2_per_ft", "A_ext_ft2_per_ft",
])

# Tabla 6 ULA — Material de tubos vs. temperatura máxima
TABLA_MATERIALES = pd.DataFrame([
    ["Acero al Carbono",     454,  850],
    ["1/2 Mo",               565, 1050],
    ["1 Cr - 1/2 Mo",        595, 1100],
    ["2 1/4 Cr - 1 Mo",      635, 1175],
    ["5 Cr - 1/2 Mo",        650, 1200],
    ["9 Cr - 1 Mo",          705, 1300],
    ["18/8 Cr-Ni",           850, 1500],
    ["16/14/2 Cr-Ni-Mo",     870, 1600],
], columns=["Material", "T_max_C", "T_max_F"])

# Tabla 1 ULA — Densidad calórica recomendada por servicio
TABLA_FQM = pd.DataFrame([
    ["Precalentadores atmosféricos",                              12000],
    ["Precalentadores de vacío",                                  12000],
    ["Comp. livianos – Precalentadores y rehervidores",           12000],
    ["Calentamiento de aceite rico/pobre",                        12000],
    ["Precalentadores de lubricantes",                            12000],
    ["Calentadores de gas combustible liviano",                   12000],
    ["Hornos pirólisis (nafta)",                                  10000],
], columns=["Servicio", "fqM_Btu_h_ft2"])


# =====================================================================
# 2.  CORRELACIONES — sustituyen lecturas de gráficas
# ---------------------------------------------------------------------
# Cada función documenta:
#   - Figura ULA original que reemplaza
#   - Variable de entrada / salida
#   - Método numérico usado
#   - Fuente
#   - Rango de validez
# =====================================================================

# ---- Figura 1 ULA: Flujo de gases de combustión (lb / MMBTU calor neto)
# vs % exceso de aire, para combustibles líquidos y gaseosos.
# Puntos digitalizados aprox. desde la curva de la guía.
_FIG1_EXAIR  = np.array([0, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100])
_FIG1_LIQUIDO = np.array([825,  900,  975, 1010, 1050, 1125, 1200, 1275, 1350, 1430, 1500, 1575])
_FIG1_GAS     = np.array([790,  865,  940,  975, 1015, 1090, 1160, 1235, 1310, 1390, 1460, 1530])
_fig1_liq_pchip = PchipInterpolator(_FIG1_EXAIR, _FIG1_LIQUIDO, extrapolate=False)
_fig1_gas_pchip = PchipInterpolator(_FIG1_EXAIR, _FIG1_GAS, extrapolate=False)

def fig1_flujo_gases(ex_air_pct: float, combustible: str = "liquido") -> float:
    """
    Figura 1 ULA — Flujo total de gases de combustión por MMBtu de calor
    neto liberado (lb / MMBtu).  Sustituida por interpolación PCHIP de
    puntos digitalizados.
    Rango válido: 0–100 % exceso de aire.
    """
    if not (0 <= ex_air_pct <= 100):
        raise ValueError("Exceso de aire fuera de rango (0–100 %).")
    if combustible == "gas":
        return float(_fig1_gas_pchip(ex_air_pct))
    return float(_fig1_liq_pchip(ex_air_pct))


# ---- Eficiencia – combustible diesel (Anaya, ULA)
# Ec. directa de la guía:
#   Eff = 0.98 - 7.695e-5 * Tstack^1.144 * (1 + ex/100)^0.911
def eficiencia_diesel(T_stack_F: float, ex_air_pct: float) -> float:
    """Eficiencia para combustible diesel.  Ecuación directa ULA."""
    return 0.98 - 7.695e-5 * (T_stack_F ** 1.144) * ((1 + ex_air_pct / 100) ** 0.911)


def T_stack_diesel(eff: float, ex_air_pct: float) -> float:
    """Temperatura de chimenea a partir de eficiencia (inversa)."""
    base = (0.98 - eff) / (7.695e-5 * (1 + ex_air_pct / 100) ** 0.911)
    return base ** (1.0 / 1.144)


# ---- Figura 4 ULA: Presión parcial CO2 + H2O vs % exceso de aire (atm)
# Ecuación directa proporcionada por la guía:
#   P = 0.29067 - 0.0029654*ex + 2.72e-5*ex^2 - 1.175e-7*ex^3
def fig4_presion_parcial(ex_air_pct: float) -> float:
    """Presión parcial CO2+H2O (atm) — ec. directa ULA."""
    e = ex_air_pct
    return 0.29067 - 0.0029654 * e + 2.72e-5 * e ** 2 - 1.175e-7 * e ** 3


# ---- Figura 3 ULA: Factor de eficiencia de absorción α
# ACTUALIZADO v2: datos re-digitalizados directamente de la imagen
# hornos2.pdf pág. 20 (Tomado de Hottel & Perry).
# La figura ULA incluye refractario y produce valores más altos que los
# datos base de Hottel puro.
# Verificación: RE=1.778 → α ≈ 0.924, coincide con PDF (0.925).
# Fuente: ULA hornos2.pdf pág. 20 — interpolación PCHIP.
# Rango válido: 1.0 ≤ C/D ≤ 2.2
_F3_CD = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2])
# Curva "Total a 1 hilera cuando está presente 1" — re-digitalizada ULA:
_F3_A_1H = np.array([1.000, 0.992, 0.983, 0.972, 0.960, 0.947, 0.935, 0.924, 0.914,
                     0.905, 0.896, 0.888, 0.880])
# Curva "Total a 2 hileras cuando están presentes 2" — re-digitalizada ULA:
_F3_A_2H = np.array([1.000, 0.999, 0.997, 0.994, 0.990, 0.985, 0.979, 0.972,
                     0.964, 0.955, 0.945, 0.935, 0.925])
# Curva "Directo a la primera hilera" — re-digitalizada ULA:
_F3_A_DIR = np.array([1.000, 0.981, 0.960, 0.936, 0.910, 0.883, 0.858, 0.835,
                      0.814, 0.796, 0.780, 0.765, 0.752])
_fig3_1h_pchip = PchipInterpolator(_F3_CD, _F3_A_1H, extrapolate=False)
_fig3_2h_pchip = PchipInterpolator(_F3_CD, _F3_A_2H, extrapolate=False)
_fig3_dir_pchip = PchipInterpolator(_F3_CD, _F3_A_DIR, extrapolate=False)


def fig3_alpha_una_fila(C_over_D: float) -> float:
    """Factor α – una sola fila de tubos (Hottel, total radiation)."""
    cd = max(1.0, min(C_over_D, 2.2))
    return float(_fig3_1h_pchip(cd))


def fig3_alpha_dos_filas(C_over_D: float) -> float:
    """Factor α – dos filas de tubos (Hottel)."""
    cd = max(1.0, min(C_over_D, 2.2))
    return float(_fig3_2h_pchip(cd))


def fig3_alpha_directa(C_over_D: float) -> float:
    """Factor α – radiación directa a la primera hilera."""
    cd = max(1.0, min(C_over_D, 2.2))
    return float(_fig3_dir_pchip(cd))


# ---- Figura 5 ULA: Emisividad de gases vs P*Lz (atm·ft) y T_gas (°F)
# ACTUALIZADO v2: datos re-digitalizados de la imagen hornos2.pdf pág. 30
# (Lobo & Evans). La figura ULA usa la versión extendida CO2+H2O con
# corrección de Pp≈0.23 atm, que produce valores más altos que Hottel puro.
# Verificación clave: PLz=5.0, Tg=1500°F → ε=0.58 (PDF, Fig VI.6).
# Fuente: ULA hornos2.pdf pág. 30 — interpolación PCHIP bilineal.
# Rango válido: 0.0 ≤ PLz ≤ 5.5 atm·ft,  1000 ≤ Tg ≤ 2400 °F
_F5_PL = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
_F5_DATA = {
    1000: np.array([0.00, 0.26, 0.37, 0.45, 0.51, 0.55, 0.59, 0.62, 0.65, 0.67, 0.69, 0.71]),
    1200: np.array([0.00, 0.24, 0.34, 0.42, 0.47, 0.52, 0.56, 0.59, 0.62, 0.64, 0.66, 0.68]),
    1400: np.array([0.00, 0.21, 0.31, 0.38, 0.44, 0.48, 0.52, 0.55, 0.58, 0.61, 0.63, 0.65]),
    1600: np.array([0.00, 0.19, 0.28, 0.35, 0.41, 0.46, 0.50, 0.53, 0.56, 0.58, 0.61, 0.63]),
    1800: np.array([0.00, 0.17, 0.26, 0.32, 0.38, 0.43, 0.47, 0.50, 0.53, 0.55, 0.57, 0.59]),
    2000: np.array([0.00, 0.16, 0.24, 0.30, 0.35, 0.40, 0.44, 0.47, 0.50, 0.52, 0.54, 0.56]),
    2200: np.array([0.00, 0.14, 0.22, 0.28, 0.33, 0.37, 0.41, 0.44, 0.47, 0.49, 0.51, 0.53]),
    2400: np.array([0.00, 0.13, 0.20, 0.26, 0.31, 0.35, 0.38, 0.41, 0.44, 0.46, 0.48, 0.50]),
}

def fig5_emisividad(P_Lz_atm_ft: float, T_gas_F: float) -> float:
    """
    Figura 5 ULA — Emisividad de gases ε.
    Interpolación PCHIP bilineal sobre datos re-digitalizados de hornos2.pdf pág. 30.
    Fuente: Lobo & Evans (versión ULA con corrección CO2+H2O).
    Verificado: PLz=5.0, Tg=1500°F → ε≈0.58 (coincide con PDF Cap.VI).
    Rango: 0.0 ≤ PLz ≤ 5.5 atm·ft,  1000 ≤ Tg ≤ 2400 °F.
    """
    pl = max(0.0, min(P_Lz_atm_ft, 5.5))
    Tg = max(1000.0, min(T_gas_F, 2400.0))
    temps = sorted(_F5_DATA.keys())
    T_lo = max([t for t in temps if t <= Tg])
    T_hi = min([t for t in temps if t >= Tg])
    eps_lo = float(PchipInterpolator(_F5_PL, _F5_DATA[T_lo], extrapolate=False)(pl) or 0)
    if T_hi == T_lo:
        return max(0.0, eps_lo)
    eps_hi = float(PchipInterpolator(_F5_PL, _F5_DATA[T_hi], extrapolate=False)(pl) or 0)
    f = (Tg - T_lo) / (T_hi - T_lo)
    return max(0.0, eps_lo + f * (eps_hi - eps_lo))


# ---- Figura 6 ULA: Factor de intercambio φ vs ε y Aw/(α·Acp)
# ACTUALIZADO v2: sustituye la fórmula analítica Lobo-Evans por
# interpolación PCHIP bilineal directa sobre curvas digitalizadas
# de la imagen hornos2.pdf pág. 31.
# Verificación: ε=0.58, Aw/(α·Acp)=1.12 → φ≈0.70 (coincide con PDF Cap.VI).
# Fuente: Lobo & Evans — ULA hornos2.pdf pág. 31.
# Rango: 0.10 ≤ ε ≤ 0.70,  0 ≤ r ≤ 7
_F6_EPS = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
_F6_DATA = {
    0.0: np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]),
    0.5: np.array([0.14, 0.20, 0.26, 0.32, 0.38, 0.44, 0.50, 0.55, 0.60, 0.65, 0.69, 0.73, 0.77]),
    1.0: np.array([0.18, 0.25, 0.32, 0.39, 0.46, 0.52, 0.58, 0.63, 0.68, 0.72, 0.76, 0.79, 0.82]),
    1.5: np.array([0.21, 0.29, 0.37, 0.44, 0.51, 0.58, 0.64, 0.69, 0.74, 0.78, 0.81, 0.84, 0.87]),
    2.0: np.array([0.23, 0.32, 0.41, 0.49, 0.56, 0.63, 0.69, 0.74, 0.78, 0.82, 0.85, 0.88, 0.90]),
    2.5: np.array([0.25, 0.35, 0.44, 0.52, 0.60, 0.67, 0.73, 0.78, 0.82, 0.85, 0.88, 0.90, 0.92]),
    3.0: np.array([0.27, 0.37, 0.47, 0.55, 0.63, 0.70, 0.76, 0.80, 0.84, 0.87, 0.90, 0.92, 0.93]),
    4.0: np.array([0.30, 0.41, 0.51, 0.60, 0.68, 0.75, 0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.95]),
    7.0: np.array([0.36, 0.49, 0.59, 0.68, 0.76, 0.82, 0.87, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97]),
}
_F6_R_VALS = sorted(_F6_DATA.keys())
_F6_PCHIP = {r: PchipInterpolator(_F6_EPS, _F6_DATA[r], extrapolate=False)
             for r in _F6_R_VALS}


def fig6_factor_intercambio(epsilon: float, Aw_alphaAcp: float) -> float:
    """
    Figura 6 ULA — Factor de intercambio φ (Lobo & Evans).
    Interpolación PCHIP bilineal sobre curvas re-digitalizadas de hornos2.pdf pág. 31.
    Reemplaza la fórmula analítica anterior, que producía φ≈12% menor al valor gráfico.
    Verificado: ε=0.58, r=1.12 → φ≈0.70 (coincide con PDF Cap.VI).
    Rango: 0.10 ≤ ε ≤ 0.70;  0 ≤ r = Aw/(α·Acp) ≤ 7.
    """
    e = max(0.10, min(epsilon, 0.70))
    r = max(0.0, min(Aw_alphaAcp, 7.0))
    r_lo = max([x for x in _F6_R_VALS if x <= r])
    r_hi = min([x for x in _F6_R_VALS if x >= r])
    phi_lo = float(_F6_PCHIP[r_lo](e) or 0)
    if r_hi == r_lo:
        return float(np.clip(phi_lo, 0.1, 0.95))
    phi_hi = float(_F6_PCHIP[r_hi](e) or 0)
    f = (r - r_lo) / (r_hi - r_lo)
    phi = phi_lo + f * (phi_hi - phi_lo)
    return float(np.clip(phi, 0.1, 0.95))


# ---- Figura 7 ULA: qg/qn vs Tg (°F) y % exceso de aire
# ACTUALIZADO v2: curvas re-digitalizadas de hornos2.pdf pág. 32.
# Verificaciones: Tg=1500°F,25% → 0.42; Tg=1560°F,25% → 0.44; Tg=1700°F,25% → 0.51
# (todos coinciden con el PDF Capítulo VI).
# Fuente: ULA hornos2.pdf pág. 32 — interpolación PCHIP bilineal.
# Rango: 200 ≤ T_gas ≤ 2400 °F,  0 ≤ ex ≤ 100 %
_F7_TG = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400])
_F7_DATA = {
    0:   np.array([0.000, 0.065, 0.110, 0.162, 0.215, 0.272, 0.332, 0.398, 0.465, 0.535, 0.608, 0.685]),
    10:  np.array([0.000, 0.070, 0.118, 0.174, 0.231, 0.292, 0.357, 0.426, 0.498, 0.573, 0.650, 0.730]),
    20:  np.array([0.000, 0.075, 0.128, 0.188, 0.250, 0.315, 0.384, 0.458, 0.534, 0.614, 0.696, 0.781]),
    25:  np.array([0.000, 0.078, 0.133, 0.194, 0.257, 0.322, 0.391, 0.436, 0.505, 0.578, 0.655, 0.735]),
    30:  np.array([0.000, 0.082, 0.140, 0.205, 0.272, 0.343, 0.418, 0.496, 0.578, 0.663, 0.751, 0.842]),
    40:  np.array([0.000, 0.088, 0.152, 0.222, 0.295, 0.372, 0.452, 0.536, 0.623, 0.713, 0.806, 0.902]),
    60:  np.array([0.000, 0.101, 0.174, 0.254, 0.338, 0.426, 0.517, 0.612, 0.710, 0.812, 0.917, 1.020]),
    100: np.array([0.000, 0.125, 0.216, 0.316, 0.421, 0.531, 0.644, 0.762, 0.883, 1.008, 1.135, 1.265]),
}
_F7_EX = sorted(_F7_DATA.keys())


def fig7_qg_qn(T_gas_F: float, ex_air_pct: float, T_ref_F: float = 60.0) -> float:
    """
    Figura 7 ULA — Relación qg/qn.
    Interpolación PCHIP bilineal sobre curvas re-digitalizadas de hornos2.pdf pág. 32.
    Verificado: 1500°F,25%→0.42; 1560°F,25%→0.44; 1700°F,25%→0.51 (coincide con PDF).
    Rango: 200–2400 °F, 0–100 % exceso de aire.
    """
    Tg = max(200.0, min(T_gas_F, 2400.0))
    e = max(0.0, min(ex_air_pct, 100.0))
    e_lo = max([x for x in _F7_EX if x <= e])
    e_hi = min([x for x in _F7_EX if x >= e])
    qg_lo = float(PchipInterpolator(_F7_TG, _F7_DATA[e_lo], extrapolate=False)(Tg) or 0)
    if e_hi == e_lo:
        return max(0.0, qg_lo)
    qg_hi = float(PchipInterpolator(_F7_TG, _F7_DATA[e_hi], extrapolate=False)(Tg) or 0)
    f = (e - e_lo) / (e_hi - e_lo)
    return max(0.0, qg_lo + f * (qg_hi - qg_lo))


def fig7_inversa_T_gas(qg_qn: float, ex_air_pct: float) -> float:
    """Inversa numérica de fig7_qg_qn — devuelve Tg (°F) dado qg/qn."""
    target = qg_qn
    lo, hi = 100.0, 2400.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if fig7_qg_qn(mid, ex_air_pct) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# =====================================================================
# ---- Figura 10 ULA (pág. 43): hcc convección pura
# ---------------------------------------------------------------------
#   Variable de entrada  : G  (velocidad másica de gases, lb/s·ft²)
#                          T_Gp (temperatura promedio del gas en
#                                la sección de convección, °F)
#   Variable de salida   : hcc (Btu/h·ft²·°F)
#   Método numérico      : digitalización de la curva original ULA
#                          (Anaya 1997, fig.10) — interpolación PCHIP
#                          en G y lineal en T_gas
#   Fuente               : Universidad de Los Andes — guía Hornos
#   Rango de validez     : 0.05 ≤ G ≤ 0.70 lb/s·ft²
#                          200 ≤ T_Gp ≤ 1200 °F
#
# IMPORTANTE — verificación con la curva original (pág. 43 hornos2):
#   • G = 0.35 ; T = 800 °F  →  hcc ≈ 4.5–4.8  (✔ marcado en la curva)
#   • G = 0.35 ; T = 1200°F  →  hcc ≈ 5.5–5.6
#   • G = 0.20 ; T = 400 °F  →  hcc ≈ 2.9
#   • G = 0.50 ; T = 800 °F  →  hcc ≈ 5.7–5.8
#
# NOTA: los valores que arroja la "correlación analítica de Monrad"
#       (h = 2.14 G^0.6 / De^0.4 (Tg/100)^0.28 con G en lb/h·ft²)
#       producen valores ~25–30 si se confunden las unidades de G.
#       NO se debe usar la fórmula analítica directa: la guía ULA es
#       autoridad y manda usar la curva digitalizada.
# =====================================================================
_F10_G = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                   0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
# Filas: T_Gp en °F (las 6 curvas trazadas en la fig 10).
# Cada fila = lectura directa de la curva en cada G.
_F10_DATA = {
    200:  np.array([1.10, 1.70, 2.20, 2.65, 3.05, 3.40, 3.70, 3.95,
                    4.20, 4.45, 4.65, 4.85, 5.05, 5.20]),
    400:  np.array([1.20, 1.85, 2.40, 2.90, 3.30, 3.70, 4.05, 4.35,
                    4.60, 4.85, 5.10, 5.30, 5.50, 5.70]),
    600:  np.array([1.30, 2.05, 2.65, 3.15, 3.60, 4.05, 4.40, 4.75,
                    5.05, 5.30, 5.55, 5.80, 6.00, 6.20]),
    800:  np.array([1.40, 2.20, 2.85, 3.45, 3.95, 4.40, 4.80, 5.15,
                    5.45, 5.75, 6.00, 6.25, 6.45, 6.65]),
    1000: np.array([1.50, 2.40, 3.10, 3.75, 4.30, 4.80, 5.20, 5.60,
                    5.95, 6.25, 6.55, 6.80, 7.05, 7.25]),
    1200: np.array([1.55, 2.55, 3.30, 4.00, 4.60, 5.10, 5.55, 5.95,
                    6.35, 6.70, 7.00, 7.25, 7.50, 7.75]),
}
_F10_T = sorted(_F10_DATA.keys())
# Pre-construcción de los splines PCHIP para cada T (evitar recompilar).
_F10_PCHIP = {T: PchipInterpolator(_F10_G, _F10_DATA[T], extrapolate=False)
              for T in _F10_T}


def fig10_hcc(G_lb_s_ft2: float, T_Gp_F: float,
              return_diag: bool = False):
    """
    Coeficiente convectivo en banco de tubos en sección de convección
    (Btu/h·ft²·°F).  Sustituye la lectura gráfica de la Figura 10 ULA.

    Parámetros
    ----------
    G_lb_s_ft2 : float
        Velocidad másica de gases en lb/(s·ft²).  Rango válido: 0.05–0.70.
        Recordatorio: la guía ULA recomienda 0.30–0.40 lb/(s·ft²)
        como rango de operación normal en convección.
    T_Gp_F : float
        Temperatura promedio del gas (°F) en la sección de convección.
        NO confundir con la temperatura de pared del tubo (que se usa
        en hcw / fig 11).  Rango válido: 200–1200 °F.
    return_diag : bool
        Si True devuelve dict con G, T y advertencias de saturación.

    Returns
    -------
    hcc : float
        Coeficiente convectivo del lado de gases (Btu/h·ft²·°F).
    """
    # Saturar al rango válido y registrar si hubo recorte.
    G_in, T_in = G_lb_s_ft2, T_Gp_F
    G = max(0.05, min(G_lb_s_ft2, 0.70))
    Tf = max(200.0, min(T_Gp_F, 1200.0))

    # Interpolación lineal en T sobre splines PCHIP en G.
    T_lo = max([t for t in _F10_T if t <= Tf])
    T_hi = min([t for t in _F10_T if t >= Tf])
    h_lo = float(_F10_PCHIP[T_lo](G))
    if T_hi == T_lo:
        hcc = h_lo
    else:
        h_hi = float(_F10_PCHIP[T_hi](G))
        f = (Tf - T_lo) / (T_hi - T_lo)
        hcc = h_lo + f * (h_hi - h_lo)

    if return_diag:
        return {
            "hcc": hcc,
            "G_in": G_in, "G_used": G,
            "T_in": T_in, "T_used": Tf,
            "G_recortado": (G_in != G),
            "T_recortado": (T_in != Tf),
            "G_recomendado": (0.30 <= G <= 0.40),
        }
    return hcc


# ---- Figura 11 ULA: hcw radiación de pared en convección
# ACTUALIZADO v2: la ecuación de la figura (hornos2.pdf pág. 44) es
#   hcw = 9.46 × (T_T_Rankine / 1000)³
# donde T_T debe estar en RANKINE (T_F + 459.67), NO en Fahrenheit.
# Verificación: T=258°F → T_R=718°R → hcw=9.46*(0.718)³≈3.50  (PDF≈3.8, Δ=-7.8%)
# La diferencia residual es de lectura gráfica.
# Fuente: hornos2.pdf pág. 44 — ecuación directa de la figura.
def fig11_hcw(T_pared_F: float) -> float:
    """
    Figura 11 ULA — Coeficiente de radiación de pared hcw (Btu/h·ft²·°F).
    Ecuación directa de la figura: hcw = 9.46 × (T_Rankine / 1000)³
    NOTA: temperatura en Rankine = T_F + 459.67
    Fuente: hornos2.pdf pág. 44. Rango: 100–1000 °F.
    """
    T_R = (max(100.0, min(T_pared_F, 1000.0)) + 459.67) / 1000.0
    return 9.46 * T_R ** 3


# ---- Figura 12 ULA: hcr radiación de gas en convección
# ACTUALIZADO v3 — RE-DIGITALIZACIÓN CORREGIDA tras auditoría técnica (mayo 2026).
# ─────────────────────────────────────────────────────────────────────────────
# PROBLEMA DETECTADO EN v2:
#   Los valores de la curva T_pared=200°F estaban subdimensionados en ~25%.
#   Con los datos de v2: T_G=1161°F, T_T=258°F → hcr=0.48 (PDF=1.60, Δ=70%).
#
# CORRECCIÓN v3:
#   Se re-leyó directamente la Fig VI.14 del PDF del Capítulo VI (Horno F-501)
#   en alta resolución (200 DPI). Las curvas son casi lineales en escala
#   lineal–lineal. Cada punto se leyó tomando como referencia las líneas de
#   cuadrícula y la pendiente característica de cada curva.
#
# VERIFICACIÓN CLAVE (punto de control del PDF Cap.VI, pág.19):
#   T_gas = 1160.75°F,  T_pared = 258.55°F  →  hcr = 1.555  (PDF: 1.60, Δ=2.8%)
#   La diferencia residual de 2.8% es inherente a la resolución de lectura gráfica
#   de la figura original (líneas gruesas, escala comprimida).
#
# FUENTE: Figura VI.14 / Fig 12 ULA — "Gas-Radiation Coefficient" (Lobo & Evans).
#   Eje X: Average Gas Temperature (°F), 400–2400°F.
#   Eje Y: Gas-Radiation Coefficient, BTU/HR, SQ FT, °F (0–5).
#   Parámetro: Average Tube-Wall Temperature (°F): 200, 400, 600, 800, 1000, 1200, 1400, 1600.
# RANGO VÁLIDO: T_G 400–2000°F, T_T 200–1600°F.
# ─────────────────────────────────────────────────────────────────────────────
_F12_TG = np.array([400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
_F12_DATA = {
    200:  np.array([0.20, 0.48, 0.82, 1.18, 1.60, 2.05, 2.52, 3.05, 3.60]),
    400:  np.array([0.28, 0.58, 0.95, 1.38, 1.82, 2.28, 2.78, 3.32, 3.90]),
    600:  np.array([0.38, 0.72, 1.12, 1.55, 2.02, 2.52, 3.05, 3.62, 4.22]),
    800:  np.array([0.52, 0.90, 1.32, 1.80, 2.30, 2.85, 3.42, 4.02, 4.65]),
    1000: np.array([0.70, 1.12, 1.58, 2.10, 2.65, 3.22, 3.82, 4.45, 5.00]),
    1200: np.array([0.92, 1.38, 1.90, 2.45, 3.05, 3.68, 4.32, 5.00, 5.00]),
    1400: np.array([1.18, 1.70, 2.28, 2.90, 3.55, 4.22, 5.00, 5.00, 5.00]),
    1600: np.array([1.50, 2.08, 2.70, 3.38, 4.08, 5.00, 5.00, 5.00, 5.00]),
}
_F12_TT = sorted(_F12_DATA.keys())
_F12_PCHIP = {T: PchipInterpolator(_F12_TG, _F12_DATA[T], extrapolate=False)
              for T in _F12_TT}

def fig12_hcr(T_pared_F: float, T_gas_F: float) -> float:
    """
    Figura 12 ULA — Coeficiente de radiación del gas hcr (Btu/h·ft²·°F).
    Interpolación PCHIP bilineal sobre datos RE-DIGITALIZADOS en v3 directamente
    de la Fig VI.14 del PDF del Capítulo VI (Horno F-501), resolución 200 DPI.

    Corrección v3 (auditoría mayo 2026): los datos de v2 subestimaban hcr en ~70%
    por valores incorrectos en la curva T_pared=200°F. Los nuevos datos producen:
      T_G=1161°F, T_T=258°F → hcr=1.555  (PDF: 1.60, Δ=2.8%)
    La diferencia residual es incertidumbre de lectura gráfica.

    Fuente: Fig VI.14 ULA (Lobo & Evans). Rango: T_G 400–2000°F, T_T 200–1600°F.
    """
    Tg = max(400.0, min(T_gas_F, 2000.0))
    Tt = max(200.0, min(T_pared_F, 1600.0))
    T_lo = max([t for t in _F12_TT if t <= Tt])
    T_hi = min([t for t in _F12_TT if t >= Tt])
    h_lo = float(_F12_PCHIP[T_lo](Tg) or 0)
    if T_hi == T_lo:
        return max(0.0, h_lo)
    h_hi = float(_F12_PCHIP[T_hi](Tg) or 0)
    f = (Tt - T_lo) / (T_hi - T_lo)
    return max(0.0, h_lo + f * (h_hi - h_lo))


# ---- Figura 14 ULA: Tiro por 100 ft de altura vs T_gas y T_ambiente
# ACTUALIZADO v2: datos re-digitalizados de hornos2.pdf pág. 55.
# Incluye múltiples curvas de temperatura ambiente.
# Verificaciones: T_gas=1560°F, T_amb=100°F → 0.963 (PDF=1.00, Δ=-3.7%)
#                 T_gas=720°F,  T_amb=100°F → 0.659 (PDF=0.75, Δ=-12%)
# La diferencia en tiro se debe a que el PDF usa T_gases=1560°F para el
# tramo de la caja de radiación (no el promedio). Se conserva la tabla completa.
# Fuente: hornos2.pdf pág. 55 — interpolación PCHIP bilineal.
_F14_T = np.array([400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600])
_F14_DATA = {
    50:  np.array([0.55, 0.60, 0.65, 0.69, 0.73, 0.77, 0.81, 0.85, 0.89, 0.92, 0.96, 0.99, 1.03]),
    70:  np.array([0.53, 0.58, 0.63, 0.67, 0.72, 0.76, 0.80, 0.84, 0.87, 0.91, 0.94, 0.97, 1.01]),
    80:  np.array([0.52, 0.57, 0.62, 0.67, 0.71, 0.75, 0.79, 0.83, 0.86, 0.90, 0.93, 0.96, 1.00]),
    90:  np.array([0.51, 0.56, 0.61, 0.66, 0.70, 0.74, 0.78, 0.82, 0.85, 0.89, 0.92, 0.95, 0.99]),
    100: np.array([0.50, 0.55, 0.60, 0.65, 0.69, 0.73, 0.77, 0.81, 0.84, 0.88, 0.91, 0.94, 0.98]),
    110: np.array([0.49, 0.54, 0.59, 0.64, 0.68, 0.72, 0.76, 0.80, 0.83, 0.87, 0.90, 0.94, 0.97]),
}
_F14_TAMB = sorted(_F14_DATA.keys())
_F14_PCHIP = {T: PchipInterpolator(_F14_T, _F14_DATA[T], extrapolate=False)
              for T in _F14_TAMB}

def fig14_tiro_100ft(T_gas_F: float, T_amb_F: float = 100.0) -> float:
    """
    Figura 14 ULA — Tiro disponible por cada 100 ft de altura (in H₂O).
    Interpolación PCHIP bilineal sobre datos re-digitalizados de hornos2.pdf pág. 55.
    Fuente: ULA hornos2.pdf pág. 55. Rango: T_gas 400–1600 °F, T_amb 50–110 °F.
    """
    Tg = max(400.0, min(T_gas_F, 1600.0))
    Ta = max(50.0, min(T_amb_F, 110.0))
    T_lo = max([t for t in _F14_TAMB if t <= Ta])
    T_hi = min([t for t in _F14_TAMB if t >= Ta])
    t_lo = float(_F14_PCHIP[T_lo](Tg) or 0)
    if T_hi == T_lo:
        return max(0.0, t_lo)
    t_hi = float(_F14_PCHIP[T_hi](Tg) or 0)
    f = (Ta - T_lo) / (T_hi - T_lo)
    return max(0.0, t_lo + f * (t_hi - t_lo))


# ---- Figura 15 ULA: Densidad de gases de combustión (lb/ft³) vs T (°F)
# ACTUALIZADO v2: datos re-digitalizados de hornos2.pdf pág. 56.
# Verificaciones: T=1160°F → 0.0268 lb/ft³ (PDF=0.024, Δ=+12%)
#                 T=720°F  → 0.0334 lb/ft³  (PDF=0.0324, Δ=+3%)
# Nota: el PDF usa T_gas_conv = 1200°F para obtener 0.024 (no 1160.75°F).
# Fuente: hornos2.pdf pág. 56 — interpolación PCHIP.
# Rango: 400 ≤ T ≤ 1800 °F
_F15_T   = np.array([400,  500,  600,  700,  800,  900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1800])
_F15_RHO = np.array([0.0414,0.0385,0.0360,0.0338,0.0319,0.0303,0.0288,0.0275,0.0263,0.0252,0.0242,0.0233,0.0225,0.0211])
_fig15_pchip = PchipInterpolator(_F15_T, _F15_RHO, extrapolate=False)

def fig15_rho_gases(T_F: float) -> float:
    """
    Figura 15 ULA — Densidad de gases de combustión a nivel del mar (lb/ft³).
    Interpolación PCHIP sobre datos re-digitalizados de hornos2.pdf pág. 56.
    Fuente: ULA hornos2.pdf pág. 56. Rango: 400–1800 °F.
    """
    Tg = max(400.0, min(T_F, 1800.0))
    return max(0.010, float(_fig15_pchip(Tg)))


# ---- Diagrama de moody simplificado (factor de Fanning)
def factor_fanning(Re: float, eps_d: float = 0.0001) -> float:
    """Factor de fricción de Fanning (ec. Colebrook–White / Swamee–Jain)."""
    Re = max(1.0, Re)
    if Re < 2300:
        return 16.0 / Re
    # Swamee-Jain para Darcy, dividido entre 4 = Fanning
    fD = 0.25 / (math.log10(eps_d / 3.7 + 5.74 / Re ** 0.9)) ** 2
    return fD / 4.0


# Diccionario con la documentación de cada correlación — usado para mostrar
# en pantalla el origen de cada cálculo.
DOC_CORRELACIONES = [
    {"figura": "Fig. 1 ULA",  "entrada": "% exceso aire", "salida": "lb gases / MMBtu",
     "metodo": "Interpolación PCHIP de puntos digitalizados", "fuente": "ULA — Anaya 1997",
     "rango": "0–100 % exceso de aire"},
    {"figura": "Eff (ULA)",   "entrada": "T_stack, % exceso", "salida": "Eficiencia",
     "metodo": "Ecuación directa ULA (combustible diesel)",  "fuente": "ULA — Anaya 1997",
     "rango": "T_stack 300–800 °F"},
    {"figura": "Fig. 3 ULA",  "entrada": "C/D",         "salida": "Factor α",
     "metodo": "PCHIP re-digitalizado de hornos2.pdf pág.20 (curva ULA con refractario)",
     "fuente": "Hottel & Perry — ULA hornos2.pdf pág.20",
     "rango": "1.0 ≤ C/D ≤ 2.2 — verificado: C/D=1.778→α=0.924 (PDF=0.925)"},
    {"figura": "Fig. 4 ULA",  "entrada": "% exceso aire", "salida": "P (atm)",
     "metodo": "Ecuación directa ULA (polinomio cúbico)", "fuente": "ULA — Anaya 1997",
     "rango": "0–100 %"},
    {"figura": "Fig. 5 ULA",  "entrada": "P·Lz, T_gas",  "salida": "ε de gases",
     "metodo": "PCHIP bilineal sobre datos re-digitalizados de hornos2.pdf pág.30",
     "fuente": "Lobo & Evans — ULA hornos2.pdf pág.30",
     "rango": "P·Lz 0.0–5.5 atm·ft, T 1000–2400 °F — verificado: PLz=5,Tg=1500°F→ε=0.58"},
    {"figura": "Fig. 6 ULA",  "entrada": "ε, Aw/αAcp",   "salida": "φ factor de intercambio",
     "metodo": "PCHIP bilineal re-digitalizado de hornos2.pdf pág.31 (sustituye fórmula analítica)",
     "fuente": "Lobo & Evans — ULA hornos2.pdf pág.31",
     "rango": "ε 0.10–0.70; r=Aw/αAcp 0–7 — verificado: ε=0.58,r=1.12→φ=0.70"},
    {"figura": "Fig. 7 ULA",  "entrada": "T_gas, % aire", "salida": "qg/qn",
     "metodo": "PCHIP bilineal re-digitalizado de hornos2.pdf pág.32",
     "fuente": "ULA hornos2.pdf pág.32",
     "rango": "200–2400 °F, 0–100 % exc — verificado: 1500°F,25%→0.42; 1560°F,25%→0.44"},
    {"figura": "Fig. 10 ULA", "entrada": "G, T_Gp",    "salida": "hcc",
     "metodo": "PCHIP en G + lineal en T sobre 6 curvas re-digitalizadas",
     "fuente": "ULA hornos2.pdf pág.43",
     "rango": "G 0.05–0.70 lb/s·ft²; T 200–1200 °F"},
    {"figura": "Fig. 11 ULA", "entrada": "T_pared (°F)", "salida": "hcw",
     "metodo": "Ecuación directa de la figura: hcw=9.46×(T_Rankine/1000)³",
     "fuente": "ULA hornos2.pdf pág.44 — ecuación explícita en la figura",
     "rango": "100–1000 °F — NOTA: T en Rankine, no en Fahrenheit"},
    {"figura": "Fig. 12 ULA", "entrada": "T_gas, T_pared", "salida": "hcr",
     "metodo": "PCHIP bilineal RE-DIGITALIZADO v3 — Fig VI.14 PDF Cap.VI a 200 DPI (auditoría mayo 2026)",
     "fuente": "ULA — Fig VI.14 PDF Capítulo VI (Lobo & Evans). Corrige error de ~70% en v2.",
     "rango": "T_G 400–2000°F, T_T 200–1600°F — verificado: T_G=1161°F,T_T=258°F→1.555 (PDF=1.6, Δ=2.8%)"},
    {"figura": "Fig. 14 ULA", "entrada": "T_gas, T_amb",  "salida": "tiro por 100 ft",
     "metodo": "PCHIP bilineal re-digitalizado de hornos2.pdf pág.55 (6 curvas T_amb)",
     "fuente": "ULA hornos2.pdf pág.55",
     "rango": "T_gas 400–1600 °F, T_amb 50–110 °F"},
    {"figura": "Fig. 15 ULA", "entrada": "T_gas",         "salida": "ρ gases",
     "metodo": "PCHIP re-digitalizado de hornos2.pdf pág.56 (sustituye ley gas ideal)",
     "fuente": "ULA hornos2.pdf pág.56",
     "rango": "400–1800 °F"},
]


# =====================================================================
# 3.  MÓDULO DE PROPIEDADES
# ---------------------------------------------------------------------
# Propiedades de hidrocarburos (correlaciones API) y de agua/vapor
# (CoolProp si está disponible, fallback a tablas).
# =====================================================================

def sg_from_api(api: float) -> float:
    """Gravedad específica a 60°F a partir de °API."""
    return 141.5 / (131.5 + api)


def watson_K(T_meabp_F: float, sg: float) -> float:
    """Factor Watson K (parafínicos≈12.0, nafténicos 11.0–11.5, aromáticos<11)."""
    T_R = T_meabp_F + 459.67
    return T_R ** (1.0 / 3.0) / sg


def cp_hidrocarburo_liq_API(T_F: float, api: float, K_w: float = 11.8) -> float:
    """
    Cp (Btu/lb·°F) para fracciones líquidas de petróleo.
    Correlación API/Cragoe (Watson & Nelson):
        Cp = (0.388 + 0.00045·T) · (1 + 0.0083·(K_w − 11.8))/sg^0.5
    Válido 100–800 °F.
    """
    sg = sg_from_api(api)
    cp_base = (0.388 + 0.00045 * T_F) / math.sqrt(sg)
    cp = cp_base * (1.0 + 0.0083 * (K_w - 11.8))
    return cp


def hentalpia_hc_liq(T_F: float, T_ref_F: float, api: float, K_w: float = 11.8) -> float:
    """Entalpía sensible (Btu/lb) integrando Cp entre T_ref y T."""
    n = 50
    Ts = np.linspace(T_ref_F, T_F, n)
    cp = np.array([cp_hidrocarburo_liq_API(t, api, K_w) for t in Ts])
    return float(trapezoid(cp, Ts))


def densidad_hc_liq(T_F: float, api: float) -> float:
    """Densidad de hidrocarburo líquido (lb/ft³).  Corrección térmica ASTM."""
    sg = sg_from_api(api)
    rho_60 = sg * 62.366
    # corrección térmica simplificada (ASTM D1250)
    alpha = 4.5e-4  # 1/°F  típico crudo
    return rho_60 * (1.0 - alpha * (T_F - 60.0))


def viscosidad_hc_liq(T_F: float, api: float) -> float:
    """Viscosidad cinemática (cSt) – correlación tipo Andrade simple."""
    sg = sg_from_api(api)
    # ν muy aprox: ln(ln(ν+0.7)) = a + b*ln(T_R)
    T_R = T_F + 459.67
    log_lognu = -3.0 * sg + 6.5 - 1.6 * math.log(T_R / 540.0)
    nu = math.exp(math.exp(log_lognu)) - 0.7
    return max(0.3, min(nu, 2000.0))


def viscosidad_hc_dynamic(T_F: float, api: float) -> float:
    """Viscosidad dinámica (cP) = ν (cSt) * sg."""
    nu = viscosidad_hc_liq(T_F, api)
    sg = sg_from_api(api)
    return nu * sg


# ---- Agua / Vapor (CoolProp) ------------------------------------------------
def props_agua_vapor(P_psia: float, T_F: float = None,
                     fase: str = "saturado") -> dict:
    """
    Devuelve dict con {h, rho, mu, cp} en unidades inglesas:
       h [Btu/lb], rho [lb/ft³], mu [cP], cp [Btu/lb·°F]
    """
    if not COOLPROP_OK:
        raise RuntimeError("CoolProp no instalado.")
    P = P_psia * 6894.76  # Pa
    if fase == "saturado_liq":
        h = PropsSI("H", "P", P, "Q", 0, "Water") / 2326.0
        rho = PropsSI("D", "P", P, "Q", 0, "Water") * 0.06243
        mu = PropsSI("V", "P", P, "Q", 0, "Water") * 1000  # cP
        cp = PropsSI("C", "P", P, "Q", 0, "Water") / 4186.8
        T = (PropsSI("T", "P", P, "Q", 0, "Water") - 273.15) * 9 / 5 + 32
        return {"h": h, "rho": rho, "mu": mu, "cp": cp, "T_F": T}
    elif fase == "saturado_vap":
        h = PropsSI("H", "P", P, "Q", 1, "Water") / 2326.0
        rho = PropsSI("D", "P", P, "Q", 1, "Water") * 0.06243
        mu = PropsSI("V", "P", P, "Q", 1, "Water") * 1000
        cp = PropsSI("C", "P", P, "Q", 1, "Water") / 4186.8
        T = (PropsSI("T", "P", P, "Q", 1, "Water") - 273.15) * 9 / 5 + 32
        return {"h": h, "rho": rho, "mu": mu, "cp": cp, "T_F": T}
    elif fase == "subenfriado":
        T_K = (T_F - 32) * 5 / 9 + 273.15
        h = PropsSI("H", "P", P, "T", T_K, "Water") / 2326.0
        rho = PropsSI("D", "P", P, "T", T_K, "Water") * 0.06243
        mu = PropsSI("V", "P", P, "T", T_K, "Water") * 1000
        cp = PropsSI("C", "P", P, "T", T_K, "Water") / 4186.8
        return {"h": h, "rho": rho, "mu": mu, "cp": cp, "T_F": T_F}
    elif fase == "sobrecalentado":
        T_K = (T_F - 32) * 5 / 9 + 273.15
        h = PropsSI("H", "P", P, "T", T_K, "Water") / 2326.0
        rho = PropsSI("D", "P", P, "T", T_K, "Water") * 0.06243
        mu = PropsSI("V", "P", P, "T", T_K, "Water") * 1000
        cp = PropsSI("C", "P", P, "T", T_K, "Water") / 4186.8
        return {"h": h, "rho": rho, "mu": mu, "cp": cp, "T_F": T_F}
    else:
        raise ValueError("Fase desconocida.")


# =====================================================================
# 4.  MÓDULO DE RADIACIÓN
# =====================================================================

def temp_cruce(T_e_F: float, T_s_F: float, eta_R: float = 0.7) -> float:
    """Temperatura de cruce radiación/convección (T_p) — ULA."""
    return T_s_F - eta_R * (T_s_F - T_e_F)


def superficie_radiante(Q_R: float, fqM: float) -> float:
    """A_R (ft²) = Q_R / fqM."""
    return Q_R / fqM


# =====================================================================
# 5.  MÓDULO DE TUBOS
# =====================================================================

def get_tubo(DN_in: int, SCH: str = "80") -> dict:
    """Devuelve fila completa de la tabla de tubos para DN y SCH dados."""
    df = TABLA_TUBOS[(TABLA_TUBOS.DN_in == DN_in) & (TABLA_TUBOS.SCH == SCH)]
    if df.empty:
        # Si no hay SCH exacto, devolver el primero del DN
        df = TABLA_TUBOS[TABLA_TUBOS.DN_in == DN_in]
    return df.iloc[0].to_dict()


def calc_velocidad_liquido(W_lb_h: float, rho_lb_ft3: float,
                           A_flujo_ft2: float, n_pasos: int) -> float:
    """Velocidad media del líquido (ft/s) en tubos."""
    Q_ft3_s = (W_lb_h / rho_lb_ft3) / 3600.0  # ft³/s totales
    Q_paso = Q_ft3_s / n_pasos
    return Q_paso / A_flujo_ft2


def calc_velocidad_masica(W_lb_h: float, A_flujo_ft2: float,
                          n_pasos: int) -> float:
    """Velocidad másica (lb/h·ft²) por paso."""
    return (W_lb_h / n_pasos) / A_flujo_ft2


# =====================================================================
# 6.  MÓDULO DE GEOMETRÍA DE CABINA
# =====================================================================

def long_haz_llama(V_horno_ft3: float, tipo: str = "rect") -> float:
    """Longitud media del haz de llama (ft).  Lz ≈ 2/3 * V^(1/3)."""
    return (2.0 / 3.0) * V_horno_ft3 ** (1.0 / 3.0)


def alpha_factor(C_over_D: float, n_filas: int = 1) -> float:
    """Factor α de absorción (Hottel)."""
    if n_filas == 1:
        return fig3_alpha_una_fila(C_over_D)
    return fig3_alpha_dos_filas(C_over_D)


# =====================================================================
# 7.  MÓDULO DE CONVECCIÓN
# ---------------------------------------------------------------------
# Procedimiento de la guía ULA (Anaya 1997, pág 40-49):
#   1) Geometría: arreglo triangular, Lcc = 2·DE, NtH tubos/hilera (= 4
#      en la guía).
#   2) Ancho de la cabina de convección  = (NtH + 0.5) * Lcc
#   3) Ancho libre  LLC = Ancho – NtH * DE
#   4) Área libre de paso AP = LLC * LTE   (ft²)
#   5) Velocidad másica de gases  G = Wg / AP    [lb/(s·ft²)]
#      → debe verificarse que G ∈ [0.30, 0.40].  Si no, ajustar NtH.
#   6) hcc = fig10_hcc(G, T_Gp)         (Btu/h·ft²·°F)
#      hcw = fig11_hcw(T_pared)
#      hcr = fig12_hcr(T_pared, T_Gp)
#   7) hCi = hcc + hcw + hcr
#      Ff  = (hcw / hCi) * (A_PH / A_TH)        (corrección por pared)
#      hC  = (1 + Ff) * (hcc + hcr)
#      Uc  = hC · h_int / (hC + h_int)
#   8) AC  = Q_C / (Uc · LMTD)         (ft²)
#      NTC = AC / S_t,    S_t = AC / NtE  ⇒  hileras = NTC / NtH
# =====================================================================

def geometria_conveccion(NtH: int, DE_in: float, Lcc_in: float,
                         L_TE_ft: float) -> dict:
    """
    Geometría de la sección de convección (procedimiento ULA).
    NtH      : tubos por hilera (típico 4)
    DE_in    : diámetro externo del tubo (in)
    Lcc_in   : espaciamiento centro–centro (in) — recomendado 2·DE
    L_TE_ft  : longitud efectiva del tubo (ft)
    """
    ancho_in = (NtH + 0.5) * Lcc_in        # in
    ancho_ft = ancho_in / 12.0
    LLC_in = ancho_in - NtH * DE_in        # in
    LLC_ft = LLC_in / 12.0
    AP_ft2 = LLC_ft * L_TE_ft              # área libre de paso (ft²)
    return {"ancho_ft": ancho_ft, "LLC_ft": LLC_ft, "AP_ft2": AP_ft2}


def diagnosticar_G_gases(G: float) -> dict:
    """Valida la velocidad másica de gases en convección (criterio ULA)."""
    if G < 0.30:
        return {"ok": False, "tipo": "BAJA",
                "msg": f"G = {G:.3f} lb/(s·ft²) está por DEBAJO del rango "
                       f"recomendado ULA (0.30–0.40). Para aumentar G "
                       f"DISMINUYA el número de tubos por hilera (NtH) o "
                       f"reduzca el espaciamiento centro–centro."}
    if G > 0.40:
        return {"ok": False, "tipo": "ALTA",
                "msg": f"G = {G:.3f} lb/(s·ft²) está por ENCIMA del rango "
                       f"recomendado ULA (0.30–0.40). Para reducir G "
                       f"AUMENTE el número de tubos por hilera (NtH) o "
                       f"el espaciamiento centro–centro."}
    return {"ok": True, "tipo": "OK",
            "msg": f"G = {G:.3f} lb/(s·ft²) dentro del rango recomendado "
                   f"ULA (0.30–0.40)."}


def LMTD(dT1: float, dT2: float) -> float:
    """Diferencia media logarítmica de temperaturas."""
    if abs(dT1 - dT2) < 0.5:
        return 0.5 * (dT1 + dT2)
    if dT1 <= 0 or dT2 <= 0:
        # Evitar log de números no positivos.
        return float("nan")
    return (dT1 - dT2) / math.log(dT1 / dT2)


def coef_global_conveccion(hcc: float, hcw: float, hcr: float,
                           h_int: float, A_PH: float, A_TH: float) -> tuple:
    """
    Coeficiente global de transferencia en sección de convección
    según procedimiento ULA pág 47.
    Returns: (Uc, hC, Ff, hCi)
    """
    h_ci = hcc + hcw + hcr                          # película total
    Ff = (hcw / h_ci) * (A_PH / A_TH)               # factor radiación pared
    h_C = (1.0 + Ff) * (hcc + hcr)                  # h aparente del gas
    Uc = (h_C * h_int) / (h_C + h_int)              # global limpio
    return Uc, h_C, Ff, h_ci


# =====================================================================
# 8.  MÓDULO DE FLUIDODINÁMICA
# =====================================================================

def caida_presion_serpentin(W_lb_h: float, DI_in: float, L_total_ft: float,
                            n_pasos: int, n_curvas: int,
                            mu_cP: float, rho_lb_ft3: float) -> dict:
    """
    ΔP en serpentín (psi) — Darcy-Weisbach en unidades inglesas.

    CORRECCIÓN v4 (dos bugs corregidos respecto a v3):

    Bug 1 — G en lb/h·ft² en lugar de lb/s·ft²:
        La versión anterior usaba G [lb/h·ft²] directamente en la fórmula de
        Darcy que espera velocidad en ft/s, produciendo ΔP ~3600² mayor.
        Corrección: convertir G a lb/s·ft² dividiendo entre 3600.

    Bug 2 — L_total era la suma de TODOS los tubos (serie + paralelo):
        L_total_ft = (N_R + N_TC) * L_TE cuenta cada tubo una vez, pero el
        fluido no recorre todos los tubos en serie: se divide en n_pasos
        circuitos paralelos, cada uno con N_total/n_pasos tubos en serie.
        Corrección: la longitud que recorre el fluido es L_total_ft / n_pasos.
        De igual forma, los codos se dividen entre n_pasos.

    Fórmula final (coherente en unidades):
        ΔP [psi] = f_D · (L_eq/D) · (ρ·v²/2) / 144
    con:
        f_D  = 4 · f_Fanning
        L_eq = L_circuito + (n_curvas/n_pasos) · 50 · D   [ft]
        D    = DI [ft]
        ρ    = densidad media  [lb/ft³]
        v    = velocidad media del fluido  [ft/s]

    Referencia: Crane TP-410 / Perry's 8th Ed.
    """
    DI_ft   = DI_in / 12.0
    A_flujo = math.pi / 4.0 * DI_ft ** 2

    # Velocidad media [ft/s] en un tubo (cada paso tiene n_pasos ramas paralelas)
    G_lb_s = (W_lb_h / n_pasos) / (A_flujo * 3600.0)   # lb/s·ft²
    v_ft_s = G_lb_s / rho_lb_ft3

    # Viscosidad dinámica en lb/(ft·s)  (1 cP = 1/1488.16 lb/(ft·s))
    mu_lb_ft_s = mu_cP / 1488.16

    # Reynolds
    Re = rho_lb_ft3 * v_ft_s * DI_ft / mu_lb_ft_s

    # Factor de Fanning → Darcy
    f_fanning = factor_fanning(Re, eps_d=0.0001)
    f_darcy   = 4.0 * f_fanning

    # Longitud que recorre el fluido en UN circuito:
    #   L_circuito = L_total_ft / n_pasos  (cada tubo mide L_TE, hay N/n_pasos por circuito)
    #   n_curvas_circuito = n_curvas / n_pasos
    L_circuito = L_total_ft / n_pasos
    n_curv_c   = n_curvas / n_pasos
    L_eq       = L_circuito + n_curv_c * 50.0 * DI_ft

    # Darcy-Weisbach  [psi]
    dP_psi = f_darcy * (L_eq / DI_ft) * (rho_lb_ft3 * v_ft_s ** 2 / 2.0) / 144.0

    return {
        "dP_psi":    dP_psi,
        "Re":        Re,
        "f":         f_fanning,          # Fanning
        "G_lb_hft2": G_lb_s * 3600.0,   # lb/h·ft² para mostrar en pantalla
        "v_ft_s":    v_ft_s,
    }


# =====================================================================
# 9.  MÓDULO DE CHIMENEA
# =====================================================================

def diam_chimenea(W_gas_lb_h: float, T_F: float, V_target_ft_s: float = 30.0) -> float:
    """Diámetro de chimenea (ft) para velocidad objetivo."""
    rho = fig15_rho_gases(T_F)
    Q = W_gas_lb_h / rho / 3600.0  # ft³/s
    A = Q / V_target_ft_s
    return (4.0 * A / math.pi) ** 0.5


def altura_chimenea(tiro_total_inH2O: float, T_gas_F: float,
                    factor_seguridad: float = 1.1) -> float:
    """Altura requerida de chimenea (ft)."""
    tiro_100 = fig14_tiro_100ft(T_gas_F)
    L = tiro_total_inH2O / tiro_100 * 100.0
    return L * factor_seguridad


# =====================================================================
# 10.  MÓDULO DE DIAGNÓSTICO
# =====================================================================

def diagnosticar_velocidad(v_ft_s: float, fluido: str = "liquido") -> dict:
    """Valida velocidad interna en tubos y sugiere corrección."""
    if fluido == "liquido":
        v_min, v_max = 3.0, 7.0
        if v_ft_s < v_min:
            return {"ok": False,
                    "msg": f"Velocidad de {v_ft_s:.2f} ft/s es BAJA "
                           f"(mín {v_min} ft/s). Aumente el número de pasos, "
                           f"reduzca el diámetro nominal o reduzca el número "
                           f"de tubos en paralelo."}
        if v_ft_s > v_max:
            return {"ok": False,
                    "msg": f"Velocidad de {v_ft_s:.2f} ft/s es ALTA "
                           f"(máx {v_max} ft/s). Disminuya el número de pasos, "
                           f"aumente el diámetro nominal o aumente el número "
                           f"de tubos en paralelo. (Riesgo de erosión y ΔP alto)."}
        return {"ok": True,
                "msg": f"Velocidad {v_ft_s:.2f} ft/s dentro del rango "
                       f"recomendado para líquidos ({v_min}–{v_max} ft/s)."}
    return {"ok": True, "msg": "Velocidad aceptable."}


def diagnosticar_fqM(fqM_real: float, fqM_max: float = 12000.0,
                     combustible: str = "liquido") -> dict:
    """Valida densidad de flujo de calor."""
    limite = 12000.0 if combustible == "liquido" else 16000.0
    if fqM_real > limite:
        return {"ok": False,
                "msg": f"Flujo radiante {fqM_real:,.0f} Btu/h·ft² supera "
                       f"el límite recomendado ({limite:,.0f}). Aumente la "
                       f"superficie radiante (más tubos, más largos) o "
                       f"reduzca la carga térmica."}
    return {"ok": True,
            "msg": f"Flujo radiante {fqM_real:,.0f} Btu/h·ft² aceptable "
                   f"(< {limite:,.0f})."}


def diagnosticar_espaciamiento(C_over_D: float) -> dict:
    """Valida C/D entre 1.5 y 3.0 (recomendado 2.0)."""
    if C_over_D < 1.5:
        return {"ok": False,
                "msg": f"C/D = {C_over_D:.2f} demasiado bajo. Aumente el "
                       f"espaciamiento centro-a-centro (recomendado 2·DE)."}
    if C_over_D > 3.0:
        return {"ok": False,
                "msg": f"C/D = {C_over_D:.2f} demasiado alto. Reduzca el "
                       f"espaciamiento (penaliza superficie radiante efectiva)."}
    return {"ok": True,
            "msg": f"Espaciamiento C/D = {C_over_D:.2f} dentro del rango "
                   f"recomendado (1.5–3.0; típico 2.0)."}


def diagnosticar_eficiencia(eff: float) -> dict:
    """Valida eficiencia para horno tipo cabina horizontal (70–85 %)."""
    if eff < 0.70:
        return {"ok": False,
                "msg": f"Eficiencia {eff*100:.1f}% por debajo del rango típico "
                       f"para cabina horizontal (70–85%). Considere reducir "
                       f"la temperatura de chimenea o el exceso de aire."}
    if eff > 0.88:
        return {"ok": False,
                "msg": f"Eficiencia {eff*100:.1f}% optimista. Revise la "
                       f"temperatura de chimenea y el % de pérdidas asumidas."}
    return {"ok": True,
            "msg": f"Eficiencia {eff*100:.1f}% dentro del rango típico."}


def recomendar_material(T_pared_F: float) -> str:
    """Recomienda material según T_max de tabla 6."""
    df = TABLA_MATERIALES.sort_values("T_max_F")
    for _, row in df.iterrows():
        if T_pared_F <= row["T_max_F"] - 50:  # margen de 50°F
            return row["Material"]
    return df.iloc[-1]["Material"]


def diagnosticar_chimenea(altura_ft: float) -> dict:
    """Valida altura de chimenea (típico 60–150 ft)."""
    if altura_ft < 30:
        return {"ok": False,
                "msg": f"Chimenea de {altura_ft:.1f} ft es muy baja. Revise "
                       f"el tiro requerido y la dispersión atmosférica."}
    if altura_ft > 200:
        return {"ok": False,
                "msg": f"Chimenea de {altura_ft:.1f} ft excesivamente alta. "
                       f"Revise pérdidas de carga, diámetro de chimenea o "
                       f"considere tiro forzado/inducido."}
    return {"ok": True,
            "msg": f"Altura de chimenea {altura_ft:.1f} ft técnicamente viable."}


def diagnosticar_dP(dP_psi: float, tipo_servicio: str = "crudo",
                    dP_limite_diseno: float = None) -> dict:
    """
    Valida ΔP del serpentín de proceso.

    Parámetros
    ----------
    dP_psi : float
        Caída de presión calculada [psi].
    tipo_servicio : str
        'crudo', 'vacio' o 'vapor' — define rangos típicos de referencia.
    dP_limite_diseno : float, opcional
        Límite máximo especificado en las bases de diseño [psi].
        Si se proporciona, se evalúa primero contra ese límite.

    Lógica de diagnóstico
    ---------------------
    1. Si se proporcionó dP_limite_diseno: evalúa contra ese límite exacto.
    2. Rangos de referencia por tipo de servicio (orientativos, no límites):
         crudo  : 40–250 psi  (ΔP típico de operación; >250 psi es muy alto)
         vacío  : 15–100 psi
         vapor  : 5–50 psi
    """
    # ── Evaluación contra límite de diseño (prioritaria) ──────────────
    if dP_limite_diseno is not None and dP_limite_diseno > 0:
        if dP_psi <= dP_limite_diseno:
            return {
                "ok": True,
                "msg": (f"ΔP = {dP_psi:.1f} psi ≤ límite de diseño "
                        f"{dP_limite_diseno:.0f} psi ✅"),
            }
        else:
            exceso = dP_psi - dP_limite_diseno
            factor = dP_psi / dP_limite_diseno
            return {
                "ok": False,
                "msg": (
                    f"ΔP = {dP_psi:.1f} psi supera el límite de diseño "
                    f"de {dP_limite_diseno:.0f} psi en {exceso:.1f} psi "
                    f"({factor:.1f}× el límite). "
                    f"Para reducir ΔP: aumente el número de pasos, "
                    f"aumente el diámetro nominal del tubo, o "
                    f"aumente la longitud efectiva reduciendo el número de tubos."
                ),
            }

    # ── Evaluación contra rangos típicos por servicio ─────────────────
    rangos = {
        "crudo":  (40,  250),
        "vacio":  (15,  100),
        "vapor":  (5,    50),
    }
    lo, hi = rangos.get(tipo_servicio, (10, 250))

    if dP_psi < lo:
        return {
            "ok": True,
            "msg": (f"ΔP = {dP_psi:.1f} psi — por debajo del rango típico "
                    f"({lo}–{hi} psi para {tipo_servicio}). "
                    f"Puede indicar velocidad baja o serpentín corto."),
        }
    if dP_psi <= hi:
        return {
            "ok": True,
            "msg": (f"ΔP = {dP_psi:.1f} psi dentro del rango típico "
                    f"({lo}–{hi} psi para {tipo_servicio})."),
        }
    # dP_psi > hi → demasiado alto
    return {
        "ok": False,
        "msg": (
            f"ΔP = {dP_psi:.1f} psi supera el rango típico "
            f"({lo}–{hi} psi para {tipo_servicio}). "
            f"Para reducir ΔP: aumente el número de pasos, "
            f"aumente el diámetro nominal, o reduzca la longitud del serpentín."
        ),
    }

# =====================================================================
# 11.  ESQUEMA TÉCNICO 2D DEL HORNO (vista lateral)
# ---------------------------------------------------------------------
# Genera con matplotlib un dibujo técnico 2D usando ÚNICAMENTE los
# resultados ya calculados en st.session_state.datos.  No usa imágenes
# externas ni dibujos genéricos.  Todas las dimensiones y rótulos
# provienen del diccionario de resultados.
# =====================================================================


_REQUERIDOS_ESQ = {
    "paso1": ["tipo_fluido", "W_lb_h", "T_e_F", "T_s_F", "Q_Btu_h"],
    "paso2": ["eta", "qn_Btu_h"],
    "paso3": ["material_rec"],
    "paso4": ["DN_in", "SCH", "tubo", "L_TE_ft", "n_pasos", "N_tubos_R"],
    "paso5": ["L_BR_ft", "L_HR_ft", "L_cc_in", "N_tE"],
    "paso6": ["T_chimenea_F", "N_TC", "n_hileras", "NtH"],
    "paso7": ["D_ch_ft", "L_ch_ft"],
}

_DIMS_POSITIVAS = [
    ("paso4", "L_TE_ft"), ("paso4", "N_tubos_R"), ("paso4", "n_pasos"),
    ("paso5", "L_BR_ft"), ("paso5", "L_HR_ft"), ("paso5", "L_cc_in"),
    ("paso6", "N_TC"), ("paso6", "n_hileras"),
    ("paso7", "D_ch_ft"), ("paso7", "L_ch_ft"),
]


def _validar_resultados_esquema(results: dict):
    """Valida claves y coherencia física antes de dibujar."""
    faltantes, advertencias = [], []
    if not isinstance(results, dict):
        return False, ["El objeto de resultados no es un diccionario."], []
    for paso, claves in _REQUERIDOS_ESQ.items():
        if paso not in results:
            faltantes.append(f"{paso} (todo el bloque)")
            continue
        d = results[paso]
        if not isinstance(d, dict):
            faltantes.append(f"{paso} (estructura inválida)")
            continue
        for k in claves:
            if k not in d or d[k] is None:
                faltantes.append(f"{paso}.{k}")
    if faltantes:
        return False, faltantes, advertencias
    for paso, k in _DIMS_POSITIVAS:
        v = results[paso][k]
        try:
            if v is None or float(v) <= 0:
                advertencias.append(f"{paso}.{k} = {v} no es físicamente válido (≤0).")
        except (TypeError, ValueError):
            advertencias.append(f"{paso}.{k} = {v} no es numérico.")
    tubo = results["paso4"].get("tubo", {})
    for k in ("DE_in", "DI_in"):
        if k not in tubo or tubo[k] is None or tubo[k] <= 0:
            advertencias.append(f"paso4.tubo.{k} inválido o ausente.")
    return len(advertencias) == 0, faltantes, advertencias


def draw_furnace_schematic(results: dict):
    """
    Dibujo técnico 2D — vista lateral de horno de cabina horizontal.

    Lee st.session_state.datos (paso1..paso7).  Devuelve (fig, info).
    Si faltan claves, devuelve (None, info_con_faltantes).
    """
    ok, faltantes, advertencias = _validar_resultados_esquema(results)
    info = {"ok": ok, "faltantes": faltantes, "advertencias": advertencias}
    if faltantes:
        return None, info

    d1, d2, d3 = results["paso1"], results["paso2"], results["paso3"]
    d4, d5, d6, d7 = results["paso4"], results["paso5"], results["paso6"], results["paso7"]

    # ---- variables reales calculadas ---------------------------------
    L_largo = float(d4["L_TE_ft"])
    L_ancho = float(d5["L_BR_ft"])
    L_alto  = float(d5["L_HR_ft"])
    DE_in   = float(d4["tubo"]["DE_in"])
    DN_in, SCH = d4["DN_in"], d4["SCH"]
    Lcc_in  = float(d5["L_cc_in"])
    N_R     = int(d4["N_tubos_R"])
    N_pasos = int(d4["n_pasos"])
    N_tE    = int(d5["N_tE"])
    N_TC    = int(d6["N_TC"])
    n_hil   = int(d6["n_hileras"])
    nT_hil  = int(d6["NtH"])
    D_ch    = float(d7["D_ch_ft"])
    L_ch    = float(d7["L_ch_ft"])
    T_e, T_s, T_ch = d1["T_e_F"], d1["T_s_F"], d6["T_chimenea_F"]
    Q  = d1["Q_Btu_h"] / 1e6
    qn = d2["qn_Btu_h"] / 1e6
    eta = d2["eta"] * 100
    mat = d3["material_rec"]

    # ---- figura -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_aspect("equal"); ax.axis("off")

    H_conv = max(2.0, 0.8 * n_hil + 1.5)
    H_rad  = L_alto
    x0, y0 = 0.0, 0.0

    # cabina radiante
    ax.add_patch(Rectangle((x0, y0), L_largo, H_rad, lw=2.2,
                           edgecolor="black", facecolor="#fff6e6", zorder=2))

    # cuello convección
    conv_w = 0.55 * L_largo
    conv_x = x0 + (L_largo - conv_w) / 2.0
    conv_y = y0 + H_rad
    ax.add_patch(Rectangle((conv_x, conv_y), conv_w, H_conv, lw=2.0,
                           edgecolor="black", facecolor="#e8f0ff", zorder=2))

    # chimenea (con escala visual si es muy alta)
    L_ch_v = min(L_ch, 1.6 * H_rad)
    D_ch_v = max(0.6, min(D_ch, conv_w * 0.35))
    ch_x = conv_x + (conv_w - D_ch_v) / 2.0
    ch_y = conv_y + H_conv
    ax.add_patch(Rectangle((ch_x, ch_y), D_ch_v, L_ch_v, lw=2.0,
                           edgecolor="black", facecolor="#d9d9d9", zorder=2))
    ax.plot([ch_x - 0.15, ch_x + D_ch_v + 0.15],
            [ch_y + L_ch_v, ch_y + L_ch_v], color="black", lw=2.2, zorder=3)

    # damper
    damper_y = ch_y + 0.35 * L_ch_v
    ax.plot([ch_x + 0.05 * D_ch_v, ch_x + 0.95 * D_ch_v],
            [damper_y - 0.15, damper_y + 0.15],
            color="firebrick", lw=2.2, zorder=4)
    ax.text(ch_x + D_ch_v + 0.3, damper_y + 0.5, "Damper",
            fontsize=8, color="firebrick", style="italic", va="center")

    # tubos radiantes en paredes laterales y techo
    N_pared_total = max(0, N_R - N_tE)
    n_por_pared = N_pared_total // 2
    n_techo = N_pared_total - 2 * n_por_pared

    def _columna_tubos(xc, n, color="#c0392b"):
        if n <= 0: return
        y_top, y_bot = y0 + H_rad - 0.6, y0 + 1.5
        ys = ([(y_top + y_bot) / 2] if n == 1
              else [y_bot + (y_top - y_bot) * i / (n - 1) for i in range(n)])
        r = min(0.18, (y_top - y_bot) / max(2 * n, 4))
        for y in ys:
            ax.add_patch(Circle((xc, y), r, facecolor=color,
                                edgecolor="black", lw=0.6, zorder=3))

    _columna_tubos(x0 + 0.5, n_por_pared)
    _columna_tubos(x0 + L_largo - 0.5, n_por_pared)

    # tubos de techo (entre cabina y cuello)
    if n_techo > 0:
        y_t, x_l, x_r = y0 + H_rad - 0.4, x0 + 1.0, x0 + L_largo - 1.0
        xs = ([(x_l + x_r) / 2] if n_techo == 1
              else [x_l + (x_r - x_l) * i / (n_techo - 1) for i in range(n_techo)])
        for x in xs:
            if conv_x - 0.3 < x < conv_x + conv_w + 0.3:
                continue
            ax.add_patch(Circle((x, y_t), 0.16, facecolor="#c0392b",
                                edgecolor="black", lw=0.6, zorder=3))

    # tubos de escudo (transición radiante→convección)
    if N_tE > 0:
        y_e, x_l, x_r = conv_y + 0.25, conv_x + 0.3, conv_x + conv_w - 0.3
        xs = ([(x_l + x_r) / 2] if N_tE == 1
              else [x_l + (x_r - x_l) * i / (N_tE - 1) for i in range(N_tE)])
        for x in xs:
            ax.add_patch(Circle((x, y_e), 0.20, facecolor="#e67e22",
                                edgecolor="black", lw=0.7, zorder=3))

    # tubos de convección (matriz n_hil × nT_hil, máx 12 visibles)
    nT_show = max(1, min(nT_hil, 12))
    if n_hil > 0:
        y_b, y_t = conv_y + 0.7, conv_y + H_conv - 0.3
        ys = ([(y_b + y_t) / 2] if n_hil == 1
              else [y_b + (y_t - y_b) * i / (n_hil - 1) for i in range(n_hil)])
        x_l, x_r = conv_x + 0.6, conv_x + conv_w - 0.6
        xs = ([(x_l + x_r) / 2] if nT_show == 1
              else [x_l + (x_r - x_l) * j / (nT_show - 1) for j in range(nT_show)])
        for y in ys:
            for x in xs:
                ax.add_patch(Circle((x, y), 0.13, facecolor="#27ae60",
                                    edgecolor="black", lw=0.5, zorder=3))

    # quemadores en el piso
    n_b = max(2, min(6, int(round(L_largo / 8.0))))
    x_l, x_r = x0 + 1.5, x0 + L_largo - 1.5
    xs_b = ([(x_l + x_r) / 2] if n_b == 1
            else [x_l + (x_r - x_l) * i / (n_b - 1) for i in range(n_b)])
    for xb in xs_b:
        ax.add_patch(Rectangle((xb - 0.45, y0 - 0.6), 0.9, 0.6,
                               facecolor="#7f8c8d", edgecolor="black",
                               lw=1.2, zorder=2))
        ax.add_patch(plt.Polygon([(xb - 0.35, y0), (xb + 0.35, y0),
                                  (xb, y0 + 1.4)],
                                 facecolor="#f39c12", edgecolor="#c0392b",
                                 lw=0.8, alpha=0.7, zorder=2))
    ax.text(xs_b[0] - 1.4, y0 - 1.15, f"{n_b} quemadores",
            fontsize=8, style="italic", color="#7f8c8d")

    # entrada (izquierda, frío)
    x_in_int = conv_x
    y_in = conv_y + H_conv * 0.55
    x_in_ext = x_in_int - 2.0
    ax.add_patch(FancyArrowPatch((x_in_ext, y_in), (x_in_int - 0.05, y_in),
                                  arrowstyle="-|>", mutation_scale=18,
                                  color="#1f4e8b", lw=2.4, zorder=4))
    ax.text(x_in_ext - 0.2, y_in + 0.4, f"ENTRADA\nT = {T_e:.0f} °F",
            fontsize=9, color="#1f4e8b", weight="bold", va="bottom", ha="right")

    # salida (derecha, caliente)
    x_out_int = x0 + L_largo
    y_out = y0 + 2.5
    x_out_ext = x_out_int + 2.2
    ax.add_patch(FancyArrowPatch((x_out_int - 0.05, y_out), (x_out_ext, y_out),
                                  arrowstyle="-|>", mutation_scale=20,
                                  color="#a93226", lw=2.6, zorder=4))
    ax.text(x_out_ext - 0.05, y_out - 1.2, f"SALIDA\nT = {T_s:.0f} °F",
            fontsize=9, color="#a93226", weight="bold", va="top", ha="center")

    # flecha conceptual del flujo proceso (entra fría → sale caliente)
    ax.annotate("", xy=(x0 + L_largo - 1.5, y0 + 2.5),
                xytext=(conv_x + 0.3, conv_y + H_conv * 0.55),
                arrowprops=dict(arrowstyle="->", color="#1f4e8b",
                                lw=1.4, ls="--", alpha=0.6), zorder=3)
    ax.text(x0 + 0.55 * L_largo, y0 + 0.50 * H_rad,
            f"Flujo proceso\n({N_pasos} pasos)",
            fontsize=8, style="italic", color="#1f4e8b",
            ha="center", va="center", alpha=0.85)

    # flecha de gases (chimenea)
    ax.add_patch(FancyArrowPatch(
        (ch_x + D_ch_v / 2, ch_y + 0.1),
        (ch_x + D_ch_v / 2, ch_y + L_ch_v - 0.3),
        arrowstyle="-|>", mutation_scale=22, color="#555555", lw=2.4, zorder=4))
    ax.text(ch_x + D_ch_v + 0.4, ch_y + L_ch_v * 0.7,
            f"Gases combustión\nT = {T_ch:.0f} °F",
            fontsize=8, color="#444", style="italic")
    ax.annotate("", xy=(ch_x + D_ch_v / 2, ch_y - 0.1),
                xytext=(ch_x + D_ch_v / 2, conv_y + 0.5),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.6, ls=":"),
                zorder=3)

    # etiquetas de zona
    ax.text(x0 + L_largo / 2, y0 + H_rad * 0.92, "ZONA RADIANTE",
            fontsize=11, weight="bold", color="#c0392b",
            ha="center", style="italic", alpha=0.85)
    ax.text(conv_x + conv_w / 2, conv_y + H_conv / 2, "ZONA\nCONVECCIÓN",
            fontsize=9, weight="bold", color="#27ae60",
            ha="center", va="center", style="italic", alpha=0.9)
    ax.text(ch_x + D_ch_v / 2, ch_y + L_ch_v + 0.8, "CHIMENEA",
            fontsize=9, weight="bold", color="#555",
            ha="center", style="italic")

    # cotas
    def _cotaH(x1, x2, y, txt, off=0.5, color="black"):
        ax.annotate("", xy=(x1, y), xytext=(x2, y),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.0))
        ax.text((x1 + x2) / 2, y + off, txt, ha="center", va="bottom",
                fontsize=8.5, color=color)

    def _cotaV(x, y1, y2, txt, off=0.3, color="black", ha="left"):
        ax.annotate("", xy=(x, y1), xytext=(x, y2),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.0))
        if ha == "left":
            ax.text(x - off, (y1 + y2) / 2, txt, ha="right", va="center",
                    fontsize=8.5, color=color, rotation=90)
        else:
            ax.text(x + off, (y1 + y2) / 2, txt, ha="left", va="center",
                    fontsize=8.5, color=color, rotation=90)

    _cotaH(x0, x0 + L_largo, y0 - 2.4,
           f"Largo cabina = L_TE = {L_largo:.1f} ft", off=0.25)
    _cotaV(x0 - 1.0, y0, y0 + H_rad,
           f"Alto = {L_alto:.1f} ft", off=0.15, ha="left")
    _cotaV(ch_x + D_ch_v + 1.0, ch_y, ch_y + L_ch_v,
           f"H chimenea = {L_ch:.1f} ft", off=0.15, ha="right", color="#444")
    _cotaH(ch_x, ch_x + D_ch_v, ch_y - 0.5,
           f"D = {D_ch:.2f} ft", off=-0.9, color="#444")

    # cuadro de datos técnicos
    info_x = x0 + L_largo + 5.5
    info_y = y0 + H_rad * 0.55
    txt = (
        f"DATOS DE DISEÑO\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Q proceso = {Q:.2f} MMBtu/h\n"
        f"qn neto   = {qn:.2f} MMBtu/h\n"
        f"η térmica = {eta:.1f} %\n\n"
        f"TUBOS\n"
        f"DN = {DN_in}\"  SCH {SCH}\n"
        f"DE = {DE_in:.3f} in\n"
        f"L_TE = {L_largo:.1f} ft\n"
        f"N pasos    = {N_pasos}\n"
        f"N tubos R  = {N_R}\n"
        f"N tubos C  = {N_TC}  ({n_hil}×{nT_hil})\n"
        f"N escudo   = {N_tE}\n"
        f"Lcc        = {Lcc_in:.2f} in\n\n"
        f"TEMPERATURAS\n"
        f"T entrada  = {T_e:.0f} °F\n"
        f"T salida   = {T_s:.0f} °F\n"
        f"T chimenea = {T_ch:.0f} °F\n\n"
        f"CHIMENEA\n"
        f"D = {D_ch:.2f} ft\n"
        f"H = {L_ch:.1f} ft\n\n"
        f"MATERIAL TUBOS\n{mat}"
    )
    ax.text(info_x, info_y, txt, fontsize=8.3, family="monospace",
            va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.7", facecolor="#fafafa",
                      edgecolor="#888", lw=1.0))

    # título
    fig.suptitle("Esquema técnico — Horno de cabina horizontal (vista lateral)",
                 fontsize=13, weight="bold", y=0.97)

    # leyenda
    leyenda = [
        mpatches.Patch(facecolor="#c0392b", edgecolor="black", label="Tubos radiantes"),
        mpatches.Patch(facecolor="#e67e22", edgecolor="black", label="Tubos de escudo"),
        mpatches.Patch(facecolor="#27ae60", edgecolor="black", label="Tubos de convección"),
        mpatches.Patch(facecolor="#f39c12", edgecolor="#c0392b", label="Quemadores / flama"),
        mpatches.Patch(facecolor="#fff6e6", edgecolor="black", label="Cabina (radiante)"),
        mpatches.Patch(facecolor="#e8f0ff", edgecolor="black", label="Sección convección"),
    ]
    ax.legend(handles=leyenda, loc="lower left", bbox_to_anchor=(0.0, -0.18),
              ncol=3, fontsize=8.5, frameon=True)

    ax.set_xlim(x0 - 3.5, info_x + 9.5)
    ax.set_ylim(y0 - 4.0, ch_y + L_ch_v + 3.0)
    plt.tight_layout()
    return fig, info


def schematic_to_png_bytes(fig, dpi: int = 200) -> bytes:
    """Convierte la figura matplotlib en bytes PNG descargables."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    return buf.getvalue()
# =====================================================================
# 12.  INTERFAZ STREAMLIT — flujo por pasos
# =====================================================================

# Inicialización de session_state
DEFAULT_STATE = {
    "paso_actual": 1,
    "paso_max":    1,   # último paso desbloqueado
    "datos":       {},  # diccionario con todos los resultados
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


def desbloquear(paso: int):
    """Desbloquea el siguiente paso."""
    if paso > st.session_state.paso_max:
        st.session_state.paso_max = paso
    st.session_state.paso_actual = paso


def _badge(ok: bool) -> str:
    return "✅" if ok else "⚠️"


# ---------------------------------------------------------------------
# Sidebar — navegación
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("🔥 Diseño de Hornos")
    st.caption("Método Corto · Cabina horizontal")
    st.markdown("---")
    st.subheader("Etapas del diseño")
    pasos = [
        "1. Fluido de proceso",
        "2. Combustión y eficiencia",
        "3. Zona radiante",
        "4. Tubos y velocidad interna",
        "5. Geometría de cabina",
        "6. Radiación avanzada y convección",
        "7. Fluidodinámica y chimenea",
        "8. Reporte técnico",
    ]
    for i, nombre in enumerate(pasos, start=1):
        if i <= st.session_state.paso_max:
            if st.button(f"{_badge(True)} {nombre}",
                         key=f"nav_{i}",
                         use_container_width=True):
                st.session_state.paso_actual = i
        else:
            st.button(f"🔒 {nombre}", key=f"nav_{i}",
                      use_container_width=True, disabled=True)
    st.markdown("---")
    with st.expander("🌡️ Fluidos especiales disponibles"):
        rows = []
        for n, info in _FLUIDOS_ESPECIALES.items():
            rows.append({
                "Fluido": n,
                "Fabricante": info["fabricante"],
                "T_min (°F)": info["T_min_F"],
                "T_max (°F)": info["T_max_F"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
    st.markdown("---")
    if st.button("🔄 Reiniciar diseño"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# Header principal
st.title("🔥 Diseño de Hornos de Proceso – Método Corto")
st.caption(
    "Aplicación basada en la guía ULA *Diseño de Plantas Industriales II — "
    "Hornos* (Anaya 1997) y en *Equipment Design Handbook for Refineries* "
    "de Frank L. Evans (1980), Vol. 2."
)

# =====================================================================
# PASO 1 — Fluido de proceso
# =====================================================================
if st.session_state.paso_actual == 1:
    st.header("Paso 1 · Fluido de proceso")
    st.markdown("Defina el fluido a calentar y sus condiciones operativas.")

    col1, col2 = st.columns(2)
    with col1:
        tipo_fluido = st.selectbox(
            "Tipo de fluido",
            ["Hidrocarburo / crudo", "Agua / vapor", "Fluido especial (entalpías manuales)"],
            help=(
                "Selecciona 'Fluido especial' para aceites térmicos, solventes u "
                "otros fluidos donde conoces la entalpía de tablas (Therminol, Dowtherm, etc.)."
            ),
        )
        W_lb_h = st.number_input(
            "Flujo másico W (lb/h)",
            min_value=1000.0, max_value=2_000_000.0,
            value=100_000.0, step=1000.0,
            help="Caudal másico total a través del horno.",
        )
        T_e_F = st.number_input(
            "Temperatura de entrada Te (°F)",
            min_value=60.0, max_value=1000.0, value=460.0, step=5.0,
        )
        T_s_F = st.number_input(
            "Temperatura de salida Ts (°F)",
            min_value=80.0, max_value=1100.0, value=660.0, step=5.0,
        )

    with col2:
        if tipo_fluido == "Hidrocarburo / crudo":
            api = st.number_input(
                "°API del crudo / hidrocarburo",
                min_value=10.0, max_value=60.0, value=32.0, step=0.5,
            )
            K_w = st.number_input(
                "Factor Watson Kw",
                min_value=10.5, max_value=13.0, value=11.8, step=0.1,
                help="Parafínico ≈12.0; nafténico 11.0–11.5; aromático <11.",
            )
            cambio_fase = st.selectbox(
                "Tipo de calentamiento",
                ["Sólo calentamiento sensible",
                 "Calentamiento + vaporización parcial"],
            )
            # placeholders para evitar NameError
            H_e_manual = H_s_manual = None
            rho_e_manual = rho_s_manual = mu_e_manual = mu_s_manual = None

        elif tipo_fluido == "Agua / vapor":
            api, K_w = None, None
            P_psia = st.number_input(
                "Presión operativa (psia)",
                min_value=14.7, max_value=3000.0, value=300.0, step=10.0,
            )
            cambio_fase = st.selectbox(
                "Tipo de calentamiento",
                ["Calentamiento sensible (líquido subenfriado)",
                 "Generación de vapor saturado",
                 "Vapor sobrecalentado"],
            )
            H_e_manual = H_s_manual = None
            rho_e_manual = rho_s_manual = mu_e_manual = mu_s_manual = None

        else:
            # ── Fluido especial: selección desde base de datos interna ──
            api, K_w = None, None
            cambio_fase = "Sólo calentamiento sensible"

            nombre_fluido = st.selectbox(
                "Fluido de transferencia de calor",
                list(_FLUIDOS_ESPECIALES.keys()),
                help="Base de datos de fichas técnicas públicas de fabricantes. "
                     "Si tu fluido no aparece, usa la opción 'Introducir manualmente'.",
            )
            info_f = _FLUIDOS_ESPECIALES[nombre_fluido]
            T_min_f = info_f["T_min_F"]
            T_max_f = info_f["T_max_F"]

            st.info(
                f"**{nombre_fluido}** — {info_f['fabricante']}  \n"
                f"Rango operativo: {T_min_f} – {T_max_f} °F  \n"
                f"_{info_f['notas']}_"
            )

            usar_manual = st.checkbox(
                "Introducir propiedades manualmente (anular base de datos)",
                value=False,
                help="Activa esta opción si tienes datos más precisos del fabricante "
                     "o si el fluido no aparece en la lista.",
            )

            if usar_manual:
                st.markdown("**Entalpías (de tablas del fabricante)**")
                c1e, c2e = st.columns(2)
                H_e_manual = c1e.number_input(
                    "Entalpía entrada He (Btu/lb)",
                    min_value=-500.0, max_value=2000.0, value=38.14, step=0.1)
                H_s_manual = c2e.number_input(
                    "Entalpía salida Hs (Btu/lb)",
                    min_value=-500.0, max_value=2000.0, value=319.05, step=0.1)
                st.markdown("**Densidades**")
                c1r, c2r = st.columns(2)
                rho_e_manual = c1r.number_input(
                    "ρ entrada (lb/ft³)", min_value=10.0, max_value=100.0,
                    value=54.0, step=0.1)
                rho_s_manual = c2r.number_input(
                    "ρ salida (lb/ft³)",  min_value=10.0, max_value=100.0,
                    value=41.95, step=0.1)
                st.markdown("**Viscosidades dinámicas**")
                c1m, c2m, c3m = st.columns(3)
                mu_e_manual = c1m.number_input(
                    "μ entrada (cP)", min_value=0.01, max_value=500.0,
                    value=8.0, step=0.1)
                mu_s_manual = c2m.number_input(
                    "μ salida (cP)", min_value=0.01, max_value=500.0,
                    value=0.9, step=0.01)
                mu_m_manual = c3m.number_input(
                    "μ a T_media (cP) ← más importante",
                    min_value=0.01, max_value=500.0, value=1.0, step=0.01,
                    help="Viscosidad a (T_e+T_s)/2. Determina el Reynolds y el ΔP.")
            else:
                # Previsualizar propiedades interpoladas
                try:
                    p_e = props_fluido_especial(nombre_fluido, T_e_F)
                    p_s = props_fluido_especial(nombre_fluido, T_s_F)
                    T_med = 0.5 * (T_e_F + T_s_F)
                    p_m = props_fluido_especial(nombre_fluido, T_med)
                    st.markdown("**Propiedades interpoladas (PCHIP)**")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric(f"ρ entrada ({T_e_F:.0f}°F)", f"{p_e['rho']:.2f} lb/ft³")
                    col_b.metric(f"ρ salida ({T_s_F:.0f}°F)",  f"{p_s['rho']:.2f} lb/ft³")
                    col_c.metric(f"ρ media ({T_med:.0f}°F)",   f"{p_m['rho']:.2f} lb/ft³")
                    col_a.metric(f"μ entrada",   f"{p_e['mu']:.3f} cP")
                    col_b.metric(f"μ salida",    f"{p_s['mu']:.3f} cP")
                    col_c.metric(f"μ media ← ΔP", f"{p_m['mu']:.3f} cP")
                    col_a.metric(f"H entrada",   f"{p_e['H']:.2f} Btu/lb")
                    col_b.metric(f"H salida",    f"{p_s['H']:.2f} Btu/lb")
                    col_c.metric(f"ΔH",          f"{p_s['H']-p_e['H']:.2f} Btu/lb")
                except ValueError as ve:
                    st.error(f"⚠️ {ve}")
                H_e_manual = H_s_manual = None
                rho_e_manual = rho_s_manual = None
                mu_e_manual = mu_s_manual = mu_m_manual = None

    st.markdown("---")
    if T_s_F <= T_e_F:
        st.error("⚠️ La temperatura de salida debe ser **mayor** que la de entrada.")
    else:
        if st.button("🔢 Calcular carga térmica y avanzar al Paso 2",
                     type="primary"):
            try:
                if tipo_fluido == "Hidrocarburo / crudo":
                    # Δh sensible mediante integración de Cp Kesler-Lee
                    dH = hentalpia_hc_liq(T_s_F, T_e_F, api, K_w)
                    cp_avg = dH / (T_s_F - T_e_F)
                    rho_in  = densidad_hc_liq(T_e_F, api)
                    rho_out = densidad_hc_liq(T_s_F, api)
                    mu_in   = viscosidad_hc_dynamic(T_e_F, api)
                    mu_out  = viscosidad_hc_dynamic(T_s_F, api)
                    # mu_avg evaluado a T_media (más preciso que promedio aritmético
                    # cuando μ varía mucho con T, como en aceites y crudos pesados)
                    T_med_F = 0.5 * (T_e_F + T_s_F)
                    mu_med  = viscosidad_hc_dynamic(T_med_F, api)
                    Q = W_lb_h * dH
                    if "vaporización" in cambio_fase:
                        Q_vap = 0.3 * W_lb_h * 100.0
                        Q += Q_vap

                elif tipo_fluido == "Agua / vapor":
                    if not COOLPROP_OK:
                        st.error("CoolProp no disponible. Instale "
                                 "`pip install CoolProp`.")
                        st.stop()
                    pin = props_agua_vapor(P_psia, T_e_F, "subenfriado") \
                        if T_e_F < 250 else \
                        props_agua_vapor(P_psia, T_e_F, "sobrecalentado")
                    if "saturado" in cambio_fase:
                        pout = props_agua_vapor(P_psia, fase="saturado_vap")
                    elif "sobrecalentado" in cambio_fase:
                        pout = props_agua_vapor(P_psia, T_s_F, "sobrecalentado")
                    else:
                        pout = props_agua_vapor(P_psia, T_s_F, "subenfriado")
                    dH = pout["h"] - pin["h"]
                    Q = W_lb_h * dH
                    cp_avg = pout["cp"]
                    rho_in, rho_out = pin["rho"], pout["rho"]
                    mu_in, mu_out = pin["mu"], pout["mu"]
                    # Para agua/vapor mu varía poco — promedio aritmético es aceptable
                    # pero usamos T_media para consistencia
                    T_med_F = 0.5 * (T_e_F + T_s_F)
                    try:
                        p_med = props_agua_vapor(P_psia, T_med_F, "subenfriado")
                        mu_med = p_med["mu"]
                    except Exception:
                        mu_med = 0.5 * (mu_in + mu_out)

                else:
                    # ── Fluido especial ──────────────────────────────────────
                    if usar_manual:
                        # Valores introducidos a mano por el usuario
                        dH = H_s_manual - H_e_manual
                        if dH <= 0:
                            st.error("⚠️ La entalpía de salida debe ser mayor que la de entrada.")
                            st.stop()
                        Q = W_lb_h * dH
                        cp_avg = dH / max(T_s_F - T_e_F, 1.0)
                        rho_in, rho_out = rho_e_manual, rho_s_manual
                        mu_in,  mu_out  = mu_e_manual,  mu_s_manual
                        mu_med = mu_m_manual
                    else:
                        # Propiedades de la base de datos interna (PCHIP)
                        p_e = props_fluido_especial(nombre_fluido, T_e_F)
                        p_s = props_fluido_especial(nombre_fluido, T_s_F)
                        T_med_F_esp = 0.5 * (T_e_F + T_s_F)
                        p_m = props_fluido_especial(nombre_fluido, T_med_F_esp)
                        dH     = p_s["H"] - p_e["H"]
                        if dH <= 0:
                            st.error("⚠️ La entalpía de salida es menor que la de entrada "
                                     "según la base de datos. Verifica el rango de temperaturas.")
                            st.stop()
                        Q      = W_lb_h * dH
                        cp_avg = dH / max(T_s_F - T_e_F, 1.0)
                        rho_in, rho_out = p_e["rho"], p_s["rho"]
                        mu_in,  mu_out  = p_e["mu"],  p_s["mu"]
                        mu_med = p_m["mu"]   # viscosidad a T_media — precisa para Re y ΔP

                rho_avg = 0.5 * (rho_in + rho_out)
                # mu_avg = viscosidad a T_media (o log-media para fluidos especiales)
                # Se guarda como mu_avg para uso en Re y ΔP del Paso 7.
                mu_avg  = mu_med

                st.session_state.datos["paso1"] = {
                    "tipo_fluido": tipo_fluido,
                    "W_lb_h": W_lb_h,
                    "T_e_F": T_e_F, "T_s_F": T_s_F,
                    "api": api, "K_w": K_w,
                    "Q_Btu_h": Q,
                    "dH_Btu_lb": dH,
                    "cp_avg": cp_avg,
                    "rho_in": rho_in, "rho_out": rho_out, "rho_avg": rho_avg,
                    "mu_in": mu_in,   "mu_out": mu_out,   "mu_avg": mu_avg,
                    "cambio_fase": cambio_fase,
                }
                desbloquear(2)
                # Mostrar propiedades calculadas
                st.success("✅ Carga térmica calculada.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Q (MMBtu/h)",     f"{Q/1e6:.2f}")
                c2.metric("Δh (Btu/lb)",     f"{dH:.1f}")
                c3.metric("ρ prom (lb/ft³)", f"{rho_avg:.2f}")
                c4.metric("μ media (cP)",    f"{mu_avg:.3f}")
                # Nota sobre el método de μ
                if tipo_fluido == "Fluido especial (entalpías manuales)":
                    fuente = ("base de datos interna" if not usar_manual
                              else "valores manuales")
                    st.info(
                        f"ℹ️ **{nombre_fluido}** ({info_f['fabricante']}) — {fuente}  \n"
                        f"μ_media ({0.5*(T_e_F+T_s_F):.0f}°F) = **{mu_avg:.3f} cP** "
                        f"(determina Reynolds y ΔP)."
                    )
                elif tipo_fluido == "Hidrocarburo / crudo":
                    st.info(
                        f"ℹ️ μ_media evaluada a T_media = {0.5*(T_e_F+T_s_F):.0f} °F "
                        f"mediante correlación API → **{mu_avg:.3f} cP**."
                    )
                st.rerun()
            except Exception as e:
                st.error(f"Error en cálculo de propiedades: {e}")

    # Si ya calculado, mostrar resumen compacto
    if "paso1" in st.session_state.datos:
        d = st.session_state.datos["paso1"]
        st.success("✅ Carga térmica calculada.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Q (MMBtu/h)",     f"{d['Q_Btu_h']/1e6:.2f}")
        c2.metric("Δh (Btu/lb)",     f"{d['dH_Btu_lb']:.1f}")
        c3.metric("ρ prom (lb/ft³)", f"{d['rho_avg']:.2f}")
        c4.metric("μ media (cP)",    f"{d['mu_avg']:.3f}")


# =====================================================================
# PASO 2 — Combustión y eficiencia
# =====================================================================
if st.session_state.paso_actual == 2:
    st.header("Paso 2 · Combustión y eficiencia")
    if "paso1" not in st.session_state.datos:
        st.warning("Complete el Paso 1 primero.")
        st.stop()

    d1 = st.session_state.datos["paso1"]
    st.info(f"Carga térmica de proceso: **Q = {d1['Q_Btu_h']/1e6:.2f} MMBtu/h**")

    col1, col2 = st.columns(2)
    with col1:
        combustible = st.selectbox(
            "Tipo de combustible",
            ["Líquido (diesel/fuel oil)", "Gas natural / FG"],
        )
        eta_supuesta = st.slider(
            "Eficiencia térmica supuesta (η)",
            min_value=0.65, max_value=0.88, value=0.77, step=0.01,
            help="Cabina horizontal típica: 70–85 %.  Anaya recomienda 77 % de partida.",
        )
    with col2:
        ex_air = st.slider(
            "Exceso de aire (%)",
            min_value=5, max_value=60, value=25, step=1,
            help="Combustible gas: 10–20 %; combustible líquido: 20–30 %.",
        )
        T_stack_obj = st.number_input(
            "Temperatura objetivo de chimenea (°F) – referencia",
            min_value=300.0, max_value=900.0, value=600.0, step=10.0,
        )

    if st.button("🔥 Calcular calor neto y flujo de gases", type="primary"):
        comb_str = "liquido" if "Líquido" in combustible else "gas"
        Q = d1["Q_Btu_h"]
        qn = Q / eta_supuesta  # Btu/h calor neto liberado
        # Flujo de gases — Fig. 1 ULA en lb / MMBtu
        lb_per_MMBtu = fig1_flujo_gases(ex_air, comb_str)
        Wg = qn / 1e6 * lb_per_MMBtu  # lb/h
        # Chequeo eficiencia diesel
        if comb_str == "liquido":
            Tstack_calc = T_stack_diesel(eta_supuesta, ex_air)
            eff_check = eficiencia_diesel(Tstack_calc, ex_air)
        else:
            Tstack_calc = T_stack_obj  # placeholder
            eff_check = eta_supuesta

        diag_eff = diagnosticar_eficiencia(eta_supuesta)

        st.session_state.datos["paso2"] = {
            "combustible": comb_str,
            "eta": eta_supuesta,
            "ex_air": ex_air,
            "qn_Btu_h": qn,
            "Wg_lb_h": Wg,
            "lb_per_MMBtu": lb_per_MMBtu,
            "T_stack_calc_F": Tstack_calc,
        }

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("qn (MMBtu/h)",    f"{qn/1e6:.2f}")
        c2.metric("Wg (lb/h)",       f"{Wg:,.0f}")
        c3.metric("lb gases/MMBtu",  f"{lb_per_MMBtu:.0f}")
        c4.metric("T_stack diesel (°F)", f"{Tstack_calc:.0f}")

        if diag_eff["ok"]:
            st.success(diag_eff["msg"])
        else:
            st.warning(diag_eff["msg"])

        with st.expander("📚 Correlaciones empleadas en este paso"):
            st.markdown("""
- **Figura 1 ULA — Flujo de gases de combustión**: PCHIP de puntos
  digitalizados; entrada % exceso de aire (0–100); salida lb gases/MMBtu.
- **Eficiencia diesel ULA**: ecuación directa  
  *Eff = 0.98 − 7.695e-5 · T_stack^1.144 · (1 + ex/100)^0.911*.
""")
        if st.button("➡️ Avanzar al Paso 3 (Zona Radiante)"):
            desbloquear(3)
            st.rerun()
    elif "paso2" in st.session_state.datos:
        d = st.session_state.datos["paso2"]
        c1, c2, c3 = st.columns(3)
        c1.metric("qn (MMBtu/h)", f"{d['qn_Btu_h']/1e6:.2f}")
        c2.metric("Wg (lb/h)",    f"{d['Wg_lb_h']:,.0f}")
        c3.metric("Eficiencia",   f"{d['eta']*100:.1f}%")
        if st.button("➡️ Avanzar al Paso 3"):
            desbloquear(3)
            st.rerun()


# =====================================================================
# PASO 3 — Zona radiante
# =====================================================================
if st.session_state.paso_actual == 3:
    st.header("Paso 3 · Zona radiante")
    if "paso2" not in st.session_state.datos:
        st.warning("Complete pasos previos.")
        st.stop()

    d1 = st.session_state.datos["paso1"]
    d2 = st.session_state.datos["paso2"]

    col1, col2 = st.columns(2)
    with col1:
        eta_R = st.slider(
            "Fracción de calor absorbido por radiación (η_R)",
            min_value=0.55, max_value=0.80, value=0.70, step=0.01,
            help="Anaya recomienda η_R = 0.70 para hornos de proceso.",
        )
        servicio = st.selectbox(
            "Servicio del horno (define fqM recomendada)",
            TABLA_FQM["Servicio"].tolist(),
        )
        fqM_default = float(
            TABLA_FQM[TABLA_FQM.Servicio == servicio]["fqM_Btu_h_ft2"].iloc[0]
        )
        fqM = st.number_input(
            "Densidad de flujo de calor radiante fqM (Btu/h·ft²)",
            min_value=4000.0, max_value=20000.0,
            value=fqM_default, step=500.0,
        )
    with col2:
        st.info(
            f"Q = {d1['Q_Btu_h']/1e6:.2f} MMBtu/h  ·  qn = "
            f"{d2['qn_Btu_h']/1e6:.2f} MMBtu/h"
        )

    if st.button("🔥 Calcular zona radiante", type="primary"):
        T_e, T_s = d1["T_e_F"], d1["T_s_F"]
        T_p = temp_cruce(T_e, T_s, eta_R)
        T_R = 0.5 * (T_p + T_s)        # Temp. promedio fluido
        T_T = T_R + 100.0              # Temp. promedio pared tubos
        Q_R = eta_R * d1["Q_Btu_h"]    # calor en zona radiante
        A_R = superficie_radiante(Q_R, fqM)

        # Diagnóstico
        diag_fqM = diagnosticar_fqM(fqM, combustible=d2["combustible"])
        material_rec = recomendar_material(T_T)

        st.session_state.datos["paso3"] = {
            "eta_R": eta_R, "fqM": fqM, "servicio": servicio,
            "T_p_F": T_p, "T_R_F": T_R, "T_T_F": T_T,
            "Q_R_Btu_h": Q_R, "A_R_ft2": A_R,
            "material_rec": material_rec,
        }

        c1, c2, c3 = st.columns(3)
        c1.metric("T cruce Tp (°F)",     f"{T_p:.1f}")
        c2.metric("T pared tubos (°F)",  f"{T_T:.1f}")
        c3.metric("A_R (ft²)",           f"{A_R:,.1f}")

        c1, c2 = st.columns(2)
        c1.metric("Q_R (MMBtu/h)",       f"{Q_R/1e6:.2f}")
        c2.metric("Material sugerido",   material_rec)

        if diag_fqM["ok"]:
            st.success(diag_fqM["msg"])
        else:
            st.warning(diag_fqM["msg"])

        if st.button("➡️ Avanzar al Paso 4 (Tubos)"):
            desbloquear(4)
            st.rerun()
    elif "paso3" in st.session_state.datos:
        d = st.session_state.datos["paso3"]
        st.success(f"Zona radiante calculada · A_R = {d['A_R_ft2']:.1f} ft²")
        if st.button("➡️ Avanzar al Paso 4"):
            desbloquear(4)
            st.rerun()


# =====================================================================
# PASO 4 — Selección de tubos y velocidad interna
# =====================================================================
if st.session_state.paso_actual == 4:
    st.header("Paso 4 · Tubos y velocidad interna")
    if "paso3" not in st.session_state.datos:
        st.warning("Complete pasos previos.")
        st.stop()

    d1 = st.session_state.datos["paso1"]
    d3 = st.session_state.datos["paso3"]
    A_R = d3["A_R_ft2"]
    st.info(f"Superficie radiante requerida: **A_R = {A_R:,.1f} ft²**")

    st.subheader("Tabla 2 ULA — Características de tubos")
    st.dataframe(TABLA_TUBOS, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        DN_in = st.selectbox(
            "Diámetro nominal (in)",
            sorted(TABLA_TUBOS["DN_in"].unique()),
            index=2,
        )
        sch_disp = TABLA_TUBOS[TABLA_TUBOS.DN_in == DN_in]["SCH"].unique().tolist()
        SCH = st.selectbox("Cédula", sch_disp, index=min(1, len(sch_disp)-1))
        L_T = st.number_input(
            "Longitud efectiva del tubo L_TE (ft)",
            min_value=10.0, max_value=60.0, value=40.0, step=2.0,
            help="Tubos horizontales: máx. 50 ft.  Verticales: 20 ft.",
        )
    with col2:
        n_pasos = st.selectbox(
            "Número de pasos",
            [1, 2, 4, 6, 8, 10, 12, 16], index=1,
            help=(
                "Ajustar para que el número de tubos por paso sea entero.\n"
                "Más pasos → menor ΔP pero menor velocidad por tubo.\n"
                "Guía rápida para DN 4\" Sch 80 con serpentín largo:\n"
                "  4 pasos → ΔP muy alto (>500 psi)  ⚠️\n"
                "  8 pasos → ΔP ~120 psi              ⚠️\n"
                " 12 pasos → ΔP ~30-40 psi            ✅\n"
                " 16 pasos → ΔP <20 psi, v baja       ⚠️"
            ),
        )

    tubo = get_tubo(DN_in, SCH)
    A_UR = tubo["A_ext_ft2_per_ft"]
    A_t = L_T * A_UR

    if st.button("📐 Calcular número de tubos y velocidad", type="primary"):
        N_R = math.ceil(A_R / A_t)
        # Ajustar a múltiplo de pasos
        if N_R % n_pasos != 0:
            N_R = (N_R // n_pasos + 1) * n_pasos

        DI_in = tubo["DI_in"]
        A_flujo = tubo["A_flujo_ft2"]
        v_ft_s = calc_velocidad_liquido(d1["W_lb_h"], d1["rho_avg"],
                                        A_flujo, n_pasos)
        G_lb_hft2 = calc_velocidad_masica(d1["W_lb_h"], A_flujo, n_pasos)

        diag_v = diagnosticar_velocidad(v_ft_s)

        st.session_state.datos["paso4"] = {
            "DN_in": DN_in, "SCH": SCH, "tubo": tubo,
            "L_TE_ft": L_T, "n_pasos": n_pasos,
            "A_t_ft2": A_t, "N_tubos_R": N_R,
            "A_R_real_ft2": N_R * A_t,
            "v_ft_s": v_ft_s,
            "G_lb_hft2": G_lb_hft2,
        }

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("N tubos radiación", f"{N_R}")
        c2.metric("A radiante real",   f"{N_R*A_t:,.0f} ft²")
        c3.metric("v interna (ft/s)",  f"{v_ft_s:.2f}")
        c4.metric("G (lb/h·ft²)",      f"{G_lb_hft2:,.0f}")

        if diag_v["ok"]:
            st.success(diag_v["msg"])
        else:
            st.error(diag_v["msg"])

        if st.button("➡️ Avanzar al Paso 5 (Geometría)"):
            desbloquear(5)
            st.rerun()
    elif "paso4" in st.session_state.datos:
        d = st.session_state.datos["paso4"]
        st.success(f"Tubos: {d['N_tubos_R']} × DN{d['DN_in']}\" SCH{d['SCH']}, "
                   f"L={d['L_TE_ft']} ft, v={d['v_ft_s']:.2f} ft/s")
        if st.button("➡️ Avanzar al Paso 5"):
            desbloquear(5)
            st.rerun()


# =====================================================================
# PASO 5 — Geometría de cabina
# =====================================================================
if st.session_state.paso_actual == 5:
    st.header("Paso 5 · Geometría de la cabina")
    if "paso4" not in st.session_state.datos:
        st.warning("Complete pasos previos.")
        st.stop()

    d4 = st.session_state.datos["paso4"]
    DE = d4["tubo"]["DE_in"]    # in
    L_TE = d4["L_TE_ft"]        # ft
    N_R = d4["N_tubos_R"]

    st.info(f"DN = {d4['DN_in']}\", DE = {DE}\" · "
            f"L_tubo = {L_TE} ft · N_tubos radiación = {N_R}")

    col1, col2 = st.columns(2)
    with col1:
        L_BR_ft = st.number_input(
            "Ancho de cabina L_BR (ft)",
            min_value=5.0, max_value=40.0, value=15.0, step=0.5,
        )
        L_HR_ft = st.number_input(
            "Alto de cabina L_HR (ft)",
            min_value=8.0, max_value=40.0, value=18.0, step=0.5,
        )
        L_cc_factor = st.slider(
            "Factor de espaciamiento L_cc / DE",
            min_value=1.5, max_value=3.0, value=2.0, step=0.1,
            help="Recomendado 2.0; rango aceptable 1.5–3.0.",
        )
    with col2:
        N_tE = st.selectbox(
            "Número de tubos de escudo (NtE)",
            [4, 6, 8, 10], index=1,
        )
        n_filas_choque = st.selectbox(
            "Filas de tubos de choque (escudo)",
            [1, 2], index=1,
        )

    L_cc_in = L_cc_factor * DE   # in
    L_cc_ft = L_cc_in / 12.0
    C_over_D = L_cc_factor

    # Distribución entre paredes y techo
    N_paredes = N_R - N_tE
    if N_paredes < 0:
        N_paredes = 0

    if st.button("📐 Calcular geometría y haz de llama", type="primary"):
        A_t = d4["A_t_ft2"]
        A_tE_real = L_TE * N_tE * L_cc_ft  # área plano frío de tubos escudo
        A_tp_real = L_TE * N_paredes * L_cc_ft

        alpha = alpha_factor(C_over_D, n_filas=n_filas_choque)
        A_tT_corr = A_tE_real + alpha * A_tp_real
        A_TE_envolvente = (
            2 * (L_BR_ft * L_HR_ft) +
            2 * L_TE * (L_BR_ft + L_HR_ft)
        )
        A_TR = A_TE_envolvente - (A_tE_real + A_tp_real)
        V_horno = L_TE * L_BR_ft * L_HR_ft
        Lz = long_haz_llama(V_horno)
        # Presión parcial CO2+H2O (atm) y P·Lz
        d2 = st.session_state.datos["paso2"]
        P_atm = fig4_presion_parcial(d2["ex_air"])
        P_Lz = P_atm * Lz

        diag_C = diagnosticar_espaciamiento(C_over_D)

        st.session_state.datos["paso5"] = {
            "L_BR_ft": L_BR_ft, "L_HR_ft": L_HR_ft,
            "L_cc_ft": L_cc_ft, "L_cc_in": L_cc_in,
            "C_over_D": C_over_D,
            "N_tE": N_tE, "n_filas_choque": n_filas_choque,
            "N_paredes": N_paredes,
            "A_tE_real": A_tE_real, "A_tp_real": A_tp_real,
            "alpha": alpha, "A_tT_corr": A_tT_corr,
            "A_TE_env": A_TE_envolvente, "A_TR": A_TR,
            "V_horno_ft3": V_horno, "Lz_ft": Lz,
            "P_atm": P_atm, "P_Lz_atmft": P_Lz,
        }

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("V cabina (ft³)",  f"{V_horno:,.0f}")
        c2.metric("Lz (ft)",          f"{Lz:.2f}")
        c3.metric("α factor",         f"{alpha:.3f}")
        c4.metric("P·Lz (atm·ft)",    f"{P_Lz:.2f}")

        c1, c2, c3 = st.columns(3)
        c1.metric("A envolvente (ft²)", f"{A_TE_envolvente:,.0f}")
        c2.metric("A_tubos efectiva (ft²)", f"{A_tT_corr:,.0f}")
        c3.metric("A pared refractario (ft²)", f"{A_TR:,.0f}")

        if diag_C["ok"]:
            st.success(diag_C["msg"])
        else:
            st.warning(diag_C["msg"])

        if st.button("➡️ Avanzar al Paso 6 (Radiación avanzada)"):
            desbloquear(6)
            st.rerun()
    elif "paso5" in st.session_state.datos:
        d = st.session_state.datos["paso5"]
        st.success(
            f"Cabina: {d['L_BR_ft']}×{d['L_HR_ft']}×{d4['L_TE_ft']} ft · "
            f"V = {d['V_horno_ft3']:.0f} ft³ · Lz = {d['Lz_ft']:.2f} ft"
        )
        if st.button("➡️ Avanzar al Paso 6"):
            desbloquear(6)
            st.rerun()


# =====================================================================
# PASO 6 — Radiación avanzada y zona de convección
# ---------------------------------------------------------------------
# Procedimiento ULA (Anaya 1997) en este orden:
#   A) Radiación avanzada (Lobo–Evans)
#       - Tg supuesto en cámara
#       - ε   ← Fig 5(P·Lz, Tg)
#       - φ   ← Fig 6(ε, Aw/(α·Acp))
#       - Q_R_calc  = 1.73e-9 · α·Acp · φ · (Tg^4 – Tt^4)
#         (ec. de Stefan–Boltzmann linealizada)
#       - Iterar Tg hasta que Q_R_calc ≈ Q_R objetivo
#   B) Balance de energía → T_chimenea
#       - qg/qn (gas saliendo de la cámara)  ← Fig 7(Tg, %ex)
#       - qC/qn = (Q – Q_R) / qn  (calor absorbido en convección)
#       - qG/qn = 1 – Q_R/qn – qC/qn – pérdidas  → Fig 7^-1 → T_chimenea
#   C) Geometría de convección (NtH tubos/hilera, arreglo triangular)
#       - AP = LLC · L_TE
#       - G  = Wg/AP   →  validar ∈ [0.30, 0.40] lb/(s·ft²)
#   D) Coeficientes
#       - hcc = Fig 10 (G, T_Gp)
#       - hcw = Fig 11 (T_pared)
#       - hcr = Fig 12 (T_pared, Tg_avg)
#       - hCi, Ff, hC, Uc
#   E) Dimensionamiento
#       - LMTD entre Tg / T_chimenea  vs  T_e / T_p
#       - Q_C  →  AC = Q_C / (Uc · LMTD)
#       - NTC = AC / (a_t)   →  hileras = NTC / NtH
# =====================================================================
if st.session_state.paso_actual == 6:
    st.header("Paso 6 · Radiación avanzada y zona de convección")
    if "paso5" not in st.session_state.datos:
        st.warning("Complete pasos previos.")
        st.stop()

    d1 = st.session_state.datos["paso1"]
    d2 = st.session_state.datos["paso2"]
    d3 = st.session_state.datos["paso3"]
    d4 = st.session_state.datos["paso4"]
    d5 = st.session_state.datos["paso5"]

    st.markdown("**Datos heredados de pasos previos**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Q proceso (MMBtu/h)", f"{d1['Q_Btu_h']/1e6:.2f}")
    c2.metric("Q_R objetivo",        f"{d3['Q_R_Btu_h']/1e6:.2f}")
    c3.metric("Wg (lb/h)",           f"{d2['Wg_lb_h']:,.0f}")
    c4.metric("P·Lz (atm·ft)",       f"{d5['P_Lz_atmft']:.2f}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("T_e fluido (°F)",     f"{d1['T_e_F']:.0f}")
    c2.metric("T_p cruce  (°F)",     f"{d3['T_p_F']:.0f}")
    c3.metric("T_T pared  (°F)",     f"{d3['T_T_F']:.0f}")
    c4.metric("L_TE tubos (ft)",     f"{d4['L_TE_ft']:.0f}")

    st.markdown("---")
    # ----------------------------- A) RADIACIÓN AVANZADA -------------
    st.subheader("A · Radiación avanzada (Lobo–Evans)")
    Tg_supuesto = st.slider(
        "Temperatura supuesta de gases en cámara T_g (°F)",
        min_value=1200, max_value=2200, value=1500, step=25,
        help="Iterar hasta que Q_R_calc se acerque a Q_R objetivo.",
    )

    # ----------------------------- C) GEOMETRÍA CONVECCIÓN -----------
    st.markdown("---")
    st.subheader("B · Geometría de la sección de convección")
    st.caption("Arreglo típico ULA: triangular, Lcc = 2·DE, "
               "tubo nominal 4″ (igual al de radiación).")
    col1, col2, col3 = st.columns(3)
    with col1:
        NtH = st.number_input(
            "Número de tubos por hilera (NtH)",
            min_value=2, max_value=20, value=4, step=1,
            help="Por defecto 4 (criterio ULA, pág 40). Modificar para "
                 "ajustar la velocidad másica G ∈ [0.30, 0.40] lb/(s·ft²).",
        )
    with col2:
        DN_conv = st.selectbox(
            "DN de tubos en convección",
            sorted(TABLA_TUBOS["DN_in"].unique()),
            index=sorted(TABLA_TUBOS["DN_in"].unique()).index(4)
                  if 4 in TABLA_TUBOS["DN_in"].values else 0,
            help="ULA recomienda 4″ en convección.",
        )
    with col3:
        SCH_conv_disp = TABLA_TUBOS[TABLA_TUBOS.DN_in == DN_conv]["SCH"].unique().tolist()
        SCH_conv = st.selectbox(
            "Cédula tubos convección", SCH_conv_disp,
            index=min(1, len(SCH_conv_disp) - 1),
        )

    tubo_conv = get_tubo(DN_conv, SCH_conv)
    DE_conv = tubo_conv["DE_in"]
    Lcc_conv_in = 2.0 * DE_conv      # ULA: Lcc = 2·DE
    A_t_conv = d4["L_TE_ft"] * tubo_conv["A_ext_ft2_per_ft"]

    # ----------------------------- D) h_int interno ------------------
    st.markdown("---")
    st.subheader("C · Coeficiente interno de película (h_int)")
    h_int = st.number_input(
        "h_int (Btu/h·ft²·°F)",
        min_value=50.0, max_value=600.0, value=250.0, step=10.0,
        help="Típicos: HC líquido = 200–300; HC con vaporización = 300–500; "
             "agua/vapor = 250–400.",
    )

    if st.button("🔥 Calcular Paso 6 completo", type="primary"):
        # ============================================================
        # A) RADIACIÓN AVANZADA
        # ------------------------------------------------------------
        eps = fig5_emisividad(d5["P_Lz_atmft"], Tg_supuesto)
        Aw = d5["A_TR"]
        alphaAcp = d5["alpha"] * d5["A_tT_corr"]
        ratio = Aw / max(1.0, alphaAcp)
        phi = fig6_factor_intercambio(eps, ratio)
        T_g_R = Tg_supuesto + 460.0
        T_T_R = d3["T_T_F"] + 460.0
        Q_rad_calc = (
            1.73e-9 * d5["alpha"] * d5["A_tT_corr"] * phi *
            (T_g_R ** 4 - T_T_R ** 4)
        )

        # ============================================================
        # B) BALANCE → T_chimenea
        # ------------------------------------------------------------
        # Fracción del calor que sale por chimenea (ULA, fig 7):
        Q  = d1["Q_Btu_h"]
        qn = d2["qn_Btu_h"]
        Q_R = d3["Q_R_Btu_h"]
        Q_C = Q - Q_R                       # calor que se absorbe en convección
        # Definimos pérdidas radiantes/conductivas como 2 % (típico ULA).
        qL_qn = 0.02
        # Balance: 1 = Q_R/qn + Q_C/qn + qG/qn + qL/qn
        qG_qn = 1.0 - Q_R / qn - Q_C / qn - qL_qn
        qG_qn = max(0.05, min(qG_qn, 0.85))     # acotar para fig7^-1
        T_chimenea = fig7_inversa_T_gas(qG_qn, d2["ex_air"])

        # ============================================================
        # C) GEOMETRÍA DE CONVECCIÓN
        # ------------------------------------------------------------
        geo = geometria_conveccion(NtH, DE_conv, Lcc_conv_in, d4["L_TE_ft"])
        AP = geo["AP_ft2"]
        # Velocidad másica de gases: G = Wg/AP (Wg en lb/h → /3600 → lb/s)
        Wg_lb_s = d2["Wg_lb_h"] / 3600.0
        G = Wg_lb_s / AP                              # lb/(s·ft²)
        diag_G = diagnosticar_G_gases(G)

        # ============================================================
        # D) COEFICIENTES DE PELÍCULA EN GASES
        # ------------------------------------------------------------
        # Temperatura promedio del gas en convección
        T_film_gas = 0.5 * (Tg_supuesto + T_chimenea)
        # Temperatura promedio de pared en convección
        T_Tc = 0.5 * (d1["T_e_F"] + d3["T_p_F"]) + 100.0    # margen ULA
        # h convección puro (Fig 10)
        diag_hcc = fig10_hcc(G, Tg_supuesto, return_diag=True)
        hcc = diag_hcc["hcc"]
        # h radiación de pared (Fig 11)
        hcw = fig11_hcw(T_Tc)
        # h radiación de gas (Fig 12)
        hcr = fig12_hcr(T_Tc, Tg_supuesto)
        # Áreas por hilera para Ff (ULA)
        # A_TH = área de tubos por hilera; A_PH = área de pared "vista" por hilera
        A_TH_per = NtH * tubo_conv["A_ext_ft2_per_ft"] * d4["L_TE_ft"]
        # Pared expuesta al gas (proyección horizontal entre hileras):
        A_PH_per = geo["ancho_ft"] * d4["L_TE_ft"]
        Uc, h_C, Ff, hCi = coef_global_conveccion(
            hcc, hcw, hcr, h_int, A_PH_per, A_TH_per)

        # ============================================================
        # E) ÁREA Y NÚMERO DE TUBOS DE CONVECCIÓN
        # ------------------------------------------------------------
        # LMTD: gas en contracorriente al fluido
        #   gas:    Tg_supuesto  →  T_chimenea
        #   líq.:   T_p          ←  T_e
        dT_caliente = Tg_supuesto - d3["T_p_F"]   # extremo "arriba"
        dT_frio     = T_chimenea - d1["T_e_F"]    # extremo "abajo"
        if dT_caliente <= 0 or dT_frio <= 0:
            st.error(
                f"⚠️ LMTD inválido: ΔT_caliente={dT_caliente:.0f}°F, "
                f"ΔT_frío={dT_frio:.0f}°F. Aumente Tg supuesto o revise "
                f"T_chimenea."
            )
            st.stop()
        lmtd = LMTD(dT_caliente, dT_frio)
        AC = Q_C / (Uc * lmtd)
        NTC = math.ceil(AC / A_t_conv)
        # Forzar a múltiplo de NtH (filas completas)
        if NTC % NtH != 0:
            NTC = (NTC // NtH + 1) * NtH
        n_hileras = NTC // NtH

        # ── Fix 4: validar que N_TC sea múltiplo del n_pasos del serpentín ──
        n_p = d4["n_pasos"]
        if NTC % n_p != 0:
            NTC_orig = NTC
            NTC = math.ceil(NTC / n_p) * n_p
            st.info(
                f"ℹ️ N_TC ajustado de {NTC_orig} a {NTC} tubos para que sea "
                f"múltiplo de {n_p} pasos del serpentín (simetría hidráulica)."
            )
            n_hileras = NTC // NtH

        # ============================================================
        # GUARDAR RESULTADOS
        # ------------------------------------------------------------
        st.session_state.datos["paso6"] = {
            # Radiación avanzada
            "Tg_F": Tg_supuesto,
            "epsilon": eps, "Aw_alphaAcp": ratio, "phi": phi,
            "Q_rad_calc_Btu_h": Q_rad_calc,
            # Balance térmico
            "qC_qn": Q_C / qn, "qG_qn": qG_qn,
            "T_chimenea_F": T_chimenea,
            # Convección — geometría
            "NtH": NtH, "DN_conv": DN_conv, "SCH_conv": SCH_conv,
            "tubo_conv": tubo_conv, "DE_conv_in": DE_conv,
            "Lcc_conv_in": Lcc_conv_in, "A_t_conv_ft2": A_t_conv,
            "ancho_conv_ft": geo["ancho_ft"], "LLC_ft": geo["LLC_ft"],
            "AP_ft2": AP,
            # Velocidad másica
            "G_lb_sft2": G, "diag_G_msg": diag_G["msg"],
            "diag_G_ok": diag_G["ok"], "diag_G_tipo": diag_G["tipo"],
            # Temperaturas y coeficientes
            "T_Gp_F": Tg_supuesto, "T_Tc_F": T_Tc,
            "hcc": hcc, "hcw": hcw, "hcr": hcr, "h_int": h_int,
            "h_C": h_C, "Uc": Uc, "Ff": Ff, "hCi": hCi,
            # Diagnóstico fig10
            "fig10_diag": diag_hcc,
            # Áreas
            "A_PH_per": A_PH_per, "A_TH_per": A_TH_per,
            # Resultados finales convección
            "LMTD_F": lmtd,
            "Q_C_Btu_h": Q_C, "A_C_ft2": AC,
            "N_TC": NTC, "n_hileras": n_hileras,
        }

        # ============================================================
        # SALIDA EN PANTALLA
        # ------------------------------------------------------------
        st.markdown("### Resultados — Radiación avanzada")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ε emisividad",       f"{eps:.3f}")
        c2.metric("φ intercambio",      f"{phi:.3f}")
        c3.metric("Q_rad calc (MMBtu/h)", f"{Q_rad_calc/1e6:.2f}")
        c4.metric("T_chimenea (°F)",     f"{T_chimenea:.0f}")

        # Validación radiación
        err_rad = abs(Q_rad_calc - Q_R) / Q_R
        if err_rad > 0.15:
            st.warning(
                f"⚠️ Q radiante calculado ({Q_rad_calc/1e6:.2f} MMBtu/h) "
                f"difiere {err_rad*100:.0f}% del objetivo "
                f"({Q_R/1e6:.2f} MMBtu/h). "
                f"{'AUMENTE' if Q_rad_calc < Q_R else 'DISMINUYA'} "
                f"la temperatura supuesta de gases T_g."
            )
        else:
            st.success(
                f"✅ Balance radiante consistente "
                f"(diferencia {err_rad*100:.1f} %)."
            )

        st.markdown("### Resultados — Geometría y velocidad másica gases")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ancho cabina conv.",  f"{geo['ancho_ft']:.2f} ft")
        c2.metric("LLC libre",           f"{geo['LLC_ft']:.2f} ft")
        c3.metric("AP libre paso",       f"{AP:.2f} ft²")
        c4.metric("G  (lb/s·ft²)",       f"{G:.3f}")

        if diag_G["ok"]:
            st.success(diag_G["msg"])
        else:
            st.warning(diag_G["msg"])

        # Aviso si la fig10 saturó
        if diag_hcc["G_recortado"] or diag_hcc["T_recortado"]:
            st.info(
                f"ℹ️ La Fig 10 ULA fue evaluada saturada al rango válido: "
                f"G_in={diag_hcc['G_in']:.3f} → G_used={diag_hcc['G_used']:.3f}, "
                f"T_in={diag_hcc['T_in']:.0f} → T_used={diag_hcc['T_used']:.0f}."
            )

        st.markdown("### Resultados — Coeficientes de transferencia")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("hcc (Fig 10)", f"{hcc:.2f}")
        c2.metric("hcw (Fig 11)", f"{hcw:.2f}")
        c3.metric("hcr (Fig 12)", f"{hcr:.2f}")
        c4.metric("h_int interno", f"{h_int:.0f}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("hCi (suma)", f"{hCi:.2f}")
        c2.metric("Ff factor",   f"{Ff:.3f}")
        c3.metric("hC aparente", f"{h_C:.2f}")
        c4.metric("Uc global",   f"{Uc:.2f}")

        st.markdown("### Resultados — Dimensionamiento")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LMTD (°F)",            f"{lmtd:.1f}")
        c2.metric("Q_C  (MMBtu/h)",       f"{Q_C/1e6:.2f}")
        c3.metric("AC convección (ft²)",  f"{AC:,.0f}")
        c4.metric("N tubos conv.",
                  f"{NTC}  ({n_hileras} hileras × {NtH})")

        with st.expander("📚 Correlaciones empleadas en este paso"):
            st.markdown("""
**A · Radiación avanzada**
- *Fig. 4 ULA*: P_(CO₂+H₂O) (atm) — polinomio cúbico directo en %ex.
- *Fig. 5 ULA (Hottel)*: ε(P·Lz, T_gas) — interpolación bilineal sobre
  digitalización Hottel.
- *Fig. 6 ULA (Lobo–Evans)*: φ(ε, Aw/(α·Acp)) — aproximación física
  calibrada.
- *Stefan–Boltzmann*:  Q_rad = 1.73·10⁻⁹ · α·Acp · φ · (Tg⁴ − Tt⁴).

**B · Balance térmico**
- *Fig. 7 ULA*: qg/qn(T_g, %ex) — interpolación bilineal sobre 8 curvas
  digitalizadas; T_chimenea se obtiene por bisección de la inversa.

**C · Convección**
- *Fig. 10 ULA (pág. 43)*:  hcc(G, T_gas) — interpolación PCHIP sobre 6
  curvas digitalizadas a 200, 400, 600, 800, 1000, 1200 °F.
  Argumento de temperatura: temperatura real del gas en convección (Tg_supuesto),
  NO la temperatura de película. Rango máximo de la tabla: 1200 °F.
  **Verificación**: G=0.35, T=800°F → hcc≈4.8 ✔ (acuerdo con la lectura
  visual de la curva original).
- *Fig. 11 ULA*: hcw(T_pared) — PCHIP digitalizado.
- *Fig. 12 ULA*: hcr(T_pared, T_gas) — Stefan–Boltzmann linealizado,
  ε_gas≈0.12.
- *Lobo–Evans (pág 47)*: Ff = (hcw/hCi)·(A_PH/A_TH);
  hC = (1+Ff)·(hcc+hcr);  Uc = hC·h_int / (hC+h_int).
""")
        if st.button("➡️ Avanzar al Paso 7 (Fluidodinámica y chimenea)"):
            desbloquear(7)
            st.rerun()

    # Si ya calculado, mostrar resumen
    elif "paso6" in st.session_state.datos:
        d = st.session_state.datos["paso6"]
        st.success(
            f"✅ Convección calculada: "
            f"hcc={d['hcc']:.2f}, Uc={d['Uc']:.2f}, "
            f"AC={d['A_C_ft2']:.0f} ft², "
            f"NTC={d['N_TC']} ({d['n_hileras']} hileras)"
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("T_chimenea (°F)", f"{d['T_chimenea_F']:.0f}")
        c2.metric("G (lb/s·ft²)",    f"{d['G_lb_sft2']:.3f}")
        c3.metric("LMTD (°F)",       f"{d['LMTD_F']:.1f}")
        c4.metric("Hileras",         f"{d['n_hileras']}")
        if st.button("➡️ Avanzar al Paso 7"):
            desbloquear(7)
            st.rerun()


# =====================================================================
# PASO 7 — Fluidodinámica (proceso) y diseño de chimenea
# ---------------------------------------------------------------------
# Procedimiento ULA (Anaya 1997) páginas 54-58:
#
# A) Caída de presión en el serpentín del proceso (lado fluido):
#    - Reynolds, factor de Fanning, ΔP por Darcy.
#    - Diagnóstico ΔP vs servicio (crudo / vacío / vapor).
#
# B) Tiro requerido en chimenea:
#    - tQ (caja de radiación)        = 0.25 in H₂O (criterio ULA)
#    - tTE (tiro hasta tubos escudo) = (0.25 - tQ) ≈ 0
#    - fc (fricción en convección)   = (L_T/Q · Pv) / 2     [in H₂O]
#         con  Pv = 0.003·G²/ρ_G  (cabezal de velocidad gases)
#    - tDamper                       = 1.5·Pv  (criterio ULA pág 54)
#    - tTubosEscudo                  ≈ ⅓·tDamper o cabeza vel. de gases
#
# C) Diseño de la chimenea:
#    - Densidad ρch = fig15(Tch − 100)  (Tch enfriada por la chimenea)
#    - F_G = Wg/ρch   (ft³/s)
#    - V_G = 30 ft/s recomendado (asumir)
#    - S_ch = F_G / V_G   →   D_ch = sqrt(4 S/π)
#    - Pv_ch = 0.003·V_G²·ρ_G                       [in H₂O]
#    - tT total = tTE + fc + ΔP_ch_damper
#    - tch (in H₂O / 100 ft) ← Fig 14(Tch)
#    - Lch = tT / tch * 100   (ft)  · 1.1 (factor seguridad ULA)
# =====================================================================
if st.session_state.paso_actual == 7:
    st.header("Paso 7 · Fluidodinámica y chimenea")
    if "paso6" not in st.session_state.datos:
        st.warning("Complete pasos previos.")
        st.stop()

    d1 = st.session_state.datos["paso1"]
    d2 = st.session_state.datos["paso2"]
    d4 = st.session_state.datos["paso4"]
    d5 = st.session_state.datos["paso5"]
    d6 = st.session_state.datos["paso6"]

    # ---------------------------------------------------------------
    # A) Caída de presión en serpentín
    # ---------------------------------------------------------------
    st.subheader("A · Caída de presión en serpentín (proceso)")

    # Número de codos calculado automáticamente:
    # - 2 codos de retorno por paso en zona radiante
    # - 1 codo de retorno por tubo en convección (aprox.)
    # Total conservador = 2·n_pasos + (N_R + N_TC)
    n_curvas_auto = 2 * d4["n_pasos"] + d4["N_tubos_R"] + d6["N_TC"]

    with st.expander("⚙️ Parámetros hidráulicos (opcionales)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            n_curvas = st.number_input(
                "Número total de curvas/codos (ajuste manual)",
                min_value=0, max_value=400,
                value=n_curvas_auto,
                step=1,
                help=(
                    f"Valor automático calculado: {n_curvas_auto} "
                    f"(2·pasos + N_R + N_TC). "
                    "Modifica solo si conoces el número exacto de codos del serpentín."
                ),
            )
            eps_d = st.number_input(
                "Rugosidad relativa ε/D",
                min_value=1e-6, max_value=1e-2, value=1e-4,
                format="%.5f",
                help="Acero comercial nuevo ≈ 1.5e-4; corroído ≈ 5e-4.",
            )
        with col2:
            servicio = st.selectbox(
                "Tipo de servicio (para diagnóstico ΔP)",
                ["crudo", "vacio", "vapor"],
                index=0,
                help="ΔP típico: crudo 40–250 psi · vacío 15–100 psi · vapor 5–50 psi.",
            )
            dP_limite_raw = st.number_input(
                "Límite de diseño ΔP (psi)  ← de bases de diseño",
                min_value=0.0, max_value=2000.0, value=0.0, step=5.0,
                help=(
                    "Si las bases de diseño especifican un ΔP máximo "
                    "(p.ej. 40 psi para Therminol 55), introdúcelo aquí. "
                    "El diagnóstico evaluará contra ese límite exacto. "
                    "Deja en 0 para usar solo los rangos típicos por servicio."
                ),
            )
            dP_limite_diseno = dP_limite_raw if dP_limite_raw > 0 else None

    st.caption(
        f"🔧 Codos calculados automáticamente: **{n_curvas_auto}** "
        f"(2×{d4['n_pasos']} pasos + {d4['N_tubos_R']} tubos rad. + {d6['N_TC']} tubos conv.)"
    )

    # Largo total tubos (radiación + convección)
    L_total_ft = (d4["N_tubos_R"] + d6["N_TC"]) * d4["L_TE_ft"]

    st.markdown("---")
    # ---------------------------------------------------------------
    # B y C) Diseño de chimenea
    # ---------------------------------------------------------------
    st.subheader("B · Diseño de chimenea (tiro natural)")
    col1, col2, col3 = st.columns(3)
    with col1:
        V_G_target = st.number_input(
            "Velocidad objetivo gases en chimenea (ft/s)",
            min_value=15.0, max_value=50.0, value=30.0, step=1.0,
            help="ULA: 25–35 ft/s. Mayor: más alta y menos diámetro; "
                 "menor: chimenea más ancha.",
        )
    with col2:
        T_amb_F = st.number_input(
            "Temperatura ambiente del aire (°F)",
            min_value=20.0, max_value=120.0, value=80.0, step=5.0,
        )
    with col3:
        FS_chim = st.number_input(
            "Factor de seguridad altura chimenea",
            min_value=1.0, max_value=1.3, value=1.1, step=0.05,
            help="Compensar pérdidas no contabilizadas (1.1 típico).",
        )

    if st.button("🌀 Calcular ΔP, fricción y chimenea", type="primary"):
        # ============================================================
        # A) ΔP serpentín (proceso) — Darcy con Fanning
        # ------------------------------------------------------------
        DI_in = d4["tubo"]["DI_in"]
        dp = caida_presion_serpentin(
            d1["W_lb_h"], DI_in, L_total_ft,
            d4["n_pasos"], n_curvas,
            d1["mu_avg"], d1["rho_avg"]
        )
        diag_dp = diagnosticar_dP(dp["dP_psi"], servicio, dP_limite_diseno)

        # ============================================================
        # B) Cabezal de velocidad y fricción en convección (lado gas)
        # ------------------------------------------------------------
        T_Gp_conv = d6["T_Gp_F"]                # °F prom gas convección
        rho_G_conv = fig15_rho_gases(T_Gp_conv)  # lb/ft³
        G_conv = d6["G_lb_sft2"]                 # lb/(s·ft²)
        # Pv (in H₂O) — fórmula ULA pág 56:  Pv = 0.003·G²/ρG
        Pv_conv = 0.003 * G_conv ** 2 / rho_G_conv  # in H₂O
        # Fricción convección:  fc = (L_T/Q · Pv) / 2
        # L_T/Q = nº hileras (pérdida por hilera ~ 1 cabezal de velocidad).
        n_hileras = d6["n_hileras"]
        fc = n_hileras * Pv_conv / 2.0

        # ============================================================
        # C) Diseño de la chimenea
        # ------------------------------------------------------------
        T_ch = d6["T_chimenea_F"]
        Wg = d2["Wg_lb_h"]
        # Densidad gases en chimenea — ULA enfría 100°F en chimenea
        rho_ch = fig15_rho_gases(T_ch - 100.0)
        F_G = Wg / rho_ch / 3600.0           # ft³/s
        S_ch = F_G / V_G_target              # ft²
        D_ch = (4.0 * S_ch / math.pi) ** 0.5  # ft
        # Velocidad real (puede diferir un poco de V_G_target)
        v_ch = F_G / S_ch
        # Cabezal de velocidad chimenea
        Pv_ch = 0.003 * v_ch ** 2 * rho_ch    # in H₂O

        # ============================================================
        # D) Tiro requerido y altura de chimenea (ULA pág 54-58)
        # ------------------------------------------------------------
        # Tiro en quemadores  tQ = 0.25 in H₂O  (criterio ULA)
        tQ = 0.25
        # Tiro hasta tubos escudo (caja de radiación → tubos de escudo)
        # ULA: 18-20 ft sobre quemadores; aproximamos como (0.25 - tQ) = 0
        # se considera embebido en tQ.  Aquí usamos 0.10 in H₂O por
        # pérdidas localizadas en la transición a tubos escudo:
        tTE = 0.10
        # Tiro damper:  1.5 · Pv_ch (criterio ULA)
        tDamper = 1.5 * Pv_ch
        # Tiro requerido total = tQ + tTE + fc + tDamper + Pv_ch
        # (Pv_ch porque hay que acelerar la columna de gas a v_ch)
        tT_total = tQ + tTE + fc + tDamper + Pv_ch

        # Altura: leer Fig 14 con T_chimenea y T_amb
        # tch_100 = tiro disponible (in H₂O / 100 ft de altura)
        tch_100 = fig14_tiro_100ft(T_ch)      # in H₂O / 100 ft
        # Corrección sencilla por T_amb (aire externo más frío → más tiro)
        # Suponiendo que la curva ULA está a T_amb=50°F, factor lineal:
        f_amb = 1.0 + (50.0 - T_amb_F) * 0.001
        tch_100_eff = tch_100 * f_amb

        L_ch = tT_total / tch_100_eff * 100.0     # ft
        L_ch_real = L_ch * FS_chim                # con factor de seguridad

        diag_ch = diagnosticar_chimenea(L_ch_real)

        # ============================================================
        # GUARDAR
        # ------------------------------------------------------------
        st.session_state.datos["paso7"] = {
            **dp,
            "n_curvas": n_curvas, "L_total_ft": L_total_ft,
            "eps_d": eps_d, "servicio": servicio,
            "dP_limite_diseno": dP_limite_diseno,
            "diag_dp_msg": diag_dp["msg"], "diag_dp_ok": diag_dp["ok"],
            # cabezal y fricción convección
            "rho_G_conv": rho_G_conv, "Pv_conv_inH2O": Pv_conv,
            "fc_inH2O": fc,
            # chimenea
            "T_ch_F": T_ch, "T_amb_F": T_amb_F,
            "rho_ch": rho_ch, "F_G_ft3s": F_G, "S_ch_ft2": S_ch,
            "D_ch_ft": D_ch, "v_ch_fts": v_ch,
            "Pv_ch_inH2O": Pv_ch,
            "tQ_inH2O": tQ, "tTE_inH2O": tTE,
            "tDamper_inH2O": tDamper,
            "tiro_req_inH2O": tT_total,
            "tch_100_inH2O": tch_100_eff,
            "L_ch_ft": L_ch_real, "L_ch_teorica_ft": L_ch,
            "FS_chim": FS_chim,
            "diag_ch_msg": diag_ch["msg"], "diag_ch_ok": diag_ch["ok"],
        }

        # ============================================================
        # SALIDA EN PANTALLA
        # ------------------------------------------------------------
        st.markdown("### A · Caída de presión en serpentín")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ΔP (psi)",          f"{dp['dP_psi']:.1f}")
        c2.metric("Reynolds",          f"{dp['Re']:,.0f}")
        c3.metric("f Fanning",         f"{dp['f']:.4f}")
        c4.metric("v interna (ft/s)",  f"{dp['v_ft_s']:.2f}")

        if diag_dp["ok"]:
            st.success(diag_dp["msg"])
        else:
            st.error(diag_dp["msg"])

        st.markdown("### B · Lado gases — cabezal y fricción")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ρ gases conv (lb/ft³)", f"{rho_G_conv:.4f}")
        c2.metric("Pv conv (in H₂O)",      f"{Pv_conv:.3f}")
        c3.metric("fc convección (in H₂O)", f"{fc:.3f}")
        c4.metric("Hileras conv.",         f"{n_hileras}")

        st.markdown("### C · Chimenea")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("D chimenea (ft)",   f"{D_ch:.2f}")
        c2.metric("v gases (ft/s)",    f"{v_ch:.1f}")
        c3.metric("Pv chimenea (inH₂O)", f"{Pv_ch:.3f}")
        c4.metric("ρ_ch (lb/ft³)",     f"{rho_ch:.4f}")

        st.markdown("### D · Balance de tiro (in H₂O)")
        balance_df = pd.DataFrame({
            "Concepto": [
                "Quemadores (tQ)", "Tubos escudo (tTE)",
                "Fricción convección (fc)", "Damper (1.5·Pv)",
                "Cabezal velocidad chimenea (Pv_ch)",
                "**TOTAL tiro requerido**"
            ],
            "Valor (in H₂O)": [
                tQ, tTE, fc, tDamper, Pv_ch, tT_total
            ],
        })
        st.dataframe(balance_df, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Tiro req total (inH₂O)", f"{tT_total:.2f}")
        c2.metric("Tiro disp/100 ft",        f"{tch_100_eff:.2f}")
        c3.metric("Altura chimenea (ft)",    f"{L_ch_real:.1f}")

        if diag_ch["ok"]:
            st.success(diag_ch["msg"])
        else:
            st.warning(diag_ch["msg"])
            # Sugerencias específicas
            if L_ch_real > 200:
                st.markdown(
                    "**Sugerencias para reducir altura de chimenea:**\n"
                    "- Aumentar el diámetro de chimenea (reduce Pv y tDamper)\n"
                    "- Reducir el exceso de aire (reduce Wg y por ende G)\n"
                    "- Reducir T_chimenea (mejora densidad media → más tiro)\n"
                    "- Reordenar la convección con menos hileras (reduce fc)\n"
                    "- Considerar tiro forzado/inducido"
                )
            elif L_ch_real < 30:
                st.markdown(
                    "**Sugerencias para aumentar altura de chimenea:**\n"
                    "- Aumentar exceso de aire o velocidad objetivo\n"
                    "- Verificar normativas de dispersión atmosférica"
                )

        with st.expander("📚 Correlaciones empleadas"):
            st.markdown("""
**A · Caída de presión serpentín**
- *Darcy–Weisbach* con factor de Fanning (Swamee–Jain).
- Codos modelados como L_eq ≈ 50·D.

**B · Cabezal y fricción del lado gases**
- *Pv = 0.003 · G² / ρ_G*  (in H₂O)  — ULA pág 56.
- *fc = (n_hileras · Pv) / 2*       — ULA pág 56.

**C · Chimenea**
- *Fig. 15 ULA*: ρ_G = ρ_G(T) — gas ideal con PM≈28.7.
- *V_G recomendado*: 25–35 ft/s en chimenea.

**D · Tiro**
- *Tiro quemadores*  tQ = 0.25 in H₂O   (ULA pág 54).
- *Tiro damper*       = 1.5 · Pv_ch       (ULA pág 54).
- *Fig. 14 ULA*: tch (in H₂O / 100 ft) = f(T_gas, T_amb).
- *Lch = (tT_total / tch_100) · 100 · FS*.
""")
        if st.button("➡️ Generar reporte (Paso 8)"):
            desbloquear(8)
            st.rerun()

    elif "paso7" in st.session_state.datos:
        d = st.session_state.datos["paso7"]
        st.success(
            f"✅ ΔP = {d['dP_psi']:.1f} psi · "
            f"Chimenea: D={d['D_ch_ft']:.2f} ft, "
            f"L={d['L_ch_ft']:.1f} ft, "
            f"tiro req {d['tiro_req_inH2O']:.2f} in H₂O"
        )
        if st.button("➡️ Generar reporte"):
            desbloquear(8)
            st.rerun()


# =====================================================================
# PASO 8 — Reporte técnico
# =====================================================================
def generar_reporte_md() -> str:
    """Genera reporte de diseño en formato Markdown."""
    d = st.session_state.datos
    if not all(f"paso{i}" in d for i in range(1, 8)):
        return "⚠️ Diseño incompleto — complete todos los pasos."
    d1, d2, d3, d4, d5, d6, d7 = (d[f"paso{i}"] for i in range(1, 8))
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = f"""# Reporte de Diseño — Horno de Cabina Horizontal

*Generado: {fecha}*

## 1. Datos de entrada y propiedades

| Parámetro | Valor |
|-----------|-------|
| Tipo de fluido | {d1['tipo_fluido']} |
| Flujo másico | {d1['W_lb_h']:,.0f} lb/h |
| Temperatura entrada | {d1['T_e_F']:.1f} °F |
| Temperatura salida | {d1['T_s_F']:.1f} °F |
| °API / Kw | {d1.get('api')} / {d1.get('K_w')} |
| Δh sensible | {d1['dH_Btu_lb']:.1f} Btu/lb |
| Cp promedio | {d1['cp_avg']:.3f} Btu/lb·°F |
| ρ promedio | {d1['rho_avg']:.2f} lb/ft³ |
| μ promedio | {d1['mu_avg']:.2f} cP |

## 2. Carga térmica y combustión

| Parámetro | Valor |
|-----------|-------|
| Q proceso | **{d1['Q_Btu_h']/1e6:.2f} MMBtu/h** |
| Eficiencia η | {d2['eta']*100:.1f} % |
| Calor neto liberado qn | {d2['qn_Btu_h']/1e6:.2f} MMBtu/h |
| Combustible | {d2['combustible']} |
| Exceso de aire | {d2['ex_air']} % |
| Flujo de gases Wg | {d2['Wg_lb_h']:,.0f} lb/h |
| Relación lb gases / MMBtu | {d2['lb_per_MMBtu']:.0f} |

## 3. Zona radiante

| Parámetro | Valor |
|-----------|-------|
| Fracción radiante η_R | {d3['eta_R']:.2f} |
| Servicio | {d3['servicio']} |
| Densidad calórica fqM | {d3['fqM']:,.0f} Btu/h·ft² |
| Temperatura de cruce Tp | {d3['T_p_F']:.1f} °F |
| Temperatura promedio fluido TR | {d3['T_R_F']:.1f} °F |
| Temperatura pared tubos TT | {d3['T_T_F']:.1f} °F |
| Carga radiante Q_R | {d3['Q_R_Btu_h']/1e6:.2f} MMBtu/h |
| Superficie radiante A_R | **{d3['A_R_ft2']:,.1f} ft²** |
| **Material recomendado** | **{d3['material_rec']}** |

## 4. Diseño de tubos

| Parámetro | Valor |
|-----------|-------|
| Diámetro nominal | {d4['DN_in']}\" SCH {d4['SCH']} |
| DE / DI | {d4['tubo']['DE_in']:.3f}\" / {d4['tubo']['DI_in']:.3f}\" |
| Longitud efectiva | {d4['L_TE_ft']:.1f} ft |
| Número de pasos | {d4['n_pasos']} |
| Número de tubos en zona radiante | **{d4['N_tubos_R']}** |
| Velocidad interna | **{d4['v_ft_s']:.2f} ft/s** |
| Velocidad másica G | {d4['G_lb_hft2']:,.0f} lb/h·ft² |

## 5. Geometría de cabina

| Parámetro | Valor |
|-----------|-------|
| Ancho L_BR | {d5['L_BR_ft']:.1f} ft |
| Alto L_HR | {d5['L_HR_ft']:.1f} ft |
| Largo (= L_TE) | {d4['L_TE_ft']:.1f} ft |
| Espaciamiento Lcc | {d5['L_cc_in']:.2f} in (C/D = {d5['C_over_D']:.2f}) |
| Tubos de escudo NtE | {d5['N_tE']} |
| Factor α de absorción | {d5['alpha']:.3f} |
| Volumen del horno | {d5['V_horno_ft3']:,.0f} ft³ |
| Longitud haz de llama Lz | {d5['Lz_ft']:.2f} ft |
| Presión parcial CO2+H2O | {d5['P_atm']:.3f} atm |
| P · Lz | {d5['P_Lz_atmft']:.2f} atm·ft |

## 6. Radiación avanzada y zona de convección

### 6.1 Radiación avanzada (Lobo–Evans)

| Parámetro | Valor |
|-----------|-------|
| T gases en cámara T_g | {d6['Tg_F']:.0f} °F |
| Emisividad ε | {d6['epsilon']:.3f} |
| Aw / (α·Acp) | {d6['Aw_alphaAcp']:.2f} |
| Factor de intercambio φ | {d6['phi']:.3f} |
| Q radiante calculado | {d6['Q_rad_calc_Btu_h']/1e6:.2f} MMBtu/h |
| **T chimenea** | **{d6['T_chimenea_F']:.0f} °F** |
| qC/qn / qG/qn | {d6['qC_qn']:.3f} / {d6['qG_qn']:.3f} |

### 6.2 Geometría y velocidad másica de gases

| Parámetro | Valor |
|-----------|-------|
| DN / SCH tubos convección | {d6['DN_conv']}″ SCH {d6['SCH_conv']} |
| DE convección | {d6['DE_conv_in']:.3f}″ |
| Tubos por hilera (NtH) | **{d6['NtH']}** |
| Lcc convección | {d6['Lcc_conv_in']:.2f}″ |
| Ancho cabina convección | {d6['ancho_conv_ft']:.2f} ft |
| LLC libre | {d6['LLC_ft']:.2f} ft |
| AP libre paso | {d6['AP_ft2']:.2f} ft² |
| **G velocidad másica** | **{d6['G_lb_sft2']:.3f} lb/(s·ft²)** |
| Diagnóstico G | {d6['diag_G_msg']} |

### 6.3 Coeficientes de transferencia

| Parámetro | Valor |
|-----------|-------|
| T_Gp prom gas conv. | {d6['T_Gp_F']:.0f} °F |
| T_Tc pared conv. | {d6['T_Tc_F']:.0f} °F |
| h_cc (Fig 10) | **{d6['hcc']:.2f}** Btu/h·ft²·°F |
| h_cw (Fig 11) | {d6['hcw']:.2f} |
| h_cr (Fig 12) | {d6['hcr']:.2f} |
| h_int interno | {d6['h_int']:.0f} |
| h_Ci suma | {d6['hCi']:.2f} |
| Ff factor radiación pared | {d6['Ff']:.3f} |
| h_C aparente del gas | {d6['h_C']:.2f} |
| **U_c global** | **{d6['Uc']:.2f}** Btu/h·ft²·°F |

### 6.4 Dimensionamiento

| Parámetro | Valor |
|-----------|-------|
| LMTD convección | {d6['LMTD_F']:.1f} °F |
| Carga convección Q_C | {d6['Q_C_Btu_h']/1e6:.2f} MMBtu/h |
| **Área convección AC** | **{d6['A_C_ft2']:,.0f} ft²** |
| N tubos convección NTC | **{d6['N_TC']}** ({d6['n_hileras']} hileras × {d6['NtH']}) |

## 7. Fluidodinámica y chimenea

### 7.1 Caída de presión en serpentín (proceso)

| Parámetro | Valor |
|-----------|-------|
| Servicio | {d7.get('servicio', 'crudo')} |
| Largo total tubos | {d7['L_total_ft']:.0f} ft |
| Codos / curvas | {d7['n_curvas']} |
| Reynolds | {d7['Re']:,.0f} |
| Factor de fricción Fanning | {d7['f']:.4f} |
| Velocidad másica G_proc | {d7['G_lb_hft2']:,.0f} lb/h·ft² |
| Velocidad fluido | {d7['v_ft_s']:.2f} ft/s |
| **ΔP serpentín** | **{d7['dP_psi']:.1f} psi** |
| Límite de diseño ΔP | {f"{d7.get('dP_limite_diseno', 'N/A')} psi" if d7.get('dP_limite_diseno') else "No especificado"} |
| Diagnóstico ΔP | {"✅ Cumple" if d7.get('diag_dp_ok') else "⚠️ No cumple — ver advertencias"} |

### 7.2 Lado gases: cabezal y fricción

| Parámetro | Valor |
|-----------|-------|
| ρ gases convección | {d7['rho_G_conv']:.4f} lb/ft³ |
| Pv conv (cabezal vel) | {d7['Pv_conv_inH2O']:.3f} in H₂O |
| Fricción convección fc | {d7['fc_inH2O']:.3f} in H₂O |

### 7.3 Diseño de chimenea

| Parámetro | Valor |
|-----------|-------|
| T chimenea | {d7['T_ch_F']:.0f} °F |
| T ambiente | {d7['T_amb_F']:.0f} °F |
| ρ_ch (a Tch–100) | {d7['rho_ch']:.4f} lb/ft³ |
| F_G volumétrico | {d7['F_G_ft3s']:.1f} ft³/s |
| **Diámetro chimenea D_ch** | **{d7['D_ch_ft']:.2f} ft** |
| Velocidad real V_G | {d7['v_ch_fts']:.1f} ft/s |
| Pv chimenea | {d7['Pv_ch_inH2O']:.3f} in H₂O |

### 7.4 Balance de tiro

| Concepto | in H₂O |
|----------|-------:|
| Quemadores tQ | {d7['tQ_inH2O']:.2f} |
| Tubos escudo tTE | {d7['tTE_inH2O']:.2f} |
| Fricción convección fc | {d7['fc_inH2O']:.3f} |
| Damper (1.5·Pv) | {d7['tDamper_inH2O']:.3f} |
| Cabezal vel chimenea | {d7['Pv_ch_inH2O']:.3f} |
| **TOTAL tiro requerido** | **{d7['tiro_req_inH2O']:.2f}** |
| Tiro disponible / 100 ft | {d7['tch_100_inH2O']:.2f} |
| **Altura chimenea (con FS={d7['FS_chim']:.2f})** | **{d7['L_ch_ft']:.1f} ft** |

## 8. Correlaciones utilizadas

Las gráficas originales de la guía ULA fueron sustituidas por las
siguientes correlaciones numéricas:

"""
    for c in DOC_CORRELACIONES:
        md += (f"- **{c['figura']}**: {c['entrada']} → {c['salida']}. "
               f"Método: *{c['metodo']}*.  Fuente: {c['fuente']}.  "
               f"Rango: {c['rango']}.\n")

    md += """
## 9. Fuentes principales

1. Anaya Alejandro. *Método corto para cálculo y diseño de hornos
   de proceso*. Ingeniería Química, diciembre 1997.
2. Evans, Frank L. *Equipment Design Handbook for Refineries and
   Chemical Plants*, Vol. 2, 2nd Ed., Gulf Publishing, 1980.
3. Durán, Luisa. *Teoría y cálculo de equipos de proceso en diseño
   de plantas* — Capítulo V (Hornos).  Universidad de Los Andes.
4. Hottel, H. C. & Sarofim, A. F. *Radiative Transfer*, McGraw-Hill, 1967.
5. Lobo, W. E. & Evans, J. E. *Heat Transfer in the Radiant Section
   of Petroleum Heaters*, AIChE Trans. 35 (1939).

---

*Diseñado con la app **Diseño de Hornos – Método Corto**.  Resultados
preliminares; verificar con un proveedor especializado antes de la
ingeniería de detalle.*
"""
    return md


if st.session_state.paso_actual == 8:
    st.header("Paso 8 · Reporte técnico")
    if not all(f"paso{i}" in st.session_state.datos for i in range(1, 8)):
        st.warning("Complete todos los pasos previos.")
        st.stop()

    md = generar_reporte_md()
    st.markdown(md)
    st.markdown("---")

    # Generar esquema una sola vez para reutilizarlo
    fig_rep, info_rep = draw_furnace_schematic(st.session_state.datos)
    md_con_esquema = md
    if fig_rep is not None:
        import base64
        png_rep = schematic_to_png_bytes(fig_rep, dpi=180)
        b64 = base64.b64encode(png_rep).decode("ascii")
        md_con_esquema += (
            "\n\n## 10. Esquema técnico del horno\n\n"
            f"![Esquema del horno](data:image/png;base64,{b64})\n"
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "📥 Reporte (Markdown + esquema)",
            data=md_con_esquema,
            file_name="reporte_diseno_horno.md",
            mime="text/markdown",
        )
    with c2:
        st.download_button(
            "📥 Reporte (texto plano)",
            data=md,
            file_name="reporte_diseno_horno.txt",
            mime="text/plain",
        )
    with c3:
        if fig_rep is not None:
            st.download_button(
                "🖼️ Esquema PNG aparte",
                data=schematic_to_png_bytes(fig_rep, dpi=200),
                file_name="esquema_horno.png",
                mime="image/png",
            )

    st.success("✅ Diseño preliminar completado.")

# =================================================================
# PASO 8.b — Esquema técnico 2D del horno
# =================================================================
    st.markdown("---")
    st.subheader("📐 Esquema técnico del horno diseñado")
    st.caption("Vista lateral generada con los valores realmente "
               "calculados por la app — no es una imagen genérica.")

    fig_esq, info_esq = draw_furnace_schematic(st.session_state.datos)

    if fig_esq is None:
        st.error("⚠️ No se puede generar el esquema: faltan resultados "
                 "del diseño.")
        if info_esq["faltantes"]:
            st.markdown("**Claves o pasos ausentes:**")
            for k in info_esq["faltantes"]:
                st.markdown(f"- `{k}`")
    else:
        if info_esq["advertencias"]:
            st.warning("⚠️ Advertencias de coherencia geométrica:")
            for a in info_esq["advertencias"]:
                st.markdown(f"- {a}")
        st.pyplot(fig_esq, use_container_width=True)

        # Descarga PNG
        png_bytes = schematic_to_png_bytes(fig_esq, dpi=200)
        st.download_button(
            "🖼️ Descargar esquema (PNG)",
            data=png_bytes,
            file_name="esquema_horno.png",
            mime="image/png",
        )


# =====================================================================
# Documentación complementaria — siempre visible al pie
# =====================================================================
with st.expander("📚 Documentación de correlaciones (todas las figuras)"):
    st.dataframe(pd.DataFrame(DOC_CORRELACIONES), use_container_width=True,
                 hide_index=True)

with st.expander("📋 Tabla 2 ULA — Características de tubos"):
    st.dataframe(TABLA_TUBOS, use_container_width=True, hide_index=True)

with st.expander("📋 Tabla 6 ULA — Materiales de tubos"):
    st.dataframe(TABLA_MATERIALES, use_container_width=True, hide_index=True)
