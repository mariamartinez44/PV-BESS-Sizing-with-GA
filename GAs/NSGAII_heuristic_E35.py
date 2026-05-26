#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSGA2 bi-objetivo — PV + BESS
Correcciones aplicadas:
  1. ILR = Ppvinst / PinverterPV  añadido en métricas, CSVs y resumen
  2. Gráfica Pareto: cada criterio se dibuja independientemente (sin fusión)
  3. Markers reducidos de s=250 → s=90
"""

from __future__ import annotations
from pathlib import Path
import re, csv, time
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.parallelization.starmap import StarmapParallelization


# ─── PARÁMETROS DEL SISTEMA ──────────────────────────────────────────────────

Plinst = 1000.0
Rmax   = 1000
Area   = 2.4
eta    = 0.2094

eff_c = 0.9624
eff_d = 0.9624
DoD   = 0.90

PmaxF = Plinst
er    = 1.1
BoP   = 0
Sc    = 1.2

OaMpv   = 12.5
OaMbess = 5.9

CAPEX_pv       = 388
CAPEX_BESS     = 185
CAPEX_inverter = 48

i  = 7.7 / 100
n  = 20
e  = 2.5 / 100

ir   = (i - e) / (1 + e)
crf  = (i * (i + 1)**n) / ((i + 1)**n - 1)
crfe = (1 + e) * (ir * (ir + 1)**n) / ((ir + 1)**n - 1)

kappa = [0, 28.79187*er, 15.07764*er, 6.55917*er,
             5.17209*er,  1.93281*er,  0.91609*er]

_SOC_LOW  = (1.0 - DoD) / 2.0
_SOC_HIGH = (1.0 - DoD) / 2.0 + DoD


# ─── LECTURA DE DATOS ────────────────────────────────────────────────────────

BASE_DIR           = Path(__file__).resolve().parent
DIRECTORIO_VENTANA = BASE_DIR / 'ventana_completa'


def read_inc(path):
    valores = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.lstrip().startswith('t'):
                continue
            partes = re.split(r'\s+', line.strip())
            if len(partes) >= 2:
                valores[int(partes[0][1:])] = float(partes[1])
    return valores


def cargar_datos_ventana(directorio):
    paths = {
        'lambda':  directorio / 'lambda_spain_localtime.inc',
        'psi':     directorio / 'psi.inc',
        'Ppvu':    directorio / 'PpvuMadridSarah20052023_localtime.inc',
        'Plu':     directorio / 'PluDataCenter.inc',
        'periodo': directorio / 'periodo.inc',
    }
    series = {k: read_inc(v) for k, v in paths.items()}
    T = range(1, 8761)
    data = {t: {
        'lambda': series['lambda'].get(t, 0.0),
        'psi':    series['psi'].get(t, 0.0),
        'Ppvu':   series['Ppvu'].get(t, 0.0),
        'Plu':    series['Plu'].get(t, 0.0),
    } for t in T}
    periodo = {t: int(series['periodo'].get(t, 6)) for t in T}
    return data, periodo, T


# ─── ESPACIO DE BÚSQUEDA ─────────────────────────────────────────────────────

PPV_MIN   = 0.1 * Plinst
PPV_MAX   = 8.0 * Plinst
C_MIN     = 0.5 * Plinst
C_MAX     = 15.0 * Plinst
RATIO_MIN = 0.1
RATIO_MAX = 1.0

XL_SIZING   = np.array([PPV_MIN, C_MIN,  RATIO_MIN, 0., 0.])
XU_SIZING   = np.array([PPV_MAX, C_MAX,  RATIO_MAX, 1., 1.])

XL_DISPATCH = np.array([1.0, 0.60,  0.0, 1.0, 0.0, 0.0, 3.0, 3.0])
XU_DISPATCH = np.array([2.0, 0.90, 15.0, 5.0, 1.0, 0.8, 6.0, 6.0])

XL = np.concatenate([XL_SIZING, XL_DISPATCH])
XU = np.concatenate([XU_SIZING, XU_DISPATCH])


# ─── DECODIFICADORES ─────────────────────────────────────────────────────────

def decodificar_sizing(x):
    C     = float(x[1])
    ratio = float(np.clip(x[2], RATIO_MIN, RATIO_MAX))
    return {
        'Ppvinst'      : float(x[0]),
        'C'            : C,
        'PinverterBESS': ratio * C,
        'PinverterPV'  : Plinst + PmaxF + ratio * C,
    }


def decodificar_dispatch(x):
    xd = np.clip(x, XL_DISPATCH, XU_DISPATCH)
    return {
        'n_ciclos'  : int(round(float(xd[0]))),
        'frac_C'    : float(xd[1]),
        'min_spread': float(xd[2]),
        'alpha'     : float(xd[3]),
        'beta'      : float(xd[4]),
        'gamma'     : float(xd[5]),
        'n_horas_c' : int(round(float(xd[6]))),
        'n_horas_d' : int(round(float(xd[7]))),
    }


def decodificar_completo(x):
    return decodificar_sizing(x[:5]), decodificar_dispatch(x[5:13])


# ─── ESTRATEGIA E3 ───────────────────────────────────────────────────────────

def planificar_dia_E3(horas_dia, data, params):
    n_ciclos   = params['n_ciclos']
    min_spread = params['min_spread']
    n_horas_c  = params['n_horas_c']
    n_horas_d  = params['n_horas_d']

    lam    = {t: data[t]['lambda'] for t in horas_dia}
    spread = max(lam.values()) - min(lam.values())

    if spread < min_spread:
        return {}

    horas_ord = sorted(horas_dia, key=lambda t: lam[t])

    n_c = min(n_horas_c * n_ciclos, len(horas_dia) // 2)
    n_d = min(n_horas_d * n_ciclos, len(horas_dia) // 2)

    ventana_c = set(horas_ord[:n_c])
    ventana_d = set(horas_ord[-n_d:]) - ventana_c

    ventana: dict[int, str] = {}
    for t in ventana_c:
        ventana[t] = 'carga'

    horas_c_sorted = sorted(ventana_c)
    for t in sorted(ventana_d, key=lambda h: -lam[h]):
        if any(tc < t for tc in horas_c_sorted):
            ventana[t] = 'descarga'

    return ventana


# ─── ESTRATEGIA E5 ───────────────────────────────────────────────────────────

def despachar_hora_E3E5(diseno, SOC, data_t, params, ventana_t, lam_med_dia):
    C    = diseno['C']
    Pinv = diseno['PinverterBESS']

    SOCmin = _SOC_LOW  * C
    SOCmax = _SOC_HIGH * C

    lam_t   = data_t['lambda']
    PL_t    = Plinst * data_t['Plu']
    Ppvmx_t = diseno['Ppvinst'] * data_t['Ppvu']
    Ppv_t   = Ppvmx_t
    pv_norm = data_t['Ppvu']

    soc_norm = float(np.clip(SOC / C, 0.0, 1.0)) if C > 1e-3 else 0.5

    lam_ref = max(abs(lam_med_dia), 1.0)
    desv    = (lam_t - lam_med_dia) / lam_ref

    frac_desc = float(1.0 / (1.0 + np.exp(-params['alpha'] * desv)))
    frac_carg = float(1.0 / (1.0 + np.exp( params['alpha'] * desv)))

    soc_corr_d = float(np.clip((soc_norm - _SOC_LOW)  / (0.5 - _SOC_LOW),  0.0, 1.0))
    soc_corr_c = float(np.clip((_SOC_HIGH - soc_norm) / (_SOC_HIGH - 0.5), 0.0, 1.0))

    pv_bonus = params['beta'] * pv_norm

    Pc_t = 0.0
    Pd_t = 0.0

    if ventana_t == 'descarga':
        frac = frac_desc * soc_corr_d * params['gamma'] \
             + frac_desc * (1.0 - params['gamma'])
        frac = float(np.clip(frac, 0.0, 1.0))
        disp = max(SOC - SOCmin, 0.0)
        Pd_t = min(Pinv * frac, disp * eff_d)

    elif ventana_t == 'carga':
        frac_base = frac_carg * (1.0 + pv_bonus)
        frac = frac_base * soc_corr_c * params['gamma'] \
             + frac_base * (1.0 - params['gamma'])
        frac = float(np.clip(frac, 0.0, 1.0))
        espacio = max(SOCmax - SOC, 0.0)
        Pc_t = min(Pinv * frac, espacio / eff_c)

    if Pd_t < 1e-4 and Pc_t < Pinv - 1e-3:
        excedente_pv = max(Ppv_t - PL_t, 0.0)
        espacio_rem  = max(SOCmax - SOC - Pc_t * eff_c, 0.0)
        if excedente_pv > 1e-3 and espacio_rem > 1e-3:
            Pc_pv = min(Pinv - Pc_t, espacio_rem / eff_c, excedente_pv)
            Pc_t  = min(Pc_t + max(Pc_pv, 0.0), Pinv)

    flujo_neto = Ppv_t + Pd_t - Pc_t - PL_t
    if flujo_neto >= 0.0:
        Ps_t = min(flujo_neto, PmaxF)
        Pb_t = 0.0
    else:
        Ps_t = 0.0
        Pb_t = min(-flujo_neto, PmaxF)

    SOC_new = float(np.clip(
        SOC + Pc_t * eff_c - Pd_t / max(eff_d, 1e-9),
        SOCmin, SOCmax,
    ))

    return {
        'Ppv'    : Ppv_t,
        'Ppvmx'  : Ppvmx_t,
        'Pc'     : Pc_t,
        'Pd'     : Pd_t,
        'Pb'     : Pb_t,
        'Ps'     : Ps_t,
        'SOC_new': SOC_new,
        'w1'     : 1 if Pc_t > 1e-4 else 0,
        'w3'     : 1 if Pb_t > 1e-4 else 0,
        'score'  : frac_desc,
    }


# ─── SIMULACIÓN ANUAL 8760h ──────────────────────────────────────────────────

def simular_E3E5(diseno, params, data, T_list, periodo=None):
    C      = diseno['C']
    SOCmin = _SOC_LOW  * C
    SOCmax = _SOC_HIGH * C
    SOC    = (SOCmin + SOCmax) / 2.0
    SOC0   = SOC

    res = {k: {} for k in ('Ppv', 'Ppvmx', 'Pc', 'Pd', 'Pb', 'Ps',
                            'SOC', 'w1', 'w3', 'score')}

    n_dias = len(T_list) // 24
    idx    = 0

    for _ in range(n_dias):
        horas_dia = T_list[idx: idx + 24]
        idx      += 24

        lam_dia_vals = [data[t]['lambda'] for t in horas_dia]
        lam_med      = float(np.median(lam_dia_vals))

        ventana = planificar_dia_E3(horas_dia, data, params)

        for t in horas_dia:
            a = despachar_hora_E3E5(
                diseno, SOC, data[t], params,
                ventana_t=ventana.get(t),
                lam_med_dia=lam_med,
            )
            for k in ('Ppv', 'Ppvmx', 'Pc', 'Pd', 'Pb', 'Ps', 'w1', 'w3', 'score'):
                res[k][t] = a[k]
            res['SOC'][t] = a['SOC_new']
            SOC = a['SOC_new']

    for t in T_list[idx:]:
        a = despachar_hora_E3E5(diseno, SOC, data[t], params,
                                ventana_t=None, lam_med_dia=50.0)
        for k in ('Ppv', 'Ppvmx', 'Pc', 'Pd', 'Pb', 'Ps', 'w1', 'w3', 'score'):
            res[k][t] = a[k]
        res['SOC'][t] = a['SOC_new']
        SOC = a['SOC_new']

    res['SOC0'] = SOC0
    return res


# ─── MÉTRICAS (con ILR) ──────────────────────────────────────────────────────

def calcular_metricas(diseno, res, data, periodo, T_list):
    Pbmax = {p: max((res['Pb'].get(t, 0.0) for t in T_list
                 if (periodo.get(t, 6) if periodo else 6) == p), default=0.0)
         for p in range(1, 7)}

    Pbmax[6] = PmaxF
    for p in range(5, 0, -1):
        Pbmax[p] = max(Pbmax[p], Pbmax[p + 1])

    Es  = er * sum(data[t]['lambda'] * res['Ps'][t] for t in T_list)
    Eb  = er * sum((data[t]['lambda'] + data[t]['psi']) * res['Pb'][t] for t in T_list)
    Eb0 = er * sum((data[t]['lambda'] + data[t]['psi']) * Plinst * data[t]['Plu']
                   for t in T_list)
    Wl  = sum(Plinst * data[t]['Plu'] for t in T_list)

    CapacityP  = sum(kappa[p] * Pbmax[p] for p in range(1, 7))
    CapacityP0 = sum(kappa[p] * PmaxF    for p in range(1, 7))
    OaM        = OaMpv * diseno['Ppvinst'] + OaMbess * diseno['C']

    OPEX  = CapacityP + Eb + OaM
    OPEX0 = CapacityP0 + Eb0
    CF    = Es + OPEX0 - OPEX

    Wc = sum(res['Pc'][t] for t in T_list)
    Wd = sum(res['Pd'][t] for t in T_list)

    # ── FIX 1: ILR = Ppvinst / PinverterPV  (igual que MILP) ────────────────
    ILR = (diseno['Ppvinst'] / diseno['PinverterPV']
           if diseno['PinverterPV'] > 1e-3 else 0.0)

    return {
        'Es': Es, 'Eb': Eb, 'Eb0': Eb0,
        'CapacityP': CapacityP, 'CapacityP0': CapacityP0,
        'OaM': OaM, 'OPEX': OPEX, 'OPEX0': OPEX0,
        'CashFlow': CF, 'Savings': OPEX0 - OPEX, 'Benefit': Es - OPEX,
        'Wl': Wl, 'ENS': 0.0,
        'Wb': sum(res['Pb'][t] for t in T_list),
        'Ws': sum(res['Ps'][t] for t in T_list),
        'Wc': Wc, 'Wd': Wd,
        'wpv'           : sum(res['Ppv'][t]   for t in T_list),
        'wpvmx'         : sum(res['Ppvmx'][t] for t in T_list),
        'wcurtail'      : sum(res['Ppvmx'][t] for t in T_list)
                          - sum(res['Ppv'][t] for t in T_list),
        'ciclos_anuales': Wc / diseno['C'] if diseno['C'] > 1e-3 else 0.0,
        'ILR'           : ILR,   # ← NUEVO
    }


def calcular_capex(diseno):
    return BoP + Sc * (CAPEX_pv       * diseno['Ppvinst']
                       + CAPEX_BESS   * diseno['C']
                       + CAPEX_inverter * (diseno['PinverterBESS'] + diseno['PinverterPV']))


def evaluar_8760(x, data_8760, periodo_8760, T_8760):
    T_list         = list(T_8760)
    diseno, params = decodificar_completo(x)
    res            = simular_E3E5(diseno, params, data_8760, T_list, periodo_8760)
    m              = calcular_metricas(diseno, res, data_8760, periodo_8760, T_list)
    inv            = calcular_capex(diseno)
    return m['CashFlow'], res, m, inv


# ─── EXPORTACIÓN CSV (con ILR) ───────────────────────────────────────────────

def guardar_csv_horario(fname, diseno, res, m, data_8760, T_8760, CF, npv_val):
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Hour','lambda','psi','Plu','PL','Ppvmx','Ppv',
                    'Pb','Ps','Pc','Pd','SOC','w1','w3','score',
                    'Benefit','CashFlow','npv'])
        for t in T_8760:
            w.writerow([t,
                round(data_8760[t]['lambda'],6), round(data_8760[t]['psi'],6),
                round(data_8760[t]['Plu'],6),
                round(Plinst*data_8760[t]['Plu'],4),
                round(res['Ppvmx'][t],4), round(res['Ppv'][t],4),
                round(res['Pb'][t],4),    round(res['Ps'][t],4),
                round(res['Pc'][t],4),    round(res['Pd'][t],4),
                round(res['SOC'][t],4),   res['w1'][t], res['w3'][t],
                round(res['score'][t],6),
                round(m['Benefit'],4), round(CF,4), round(npv_val,4)])
    print(f"CSV horario: {fname}")


def guardar_csv_resumen(fname, diseno, params, res, m, inv, CF, periodo=None):
    opex_val  = CF / crfe
    npv_val   = opex_val - inv
    TIR       = npf.irr([-inv] + [CF]*n) * 100 if inv > 0 else 0.0
    BCratio   = npf.pv(rate=i, nper=n, pmt=-CF, fv=0) / inv if inv > 0 else 0.0
    NPERaprox = inv / CF if CF > 0 else 0.0
    try:
        NPER = np.log(CF / (CF + i*(-inv))) / np.log(1+i) if (CF + i*(-inv)) > 0 else 0.0
    except Exception:
        NPER = 0.0
    OPEXgross = OaMpv*diseno['Ppvinst'] + OaMbess*diseno['C']
    Wl_val    = m['Wl']
    LCOEgross = 1000*(inv + OPEXgross/crfe) / (Wl_val/crf) if Wl_val > 0 else 0.0
    LCOEnet   = 1000*(inv + (m['OPEX']-m['OPEX0']-m['Es'])/crfe) / (Wl_val/crf) \
                if Wl_val > 0 else 0.0
    Crate = diseno['PinverterBESS'] / diseno['C'] if diseno['C'] > 0 else 0.0
    nx    = 1000 * diseno['Ppvinst'] / (Rmax * eta * Area)
    gap   = (npv_val - 3_623_548) / 3_623_548 * 100

    # ── FIX 1: ILR en el CSV resumen ─────────────────────────────────────────
    ILR = m.get('ILR', 0.0)

    T_keys   = sorted(res['Pb'].keys())
    Pbmax_ph = {}
    for p in range(1, 7):
        vals = [res['Pb'][t] for t in T_keys
                if (periodo.get(t, 6) if periodo else 6) == p]
        Pbmax_ph[p] = max(vals) if vals else 0.0

    headers = ['Ppvinst_kW','C_kWh','PinverterBESS_kW','PinverterPV_kW',
               'nx','Crate','ILR',                         # ← ILR añadido
               'PbmaxP1','PbmaxP2','PbmaxP3','PbmaxP4','PbmaxP5','PbmaxP6',
               'n_ciclos','frac_C','min_spread','alpha','beta','gamma',
               'n_horas_c','n_horas_d','SOC0_kWh',
               'Wl','Wb','Ws','Wc','Wd','wpvmx','wpv','wcurtail','ciclos_anuales',
               'Es','Eb','Eb0','CapacityP','CapacityP0','OaM',
               'OPEX','OPEX0','Savings','Benefit','CashFlow',
               'Investment','PVcost','BESSbatCost','BESSinvCost','SoftCosts',
               'OPEX_VP','npv','TIR','Payback','PaybackSimple',
               'LCOEnet','LCOEgross','BCR','gap_vs_MILP_pct']

    values = [
        round(diseno['Ppvinst'],2), round(diseno['C'],2),
        round(diseno['PinverterBESS'],2), round(diseno['PinverterPV'],2),
        round(nx,2), round(Crate,4), round(ILR,4),           # ← ILR añadido
        round(Pbmax_ph[1],2), round(Pbmax_ph[2],2),
        round(Pbmax_ph[3],2), round(Pbmax_ph[4],2),
        round(Pbmax_ph[5],2), round(Pbmax_ph[6],2),
        params['n_ciclos'], round(params['frac_C'],3),
        round(params['min_spread'],2), round(params['alpha'],4),
        round(params['beta'],4),       round(params['gamma'],4),
        params['n_horas_c'],           params['n_horas_d'],
        round(res['SOC0'],2),
        round(m['Wl'],2),     round(m['Wb'],2),       round(m['Ws'],2),
        round(m['Wc'],2),     round(m['Wd'],2),
        round(m['wpvmx'],2),  round(m['wpv'],2),       round(m['wcurtail'],2),
        round(m['ciclos_anuales'],1),
        round(m['Es'],2),     round(m['Eb'],2),        round(m['Eb0'],2),
        round(m['CapacityP'],2), round(m['CapacityP0'],2), round(m['OaM'],2),
        round(m['OPEX'],2),   round(m['OPEX0'],2),
        round(m['Savings'],2),round(m['Benefit'],2),   round(CF,2),
        round(inv,2),
        round(CAPEX_pv       * diseno['Ppvinst'],2),
        round(CAPEX_BESS     * diseno['C'],2),
        round(CAPEX_inverter * diseno['PinverterBESS'],2),
        round(inv*(Sc-1)/Sc,2),
        round(opex_val,2), round(npv_val,2),
        round(TIR,4),      round(NPER,4),    round(NPERaprox,4),
        round(LCOEnet,4),  round(LCOEgross,4), round(BCratio,4), round(gap,4),
    ]

    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerow(values)
    print(f"CSV resumen: {fname}")


# ─── PROBLEMA PYMOO ──────────────────────────────────────────────────────────

class ProblemaNSGA2_E3E5(ElementwiseProblem):
    def __init__(self, data_8760, periodo_8760, T_8760, **kwargs):
        super().__init__(
            n_var=13, n_obj=2,
            xl=XL, xu=XU, **kwargs)
        self.data_8760    = data_8760
        self.periodo_8760 = periodo_8760
        self.T_8760       = T_8760

    def _evaluate(self, x, out):
        CF, _, m, inv = evaluar_8760(x, self.data_8760,
                                     self.periodo_8760, self.T_8760)
        out['F'] = [-CF / crfe, inv]


# ─── CALLBACK ────────────────────────────────────────────────────────────────

class RegistrarConvergencia(Callback):
    def __init__(self, period=20, tol=1.0):
        super().__init__()
        self.historial_opex  = []
        self.historial_capex = []
        self.historial_gen   = []
        self.t0          = time.time()
        self.period      = period
        self.tol         = tol
        self._sin_mejora = 0
        self._mejor_opex = -np.inf

    def notify(self, algorithm):
        gen   = algorithm.n_gen
        F     = algorithm.opt.get('F')
        opex  = -float(np.min(F[:, 0]))
        capex =  float(np.min(F[:, 1]))
        self.historial_opex.append(opex)
        self.historial_capex.append(capex)
        self.historial_gen.append(gen)

        if opex - self._mejor_opex > self.tol:
            self._sin_mejora = 0
            self._mejor_opex = opex
        else:
            self._sin_mejora += 1

        print(f"Gen {gen:4d} | OPEX: {opex:>14,.0f} USD | "
              f"CAPEX mín: {capex/1e6:>6.2f} MUSD | "
              f"sin mejora: {self._sin_mejora:3d}/{self.period} | "
              f"t: {time.time()-self.t0:.1f}s")

        if self._sin_mejora >= self.period:
            print(f"\nParada temprana: {self.period} gen sin mejora")
            algorithm.termination.force_termination = True


# ─── IMPRESIÓN DE RESULTADOS (con ILR) ───────────────────────────────────────

def imprimir_resultado(idx, label, pareto_X, real_res, real_m, real_capex, real_cf):
    diseno, params = decodificar_completo(pareto_X[idx])
    m   = real_m[idx]
    CF  = real_cf[idx]
    inv = real_capex[idx]
    npv = CF/crfe - inv
    TIR = npf.irr([-inv] + [CF]*n) * 100 if inv > 0 else 0
    BCr = npf.pv(rate=i, nper=n, pmt=-CF, fv=0) / inv if inv > 0 else 0
    try:
        NPER = np.log(CF/(CF+i*(-inv))) / np.log(1+i) if (CF+i*(-inv)) > 0 else 0
    except Exception:
        NPER = 0
    gap = (npv - 3_623_548) / 3_623_548 * 100
    ILR = m.get('ILR', 0.0)   # ← FIX 1

    print(f"\n{'='*62}\nRESULTADOS — {label}\n{'='*62}")
    print(f"  C={diseno['C']:,.0f} kWh  Pinv={diseno['PinverterBESS']:,.0f} kW"
          f"  PV={diseno['Ppvinst']:,.0f} kW")
    print(f"  PinverterPV={diseno['PinverterPV']:,.0f} kW  ILR={ILR:.4f}")  # ← ILR
    print(f"  n_ciclos={params['n_ciclos']}  frac_C={params['frac_C']:.2f}"
          f"  min_spread={params['min_spread']:.0f}")
    print(f"  alpha={params['alpha']:.3f}  beta={params['beta']:.3f}"
          f"  gamma={params['gamma']:.3f}")
    print(f"  n_horas_c={params['n_horas_c']}  n_horas_d={params['n_horas_d']}")
    print(f"  Ciclos/año={m['ciclos_anuales']:.0f}"
          f"  Wc={m['Wc']:,.0f}  Wd={m['Wd']:,.0f} kWh")
    print(f"  NPV={npv:,.0f}  TIR={TIR:.2f}%  PB={NPER:.1f}yr  BCR={BCr:.3f}")
    print(f"  Gap vs MILP: {gap:+.2f}%")
    return npv, CF


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    from adjustText import adjust_text

    print("Cargando datos 8760h...")
    data_8760, periodo_8760, T_8760 = cargar_datos_ventana(DIRECTORIO_VENTANA)
    T_list = list(T_8760)

    diseno_ref = {
        'Ppvinst': 5991.87, 'C': 9568.53, 'PinverterBESS': 1999.61,
        'PinverterPV': 3999.61,
    }
    params_ref = {
        'n_ciclos': 1, 'frac_C': 0.80, 'min_spread': 15.0,
        'alpha': 1.5, 'beta': 0.5, 'gamma': 0.4,
        'n_horas_c': 4, 'n_horas_d': 4,
    }

    print("\nMidiendo tiempo por evaluación...")
    t0      = time.time()
    res_ref = simular_E3E5(diseno_ref, params_ref, data_8760, T_list, periodo_8760)
    m_ref   = calcular_metricas(diseno_ref, res_ref, data_8760, periodo_8760, T_list)
    t_eval  = time.time() - t0
    inv_ref = calcular_capex(diseno_ref)
    CF_ref  = m_ref['CashFlow']

    POP_SIZE  = 80
    N_MAX_GEN = 40
    PERIOD    = 20
    N_CORES   = 4

    t_est = t_eval * POP_SIZE * N_MAX_GEN / N_CORES / 60
    print(f"\nTiempo estimado: {t_est:.1f} min"
          f"  (pop={POP_SIZE} gen={N_MAX_GEN} cores={N_CORES})")

    pool     = Pool(N_CORES)
    runner   = StarmapParallelization(pool.starmap)
    problema = ProblemaNSGA2_E3E5(data_8760, periodo_8760, T_8760,
                                   elementwise_runner=runner)

    algoritmo = NSGA2(
        pop_size=POP_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.8, eta=10),
        mutation=PM(eta=5),
        eliminate_duplicates=True,
    )
    criterio = get_termination('n_gen', N_MAX_GEN)
    callback = RegistrarConvergencia(period=PERIOD, tol=1.0)

    print("\nIniciando NSGA2 E3+E5 v4...")
    t0 = time.time()
    resultado = minimize(problema, algoritmo, termination=criterio,
                         seed=94652, verbose=False, callback=callback)
    t_ga = time.time() - t0
    pool.close(); pool.join()
    print(f"\nNSGA2 completado en {t_ga/60:.1f} min"
          f" | {resultado.algorithm.n_gen} gen")

    # ── Post-proceso: re-evaluar frente de Pareto ────────────────────────────
    pareto_X     = resultado.X
    n_pareto     = len(pareto_X)
    real_opex    = np.full(n_pareto, np.nan)
    real_capex_a = np.full(n_pareto, np.nan)
    real_npv     = np.full(n_pareto, np.nan)
    real_cf      = np.full(n_pareto, np.nan)
    real_res     = [None] * n_pareto
    real_m       = [None] * n_pareto

    print(f"\nFrente de Pareto — {n_pareto} soluciones. Extrayendo métricas...")
    for idx in range(n_pareto):
        CF_i, res_i, m_i, inv_i = evaluar_8760(
            pareto_X[idx], data_8760, periodo_8760, T_8760)
        real_capex_a[idx] = inv_i
        real_cf[idx]      = CF_i
        real_opex[idx]    = CF_i / crfe
        real_npv[idx]     = real_opex[idx] - inv_i
        real_res[idx]     = res_i
        real_m[idx]       = m_i
        _, pi = decodificar_completo(pareto_X[idx])
        ILR_i = m_i.get('ILR', 0.0)
        print(f"  [{idx+1:>3}/{n_pareto}] NPV={real_npv[idx]:>10,.0f} | "
              f"CAPEX={real_capex_a[idx]/1e6:.2f}M | "
              f"ILR={ILR_i:.3f} | "                              # ← ILR en log
              f"ciclos={m_i['ciclos_anuales']:.0f} | "
              f"n_c={pi['n_ciclos']} nh_c={pi['n_horas_c']}"
              f" nh_d={pi['n_horas_d']}")

    # ── Métricas financieras por solución ────────────────────────────────────
    idx_v   = np.arange(n_pareto)
    npv_v   = real_npv[idx_v]
    capex_v = real_capex_a[idx_v]
    bcr_v   = (npv_v + capex_v) / np.where(capex_v > 0, capex_v, 1e-9)

    real_tir  = np.full(n_pareto, np.nan)
    real_lcoe = np.full(n_pareto, np.nan)
    real_pb   = np.full(n_pareto, np.nan)
    real_bcr  = np.full(n_pareto, np.nan)

    for idx in range(n_pareto):
        CF_i  = real_cf[idx]
        inv_i = real_capex_a[idx]
        m_i   = real_m[idx]

        if inv_i > 0 and CF_i > 0:
            try:
                real_tir[idx] = npf.irr([-inv_i] + [CF_i] * n) * 100
            except Exception:
                real_tir[idx] = float('nan')
            denom = CF_i + i * (-inv_i)
            real_pb[idx] = (np.log(CF_i / denom) / np.log(1 + i)
                            if denom > 0 else float('nan'))
            real_bcr[idx] = npf.pv(rate=i, nper=n, pmt=-CF_i, fv=0) / inv_i

        Wl_i = m_i['Wl']
        if Wl_i > 0:
            real_lcoe[idx] = (1000 * (inv_i + (m_i['OPEX'] - m_i['OPEX0'] - m_i['Es']) / crfe)
                              / (Wl_i / crf))

    idx_best_npv  = idx_v[int(np.nanargmax(npv_v))]
    idx_best_bcr  = idx_v[int(np.nanargmax(bcr_v))]
    idx_best_tir  = idx_v[int(np.nanargmax(
                        np.where(np.isnan(real_tir), -np.inf, real_tir)))]
    lcoe_valid    = np.where(~np.isnan(real_lcoe), real_lcoe, np.inf)
    idx_best_lcoe = idx_v[int(np.argmin(lcoe_valid))]

    imprimir_resultado(idx_best_npv, 'NSGA2 — Máx NPV',
                       pareto_X, real_res, real_m, real_capex_a, real_cf)

    d_best   = decodificar_completo(pareto_X[idx_best_npv])[0]
    p_best   = decodificar_completo(pareto_X[idx_best_npv])[1]
    CF_best  = real_cf[idx_best_npv]
    inv_best = real_capex_a[idx_best_npv]
    npv_best = CF_best / crfe - inv_best

    guardar_csv_horario('solution_nsga2_E3E5_hourly.csv',
                        d_best, real_res[idx_best_npv], real_m[idx_best_npv],
                        data_8760, T_8760, CF_best, npv_best)
    guardar_csv_resumen('solution_nsga2_E3E5_summary.csv',
                        d_best, p_best, real_res[idx_best_npv],
                        real_m[idx_best_npv], inv_best, CF_best,
                        periodo=periodo_8760)

    # ── CSV Pareto completo (con ILR) ────────────────────────────────────────
    with open('pareto_nsga2_E3E5.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'OPEX_VP', 'CAPEX', 'NPV', 'CF',
                    'TIR_pct', 'BCR', 'Payback_yr', 'LCOEnet_USDMWh',
                    'ILR',                                    # ← ILR añadido
                    'ciclos_anuales',
                    'Ppvinst_kW', 'C_kWh', 'PinverterBESS_kW', 'PinverterPV_kW',
                    'n_ciclos', 'frac_C', 'min_spread',
                    'alpha', 'beta', 'gamma', 'n_horas_c', 'n_horas_d'])
        for idx in range(n_pareto):
            di, pi = decodificar_completo(pareto_X[idx])
            ILR_row = real_m[idx].get('ILR', 0.0)
            w.writerow([idx,
                        round(real_opex[idx],    2),
                        round(real_capex_a[idx], 2),
                        round(real_npv[idx],     2),
                        round(real_cf[idx],      2),
                        round(float(real_tir[idx]),  4) if not np.isnan(real_tir[idx])  else '',
                        round(float(real_bcr[idx]),  4) if not np.isnan(real_bcr[idx])  else '',
                        round(float(real_pb[idx]),   4) if not np.isnan(real_pb[idx])   else '',
                        round(float(real_lcoe[idx]), 4) if not np.isnan(real_lcoe[idx]) else '',
                        round(ILR_row, 4),                   # ← ILR añadido
                        round(real_m[idx]['ciclos_anuales'], 1),
                        round(di['Ppvinst'],       2),
                        round(di['C'],             2),
                        round(di['PinverterBESS'], 2),
                        round(di['PinverterPV'],   2),       # ← también añadido
                        pi['n_ciclos'], round(pi['frac_C'], 3),
                        round(pi['min_spread'], 2),
                        round(pi['alpha'], 4), round(pi['beta'],  4),
                        round(pi['gamma'], 4), pi['n_horas_c'], pi['n_horas_d']])
    print("CSV Pareto: pareto_nsga2_E3E5.csv")

    # ── Configuración de colores ──────────────────────────────────────────────
    # FIX 2: cada criterio tiene su propia entrada — sin fusión por índice
    colores_criterios = {
        'idx_best_npv' : ('#e87722', 'D'),   # naranja / diamante
        'idx_best_bcr' : ('#1f77b4', 's'),   # azul    / cuadrado
        'idx_best_tir' : ('#d62728', '^'),   # rojo    / triángulo arriba
        'idx_best_lcoe': ('#9467bd', 'v'),   # morado  / triángulo abajo
    }
    indices_criterios = {
        'idx_best_npv' : idx_best_npv,
        'idx_best_bcr' : idx_best_bcr,
        'idx_best_tir' : idx_best_tir,
        'idx_best_lcoe': idx_best_lcoe,
    }

    # ── Textos en dos idiomas ─────────────────────────────────────────────────
    idiomas = {
        'es': {
            'pareto_title' : 'NSGA2 E3+E5  —  Frente de Pareto',
            'pareto_sub'   : 'PVNCF vs CAPEX',
            'xlabel'       : 'CAPEX (M USD)',
            'ylabel'       : 'PVNCF = CF/CRFE (M USD)',
            'cb_label'     : 'NPV (M USD)',
            'legend_title' : 'Soluciones destacadas',
            'conv_title'   : 'NSGA2 E3+E5  —  Convergencia',
            'conv_sub'     : 'Evolución del mejor PVNCF',
            'conv_xlabel'  : 'Generación',
            'conv_ylabel'  : 'PVNCF (M USD)',
            'conv_label'   : 'Mejor PVNCF por generación',
            'conv_best'    : 'PVNCF Máx NPV',
            'criterios'    : {
                'idx_best_npv' : '★ Máx NPV',
                'idx_best_bcr' : '■ Máx BCR',
                'idx_best_tir' : '▲ Máx TIR',
                'idx_best_lcoe': '▼ Mín LCOE',
            },
            'suffix': 'es',
        },
        'en': {
            'pareto_title' : 'NSGA2 E3+E5  —  Pareto Front',
            'pareto_sub'   : 'PVNCF vs CAPEX',
            'xlabel'       : 'CAPEX (M USD)',
            'ylabel'       : 'PVNCF = CF/CRFE (M USD)',
            'cb_label'     : 'NPV (M USD)',
            'legend_title' : 'Highlighted solutions',
            'conv_title'   : 'NSGA2 E3+E5  —  Convergence',
            'conv_sub'     : 'Best PVNCF evolution',
            'conv_xlabel'  : 'Generation',
            'conv_ylabel'  : 'PVNCF (M USD)',
            'conv_label'   : 'Best PVNCF per generation',
            'conv_best'    : 'PVNCF Max NPV',
            'criterios'    : {
                'idx_best_npv' : '★ Max NPV',
                'idx_best_bcr' : '■ Max BCR',
                'idx_best_tir' : '▲ Max TIR',
                'idx_best_lcoe': '▼ Min LCOE',
            },
            'suffix': 'en',
        },
    }

    def _label_idioma(idx, prefix):
        """Etiqueta multi-línea con las 5 métricas clave + ILR."""
        npv_m = real_npv[idx] / 1e6
        tir   = real_tir[idx]
        bcr   = real_bcr[idx]
        pb    = real_pb[idx]
        lcoe  = real_lcoe[idx]
        ILR_l = real_m[idx].get('ILR', 0.0)          # ← ILR en etiqueta
        lines = [
            prefix,
            f'NPV={npv_m:.2f}M',
            f'TIR={tir:.1f}%'  if not np.isnan(tir)  else 'TIR=—',
            f'BCR={bcr:.3f}'   if not np.isnan(bcr)  else 'BCR=—',
            f'PB={pb:.1f}yr'   if not np.isnan(pb)   else 'PB=—',
            f'LCOE={lcoe:.1f}' if not np.isnan(lcoe) else 'LCOE=—',
            f'ILR={ILR_l:.3f}',                       # ← ILR en etiqueta
        ]
        return '\n'.join(lines)

    # ── Bucle de gráficas por idioma — figura combinada (Pareto | Convergencia)
    for lang, L in idiomas.items():

        fig, (ax_p, ax_c) = plt.subplots(
            1, 2,
            figsize=(18, 7),
            gridspec_kw={'width_ratios': [1.5, 1]},
        )
        fig.suptitle(L['pareto_title'], fontsize=13, fontweight='bold', y=1.01)

        # ── Panel izquierdo: Frente de Pareto ────────────────────────────────
        sc = ax_p.scatter(real_capex_a / 1e6, real_opex / 1e6,
                          c=real_npv / 1e6, cmap='RdYlBu', s=60,
                          zorder=3, alpha=0.85)
        cb = fig.colorbar(sc, ax=ax_p, pad=0.02)
        cb.set_label(L['cb_label'], fontsize=9)

        x_all = real_capex_a / 1e6
        y_all = real_opex    / 1e6
        x_min, x_max = x_all.min(), x_all.max()
        y_min, y_max = y_all.min(), y_all.max()
        x_rng = max(x_max - x_min, 1e-6)
        y_rng = max(y_max - y_min, 1e-6)

        texts_p = []
        for key, idx_h in indices_criterios.items():
            col, mk = colores_criterios[key]
            prefix  = L['criterios'][key]
            x_pt    = real_capex_a[idx_h] / 1e6
            y_pt    = real_opex[idx_h]    / 1e6

            ax_p.scatter(x_pt, y_pt,
                         marker=mk, color=col,
                         s=90, zorder=6, label=prefix,
                         edgecolors='black', linewidths=0.8)

            lbl = _label_idioma(idx_h, prefix)
            txt = ax_p.text(
                x_pt, y_pt, lbl,
                fontsize=7.5, zorder=10,
                bbox=dict(boxstyle='round,pad=0.30', fc='white',
                          ec=col, lw=1.3, alpha=0.95),
            )
            texts_p.append(txt)

        adjust_text(
            texts_p,
            ax=ax_p,
            expand=(1.6, 2.0),
            force_text=(1.0, 1.2),
            force_points=(0.6, 0.9),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2),
        )

        ax_p.set_xlabel(L['xlabel'], fontsize=10)
        ax_p.set_ylabel(L['ylabel'], fontsize=10)
        ax_p.set_title(L['pareto_sub'], fontsize=11)
        ax_p.set_xlim(x_min - 0.08 * x_rng, x_max + 0.22 * x_rng)
        ax_p.set_ylim(y_min - 0.12 * y_rng, y_max + 0.22 * y_rng)
        ax_p.legend(fontsize=8, loc='lower right',
                    title=L['legend_title'], title_fontsize=8)
        ax_p.grid(alpha=0.3)
        ax_p.text(0.02, 0.98, '(a)', transform=ax_p.transAxes,
                  fontsize=11, fontweight='bold', va='top')

        # ── Panel derecho: Convergencia ───────────────────────────────────────
        ax_c.plot(callback.historial_gen,
                  np.array(callback.historial_opex) / 1e6,
                  color='steelblue', lw=2, label=L['conv_label'])
        ax_c.axhline(CF_best / crfe / 1e6, color='#e87722', ls='--', lw=1.5,
                     label=f'{L["conv_best"]}:\n{CF_best/crfe/1e6:.2f} M USD')
        ax_c.set_xlabel(L['conv_xlabel'], fontsize=10)
        ax_c.set_ylabel(L['conv_ylabel'], fontsize=10)
        ax_c.set_title(L['conv_sub'], fontsize=11)
        ax_c.legend(fontsize=8.5)
        ax_c.grid(alpha=0.3)
        ax_c.text(0.02, 0.98, '(b)', transform=ax_c.transAxes,
                  fontsize=11, fontweight='bold', va='top')

        fig.tight_layout()
        fname = f'nsga2_E3E5_{L["suffix"]}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Figura combinada ({lang}): {fname}')

    # ── Resumen en consola (con ILR) ──────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("RESUMEN DE SOLUCIONES DESTACADAS DEL FRENTE DE PARETO")
    print(f"{'─'*55}")
    criterios_resumen = [
        ('Máx NPV',  idx_best_npv),
        ('Máx BCR',  idx_best_bcr),
        ('Máx TIR',  idx_best_tir),
        ('Mín LCOE', idx_best_lcoe),
    ]
    for nombre, idx_h in criterios_resumen:
        ILR_s = real_m[idx_h].get('ILR', 0.0)
        print(f"\n  [{nombre}]  (idx={idx_h})")
        print(f"    NPV     = {real_npv[idx_h]:>12,.0f} USD")
        print(f"    ILR     = {ILR_s:>7.4f}")                      # ← ILR
        print(f"    TIR     = {real_tir[idx_h]:>7.2f} %"        if not np.isnan(real_tir[idx_h])  else "    TIR     = —")
        print(f"    BCR     = {real_bcr[idx_h]:>7.4f}"          if not np.isnan(real_bcr[idx_h])  else "    BCR     = —")
        print(f"    Payback = {real_pb[idx_h]:>6.1f} años"      if not np.isnan(real_pb[idx_h])   else "    Payback = —")
        print(f"    LCOEnet = {real_lcoe[idx_h]:>7.2f} USD/MWh" if not np.isnan(real_lcoe[idx_h]) else "    LCOEnet = —")
        print(f"    CAPEX   = {real_capex_a[idx_h]/1e6:>6.2f} M USD")