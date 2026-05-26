"""
NSGA2 bi-objetivo con MPC de 24h como política de despacho
===========================================================
Arquitectura:
  - El GA optimiza variables de diseño únicamente (igual que NSGA2+LP):
      Ppvinst, C, ratio_inverter, PbmaxP1..P5
  - El DESPACHO en cada hora t se decide con un LP de ventana de 24h:
      → Conoce lambda[t..t+23], Ppvu[t..t+23], Plu[t..t+23]
      → Resuelve el LP de 24h con SOC_inicial = SOC[t]
      → Ejecuta solo la primera acción: Pc[t], Pd[t], Pb[t], Ps[t]
      → Avanza a t+1 con el SOC resultante
  - Esto elimina lam_buy, lam_sell, soc_lo, soc_hi del cromosoma
  - El MPC ve precios futuros próximos → evita ciclado innecesario
    naturalmente, igual que el LP anual pero sin visión perfecta

Ventajas sobre heurística de umbrales:
  - No cicla sin beneficio económico (el LP de 24h lo evita)
  - No necesita restricción dura de ciclos
  - Política adaptativa: reacciona a la forma de la curva de precios

Ventajas sobre NSGA2+LP (8760h):
  - Más rápido: ~0.12s/evaluación vs ~5s/evaluación
  - Permite más generaciones con el mismo tiempo de cómputo

Trade-off:
  - El MPC comete errores de "horizonte corto" (no ve más allá de 24h)
  - Sesgo de aproximación mayor que NSGA2+LP pero menor que heurístico

Parámetros GA: pop=35, gen=25, cores=4
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import numpy_financial as npf
from sklearn.preprocessing import StandardScaler
import time
import csv
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool

import gurobipy as gp
from gurobipy import GRB, quicksum

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.parallelization.starmap import StarmapParallelization

from LP_model_funcional import (
    cargar_datos_ventana,
    Plinst, PmaxF,
    BoP, Sc, CAPEX_pv, CAPEX_BESS, CAPEX_BESS_inverter,
    crfe, crf, OaMpv, OaMbess, i, n, kappa,
    Rmax, eta, Area, DoD,
    eff_c, eff_d, eff_pv, er,
)

DIRECTORIO_VENTANA = Path(__file__).resolve().parent / 'ventana_completa'

PPV_MAX   = 8.0  * Plinst
C_MAX     = 15.0 * Plinst
RATIO_MIN = 0.1
RATIO_MAX = 2.0
H_MPC     = 24   # horizonte MPC en horas

SEMANAS_POR_ESTACION = {
    'Primavera': 13.0,
    'Verano'   : 13.0,
    'Otono'    : 13.0,
    'Invierno' : 13.0,
}
BLOQUES_ESTACION = {
    'Primavera': (1,    2184),
    'Verano'   : (2185, 4368),
    'Otono'    : (4369, 6552),
    'Invierno' : (6553, 8760),
}


# Semanas tipo
def seleccionar_semana_tipo(data_8760, periodo_8760,
                            h_inicio, h_fin, estacion):
    H_SEM   = 168
    n_horas = h_fin - h_inicio + 1
    n_sem   = n_horas // H_SEM
    if n_sem == 0:
        raise ValueError(f"Bloque {estacion} demasiado corto")

    semanas = []
    for s in range(n_sem):
        ini = h_inicio + s * H_SEM
        semanas.append(
            [data_8760[ini + h]['lambda'] for h in range(H_SEM)]
            + [data_8760[ini + h]['Ppvu']   for h in range(H_SEM)]
        )

    X     = np.array(semanas)
    X_s   = StandardScaler().fit_transform(X)
    s_rep = int(np.argmin(np.linalg.norm(X_s - X_s.mean(0), axis=1)))
    h_rep = h_inicio + s_rep * H_SEM

    data_sem, periodo_sem = {}, {}
    for h in range(H_SEM):
        t_local          = h + 1
        t_orig           = h_rep + h
        data_sem[t_local]    = data_8760[t_orig].copy()
        periodo_sem[t_local] = periodo_8760[t_orig]

    print(f"  {estacion}: semana {s_rep+1}/{n_sem} "
          f"(horas {h_rep}–{h_rep+H_SEM-1}) "
          f"→ {SEMANAS_POR_ESTACION[estacion]:.1f} sem/año")

    return {
        'data'     : data_sem,
        'periodo'  : periodo_sem,
        'T'        : range(1, H_SEM + 1),
        'n_semanas': SEMANAS_POR_ESTACION[estacion],
        'estacion' : estacion,
    }


def construir_semanas_tipo(data_8760, periodo_8760):
    print("Seleccionando semanas tipo por estación...")
    semanas = [
        seleccionar_semana_tipo(data_8760, periodo_8760, h0, h1, est)
        for est, (h0, h1) in BLOQUES_ESTACION.items()
    ]
    print(f"  Total semanas: {sum(s['n_semanas'] for s in semanas):.1f}")
    return semanas


# Simulación MPC hora a hora
def resolver_mpc_ventana(diseno, SOC_init, data_ventana, periodo_ventana, T_ventana):
    """
    Resuelve un LP de H_MPC horas con SOC inicial dado.
    Retorna las acciones de la primera hora solamente.
    Si falla, compra lo necesario de la red.

    T_ventana: lista de claves en data_ventana para las H_MPC horas.
    """
    Ppvinst       = diseno['Ppvinst']
    C             = diseno['C']
    PinverterBESS = diseno['PinverterBESS']
    Pbmax         = {p: diseno[f'PbmaxP{p}'] for p in range(1, 7)}

    SOCmin = ((1 - DoD) / 2)       * C
    SOCmax = ((1 - DoD) / 2 + DoD) * C
    SOC_init_clip = float(np.clip(SOC_init, SOCmin, SOCmax))

    m = gp.Model('MPC')
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 2.0)   # máx 2s por ventana

    T = list(T_ventana)

    SOC = {t: m.addVar(lb=SOCmin, ub=SOCmax, name=f'SOC[{t}]') for t in T}
    Ppv = {t: m.addVar(lb=0, name=f'Ppv[{t}]') for t in T}
    Pc  = {t: m.addVar(lb=0, ub=PinverterBESS, name=f'Pc[{t}]') for t in T}
    Pd  = {t: m.addVar(lb=0, ub=PinverterBESS, name=f'Pd[{t}]') for t in T}
    Pb  = {t: m.addVar(lb=0, name=f'Pb[{t}]') for t in T}
    Ps  = {t: m.addVar(lb=0, name=f'Ps[{t}]') for t in T}

    for k, t in enumerate(T):
        p    = periodo_ventana[t]
        PL_t = Plinst * data_ventana[t]['Plu']
        Ppvmx_t = Ppvinst * eff_pv * data_ventana[t]['Ppvu']

        m.addConstr(Pd[t] + Pb[t] + Ppv[t] == Pc[t] + Ps[t] + PL_t)
        m.addConstr(Ppv[t] <= Ppvmx_t)
        m.addConstr(Pb[t]  <= PmaxF)
        m.addConstr(Ps[t]  <= PmaxF)
        m.addConstr(Pb[t]  <= Pbmax[p])

        if k == 0:
            m.addConstr(SOC[t] == SOC_init_clip + Pc[t] * eff_c - Pd[t] / eff_d)
        else:
            t_prev = T[k - 1]
            m.addConstr(SOC[t] == SOC[t_prev] + Pc[t] * eff_c - Pd[t] / eff_d)

    # Objetivo: maximizar beneficio neto en la ventana
    eps = 1e-4
    obj = quicksum(
        er * data_ventana[t]['lambda'] * Ps[t]
        - er * (data_ventana[t]['lambda'] + data_ventana[t]['psi']) * Pb[t]
        - eps * (Pc[t] + Pd[t] + Pb[t] + Ps[t])
        for t in T
    )
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

    t0 = T[0]
    if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
        try:
            return {
                'Pc' : Pc[t0].X,
                'Pd' : Pd[t0].X,
                'Pb' : Pb[t0].X,
                'Ps' : Ps[t0].X,
                'Ppv': Ppv[t0].X,
                'ok' : True,
            }
        except Exception:
            pass

    # Fallback: cubrir la carga comprando de la red, BESS inactiva
    PL_t0   = Plinst * data_ventana[t0]['Plu']
    Ppv_t0  = min(Ppvinst * eff_pv * data_ventana[t0]['Ppvu'], PL_t0)
    p0      = periodo_ventana[t0]
    pb_max0 = min(PmaxF, Pbmax[p0])
    return {
        'Pc' : 0.0,
        'Pd' : 0.0,
        'Pb' : min(max(PL_t0 - Ppv_t0, 0.0), pb_max0),
        'Ps' : max(Ppv_t0 - PL_t0, 0.0),
        'Ppv': Ppv_t0,
        'ok' : False,
    }


# Simulación MPC hora a hora
def simular_mpc(diseno, data, periodo, T_list):
    """
    Recorre T_list hora a hora.
    En cada hora t resuelve un LP de H_MPC horas y ejecuta solo la
    primera acción. El horizonte se extiende circularmente al final
    para manejar el borde del período.

    Retorna dict con series temporales y SOC0.
    """
    C      = diseno['C']
    SOCmin = ((1 - DoD) / 2)       * C
    SOCmax = ((1 - DoD) / 2 + DoD) * C

    # SOC inicial: punto medio del rango operativo
    SOC = (SOCmin + SOCmax) / 2.0
    SOC0 = SOC

    n_t  = len(T_list)
    res  = {k: {} for k in ('Ppv', 'Ppvmx', 'Pc', 'Pd', 'Pb', 'Ps', 'SOC', 'ENS')}

    for idx, t in enumerate(T_list):
        # Construir ventana MPC de H_MPC horas
        # Índices locales de la ventana (wrap-around al final del período)
        win_idx  = [(idx + h) % n_t for h in range(H_MPC)]
        win_T    = [T_list[i] for i in win_idx]
        win_data = {h + 1: data[win_T[h]] for h in range(H_MPC)}
        win_per  = {h + 1: periodo[win_T[h]] for h in range(H_MPC)}
        win_keys = list(range(1, H_MPC + 1))

        # Resolver MPC y ejecutar primera acción
        accion = resolver_mpc_ventana(diseno, SOC, win_data, win_per, win_keys)

        Pc_t  = accion['Pc']
        Pd_t  = accion['Pd']
        Pb_t  = accion['Pb']
        Ps_t  = accion['Ps']
        Ppv_t = accion['Ppv']

        PL_t    = Plinst * data[t]['Plu']
        Ppvmx_t = diseno['Ppvinst'] * eff_pv * data[t]['Ppvu']

        # Balance de energía: verificar ENS residual
        balance = Pd_t + Pb_t + Ppv_t - Pc_t - Ps_t - PL_t
        ENS_t   = max(-balance, 0.0)   # déficit no cubierto

        SOC_new = float(np.clip(SOC + Pc_t * eff_c - Pd_t / eff_d, SOCmin, SOCmax))

        res['Ppvmx'][t] = Ppvmx_t
        res['Ppv'][t]   = min(Ppv_t, Ppvmx_t)
        res['Pc'][t]    = Pc_t
        res['Pd'][t]    = Pd_t
        res['Pb'][t]    = Pb_t
        res['Ps'][t]    = Ps_t
        res['ENS'][t]   = ENS_t
        res['SOC'][t]   = SOC_new
        SOC = SOC_new

    res['SOC0'] = SOC0
    return res


# Métricas financieras y operativas a partir de resultados de simulación MPC
def calcular_metricas(diseno, res, data, periodo, T_list, n_semanas=1.):
    Pbmax  = {p: diseno[f'PbmaxP{p}'] for p in range(1, 7)}
    scale  = n_semanas

    Es  = er * scale * sum(data[t]['lambda'] * res['Ps'][t] for t in T_list)
    Eb  = er * scale * sum((data[t]['lambda'] + data[t]['psi']) * res['Pb'][t] for t in T_list)
    Eb0 = er * scale * sum((data[t]['lambda'] + data[t]['psi']) * Plinst * data[t]['Plu'] for t in T_list)
    Wl  = scale * sum(Plinst * data[t]['Plu'] for t in T_list)
    ENS = scale * sum(res['ENS'][t] for t in T_list)

    CapacityP  = sum(kappa[p] * Pbmax[p] for p in range(1, 7))
    CapacityP0 = sum(kappa[p] * PmaxF    for p in range(1, 7))
    OaM        = OaMpv * diseno['Ppvinst'] + OaMbess * diseno['C']

    OPEX  = CapacityP + Eb  + OaM
    OPEX0 = CapacityP0 + Eb0
    CF    = Es + OPEX0 - OPEX

    return {
        'Es': Es, 'Eb': Eb, 'Eb0': Eb0,
        'CapacityP': CapacityP, 'CapacityP0': CapacityP0,
        'OaM': OaM, 'OPEX': OPEX, 'OPEX0': OPEX0,
        'CashFlow': CF, 'Savings': OPEX0 - OPEX, 'Benefit': Es - OPEX,
        'Wl': Wl, 'ENS': ENS,
        'Wb': scale * sum(res['Pb'][t] for t in T_list),
        'Ws': scale * sum(res['Ps'][t] for t in T_list),
        'Wc': scale * sum(res['Pc'][t] for t in T_list),
        'Wd': scale * sum(res['Pd'][t] for t in T_list),
        'wpv':      scale * sum(res['Ppv'][t]   for t in T_list),
        'wpvmx':    scale * sum(res['Ppvmx'][t] for t in T_list) / eff_pv,
        'wcurtail': scale * (
            sum(res['Ppvmx'][t] for t in T_list) / eff_pv
            - sum(res['Ppv'][t]  for t in T_list)
        ),
    }


# Evaluación sobre semanas tipo
def evaluar_diseno_semanas(diseno, semanas_tipo):
    """
    Evalúa el diseño usando MPC en las 4 semanas tipo.
    CapacityP y OaM se agregan una sola vez al año.
    """
    Es_anual  = 0.0
    Eb_anual  = 0.0
    Eb0_anual = 0.0
    ENS_anual = 0.0
    Wc_anual  = 0.0

    for sem in semanas_tipo:
        T_list = list(sem['T'])
        res    = simular_mpc(diseno, sem['data'], sem['periodo'], T_list)
        m      = calcular_metricas(diseno, res, sem['data'], sem['periodo'],
                                   T_list, n_semanas=sem['n_semanas'])
        Es_anual  += m['Es']
        Eb_anual  += m['Eb']
        Eb0_anual += m['Eb0']
        ENS_anual += m['ENS']
        Wc_anual  += m['Wc']

    CapacityP  = sum(kappa[p] * diseno[f'PbmaxP{p}'] for p in range(1, 7))
    CapacityP0 = sum(kappa[p] * PmaxF               for p in range(1, 7))
    OaM        = OaMpv * diseno['Ppvinst'] + OaMbess * diseno['C']

    OPEX  = CapacityP  + Eb_anual  + OaM
    OPEX0 = CapacityP0 + Eb0_anual
    CF    = Es_anual + OPEX0 - OPEX

    ciclos = Wc_anual / diseno['C'] if diseno['C'] > 0 else 1e9
    return CF, 'optimal', ENS_anual, ciclos


# Evaluación 8760h
def evaluar_8760(diseno, data_8760, periodo_8760, T_8760):
    T_list = list(T_8760)
    res    = simular_mpc(diseno, data_8760, periodo_8760, T_list)
    m      = calcular_metricas(diseno, res, data_8760, periodo_8760,
                               T_list, n_semanas=1.)
    ciclos_reales       = m['Wc'] / diseno['C'] if diseno['C'] > 0 else 1e9
    m['ciclos_anuales'] = ciclos_reales
    return m['CashFlow'], res, m


# Validación cruzada
def validar_aproximacion(diseno_ref, semanas_tipo,
                          data_8760, periodo_8760, T_8760):
    cf_sem, _, _, _ = evaluar_diseno_semanas(diseno_ref, semanas_tipo)
    cf_8760, _, _   = evaluar_8760(diseno_ref, data_8760, periodo_8760, T_8760)

    print(f"\nValidación de aproximación (MPC):")
    print(f"  CashFlow semanas tipo : {cf_sem:>12,.0f} USD/año")
    print(f"  CashFlow 8760h        : {cf_8760:>12,.0f} USD/año")

    if cf_8760 != 0:
        error = abs(cf_sem - cf_8760) / abs(cf_8760) * 100
        sesgo = (cf_sem - cf_8760) / cf_8760 * 100
        print(f"  Error relativo        : {error:.2f}%")
        print(f"  Sesgo (+sobreestima)  : {sesgo:+.2f}%")
        print(f"  {'OK' if error <= 20 else 'ADVERTENCIA: error > 20%'}")
        return error
    return float('inf')


# Decodificador
def decodificar(x):
    """
    Cromosoma de 8 variables — solo diseño, sin política de control.
    El MPC se encarga del despacho.
    """
    C     = float(x[1])
    ratio = float(np.clip(x[2], RATIO_MIN, RATIO_MAX))
    p1 = float(np.clip(x[3], 0., PmaxF))
    p2 = float(np.clip(p1 + x[4], 0., PmaxF))
    p3 = float(np.clip(p2 + x[5], 0., PmaxF))
    p4 = float(np.clip(p3 + x[6], 0., PmaxF))
    p5 = float(np.clip(p4 + x[7], 0., PmaxF))
    return {
        'Ppvinst'      : float(x[0]),
        'C'            : C,
        'PinverterBESS': ratio * C,
        'PbmaxP1'      : p1,
        'PbmaxP2'      : p2,
        'PbmaxP3'      : p3,
        'PbmaxP4'      : p4,
        'PbmaxP5'      : p5,
        'PbmaxP6'      : PmaxF,
    }


# Problema pymoo
class ProblemaNSGA2_MPC(ElementwiseProblem):
    """
    Bi-objetivo: -NPV y CAPEX.
    Despacho via MPC de 24h — sin variables de política en el cromosoma.
    Sin restricción dura de ciclos: el MPC los controla naturalmente
    mediante optimización económica en cada ventana.
    """

    def __init__(self, semanas_tipo, **kwargs):
        super().__init__(
            n_var=8, n_obj=2, n_ieq_constr=0,
            xl=np.array([0.,      0.,    RATIO_MIN,
                         0.,      0.,    0.,    0.,    0.]),
            xu=np.array([PPV_MAX, C_MAX, RATIO_MAX,
                         PmaxF,   PmaxF, PmaxF, PmaxF, PmaxF]),
            **kwargs,
        )
        self.semanas_tipo = semanas_tipo

    def _evaluate(self, x, out):
        diseno = decodificar(x)

        if diseno['C'] < 1e-3 or diseno['Ppvinst'] < 1e-3:
            out["F"] = [1e12, 1e12]
            return

        cashflow, status, ens_anual, _ = evaluar_diseno_semanas(
            diseno, self.semanas_tipo
        )

        if status != 'optimal' or ens_anual > 1.0:
            out["F"] = [1e12, 1e12]
            return

        inv = BoP + Sc * (
            CAPEX_pv              * diseno['Ppvinst']
            + CAPEX_BESS          * diseno['C']
            + CAPEX_BESS_inverter * diseno['PinverterBESS']
        )
        npv = cashflow / crfe - inv
        out["F"] = [-npv, inv]


# Callback para registrar convergencia y aplicar parada temprana por falta de mejora
class RegistrarConvergencia(Callback):
    def __init__(self, period=15, tol=1.0):
        super().__init__()
        self.historial_npv   = []
        self.historial_capex = []
        self.historial_gen   = []
        self.t0              = time.time()
        self.period          = period
        self.tol             = tol
        self._sin_mejora     = 0
        self._mejor_npv      = -np.inf

    def notify(self, algorithm):
        gen     = algorithm.n_gen
        F       = algorithm.opt.get("F")
        npv     = -float(np.min(F[:, 0]))
        capex   =  float(np.min(F[:, 1]))
        elapsed = time.time() - self.t0

        self.historial_npv.append(npv)
        self.historial_capex.append(capex)
        self.historial_gen.append(gen)

        if npv - self._mejor_npv > self.tol:
            self._sin_mejora = 0
            self._mejor_npv  = npv
        else:
            self._sin_mejora += 1

        print(f"Gen {gen:4d} | NPV: {npv:>14,.0f} USD | "
              f"CAPEX mín: {capex/1e6:>6.2f} MUSD | "
              f"sin mejora: {self._sin_mejora:3d}/{self.period} | "
              f"t: {elapsed:.1f}s")

        if self._sin_mejora >= self.period:
            print(f"\nParada temprana: {self.period} generaciones sin mejora")
            algorithm.termination.force_termination = True


# Main
if __name__ == "__main__":

    # 1. Cargar datos
    print("Cargando datos 8760h...")
    data_8760, periodo_8760, T_8760 = cargar_datos_ventana(DIRECTORIO_VENTANA)

    # 2. Semanas tipo
    semanas_tipo = construir_semanas_tipo(data_8760, periodo_8760)

    # 3. Diseño de referencia (MILP óptimo)
    diseno_ref = {
        'Ppvinst': 5991.87, 'C': 9568.53, 'PinverterBESS': 1999.61,
        'PbmaxP1': 0., 'PbmaxP2': 0., 'PbmaxP3': 0.,
        'PbmaxP4': 0., 'PbmaxP5': 0., 'PbmaxP6': PmaxF,
    }

    # 4. Tiempo por evaluación y diagnóstico
    print("\nMidiendo tiempo por evaluación MPC...")
    t0 = time.time()
    cf_diag, _, ens_diag, ciclos_diag = evaluar_diseno_semanas(
        diseno_ref, semanas_tipo
    )
    t_eval = time.time() - t0
    C_ref   = diseno_ref['C']
    inv_ref = BoP + Sc * (CAPEX_pv * diseno_ref['Ppvinst']
                          + CAPEX_BESS * C_ref
                          + CAPEX_BESS_inverter * diseno_ref['PinverterBESS'])
    npv_diag = cf_diag / crfe - inv_ref
    print(f"  Tiempo por evaluación  : {t_eval:.2f}s")
    print(f"  CashFlow (proxy MPC)   : {cf_diag:>12,.0f} USD/año")
    print(f"  ENS anual              : {ens_diag:>12,.4f} kWh")
    print(f"  Ciclos anuales         : {ciclos_diag:>12,.1f} ciclos/año")
    print(f"  NPV estimado           : {npv_diag:>12,.0f} USD")

    # 5. Validación cruzada
    error_pct = validar_aproximacion(
        diseno_ref, semanas_tipo, data_8760, periodo_8760, T_8760
    )
    if error_pct > 25:
        print(f"ADVERTENCIA: error de aproximación alto ({error_pct:.1f}%)")

    # 6. GA
    POP_SIZE  = 35
    N_MAX_GEN = 25
    PERIOD    = 10
    N_CORES   = 4

    t_est = t_eval * POP_SIZE * N_MAX_GEN / N_CORES / 60
    print(f"\nTiempo estimado NSGA2-MPC: {t_est:.1f} min "
          f"(pop={POP_SIZE}, gen={N_MAX_GEN}, cores={N_CORES})")

    pool     = Pool(N_CORES)
    runner   = StarmapParallelization(pool.starmap)
    problema = ProblemaNSGA2_MPC(semanas_tipo, elementwise_runner=runner)

    algoritmo = NSGA2(
        pop_size=POP_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    criterio = get_termination("n_gen", N_MAX_GEN)
    callback = RegistrarConvergencia(period=PERIOD, tol=1.0)

    print("\nIniciando NSGA2-MPC...")
    t0 = time.time()
    resultado_ga = minimize(
        problema, algoritmo,
        termination=criterio,
        seed=42, verbose=False, callback=callback,
    )
    t_ga = time.time() - t0
    pool.close(); pool.join()
    print(f"\nNSGA2-MPC completado en {t_ga/60:.1f} min "
          f"| {resultado_ga.algorithm.n_gen} generaciones")

    # 7. Frente de Pareto proxy
    pareto_X     = resultado_ga.X
    pareto_npv   = -resultado_ga.F[:, 0]
    pareto_capex =  resultado_ga.F[:, 1]
    n_pareto     = len(pareto_npv)

    print(f"\nFrente de Pareto proxy — {n_pareto} soluciones")
    print(f"  NPV  rango: [{pareto_npv.min():,.0f}, {pareto_npv.max():,.0f}] USD")
    print(f"  CAPEX rango: [{pareto_capex.min()/1e6:.2f}, {pareto_capex.max()/1e6:.2f}] MUSD")

    # 8. Validación 8760h de TODAS las soluciones del frente
    print(f"\nValidando {n_pareto} soluciones en 8760h completas (MPC)...")
    print(f"  (estimado: ~{n_pareto * t_eval * 8760/168 / 60:.1f} min)")
    print(f"{'─'*65}")

    t0_val     = time.time()
    real_npv   = np.full(n_pareto, np.nan)
    real_capex = np.full(n_pareto, np.nan)
    real_cf    = np.full(n_pareto, np.nan)
    real_res   = [None] * n_pareto
    real_m     = [None] * n_pareto

    for idx in range(n_pareto):
        diseno_i = decodificar(pareto_X[idx])
        inv_i    = BoP + Sc * (
            CAPEX_pv              * diseno_i['Ppvinst']
            + CAPEX_BESS          * diseno_i['C']
            + CAPEX_BESS_inverter * diseno_i['PinverterBESS']
        )
        cf_i, res_i, m_i = evaluar_8760(diseno_i, data_8760, periodo_8760, T_8760)
        real_capex[idx] = inv_i
        real_cf[idx]    = cf_i
        real_npv[idx]   = cf_i / crfe - inv_i
        real_res[idx]   = res_i
        real_m[idx]     = m_i

        sesgo_i = (pareto_npv[idx] - real_npv[idx]) / abs(real_npv[idx]) * 100 \
                  if real_npv[idx] != 0 else float('nan')
        print(f"  [{idx+1:>3}/{n_pareto}] "
              f"proxy: {pareto_npv[idx]:>12,.0f} | "
              f"real: {real_npv[idx]:>12,.0f} | "
              f"ciclos: {m_i['ciclos_anuales']:>6.0f} | "
              f"sesgo: {sesgo_i:>+7.1f}%")

    t_val = time.time() - t0_val
    print(f"{'─'*65}")
    print(f"  Validación completada en {t_val/60:.1f} min")

    # Selecciones sobre frente real
    idx_real_npv = int(np.nanargmax(real_npv))
    bcr_real     = (real_npv + real_capex) / np.where(real_capex > 0, real_capex, 1e-9)
    idx_real_bcr = int(np.nanargmax(bcr_real))

    # Guardar tabla completa
    with open("pareto_mpc_8760h.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "NPV_proxy", "CAPEX_proxy",
                         "NPV_8760h", "CAPEX_8760h", "CF_8760h",
                         "ciclos_anuales", "sesgo_pct",
                         "Ppvinst", "C", "PinverterBESS",
                         "PbmaxP1", "PbmaxP2", "PbmaxP3", "PbmaxP4", "PbmaxP5"])
        for idx in range(n_pareto):
            d_i = decodificar(pareto_X[idx])
            s_i = (pareto_npv[idx] - real_npv[idx]) / abs(real_npv[idx]) * 100 \
                  if real_npv[idx] != 0 else float('nan')
            writer.writerow([
                idx, pareto_npv[idx], pareto_capex[idx],
                real_npv[idx], real_capex[idx], real_cf[idx],
                real_m[idx]['ciclos_anuales'], s_i,
                d_i['Ppvinst'], d_i['C'], d_i['PinverterBESS'],
                d_i['PbmaxP1'], d_i['PbmaxP2'], d_i['PbmaxP3'],
                d_i['PbmaxP4'], d_i['PbmaxP5'],
            ])
    print("  Tabla completa guardada en pareto_mpc_8760h.csv")

    # 9. Resultados detallados
    def imprimir_resultado(idx_sel, label):
        d   = decodificar(pareto_X[idx_sel])
        r   = real_res[idx_sel]
        m   = real_m[idx_sel]
        CF  = real_cf[idx_sel]
        inv = real_capex[idx_sel]

        npv_val          = CF / crfe - inv
        TIR              = npf.irr([-inv] + [CF] * n) * 100 if inv > 0 else 0
        BCratio          = npf.pv(rate=i, nper=n, pmt=-CF, fv=0) / inv if inv > 0 else 0
        OPEXgross        = OaMpv * d['Ppvinst'] + OaMbess * d['C']
        Wl_val           = m['Wl']
        LCOEgross        = 1000*(inv + OPEXgross/crfe)/(Wl_val/crf) if Wl_val > 0 else 0
        LCOEnet          = 1000*(inv + (m['OPEX'] - m['OPEX0'] - m['Es'])/crfe)/(Wl_val/crf) if Wl_val > 0 else 0
        NPERaprox        = inv / CF if CF > 0 else 0
        try:
            NPER = np.log(CF / (CF + i*(-inv))) / np.log(1+i) if (CF + i*(-inv)) > 0 else 0
        except Exception:
            NPER = 0
        Crate            = d['PinverterBESS'] / d['C'] if d['C'] > 0 else 0
        nx               = 1000 * d['Ppvinst'] / (Rmax * eta * Area)
        BESSbatteryCost  = CAPEX_BESS          * d['C']
        BESSinverterCost = CAPEX_BESS_inverter * d['PinverterBESS']
        PVsystemCost     = CAPEX_pv            * d['Ppvinst']
        sesgo            = (pareto_npv[idx_sel] - npv_val) / abs(npv_val) * 100 if npv_val != 0 else 0
        gap_milp         = (npv_val - 3_581_528) / 3_581_528 * 100

        print(f"\n{'='*60}")
        print(f"RESULTADOS — {label}")
        print(f"{'='*60}")
        print(f"Parámetros de diseño:")
        print(f"  BESS capacity (WbessInst):           {d['C']:>12,.2f} kWh")
        print(f"  BESS inverter capacity (PbessInst):  {d['PinverterBESS']:>12,.2f} kW")
        print(f"  PV System capacity (PpvInst):        {d['Ppvinst']:>12,.2f} kW")
        print(f"  Npan:                                {nx:>12,.2f} modules of 500W")
        print(f"  C-rate (diseño):                     {Crate:>12,.4f} 1/h")
        print(f"  Contracted power per period:")
        for p in range(1, 7):
            print(f"    PbmaxP{p}: {d[f'PbmaxP{p}']:>10,.2f} kW")
        print(f"------Verificación BESS------------------")
        print(f"  Ciclos anuales (MPC 8760h):          {m['ciclos_anuales']:>12,.1f} ciclos/año")
        print(f"  SOC0 (punto medio rango operativo):  {r['SOC0']:>12,.2f} kWh")
        print(f"------Energy Dispatch--------------------")
        print(f"  Energy load consumption (Wl):        {m['Wl']:>12,.2f} kWh/year")
        print(f"  Energy bought from market (Wb):      {m['Wb']:>12,.2f} kWh/year")
        print(f"  Energy sold to market (Ws):          {m['Ws']:>12,.2f} kWh/year")
        print(f"  BESS Energy charged (Wc):            {m['Wc']:>12,.2f} kWh/year")
        print(f"  BESS Energy discharged (Wd):         {m['Wd']:>12,.2f} kWh/year")
        print(f"  PV energy generated (wpvmx):         {m['wpvmx']:>12,.2f} kWh/year")
        print(f"  PV energy injected (wpv):            {m['wpv']:>12,.2f} kWh/year")
        print(f"  PV energy curtailed (Wcurtail):      {m['wcurtail']:>12,.2f} kWh/year")
        print(f"------Financial results------------------")
        print(f"  Operational Benefit:                 {m['Benefit']:>12,.2f} USD/year")
        print(f"  OPEX with project:                   {m['OPEX']:>12,.2f} USD/year")
        print(f"  OPEX without project (OPEX_0):       {m['OPEX0']:>12,.2f} USD/year")
        print(f"  Operational Savings:                 {m['Savings']:>12,.2f} USD/year")
        print(f"  Energy expenses (Eb):                {m['Eb']:>12,.2f} USD/year")
        print(f"  Energy expenses w/o project (Eb0):   {m['Eb0']:>12,.2f} USD/year")
        print(f"  Energy earnings (Es):                {m['Es']:>12,.2f} USD/year")
        print(f"  Capacity Charges (CP):               {m['CapacityP']:>12,.2f} USD/year")
        print(f"  Capacity Charges w/o project (CP0):  {m['CapacityP0']:>12,.2f} USD/year")
        print(f"  CAPEX:                               {inv:>12,.2f} USD")
        print(f"    BESS battery cost:                 {BESSbatteryCost:>12,.2f} USD")
        print(f"    BESS inverter cost:                {BESSinverterCost:>12,.2f} USD")
        print(f"    Soft Costs:                        {inv*(Sc-1):>12,.2f} USD")
        print(f"    PV system cost:                    {PVsystemCost:>12,.2f} USD")
        print(f"  Net Present Value:                   {npv_val:>12,.2f} USD")
        print(f"  Project Cash Flow:                   {CF:>12,.2f} USD/year")
        print(f"  Internal Rate of Return:             {TIR:>12,.2f} %")
        print(f"  Pay Back Time:                       {NPER:>12,.2f} years")
        print(f"  Simple Pay Back Time:                {NPERaprox:>12,.2f} years")
        print(f"  Net LCOE:                            {LCOEnet:>12,.2f} USD/MWh")
        print(f"  Gross LCOE:                          {LCOEgross:>12,.2f} USD/MWh")
        print(f"  Benefit-Cost Ratio:                  {BCratio:>12,.2f}")
        print(f"------MPC vs referencia------------------")
        print(f"  NPV proxy (semanas tipo):            {pareto_npv[idx_sel]:>12,.0f} USD")
        print(f"  NPV real  (8760h MPC):               {npv_val:>12,.0f} USD")
        print(f"  NPV MILP  referencia:                  3,581,528 USD")
        print(f"  NPV NSGA2+LP:                          3,573,288 USD")
        print(f"  Sesgo aproximación:                  {sesgo:>+11.2f}%")
        print(f"  Gap vs MILP:                         {gap_milp:>+11.2f}%")
        return npv_val, CF

    npv_best, CF_best = imprimir_resultado(
        idx_real_npv, "NSGA2-MPC — Máximo NPV real (8760h)"
    )
    imprimir_resultado(
        idx_real_bcr, "NSGA2-MPC — Máximo BCR real (8760h)"
    )

    # Guardar despacho de la mejor solución
    r_best = real_res[idx_real_npv]
    with open("solucion_nsga2_mpc.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Hour", "Pb", "Ps", "Pc", "Pd", "SOC", "CF", "NPV"])
        for t in T_8760:
            writer.writerow([t,
                              r_best['Pb'][t], r_best['Ps'][t],
                              r_best['Pc'][t], r_best['Pd'][t],
                              r_best['SOC'][t], CF_best, npv_best])
    print("\nDespacho horario guardado en solucion_nsga2_mpc.csv")

    # 10. Gráficas
    sesgos = np.array([
        (pareto_npv[i] - real_npv[i]) / abs(real_npv[i]) * 100
        if real_npv[i] != 0 else np.nan
        for i in range(n_pareto)
    ])

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    ax = axes[0]
    bcr_plot = (real_npv + real_capex) / np.where(real_capex > 0, real_capex, 1e-9)
    sc = ax.scatter(real_capex / 1e6, real_npv / 1e6,
                    c=bcr_plot, cmap='RdYlGn', s=80, zorder=3,
                    vmin=0.5, vmax=2.5)
    ax.scatter(pareto_capex / 1e6, pareto_npv / 1e6,
               c='lightgrey', s=40, zorder=2, alpha=0.6, label='Proxy')
    for idx in range(n_pareto):
        ax.annotate('',
            xy=(real_capex[idx]/1e6, real_npv[idx]/1e6),
            xytext=(pareto_capex[idx]/1e6, pareto_npv[idx]/1e6),
            arrowprops=dict(arrowstyle='->', color='grey', lw=0.7),
        )
    ax.scatter(real_capex[idx_real_npv]/1e6, real_npv[idx_real_npv]/1e6,
               c='blue', s=200, marker='*', zorder=5, label='Máx NPV real')
    ax.scatter(real_capex[idx_real_bcr]/1e6, real_npv[idx_real_bcr]/1e6,
               c='black', s=200, marker='D', zorder=5, label='Máx BCR real')
    ax.axhline(3.581528, color='red',    ls='--', lw=1.2, label='MILP ref.')
    ax.axhline(3.573288, color='orange', ls='--', lw=1.2, label='NSGA2+LP')
    fig.colorbar(sc, ax=ax, label='BCR real')
    ax.set_xlabel("CAPEX (M USD)")
    ax.set_ylabel("NPV (M USD)")
    ax.set_title("Frente de Pareto real\n(NSGA2-MPC 24h)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(callback.historial_gen,
            np.array(callback.historial_npv) / 1e6,
            color='steelblue', lw=2, label='Proxy MPC')
    ax.axhline(3.581528, color='red',    ls='--', lw=1.2, label='MILP ref.')
    ax.axhline(3.573288, color='orange', ls='--', lw=1.2, label='NSGA2+LP')
    ax.axhline(npv_best / 1e6, color='blue', ls=':', lw=1.5,
               label=f'Mejor real: {npv_best/1e6:.2f} MUSD')
    ax.set_xlabel("Generación")
    ax.set_ylabel("Mejor NPV del frente (M USD)")
    ax.set_title("Convergencia NPV — NSGA2-MPC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.bar(range(n_pareto),
           np.where(np.isnan(sesgos), 0, sesgos),
           color='steelblue')
    ax.axhline(0, color='black', lw=0.8)
    sesgos_ok = sesgos[~np.isnan(sesgos)]
    if len(sesgos_ok):
        ax.axhline(sesgos_ok.mean(), color='red', ls='--', lw=1.2,
                   label=f'Sesgo medio: {sesgos_ok.mean():+.1f}%')
    ax.set_xlabel("Solución del frente")
    ax.set_ylabel("Sesgo proxy vs real (%)")
    ax.set_title("Sesgo por solución\n(proxy MPC - real 8760h)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("convergencia_nsga2_mpc.png", dpi=150)
    plt.show()
    print("Gráfica guardada en convergencia_nsga2_mpc.png")