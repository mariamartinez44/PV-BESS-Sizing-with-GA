"""
NSGA2 bi-objetivo con LP de despacho (Gurobi)
==============================================
Objetivos:
  F[0] = -NPV    (maximizar NPV   → minimizar -NPV)
  F[1] =  CAPEX  (minimizar CAPEX → minimizar  CAPEX)

Cromosoma (8 variables reales) — idéntico al GA mono-objetivo:
  [0]  Ppvinst        — capacidad PV instalada (kW)
  [1]  C              — capacidad BESS (kWh)
  [2]  ratio_inverter — PinverterBESS / C  (C-rate, adimensional)
  [3]  PbmaxP1        — potencia contratada período 1 (kW)
  [4]  delta_P2       — incremento acumulado P2 (kW)
  [5]  delta_P3       — incremento acumulado P3 (kW)
  [6]  delta_P4       — incremento acumulado P4 (kW)
  [7]  delta_P5       — incremento acumulado P5 (kW)
  PbmaxP6 = PmaxF siempre.

El despacho se optimiza con el LP de Gurobi (resolver_lp_operacion)

Arquitectura de evaluación:
  - 4 semanas tipo estacionales (168h cada una)
  - Factor de escala: n_semanas × CashFlow_semana
  - CapacityP y OaM se agregan una sola vez al año
  - Validación cruzada contra 8760h antes del GA

Parámetros: pop=35, gen=25, cores=4 
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

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.parallelization.starmap import StarmapParallelization

# Importar LP y parámetros compartidos desde LP_model_funcional

from LP_model_funcional import (
    cargar_datos_ventana,
    resolver_lp_operacion,
    Plinst, PmaxF,
    BoP, Sc, CAPEX_pv, CAPEX_BESS, CAPEX_BESS_inverter,
    crfe, crf, OaMpv, OaMbess, i, n, kappa,
    Rmax, eta, Area, DoD,
)

DIRECTORIO_VENTANA = Path(__file__).resolve().parent / 'ventana_completa'

PPV_MAX   = 8.0  * Plinst
C_MAX     = 15.0 * Plinst
RATIO_MIN = 0.1
RATIO_MAX = 2.0

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


# Selección de semana representativa por estación — método de k-means con k=1 sobre lambda y Ppvu
def seleccionar_semana_tipo(data_8760, periodo_8760,
                            h_inicio, h_fin, estacion):
    H_SEM  = 168
    n_horas = h_fin - h_inicio + 1
    n_sem   = n_horas // H_SEM
    if n_sem == 0:
        raise ValueError(f"Bloque {estacion} demasiado corto: {n_horas}h")

    semanas = []
    for s in range(n_sem):
        ini = h_inicio + s * H_SEM
        semanas.append(
            [data_8760[ini + h]['lambda'] for h in range(H_SEM)]
            + [data_8760[ini + h]['Ppvu']   for h in range(H_SEM)]
        )

    X      = np.array(semanas)
    X_s    = StandardScaler().fit_transform(X)
    s_rep  = int(np.argmin(np.linalg.norm(X_s - X_s.mean(0), axis=1)))
    h_rep  = h_inicio + s_rep * H_SEM

    data_sem, periodo_sem = {}, {}
    for h in range(H_SEM):
        t_local          = h + 1
        t_orig           = h_rep + h
        data_sem[t_local]    = data_8760[t_orig].copy()
        periodo_sem[t_local] = periodo_8760[t_orig]

    print(f"  {estacion}: semana {s_rep+1}/{n_sem} "
          f"(horas {h_rep}–{h_rep+H_SEM-1}) "
          f"→ representa {SEMANAS_POR_ESTACION[estacion]:.1f} semanas/año")

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


# Evaluación con LP sobre semanas tipo y factor de escala anual
def evaluar_diseno_semanas(diseno, semanas_tipo):
    """
    CashFlow anual = Σ_estacion [ n_semanas × (Es_sem - Eb_sem - Eb0_sem) ]
                    - CapacityP - OaM + CapacityP0
    CapacityP y OaM son anuales fijos.
    """
    Es_anual  = 0.0
    Eb_anual  = 0.0
    Eb0_anual = 0.0

    for sem in semanas_tipo:
        cf, resultado, status = resolver_lp_operacion(
            diseno,
            sem['data'],
            sem['periodo'],
            sem['T'],
        )
        if status != 'optimal':
            return -1e9, f"infeasible_{sem['estacion']}"

        Es_anual  += resultado['Es']  * sem['n_semanas']
        Eb_anual  += resultado['Eb']  * sem['n_semanas']
        Eb0_anual += resultado['Eb0'] * sem['n_semanas']

    CapacityP  = sum(kappa[p] * diseno[f'PbmaxP{p}'] for p in range(1, 7))
    CapacityP0 = sum(kappa[p] * PmaxF               for p in range(1, 7))
    OaM        = OaMpv * diseno['Ppvinst'] + OaMbess * diseno['C']

    OPEX  = CapacityP  + Eb_anual  + OaM
    OPEX0 = CapacityP0 + Eb0_anual

    cashflow_anual = Es_anual + OPEX0 - OPEX
    return cashflow_anual, 'optimal'


#  Validación cruzada contra 8760h para el diseño de referencia
def validar_aproximacion(diseno_ref, semanas_tipo,
                         data_8760, periodo_8760, T_8760):
    cf_sem, status_sem   = evaluar_diseno_semanas(diseno_ref, semanas_tipo)
    cf_8760, _, status_8760 = resolver_lp_operacion(
        diseno_ref, data_8760, periodo_8760, T_8760
    )

    print(f"\nValidación de aproximación (diseño de referencia):")
    print(f"  CashFlow semanas tipo : {cf_sem:>12,.0f} USD/año  [{status_sem}]")
    print(f"  CashFlow 8760h        : {cf_8760:>12,.0f} USD/año  [{status_8760}]")

    if status_8760 == 'optimal' and cf_8760 != 0:
        error = abs(cf_sem - cf_8760) / abs(cf_8760) * 100
        sesgo = (cf_sem - cf_8760) / cf_8760 * 100
        print(f"  Error relativo        : {error:.2f}%")
        print(f"  Sesgo (+sobreestima)  : {sesgo:+.2f}%")
        print(f"  {'OK: error aceptable' if error <= 20 else 'ADVERTENCIA: error > 20%'}")
        return error
    return float('inf')


#  Decodificador del cromosoma
def decodificar(x):
    """
    Convierte el vector x de 8 variables reales al dict de diseño.
    PbmaxP1..P5 se construyen acumulativamente para garantizar P1≤P2≤...≤P5.
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
class ProblemaNSGA2_LP(ElementwiseProblem):
    """
    Bi-objetivo:
      F[0] = -NPV    (maximizar NPV)
      F[1] =  CAPEX  (minimizar CAPEX)

    El frente de Pareto resultante muestra la frontera eficiente
    NPV vs CAPEX: para cada nivel de inversión, el máximo NPV alcanzable
    con despacho óptimo (LP).

    Constraints implícitos ya manejados por el LP
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

        # Filtro mínimo: evitar solución trivial sin instalación
        if diseno['C'] < 1e-3 or diseno['Ppvinst'] < 1e-3:
            out["F"] = [1e12, 1e12]
            return

        cashflow, status = evaluar_diseno_semanas(diseno, self.semanas_tipo)
        if status != 'optimal':
            out["F"] = [1e12, 1e12]
            return

        inv = BoP + Sc * (
            CAPEX_pv              * diseno['Ppvinst']
            + CAPEX_BESS          * diseno['C']
            + CAPEX_BESS_inverter * diseno['PinverterBESS']
        )
        npv = cashflow / crfe - inv

        out["F"] = [-npv, inv]


# Callback de convergencia para parada temprana si no hay mejora significativa en NPV durante varias generaciones
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


#  Main
if __name__ == "__main__":

    # 1. Cargar datos
    print("Cargando datos 8760h...")
    data_8760, periodo_8760, T_8760 = cargar_datos_ventana(DIRECTORIO_VENTANA)

    # 2. Construir semanas tipo estacionales
    semanas_tipo = construir_semanas_tipo(data_8760, periodo_8760)

    # 3. Validación cruzada (mismo diseño de referencia que el GA)
    diseno_ref = {
        'Ppvinst': 5991.87, 'C': 9568.53, 'PinverterBESS': 1999.61,
        'PbmaxP1': 0., 'PbmaxP2': 0., 'PbmaxP3': 0.,
        'PbmaxP4': 0., 'PbmaxP5': 0., 'PbmaxP6': PmaxF,
    }
    error_pct = validar_aproximacion(
        diseno_ref, semanas_tipo, data_8760, periodo_8760, T_8760
    )
    if error_pct > 20:
        print(f"\nERROR: aproximación demasiado imprecisa ({error_pct:.1f}%)")
        raise SystemExit(1)

    # 4. Tiempo por evaluación
    t0 = time.time()
    evaluar_diseno_semanas(diseno_ref, semanas_tipo)
    t_eval = time.time() - t0
    print(f"\nTiempo por evaluación (semanas tipo): {t_eval:.2f}s")

    # 5. Configuración del GA
    POP_SIZE  = 35
    N_MAX_GEN = 25
    PERIOD    = 10
    N_CORES   = 4

    t_est = t_eval * POP_SIZE * N_MAX_GEN / N_CORES / 60
    print(f"Tiempo estimado NSGA2: {t_est:.1f} min "
          f"(pop={POP_SIZE}, gen={N_MAX_GEN}, cores={N_CORES})")

    pool     = Pool(N_CORES)
    runner   = StarmapParallelization(pool.starmap)
    problema = ProblemaNSGA2_LP(semanas_tipo, elementwise_runner=runner)

    algoritmo = NSGA2(
        pop_size=POP_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    criterio = get_termination("n_gen", N_MAX_GEN)
    callback = RegistrarConvergencia(period=PERIOD, tol=1.0)

    # 6. Ejecutar NSGA2
    print("\nIniciando NSGA2 + LP...")
    t0 = time.time()
    resultado_ga = minimize(
        problema, algoritmo,
        termination=criterio,
        seed=42, verbose=False, callback=callback,
    )
    t_ga = time.time() - t0
    pool.close(); pool.join()
    print(f"\nNSGA2 completado en {t_ga/60:.1f} min "
          f"| {resultado_ga.algorithm.n_gen} generaciones")

    # 7. Frente de Pareto proxy (semanas tipo)
    pareto_X     = resultado_ga.X
    pareto_npv_proxy  = -resultado_ga.F[:, 0]
    pareto_capex_proxy =  resultado_ga.F[:, 1]
    n_pareto = len(pareto_npv_proxy)

    print(f"\n{'='*60}")
    print(f"Frente de Pareto proxy — {n_pareto} soluciones no dominadas")
    print(f"  NPV  rango: [{pareto_npv_proxy.min():,.0f}, {pareto_npv_proxy.max():,.0f}] USD")
    print(f"  CAPEX rango: [{pareto_capex_proxy.min()/1e6:.2f}, {pareto_capex_proxy.max()/1e6:.2f}] MUSD")

    # 8. Validación 8760h de todas las soluciones del frente
    print(f"\n{'='*60}")
    print(f"Validando las {n_pareto} soluciones del frente en 8760h completas...")
    print(f"  (estimado: ~{n_pareto * 5:.0f}s con LP secuencial)")
    print(f"{'─'*60}")

    t0_val = time.time()
    real_npv   = np.full(n_pareto, np.nan)
    real_capex = np.full(n_pareto, np.nan)
    real_cf    = np.full(n_pareto, np.nan)
    real_res   = [None] * n_pareto
    real_status = [''] * n_pareto

    for idx in range(n_pareto):
        diseno_i = decodificar(pareto_X[idx])
        inv_i = BoP + Sc * (
            CAPEX_pv              * diseno_i['Ppvinst']
            + CAPEX_BESS          * diseno_i['C']
            + CAPEX_BESS_inverter * diseno_i['PinverterBESS']
        )
        cf_i, res_i, st_i = resolver_lp_operacion(
            diseno_i, data_8760, periodo_8760, T_8760
        )
        real_status[idx] = st_i
        real_capex[idx]  = inv_i
        if st_i == 'optimal':
            real_npv[idx] = cf_i / crfe - inv_i
            real_cf[idx]  = cf_i
            real_res[idx] = res_i
        else:
            real_npv[idx] = np.nan
            real_cf[idx]  = np.nan

        sesgo_i = (pareto_npv_proxy[idx] - real_npv[idx]) / abs(real_npv[idx]) * 100 \
                  if not np.isnan(real_npv[idx]) and real_npv[idx] != 0 else float('nan')
        print(f"  [{idx+1:>3}/{n_pareto}] "
              f"NPV proxy: {pareto_npv_proxy[idx]:>12,.0f} | "
              f"NPV 8760h: {real_npv[idx]:>12,.0f} | "
              f"sesgo: {sesgo_i:>+7.1f}% | {st_i}")

    t_val = time.time() - t0_val
    print(f"{'─'*60}")
    print(f"  Validación completada en {t_val:.1f}s")

    # Filtrar solo óptimas
    mask_ok = np.array([s == 'optimal' for s in real_status])
    if not mask_ok.any():
        print("ERROR: ninguna solución del frente fue óptima en 8760h.")
        raise SystemExit(1)

    # Frente real 8760h — recalcular dominancia sobre NPV y CAPEX reales
    npv_ok   = real_npv[mask_ok]
    capex_ok = real_capex[mask_ok]
    idx_ok   = np.where(mask_ok)[0]

    # Selecciones sobre el frente real
    idx_real_npv  = idx_ok[int(np.argmax(npv_ok))]
    idx_real_bcr  = idx_ok[int(np.argmax(
        (npv_ok + capex_ok) / np.where(capex_ok > 0, capex_ok, 1e-9)
    ))]

    print(f"\n{'='*60}")
    print(f"Resumen del frente real (8760h):")
    print(f"  Soluciones óptimas: {mask_ok.sum()} / {n_pareto}")
    print(f"  NPV  rango real: [{npv_ok.min():,.0f}, {npv_ok.max():,.0f}] USD")
    print(f"  CAPEX rango real: [{capex_ok.min()/1e6:.2f}, {capex_ok.max()/1e6:.2f}] MUSD")

    # Guardar tabla completa del frente real
    with open("pareto_real_8760h.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx", "status",
            "NPV_proxy_USD", "CAPEX_proxy_USD",
            "NPV_8760h_USD", "CAPEX_8760h_USD", "CF_8760h_USD",
            "sesgo_pct",
            "Ppvinst", "C", "PinverterBESS",
            "PbmaxP1", "PbmaxP2", "PbmaxP3", "PbmaxP4", "PbmaxP5", "PbmaxP6",
        ])
        for idx in range(n_pareto):
            d_i = decodificar(pareto_X[idx])
            sesgo_i = (pareto_npv_proxy[idx] - real_npv[idx]) / abs(real_npv[idx]) * 100 \
                      if not np.isnan(real_npv[idx]) and real_npv[idx] != 0 else float('nan')
            writer.writerow([
                idx, real_status[idx],
                pareto_npv_proxy[idx], pareto_capex_proxy[idx],
                real_npv[idx], real_capex[idx], real_cf[idx],
                sesgo_i,
                d_i['Ppvinst'], d_i['C'], d_i['PinverterBESS'],
                d_i['PbmaxP1'], d_i['PbmaxP2'], d_i['PbmaxP3'],
                d_i['PbmaxP4'], d_i['PbmaxP5'], d_i['PbmaxP6'],
            ])
    print("  Tabla completa guardada en pareto_real_8760h.csv")

    # 9. Resultados detallados: solución de máximo NPV real
    def imprimir_resultado(idx_sel, label):
        d   = decodificar(pareto_X[idx_sel])
        r   = real_res[idx_sel]
        CF  = real_cf[idx_sel]
        inv = real_capex[idx_sel]

        npv_8760         = CF / crfe - inv
        TIR              = npf.irr([-inv] + [CF] * n) * 100 if inv > 0 else 0
        BCratio          = npf.pv(rate=i, nper=n, pmt=-CF, fv=0) / inv if inv > 0 else 0
        OPEXgross        = OaMpv * d['Ppvinst'] + OaMbess * d['C']
        Wl_val           = r['Wl']
        LCOEgross        = 1000*(inv + OPEXgross/crfe)/(Wl_val/crf) if Wl_val > 0 else 0
        LCOEnet          = 1000*(inv + (r['OPEX'] - r['OPEX0'] - r['Es'])/crfe)/(Wl_val/crf) if Wl_val > 0 else 0
        NPERaprox        = inv / CF if CF > 0 else 0
        try:
            NPER = np.log(CF / (CF + i * (-inv))) / np.log(1+i) if (CF + i*(-inv)) > 0 else 0
        except Exception:
            NPER = 0
        Crate            = d['PinverterBESS'] / d['C'] if d['C'] > 0 else 0
        nx               = 1000 * d['Ppvinst'] / (Rmax * eta * Area)
        BESSbatteryCost  = CAPEX_BESS          * d['C']
        BESSinverterCost = CAPEX_BESS_inverter * d['PinverterBESS']
        PVsystemCost     = CAPEX_pv            * d['Ppvinst']
        SOCmin_val       = ((1 - DoD) / 2) * d['C']
        SOCmax_val       = ((1 - DoD) / 2 + DoD) * d['C']
        sesgo_aprox      = (pareto_npv_proxy[idx_sel] - npv_8760) / abs(npv_8760) * 100 if npv_8760 != 0 else 0
        gap_milp         = (npv_8760 - 3_581_528) / 3_581_528 * 100
        T_list           = list(T_8760)

        print(f"\n{'='*60}")
        print(f"RESULTADOS FINALES — {label}")
        print(f"{'='*60}")
        print(f"Parámetros de diseño:")
        print(f"  BESS capacity (WbessInst):           {d['C']:>12,.2f} kWh")
        print(f"  BESS inverter capacity (PbessInst):  {d['PinverterBESS']:>12,.2f} kW")
        print(f"  PV System capacity (PpvInst):        {d['Ppvinst']:>12,.2f} kW")
        print(f"  Npan:                                {nx:>12,.2f} modules of 500W")
        print(f"  C-rate:                              {Crate:>12,.4f} 1/h")
        print(f"  Contracted power per period:")
        for p in range(1, 7):
            print(f"    PbmaxP{p}: {d[f'PbmaxP{p}']:>10,.2f} kW")
        print(f"------Verificación SOC0--------------------")
        print(f"  SOCmin  = {SOCmin_val:>10,.2f} kWh")
        print(f"  SOC0    = {r['SOC0']:>10,.2f} kWh  ← debe estar en [SOCmin, SOCmax]")
        print(f"  SOCmax  = {SOCmax_val:>10,.2f} kWh")
        print(f"  SOC[T]  = {r['SOC'][T_list[-1]]:>10,.2f} kWh  ← debe == SOC0")
        print(f"------Energy Dispatch--------------------")
        print(f"  Energy load consumption (Wl):        {r['Wl']:>12,.2f} kWh/year")
        print(f"  Energy bought from market (Wb):      {r['Wb']:>12,.2f} kWh/year")
        print(f"  Energy sold to market (Ws):          {r['Ws']:>12,.2f} kWh/year")
        print(f"  BESS Energy charged (Wc):            {r['Wc']:>12,.2f} kWh/year")
        print(f"  BESS Energy discharged (Wd):         {r['Wd']:>12,.2f} kWh/year")
        print(f"  PV energy generated (wpvmx):         {r['wpvmx']:>12,.2f} kWh/year")
        print(f"  PV energy injected (wpv):            {r['wpv']:>12,.2f} kWh/year")
        print(f"  PV energy curtailed (Wcurtail):      {r['wcurtail']:>12,.2f} kWh/year")
        print(f"------Financial results--------------------")
        print(f"  Operational Benefit:                 {r['Benefit']:>12,.2f} USD/year")
        print(f"  OPEX with project:                   {r['OPEX']:>12,.2f} USD/year")
        print(f"  OPEX without project (OPEX_0):       {r['OPEX0']:>12,.2f} USD/year")
        print(f"  Operational Savings:                 {r['Savings']:>12,.2f} USD/year")
        print(f"  Energy expenses (Eb):                {r['Eb']:>12,.2f} USD/year")
        print(f"  Energy expenses w/o project (Eb0):   {r['Eb0']:>12,.2f} USD/year")
        print(f"  Energy earnings (Es):                {r['Es']:>12,.2f} USD/year")
        print(f"  Capacity Charges (CP):               {r['CapacityP']:>12,.2f} USD/year")
        print(f"  Capacity Charges w/o project (CP0):  {r['CapacityP0']:>12,.2f} USD/year")
        print(f"  CAPEX:                               {inv:>12,.2f} USD")
        print(f"    BESS battery cost:                 {BESSbatteryCost:>12,.2f} USD")
        print(f"    BESS inverter cost:                {BESSinverterCost:>12,.2f} USD")
        print(f"    Soft Costs:                        {inv*(Sc-1):>12,.2f} USD")
        print(f"    PV system cost:                    {PVsystemCost:>12,.2f} USD")
        print(f"  Net Present Value:                   {npv_8760:>12,.2f} USD")
        print(f"  Project Cash Flow:                   {CF:>12,.2f} USD/year")
        print(f"  Internal Rate of Return:             {TIR:>12,.2f} %")
        print(f"  Pay Back Time:                       {NPER:>12,.2f} years")
        print(f"  Simple Pay Back Time:                {NPERaprox:>12,.2f} years")
        print(f"  Net LCOE:                            {LCOEnet:>12,.2f} USD/MWh")
        print(f"  Gross LCOE:                          {LCOEgross:>12,.2f} USD/MWh")
        print(f"  Benefit-Cost Ratio:                  {BCratio:>12,.2f}")
        print(f"------NSGA2+LP vs referencia-------------")
        print(f"  NPV semanas tipo (proxy):            {pareto_npv_proxy[idx_sel]:>12,.0f} USD")
        print(f"  NPV 8760h (real):                    {npv_8760:>12,.0f} USD")
        print(f"  NPV MILP referencia:                   3,581,528 USD")
        print(f"  NPV GA mono-objetivo:                  3,708,107 USD")
        print(f"  Sesgo aproximación:                  {sesgo_aprox:>+11.2f}%")
        print(f"  Gap vs MILP:                         {gap_milp:>+11.2f}%")

        return npv_8760, inv, CF

    npv_final, inv_final, CF_final = imprimir_resultado(
        idx_real_npv, "NSGA2 + LP — Máximo NPV real (8760h)"
    )
    imprimir_resultado(
        idx_real_bcr, "NSGA2 + LP — Máximo BCR real (8760h)"
    )

    # Guardar despacho de la solución de máximo NPV real
    r_best = real_res[idx_real_npv]
    with open("solucion_nsga2_lp.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Hour", "Pb", "Ps", "Pc", "Pd", "SOC", "CF", "NPV"])
        for t in T_8760:
            writer.writerow([t,
                              r_best['Pb'][t], r_best['Ps'][t],
                              r_best['Pc'][t], r_best['Pd'][t],
                              r_best['SOC'][t], CF_final, npv_final])
    print("\nDespacho horario guardado en solucion_nsga2_lp.csv")

    # 10. Gráficas: frente proxy vs frente real
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    #  Frente de Pareto: proxy vs real
    ax = axes[0]
    bcr_real = (npv_ok + capex_ok) / np.where(capex_ok > 0, capex_ok, 1e-9)
    sc = ax.scatter(capex_ok / 1e6, npv_ok / 1e6,
                    c=bcr_real, cmap='RdYlGn', s=80, zorder=3,
                    vmin=0.5, vmax=2.5, label='Frente real (8760h)')
    ax.scatter(pareto_capex_proxy / 1e6, pareto_npv_proxy / 1e6,
               c='lightgrey', s=40, zorder=2, alpha=0.6,
               marker='o', label='Proxy (semanas tipo)')
    # Flechas proxy → real para cada solución óptima
    for idx in idx_ok:
        ax.annotate('',
            xy=(real_capex[idx]/1e6, real_npv[idx]/1e6),
            xytext=(pareto_capex_proxy[idx]/1e6, pareto_npv_proxy[idx]/1e6),
            arrowprops=dict(arrowstyle='->', color='grey', lw=0.8),
        )
    ax.scatter(real_capex[idx_real_npv]/1e6, real_npv[idx_real_npv]/1e6,
               c='blue', s=200, marker='*', zorder=5, label='Máx NPV real')
    ax.scatter(real_capex[idx_real_bcr]/1e6, real_npv[idx_real_bcr]/1e6,
               c='black', s=200, marker='D', zorder=5, label='Máx BCR real')
    ax.axhline(3.581528, color='red', ls='--', lw=1.2, label='MILP ref.')
    ax.axhline(3.708107, color='orange', ls='--', lw=1.2, label='GA mono')
    fig.colorbar(sc, ax=ax, label='BCR real')
    ax.set_xlabel("CAPEX (M USD)")
    ax.set_ylabel("NPV (M USD)")
    ax.set_title("Frente de Pareto: proxy vs real\n(flechas = corrección por 8760h)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Sesgo por solución (proxy - real) / real
    ax = axes[1]
    sesgos = np.array([
        (pareto_npv_proxy[idx] - real_npv[idx]) / abs(real_npv[idx]) * 100
        if not np.isnan(real_npv[idx]) and real_npv[idx] != 0 else np.nan
        for idx in range(n_pareto)
    ])
    colores = ['steelblue' if not np.isnan(s) else 'lightgrey' for s in sesgos]
    ax.bar(range(n_pareto), np.where(np.isnan(sesgos), 0, sesgos), color=colores)
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(sesgos[~np.isnan(sesgos)].mean(), color='red', ls='--', lw=1.2,
               label=f'Sesgo medio: {sesgos[~np.isnan(sesgos)].mean():+.1f}%')
    ax.set_xlabel("Solución del frente (índice)")
    ax.set_ylabel("Sesgo proxy (%)")
    ax.set_title("Sesgo por solución\n(proxy - real) / real")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Convergencia NPV (proxy)
    ax = axes[2]
    ax.plot(callback.historial_gen,
            np.array(callback.historial_npv) / 1e6,
            color='steelblue', lw=2, label='Proxy (semanas tipo)')
    ax.axhline(3.581528, color='red', ls='--', lw=1.2, label='MILP ref.')
    ax.axhline(3.708107, color='orange', ls='--', lw=1.2, label='GA mono')
    ax.axhline(npv_final / 1e6, color='blue', ls=':', lw=1.5,
               label=f'Mejor real: {npv_final/1e6:.2f} MUSD')
    ax.set_xlabel("Generación")
    ax.set_ylabel("Mejor NPV del frente (M USD)")
    ax.set_title("Convergencia NPV — NSGA2 + LP")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("convergencia_nsga2_lp.png", dpi=150)
    plt.show()
    print("Gráfica guardada en convergencia_nsga2_lp.png")