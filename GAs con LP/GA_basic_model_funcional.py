"""
GA mono-objetivo con semanas tipo estacionales
Unidad temporal: 168h (1 semana) por estación — elimina el error de
acoplamiento SOC que ocurre con días de 24h independientes.

Arquitectura:
  - 4 semanas tipo (Primavera, Verano, Otoño, Invierno) de 168h cada una
  - Cada semana se selecciona como la más representativa de su estación
    (mínima distancia al centroide del perfil semanal de lambda y Ppvu)
  - Factor de escala: semanas reales por año por estación
  - Validación cruzada obligatoria antes del GA
"""
from __future__ import annotations

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from pymoo.parallelization.starmap import StarmapParallelization
from multiprocessing.pool import Pool

from LP_model_funcional import (
    cargar_datos_ventana,
    resolver_lp_operacion,
    Plinst, PmaxF,
    BoP, Sc, CAPEX_pv, CAPEX_BESS, CAPEX_BESS_inverter,
    crfe, crf, OaMpv, OaMbess, i, n, kappa,
    Rmax, eta, Area,
)

from pathlib import Path
import numpy as np
import numpy_financial as npf
from sklearn.preprocessing import StandardScaler
import time
import csv
import matplotlib.pyplot as plt

DIRECTORIO_VENTANA = Path(__file__).resolve().parent / 'ventana_completa'

PPV_MAX   = 8.0  * Plinst
C_MAX     = 15.0 * Plinst
RATIO_MIN = 0.1
RATIO_MAX = 2.0

# Semanas por estación en el año (52 semanas + 1 día → se distribuye)
SEMANAS_POR_ESTACION = {
    'Primavera': 13.0,
    'Verano'   : 13.0,
    'Otono'    : 13.0,
    'Invierno' : 13.0,
}  # suma = 52 semanas ≈ 364 días (el día restante se ignora — error < 0.3%)

# Rangos de horas por estación en el año (horas 1-8760)
# Primavera: mar-may (~2160-4344), Verano: jun-ago (4345-6552),
# Otoño: sep-nov (6553-8016), Invierno: dic-feb (1-2159 + 8017-8760)
# Simplificado en bloques consecutivos de ~91 días:
BLOQUES_ESTACION = {
    'Primavera': (1,    2184),   # horas 1-2184    (~91 días)
    'Verano'   : (2185, 4368),   # horas 2185-4368
    'Otono'    : (4369, 6552),   # horas 4369-6552
    'Invierno' : (6553, 8760),   # horas 6553-8760
}



# Selección de semana representativa por estación

def seleccionar_semana_tipo(data_8760: dict, periodo_8760: dict,
                            h_inicio: int, h_fin: int,
                            estacion: str) -> dict:
    """
    Dentro del bloque [h_inicio, h_fin], identifica la semana de 168h
    más representativa (más cercana al perfil medio de la estación).

    Retorna dict con 'data', 'periodo', 'T', 'n_semanas', 'estacion'.
    """
    H_SEM = 168
    horas_bloque = list(range(h_inicio, h_fin + 1))
    n_horas = len(horas_bloque)

    # Cuántas semanas completas caben en el bloque
    n_semanas_completas = n_horas // H_SEM
    if n_semanas_completas == 0:
        raise ValueError(f"Bloque {estacion} demasiado corto: {n_horas}h < 168h")

    # Construir matriz de features por semana: (n_semanas, 168*2)
    # Features: lambda y Ppvu hora a hora dentro de la semana
    semanas = []
    for s in range(n_semanas_completas):
        inicio = h_inicio + s * H_SEM
        fila_lambda = [data_8760[inicio + h]['lambda'] for h in range(H_SEM)]
        fila_ppvu   = [data_8760[inicio + h]['Ppvu']   for h in range(H_SEM)]
        semanas.append(fila_lambda + fila_ppvu)

    X = np.array(semanas)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Semana representativa = la más cercana al perfil medio
    perfil_medio = X_scaled.mean(axis=0)
    distancias   = np.linalg.norm(X_scaled - perfil_medio, axis=1)
    s_rep        = int(np.argmin(distancias))
    h_inicio_rep = h_inicio + s_rep * H_SEM

    # Extraer datos de esa semana con índice local 1..168
    data_sem    = {}
    periodo_sem = {}
    for h in range(H_SEM):
        t_orig       = h_inicio_rep + h
        t_local      = h + 1
        data_sem[t_local]    = data_8760[t_orig].copy()
        periodo_sem[t_local] = periodo_8760[t_orig]

    print(f"  {estacion}: semana {s_rep+1}/{n_semanas_completas} "
          f"(horas {h_inicio_rep}-{h_inicio_rep+H_SEM-1}) "
          f"→ representa {SEMANAS_POR_ESTACION[estacion]:.1f} semanas/año")

    return {
        'data'      : data_sem,
        'periodo'   : periodo_sem,
        'T'         : range(1, H_SEM + 1),
        'n_semanas' : SEMANAS_POR_ESTACION[estacion],
        'estacion'  : estacion,
    }


def construir_semanas_tipo(data_8760: dict,
                           periodo_8760: dict) -> list[dict]:
    print("Seleccionando semanas tipo por estación...")
    semanas = []
    for estacion, (h_ini, h_fin) in BLOQUES_ESTACION.items():
        sem = seleccionar_semana_tipo(
            data_8760, periodo_8760, h_ini, h_fin, estacion
        )
        semanas.append(sem)

    total_semanas = sum(s['n_semanas'] for s in semanas)
    print(f"  Total semanas: {total_semanas:.1f} (año ≈ 52 semanas)")
    return semanas



# Evaluación sobre semanas tipo

def evaluar_diseno_semanas(diseno: dict,
                           semanas_tipo: list[dict]) -> tuple[float, str]:
    """
    CashFlow anual = Σ_estacion [ n_semanas_estacion × CashFlow_semana ]

    CashFlow_semana ya incluye OPEX y CapacityP proporcionales a 168h.
    Pero CapacityP y OaM son anuales fijos — hay que reconstruirlos.

    Para evitar el error de escala: extraemos solo Es y Eb de cada semana
    y sumamos CapacityP y OaM una sola vez al año.
    """
    Es_anual  = 0.0
    Eb_anual  = 0.0
    Eb0_anual = 0.0

    for sem in semanas_tipo:
        horas_año = sem['n_semanas'] * 168   # horas reales que representa

        cf, resultado, status = resolver_lp_operacion(
            diseno,
            sem['data'],
            sem['periodo'],
            sem['T'],
        )

        if status != 'optimal':
            return -1e9, f"infeasible_{sem['estacion']}"

        # Escalar los flujos de energía al año completo
        # resultado['Es'] y 'Eb' están en USD/168h → escalar a USD/año
        Es_anual  += resultado['Es']  * sem['n_semanas']
        Eb_anual  += resultado['Eb']  * sem['n_semanas']
        Eb0_anual += resultado['Eb0'] * sem['n_semanas']

    # Costos fijos anuales — una sola vez
    CapacityP  = sum(kappa[p] * diseno[f'PbmaxP{p}'] for p in range(1, 7))
    CapacityP0 = sum(kappa[p] * PmaxF               for p in range(1, 7))
    OaM        = OaMpv * diseno['Ppvinst'] + OaMbess * diseno['C']

    OPEX  = CapacityP  + Eb_anual  + OaM
    OPEX0 = CapacityP0 + Eb0_anual

    cashflow_anual = Es_anual + OPEX0 - OPEX
    return cashflow_anual, 'optimal'



# Validación cruzada

def validar_aproximacion(diseno_ref: dict, semanas_tipo: list[dict],
                         data_8760: dict, periodo_8760: dict,
                         T_8760) -> float:
    cf_sem, status_sem = evaluar_diseno_semanas(diseno_ref, semanas_tipo)
    cf_8760, _, status_8760 = resolver_lp_operacion(
        diseno_ref, data_8760, periodo_8760, T_8760
    )

    print(f"\nValidación de aproximación (diseño MILP óptimo):")
    print(f"  CashFlow semanas tipo: {cf_sem:>12,.0f} USD/año  (status: {status_sem})")
    print(f"  CashFlow 8760h:        {cf_8760:>12,.0f} USD/año  (status: {status_8760})")

    if status_8760 == 'optimal' and cf_8760 != 0:
        error = abs(cf_sem - cf_8760) / abs(cf_8760) * 100
        print(f"  Error relativo:        {error:.2f}%")
        sesgo = (cf_sem - cf_8760) / cf_8760 * 100
        print(f"  Sesgo (+ = sobreestima): {sesgo:+.2f}%")
        if error > 15:
            print("  ADVERTENCIA: error > 15%")
        else:
            print("  OK: error aceptable para el GA")
        return error
    return float('inf')



# Callback

class RegistrarConvergencia(Callback):
    def __init__(self, period: int = 15, tol: float = 1.0):
        super().__init__()
        self.historial_npv = []
        self.historial_gen = []
        self.t0            = time.time()
        self.period        = period
        self.tol           = tol
        self._sin_mejora   = 0
        self._mejor_npv    = -np.inf

    def notify(self, algorithm):
        gen     = algorithm.n_gen
        npv     = -float(algorithm.opt.get("F")[0][0])
        elapsed = time.time() - self.t0
        self.historial_npv.append(npv)
        self.historial_gen.append(gen)

        if npv - self._mejor_npv > self.tol:
            self._sin_mejora = 0
            self._mejor_npv  = npv
        else:
            self._sin_mejora += 1

        print(f"Gen {gen:4d} | NPV: {npv:>14,.0f} USD | "
              f"sin mejora: {self._sin_mejora:3d}/{self.period} | "
              f"t: {elapsed:.1f}s")

        if self._sin_mejora >= self.period:
            print(f"\nParada temprana: {self.period} gen sin mejora")
            algorithm.termination.force_termination = True



# Problema pymoo

class ProblemaPVBESS(ElementwiseProblem):

    def __init__(self, semanas_tipo: list[dict], **kwargs):
        super().__init__(
            n_var=8, n_obj=1, n_ieq_constr=0,
            xl=np.array([0.,      0.,    RATIO_MIN,
                         0.,      0.,    0.,    0.,    0.]),
            xu=np.array([PPV_MAX, C_MAX, RATIO_MAX,
                         PmaxF,   PmaxF, PmaxF, PmaxF, PmaxF]),
            **kwargs,
        )
        self.semanas_tipo = semanas_tipo

    def _evaluate(self, x, out):
        diseno = self.decodificar(x)
        if diseno['C'] < 1e-3:
            out["F"] = [1e12]
            return

        cashflow, status = evaluar_diseno_semanas(diseno, self.semanas_tipo)
        if status != 'optimal':
            out["F"] = [1e12]
            return

        investment = BoP + Sc * (
            CAPEX_pv            * diseno['Ppvinst']
            + CAPEX_BESS        * diseno['C']
            + CAPEX_BESS_inverter * diseno['PinverterBESS']
        )
        out["F"] = [-(cashflow / crfe - investment)]

    @staticmethod
    def decodificar(x: np.ndarray) -> dict:
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


# Main

if __name__ == "__main__":

    # ── 1. Cargar datos 
    print("Cargando datos 8760h...")
    data_8760, periodo_8760, T_8760 = cargar_datos_ventana(DIRECTORIO_VENTANA)

    # ── 2. Construir semanas tipo 
    semanas_tipo = construir_semanas_tipo(data_8760, periodo_8760)

    # ── 3. Validación obligatoria 
    diseno_milp = {
        'Ppvinst': 5991.87, 'C': 9568.53, 'PinverterBESS': 1999.61,
        'PbmaxP1': 0., 'PbmaxP2': 0., 'PbmaxP3': 0.,
        'PbmaxP4': 0., 'PbmaxP5': 0., 'PbmaxP6': PmaxF,
    }
    error_pct = validar_aproximacion(
        diseno_milp, semanas_tipo, data_8760, periodo_8760, T_8760
    )
    if error_pct > 20:
        print(f"\nERROR: aproximación demasiado imprecisa ({error_pct:.1f}%)")
        raise SystemExit(1)

    # ── 4. Tiempo por evaluación 
    t0 = time.time()
    cf_ref, _ = evaluar_diseno_semanas(diseno_milp, semanas_tipo)
    t_eval = time.time() - t0
    inv_ref = BoP + Sc * (CAPEX_pv*diseno_milp['Ppvinst']
                          + CAPEX_BESS*diseno_milp['C']
                          + CAPEX_BESS_inverter*diseno_milp['PinverterBESS'])
    print(f"\nTiempo por evaluación: {t_eval:.2f}s")
    print(f"NPV semanas tipo (ref): {cf_ref/crfe - inv_ref:,.0f} USD")

    # ── 5. GA 
    POP_SIZE  = 40
    N_MAX_GEN = 30
    PERIOD    = 15
    N_CORES   = 4

    t_est = t_eval * POP_SIZE * N_MAX_GEN / N_CORES / 60
    print(f"Tiempo estimado GA: {t_est:.1f} min")

    pool    = Pool(N_CORES)
    runner  = StarmapParallelization(pool.starmap)
    problema = ProblemaPVBESS(semanas_tipo, elementwise_runner=runner)

    algoritmo = GA(
        pop_size=POP_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    criterio = get_termination("n_gen", N_MAX_GEN)
    callback = RegistrarConvergencia(period=PERIOD, tol=1.0)

    print("\nIniciando GA...")
    t0 = time.time()
    resultado_ga = minimize(
        problema, algoritmo,
        termination=criterio,
        seed=42, verbose=False, callback=callback,
    )
    t_ga = time.time() - t0
    pool.close(); pool.join()

    # ── 6. Resultado 
    mejor_x      = resultado_ga.X
    mejor_npv_dt = -float(resultado_ga.F[0])
    diseno       = ProblemaPVBESS.decodificar(mejor_x)

    print(f"\n{'='*55}")
    print(f"GA: {t_ga/60:.1f} min, {resultado_ga.algorithm.n_gen} gen")
    print(f"NPV semanas tipo: {mejor_npv_dt:,.0f} USD")
    for k, v in diseno.items():
        print(f"  {k:<20}: {v:>10,.2f}")

    # ── 7. Validación 8760h 
    print("\nValidando sobre 8760h completas...")
    t0 = time.time()
    cf_8760, res_8760, st_8760 = resolver_lp_operacion(
        diseno, data_8760, periodo_8760, T_8760
    )
    print(f"  Completado en {time.time()-t0:.1f}s  (status: {st_8760})")

    if st_8760 != 'optimal':
        print(f"  Diseño infactible en 8760h ({st_8760})")
    else:
        r  = res_8760
        d  = diseno
        CF = cf_8760

        inv = BoP + Sc * (
            CAPEX_pv            * d['Ppvinst']
            + CAPEX_BESS        * d['C']
            + CAPEX_BESS_inverter * d['PinverterBESS']
        )
        npv_8760      = CF / crfe - inv
        TIR           = npf.irr([-inv] + [CF] * n) * 100 if inv > 0 else 0
        BCratio       = npf.pv(rate=i, nper=n, pmt=-CF, fv=0) / inv if inv > 0 else 0
        OPEXgross     = OaMpv * d['Ppvinst'] + OaMbess * d['C']
        Wl_val        = r['Wl']
        LCOEgross     = 1000*(inv + OPEXgross/crfe)/(Wl_val/crf) if Wl_val > 0 else 0
        LCOEnet       = 1000*(inv + (r['OPEX'] - r['OPEX0'] - r['Es'])/crfe)/(Wl_val/crf) if Wl_val > 0 else 0
        NPERaprox     = inv / CF if CF > 0 else 0
        try:
            NPER = np.log(CF / (CF + i*(-inv))) / np.log(1+i) if (CF + i*(-inv)) > 0 else 0
        except:
            NPER = 0
        Crate            = d['PinverterBESS'] / d['C'] if d['C'] > 0 else 0
        ciclos_anuales = r['Wc'] / d['C'] if d['C'] > 0 else 0
        nx               = 1000 * d['Ppvinst'] / (Rmax * eta * Area)
        BESSbatteryCost  = CAPEX_BESS          * d['C']
        BESSinverterCost = CAPEX_BESS_inverter * d['PinverterBESS']
        PVsystemCost     = CAPEX_pv            * d['Ppvinst']

        sesgo_aprox = (mejor_npv_dt - npv_8760) / abs(npv_8760) * 100
        gap_milp    = (npv_8760 - 3_581_528) / 3_581_528 * 100

        print(f"\nResults:")
        print(f"BESS capacity (WbessInst): {d['C']:,.2f} kWh")
        print(f"BESS inverter capacity (PbessInst): {d['PinverterBESS']:,.2f} kW")
        print(f"PV System capacity (PpvInst): {d['Ppvinst']:,.2f} kW")
        print(f"Npan: {nx:,.2f} modules of 500W")
        print(f"Contracted power per period")
        for p in range(1, 7):
            print(f"PbmaxP{p}: {d[f'PbmaxP{p}']:,.2f} kW")
        print(f"------Energy Dispatch--------------------")
        print(f"Energy load consumption (Wl): {r['Wl']:,.2f} kWh/year")
        print(f"Energy bought from the market (Wb): {r['Wb']:,.2f} kWh/year")
        print(f"Energy sold to the market (Ws): {r['Ws']:,.2f} kWh/year")
        print(f"BESS Energy charged (Wc): {r['Wc']:,.2f} kWh/year")
        print(f"BESS Energy discharged (Wd): {r['Wd']:,.2f} kWh/year")
        print(f"PV energy generated (wpvmx): {r['wpvmx']:,.2f} kWh/year")
        print(f"PV energy injected (wpv): {r['wpv']:,.2f} kWh/year")
        print(f"PV energy curtailed (Wcurtail): {r['wcurtail']:,.2f} kWh/year")
        print(f"Initial SOC:  {r['SOC0']:,.2f} kWh")
        print(f"BESS C-rate:  {Crate:,.2f} 1/h")
        print(f"BESS Cycles per year:               {ciclos_anuales:>12,.1f} ciclos/año")
        print(f"------Financial results--------------------")
        print(f"Operational Benefit: {r['Benefit']:,.2f} USD/year")
        print(f"Operational Expenditure with the project (OPEX): {r['OPEX']:,.2f} USD/year")
        print(f"Operational Expenditure without the project (OPEX_0): {r['OPEX0']:,.2f} USD/year")
        print(f"Operational Savings (Savings): {r['Savings']:,.2f} USD/year")
        print(f"Energy expenses with the project (Eb): {r['Eb']:,.2f} USD/year")
        print(f"Energy expenses without the project (Eb0): {r['Eb0']:,.2f} USD/year")
        print(f"Energy earnings of the project (Es): {r['Es']:,.2f} USD/year")
        print(f"Capacity Charges (CP): {r['CapacityP']:,.2f} USD/year")
        print(f"Capacity Charges without the project (CP0): {r['CapacityP0']:,.2f} USD/year")
        print(f"Capital Expenditure (CAPEX): {inv:,.2f} USD")
        print(f"BESS battery cost: {BESSbatteryCost:,.2f} USD")
        print(f"BESS inverter cost: {BESSinverterCost:,.2f} USD")
        print(f"Soft Costs: {inv*(Sc-1):,.2f} USD")
        print(f"PV system cost: {PVsystemCost:,.2f} USD")
        print(f"Net Present Value: {npv_8760:,.2f} USD")
        print(f"Project Cash Flow: {CF:,.2f} USD/year")
        print(f"Internal Rate of Return:      {TIR:,.2f} %")
        print(f"Pay Back Time:     {NPER:,.2f} years")
        print(f"Simple Pay Back Time:    {NPERaprox:,.2f} years")
        print(f"Net LCOE (include earnings and savings):     {LCOEnet:,.2f} USD/MWh")
        print(f"Gross LCOE: (only CAPEX and OPEX)    {LCOEgross:,.2f} USD/MWh")
        print(f"Benefit-Cost Ratio {BCratio:,.2f}")
        print(f"------GA vs MILP--------------------")
        print(f"NPV semanas tipo (GA proxy): {mejor_npv_dt:,.2f} USD")
        print(f"NPV 8760h (GA real):         {npv_8760:,.2f} USD")
        print(f"NPV MILP referencia:         3,581,528.00 USD")
        print(f"Sesgo aproximación:          {sesgo_aprox:+.2f}%")
        print(f"Gap GA vs MILP:              {gap_milp:+.2f}%")

        with open("solucion_ga.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Hour","Pb","Ps","Pc","Pd","SOC","Benefit","npv"])
            for t in T_8760:
                writer.writerow([t, r['Pb'][t], r['Ps'][t],
                                  r['Pc'][t], r['Pd'][t],
                                  r['SOC'][t], r['Benefit'], npv_8760])
        print("Solution written to solucion_ga.csv")

    # ── 8. Convergencia 
    plt.figure(figsize=(10, 4))
    plt.plot(callback.historial_gen, callback.historial_npv, linewidth=2)
    plt.axhline(y=3_581_528, color='red', linestyle='--',
                linewidth=1, label='MILP óptimo')
    plt.xlabel("Generación"); plt.ylabel("NPV (USD)")
    plt.title("Convergencia GA — semanas tipo estacionales")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("convergencia_ga.png", dpi=150); plt.show()