from pathlib import Path
import re
import numpy as np
import numpy_financial as npf
import gurobipy as gp
from gurobipy import GRB, quicksum

BASE_DIR           = Path(__file__).resolve().parent
DIRECTORIO_VENTANA = BASE_DIR / 'ventana_completa'

# Parámetros técnicos
Plinst  = 1000.0
Rmax    = 1000
Area    = 2.4
eta     = 0.2094
eff_c   = 0.9624
eff_d   = 0.9624
eff_pv  = 0.9624
DoD     = 0.90
PmaxF   = Plinst

er      = 1.1
BoP     = 0
Sc      = 1.2
OaMpv   = 12.5
OaMbess = 5.9
CAPEX_pv             = 436
CAPEX_BESS           = 185
CAPEX_BESS_inverter  = 48
i    = 7.7  / 100
n    = 20
e    = 2.5  / 100
ir   = (i - e) / (1 + e)
crf  = (i * (i + 1)**n) / ((i + 1)**n - 1)
crfe = (1 + e) * (ir * (ir + 1)**n) / ((ir + 1)**n - 1)

kappa = [0,
         28.79187 * er,
         15.07764 * er,
          6.55917 * er,
          5.17209 * er,
          1.93281 * er,
          0.91609 * er]


# Lectura de archivos
def read_inc(path):
    valores = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.lstrip().startswith("t"):
                continue
            partes = re.split(r"\s+", line.strip())
            if len(partes) >= 2:
                hora  = int(partes[0][1:])
                valor = float(partes[1])
                valores[hora] = valor
    return valores


def cargar_datos_ventana(directorio):
    paths = {
        'lambda':  directorio / 'lambda.inc',
        'psi':     directorio / 'psi.inc',
        'Ppvu':    directorio / 'PpvuMadridSarah20052023.inc',
        'Plu':     directorio / 'PluDataCenter.inc',
        'periodo': directorio / 'periodo.inc',
    }
    series   = {nombre: read_inc(ruta) for nombre, ruta in paths.items()}
    T_ventana = range(1, 8761)
    data_ventana = {
        t: {
            'lambda': series['lambda'].get(t, 0.0),
            'psi':    series['psi'].get(t, 0.0),
            'Ppvu':   series['Ppvu'].get(t, 0.0),
            'Plu':    series['Plu'].get(t, 0.0),
        }
        for t in T_ventana
    }
    periodo_ventana = {t: int(series['periodo'].get(t, 6)) for t in T_ventana}
    return data_ventana, periodo_ventana, T_ventana


# LP de operación
def resolver_lp_operacion(diseno, data_ventana, periodo_ventana, T_ventana):

    Ppvinst       = diseno['Ppvinst']
    C             = diseno['C']
    PinverterBESS = diseno['PinverterBESS']
    Pbmax = {p: diseno[f'PbmaxP{p}'] for p in range(1, 7)}

    SOCmin = ((1 - DoD) / 2)       * C
    SOCmax = ((1 - DoD) / 2 + DoD) * C

    m = gp.Model('LP_operacion')
    m.setParam('OutputFlag', 0)

    SOC = {t: m.addVar(lb=SOCmin, ub=SOCmax * 1.01, name=f'SOC[{t}]') for t in T_ventana}
    SOC0 = m.addVar(lb=SOCmin, ub=SOCmax * 1.01, name='SOC0')
    #m.addConstr(SOC0 == 9097.15)
    #SOC0 = m.addVar(lb=SOCmin, ub=SOCmax, name='SOC0')
    #SOC = {t: m.addVar(lb=SOCmin, ub=SOCmax, name=f'SOC[{t}]') for t in T_ventana}
    Ppv   = {t: m.addVar(lb=0,             name=f'Ppv[{t}]')   for t in T_ventana}
    Ppvmx = {t: m.addVar(lb=0,             name=f'Ppvmx[{t}]') for t in T_ventana}
    Pd    = {t: m.addVar(lb=0,             name=f'Pd[{t}]')    for t in T_ventana}
    Pc    = {t: m.addVar(lb=0,             name=f'Pc[{t}]')    for t in T_ventana}
    Pb    = {t: m.addVar(lb=0,             name=f'Pb[{t}]')    for t in T_ventana}
    Ps    = {t: m.addVar(lb=0,             name=f'Ps[{t}]')    for t in T_ventana}

    Es         = m.addVar(lb=0,             name='Es')
    Eb         = m.addVar(lb=0,             name='Eb')
    Eb0        = m.addVar(lb=0,             name='Eb0')
    CapacityP  = m.addVar(lb=0,             name='CapacityP')
    CapacityP0 = m.addVar(lb=0,             name='CapacityP0')
    Investment0 = m.addVar(lb=0,             name='Investment0')
    OPEX       = m.addVar(lb=0,             name='OPEX')
    OPEX0      = m.addVar(lb=-GRB.INFINITY, name='OPEX0')
    Savings    = m.addVar(lb=-GRB.INFINITY, name='Savings')
    Benefit    = m.addVar(lb=-GRB.INFINITY, name='Benefit')
    CashFlow_var = m.addVar(lb=-GRB.INFINITY, name='CashFlow_var')
    npv_var    = m.addVar(lb=-GRB.INFINITY, name='npv_var')
    wpv        = m.addVar(lb=-GRB.INFINITY, name='wpv')
    wpvmx      = m.addVar(lb=-GRB.INFINITY, name='wpvmx')
    wcurtail   = m.addVar(lb=-GRB.INFINITY, name='wcurtail')
    Wb         = m.addVar(lb=-GRB.INFINITY, name='Wb')
    Ws         = m.addVar(lb=-GRB.INFINITY, name='Ws')
    Wc         = m.addVar(lb=-GRB.INFINITY, name='Wc')
    Wd         = m.addVar(lb=-GRB.INFINITY, name='Wd')
    Wl         = m.addVar(lb=-GRB.INFINITY, name='Wl')

    for t in T_ventana:
        p    = periodo_ventana[t]
        PL_t = Plinst * data_ventana[t]['Plu']

        m.addConstr(Pc[t] + Pd[t] <= PinverterBESS)  # no se puede cargar o descargar más allá de la capacidad del inversor
        m.addConstr(Pb[t] + Ps[t] <= PmaxF)           # no se puede comprar o vender más allá de la potencia contratada
        m.addConstr(Pd[t] + Pb[t] + Ppv[t] == Pc[t] + Ps[t] + PL_t)
        m.addConstr(Ppvmx[t] == Ppvinst * eff_pv * data_ventana[t]['Ppvu'])
        m.addConstr(Ppv[t]   <= Ppvmx[t])
        m.addConstr(Pc[t]    <= PinverterBESS)
        m.addConstr(Pd[t]    <= PinverterBESS)
        m.addConstr(Pb[t]    <= PmaxF)
        m.addConstr(Ps[t]    <= PmaxF)
        m.addConstr(Pb[t]    <= Pbmax[p])

        if t == T_ventana[0]:
            m.addConstr(SOC[t] == SOC0 + Pc[t] * eff_c - Pd[t] / eff_d)
        else:
            m.addConstr(SOC[t] == SOC[t-1] + Pc[t] * eff_c - Pd[t] / eff_d)
    
    for p in range(1, 6):
        m.addConstr(Pbmax[p+1] >= Pbmax[p])

    m.addConstr(SOC[T_ventana[-1]] == SOC0)

    m.addConstr(Es  == er * quicksum(data_ventana[t]['lambda'] * Ps[t] for t in T_ventana))
    m.addConstr(Eb  == er * quicksum((data_ventana[t]['lambda'] + data_ventana[t]['psi']) * Pb[t] for t in T_ventana))
    m.addConstr(Eb0 == er * quicksum((data_ventana[t]['lambda'] + data_ventana[t]['psi']) * Plinst * data_ventana[t]['Plu'] for t in T_ventana))
    m.addConstr(CapacityP  == quicksum(kappa[p] * Pbmax[p] for p in range(1, 7)))
    m.addConstr(CapacityP0 == sum(kappa[p] * PmaxF for p in range(1, 7)))
    m.addConstr(OPEX0 == Eb0 + CapacityP0)
    m.addConstr(OPEX == CapacityP + Eb + OaMpv*Ppvinst + OaMbess*C)
    m.addConstr(Savings == OPEX0 - OPEX)
    m.addConstr(Benefit == Es - OPEX)
    m.addConstr(CashFlow_var == Es + OPEX0 - OPEX)
    m.addConstr(Investment0 == BoP + Sc * (CAPEX_pv*Ppvinst + CAPEX_BESS*C + CAPEX_BESS_inverter*PinverterBESS))
    m.addConstr(npv_var == (CashFlow_var/(crfe) - Investment0))
    m.addConstr(wpv      == quicksum(Ppv[t]   for t in T_ventana))
    m.addConstr(wpvmx    == (1 / eff_pv) * quicksum(Ppvmx[t] for t in T_ventana))
    m.addConstr(wcurtail == wpvmx / eff_pv - wpv)
    m.addConstr(Wb       == quicksum(Pb[t] for t in T_ventana))
    m.addConstr(Ws       == quicksum(Ps[t] for t in T_ventana))
    m.addConstr(Wc       == quicksum(Pc[t] for t in T_ventana))
    m.addConstr(Wd       == quicksum(Pd[t] for t in T_ventana))
    m.addConstr(Wl       == sum(Plinst * data_ventana[t]['Plu'] for t in T_ventana))

    eps = 1e-4
    
    m.setObjective(npv_var, GRB.MAXIMIZE)
    m.setParam('DualReductions', 0)
    m.optimize()


    if m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write("iis_debug.ilp")
        # Leer el IIS y mostrar las restricciones conflictivas
        print("\nRestricciones en el IIS:")
        for c in m.getConstrs():
            if c.IISConstr:
                print(f"  {c.ConstrName}")
        print("\nVariables en el IIS:")
        for v in m.getVars():
            if v.IISLB or v.IISUB:
                print(f"  {v.VarName}: lb={v.lb}, ub={v.ub}")
        print("Periodos de las últimas horas:")
        for t in range(8750, 8761):
            print(f"  t={t}: periodo={periodo_ventana[t]}, Ppvu={data_ventana[t]['Ppvu']:.4f}, Plu={data_ventana[t]['Plu']:.4f}")

    if m.Status == GRB.OPTIMAL:
        
        resultado = {
            'SOC'       : {t: SOC[t].X   for t in T_ventana},
            'Ppv'       : {t: Ppv[t].X   for t in T_ventana},
            'Ppvmx'     : {t: Ppvmx[t].X for t in T_ventana},
            'Pc'        : {t: Pc[t].X    for t in T_ventana},
            'Pd'        : {t: Pd[t].X    for t in T_ventana},
            'Pb'        : {t: Pb[t].X    for t in T_ventana},
            'Ps'        : {t: Ps[t].X    for t in T_ventana},
            'SOC0'      : SOC0.X,
            'Es'        : Es.X,
            'Eb'        : Eb.X,
            'Eb0'       : Eb0.X,
            'CapacityP' : CapacityP.X,
            'CapacityP0': CapacityP0.X,
            'OPEX'      : OPEX.X,
            'OPEX0'     : OPEX0.X,
            'Savings'   : Savings.X,
            'Benefit'   : Benefit.X,
            'CashFlow'  : CashFlow_var.X,
            'npv'       : npv_var.X,
            'wpv'       : wpv.X,
            'wpvmx'     : wpvmx.X,
            'wcurtail'  : wcurtail.X,
            'Wb'        : Wb.X,
            'Ws'        : Ws.X,
            'Wc'        : Wc.X,
            'Wd'        : Wd.X,
            'Wl'        : Wl.X,
        }

        return npv_var.X, resultado, 'optimal'

    elif m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write("iis_debug.ilp")
        return -1e9, {}, 'infeasible'
    else:
        return -1e9, {}, f'status_{m.Status}'


# Bloque de prueba

if __name__ == "__main__":

    data_ventana, periodo_ventana, T_ventana = cargar_datos_ventana(
        DIRECTORIO_VENTANA
    )

    diseno_prueba = {
        'Ppvinst'      : 5989.33,
        'C'            : 9575.95,
        'PinverterBESS': 1995.54,
        'PbmaxP1'      : 0.0,
        'PbmaxP2'      : 0.0,
        'PbmaxP3'      : 0.0,
        'PbmaxP4'      : 0.0,
        'PbmaxP5'      : 0.0,
        'PbmaxP6'      : PmaxF,
    }

    cashflow, resultado, status = resolver_lp_operacion(
        diseno_prueba, data_ventana, periodo_ventana, T_ventana
    )

    if status != 'optimal':
        print(f"Sin solución óptima, estado: {status}")
    else:
        r  = resultado
        d  = diseno_prueba

        # ── Métricas financieras post-optimización (igual que el MINLP)
        Io = BoP + Sc * (
            CAPEX_pv           * d['Ppvinst']
            + CAPEX_BESS       * d['C']
            + CAPEX_BESS_inverter * d['PinverterBESS']
        )
        npv_val = resultado['npv'] 
        CF = (npv_val + Io) * crfe
        TIR      = npf.irr([-Io] + [CF] * n) * 100 if Io > 0 else 0
        BCratio  = npf.pv(rate=i, nper=n, pmt=-CF, fv=0) / Io if Io > 0 else 0
        OPEXgross = OaMpv * d['Ppvinst'] + OaMbess * d['C']
        Wl_val   = r['Wl']
        LCOEgross = 1000 * (Io + OPEXgross / crfe) / (Wl_val / crf) if Wl_val > 0 else 0
        LCOEnet   = 1000 * (Io + (r['OPEX'] - r['OPEX0'] - r['Es']) / crfe) / (Wl_val / crf) if Wl_val > 0 else 0
        NPERaprox = Io / CF if CF > 0 else 0
        try:
            NPER = np.log((CF) / (CF + i * (-Io))) / np.log(1 + i) if (CF + i * (-Io)) > 0 else 0
        except:
            NPER = 0

        Crate    = d['PinverterBESS'] / d['C']
        nx       = 1000 * d['Ppvinst'] / (Rmax * eta * Area)

        BESSbatteryCost  = CAPEX_BESS          * d['C']
        BESSinverterCost = CAPEX_BESS_inverter * d['PinverterBESS']
        PVsystemCost     = CAPEX_pv            * d['Ppvinst']

        # ── Parámetros de diseño (fijos — no optimizados por el LP)
        print(f"Parámetros de diseño (fijos):")
        print(f"BESS capacity (WbessInst):          {d['C']:>12,.2f} kWh")
        print(f"BESS inverter capacity (PbessInst): {d['PinverterBESS']:>12,.2f} kW")
        print(f"PV System capacity (PpvInst):       {d['Ppvinst']:>12,.2f} kW")
        print(f"Npan:                               {nx:>12,.2f} modules of 500W")
        print(f"Contracted power per period")
        for p in range(1, 7):
            print(f"  PbmaxP{p}: {d[f'PbmaxP{p}']:>10,.2f} kW")

        # ── Despacho de energía
        print(f"------Energy Dispatch--------------------")
        print(f"Energy load consumption (Wl):       {r['Wl']:>12,.2f} kWh/year")
        print(f"Energy bought from the market (Wb): {r['Wb']:>12,.2f} kWh/year")
        print(f"Energy sold to the market (Ws):     {r['Ws']:>12,.2f} kWh/year")
        print(f"BESS Energy charged (Wc):           {r['Wc']:>12,.2f} kWh/year")
        print(f"BESS Energy discharged (Wd):        {r['Wd']:>12,.2f} kWh/year")
        print(f"PV energy generated (wpvmx):        {r['wpvmx']:>12,.2f} kWh/year")
        print(f"PV energy injected (wpv):           {r['wpv']:>12,.2f} kWh/year")
        print(f"PV energy curtailed (Wcurtail):     {r['wcurtail']:>12,.2f} kWh/year")
        print(f"Initial SOC:                        {r['SOC0']:>12,.2f} kWh")
        print(f"BESS C-rate:                        {Crate:>12,.2f} 1/h")

        # ── Resultados financieros
        print(f"------Financial results--------------------")
        print(f"Operational Benefit:                          {r['Benefit']:>12,.2f} USD/year")
        print(f"Operational Expenditure with project (OPEX):  {r['OPEX']:>12,.2f} USD/year")
        print(f"Operational Expenditure without project (OPEX_0): {r['OPEX0']:>12,.2f} USD/year")
        print(f"Operational Savings (Savings):                {r['Savings']:>12,.2f} USD/year")
        print(f"Energy expenses with project (Eb):            {r['Eb']:>12,.2f} USD/year")
        print(f"Energy expenses without project (Eb0):        {r['Eb0']:>12,.2f} USD/year")
        print(f"Energy earnings of the project (Es):          {r['Es']:>12,.2f} USD/year")
        print(f"Capacity Charges (CP):                        {r['CapacityP']:>12,.2f} USD/year")
        print(f"Capacity Charges without project (CP0):       {r['CapacityP0']:>12,.2f} USD/year")
        print(f"Capital Expenditure (CAPEX):                  {Io:>12,.2f} USD")
        print(f"BESS battery cost:                            {BESSbatteryCost:>12,.2f} USD")
        print(f"BESS inverter cost:                           {BESSinverterCost:>12,.2f} USD")
        print(f"Soft Costs:                                   {Io*(Sc-1)/Sc:>12,.2f} USD")
        print(f"PV system cost:                               {PVsystemCost:>12,.2f} USD")
        print(f"Net Present Value:                            {npv_val:>12,.2f} USD")
        print(f"Project Cash Flow:                            {CF:>12,.2f} USD/year")
        print(f"Internal Rate of Return:                      {TIR:>12,.2f} %")
        print(f"Pay Back Time:                                {NPER:>12,.2f} years")
        print(f"Simple Pay Back Time:                         {NPERaprox:>12,.2f} years")
        print(f"Net LCOE (earnings and savings):              {LCOEnet:>12,.2f} USD/MWh")
        print(f"Gross LCOE (only CAPEX and OPEX):             {LCOEgross:>12,.2f} USD/MWh")
        print(f"Benefit-Cost Ratio:                           {BCratio:>12,.2f}")

        # Verificar si hay ciclos simultáneos de carga y descarga en BESS o compra y venta en red
        cycling_c_d = sum(
            min(r['Pc'][t], r['Pd'][t]) for t in T_ventana
        )
        cycling_b_s = sum(
            min(r['Pb'][t], r['Ps'][t]) for t in T_ventana
        )
        # Pruebas para verificar congruencia con MINLP
        print(f"Cycling BESS (Pc∧Pd simultáneo): {cycling_c_d:,.2f} kWh")
        print(f"Cycling red  (Pb∧Ps simultáneo): {cycling_b_s:,.2f} kWh")

        print(f"SOC0 LP: {r['SOC0']:,.2f} kWh")
        print(f"SOC0 MINLP: 9,097.15 kWh")

        # Comparar con valores del MINLP
        Wb_MINLP = 2_426_098.39
        Ws_MINLP = 2_068_566.56
        Wc_MINLP = 3_252_831.23
        Wd_MINLP = 3_023_104.53
        Es_MINLP =   131_291.47
        Eb_MINLP =   230_264.19
        CF_MINLP =   741_184.67

        print(f"\n{'='*55}")
        print(f"COMPARACIÓN LP vs MINLP")
        print(f"{'='*55}")
        print(f"{'Variable':<12} {'MINLP':>14} {'LP':>14} {'Δ':>14}")
        print(f"{'-'*55}")
        print(f"{'Wb (kWh)':<12} {Wb_MINLP:>14,.2f} {r['Wb']:>14,.2f} {r['Wb']-Wb_MINLP:>+14,.2f}")
        print(f"{'Ws (kWh)':<12} {Ws_MINLP:>14,.2f} {r['Ws']:>14,.2f} {r['Ws']-Ws_MINLP:>+14,.2f}")
        print(f"{'Wc (kWh)':<12} {Wc_MINLP:>14,.2f} {r['Wc']:>14,.2f} {r['Wc']-Wc_MINLP:>+14,.2f}")
        print(f"{'Wd (kWh)':<12} {Wd_MINLP:>14,.2f} {r['Wd']:>14,.2f} {r['Wd']-Wd_MINLP:>+14,.2f}")
        print(f"{'Es (USD)':<12} {Es_MINLP:>14,.2f} {r['Es']:>14,.2f} {r['Es']-Es_MINLP:>+14,.2f}")
        print(f"{'Eb (USD)':<12} {Eb_MINLP:>14,.2f} {r['Eb']:>14,.2f} {r['Eb']-Eb_MINLP:>+14,.2f}")
        print(f"{'CF (USD)':<12} {CF_MINLP:>14,.2f} {CF:>14,.2f} {CF-CF_MINLP:>+14,.2f}")
        print(f"crfe = {crfe:.8f}")
        print(f"crf  = {crf:.8f}")
        print(f"ir   = {ir:.8f}")

        # Verificar balance horario
        balance_errors = sum(
            abs(r['Pd'][t] + r['Pb'][t] + r['Ppv'][t] - r['Pc'][t] - r['Ps'][t]
                - Plinst * data_ventana[t]['Plu'])
            for t in T_ventana
        )
        print(f"\nError balance total?: {balance_errors:.6f} kWh")

        # Detectar horas con simultaneidad residual
        sim_cd = [(t, r['Pc'][t], r['Pd'][t])
                for t in T_ventana
                if r['Pc'][t] > 1e-3 and r['Pd'][t] > 1e-3]
        sim_bs = [(t, r['Pb'][t], r['Ps'][t])
                for t in T_ventana
                if r['Pb'][t] > 1e-3 and r['Ps'][t] > 1e-3]

        print(f"Horas con Pc>0 y Pd>0: {len(sim_cd)}")
        print(f"Horas con Pb>0 y Ps>0: {len(sim_bs)}")

        if sim_cd:
            print(f"  Primeras 5 horas con cycling BESS:")
            for t, pc, pd in sim_cd[:5]:
                print(f"    t={t}: Pc={pc:.3f} kW, Pd={pd:.3f} kW")

        if sim_bs:
            print(f"  Primeras 5 horas con cycling red:")
            for t, pb, ps in sim_bs[:5]:
                print(f"    t={t}: Pb={pb:.3f} kW, Ps={ps:.3f} kW")