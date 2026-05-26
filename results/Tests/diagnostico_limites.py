#!/usr/bin/env python3
"""
diagnostico_limites.py  (v2)
============================
Verifica que todas las variables del CSV horario estén dentro
de los límites físicos del sistema BESS+PV.

Uso:
    python diagnostico_limites.py

"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import csv
import numpy as np
from pathlib import Path

df = pd.read_csv("solution_nsga2_E3E5_v6_hourly.csv")
df['balance'] = df['Ppv'] + df['Pb'] + df['Pd'] - df['PL'] - df['Ps'] - df['Pc']
print(df['balance'].describe())
print(f"Error máximo: {df['balance'].abs().max():.2f} kW")
print(f"Error medio : {df['balance'].abs().mean():.4f} kW")

"""
debug_balance.py
================
Corre en la misma carpeta que los CSVs.
Muestra exactamente QUÉ ocurre en las horas con error de balance.
"""

hourly_file  = Path("solution_nsga2_E3E5_v6_hourly.csv")
summary_file = Path("solution_nsga2_E3E5_v6_summary.csv")

with open(summary_file, newline='') as f:
    summary = next(csv.DictReader(f))

C    = float(summary['C_kWh'])
Pinv = float(summary['PinverterBESS_kW'])
Plinst = 1000.0

rows = []
with open(hourly_file, newline='') as f:
    for row in csv.DictReader(f):
        rows.append({k: float(v) if k not in ('w1','w3') else int(v)
                     for k, v in row.items()})

# Calcular balance hora a hora
for r in rows:
    r['balance'] = r['Ppv'] + r['Pd'] + r['Pb'] - r['Pc'] - r['Ps'] - r['PL']

errors = [r for r in rows if abs(r['balance']) > 0.5]

print(f"Total horas: {len(rows)}")
print(f"Horas con |balance| > 0.5 kW: {len(errors)}")
print()

# Agrupar por valor de error redondeado
from collections import Counter
error_vals = Counter(round(r['balance'], 0) for r in errors)
print("Distribución de errores:")
for val, cnt in sorted(error_vals.items()):
    print(f"  balance ≈ {val:+.0f} kW  →  {cnt} horas ({cnt/len(rows)*100:.1f}%)")

print()
print("=== PRIMERAS 20 HORAS CON ERROR ===")
print(f"{'t':>5} {'h':>3} {'λ':>6} {'Ppvmx':>7} {'Ppv':>7} {'PL':>7} "
      f"{'Pc':>7} {'Pd':>7} {'Pb':>7} {'Ps':>7} {'w1':>3} {'w3':>3} {'bal':>8}")
print("-"*90)
for r in errors[:20]:
    t = int(r['Hour'])
    h = (t-1) % 24
    print(f"{t:>5} {h:>3} {r['lambda']:>6.1f} {r['Ppvmx']:>7.1f} {r['Ppv']:>7.1f} "
          f"{r['PL']:>7.1f} {r['Pc']:>7.1f} {r['Pd']:>7.1f} {r['Pb']:>7.1f} "
          f"{r['Ps']:>7.1f} {r['w1']:>3} {r['w3']:>3} {r['balance']:>+8.2f}")

print()
# Buscar el patrón: ¿qué tienen en común las horas con error?
print("=== ANÁLISIS DEL PATRÓN ===")
print()

# ¿Hay PV en esas horas?
con_pv    = sum(1 for r in errors if r['Ppv'] > 1)
sin_pv    = sum(1 for r in errors if r['Ppv'] <= 1)
print(f"Horas error con PV>1kW  : {con_pv}")
print(f"Horas error sin PV      : {sin_pv}")

# ¿Qué hace el BESS?
cargando  = sum(1 for r in errors if r['Pc'] > 1)
descarg   = sum(1 for r in errors if r['Pd'] > 1)
inactivo  = sum(1 for r in errors if r['Pc'] <= 1 and r['Pd'] <= 1)
print(f"Horas error BESS carga  : {cargando}")
print(f"Horas error BESS desc   : {descarg}")
print(f"Horas error BESS inact  : {inactivo}")

# ¿Qué hace la red?
comprando = sum(1 for r in errors if r['Pb'] > 1)
vendiendo = sum(1 for r in errors if r['Ps'] > 1)
print(f"Horas error comprando   : {comprando}")
print(f"Horas error vendiendo   : {vendiendo}")

# Balance esperado si no hay BESS ni PV
# Ppv + Pb = PL + Ps  → Pb = PL - Ppv (si no hay BESS)
print()

# FLUJO NETO Y TOTALES DE ENERGÍA
# Cada variable en el CSV es potencia en kW para una hora → kWh = kW × 1h

print()
print("=" * 55)
print("BALANCE ENERGÉTICO ANUAL (kWh)")
print("=" * 55)

# GENERACIÓN (fuentes que aportan energía al sistema)
Wpv    = sum(r['Ppv'] for r in rows)   # PV generado e inyectado
Wbess_d = sum(r['Pd'] for r in rows)  # descarga BESS → aporta energía
Wb     = sum(r['Pb'] for r in rows)   # compra a red → aporta energía

total_entradas = Wpv + Wbess_d + Wb

# CONSUMO (destinos de la energía)
WL     = sum(r['PL'] for r in rows)   # demanda del data center
Wbess_c = sum(r['Pc'] for r in rows)  # carga BESS → consume energía
Ws     = sum(r['Ps'] for r in rows)   # venta a red → sale del sistema

total_salidas = WL + Wbess_c + Ws

# FLUJO NETO HORARIO: Ppv + Pd - Pc - PL (positivo: excedente a red)
# Pb y Ps son consecuencia del flujo neto, no entran en su cálculo
flujos_netos = [r['Ppv'] + r['Pd'] - r['Pc'] - r['PL'] for r in rows]
horas_excedente = sum(1 for f in flujos_netos if f > 0)
horas_deficit   = sum(1 for f in flujos_netos if f < 0)
horas_neutro    = sum(1 for f in flujos_netos if f == 0)

print(f"\n--- ENTRADAS AL SISTEMA ---")
print(f"  PV generado (Wpv)         : {Wpv:>12,.1f} kWh")
print(f"  Descarga BESS (Wd)        : {Wbess_d:>12,.1f} kWh")
print(f"  Compra a red (Wb)         : {Wb:>12,.1f} kWh")
print(f"  TOTAL ENTRADAS            : {total_entradas:>12,.1f} kWh")

print(f"\n--- SALIDAS DEL SISTEMA ---")
print(f"  Demanda DC (Wl)           : {WL:>12,.1f} kWh")
print(f"  Carga BESS (Wc)           : {Wbess_c:>12,.1f} kWh")
print(f"  Venta a red (Ws)          : {Ws:>12,.1f} kWh")
print(f"  TOTAL SALIDAS             : {total_salidas:>12,.1f} kWh")

print(f"\n--- VERIFICACIÓN DE BALANCE ---")
print(f"  Entradas - Salidas        : {total_entradas - total_salidas:>+12,.2f} kWh (idealmente 0)")

print(f"\n--- FLUJO NETO (Ppv + Pd - Pc - PL) ---")
print(f"  Flujo neto total          : {sum(flujos_netos):>+12,.1f} kWh")
print(f"  Horas con excedente (→Ps) : {horas_excedente:>12,}  h")
print(f"  Horas con déficit   (→Pb) : {horas_deficit:>12,}  h")
print(f"  Horas neutras             : {horas_neutro:>12,}  h")
print(f"  Flujo neto máximo         : {max(flujos_netos):>+12,.1f} kW")
print(f"  Flujo neto mínimo         : {min(flujos_netos):>+12,.1f} kW")
print(f"  Flujo neto promedio       : {sum(flujos_netos)/len(flujos_netos):>+12,.2f} kW")

print(f"\n--- MÉTRICAS ADICIONALES ---")
print(f"  Ciclos anuales (Wc/C)     : {Wbess_c/C:>12,.1f}")
print(f"  Autosuficiencia (Wpv/Wl)  : {Wpv/WL*100:>11,.1f} %")
print(f"  Autoconsumo  (1-Ws/Wpv)   : {(1-Ws/Wpv)*100:>11,.1f} %  (PV usado internamente)")
print(f"  Dependencia red (Wb/Wl)   : {Wb/WL*100:>11,.1f} %  (fracción de demanda comprada)")
print("=" * 55)

## RESUMEN EJECUTIVO

## INTRODUCCIÓN

## PROBLEMÁTICA: BRECHA EN LA LITERATURA Y APORTE
## NO MARCO TEÓRICO, SOLO ESTADO DEL ARTE (REFERENCIAS TÉCNICAS COMO LA CREG, GAs, HEURISTICAS, ETC)
# TABLA CON TODOS LOS ARTÍCULOS POR TEMA (TECNOLOGÍA USADA, REFERENCIAS DE MÁS ACTUAL A MÁS ANTIGUA, PRIMERA FILA MI PROPUESTA,
#  REGIÓN EN LA QUE SE ESTÁ TRABAJANDO, CRITERIOS DE EVALUACIÓN* APORTE A LA LITERATURA))
# CERRAR CON BRECHA Y APORTE
# TABLAS DE SISTEMAS

## METODOLOGÍA:
# MODELO DE OPTIMIZACIÓN (ecuación, parámetros, variables, objetivos, restricciones, modelo)
# ESCRIBIR TODAS LAS FUNCIONES DE LA HEURÍSTICA
# HEURÍSTICAS DEL MODELO
# PARTE DE DISC (PROTOCOLO DE PRUEBAS)
# DIAGRAMA DE FUNCIONAMIENTO DE GA QUE UNA TODO (POBLACIÓN, SELECCIÓN, CRUCE, MUTACIÓN, EVALUACIÓN, REEMPLAZO)

### RESULTADOS:
## ESCENARIOS:
# NSGA2 VS GA
# POTENCIA REACTIVA VS NO POTENCIA REACTIVA

## VIABILIDAD TÉCNICA:
# RESULTADOS DISC: PRUEBAS DE VALIDACIÓN DEL GA
#HORA A HORA POTENCIA QUE GENERA EL PARQUE, QUE SE DESCARGA, QUE SE CARGA, QUE SE CONSUME DE LA RED
#GRAFICAR LA POTENCIA SOLAR PARA DOCUMENTO. ANÁLISIS DE PICOS Y RELACIONARLO CON EL CÓDIGO
#REPORTAR EL SOC
#GRAFICAR COMPRA Y VENTA A LA RED
#GRAFICAR DEMANDA
# SOLO PARA POTENCIA REACTIVA + NSGA2: DECIR QUE SE CUMPLE TAMBIÉN EN LOS DEMÁS ESCENARIOS

## VIABILIDAD FINANCIERA:
# RESUMEN EJECUTIVO SOLO IMPORTANTES (VPN, PAYBACK, TIR, LCOE, CAPEX, OPEX (INVERSIÓN INICIAL))
# LISTADO DE EQUIPOS Y COSTOS (TABLA DE CAPEX CON CAPACIDADES Y PRECIOS)
# RENTABILIDAD: ANÁLISIS DE LOS FLUJOS DE CAJA (TABLA DE FLUJOS DE CAJA CON INGRESOS, COSTOS, IMPUESTOS, ETC)

## CONCLUSIONES

## APÉNDICES
# TABLAS DE LAS VARIBLES, PARAMETROS, ETC.

## ANEXOS: CÓDIGOS DE PYTHON Y DOCUMENTOS IMPORTANTES


