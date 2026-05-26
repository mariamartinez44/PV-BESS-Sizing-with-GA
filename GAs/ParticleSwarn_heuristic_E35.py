#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

NSGA2 bi-objetivo para el dimensionamiento conjunto de un sistema
PV + BESS (Battery Energy Storage System) conectado a red, con
despacho heurístico basado en dos estrategias combinadas:

    Estrategia Look-ahead diario (E3):
        Planifica al inicio de cada día qué horas serán de carga y cuáles
        de descarga, usando el ranking de precios de mercado del día completo.
        Es el equivalente heurístico de la estrategia TS1 de Uniejewski & Weron.

    Estrategia Score multi-señal (E5):
        Dentro de cada ventana asignada por E3, modula la potencia real
        de carga o descarga usando una función sigmoide sobre la desviación
        del precio respecto a la mediana diaria. Horas con precios más
        extremos reciben más potencia; horas mediocres, menos.

Objetivos del NSGA2 (bi-objetivo):
    F1: maximizar el valor presente neto del flujo de caja anual (OPEX = CF/crfe)
    F2: minimizar la inversión inicial (CAPEX)

El frente de Pareto resultante muestra el trade-off entre rentabilidad
y coste de inversión, permitiendo seleccionar soluciones según criterio
(máx NPV o máx BCR).

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


# PARÁMETROS DEL SISTEMA — constantes físicas y financieras

# Potencia nominal de la carga del data center (kW).
# Se usa como referencia para escalar los perfiles de consumo normalizados.
Plinst = 1000.0

# Parámetros del campo PV para convertir potencia instalada a número de módulos.
# Rmax: irradiancia de referencia (W/m²); Area: área de cada módulo (m²);
# eta: eficiencia del módulo. Se usan en la métrica nx (número de módulos).
Rmax = 1000
Area = 2.4
eta  = 0.2094

# Eficiencias de carga, descarga e inversor PV (adimensionales, entre 0 y 1).
# Se asume el mismo valor para carga y descarga del BESS (round-trip ≈ 0.926).
eff_c  = 0.9624   # eficiencia de carga del BESS
eff_d  = 0.9624   # eficiencia de descarga del BESS
# eficiencia del inversor PV ya incluida en el perfil Ppvu

# Profundidad de descarga máxima permitida (DoD = 0.90 : se usa el 90% de C).
# Define los límites operativos de SOC: [5%, 95%] de la capacidad nominal C.
DoD = 0.90

# Potencia de referencia para el cálculo de cargos de capacidad sin BESS.
# Es igual a Plinst porque sin batería toda la demanda se compra a red.
PmaxF = Plinst

# Factor de escalado de ingresos y costes energéticos 
er = 1.1

# Costes de obra civil e instalación adicionales al equipamiento (EUR).
# BoP = 0 significa que no se añaden costes de Balance of Plant separados.
BoP = 0

# Factor de costes blandos: ingeniería, permisos, margen del EPC.
# Sc = 1.2; el coste total es un 20% mayor que el coste de equipos.
Sc = 1.2

# Costes de operación y mantenimiento anuales (EUR/kW·año para PV,
# EUR/kWh·año para BESS). Se aplican sobre la potencia/capacidad instalada.
OaMpv   = 12.5   # O&M del campo PV (EUR/kW_pico/año)
OaMbess = 5.9    # O&M de la batería (EUR/kWh/año)

# Costes de inversión unitarios (EUR/kW o EUR/kWh).
# Se usan para calcular el CAPEX total en calcular_capex().
CAPEX_pv            = 388   # EUR/kW instalado de PV (modificado para coincidir con EMBER sin el del inversor del panel)
CAPEX_BESS          = 185   # EUR/kWh de capacidad de la batería
CAPEX_inverter = 48    # EUR/kW del inversor del BESS y el PV (se asume el mismo coste por kW para ambos inversores, aunque el del PV se escala con la potencia total que maneja)

# Parámetros financieros del proyecto.
# i: tasa de descuento nominal; n: vida útil del proyecto (años);
# e: tasa de inflación esperada.
i = 7.7 / 100
n = 20
e = 2.5 / 100

# Tasa de descuento real (deflactada por inflación), usada en crfe.
ir = (i - e) / (1 + e)

# Factor de recuperación de capital nominal (CRF): convierte CAPEX en
# anualidad equivalente a la tasa nominal i durante n años.
crf = (i * (i + 1)**n) / ((i + 1)**n - 1)

# Factor de recuperación de capital con escalación (CRFE): igual que CRF
# pero para flujos de caja que crecen con la inflación e. Se usa para
# calcular el valor presente de los flujos anuales CF: OPEX = CF / crfe.
crfe = (1 + e) * (ir * (ir + 1)**n) / ((ir + 1)**n - 1)

# Cargos de capacidad por periodo tarifario (EUR/kW·año).
# kappa[p] es el cargo para el periodo p (p=1 pico máximo, p=6 valle).
# kappa[0] = 0 es un placeholder (los periodos van de 1 a 6).
# Estos cargos penalizan la potencia máxima comprada a red en cada periodo,
# incentivando al BESS a reducir los picos de demanda.
kappa = [0, 28.79187*er, 15.07764*er, 6.55917*er,
             5.17209*er,  1.93281*er,  0.91609*er]

# Límites de SOC operativos derivados del DoD.
# Con DoD=0.90 la batería opera entre el 5% y el 95% de su capacidad C.
_SOC_LOW  = (1.0 - DoD) / 2.0          # 0.05; piso mínimo de SOC (5% de C)
_SOC_HIGH = (1.0 - DoD) / 2.0 + DoD   # 0.95; techo máximo de SOC (95% de C)


# LECTURA DE DATOS

BASE_DIR           = Path(__file__).resolve().parent
DIRECTORIO_VENTANA = BASE_DIR / 'ventana_completa'


def read_inc(path):
    """
    Lee un archivo .inc de GAMS con formato 't<índice>  <valor>'
    y devuelve un diccionario {índice_entero: valor_float}.

    Solo procesa líneas que comienzan con 't'.
    """
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
    """
    Carga las series horarias del año completo (8760 horas) desde los
    archivos .inc del directorio indicado y las organiza en un diccionario
    indexado por hora t (t=1..8760).

    Archivos leídos:
        lambda_spain_localtime.inc  : precio de mercado (EUR/MWh) en cada hora
        psi.inc     : cargo de energía adicional (EUR/MWh), p.ej. peaje
        PpvuMadridSarah20052023_localtime.inc : perfil normalizado de generación PV [0,1]
        PluDataCenter.inc           : perfil normalizado de consumo del DC [0,1]
        periodo.inc : periodo tarifario (1=pico..6=valle) en cada hora

    Devuelve:
        data    : dict {t: {'lambda', 'psi', 'Ppvu', 'Plu'}}
        periodo : dict {t: entero 1..6}
        T       : range(1, 8761)
    """
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


# ESPACIO DE BÚSQUEDA DEL NSGA2 — límites de los 13 genes
#
# El cromosoma tiene 13 genes divididos en dos bloques:
#
# Genes de sizing (x[0..4]) — definen el hardware instalado:
#   x[0] Ppvinst  : potencia pico PV instalada (kW)
#   x[1] C        : capacidad de la batería (kWh)
#   x[2] ratio    : C-rate del inversor BESS (Pinv = ratio × C)
#   x[3], x[4]    : Variables no usadas actualemnte, en versiones
#                   anteriores contenían PbmaxP (potencia máxima por periodo
#                   tarifario) como genes del GA.
#
# Genes de despacho (x[5..12]) — parámetros de las estrategias nuevas implementadas:
#   x[5]  n_ciclos   : número de ciclos carga-descarga planificados por día (1-2)
#   x[6]  frac_C     : fracción de C usada por ciclo (0.6-0.9), controla
#                      cuánta energía se mueve por ciclo
#   x[7]  min_spread : spread mínimo de precios (λ_max - λ_min) en EUR/MWh
#                      por debajo del cual el BESS no opera ese día. Ayuda
#                      al GA a filtrar días sin arbitraje rentable
#   x[8]  alpha      : pendiente de la sigmoide de modulación de potencia;
#                      alpha alto : despacho más agresivo,
#                      alpha bajo : potencia más gradual
#   x[9]  beta       : amplificador del incentivo a cargar cuando hay
#                      generación PV disponible; beta=0 ignora el PV
#   x[10] gamma      : peso de la corrección de SOC sobre el umbral de
#                      potencia; gamma=0 ignora el estado de carga actual
#   x[11] n_horas_c  : horas de ventana de carga por ciclo (3-6)
#   x[12] n_horas_d  : horas de ventana de descarga por ciclo (3-6)
#
# Los límites PPV_MIN/MAX, C_MIN/MAX, RATIO_MIN/MAX son cotas físicas
# razonables del mercado. Acotar RATIO_MAX a 1.0 evita inversores
# sobredimensionados (el MILP de referencia tiene C-rate ≈ 0.21).
# Estos rangos guían al GA para explorar soluciones técnicamente viables
# sin desperdiciar evaluaciones en configuraciones irreales.

PPV_MIN   = 0.1 * Plinst   # 100 kW — instalación mínima significativa
PPV_MAX   = 8.0 * Plinst   # 8000 kW — límite superior razonable para el sitio
C_MIN     = 0.5 * Plinst   # 500 kWh — batería mínima para arbitraje horario
C_MAX     = 15.0 * Plinst  # 15000 kWh — batería máxima coherente con el sitio
RATIO_MIN = 0.1             # C-rate mínimo: Pinv = 10% de C (inversor pequeño)
RATIO_MAX = 1.0             # C-rate máximo: Pinv = 2C (inversor igual a capacidad, se cambió de 2 a 1 para 
                            # evitar configuraciones irreales)

XL_SIZING   = np.array([PPV_MIN, C_MIN,  RATIO_MIN, 0., 0.])
XU_SIZING   = np.array([PPV_MAX, C_MAX,  RATIO_MAX, 1., 1.])

#Aquí están los valores de los genes de despacho, es más empírico que físico, 
# se han acotado a rangos razonables para que el GA explore estrategias de despacho 
# sin generar combinaciones extremas que no tengan sentido operativo.
XL_DISPATCH = np.array([1.0, 0.60,  0.0, 1.0, 0.0, 0.0, 3.0, 3.0])
XU_DISPATCH = np.array([2.0, 0.90, 15.0, 5.0, 1.0, 0.8, 6.0, 6.0])

XL = np.concatenate([XL_SIZING, XL_DISPATCH])   # vector de límites inferiores
XU = np.concatenate([XU_SIZING, XU_DISPATCH])   # vector de límites superiores


# DECODIFICADORES — convierten el vector de genes en parámetros operativos

def decodificar_sizing(x):
    """
    Extrae los parámetros de hardware del bloque de genes de sizing (x[0..4]).

    Pinv = ratio × C garantiza que el inversor BESS nunca supera la capacidad
    de la batería (C-rate ≤ 1). El clip asegura que aunque la mutación
    del GA produzca valores fuera de rango, el despacho recibe valores válidos.

    PbmaxP (potencia máxima comprada a red por periodo tarifario) no es
    un gen del GA, se calcula post-hoc en calcular_metricas() como el
    máximo de Pb[t] observado durante la simulación en cada periodo,
    igual que la restricción del MILP original.
    """
    C     = float(x[1])
    ratio = float(np.clip(x[2], RATIO_MIN, RATIO_MAX))
    return {
        'Ppvinst'      : float(x[0]),   # kW pico de PV instalado
        'C'            : C,              # kWh de capacidad de la batería
        'PinverterBESS': ratio * C,      # kW del inversor BESS
        'PinverterPV'  : Plinst + PmaxF + ratio * C,      # kW del inversor PV
    }


def decodificar_dispatch(x):
    """
    Extrae los parámetros de la estrategia de despacho del bloque x[5..12].

    El clip garantiza que todos los genes estén dentro de sus rangos
    aunque la mutación del GA genere valores fuera de límites.
    n_ciclos y n_horas_c/d se redondean a entero porque son discretos.
    """
    xd = np.clip(x, XL_DISPATCH, XU_DISPATCH)
    return {
        'n_ciclos'  : int(round(float(xd[0]))),   # ciclos por día planificados
        'frac_C'    : float(xd[1]),                # fracción de C por ciclo
        'min_spread': float(xd[2]),                # umbral de spread diario (EUR/MWh)
        'alpha'     : float(xd[3]),                # pendiente de la sigmoide
        'beta'      : float(xd[4]),                # amplificador de incentivo PV
        'gamma'     : float(xd[5]),                # peso de la corrección de SOC
        'n_horas_c' : int(round(float(xd[6]))),   # horas de ventana de carga
        'n_horas_d' : int(round(float(xd[7]))),   # horas de ventana de descarga
    }


def decodificar_completo(x):
    """
    Decodifica el cromosoma completo de 13 genes en (diseno, params).
    diseno contiene los parámetros de hardware; params, los de despacho.
    """
    return decodificar_sizing(x[:5]), decodificar_dispatch(x[5:13])


# ESTRATEGIA E3 — planificación look-ahead diaria

def planificar_dia_E3(horas_dia, data, params):
    """
    Asigna a cada hora del día una etiqueta de 'carga', 'descarga' o ninguna,
    usando únicamente los precios conocidos al inicio del día (look-ahead).

    Algoritmo:
        1. Calcula el spread del día (λ_max - λ_min). Si es menor que
           min_spread, el arbitraje no es rentable y el BESS permanece
           inactivo todo el día (devuelve diccionario vacío).

        2. Ordena las horas por precio ascendente. Las n_c horas más baratas
           se asignan a ventana de carga; las n_d horas más caras a descarga.
           n_c = n_horas_c × n_ciclos (hasta la mitad del día).

        3. Garantía de causalidad: una hora de descarga solo se acepta si
           existe al menos una hora de carga que la preceda temporalmente
           (para que la energía que se descarga haya sido cargada antes).

    Esta separación entre planificación (E3) y modulación de potencia (E5)
    permite que el score de E5 module cuánta potencia usar sin bloquear
    la activación de la ventana, corrigiendo la subutilización de versiones
    anteriores donde un score bajo cancelaba toda la operación.

    Parámetros:
        horas_dia : lista de índices de hora (e.g. [1,2,...,24])
        data      : diccionario {t: {'lambda': ...}} con precios horarios
        params    : diccionario con n_ciclos, min_spread, n_horas_c, n_horas_d

    Devuelve:
        dict {t: 'carga'|'descarga'} para las horas asignadas;
        dict vacío si el día no opera.
    """
    n_ciclos   = params['n_ciclos']
    min_spread = params['min_spread']
    n_horas_c  = params['n_horas_c']
    n_horas_d  = params['n_horas_d']

    lam    = {t: data[t]['lambda'] for t in horas_dia}
    spread = max(lam.values()) - min(lam.values())

    # Si el spread del día es insuficiente, no hay margen de arbitraje rentable
    if spread < min_spread:
        return {}

    # Ordenar horas por precio ascendente para identificar extremos
    horas_ord = sorted(horas_dia, key=lambda t: lam[t])

    # Número total de horas de carga y descarga, acotado a la mitad del día
    # para que siempre queden horas libres entre ventanas
    n_c = min(n_horas_c * n_ciclos, len(horas_dia) // 2)
    n_d = min(n_horas_d * n_ciclos, len(horas_dia) // 2)

    ventana_c = set(horas_ord[:n_c])               # horas más baratas : carga
    ventana_d = set(horas_ord[-n_d:]) - ventana_c  # horas más caras : descarga

    ventana: dict[int, str] = {}
    for t in ventana_c:
        ventana[t] = 'carga'

    # Aceptar descarga solo si existe una hora de carga anterior en el día
    horas_c_sorted = sorted(ventana_c)
    for t in sorted(ventana_d, key=lambda h: -lam[h]):
        if any(tc < t for tc in horas_c_sorted):
            ventana[t] = 'descarga'

    return ventana


# ESTRATEGIA E5 — modulación de potencia hora a hora

def calcular_fraccion_potencia(lam_t, lam_med_dia, soc_norm, params, modo):
    """
    Calcula la fracción de Pinv a usar en una hora dada [0, 1].

    El score mide la desviación relativa del precio respecto a la mediana:
        score = alpha × |λ(t) - λ_mediana| / λ_mediana

    Un score alto significa que el precio es extremo (muy alto o muy bajo),
    lo que justifica operar con más potencia. Un score bajo (precio cercano
    a la mediana) produce potencia reducida, evitando operar en horas
    donde el arbitraje no compensa las pérdidas de eficiencia.

    La corrección de SOC (gamma) ajusta el umbral de activación:
        - En descarga: si el SOC es bajo, sube el umbral : menos descarga
        - En carga: si el SOC es alto, sube el umbral : menos carga
    Protege la batería sin bloquear completamente la operación.
    """
    alpha = params['alpha']
    beta  = params['beta']
    gamma = params['gamma']

    lam_ref = max(abs(lam_med_dia), 1.0)
    score   = alpha * abs(lam_t - lam_med_dia) / lam_ref

    if modo == 'descarga':
        umbral = 0.05 + gamma * max(0.0, 0.5 - soc_norm)
        frac   = np.clip((score - umbral) / max(1.0 - umbral, 1e-6), 0.0, 1.0)
    else:  # 'carga'
        umbral = 0.05 + gamma * max(0.0, soc_norm - 0.5)
        frac   = np.clip((score - umbral) / max(1.0 - umbral, 1e-6), 0.0, 1.0)

    return float(frac)


def despachar_hora_E3E5(diseno, SOC, data_t, params, ventana_t, lam_med_dia):
    """
    Decide la potencia de carga (Pc) y descarga (Pd) para una hora concreta,
    combinando la ventana asignada por E3 con la modulación de potencia de E5.

    Lógica principal:
        - Si la hora está en ventana 'descarga': siempre intenta descargar.
          La fracción de Pinv usada viene de una sigmoide sobre la desviación
          del precio respecto a la mediana diaria:
              frac = sigmoid(alpha × desv)
          donde desv = (λ(t) - λ_mediana) / λ_mediana.
          Horas con precios muy por encima de la mediana (desv >> 0) reciben
          frac ≈ 1 (potencia máxima). Horas moderadas reciben fracciones
          intermedias, produciendo un perfil de descarga más gradual.

        - Si la hora está en ventana 'carga': siempre intenta cargar.
          Se usa la sigmoide inversa (sobre -desv) para que precios bajos
          (desv << 0) generen frac ≈ 1. El gen beta amplifica la fracción
          cuando hay generación PV disponible, con el fin de absorber excedentes.

        - La corrección de SOC (soc_corr_d / soc_corr_c) reduce la potencia
          cuando el SOC se acerca a sus límites físicos, evitando truncamientos
          bruscos por las restricciones de energía disponible/espacio.

        - Absorción de excedente PV: si hay generación PV sobrante después
          de atender la carga del DC, se intenta almacenar en la batería
          independientemente de si la hora es de carga por E3. Esto simula
          la prioridad de autoconsumo antes de vender a red.

    Parámetros:
        diseno     : dict con 'C', 'PinverterBESS', 'Ppvinst'
        SOC        : estado de carga actual de la batería (kWh)
        data_t     : dict con 'lambda', 'psi', 'Ppvu', 'Plu' para esta hora
        params     : parámetros de despacho (alpha, beta, gamma, ...)
        ventana_t  : 'carga', 'descarga' o None (hora inactiva según E3)
        lam_med_dia: mediana del precio del día (EUR/MWh), referencia del score

    Devuelve dict con:
        Ppv    : generación PV real inyectada (kW)
        Ppvmx  : generación PV máxima disponible (kW)
        Pc     : potencia de carga del BESS (kW)
        Pd     : potencia de descarga del BESS (kW)
        Pb     : potencia comprada a red (kW)
        Ps     : potencia vendida a red (kW)
        SOC_new: estado de carga al final de la hora (kWh)
        w1     : indicador binario — 1 si el BESS cargó
        w3     : indicador binario — 1 si se compró energía a red
        score  : fracción de descarga calculada (para diagnóstico)
    """
    C    = diseno['C']
    Pinv = diseno['PinverterBESS']
    Ppv_inv = Pinv + Plinst + PmaxF  # potencia máxima que puede manejar el inversor PV sin saturar el sistema

    # Límites de SOC derivados del DoD (en kWh absolutos)
    SOCmin = _SOC_LOW  * C
    SOCmax = _SOC_HIGH * C

    lam_t   = data_t['lambda']
    PL_t    = Plinst * data_t['Plu']       # demanda del DC en kW (= Plinst × perfil)
    Ppvmx_t = diseno['Ppvinst'] * data_t['Ppvu']  # PV disponible en kW
    Ppv_t   = Ppvmx_t   # PV real = PV disponible (sin curtailment inicial)
    pv_norm = data_t['Ppvu']               # perfil normalizado [0,1] para beta

    # SOC normalizado [0,1] para la corrección gamma
    soc_norm = float(np.clip(SOC / C, 0.0, 1.0)) if C > 1e-3 else 0.5

    # Desviación del precio respecto a la mediana del día, normalizada.
    # desv > 0 : precio por encima de mediana (hora cara, incentivo a descargar)
    # desv < 0 : precio por debajo de mediana (hora barata, incentivo a cargar)
    lam_ref = max(abs(lam_med_dia), 1.0)
    desv    = (lam_t - lam_med_dia) / lam_ref

    # Sigmoide de descarga: alta cuando el precio es muy superior a la mediana.
    # Sigmoide de carga: alta cuando el precio es muy inferior a la mediana.
    # alpha controla la pendiente: alpha≈1 : curva suave, alpha≈5 : casi escalón.
    frac_desc = float(1.0 / (1.0 + np.exp(-params['alpha'] * desv)))
    frac_carg = float(1.0 / (1.0 + np.exp( params['alpha'] * desv)))

    # Corrección de SOC: reduce la potencia si el SOC está cerca de sus límites.
    # soc_corr_d : 0 cuando SOC ≈ SOCmin (no queda energía para descargar)
    # soc_corr_c : 0 cuando SOC ≈ SOCmax (no queda espacio para cargar)
    soc_corr_d = float(np.clip((soc_norm - _SOC_LOW)  / (0.5 - _SOC_LOW),  0.0, 1.0))
    soc_corr_c = float(np.clip((_SOC_HIGH - soc_norm) / (_SOC_HIGH - 0.5), 0.0, 1.0))

    # Bonus de incentivo a cargar cuando hay generación PV: más sol : más carga.
    # beta=0 : el PV no influye en la decisión de carga (solo arbitraje puro).
    # beta=1 : la fracción de carga se amplifica proporcionalmente al PV disponible.
    pv_bonus = params['beta'] * pv_norm

    Pc_t = 0.0
    Pd_t = 0.0

    # Decisión de despacho según la ventana asignada por E3
    if ventana_t == 'descarga':
        # gamma pondera entre la fracción pura (sin corrección SOC) y la
        # fracción corregida. gamma=0 : sin corrección; gamma=1 : corrección máxima.
        frac = frac_desc * soc_corr_d * params['gamma'] \
             + frac_desc * (1.0 - params['gamma'])
        frac = float(np.clip(frac, 0.0, 1.0))
        # disp limita la descarga a la energía realmente disponible sobre SOCmin
        disp = max(SOC - SOCmin, 0.0)
        Pd_t = min(Pinv * frac, disp * eff_d)

    elif ventana_t == 'carga':
        # pv_bonus amplifica frac_carg antes del clip para reflejar el
        # incentivo adicional de absorber excedentes PV en esta hora
        frac_base = frac_carg * (1.0 + pv_bonus)
        frac = frac_base * soc_corr_c * params['gamma'] \
             + frac_base * (1.0 - params['gamma'])
        frac = float(np.clip(frac, 0.0, 1.0))
        espacio = max(SOCmax - SOC, 0.0)
        Pc_t = min(Pinv * frac, espacio / eff_c)

    # Absorción de excedente PV independiente de la ventana E3.
    # Si hay generación PV sobrante (Ppv_t > PL_t) y el BESS no está
    # descargando ni saturado, se intenta almacenar el excedente.
    # Pc_t representa solo la potencia del inversor BESS; el PV tiene
    # su propio inversor, por lo que no se suman directamente.
    if Pd_t < 1e-4 and Pc_t < Pinv - 1e-3:
        excedente_pv = max(Ppv_t - PL_t, 0.0)
        espacio_rem  = max(SOCmax - SOC - Pc_t * eff_c, 0.0)
        if excedente_pv > 1e-3 and espacio_rem > 1e-3:
            Pc_pv = min(Pinv - Pc_t, espacio_rem / eff_c, excedente_pv)
            Pc_t  = min(Pc_t + max(Pc_pv, 0.0), Pinv)

    # Balance de energía en la frontera de red.
    # flujo_neto > 0 : excedente: se vende a red (Ps > 0, Pb = 0)
    # flujo_neto < 0 : déficit: se compra a red (Pb > 0, Ps = 0)
    # No se aplica límite de Pbmax aquí. Se calcula post-hoc como el
    # máximo observado por periodo para los cargos de capacidad (kappa).
    flujo_neto = Ppv_t + Pd_t - Pc_t - PL_t
    if flujo_neto >= 0.0:
        Ps_t = min(flujo_neto, PmaxF)   # límite de exportación
        Pb_t = 0.0
    else:
        Ps_t = 0.0
        Pb_t = min(-flujo_neto, PmaxF)  # límite de importación

    # Actualización del SOC respetando los límites físicos del DoD.
    # La energía almacenada se multiplica por eff_c; la extraída se divide
    # por eff_d para reflejar las pérdidas de conversión en cada dirección.
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


# SIMULACIÓN ANUAL — 8760 horas

def simular_E3E5(diseno, params, data, T_list, periodo=None):
    """
    Simula el despacho del sistema PV+BESS durante las 8760 horas del año.

    Para cada día (bloque de 24 horas):
        1. Calcula la mediana de precios del día (referencia para E5).
        2. Llama a planificar_dia_E3() para asignar ventanas de carga/descarga.
        3. Ejecuta despachar_hora_E3E5() hora a hora, actualizando el SOC.

    El SOC inicial se fija en el punto medio de la banda operativa [SOCmin, SOCmax]
    para evitar sesgar los resultados hacia arranques con batería llena o vacía.

    Parámetros:
        diseno  : dict con 'C', 'PinverterBESS', 'Ppvinst'
        params  : dict con parámetros de despacho (alpha, beta, gamma, ...)
        data    : dict {t: {'lambda', 'psi', 'Ppvu', 'Plu'}}
        T_list  : lista de índices de hora [1, 2, ..., 8760]
        periodo : dict {t: 1..6} con el periodo tarifario de cada hora

    Devuelve:
        dict con series horarias de Ppv, Ppvmx, Pc, Pd, Pb, Ps, SOC,
        w1 (indicador carga), w3 (indicador compra) y score.
        Incluye también 'SOC0' (estado de carga inicial).
    """
    C    = diseno['C']
    Pinv = diseno['PinverterBESS']

    SOCmin = _SOC_LOW  * C
    SOCmax = _SOC_HIGH * C
    SOC    = (SOCmin + SOCmax) / 2.0   # inicio en el punto medio de la banda
    SOC0   = SOC

    res = {k: {} for k in ('Ppv', 'Ppvmx', 'Pc', 'Pd', 'Pb', 'Ps',
                            'SOC', 'w1', 'w3', 'score')}

    n_dias = len(T_list) // 24
    idx    = 0

    for _ in range(n_dias):
        horas_dia    = T_list[idx: idx + 24]
        idx         += 24

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

    # Horas sobrantes si T_list no es múltiplo exacto de 24
    for t in T_list[idx:]:
        a = despachar_hora_E3E5(diseno, SOC, data[t], params,
                                ventana_t=None, lam_med_dia=50.0)
        for k in ('Ppv', 'Ppvmx', 'Pc', 'Pd', 'Pb', 'Ps', 'w1', 'w3', 'score'):
            res[k][t] = a[k]
        res['SOC'][t] = a['SOC_new']
        SOC = a['SOC_new']

    res['SOC0'] = SOC0
    return res


# CÁLCULO DE MÉTRICAS FINANCIERAS Y ENERGÉTICAS

def calcular_metricas(diseno, res, data, periodo, T_list):
    """
    Calcula todas las métricas energéticas y financieras a partir de los
    resultados horarios de la simulación.

    Estructura de ingresos y costes:
        Es  : ingresos por venta de energía a red (er × Σ λ(t) × Ps(t))
        Eb  : coste de compra de energía a red   (er × Σ (λ(t)+ψ(t)) × Pb(t))
        Eb0 : coste de compra sin BESS/PV        (er × Σ (λ(t)+ψ(t)) × PL(t))
              : referencia contrafactual sin inversión
        CapacityP  : cargos de capacidad con BESS  (Σ κ(p) × Pbmax(p))
        CapacityP0 : cargos de capacidad sin BESS  (Σ κ(p) × PmaxF)
              : PmaxF = Plinst porque sin batería la demanda pico = carga máxima
        OaM : costes de O&M anuales (PV + BESS)

        OPEX  = CapacityP + Eb + OaM   : coste operativo total con inversión
        OPEX0 = CapacityP0 + Eb0       : coste operativo total sin inversión
        CF    = Es + OPEX0 - OPEX      : flujo de caja anual (ahorro + ingresos)

    PbmaxP[p] se calcula como max(Pb[t]) para cada periodo tarifario p,
    replicando exactamente la restricción del MILP original:
        m.addConstrs(Pbmax[periodo[t]] >= Pb[t] for t in T)
    No es un gen del GA — emerge de la simulación real.

    ciclos_anuales = Wc / C : número de ciclos completos equivalentes por año.
    
    """
    # Potencia máxima comprada por periodo tarifario (post-hoc, no gen del GA)
    Pbmax = {p: max((res['Pb'].get(t, 0.0) for t in T_list
                 if (periodo.get(t, 6) if periodo else 6) == p), default=0.0)
         for p in range(1, 7)}

    Pbmax[6] = PmaxF  # el valle siempre paga por la carga completa
    for p in range(5, 0, -1):  # p=5,4,3,2,1
        Pbmax[p] = max(Pbmax[p], Pbmax[p + 1])  # jerarquía no decreciente valle a pico

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
    }


def calcular_capex(diseno):
    """
    Calcula la inversión inicial total (CAPEX) incluyendo costes blandos.
    Sc = 1.2 añade un 20% sobre el coste de equipos por ingeniería y EPC.
    BoP = 0 en este caso (sin Balance of Plant adicional).
    """
    return BoP + Sc * (CAPEX_pv            * diseno['Ppvinst']
                       + CAPEX_BESS          * diseno['C']
                       + CAPEX_inverter  * (diseno['PinverterBESS'] +diseno['PinverterPV']))


def evaluar_8760(x, data_8760, periodo_8760, T_8760):
    """
    Función de evaluación completa para un individuo del NSGA2.
    Decodifica el cromosoma, simula las 8760 horas, calcula métricas
    y CAPEX. Devuelve (CashFlow, res, métricas, inversión).
    """
    T_list         = list(T_8760)
    diseno, params = decodificar_completo(x)
    res            = simular_E3E5(diseno, params, data_8760, T_list, periodo_8760)
    m              = calcular_metricas(diseno, res, data_8760, periodo_8760, T_list)
    inv            = calcular_capex(diseno)
    return m['CashFlow'], res, m, inv


# EXPORTACIÓN DE RESULTADOS A CSV

def guardar_csv_horario(fname, diseno, res, m, data_8760, T_8760, CF, npv_val):
    """
    Guarda una fila por hora con todas las variables de despacho y financieras.
    Benefit y CashFlow se repiten en todas las filas como referencia anual.
    """
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
    """
    Guarda una única fila con todas las métricas técnicas, financieras y
    de dimensionamiento de la solución óptima seleccionada.

    Métricas financieras calculadas aquí:
        npv     = CF/crfe - inv  (valor presente neto)
        TIR     = tasa interna de retorno
        Payback = años hasta recuperar la inversión (exacto con descuento)
        BCR     = benefit-cost ratio = PV(CF) / inv
        gap     = desviación porcentual del NPV respecto al MILP de referencia

    LCOEnet y LCOEgross expresan el coste nivelado de energía en EUR/MWh,
    útiles para comparar con tarifas de mercado.
    """
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
    nx    = 1000 * diseno['Ppvinst'] / (Rmax * eta * Area)   # número de módulos PV
    gap   = (npv_val - 3_623_548) / 3_623_548 * 100          # gap vs MILP referencia

    # PbmaxP por periodo: máximo de Pb observado en la simulación para cada
    # periodo tarifario. Equivale a la variable Pbmax del MILP original.
    T_keys   = sorted(res['Pb'].keys())
    Pbmax_ph = {}
    for p in range(1, 7):
        vals = [res['Pb'][t] for t in T_keys
                if (periodo.get(t, 6) if periodo else 6) == p]
        Pbmax_ph[p] = max(vals) if vals else 0.0

    headers = ['Ppvinst_kW','C_kWh','PinverterBESS_kW','nx','Crate',
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
        round(diseno['PinverterBESS'],2), round(nx,2), round(Crate,4),
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
        round(CAPEX_pv            * diseno['Ppvinst'],2),
        round(CAPEX_BESS          * diseno['C'],2),
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


# DEFINICIÓN DEL PROBLEMA PARA PYMOO

class ProblemaNSGA2_E3E5(ElementwiseProblem):
    """
    Problema bi-objetivo para pymoo con evaluación elemento a elemento.

    Objetivos (se minimiza en pymoo, por eso se niega F1):
        F1 = -CF/crfe = -OPEX  : maximizar el valor presente del flujo de caja
        F2 = CAPEX              : minimizar la inversión inicial

    """
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



# CALLBACK DE CONVERGENCIA

class RegistrarConvergencia(Callback):
    """
    Registra la evolución del frente de Pareto generación a generación
    e implementa parada temprana si no hay mejora durante 'period' generaciones.

    Se monitorea el mejor valor de F1 (OPEX máximo) del frente óptimo actual.
    Si no mejora más de 'tol' USD en 'period' generaciones consecutivas,
    se fuerza la terminación anticipada para ahorrar tiempo de cómputo.
    """
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
        opex  = -float(np.min(F[:, 0]))   # mejor OPEX actual (desnegado)
        capex =  float(np.min(F[:, 1]))   # menor CAPEX actual
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



# IMPRESIÓN DE RESULTADOS PARA VER RÁPIDO

def imprimir_resultado(idx, label, pareto_X, real_res, real_m, real_capex, real_cf):
    """
    Imprime por consola las métricas clave de una solución del frente de Pareto.
    gap_vs_MILP compara el NPV obtenido con el NPV del MILP de referencia
    (3,581,528 USD), indicando cuánto se aleja la solución heurística del óptimo.
    """
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
    nx  = 1000 * diseno['Ppvinst'] / (Rmax * eta * Area)

    print(f"\n{'='*62}\nRESULTADOS — {label}\n{'='*62}")
    print(f"  C={diseno['C']:,.0f} kWh  Pinv={diseno['PinverterBESS']:,.0f} kW"
          f"  PV={diseno['Ppvinst']:,.0f} kW")
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


# MAIN

if __name__ == '__main__':

    print("Cargando datos 8760h...")
    data_8760, periodo_8760, T_8760 = cargar_datos_ventana(DIRECTORIO_VENTANA)
    T_list = list(T_8760)

    # Solución de referencia del MILP para medir tiempo de evaluación
    # y verificar que el despacho produce resultados razonables antes
    # de lanzar el NSGA2. Los parámetros corresponden al sizing del MILP.
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

    # Configuración del NSGA2.
    # POP_SIZE: número de individuos por generación. Mayor población : mejor
    #   exploración del espacio de búsqueda.
    # N_MAX_GEN: máximo de generaciones.
    # N_CORES: paralelización por multiprocessing. Cada core evalúa un proceso
    # individuos independientemente (el despacho no tiene estado compartido).
    POP_SIZE  = 60
    N_MAX_GEN = 40
    PERIOD    = 20   # generaciones sin mejora para parada temprana
    N_CORES   = 4

    t_est = t_eval * POP_SIZE * N_MAX_GEN / N_CORES / 60
    print(f"\nTiempo estimado: {t_est:.1f} min"
          f"  (pop={POP_SIZE} gen={N_MAX_GEN} cores={N_CORES})")

    pool     = Pool(N_CORES)
    runner   = StarmapParallelization(pool.starmap)
    problema = ProblemaNSGA2_E3E5(data_8760, periodo_8760, T_8760,
                                   elementwise_runner=runner)

    # SBX (Simulated Binary Crossover): operador de cruce estándar para
    # variables continuas. eta=15 : distribución concentrada cerca de los
    # padres (exploración local). prob=0.9 : alta tasa de cruce.
    # PM (Polynomial Mutation): mutación con distribución polinomial.
    # eta=20 : mutaciones pequeñas (refinamiento). Combinado con SBX
    # produce una búsqueda equilibrada entre exploración y explotación.
    algoritmo = NSGA2(
        pop_size=POP_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    criterio = get_termination('n_gen', N_MAX_GEN)
    callback = RegistrarConvergencia(period=PERIOD, tol=1.0)

    print("\nIniciando NSGA2 E3+E5 v3...")
    t0 = time.time()
    resultado = minimize(problema, algoritmo, termination=criterio,
                         seed=42, verbose=False, callback=callback)
    t_ga = time.time() - t0
    pool.close(); pool.join()
    print(f"\nNSGA2 completado en {t_ga/60:.1f} min"
          f" | {resultado.algorithm.n_gen} gen")

    # Post-proceso: re-evaluar todas las soluciones del frente de Pareto
    # para obtener métricas reales (la evaluación interna del NSGA2 puede
    # tener pequeñas diferencias por el orden de evaluación paralela).
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
        print(f"  [{idx+1:>3}/{n_pareto}] NPV={real_npv[idx]:>10,.0f} | "
              f"CAPEX={real_capex_a[idx]/1e6:.2f}M | "
              f"ciclos={m_i['ciclos_anuales']:.0f} | "
              f"n_c={pi['n_ciclos']} nh_c={pi['n_horas_c']}"
              f" nh_d={pi['n_horas_d']}")

    idx_v        = np.arange(n_pareto)
    npv_v        = real_npv[idx_v]
    capex_v      = real_capex_a[idx_v]
    bcr_v        = (npv_v + capex_v) / np.where(capex_v > 0, capex_v, 1e-9)
    idx_best_npv = idx_v[int(np.nanargmax(npv_v))]
    idx_best_bcr = idx_v[int(np.nanargmax(bcr_v))]

    imprimir_resultado(idx_best_npv, 'NSGA2 E3+E5 v3 — Máx NPV',
                       pareto_X, real_res, real_m, real_capex_a, real_cf)
    imprimir_resultado(idx_best_bcr, 'NSGA2 E3+E5 v3 — Máx BCR',
                       pareto_X, real_res, real_m, real_capex_a, real_cf)

    d_best  = decodificar_completo(pareto_X[idx_best_npv])[0]
    p_best  = decodificar_completo(pareto_X[idx_best_npv])[1]
    CF_best  = real_cf[idx_best_npv]
    inv_best = real_capex_a[idx_best_npv]
    npv_best = CF_best/crfe - inv_best

    guardar_csv_horario('solution_nsga2_E3E5_hourly.csv',
                        d_best, real_res[idx_best_npv], real_m[idx_best_npv],
                        data_8760, T_8760, CF_best, npv_best)
    guardar_csv_resumen('solution_nsga2_E3E5_summary.csv',
                        d_best, p_best, real_res[idx_best_npv],
                        real_m[idx_best_npv], inv_best, CF_best,
                        periodo=periodo_8760)

    # CSV con todas las soluciones del frente de Pareto
    with open('pareto_nsga2_E3E5.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx','OPEX','CAPEX','NPV','CF','ciclos','BCR',
                    'Ppvinst','C','PinverterBESS','n_ciclos','frac_C',
                    'min_spread','alpha','beta','gamma','n_horas_c','n_horas_d'])
        for idx in range(n_pareto):
            di, pi = decodificar_completo(pareto_X[idx])
            bcr = ((real_npv[idx] + real_capex_a[idx]) / real_capex_a[idx]
                   if real_capex_a[idx] > 0 else float('nan'))
            w.writerow([idx,
                        real_opex[idx], real_capex_a[idx],
                        real_npv[idx],  real_cf[idx],
                        real_m[idx]['ciclos_anuales'], bcr,
                        di['Ppvinst'], di['C'], di['PinverterBESS'],
                        pi['n_ciclos'], pi['frac_C'], pi['min_spread'],
                        pi['alpha'], pi['beta'], pi['gamma'],
                        pi['n_horas_c'], pi['n_horas_d']])
    print("CSV Pareto: pareto_nsga2_E3E5.csv")

    # Gráficas de resultados
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('NSGA2 E3+E5  |  OPEX vs CAPEX', fontsize=13, fontweight='bold')

    ax = axes[0]
    sc = ax.scatter(real_capex_a/1e6, real_opex/1e6,
                    c=real_npv/1e6, cmap='RdYlGn', s=80, zorder=3)
    ax.scatter(real_capex_a[idx_best_npv]/1e6, real_opex[idx_best_npv]/1e6,
               c='green', s=200, marker='D', zorder=5, label='Máx NPV')
    ax.scatter(real_capex_a[idx_best_bcr]/1e6, real_opex[idx_best_bcr]/1e6,
               c='black', s=200, marker='s', zorder=5, label='Máx BCR')
    fig.colorbar(sc, ax=ax).set_label('NPV (M USD)')
    ax.set_xlabel('CAPEX (M USD)'); ax.set_ylabel('OPEX=CF/CRFE (M USD)')
    ax.set_title('Frente de Pareto'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(callback.historial_gen,
            np.array(callback.historial_opex)/1e6, color='steelblue', lw=2)
    ax.axhline(CF_best/crfe/1e6, color='green', ls='--', lw=1.5,
               label=f'OPEX Máx NPV: {CF_best/crfe/1e6:.2f} MUSD')
    ax.set_xlabel('Generación'); ax.set_ylabel('OPEX (M USD)')
    ax.set_title('Convergencia'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergencia_nsga2_E3E5.png', dpi=150)
    plt.show()
    print("Gráfica: convergencia_nsga2_E3E5.png")