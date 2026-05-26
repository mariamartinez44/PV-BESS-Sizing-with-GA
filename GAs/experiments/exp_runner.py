#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_runner.py
=============
Utilidades compartidas: ejecutar una corrida del NSGA2, calcular HV,
guardar/cargar checkpoint y estadísticas.
"""

from __future__ import annotations
import csv
import importlib
import json
import sys
import time
from copy import deepcopy
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.parallelization.starmap import StarmapParallelization
from pymoo.termination import get_termination

from exp_config import BASELINE, REF_POINT, RESULTS_DIR, FIGURES_DIR

# Importar módulo principal dinámicamente (respeta NSGA2_MODULE en exp_config)
def _load_main_module():
    from exp_config import NSGA2_MODULE
    here   = Path(__file__).resolve().parent   # carpeta de los scripts
    parent = here.parent                        # carpeta donde vive el .py principal
    for p in (str(here), str(parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module(NSGA2_MODULE)

_mod = _load_main_module()
cargar_datos_ventana    = _mod.cargar_datos_ventana
DIRECTORIO_VENTANA      = _mod.DIRECTORIO_VENTANA
ProblemaNSGA2_E3E5      = _mod.ProblemaNSGA2_E3E5
RegistrarConvergencia   = _mod.RegistrarConvergencia
evaluar_8760            = _mod.evaluar_8760
crfe                    = _mod.crfe


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint(exp_name: str) -> dict:
    """Carga checkpoint individual por experimento."""
    f = RESULTS_DIR / f'checkpoint_{exp_name}.json'
    if f.exists():
        with open(f) as fh:
            return json.load(fh)
    return {}


def save_checkpoint(exp_name: str, cp: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    f = RESULTS_DIR / f'checkpoint_{exp_name}.json'
    with open(f, 'w') as fh:
        json.dump(cp, fh, indent=2)


# ── Hipervolumen ──────────────────────────────────────────────────────────────

def compute_hv(F: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Calcula el HV del frente de Pareto F (n×2, espacio de minimización).
    F[:,0] = -OPEX,  F[:,1] = CAPEX.
    """
    try:
        return float(HV(ref_point=ref_point)(F))
    except Exception:
        return 0.0


# ── Ejecución de una corrida ──────────────────────────────────────────────────

def run_single(
    exp_name: str,
    variant_id: str,
    params_override: dict,
    seed: int,
    data_8760, periodo_8760, T_8760,
    ref_point: np.ndarray,
    checkpoint: dict,
    save_conv_plot: bool = True,
) -> dict:
    """
    Ejecuta el NSGA2 para una combinación (variante, semilla).
    Si ya existe en checkpoint, devuelve el resultado cacheado.

    Parámetros
    ----------
    exp_name       : nombre del experimento ('D1', 'P1', etc.)
    variant_id     : identificador de la variante ('P1-pop30', etc.)
    params_override: parámetros que sobreescriben el baseline
    seed           : semilla aleatoria de esta corrida
    ref_point      : punto de referencia para HV
    checkpoint     : dict mutable; se actualiza in-place y se guarda a disco

    Devuelve dict con: exp_name, variant_id, seed, hv, best_npv,
                       n_gen, t_min, pop_size, pc, eta_c, eta_m, period
    """
    run_key = f'{variant_id}__s{seed}'
    if run_key in checkpoint:
        print(f'  [skip] {run_key}')
        return checkpoint[run_key]

    # Merge baseline con overrides
    cfg = deepcopy(BASELINE)
    cfg.update(params_override)

    pop_size  = cfg['pop_size']
    n_max_gen = cfg['n_max_gen']
    period    = cfg['period']
    n_cores   = cfg['n_cores']
    pc        = cfg['pc']
    eta_c     = cfg['eta_c']
    eta_m     = cfg['eta_m']

    print(f'\n  ▶ {run_key}  '
          f'pop={pop_size} gen={n_max_gen} period={period} '
          f'pc={pc} ηc={eta_c} ηm={eta_m}')

    pool    = Pool(n_cores)
    runner  = StarmapParallelization(pool.starmap)
    problem = ProblemaNSGA2_E3E5(
        data_8760, periodo_8760, T_8760,
        elementwise_runner=runner,
    )
    algo = NSGA2(
        pop_size             = pop_size,
        sampling             = FloatRandomSampling(),
        crossover            = SBX(prob=pc, eta=eta_c),
        mutation             = PM(eta=eta_m),
        eliminate_duplicates = True,
    )
    cb   = RegistrarConvergencia(period=period, tol=1.0)
    term = get_termination('n_gen', n_max_gen)

    t0  = time.time()
    res = minimize(problem, algo, termination=term,
                   seed=seed, verbose=False, callback=cb)
    t_ga = time.time() - t0
    pool.close(); pool.join()

    # HV del frente resultante
    hv_val = compute_hv(res.F, ref_point)

    # Mejor NPV del frente
    npv_list = []
    for xi in res.X:
        CF_i, _, _, inv_i = evaluar_8760(xi, data_8760, periodo_8760, T_8760)
        npv_list.append(CF_i / crfe - inv_i)
    best_npv = float(np.max(npv_list)) if npv_list else float('nan')

    n_gen_done = res.algorithm.n_gen
    conv_gen   = cb.historial_gen[-1] if cb.historial_gen else n_max_gen

    row = dict(
        exp_name   = exp_name,
        variant_id = variant_id,
        seed       = seed,
        hv         = hv_val,
        best_npv   = best_npv,
        n_gen      = n_gen_done,
        conv_gen   = conv_gen,
        t_min      = round(t_ga / 60, 3),
        pop_size   = pop_size,
        n_max_gen  = n_max_gen,
        period     = period,
        pc         = pc,
        eta_c      = eta_c,
        eta_m      = eta_m,
    )

    checkpoint[run_key] = row
    save_checkpoint(exp_name, checkpoint)

    if save_conv_plot:
        _save_conv_plot(cb, variant_id, seed, best_npv, n_gen_done)

    print(f'     HV={hv_val:.6f}  NPV={best_npv:,.0f}  '
          f't={t_ga/60:.1f}min  gen={n_gen_done}')
    return row


def _save_conv_plot(cb, variant_id: str, seed: int, best_npv: float, n_gen: int):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cb.historial_gen,
            np.array(cb.historial_opex) / 1e6,
            color='steelblue', lw=1.5)
    if not np.isnan(best_npv):
        ax.axhline((best_npv) / 1e6, color='green', ls='--', lw=1,
                   label=f'NPV={best_npv:,.0f} USD')
    ax.set_xlabel('Generación')
    ax.set_ylabel('PVNCF = CF/CRFE  (M USD)')
    ax.set_title(f'{variant_id} | seed={seed} | gen={n_gen}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = FIGURES_DIR / f'conv_{variant_id}_s{seed}.png'
    fig.savefig(fname, dpi=100)
    plt.close(fig)


# ── Guardar CSV de resultados ─────────────────────────────────────────────────

def save_csv(rows: list[dict], fname: Path):
    if not rows:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(fname, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'  CSV: {fname}')


# ── Estadísticas por grupo de corridas ───────────────────────────────────────

def compute_stats(variant_id: str, rows: list[dict]) -> dict:
    hvs  = np.array([r['hv']       for r in rows])
    npvs = np.array([r['best_npv'] for r in rows if not np.isnan(r['best_npv'])])
    t    = np.array([r['t_min']    for r in rows])

    mean_hv   = float(np.mean(hvs))
    median_hv = float(np.median(hvs))
    std_hv    = float(np.std(hvs, ddof=1)) if len(hvs) > 1 else 0.0
    cv_pct    = std_hv / mean_hv * 100 if mean_hv > 0 else float('nan')
    se        = std_hv / np.sqrt(len(hvs))
    ci95_lo   = mean_hv - 1.96 * se
    ci95_hi   = mean_hv + 1.96 * se

    # Índice de la ejecución más cercana a la mediana (frente representativo)
    idx_rep = int(np.argmin(np.abs(hvs - median_hv)))

    r0 = rows[0]
    return dict(
        variant_id = variant_id,
        n_runs     = len(rows),
        mean_hv    = mean_hv,
        median_hv  = median_hv,
        std_hv     = std_hv,
        cv_pct     = cv_pct,
        ci95_lo    = ci95_lo,
        ci95_hi    = ci95_hi,
        mean_npv   = float(np.mean(npvs))   if len(npvs) else float('nan'),
        median_npv = float(np.median(npvs)) if len(npvs) else float('nan'),
        max_npv    = float(np.max(npvs))    if len(npvs) else float('nan'),
        mean_t_min = float(np.mean(t)),
        pop_size   = r0['pop_size'],
        n_max_gen  = r0['n_max_gen'],
        period     = r0['period'],
        pc         = r0['pc'],
        eta_c      = r0['eta_c'],
        eta_m      = r0['eta_m'],
        seed_rep   = rows[idx_rep]['seed'],    # semilla del frente representativo
        hv_rep     = rows[idx_rep]['hv'],      # HV de esa ejecución
        npv_rep    = rows[idx_rep]['best_npv'],
    )
