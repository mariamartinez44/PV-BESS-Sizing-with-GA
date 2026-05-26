#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_config.py — CONFIGURACIÓN FINAL (165 semillas, equipo 32 GB)
=================================================================
Cambios respecto al experimento D1 original:
  - N_SEEDS    : 1 → 165        (prueba estadística completa)
  - pop_size   : 60 → 80        (ganador experimento P1)
  - pc         : 0.90 → 0.80    (ganador experimento P2-F)
  - eta_c      : 15 → 10        (ganador experimento P2-F)
  - eta_m      : 20 → 5         (ganador experimento P2-F)
  - n_cores    : 4 → 16         (aprovecha los 32 GB / CPU del equipo nuevo)

Edita N_CORES si tu equipo tiene más o menos núcleos físicos.
"""

from pathlib import Path
import numpy as np

# ── Módulo principal del GA ─────────────────────────────────────────────────
NSGA2_MODULE = 'NSGAII_heuristic_E35'   # sin .py; debe estar en el mismo directorio

# ── Parámetros estadísticos ─────────────────────────────────────────────────
N_VAR   = 11    # genes activos del cromosoma (x[3],x[4] no se usan)
N_SEEDS = 165   # 15 × N_VAR  (regla TCL para robustez estadística)

# ── Baseline FINAL del sistema ──────────────────────────────────────────────
BASELINE = dict(
    pop_size  = 80,     # ← P1: mejor que 60
    n_max_gen = 40,     # sin cambio
    period    = 20,     # sin cambio (P3: no relevante)
    n_cores   = 16,     # ← AUMENTADO para equipo de 32 GB
                        #    ajusta a los núcleos físicos reales de tu máquina
    pc        = 0.80,   # ← P2-F: mejor que 0.90
    eta_c     = 10,     # ← P2-F: mejor que 15
    eta_m     = 5,      # ← P2-F: el factor más importante
)

# ── Punto de referencia para el hipervolumen ────────────────────────────────
# F = [-OPEX, CAPEX]  (ambos se minimizan en pymoo)
REF_POINT = np.array([0.0, 15_000_000.0])

# ── Directorios de salida ───────────────────────────────────────────────────
RESULTS_DIR = Path('results')
FIGURES_DIR = RESULTS_DIR / 'figures'

# ── Semillas reproducibles ──────────────────────────────────────────────────
# Mismas semillas que D1/FINAL para comparabilidad directa.
def get_seeds(group: str) -> list[int]:
    if group in ('D1', 'FINAL'):
        rng = np.random.default_rng(0)
        return rng.integers(0, 100_000, size=165).tolist()
    # otros grupos: nueva secuencia independiente
    rng = np.random.default_rng(1)
    return rng.integers(0, 100_000, size=N_SEEDS).tolist()
