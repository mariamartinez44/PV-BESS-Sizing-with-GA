#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_D1_seeds.py
===============
Experimento D1 — Análisis de robustez estadística.

Corre el baseline con N_SEEDS = 15 × N_VAR semillas independientes.
Objetivo: cuantificar la varianza estocástica del GA y determinar
si los resultados son estadísticamente confiables.

Salidas
-------
  results/D1_seeds.csv          una fila por semilla
  results/D1_stats.csv          media, mediana, σ, CV%, IC95 del HV
  results/figures/D1_boxplot.png
  results/figures/conv_D1-baseline_s<N>.png  (una por semilla)

Uso
---
  python exp_D1_seeds.py
  python exp_D1_seeds.py --resume   # reanuda desde checkpoint existente
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from exp_config import N_SEEDS, REF_POINT, RESULTS_DIR, FIGURES_DIR, get_seeds
from exp_runner import (
    load_checkpoint, save_checkpoint, save_csv, compute_stats,
    run_single, cargar_datos_ventana, DIRECTORIO_VENTANA,
)

EXP_NAME   = 'D1'
VARIANT_ID = 'D1-baseline'


def main(resume: bool = True):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f'{"="*60}')
    print(f'EXPERIMENTO D1 — Robustez estadística baseline')
    print(f'  N_SEEDS = {N_SEEDS}  (15 × 11 variables activas)')
    print(f'{"="*60}')

    print('\nCargando datos 8760h...')
    data_8760, periodo_8760, T_8760 = cargar_datos_ventana(DIRECTORIO_VENTANA)

    seeds = get_seeds('D1')   # lista reproducible de N_SEEDS semillas
    cp    = load_checkpoint(EXP_NAME) if resume else {}

    # ── Correr todas las semillas ─────────────────────────────────────────────
    rows = []
    csv_path = RESULTS_DIR / 'D1_seeds.csv'
    for k, seed in enumerate(seeds):
        print(f'\n[{k+1}/{N_SEEDS}]', end='')
        row = run_single(
            exp_name        = EXP_NAME,
            variant_id      = VARIANT_ID,
            params_override = {},           # todo baseline
            seed            = seed,
            data_8760       = data_8760,
            periodo_8760    = periodo_8760,
            T_8760          = T_8760,
            ref_point       = REF_POINT,
            checkpoint      = cp,
            save_conv_plot  = True,
        )
        rows.append(row)
        # Guardar CSV incrementalmente tras cada semilla
        save_csv(rows, csv_path)

    # ── Estadísticas ──────────────────────────────────────────────────────────
    st = compute_stats(VARIANT_ID, rows)
    save_csv([st], RESULTS_DIR / 'D1_stats.csv')

    hvs       = np.array([r['hv'] for r in rows])
    median_hv = st['median_hv']
    cv        = st['cv_pct']

    print(f'\n{"─"*50}')
    print(f'D1 — Resultados estadísticos ({N_SEEDS} semillas)')
    print(f'  Media HV   : {st["mean_hv"]:.6f}')
    print(f'  Mediana HV : {median_hv:.6f}')
    print(f'  σ HV       : {st["std_hv"]:.6f}')
    print(f'  CV         : {cv:.2f}%')
    print(f'  IC95       : [{st["ci95_lo"]:.6f}, {st["ci95_hi"]:.6f}]')
    print(f'  NPV máx    : {st["max_npv"]:,.0f} USD')
    print(f'  Semilla rep.: {st["seed_rep"]}  (HV={st["hv_rep"]:.6f})')

    if cv < 1.0:
        msg = ('→ CV < 1 %: GA muy robusto. '
               'Una sola réplica es suficiente para comparar configuraciones.')
    elif cv < 3.0:
        msg = ('→ CV 1–3 %: varianza moderada. '
               'Usar ≥ 3 réplicas por experimento.')
    else:
        msg = ('→ CV > 3 %: alta varianza estocástica. '
               'Usar ≥ 10 réplicas y test de Wilcoxon para comparaciones.')
    print(f'\n  {msg}')

    # ── Gráfica: boxplot + distribución de HV ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'D1 — Robustez estadística baseline  (N={N_SEEDS} semillas)',
                 fontsize=13, fontweight='bold')

    # Panel izquierdo: boxplot
    ax = axes[0]
    bp = ax.boxplot(hvs, patch_artist=True,
                    medianprops=dict(color='black', lw=2),
                    widths=0.5)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.6)
    ax.axhline(median_hv, color='red', ls='--', lw=1.5,
               label=f'Mediana = {median_hv:.5f}')
    ax.set_xticklabels(['Baseline'])
    ax.set_ylabel('Hipervolumen (HV)')
    ax.set_title(f'Boxplot  —  CV = {cv:.2f} %')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel derecho: histograma con líneas de estadísticos
    ax2 = axes[1]
    ax2.hist(hvs, bins=max(10, N_SEEDS // 10), color='steelblue',
             alpha=0.7, edgecolor='white')
    ax2.axvline(st['mean_hv'],   color='orange', lw=2, ls='-',
                label=f'Media = {st["mean_hv"]:.5f}')
    ax2.axvline(median_hv,       color='red',    lw=2, ls='--',
                label=f'Mediana = {median_hv:.5f}')
    ax2.axvline(st['ci95_lo'],   color='gray',   lw=1, ls=':')
    ax2.axvline(st['ci95_hi'],   color='gray',   lw=1, ls=':',
                label='IC 95 %')
    ax2.set_xlabel('Hipervolumen (HV)')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución del HV')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fname = FIGURES_DIR / 'D1_boxplot.png'
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'\n  Gráfica: {fname}')
    print(f'\n✓ D1 completado.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Reanudar desde checkpoint (por defecto activo)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Ignorar checkpoint y empezar desde cero')
    args = parser.parse_args()
    main(resume=args.resume)
