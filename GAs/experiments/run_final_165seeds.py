#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_final_165seeds.py
=====================
Prueba de robustez FINAL con la configuración ganadora de todos los
experimentos de hiperparámetros (P1, P2, P3).

Configuración final
-------------------
  pop_size  = 80      (P1)
  n_max_gen = 40
  period    = 20
  pc        = 0.80    (P2-F)
  eta_c     = 10      (P2-F)
  eta_m     = 5       (P2-F)
  n_cores   = 16      (equipo 32 GB RAM)
  N_SEEDS   = 165     (15 × 11 variables activas, regla TCL)

Salidas
-------
  results/FINAL_seeds.csv          una fila por semilla
  results/FINAL_stats.csv          estadísticas agregadas
  results/figures/FINAL_boxplot.png
  results/figures/conv_FINAL-config_s<N>.png  (una por semilla)

Uso
---
  python run_final_165seeds.py               # empieza / reanuda automáticamente
  python run_final_165seeds.py --no-resume   # ignora checkpoint, empieza de cero
  python run_final_165seeds.py --cores 8     # sobreescribe n_cores en runtime

Estimación de tiempo (equipo 32 GB)
------------------------------------
  • Con 16 cores: ~3–4 min/semilla → ~8–11 h totales
  • Con  8 cores: ~6–7 min/semilla → ~16–19 h totales
  • Con  4 cores: ~8–9 min/semilla → ~22–25 h totales
  El script guarda checkpoint tras cada semilla, por lo que puede
  interrumpirse y reanudarse sin perder trabajo.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from exp_config import N_SEEDS, REF_POINT, RESULTS_DIR, FIGURES_DIR, get_seeds, BASELINE
from exp_runner import (
    load_checkpoint, save_checkpoint, save_csv, compute_stats,
    run_single, cargar_datos_ventana, DIRECTORIO_VENTANA,
)

EXP_NAME   = 'FINAL'
VARIANT_ID = 'FINAL-config'


def main(resume: bool = True, cores_override: int = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Aplicar override de cores si se pasó por CLI
    if cores_override is not None:
        BASELINE['n_cores'] = cores_override

    cfg = BASELINE  # ya tiene la configuración final

    print('=' * 60)
    print('PRUEBA FINAL — 165 semillas con configuración ganadora')
    print('=' * 60)
    print(f'  pop_size  = {cfg["pop_size"]}')
    print(f'  n_max_gen = {cfg["n_max_gen"]}')
    print(f'  pc        = {cfg["pc"]}')
    print(f'  eta_c     = {cfg["eta_c"]}')
    print(f'  eta_m     = {cfg["eta_m"]}')
    print(f'  n_cores   = {cfg["n_cores"]}')
    print(f'  N_SEEDS   = {N_SEEDS}')
    print('=' * 60)

    print('\nCargando datos 8760h...')
    data_8760, periodo_8760, T_8760 = cargar_datos_ventana(DIRECTORIO_VENTANA)

    seeds = get_seeds('FINAL')          # mismas 165 semillas que D1
    cp    = load_checkpoint(EXP_NAME) if resume else {}

    already_done = sum(
        1 for s in seeds
        if f'{VARIANT_ID}__s{s}' in cp
    )
    print(f'\n  Semillas ya completadas (checkpoint): {already_done}/{N_SEEDS}')
    if already_done > 0 and resume:
        print('  (usa --no-resume para ignorar el checkpoint)')

    # ── Correr todas las semillas ────────────────────────────────────────────
    rows = []
    csv_path = RESULTS_DIR / 'FINAL_seeds.csv'

    for k, seed in enumerate(seeds):
        print(f'\n[{k+1}/{N_SEEDS}]', end='')
        row = run_single(
            exp_name        = EXP_NAME,
            variant_id      = VARIANT_ID,
            params_override = {},           # todo ya está en BASELINE
            seed            = seed,
            data_8760       = data_8760,
            periodo_8760    = periodo_8760,
            T_8760          = T_8760,
            ref_point       = REF_POINT,
            checkpoint      = cp,
            save_conv_plot  = True,
        )
        rows.append(row)
        save_csv(rows, csv_path)           # guardado incremental

    # ── Estadísticas ─────────────────────────────────────────────────────────
    st = compute_stats(VARIANT_ID, rows)
    save_csv([st], RESULTS_DIR / 'FINAL_stats.csv')

    hvs       = np.array([r['hv']      for r in rows])
    times     = np.array([r['t_min']   for r in rows])
    median_hv = st['median_hv']
    cv        = st['cv_pct']

    print(f'\n{"─"*50}')
    print(f'FINAL — Resultados estadísticos ({N_SEEDS} semillas)')
    print(f'  Media HV   : {st["mean_hv"]:.6f}')
    print(f'  Mediana HV : {median_hv:.6f}')
    print(f'  σ HV       : {st["std_hv"]:.6f}')
    print(f'  CV         : {cv:.2f}%')
    print(f'  IC95       : [{st["ci95_lo"]:.6f}, {st["ci95_hi"]:.6f}]')
    print(f'  NPV máx    : {st["max_npv"]:,.0f} USD')
    print(f'  NPV mediana: {st["median_npv"]:,.0f} USD')
    print(f'  Semilla rep.: {st["seed_rep"]}  (HV={st["hv_rep"]:.6f})')
    print(f'  Tiempo medio: {st["mean_t_min"]:.2f} min/semilla')
    print(f'  Tiempo total: {times.sum()/60:.1f} h')

    if cv < 1.0:
        msg = '→ CV < 1%: GA muy robusto con configuración final.'
    elif cv < 3.0:
        msg = '→ CV 1–3%: varianza moderada; resultado confiable.'
    else:
        msg = '→ CV > 3%: varianza alta; revisar hiperparámetros.'
    print(f'\n  {msg}')

    # ── Gráfica: boxplot + histograma ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f'FINAL — Configuración ganadora  (pop=80, pc=0.80, ηc=10, ηm=5)\n'
        f'N={N_SEEDS} semillas  |  CV={cv:.2f}%',
        fontsize=12, fontweight='bold'
    )

    # Boxplot
    ax = axes[0]
    bp = ax.boxplot(hvs, patch_artist=True,
                    medianprops=dict(color='black', lw=2),
                    widths=0.5)
    bp['boxes'][0].set_facecolor('#2196F3')
    bp['boxes'][0].set_alpha(0.65)
    ax.axhline(median_hv, color='red', ls='--', lw=1.5,
               label=f'Mediana = {median_hv:.4e}')
    ax.set_xticklabels(['Config Final'])
    ax.set_ylabel('Hipervolumen (HV)')
    ax.set_title(f'Boxplot  —  CV = {cv:.2f}%')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Histograma
    ax2 = axes[1]
    ax2.hist(hvs, bins=max(10, N_SEEDS // 10),
             color='#2196F3', alpha=0.7, edgecolor='white')
    ax2.axvline(st['mean_hv'],  color='orange', lw=2, ls='-',
                label=f'Media = {st["mean_hv"]:.4e}')
    ax2.axvline(median_hv,      color='red',    lw=2, ls='--',
                label=f'Mediana = {median_hv:.4e}')
    ax2.axvline(st['ci95_lo'],  color='gray',   lw=1, ls=':')
    ax2.axvline(st['ci95_hi'],  color='gray',   lw=1, ls=':',
                label='IC 95%')
    ax2.set_xlabel('Hipervolumen (HV)')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución del HV')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fname = FIGURES_DIR / 'FINAL_boxplot.png'
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'\n  Gráfica guardada: {fname}')
    print(f'\n✓ Prueba FINAL completada.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prueba final de 165 semillas con configuración ganadora.'
    )
    parser.add_argument(
        '--resume', action='store_true', default=True,
        help='Reanudar desde checkpoint (por defecto activo)'
    )
    parser.add_argument(
        '--no-resume', dest='resume', action='store_false',
        help='Ignorar checkpoint y empezar desde cero'
    )
    parser.add_argument(
        '--cores', type=int, default=None,
        metavar='N',
        help='Sobreescribir n_cores en runtime (por defecto usa exp_config.py)'
    )
    args = parser.parse_args()
    main(resume=args.resume, cores_override=args.cores)
