#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_P1_population.py
====================
Experimento P1 — Tamaño de población.

Variantes: pop = 30, 50, 60 (baseline), 80
Cada variante se corre con las mismas N_SEEDS semillas para
garantizar comparación justa (mismas condiciones iniciales).

Salidas
-------
  results/P1_all_runs.csv       una fila por (variante, semilla)
  results/P1_stats.csv          estadísticos por variante + p-valor Wilcoxon
  results/figures/P1_boxplot.png
  results/figures/P1_efficiency.png   HV/tiempo vs pop_size

Uso
---
  python exp_P1_population.py
  python exp_P1_population.py --no-resume
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from exp_config import BASELINE, N_SEEDS, REF_POINT, RESULTS_DIR, FIGURES_DIR, get_seeds
from exp_runner import (
    load_checkpoint, save_checkpoint, save_csv, compute_stats,
    run_single, cargar_datos_ventana, DIRECTORIO_VENTANA,
)

EXP_NAME = 'P1'

# ── Variantes a probar ────────────────────────────────────────────────────────
# El baseline (pop=60) se incluye aquí para tener sus HVs con las mismas
# semillas que las otras variantes (permite Wilcoxon pareado).
VARIANTS = [
    dict(variant_id='P1-pop30',  params=dict(pop_size=30)),
    dict(variant_id='P1-pop50',  params=dict(pop_size=50)),
    dict(variant_id='P1-pop60',  params=dict(pop_size=60)),   # baseline
    dict(variant_id='P1-pop80',  params=dict(pop_size=80)),
]
BASELINE_VARIANT = 'P1-pop60'


def main(resume: bool = True):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f'{"="*60}')
    print(f'EXPERIMENTO P1 — Tamaño de población')
    print(f'  Variantes : {[v["variant_id"] for v in VARIANTS]}')
    print(f'  N_SEEDS   : {N_SEEDS} semillas por variante')
    print(f'{"="*60}')

    print('\nCargando datos 8760h...')
    data_8760, periodo_8760, T_8760 = cargar_datos_ventana(DIRECTORIO_VENTANA)

    seeds = get_seeds('P1')   # mismas semillas para todas las variantes
    cp    = load_checkpoint(EXP_NAME) if resume else {}

    # ── Correr todas las variantes ────────────────────────────────────────────
    all_rows: dict[str, list[dict]] = {}

    for v in VARIANTS:
        vid  = v['variant_id']
        prms = v['params']
        pop  = prms.get('pop_size', BASELINE['pop_size'])
        print(f'\n{"─"*50}')
        print(f'  Variante: {vid}  (pop={pop})')
        rows = []
        for k, seed in enumerate(seeds):
            print(f'  [{k+1}/{N_SEEDS}]', end='')
            row = run_single(
                exp_name        = EXP_NAME,
                variant_id      = vid,
                params_override = prms,
                seed            = seed,
                data_8760       = data_8760,
                periodo_8760    = periodo_8760,
                T_8760          = T_8760,
                ref_point       = REF_POINT,
                checkpoint      = cp,
                save_conv_plot  = False,   # P1: omitir plots individuales
            )
            rows.append(row)
            # Guardar CSV incrementalmente
            flat_so_far = [r for rs in all_rows.values() for r in rs] + rows
            save_csv(flat_so_far, RESULTS_DIR / 'P1_all_runs.csv')
        all_rows[vid] = rows

    # ── CSV de todas las corridas ─────────────────────────────────────────────
    flat = [r for rows in all_rows.values() for r in rows]
    save_csv(flat, RESULTS_DIR / 'P1_all_runs.csv')

    # ── Estadísticos + Wilcoxon ───────────────────────────────────────────────
    stats_list = []
    baseline_hvs = np.array([r['hv'] for r in all_rows[BASELINE_VARIANT]])

    for v in VARIANTS:
        vid  = v['variant_id']
        rows = all_rows[vid]
        st   = compute_stats(vid, rows)
        hvs  = np.array([r['hv'] for r in rows])

        # Test de Wilcoxon pareado vs baseline (mismo orden de semillas)
        if vid == BASELINE_VARIANT:
            st['p_wilcoxon'] = float('nan')
            st['wilcoxon_sig'] = '—'
        else:
            try:
                _, pval = wilcoxon(baseline_hvs, hvs)
                st['p_wilcoxon'] = float(pval)
                st['wilcoxon_sig'] = ('*' if pval < 0.05 else
                                      '(**p<0.01)' if pval < 0.01 else 'ns')
            except Exception:
                st['p_wilcoxon'] = float('nan')
                st['wilcoxon_sig'] = 'error'

        stats_list.append(st)
        pop = v['params'].get('pop_size', BASELINE['pop_size'])
        marker = ' ← baseline' if vid == BASELINE_VARIANT else ''
        print(f'\n  {vid}  pop={pop}{marker}')
        print(f'    mediana HV={st["median_hv"]:.6f}  '
              f'CV={st["cv_pct"]:.2f}%  '
              f'p_wilcoxon={st["p_wilcoxon"]:.4f}  {st["wilcoxon_sig"]}')

    save_csv(stats_list, RESULTS_DIR / 'P1_stats.csv')

    # ── Gráfica 1: Boxplot comparativo ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    hv_data    = [np.array([r['hv'] for r in all_rows[v['variant_id']]]) for v in VARIANTS]
    xlabels    = [f"pop={v['params'].get('pop_size', BASELINE['pop_size'])}" for v in VARIANTS]
    colors     = ['#5b9bd5', '#70ad47', '#ed7d31', '#ffc000']

    bp = ax.boxplot(hv_data, labels=xlabels, patch_artist=True,
                    medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Línea de mediana del baseline
    baseline_median = float(np.median(baseline_hvs))
    ax.axhline(baseline_median, color='red', ls='--', lw=1.5,
               label=f'Baseline mediana = {baseline_median:.5f}')

    # Anotar p-valores Wilcoxon
    for k, st in enumerate(stats_list):
        if st['wilcoxon_sig'] not in ('—', 'error', float('nan')):
            ymax = max([r['hv'] for r in all_rows[VARIANTS[k]['variant_id']]])
            ax.text(k + 1, ymax * 1.002, st['wilcoxon_sig'],
                    ha='center', fontsize=8, color='darkred')

    ax.set_title('P1 — Hipervolumen por tamaño de población',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Tamaño de población')
    ax.set_ylabel('Hipervolumen (HV)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fname = FIGURES_DIR / 'P1_boxplot.png'
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'\n  Gráfica boxplot: {fname}')

    # ── Gráfica 2: eficiencia (mediana HV / tiempo medio) ────────────────────
    pops     = [v['params'].get('pop_size', BASELINE['pop_size']) for v in VARIANTS]
    med_hvs  = [st['median_hv']  for st in stats_list]
    mean_ts  = [st['mean_t_min'] for st in stats_list]
    eff      = [h / t if t > 0 else 0 for h, t in zip(med_hvs, mean_ts)]

    fig, axes2 = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle('P1 — Análisis de eficiencia', fontweight='bold')

    axes2[0].plot(pops, med_hvs, 'o-', color='steelblue', lw=2, ms=8)
    axes2[0].axvline(BASELINE['pop_size'], color='red', ls='--', lw=1,
                     label='Baseline')
    axes2[0].set_xlabel('pop_size')
    axes2[0].set_ylabel('Mediana HV')
    axes2[0].set_title('Calidad del frente')
    axes2[0].legend()
    axes2[0].grid(alpha=0.3)

    axes2[1].plot(pops, eff, 's-', color='darkorange', lw=2, ms=8)
    axes2[1].axvline(BASELINE['pop_size'], color='red', ls='--', lw=1)
    axes2[1].set_xlabel('pop_size')
    axes2[1].set_ylabel('Mediana HV / tiempo medio (HV·min⁻¹)')
    axes2[1].set_title('Eficiencia (HV por minuto)')
    axes2[1].grid(alpha=0.3)

    fig.tight_layout()
    fname2 = FIGURES_DIR / 'P1_efficiency.png'
    fig.savefig(fname2, dpi=150)
    plt.close(fig)
    print(f'  Gráfica eficiencia: {fname2}')
    print(f'\n✓ P1 completado.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    args = parser.parse_args()
    main(resume=args.resume)
