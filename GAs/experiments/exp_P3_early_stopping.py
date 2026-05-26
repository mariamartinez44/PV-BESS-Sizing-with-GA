#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_P3_early_stopping.py
========================
Experimento P3 — Periodo de parada temprana.

Con N_gen=40, el baseline usa period=20 (50% de las generaciones).
Se evalúa si detener antes o después cambia significativamente el HV.

Variantes:
  P3-p10   period=10  (25% de N_gen)
  P3-p20   period=20  (50%, baseline)
  P3-p30   period=30  (75%)
  P3-inf   period=9999  (sin parada temprana — corre las 40 gen siempre)

Nota: period=9999 es funcionalmente equivalente a "sin parada" cuando
N_gen=40, ya que el GA nunca acumula 9999 generaciones sin mejora.

Salidas
-------
  results/P3_all_runs.csv
  results/P3_stats.csv
  results/figures/P3_boxplot.png
  results/figures/P3_hv_vs_time.png     trade-off calidad vs tiempo

Uso
---
  python exp_P3_early_stopping.py
  python exp_P3_early_stopping.py --no-resume
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from exp_config import BASELINE, N_SEEDS, REF_POINT, RESULTS_DIR, FIGURES_DIR, get_seeds
from exp_runner import (
    load_checkpoint, save_csv, compute_stats,
    run_single, cargar_datos_ventana, DIRECTORIO_VENTANA,
)

EXP_NAME = 'P3'

VARIANTS = [
    dict(variant_id='P3-p10',  label='period=10\n(25% Ngen)', period=10),
    dict(variant_id='P3-p20',  label='period=20\n(50% Ngen, baseline)', period=20),
    dict(variant_id='P3-p30',  label='period=30\n(75% Ngen)', period=30),
    dict(variant_id='P3-inf',  label='sin parada\n(period=∞)', period=9999),
]
BASELINE_VARIANT = 'P3-p20'


def main(resume: bool = True):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f'{"="*60}')
    print('EXPERIMENTO P3 — Periodo de parada temprana')
    print(f'  N_gen = {BASELINE["n_max_gen"]}  |  '
          f'{len(VARIANTS)} variantes × {N_SEEDS} semillas')
    print(f'{"="*60}')

    print('\nCargando datos 8760h...')
    data_8760, periodo_8760, T_8760 = cargar_datos_ventana(DIRECTORIO_VENTANA)

    seeds = get_seeds('P3')
    cp    = load_checkpoint(EXP_NAME) if resume else {}

    all_rows: dict[str, list[dict]] = {}

    for v in VARIANTS:
        vid    = v['variant_id']
        period = v['period']
        print(f'\n{"─"*50}')
        print(f'  Variante: {vid}  (period={period})')
        rows = []
        for k, seed in enumerate(seeds):
            print(f'  [{k+1}/{N_SEEDS}]', end='')
            row = run_single(
                exp_name        = EXP_NAME,
                variant_id      = vid,
                params_override = dict(period=period),
                seed            = seed,
                data_8760       = data_8760,
                periodo_8760    = periodo_8760,
                T_8760          = T_8760,
                ref_point       = REF_POINT,
                checkpoint      = cp,
                save_conv_plot  = False,
            )
            rows.append(row)
            # Guardar CSV incrementalmente
            flat_so_far = [r for rs in all_rows.values() for r in rs] + rows
            save_csv(flat_so_far, RESULTS_DIR / 'P3_all_runs.csv')
        all_rows[vid] = rows

    flat = [r for rows in all_rows.values() for r in rows]
    save_csv(flat, RESULTS_DIR / 'P3_all_runs.csv')

    # ── Estadísticos + Wilcoxon ───────────────────────────────────────────────
    baseline_hvs = np.array([r['hv'] for r in all_rows[BASELINE_VARIANT]])
    stats_list   = []

    for v in VARIANTS:
        vid  = v['variant_id']
        rows = all_rows[vid]
        st   = compute_stats(vid, rows)
        hvs  = np.array([r['hv'] for r in rows])

        if vid == BASELINE_VARIANT:
            st['p_wilcoxon'] = float('nan')
            st['wilcoxon_sig'] = '—'
        else:
            try:
                _, pval = wilcoxon(baseline_hvs, hvs)
                st['p_wilcoxon'] = float(pval)
                st['wilcoxon_sig'] = ('**' if pval < 0.01 else
                                      '*'  if pval < 0.05 else 'ns')
            except Exception:
                st['p_wilcoxon'] = float('nan')
                st['wilcoxon_sig'] = 'err'

        stats_list.append(st)
        print(f'\n  {vid}: medHV={st["median_hv"]:.6f}  '
              f'CV={st["cv_pct"]:.2f}%  '
              f'meanT={st["mean_t_min"]:.1f}min  '
              f'p={st["p_wilcoxon"]:.4f}  {st["wilcoxon_sig"]}')

    save_csv(stats_list, RESULTS_DIR / 'P3_stats.csv')

    # ── Gráfica 1: Boxplot ────────────────────────────────────────────────────
    labels  = [v['label'] for v in VARIANTS]
    hv_data = [np.array([r['hv'] for r in all_rows[v['variant_id']]]) for v in VARIANTS]
    colors  = ['#5b9bd5', '#ed7d31', '#70ad47', '#9e480e']

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(hv_data, labels=labels, patch_artist=True,
                    medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    baseline_med = float(np.median(baseline_hvs))
    ax.axhline(baseline_med, color='red', ls='--', lw=1.5,
               label=f'Baseline mediana = {baseline_med:.5f}')

    for k, (v, st) in enumerate(zip(VARIANTS, stats_list)):
        sig = st['wilcoxon_sig']
        if sig not in ('—',):
            ymax = max([r['hv'] for r in all_rows[v['variant_id']]])
            ax.text(k + 1, ymax * 1.001, sig,
                    ha='center', fontsize=10, color='darkred', fontweight='bold')

    ax.set_title('P3 — Hipervolumen por periodo de parada temprana',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Hipervolumen (HV)')
    ax.set_xlabel('Variante  (** p<0.01, * p<0.05, ns = no significativo)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fname = FIGURES_DIR / 'P3_boxplot.png'
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'\n  Gráfica boxplot: {fname}')

    # ── Gráfica 2: trade-off HV vs tiempo ────────────────────────────────────
    # Muestra si ganar calidad con period mayor sale caro en tiempo
    med_hvs = [st['median_hv']  for st in stats_list]
    mean_ts = [st['mean_t_min'] for st in stats_list]
    xlabels_short = [str(v['period']) if v['period'] < 9999 else '∞'
                     for v in VARIANTS]
    colors2 = ['#5b9bd5', '#ed7d31', '#70ad47', '#9e480e']

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
    fig2.suptitle('P3 — Trade-off calidad vs tiempo de cómputo', fontweight='bold')

    # Mediana HV
    axes2[0].bar(xlabels_short, med_hvs, color=colors2, alpha=0.8, edgecolor='black', lw=0.5)
    axes2[0].axhline(baseline_med, color='red', ls='--', lw=1.5, label='Baseline')
    axes2[0].set_xlabel('period')
    axes2[0].set_ylabel('Mediana HV')
    axes2[0].set_title('Calidad (HV)')
    axes2[0].legend(fontsize=9)
    axes2[0].grid(axis='y', alpha=0.3)

    # Tiempo medio
    axes2[1].bar(xlabels_short, mean_ts, color=colors2, alpha=0.8, edgecolor='black', lw=0.5)
    axes2[1].set_xlabel('period')
    axes2[1].set_ylabel('Tiempo medio (min)')
    axes2[1].set_title('Tiempo de cómputo')
    axes2[1].grid(axis='y', alpha=0.3)

    fig2.tight_layout()
    fname2 = FIGURES_DIR / 'P3_hv_vs_time.png'
    fig2.savefig(fname2, dpi=150)
    plt.close(fig2)
    print(f'  Gráfica trade-off: {fname2}')
    print(f'\n✓ P3 completado.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    args = parser.parse_args()
    main(resume=args.resume)
