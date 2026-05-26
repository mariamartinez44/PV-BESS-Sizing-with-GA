#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_analyze.py
==============
Análisis final — lee los CSVs de resultados y genera:
  1. Tabla resumen consolidada (summary_all.csv)
  2. Boxplot comparativo de todos los experimentos
  3. Ranking de variantes por mediana HV
  4. Interpretación automática del CV (D1) y Wilcoxon (P1/P2/P3)
  5. Identificación del frente representativo por experimento

Correr DESPUÉS de haber ejecutado todos los experimentos individuales.

Uso
---
  python exp_analyze.py             # analiza todo lo disponible
  python exp_analyze.py --only P2   # solo el grupo P2
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from exp_config import BASELINE, RESULTS_DIR, FIGURES_DIR

MINLP_NPV = 3_581_528   # USD — NPV de referencia del MILP


# ── Utilidades de lectura ─────────────────────────────────────────────────────

def read_csv(fname: Path) -> list[dict]:
    if not fname.exists():
        return []
    with open(fname) as f:
        return list(csv.DictReader(f))


def float_or_nan(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


# ── Cargar todos los stats CSVs ───────────────────────────────────────────────

def load_all_stats() -> list[dict]:
    """
    Lee los archivos *_stats.csv de cada experimento y los consolida.
    Añade gap_vs_minlp a cada fila.
    """
    files = {
        'D1': RESULTS_DIR / 'D1_stats.csv',
        'P1': RESULTS_DIR / 'P1_stats.csv',
        'P2': RESULTS_DIR / 'P2_stats.csv',
        'P3': RESULTS_DIR / 'P3_stats.csv',
    }
    all_rows = []
    for group, fname in files.items():
        rows = read_csv(fname)
        for r in rows:
            r['group'] = group
            npv = float_or_nan(r.get('max_npv', 'nan'))
            r['gap_vs_minlp_pct'] = (npv - MINLP_NPV) / MINLP_NPV * 100 if not np.isnan(npv) else float('nan')
        all_rows.extend(rows)
    return all_rows


def load_all_runs() -> dict[str, list[dict]]:
    """Carga los CSVs de corridas individuales por grupo."""
    files = {
        'D1': RESULTS_DIR / 'D1_seeds.csv',
        'P1': RESULTS_DIR / 'P1_all_runs.csv',
        'P2': RESULTS_DIR / 'P2_all_runs.csv',
        'P3': RESULTS_DIR / 'P3_all_runs.csv',
    }
    data = {}
    for group, fname in files.items():
        rows = read_csv(fname)
        data[group] = rows
    return data


# ── Guardar CSV consolidado ───────────────────────────────────────────────────

def save_summary(rows: list[dict], fname: Path):
    if not rows:
        print('  Sin datos para resumen.')
        return
    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with open(fname, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in all_keys})
    print(f'  CSV consolidado: {fname}')


# ── Gráfica: boxplot multi-grupo ──────────────────────────────────────────────

def plot_all_boxplots(runs_by_group: dict, stats_by_group: dict, only: str | None):
    """
    Una figura con un panel por grupo (D1, P1, P2, P3),
    cada uno con sus boxplots de HV por variante.
    """
    groups = [g for g in ['D1', 'P1', 'P2', 'P3']
              if g in runs_by_group and (only is None or only == g)]
    if not groups:
        return

    n_groups = len(groups)
    fig = plt.figure(figsize=(6 * n_groups, 5))
    gs  = gridspec.GridSpec(1, n_groups, figure=fig)

    palette = plt.cm.tab10.colors

    for col, group in enumerate(groups):
        ax  = fig.add_subplot(gs[col])
        all_runs = runs_by_group[group]

        # Agrupar corridas por variant_id, en orden de aparición
        seen  = []
        by_v: dict[str, list[float]] = {}
        for r in all_runs:
            vid = r.get('variant_id', r.get('exp_name', '?'))
            if vid not in by_v:
                by_v[vid] = []
                seen.append(vid)
            by_v[vid].append(float_or_nan(r['hv']))

        labels  = list(seen)
        hv_data = [by_v[v] for v in labels]
        short_l = [l.replace('D1-baseline', 'baseline')
                    .replace('P1-pop', 'pop=')
                    .replace('P2-', '')
                    .replace('P3-', '') for l in labels]

        bp = ax.boxplot(hv_data, labels=short_l, patch_artist=True,
                        medianprops=dict(color='black', lw=2))
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        # Línea de baseline si aplica
        bl = stats_by_group.get(group, {}).get('baseline_median')
        if bl:
            ax.axhline(bl, color='red', ls='--', lw=1.2, alpha=0.8,
                       label=f'Baseline = {bl:.5f}')
            ax.legend(fontsize=7)

        ax.set_title(f'{group}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Hipervolumen (HV)' if col == 0 else '')
        ax.tick_params(axis='x', labelsize=7, rotation=30)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Comparativo de hipervolumen por experimento',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fname = FIGURES_DIR / 'summary_all_boxplots.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Gráfica multi-boxplot: {fname}')


# ── Gráfica: ranking de variantes ─────────────────────────────────────────────

def plot_ranking(all_stats: list[dict]):
    """
    Gráfica de barras horizontales ordenada por mediana HV descendente.
    Incluye barras de error (±σ) y línea del baseline.
    """
    rows = [r for r in all_stats if float_or_nan(r.get('median_hv', 'nan')) > 0]
    if not rows:
        return

    rows_sorted = sorted(rows,
                         key=lambda r: float_or_nan(r.get('median_hv', '0')),
                         reverse=True)

    labels   = [f"{r['group']} — {r['variant_id']}" for r in rows_sorted]
    medians  = [float_or_nan(r['median_hv']) for r in rows_sorted]
    stds     = [float_or_nan(r['std_hv'])    for r in rows_sorted]

    # Mediana del baseline para línea de referencia
    baseline_med = next(
        (float_or_nan(r['median_hv']) for r in rows_sorted
         if 'baseline' in r['variant_id'].lower() or r.get('group') == 'D1'),
        None,
    )

    fig, ax = plt.subplots(figsize=(10, max(5, len(rows_sorted) * 0.4)))
    y = np.arange(len(rows_sorted))
    bars = ax.barh(y, medians, xerr=stds, color='steelblue', alpha=0.75,
                   error_kw=dict(ecolor='black', capsize=4, lw=1.2))

    if baseline_med:
        ax.axvline(baseline_med, color='red', ls='--', lw=1.5,
                   label=f'Baseline med = {baseline_med:.5f}')
        ax.legend(fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Mediana del Hipervolumen (±1σ)')
    ax.set_title('Ranking de variantes por calidad del frente de Pareto',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fname = FIGURES_DIR / 'ranking_median_hv.png'
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'  Gráfica ranking: {fname}')


# ── Impresión del reporte de texto ────────────────────────────────────────────

def print_report(all_stats: list[dict]):
    print('\n' + '='*65)
    print('REPORTE FINAL DE EXPERIMENTOS')
    print('='*65)

    groups = ['D1', 'P1', 'P2', 'P3']
    for group in groups:
        rows = [r for r in all_stats if r.get('group') == group]
        if not rows:
            continue
        print(f'\n── {group} ──────────────────────────────────────────────')
        for r in rows:
            vid   = r['variant_id']
            med   = float_or_nan(r.get('median_hv', 'nan'))
            cv    = float_or_nan(r.get('cv_pct', 'nan'))
            pval  = float_or_nan(r.get('p_wilcoxon', 'nan'))
            gap   = float_or_nan(r.get('gap_vs_minlp_pct', 'nan'))
            npv_r = float_or_nan(r.get('npv_rep', 'nan'))
            seed_r = r.get('seed_rep', '?')

            sig_str = ''
            if not np.isnan(pval):
                sig_str = ('  **' if pval < 0.01 else
                           '  *'  if pval < 0.05 else '  ns')

            print(f'  {vid:20s}  medHV={med:.5f}  CV={cv:5.2f}%'
                  f'  gap={gap:+.1f}%  seed_rep={seed_r}{sig_str}')

    # D1 interpretación CV
    d1_rows = [r for r in all_stats if r.get('group') == 'D1']
    if d1_rows:
        cv_d1 = float_or_nan(d1_rows[0].get('cv_pct', 'nan'))
        print(f'\n── Interpretación D1 (CV baseline = {cv_d1:.2f}%):')
        if cv_d1 < 1:
            print('   CV < 1%: GA muy robusto. Una réplica por configuración es suficiente.')
        elif cv_d1 < 3:
            print('   CV 1–3%: varianza moderada. Usar ≥ 3 réplicas por configuración.')
        else:
            print('   CV > 3%: alta varianza. Usar ≥ 10 réplicas y test de Wilcoxon.')

    # Mejor variante global
    valid = [r for r in all_stats
             if not np.isnan(float_or_nan(r.get('median_hv', 'nan')))]
    if valid:
        best = max(valid, key=lambda r: float_or_nan(r.get('median_hv', '0')))
        print(f'\n── Mejor variante global:')
        print(f'   {best["group"]} — {best["variant_id"]}')
        print(f'   mediana HV = {float_or_nan(best["median_hv"]):.6f}')
        print(f'   NPV representativo = {float_or_nan(best.get("npv_rep","nan")):,.0f} USD')
        print(f'   Semilla representativa = {best.get("seed_rep", "?")}')

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(only: str | None = None):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print('\nCargando resultados...')
    all_stats = load_all_stats()
    all_runs  = load_all_runs()

    if only:
        all_stats = [r for r in all_stats if r.get('group') == only]
        all_runs  = {k: v for k, v in all_runs.items() if k == only}

    if not all_stats and not any(all_runs.values()):
        print('No se encontraron resultados. Corre primero los scripts de experimento.')
        return

    # Calcular mediana baseline por grupo para referencia en gráficas
    stats_by_group: dict[str, dict] = {}
    for group in ['D1', 'P1', 'P2', 'P3']:
        rows = [r for r in all_stats if r.get('group') == group]
        if rows:
            # baseline = primera fila o la que tenga "baseline" en el nombre
            bl_row = next((r for r in rows if 'baseline' in r['variant_id'].lower()
                           or r['variant_id'] in ('D1-baseline', 'P1-pop60',
                                                   'P2-ref', 'P3-p20')), rows[0])
            stats_by_group[group] = {
                'baseline_median': float_or_nan(bl_row.get('median_hv', 'nan'))
            }

    save_summary(all_stats, RESULTS_DIR / 'summary_all.csv')
    plot_all_boxplots(all_runs, stats_by_group, only)
    plot_ranking(all_stats)
    print_report(all_stats)

    print(f'✓ Análisis completado. Figuras en: {FIGURES_DIR.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', type=str, default=None,
                        help='Analizar solo un grupo: D1, P1, P2 o P3')
    args = parser.parse_args()
    main(only=args.only)
