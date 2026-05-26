#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_P2_crossover_mutation.py
============================
Experimento P2 — Patrones conjuntos de cruce-mutación.

Reglas del protocolo:
  - η_m restringido a {5, 10}  (tutor: "mutación entre 5 y 10")
  - p_c ∈ {0.80, 0.90, 0.95}  (p_c=0.70 excluido)
  - El baseline (p_c=0.90, η_c=15, η_m=20) se incluye como referencia;
    su η_m=20 está fuera del rango objetivo, lo que hace visible la mejora.

Variantes:
  P2-ref  p_c=0.90 η_c=15 η_m=20  Referencia (baseline)
  P2-B    p_c=0.90 η_c=5  η_m=5   Exploración alta
  P2-C    p_c=0.90 η_c=5  η_m=10  Exploración media
  P2-D    p_c=0.90 η_c=10 η_m=5   Cruce medio, mut. alta
  P2-E    p_c=0.90 η_c=10 η_m=10  Equilibrado (Deb 2001)
  P2-F    p_c=0.80 η_c=10 η_m=5   pc baja + exploración
  P2-G    p_c=0.95 η_c=15 η_m=5   pc alta + mut. alta
  P2-H    p_c=0.95 η_c=10 η_m=10  pc alta + equilibrado

Salidas
-------
  results/P2_all_runs.csv
  results/P2_stats.csv
  results/figures/P2_boxplot.png
  results/figures/P2_heatmap_median_hv.png

Uso
---
  python exp_P2_crossover_mutation.py
  python exp_P2_crossover_mutation.py --no-resume
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

EXP_NAME = 'P2'

VARIANTS = [
    dict(variant_id='P2-ref', label='Ref\nηc=15,ηm=20', params=dict(pc=0.90, eta_c=15, eta_m=20)),
    dict(variant_id='P2-B',   label='B\nηc=5,ηm=5',    params=dict(pc=0.90, eta_c=5,  eta_m=5 )),
    dict(variant_id='P2-C',   label='C\nηc=5,ηm=10',   params=dict(pc=0.90, eta_c=5,  eta_m=10)),
    dict(variant_id='P2-D',   label='D\nηc=10,ηm=5',   params=dict(pc=0.90, eta_c=10, eta_m=5 )),
    dict(variant_id='P2-E',   label='E\nηc=10,ηm=10',  params=dict(pc=0.90, eta_c=10, eta_m=10)),
    dict(variant_id='P2-F',   label='F\npc=0.80\nηc=10,ηm=5',  params=dict(pc=0.80, eta_c=10, eta_m=5 )),
    dict(variant_id='P2-G',   label='G\npc=0.95\nηc=15,ηm=5',  params=dict(pc=0.95, eta_c=15, eta_m=5 )),
    dict(variant_id='P2-H',   label='H\npc=0.95\nηc=10,ηm=10', params=dict(pc=0.95, eta_c=10, eta_m=10)),
]
BASELINE_VARIANT = 'P2-ref'


def main(resume: bool = True):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f'{"="*60}')
    print('EXPERIMENTO P2 — Patrones cruce-mutación')
    print(f'  {len(VARIANTS)} variantes × {N_SEEDS} semillas')
    print(f'{"="*60}')

    print('\nCargando datos 8760h...')
    data_8760, periodo_8760, T_8760 = cargar_datos_ventana(DIRECTORIO_VENTANA)

    seeds = get_seeds('P2')
    cp    = load_checkpoint(EXP_NAME) if resume else {}

    all_rows: dict[str, list[dict]] = {}

    for v in VARIANTS:
        vid  = v['variant_id']
        prms = v['params']
        print(f'\n{"─"*50}')
        print(f'  Variante: {vid}  {v["label"].replace(chr(10), " ")}')
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
                save_conv_plot  = False,
            )
            rows.append(row)
            # Guardar CSV incrementalmente
            flat_so_far = [r for rs in all_rows.values() for r in rs] + rows
            save_csv(flat_so_far, RESULTS_DIR / 'P2_all_runs.csv')
        all_rows[vid] = rows

    flat = [r for rows in all_rows.values() for r in rows]
    save_csv(flat, RESULTS_DIR / 'P2_all_runs.csv')

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
              f'p={st["p_wilcoxon"]:.4f}  {st["wilcoxon_sig"]}')

    save_csv(stats_list, RESULTS_DIR / 'P2_stats.csv')

    # ── Gráfica 1: Boxplot ────────────────────────────────────────────────────
    labels   = [v['label'] for v in VARIANTS]
    hv_data  = [np.array([r['hv'] for r in all_rows[v['variant_id']]]) for v in VARIANTS]
    colors   = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(13, 5))
    bp = ax.boxplot(hv_data, labels=labels, patch_artist=True,
                    medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    baseline_med = float(np.median(baseline_hvs))
    ax.axhline(baseline_med, color='red', ls='--', lw=1.5,
               label=f'Baseline mediana = {baseline_med:.5f}')

    # Anotar significancia estadística
    for k, (v, st) in enumerate(zip(VARIANTS, stats_list)):
        sig = st['wilcoxon_sig']
        if sig not in ('—',):
            ymax = max([r['hv'] for r in all_rows[v['variant_id']]])
            ax.text(k + 1, ymax * 1.001, sig,
                    ha='center', fontsize=9, color='darkred', fontweight='bold')

    ax.set_title('P2 — Hipervolumen por patrón cruce-mutación',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Hipervolumen (HV)')
    ax.set_xlabel('Variante  (** p<0.01, * p<0.05, ns = no significativo)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fname = FIGURES_DIR / 'P2_boxplot.png'
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'\n  Gráfica boxplot: {fname}')

    # ── Gráfica 2: mapa de calor mediana HV (η_c × η_m, por pc) ─────────────
    # Solo variantes con pc=0.90 para el heatmap (más limpio)
    eta_c_vals = [5, 10, 15]
    eta_m_vals = [5, 10, 20]
    hm_data    = np.full((len(eta_m_vals), len(eta_c_vals)), np.nan)

    vid_by_params = {
        (v['params']['pc'], v['params']['eta_c'], v['params']['eta_m']): v['variant_id']
        for v in VARIANTS
    }
    for j, ec in enumerate(eta_c_vals):
        for i, em in enumerate(eta_m_vals):
            key = (0.90, ec, em)
            if key in vid_by_params:
                vid  = vid_by_params[key]
                rows = all_rows.get(vid, [])
                if rows:
                    hm_data[i, j] = float(np.median([r['hv'] for r in rows]))

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    masked = np.ma.masked_invalid(hm_data)
    im = ax2.imshow(masked, cmap='RdYlGn', aspect='auto',
                    vmin=np.nanmin(hm_data), vmax=np.nanmax(hm_data))
    ax2.set_xticks(range(len(eta_c_vals)))
    ax2.set_yticks(range(len(eta_m_vals)))
    ax2.set_xticklabels([f'η_c={v}' for v in eta_c_vals])
    ax2.set_yticklabels([f'η_m={v}' for v in eta_m_vals])
    ax2.set_xlabel('Índice distribución cruce (η_c)')
    ax2.set_ylabel('Índice distribución mutación (η_m)')
    ax2.set_title('P2 — Mediana HV  (p_c=0.90)\n'
                  'Verde = mejor, Rojo = peor', fontsize=11)
    plt.colorbar(im, ax=ax2, label='Mediana HV')
    for i in range(len(eta_m_vals)):
        for j in range(len(eta_c_vals)):
            val = hm_data[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f'{val:.4f}', ha='center', va='center',
                         fontsize=9, fontweight='bold')
    fig2.tight_layout()
    fname2 = FIGURES_DIR / 'P2_heatmap_median_hv.png'
    fig2.savefig(fname2, dpi=150)
    plt.close(fig2)
    print(f'  Gráfica heatmap: {fname2}')
    print(f'\n✓ P2 completado.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    args = parser.parse_args()
    main(resume=args.resume)
