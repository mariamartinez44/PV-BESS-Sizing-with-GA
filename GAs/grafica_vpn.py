#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_FINAL_bimodal.py
==================
Analiza la distribución bimodal del HV en FINAL y correlaciona
con el NPV para entender los dos modos del GA.

Uso:
  python plot_FINAL_bimodal.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path('results/FINAL_seeds.csv')

# ── Leer datos ────────────────────────────────────────────────────────────────
rows = []
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        try:
            rows.append(dict(
                seed     = int(row['seed']),
                hv       = float(row['hv']),
                best_npv = float(row['best_npv']),
                n_gen    = int(row['n_gen']),
                t_min    = float(row['t_min']),
            ))
        except (ValueError, KeyError):
            continue

hvs  = np.array([r['hv']       for r in rows])
npvs = np.array([r['best_npv'] for r in rows])
gens = np.array([r['n_gen']    for r in rows])

# ── Encontrar el umbral que separa los dos grupos ────────────────────────────
# Se usa el valle del histograma como umbral (entre 0.75 y 0.85 × 10^14)
# Ajusta THRESHOLD si la separación visual está en otro punto
counts, edges = np.histogram(hvs, bins=30)
# El umbral es el punto de mínimo entre los dos picos
mid_idx  = len(counts) // 2
valley   = np.argmin(counts[5:mid_idx+5]) + 5   # buscar valle en zona central
THRESHOLD = float((edges[valley] + edges[valley + 1]) / 2)

mask_low  = hvs < THRESHOLD
mask_high = hvs >= THRESHOLD

n_low  = mask_low.sum()
n_high = mask_high.sum()

print(f'Umbral HV    : {THRESHOLD:.3e}')
print(f'Grupo bajo   : {n_low:3d} semillas  ({n_low/len(hvs)*100:.1f}%)')
print(f'Grupo alto   : {n_high:3d} semillas  ({n_high/len(hvs)*100:.1f}%)')
print()
print(f'NPV medio grupo bajo  : {npvs[mask_low].mean():>12,.0f} USD')
print(f'NPV medio grupo alto  : {npvs[mask_high].mean():>12,.0f} USD')
print(f'Diferencia de NPV     : {npvs[mask_high].mean()-npvs[mask_low].mean():>12,.0f} USD  '
      f'({(npvs[mask_high].mean()/npvs[mask_low].mean()-1)*100:.1f}% mejor en grupo alto)')
print()
print(f'Generaciones medias grupo bajo : {gens[mask_low].mean():.1f}')
print(f'Generaciones medias grupo alto : {gens[mask_high].mean():.1f}')

# ── Figura ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('FINAL — Análisis de distribución bimodal del HV',
             fontsize=13, fontweight='bold')

# Panel 1: histograma HV coloreado por grupo
ax = axes[0, 0]
ax.hist(hvs[mask_low]  / 1e14, bins=15, color='tomato',
        alpha=0.7, label=f'Grupo bajo  (n={n_low})', edgecolor='white')
ax.hist(hvs[mask_high] / 1e14, bins=15, color='steelblue',
        alpha=0.7, label=f'Grupo alto  (n={n_high})', edgecolor='white')
ax.axvline(THRESHOLD / 1e14, color='black', ls='--', lw=1.5, label='Umbral')
ax.set_xlabel('HV (×10¹⁴)')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución bimodal del HV')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 2: boxplot NPV por grupo
ax2 = axes[0, 1]
bp = ax2.boxplot(
    [npvs[mask_low] / 1e6, npvs[mask_high] / 1e6],
    labels=['Grupo bajo\n(HV bajo)', 'Grupo alto\n(HV alto)'],
    patch_artist=True,
    medianprops=dict(color='black', lw=2),
)
bp['boxes'][0].set_facecolor('tomato')
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('steelblue')
bp['boxes'][1].set_alpha(0.7)
ax2.set_ylabel('NPV (M USD)')
ax2.set_title('NPV por grupo de convergencia')
ax2.grid(axis='y', alpha=0.3)

# Anotar diferencia
diff_pct = (npvs[mask_high].mean() / npvs[mask_low].mean() - 1) * 100
ax2.text(1.5, (npvs[mask_low].mean() + npvs[mask_high].mean()) / 2 / 1e6,
         f'+{diff_pct:.1f}%', ha='center', va='center',
         fontsize=11, fontweight='bold', color='darkgreen')

# Panel 3: scatter HV vs NPV
ax3 = axes[1, 0]
ax3.scatter(hvs[mask_low]  / 1e14, npvs[mask_low]  / 1e6,
            color='tomato',    alpha=0.6, s=30, label='Grupo bajo')
ax3.scatter(hvs[mask_high] / 1e14, npvs[mask_high] / 1e6,
            color='steelblue', alpha=0.6, s=30, label='Grupo alto')
ax3.axvline(THRESHOLD / 1e14, color='black', ls='--', lw=1, alpha=0.5)
ax3.set_xlabel('HV (×10¹⁴)')
ax3.set_ylabel('NPV (M USD)')
ax3.set_title('Correlación HV ↔ NPV')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Correlación
corr = np.corrcoef(hvs, npvs)[0, 1]
ax3.text(0.05, 0.95, f'r = {corr:.3f}',
         transform=ax3.transAxes, fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel 4: generaciones por grupo (muestra si el grupo bajo terminó antes)
ax4 = axes[1, 1]
bp2 = ax4.boxplot(
    [gens[mask_low], gens[mask_high]],
    labels=['Grupo bajo', 'Grupo alto'],
    patch_artist=True,
    medianprops=dict(color='black', lw=2),
)
bp2['boxes'][0].set_facecolor('tomato')
bp2['boxes'][0].set_alpha(0.7)
bp2['boxes'][1].set_facecolor('steelblue')
bp2['boxes'][1].set_alpha(0.7)
ax4.set_ylabel('Generaciones ejecutadas')
ax4.set_title('¿El grupo bajo terminó antes?')
ax4.grid(axis='y', alpha=0.3)

fig.tight_layout()
out = Path('results/figures/FINAL_bimodal_analysis.png')
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
plt.show()
print(f'\nGráfica: {out}')

# ── Diagnóstico final ─────────────────────────────────────────────────────────
print('\n' + '='*55)
print('DIAGNÓSTICO')
print('='*55)
print(f'El GA tiene dos modos de convergencia con los parámetros actuales:')
print(f'  • {n_low} semillas ({n_low/len(hvs)*100:.0f}%) convergen a óptimo local')
print(f'    → NPV medio: {npvs[mask_low].mean():,.0f} USD')
print(f'  • {n_high} semillas ({n_high/len(hvs)*100:.0f}%) encuentran el frente real')
print(f'    → NPV medio: {npvs[mask_high].mean():,.0f} USD')
print()
print('Causa probable: parada temprana (period=20) con N_gen=40')
print('detiene el GA en el 50% de las generaciones si no hay mejora,')
print('atrapándolo en óptimos locales antes de explorar.')
print()
print('Experimentos recomendados a priorizar:')
print('  P3 (period=30 o sin parada) — para reducir convergencia prematura')
print('  P2 (eta_c bajo) — para aumentar exploración y escapar óptimos locales')