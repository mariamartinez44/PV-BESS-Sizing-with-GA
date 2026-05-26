"""
fig_dispatch_E3E5.py
====================
Generates Fig. X — Layered illustration of the E3+E5 heuristic dispatch
strategy for a representative winter day.

Data source: optimal NSGA-II chromosome (max-NPV solution)
  C_bess=10,124 kWh, P_inv=3,770 kW, P_pv=3,034 kWp

Dependencies: matplotlib, numpy
Usage:
    python fig_dispatch_E3E5.py
Outputs:
    fig_dispatch_E3E5.pdf  (vector, for LaTeX)
    fig_dispatch_E3E5.png  (300 dpi, for preview)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── Colour palette (matches paper) ──────────────────────────────────────────
C_AZUL   = '#0070C0'
C_ROJO   = '#C0392B'
C_VERDE  = '#1E8449'
C_AMAR   = '#B7950B'
C_GRIS   = '#BDC3C7'
C_NEGRO  = '#2C3E50'
C_MORADO = '#6C3483'

# ── Optimal chromosome parameters ───────────────────────────────────────────
N_CYCLES   = 2
N_H_C      = 6       # charge window hours per cycle
N_H_D      = 5       # discharge window hours per cycle
SPREAD_MIN = 0.0
ALPHA      = 2.129   # sigmoid slope
BETA       = 0.890   # PV surplus bonus
GAMMA      = 0.173   # SOC correction weight

C_BESS  = 10124   # kWh — BESS capacity
P_INV   = 3770    # kW  — BESS inverter power
DOD     = 0.90
BETA_MIN = (1 - DOD) / 2       # 0.05
BETA_MAX = BETA_MIN + DOD       # 0.95
SOC_INIT = 0.50 * C_BESS
ETA_C = ETA_D = 0.9624

# ── Winter day prices (EUR/kWh → EUR/MWh ×1000) ─────────────────────────────
lam = np.array([63.33, 50.09, 47.50, 43.50, 42.50, 42.09,
                42.50, 42.59, 43.37, 42.29, 25.00,  3.90,
                 3.20,  2.06,  1.73,  5.72, 18.49, 37.00,
                47.50, 54.97, 60.90, 60.00, 47.50, 42.09])
hours    = np.arange(1, 25)
lam_med  = np.median(lam)

# ── Strategy E3: window assignment ──────────────────────────────────────────
spread = lam.max() - lam.min()
n_c = N_H_C * N_CYCLES   # 12 total charge hours
n_d = N_H_D * N_CYCLES   # 10 total discharge hours

rank_asc  = np.argsort(lam)           # cheapest first
rank_desc = np.argsort(lam)[::-1]     # most expensive first

charge_idx    = set(rank_asc[:n_c].tolist())
discharge_idx = set(rank_desc[:n_d].tolist())

# Causality check: discharge hour accepted only if ≥1 charge hour precedes it
# This rejects early-morning high-price hours (h1-h4) because no charge
# has occurred yet at the start of the day.
final_discharge = set()
causality_rejected = set()
for d in sorted(discharge_idx):
    if any(c < d for c in charge_idx):
        final_discharge.add(d)
    else:
        causality_rejected.add(d)

# Convert to 1-based hour lists for plotting
charge_h    = sorted([i+1 for i in charge_idx])
discharge_h = sorted([i+1 for i in final_discharge])
rejected_h  = sorted([i+1 for i in causality_rejected])

print("=== E3 Window Assignment ===")
print(f"  Spread: {spread:.2f} EUR/MWh (> spread_min={SPREAD_MIN}) → BESS active")
print(f"  Charge hours    ({len(charge_h):2d}): {charge_h}")
print(f"  Discharge hours ({len(discharge_h):2d}): {discharge_h}")
print(f"  Causality rejected ({len(rejected_h)}): {rejected_h}")
print(f"  Note: h{rejected_h} are the highest-price hours but occur before")
print(f"        any charge window → E3 cannot schedule discharge there.")

# ── Strategy E5: sigmoid modulation ─────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

delta = (lam - lam_med) / max(abs(lam_med), 1e-6)
fc    = sigmoid(-ALPHA * delta)   # charge fraction
fd    = sigmoid( ALPHA * delta)   # discharge fraction

# ── Simulate hour-by-hour dispatch (E3 + E5) ────────────────────────────────
SOC = SOC_INIT
Pc  = np.zeros(24)
Pd  = np.zeros(24)
SOC_arr = np.zeros(24)

for i in range(24):
    h = i + 1
    SOC_norm = SOC / C_BESS

    # SOC correction factors (Eq. soc_corr)
    soc_corr_d = max(0, min(1, (SOC_norm - BETA_MIN) / (0.5 - BETA_MIN)))
    soc_corr_c = max(0, min(1, (BETA_MAX - SOC_norm) / (BETA_MAX - 0.5)))

    fc_corr = fc[i] * (GAMMA * soc_corr_c + (1 - GAMMA))
    fd_corr = fd[i] * (GAMMA * soc_corr_d + (1 - GAMMA))

    SOC_max = BETA_MAX * C_BESS
    SOC_min = BETA_MIN * C_BESS

    if h in charge_h:
        p = min(P_INV * fc_corr, (SOC_max - SOC) / ETA_C)
        p = max(0, p)
        Pc[i] = p
        SOC  += p * ETA_C
    elif h in discharge_h:
        p = min(P_INV * fd_corr, (SOC - SOC_min) * ETA_D)
        p = max(0, p)
        Pd[i] = p
        SOC  -= p / ETA_D

    SOC_arr[i] = SOC

print("\n=== Dispatch results ===")
print(f"  Total Wc = {Pc.sum():.0f} kWh")
print(f"  Total Wd = {Pd.sum():.0f} kWh")
print(f"  Hours with Pc>0: {[i+1 for i in range(24) if Pc[i]>0]}")
print(f"  Hours with Pd>0: {[i+1 for i in range(24) if Pd[i]>0]}")

# Hours whose SOC is full (charge assigned but Pc≈0)
soc_full_h = [h for h in charge_h
              if Pc[h-1] < 10 and SOC_arr[h-2 if h>1 else 0] > 0.90*C_BESS]

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9))
gs  = gridspec.GridSpec(3, 2, width_ratios=[1, 5.5], hspace=0.55, wspace=0.05)

ax_g = [fig.add_subplot(gs[i, 0]) for i in range(3)]
ax0  = fig.add_subplot(gs[0, 1])
ax1  = fig.add_subplot(gs[1, 1], sharex=ax0)
ax2  = fig.add_subplot(gs[2, 1], sharex=ax0)


def style_gene_panel(ax, title, color, genes):
    """Style the left gene-column panels."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(color + '18')
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(1.5)
    ax.text(0.5, 0.93, title, transform=ax.transAxes,
            ha='center', va='top', fontsize=8, fontweight='bold', color=color)
    for i, (gene, desc) in enumerate(genes):
        y = 0.76 - i * 0.21
        ax.text(0.06, y, gene, transform=ax.transAxes,
                ha='left', va='center', fontsize=7.5, fontweight='bold', color=color)
        if desc:
            ax.text(0.06, y - 0.09, desc, transform=ax.transAxes,
                    ha='left', va='center', fontsize=6.5, color=C_NEGRO)


style_gene_panel(ax_g[0], 'Strategy E3', C_AZUL, [
    ('$n_{\\mathrm{cycles}}=2$',    'cycles per day'),
    ('$n_{h,c}=6$',                 'charge hrs/cycle'),
    ('$n_{h,d}=5$',                 'discharge hrs/cycle'),
    ('$\\mathit{spread}_{\\min}=0$', 'activation threshold'),
])
style_gene_panel(ax_g[1], 'Strategy E5', C_MORADO, [
    ('$\\alpha=2.13$', 'sigmoid slope'),
    ('$\\gamma=0.17$', 'SOC correction'),
    ('$\\beta=0.89$',  'PV surplus bonus'),
])
style_gene_panel(ax_g[2], 'Physical output', C_ROJO, [
    ('$P_{c,t},\\,P_{d,t}$ (kW)', ''),
    ('$SOC_t$ (kWh)',              ''),
    ('$P^{\\mathrm{inv}}_{\\mathrm{bess}}=3{,}770$\\,kW', ''),
])


def add_window_shading(ax):
    """Shade charge/discharge/rejected windows across all panels."""
    for h in charge_h:
        ax.axvspan(h - 0.5, h + 0.5, alpha=0.09, color=C_AZUL, zorder=0)
    for h in discharge_h:
        ax.axvspan(h - 0.5, h + 0.5, alpha=0.09, color=C_ROJO, zorder=0)
    for h in rejected_h:
        ax.axvspan(h - 0.5, h + 0.5, alpha=0.06, color=C_AMAR, zorder=0)


# ── Panel 1: Price λ_t ───────────────────────────────────────────────────────
add_window_shading(ax0)
bar_colors = [C_AZUL if h in charge_h
              else (C_ROJO   if h in discharge_h
              else (C_AMAR   if h in rejected_h
              else  C_GRIS)) for h in hours]
ax0.bar(hours, lam, color=bar_colors, alpha=0.80, width=0.75,
        edgecolor='white', linewidth=0.4, zorder=2)
ax0.axhline(lam_med, color=C_AMAR, linestyle='--', linewidth=1.5, zorder=3)
ax0.text(24.7, lam_med, '$\\lambda^{\\mathrm{med}}$',
         va='center', fontsize=7, color=C_AMAR)
ax0.text(12, lam_med + 2.5,
         f'$\\Delta\\lambda={spread:.1f}>'
         f'\\mathit{{spread}}_{{\\min}}={SPREAD_MIN:.0f}$ $\\Rightarrow$ BESS active',
         ha='center', fontsize=7, color=C_AMAR, style='italic')

# Annotate causality-rejected hours
for h in rejected_h:
    ax0.annotate(f'h{h}\nrejected\n(causality)',
                 xy=(h, lam[h-1]),
                 xytext=(h, lam[h-1] + 4),
                 fontsize=5.5, color=C_AMAR, ha='center',
                 arrowprops=dict(arrowstyle='->', color=C_AMAR, lw=0.7))

ax0.set_ylabel('$\\lambda_t$ (EUR/MWh)', fontsize=8)
ax0.set_ylim(0, 80); ax0.set_yticks([0, 20, 40, 60])
ax0.tick_params(axis='y', labelsize=7)
ax0.tick_params(axis='x', labelbottom=False)
ax0.set_title('Layer 1 — Strategy E3:  daily price ranking & window assignment',
              fontsize=9, fontweight='bold', loc='left', pad=4)
ax0.legend(handles=[
    mpatches.Patch(color=C_AZUL,  alpha=0.8, label=f'CHARGE ($n_c={n_c}$ h assigned)'),
    mpatches.Patch(color=C_ROJO,  alpha=0.8, label=f'DISCHARGE ({len(discharge_h)} h accepted)'),
    mpatches.Patch(color=C_AMAR,  alpha=0.8, label=f'DISCHARGE rejected — causality ({len(rejected_h)} h)'),
    mpatches.Patch(color=C_GRIS,  alpha=0.8, label='IDLE'),
], loc='upper right', fontsize=7, framealpha=0.9)
ax0.spines['top'].set_visible(False); ax0.spines['right'].set_visible(False)

# ── Panel 2: Sigmoid fraction f_t ────────────────────────────────────────────
add_window_shading(ax1)
for i, h in enumerate(hours):
    if h in charge_h:
        is_full = (Pc[i] < 10)
        alpha_v = 0.35 if is_full else 0.85
        hatch_v = '//'  if is_full else None
        ax1.bar(h, fc[i], color=C_AZUL, alpha=alpha_v, width=0.75,
                edgecolor='white', linewidth=0.4, hatch=hatch_v, zorder=2)
    elif h in discharge_h:
        ax1.bar(h, fd[i], color=C_ROJO, alpha=0.85, width=0.75,
                edgecolor='white', linewidth=0.4, zorder=2)
    elif h in rejected_h:
        # Show the f_d value they would have had (but were rejected)
        ax1.bar(h, fd[i], color=C_AMAR, alpha=0.40, width=0.75,
                edgecolor=C_AMAR, linewidth=0.6, linestyle='--', zorder=2)
    else:
        ax1.bar(h, 0.5, color=C_GRIS, alpha=0.15, width=0.75,
                edgecolor=C_GRIS, linewidth=0.6, zorder=2)

ax1.axhline(0.5, color=C_AMAR, linestyle='--', linewidth=1.2, zorder=3)
ax1.text(24.7, 0.5, '$f=0.5$', va='center', fontsize=7, color=C_AMAR)

# Annotations
ax1.annotate('$f_c>0.87$ but\nSOC full $\\Rightarrow P_c=0$',
             xy=(15, fc[14]), xytext=(16.5, 0.97),
             fontsize=7, color=C_AZUL, ha='center',
             arrowprops=dict(arrowstyle='->', color=C_AZUL, lw=1.0))
ax1.annotate('$\\alpha$ controls\nslope steepness',
             xy=(21, fd[20]), xytext=(19.0, 0.83),
             fontsize=7, color=C_MORADO,
             arrowprops=dict(arrowstyle='->', color=C_MORADO, lw=1.0))
ax1.annotate('$f_d$ high but\ncausality rejected',
             xy=(1, fd[0]), xytext=(3.5, 0.85),
             fontsize=6.5, color=C_AMAR, ha='center',
             arrowprops=dict(arrowstyle='->', color=C_AMAR, lw=0.8))

ax1.set_ylabel('$f_t \\in [0,1]$', fontsize=8)
ax1.set_ylim(0, 1.15); ax1.set_yticks([0, 0.5, 1.0])
ax1.tick_params(axis='y', labelsize=7)
ax1.tick_params(axis='x', labelbottom=False)
ax1.set_title('Layer 2 — Strategy E5:  sigmoid power fraction modulation',
              fontsize=9, fontweight='bold', loc='left', pad=4)
ax1.legend(handles=[
    mpatches.Patch(color=C_AZUL, alpha=0.85, label='$f_c$ charge fraction'),
    mpatches.Patch(color=C_ROJO, alpha=0.85, label='$f_d$ discharge fraction'),
    mpatches.Patch(color=C_AZUL, alpha=0.35, hatch='//', label='SOC full $\\Rightarrow P_c=0$'),
    mpatches.Patch(color=C_AMAR, alpha=0.40, label='$f_d$ computed but E3-rejected'),
    mpatches.Patch(color=C_GRIS, alpha=0.3,  label='IDLE'),
], loc='upper right', fontsize=7, framealpha=0.9)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# ── Panel 3: Power Pc/Pd + SOC ───────────────────────────────────────────────
add_window_shading(ax2)
ax2.bar(hours,  Pc/1000, color=C_AZUL, alpha=0.85, width=0.75,
        edgecolor='white', linewidth=0.4, zorder=2, label='$P_{c,t}$ charge')
ax2.bar(hours, -Pd/1000, color=C_ROJO, alpha=0.85, width=0.75,
        edgecolor='white', linewidth=0.4, zorder=2, label='$-P_{d,t}$ discharge')
ax2.axhline(0, color=C_NEGRO, linewidth=0.8)
ax2.set_ylabel('$P_{c,t},\\,-P_{d,t}$ (MW)', fontsize=8)
ax2.set_ylim(-3.2, 2.4); ax2.set_yticks([-3, -2, -1, 0, 1, 2])
ax2.tick_params(axis='y', labelsize=7)

# SOC on secondary right axis
ax3r = ax2.twinx()
ax3r.plot(hours, SOC_arr/1000, color=C_VERDE, linewidth=2.0,
          drawstyle='steps-post', zorder=3, label='$SOC_t$',
          marker='o', markersize=2.5, markerfacecolor=C_VERDE)
ax3r.axhline(BETA_MAX * C_BESS / 1000, color=C_VERDE,
             linestyle='--', linewidth=0.8, alpha=0.6)
ax3r.axhline(BETA_MIN * C_BESS / 1000, color=C_VERDE,
             linestyle='--', linewidth=0.8, alpha=0.6)
ax3r.text(25.1, BETA_MAX * C_BESS / 1000, '95%', fontsize=7,
          color=C_VERDE, va='center')
ax3r.text(25.1, BETA_MIN * C_BESS / 1000, '5%',  fontsize=7,
          color=C_VERDE, va='center')
ax3r.set_ylabel('$SOC_t$ (MWh)', fontsize=8, color=C_VERDE)
ax3r.tick_params(axis='y', labelsize=7, colors=C_VERDE)
ax3r.set_ylim(-3.2, 14); ax3r.spines['right'].set_edgecolor(C_VERDE)

ax2.set_xlabel('Hour of day', fontsize=9)
ax2.set_xlim(0.5, 24.5); ax2.set_xticks(range(1, 25, 2))
ax2.tick_params(axis='x', labelsize=7)
ax2.set_title('Layer 3 — Physical output:  actual dispatch & state of charge',
              fontsize=9, fontweight='bold', loc='left', pad=4)
l1, lb1 = ax2.get_legend_handles_labels()
l2, lb2 = ax3r.get_legend_handles_labels()
ax2.legend(l1 + l2, lb1 + lb2, loc='upper right', fontsize=7, framealpha=0.9)
ax2.spines['top'].set_visible(False)

# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(0.985, 0.005,
         f'Winter day — $C_{{bess}}={C_BESS:,}$ kWh, '
         f'$P^{{inv}}_{{bess}}={P_INV:,}$ kW, '
         f'$P^{{inst}}_{{pv}}=3{{,}}034$ kWp. '
         f'Note: h{rejected_h} are highest-price hours but precede all '
         f'charge windows and are causality-rejected by E3.',
         ha='right', fontsize=6.5, color=C_GRIS, style='italic')

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ('pdf', 'png'):
    fname = f'fig_dispatch_E3E5.{ext}'
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    print(f'Saved: {fname}')

plt.close()
