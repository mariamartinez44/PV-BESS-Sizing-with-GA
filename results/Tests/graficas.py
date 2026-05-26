"""
plot_weekly_dispatch.py
========================
Genera 5 figuras independientes, cada una con enero (izquierda) y julio (derecha).
Estilo limpio: líneas sólidas, sin rellenos, fondo blanco, grilla horizontal suave.

Figuras generadas:
  1. soc_weekly.png         — SOC (kWh)
  2. ppv_weekly.png         — P_pv (kW)
  3. curtailment_weekly.png — Curtailment (kWh)
  4. pb_weekly.png          — P_b (kW)
  5. pcpd_weekly.png        — P_c / P_d (kW)

Uso:
  python plot_weekly_dispatch.py
  python plot_weekly_dispatch.py --csv mi_solucion.csv --dpi 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Semanas de interés (horas 1-indexed, igual que el CSV)
# ──────────────────────────────────────────────────────────────────────────────
ENERO_INI, ENERO_FIN = 1,    168   # 1-ene .. 7-ene
JULIO_INI, JULIO_FIN = 4345, 4512  # 1-jul .. 7-jul  (día 181)

# ──────────────────────────────────────────────────────────────────────────────
# Colores (igual que imagen de referencia)
# ──────────────────────────────────────────────────────────────────────────────
COLOR_BLACK  = '#000000'
COLOR_RED    = '#CC0000'
COLOR_GREEN  = '#2E7D32'
COLOR_ORANGE = '#E65100'
COLOR_BLUE   = '#1565C0'

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def cargar_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Hour", "SOC", "Pc", "Pd", "Ppv", "Ppvmx", "Pb", "Ps"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: columnas faltantes en el CSV: {missing}\n"
                 f"Columnas disponibles: {list(df.columns)}")
    df = df.sort_values("Hour").reset_index(drop=True)
    df["curtail"] = (df["Ppvmx"] - df["Ppv"]).clip(lower=0)
    return df


def get_semana(df: pd.DataFrame, h_ini: int, h_fin: int) -> pd.DataFrame:
    sub = df[(df["Hour"] >= h_ini) & (df["Hour"] <= h_fin)].copy()
    if len(sub) == 0:
        sys.exit(f"ERROR: sin datos para horas {h_ini}–{h_fin}")
    sub["x"] = np.arange(len(sub))
    return sub


def estilo_ax(ax, n_horas: int, titulo: str, ylabel: str):
    ax.set_title(titulo, fontsize=9, fontweight='bold', pad=5)
    ax.set_ylabel(ylabel, fontsize=8)

    # Grilla horizontal suave
    ax.yaxis.grid(True, color='#E0E0E0', lw=0.6, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Bordes mínimos
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BBBBBB')
    ax.spines['bottom'].set_color('#BBBBBB')

    # Eje X: etiquetas cada 24h (cada día)
    ticks  = np.arange(0, n_horas + 1, 24)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
    ax.set_xlim(0, n_horas - 1)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_facecolor('white')


def guardar(fig, fname: str, dpi: int):
    fig.savefig(fname, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  Guardada: {fname}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Función genérica: figura con dos paneles (enero | julio)
# ──────────────────────────────────────────────────────────────────────────────

def figura_dos_paneles(df_en, df_ju, plot_fn,
                       titulo_en, titulo_ju, ylabel,
                       fname, dpi, figsize=(11, 3.2)):
    fig, (ax_en, ax_ju) = plt.subplots(
        1, 2, figsize=figsize, sharey=False, facecolor='white')
    fig.subplots_adjust(wspace=0.30, left=0.07, right=0.97,
                        top=0.87, bottom=0.14)

    plot_fn(ax_en, df_en)
    estilo_ax(ax_en, n_horas=len(df_en), titulo=titulo_en, ylabel=ylabel)

    plot_fn(ax_ju, df_ju)
    estilo_ax(ax_ju, n_horas=len(df_ju), titulo=titulo_ju, ylabel="")

    guardar(fig, fname, dpi)


# ──────────────────────────────────────────────────────────────────────────────
# Funciones de dibujo por variable
# ──────────────────────────────────────────────────────────────────────────────

def plot_soc(ax, df):
    ax.plot(df["x"].values, df["SOC"].values,
            color=COLOR_BLACK, lw=1.3)


def plot_ppv(ax, df):
    ax.plot(df["x"].values, df["Ppv"].values,
            color=COLOR_RED, lw=1.3)


def plot_curtail(ax, df):
    # Barras verticales negras delgadas (como la imagen de referencia)
    ax.bar(df["x"].values, df["curtail"].values,
           width=0.85, color=COLOR_BLACK, linewidth=0)


def plot_pbps(ax, df):
    ax.plot(df["x"].values, df["Pb"].values,
            color=COLOR_GREEN, lw=1.3, label="$P_b$ (vendida)")
    ax.plot(df["x"].values, df["Ps"].values,
            color=COLOR_RED, lw=1.3, label="$P_s$ (comprada)")
    ax.legend(fontsize=7.5, loc='upper right',
              framealpha=0.8, handlelength=1.5, ncol=2,
              borderpad=0.5, labelspacing=0.3)


def plot_pcpd(ax, df):
    ax.plot(df["x"].values, df["Pc"].values,
            color=COLOR_ORANGE, lw=1.3, label="$P_c$ (carga)")
    ax.plot(df["x"].values, df["Pd"].values,
            color=COLOR_BLUE,   lw=1.3, label="$P_d$ (descarga)")
    ax.legend(fontsize=7.5, loc='upper right',
              framealpha=0.8, handlelength=1.5, ncol=2,
              borderpad=0.5, labelspacing=0.3)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gráficas semanales de despacho NSGA2 PV+BESS")
    parser.add_argument("--csv", default="solution_nsga2_E3E5_hourly.csv",
                        help="CSV horario generado por NSGA2_8760.py")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Resolución de salida (default: 150)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"ERROR: no se encontró '{csv_path}'.\n"
                 f"Asegúrate de correr primero NSGA2_8760.py para generar el CSV.")

    print(f"Leyendo {csv_path} ...")
    df = cargar_csv(csv_path)
    print(f"  {len(df)} horas cargadas  "
          f"(rango: hora {int(df['Hour'].min())} – {int(df['Hour'].max())})")

    df_en = get_semana(df, ENERO_INI, ENERO_FIN)
    df_ju = get_semana(df, JULIO_INI, JULIO_FIN)

    print(f"  Enero: horas {int(df_en['Hour'].min())}–{int(df_en['Hour'].max())}"
          f"  ({len(df_en)} filas)")
    print(f"  Julio: horas {int(df_ju['Hour'].min())}–{int(df_ju['Hour'].max())}"
          f"  ({len(df_ju)} filas)")

    print("\nGenerando figuras...")

    figura_dos_paneles(
        df_en, df_ju, plot_soc,
        titulo_en="SOC (kWh) - First week January",
        titulo_ju="SOC (kWh) - First week July",
        ylabel="SOC (kWh)",
        fname="soc_weekly.png", dpi=args.dpi,
    )

    figura_dos_paneles(
        df_en, df_ju, plot_ppv,
        titulo_en="$P_{pv}$ (kW) - First week January",
        titulo_ju="$P_{pv}$ (kW) - First week July",
        ylabel="$P_{pv}$ (kW)",
        fname="ppv_weekly.png", dpi=args.dpi,
    )

    figura_dos_paneles(
        df_en, df_ju, plot_curtail,
        titulo_en="Curtailment (kWh) - First week January",
        titulo_ju="Curtailment (kWh) - First week July",
        ylabel="Curtailment (kWh)",
        fname="curtailment_weekly.png", dpi=args.dpi,
    )

    figura_dos_paneles(
        df_en, df_ju, plot_pbps,
        titulo_en="$P_{b}$/$P_{s}$ (kW) - First week January",
        titulo_ju="$P_{b}$/$P_{s}$ (kW) - First week July",
        ylabel="Potencia (kW)",
        fname="pbps_weekly.png", dpi=args.dpi,
    )

    figura_dos_paneles(
        df_en, df_ju, plot_pcpd,
        titulo_en="$P_{c}$/$P_{d}$ (kW) - First week January",
        titulo_ju="$P_{c}$/$P_{d}$ (kW) - First week July",
        ylabel="Potencia (kW)",
        fname="pcpd_weekly.png", dpi=args.dpi,
        figsize=(11, 3.5),
    )

    print("\nListo. Archivos generados:")
    for f in ["soc_weekly.png", "ppv_weekly.png", "curtailment_weekly.png",
              "pb_weekly.png", "pcpd_weekly.png"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()