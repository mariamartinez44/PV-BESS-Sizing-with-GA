# Prueba Final — 165 Semillas con Configuración Ganadora

## Qué hace este script

Corre el NSGA-II con la **configuración ganadora** de todos los experimentos
de hiperparámetros (P1, P2, P3) sobre **165 semillas independientes** para
validar estadísticamente la robustez del sistema.

### Configuración final

| Parámetro  | D1 original | **FINAL** | Fuente |
|------------|-------------|-----------|--------|
| `pop_size` | 60          | **80**    | P1     |
| `n_max_gen`| 40          | 40        | —      |
| `pc`       | 0.90        | **0.80**  | P2-F   |
| `eta_c`    | 15          | **10**    | P2-F   |
| `eta_m`    | 20          | **5**     | P2-F   |
| `n_cores`  | 4           | **16**    | nuevo equipo |
| `N_SEEDS`  | 1 (config)  | **165**   | regla TCL |

---

## Instalación en el equipo nuevo

```bash
# 1. Clona / copia los archivos del proyecto
cp -r "Códigos Algoritmo Genético/Modelos GA/GA/" ~/proyecto_ga/
cd ~/proyecto_ga

# 2. Copia los dos nuevos archivos a la carpeta de experimentos
cp exp_config.py          experimentos_nsga2/exp_config.py   # REEMPLAZA el original
cp run_final_165seeds.py  experimentos_nsga2/

# 3. Instala dependencias (si no están instaladas)
pip install pymoo matplotlib numpy scipy
```

---

## Cómo correr

```bash
cd experimentos_nsga2/

# Opción A — usa 16 cores (definidos en exp_config.py)
python run_final_165seeds.py

# Opción B — sobreescribe cores en runtime sin tocar el config
python run_final_165seeds.py --cores 12

# Opción C — empieza desde cero ignorando cualquier checkpoint previo
python run_final_165seeds.py --no-resume

# Reanudar tras una interrupción (comportamiento por defecto)
python run_final_165seeds.py --resume
```

### El script guarda checkpoint tras **cada semilla**

Si se interrumpe (apagado, Ctrl+C, etc.) simplemente vuelve a correrlo
y retoma donde se quedó.

---

## Estimación de tiempo en equipo de 32 GB

Basado en el experimento D1 previo (**6.34 min/semilla** con pop=60, 4 cores):

| Cores usados | Eficiencia paralela | Min/semilla | **Total 165 seeds** |
|:---:|:---:|:---:|:---:|
| 4  | baseline | ~8.5 min | **~23 h** |
| 8  | 60 %     | ~7.0 min | **~19 h** |
| 12 | 65 %     | ~4.3 min | **~12 h** |
| **16** | **65 %** | **~3.3 min** | **~9 h** ✓ |

> **Recomendación:** usa `--cores 16` si la máquina tiene ≥ 16 núcleos físicos
> (no lógicos/hyperthreading). Si el sistema tiene 8 físicos + HT, prueba `--cores 8`.

---

## Salidas generadas

```
results/
├── FINAL_seeds.csv          ← una fila por semilla (HV, NPV, tiempo, etc.)
├── FINAL_stats.csv          ← media, mediana, CV%, IC95 del HV
└── figures/
    ├── FINAL_boxplot.png    ← boxplot + histograma del HV
    └── conv_FINAL-config_s<N>.png   ← curva de convergencia por semilla
```

---

## Interpretación del CV%

| CV%   | Significado |
|-------|-------------|
| < 1%  | GA muy robusto; una sola réplica es suficiente para comparar |
| 1–3%  | Varianza moderada; usar ≥ 3 réplicas por experimento |
| > 3%  | Alta varianza; revisar hiperparámetros o aumentar generaciones |

El experimento D1 (configuración anterior) obtuvo **CV = 19.9%**.
Con la nueva configuración (especialmente `eta_m=5`) se espera un CV notablemente menor.
