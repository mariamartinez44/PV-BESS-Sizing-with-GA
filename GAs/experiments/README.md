# Scripts de Experimentos — NSGA2 PV+BESS

## Archivos

| Archivo | Función |
|---|---|
| `exp_config.py` | Configuración global (editar antes de correr) |
| `exp_runner.py` | Utilidades compartidas (HV, checkpoint, estadísticas) |
| `exp_D1_seeds.py` | D1: robustez estadística (165 semillas sobre baseline) |
| `exp_P1_population.py` | P1: tamaño de población (30 / 50 / 60 / 80) |
| `exp_P2_crossover_mutation.py` | P2: patrones cruce-mutación (8 combinaciones) |
| `exp_P3_early_stopping.py` | P3: periodo de parada temprana (10 / 20 / 30 / ∞) |
| `exp_analyze.py` | Análisis final: resumen, ranking, gráficas consolidadas |

## Configuración inicial (OBLIGATORIO)

Edita `exp_config.py`:

```python
NSGA2_MODULE = 'nsga2_pv_bess'  # nombre de tu archivo principal sin .py
N_VAR        = 11                # variables activas del cromosoma
REF_POINT    = np.array([0.0, 15_000_000.0])  # ajustar con valores reales
```

El `REF_POINT` para el hipervolumen debe ser peor que cualquier solución:
- `REF_POINT[0]` = 0.0 (F1 = -OPEX; el peor OPEX es 0)
- `REF_POINT[1]` = CAPEX máximo esperado en USD (p.ej. 15 millones)

## Orden de ejecución recomendado

```bash
# 1. Primero D1 para saber el CV y calibrar el REF_POINT
python exp_D1_seeds.py

# 2. Actualiza REF_POINT en exp_config.py con los valores observados
# 3. Luego los experimentos de parámetros (independientes entre sí)
python exp_P1_population.py
python exp_P2_crossover_mutation.py
python exp_P3_early_stopping.py

# 4. Análisis final (lee los CSVs de todos los experimentos)
python exp_analyze.py
```

## Checkpoint (reanudación)

Cada script guarda su progreso en `results/checkpoint_<EXP>.json`.
Si el proceso se interrumpe, vuelve a lanzar el mismo script y
saltará automáticamente las corridas ya completadas:

```bash
python exp_P2_crossover_mutation.py          # reanuda desde donde quedó
python exp_P2_crossover_mutation.py --no-resume  # empieza desde cero
```

## Salidas por experimento

### D1
```
results/
  D1_seeds.csv          # una fila por semilla: HV, NPV, t_min, n_gen
  D1_stats.csv          # media, mediana, σ, CV%, IC95, semilla representativa
  figures/
    D1_boxplot.png      # boxplot + histograma del HV
    conv_D1-baseline_s<N>.png  # convergencia por semilla
```

### P1
```
results/
  P1_all_runs.csv       # una fila por (variante, semilla)
  P1_stats.csv          # estadísticos + p-valor Wilcoxon vs baseline pop=60
  figures/
    P1_boxplot.png      # boxplot comparativo con anotaciones de significancia
    P1_efficiency.png   # mediana HV y HV/tiempo vs pop_size
```

### P2
```
results/
  P2_all_runs.csv
  P2_stats.csv
  figures/
    P2_boxplot.png           # boxplot de 8 patrones
    P2_heatmap_median_hv.png # mapa de calor η_c × η_m (p_c=0.90)
```

### P3
```
results/
  P3_all_runs.csv
  P3_stats.csv
  figures/
    P3_boxplot.png       # boxplot por periodo
    P3_hv_vs_time.png    # trade-off calidad vs tiempo
```

### Análisis final
```
results/
  summary_all.csv              # tabla consolidada de todos los experimentos
  figures/
    summary_all_boxplots.png   # un panel por grupo
    ranking_median_hv.png      # barras horizontales ordenadas por mediana HV
```

## Interpretación del CV (D1)

| CV | Interpretación | Acción |
|---|---|---|
| < 1% | GA muy robusto | Una réplica por configuración es suficiente |
| 1–3% | Varianza moderada | ≥ 3 réplicas por configuración |
| > 3% | Alta varianza | ≥ 10 réplicas, usar test de Wilcoxon |

## Frente representativo

El **frente de Pareto final** a reportar no es el de la ejecución con
mayor HV, sino el de la ejecución cuyo HV es más cercano a la **mediana**.
Esta semilla se reporta en `seed_rep` y `hv_rep` en los CSVs de stats.

## Dependencias

```
pymoo >= 0.6
numpy
scipy
matplotlib
```
