# Optimización Bi-objetivo de Sistemas PV + BESS mediante Algoritmos Evolutivos

Repositorio de código y resultados para la optimización del tamaño (sizing) de sistemas fotovoltaicos con almacenamiento en baterías (PV + BESS), usando algoritmos evolutivos multi-objetivo. El problema se formula como una optimización bi-objetivo sobre una ventana representativa de operación, buscando maximizar el VPN del sistema y minimizar el curtailment energético.

---

## Estructura del repositorio

```
Modelos GA/
├── GAs/                          # Algoritmos evolutivos (versión final)
│   ├── NSGAII_heuristic_E35.py       # NSGA-II bi-objetivo (algoritmo principal)
│   ├── SPEAII_heuristic_E35.py       # SPEA-II bi-objetivo
│   ├── AntCollony_heuristic_E35.py   # Ant Colony Optimization
│   ├── ParticleSwarn_heuristic_E35.py# Particle Swarm Optimization
│   ├── grafica_vpn.py                # Visualización de soluciones VPN
│   └── experiments/                  # Suite de experimentos de hiperparámetros
│       ├── exp_config.py             # Configuración centralizada
│       ├── exp_runner.py             # Ejecutor de experimentos con checkpointing
│       ├── exp_D1_seeds.py           # Exp D1: estudio de semillas (baseline)
│       ├── exp_P1_population.py      # Exp P1: tamaño de población
│       ├── exp_P2_crossover_mutation.py  # Exp P2: cruce y mutación
│       ├── exp_P3_early_stopping.py  # Exp P3: criterio de parada temprana
│       ├── exp_analyze.py            # Análisis estadístico de resultados
│       └── run_final_165seeds.py     # Validación final con 165 semillas
│
├── GAs con LP/                   # 🚧 Trabajo futuro — integración con LP
│   ├── LP_model_funcional.py         # Modelo de Programación Lineal base
│   ├── GA_basic_model_funcional.py   # GA acoplado con LP
│   ├── NSGAII_model.py               # NSGA-II + LP (en desarrollo)
│   └── NSGAII_MPC.py                 # NSGA-II + Control Predictivo (en desarrollo)
│
├── ventana_representativa/       # Datos de la ventana de optimización (E3–E5)
│   ├── lambda_ventana.inc            # Precio de electricidad (€/kWh)
│   ├── Ppvu_ventana.inc              # Generación FV unitaria (kW/kWp)
│   ├── Plu_ventana.inc               # Demanda de carga (kW)
│   ├── psi_ventana.inc               # Parámetros auxiliares
│   ├── periodo_ventana.inc           # Horizonte temporal
│   └── mapa_reindexacion.txt         # Mapa de índices de la ventana
│
├── ventana_completa/             # Serie temporal completa (año entero)
│   ├── lambda_spain_localtime.inc    # Precio spot España (hora local)
│   ├── PpvuMadridSarah20052023.inc   # Generación FV Madrid (hora local)
│   ├── Plu.inc / Plu2.inc            # Demanda (variantes)
│   ├── PluDataCenter.inc             # Demanda data center
│   ├── psi.inc                       # Parámetros
│   └── periodo.inc                   # Horizonte completo
│
└── results/                      # Resultados y figuras
    ├── figures/                      # Frentes de Pareto y convergencia (finales)
    ├── Solutions/                    # CSVs con soluciones Pareto (NSGA-II y SPEA-II)
    └── Tests/                        # Experimentos de hiperparámetros
        ├── results_baseline_experiments/   # Resultados de D1, P1, P2, P3
        └── results_final_experiments/      # Validación final (165 semillas)
```

---

## Algoritmos implementados

| Algoritmo | Archivo | Estado |
|-----------|---------|--------|
| NSGA-II | `GAs/NSGAII_heuristic_E35.py` | ✅ Final |
| SPEA-II | `GAs/SPEAII_heuristic_E35.py` | ✅ Final |
| Ant Colony Optimization | `GAs/AntCollony_heuristic_E35.py` | ✅ Final |
| Particle Swarm Optimization | `GAs/ParticleSwarn_heuristic_E35.py` | ✅ Final |
| NSGA-II + LP | `GAs con LP/NSGAII_model.py` | 🚧 Trabajo futuro |
| NSGA-II + MPC | `GAs con LP/NSGAII_MPC.py` | 🚧 Trabajo futuro |

---

## Formulación del problema

El sistema optimiza simultáneamente dos objetivos sobre un horizonte de 20 años:

- **Objetivo 1 — Maximizar VPN** (Valor Presente Neto del proyecto)
- **Objetivo 2 — Minimizar Curtailment** (energía FV no aprovechada)

**Variables de decisión:** potencia instalada FV (kWp) y capacidad de almacenamiento BESS (kWh).

**Parámetros económicos principales:**

| Parámetro | Valor |
|-----------|-------|
| CAPEX FV | 388 €/kWp |
| CAPEX BESS | 185 €/kWh |
| CAPEX Inversor | 48 €/kVA |
| Tasa de descuento | 7.7 % |
| Escalación energía | 2.5 % |
| Vida útil | 20 años |
| Eficiencia carga/descarga BESS | 96.24 % |
| DoD máximo | 90 % |

---

## Configuración final del NSGA-II

Obtenida tras los experimentos de hiperparámetros D1 → P1 → P2 → P3:

| Parámetro | Baseline (D1) | **Final** | Experimento |
|-----------|:---:|:---:|:---:|
| `pop_size` | 60 | **80** | P1 |
| `n_max_gen` | 40 | 40 | — |
| `pc` (prob. cruce) | 0.90 | **0.80** | P2 |
| `eta_c` (dist. cruce SBX) | 15 | **10** | P2 |
| `eta_m` (dist. mutación PM) | 20 | **5** | P2 |

La configuración final fue validada sobre **165 semillas independientes** para garantizar robustez estadística (criterio del Teorema Central del Límite).

---

## Instalación

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd Modelos-GA

# Instalar dependencias
pip install pymoo matplotlib numpy scipy numpy-financial
```

**Dependencias principales:** `pymoo`, `numpy`, `matplotlib`, `scipy`, `numpy-financial`

---

## Uso

### Ejecutar NSGA-II principal

```bash
cd "GAs"
python NSGAII_heuristic_E35.py
```

### Ejecutar experimentos de hiperparámetros

```bash
cd "GAs/experiments"

# Experimento D1: estudio de semillas baseline
python exp_D1_seeds.py

# Experimento P1: tamaño de población
python exp_P1_population.py

# Experimento P2: cruce y mutación
python exp_P2_crossover_mutation.py

# Experimento P3: parada temprana
python exp_P3_early_stopping.py

# Analizar resultados
python exp_analyze.py
```

### Ejecutar validación final (165 semillas)

```bash
cd "GAs/experiments"

# Ejecución estándar (usa 16 cores por defecto)
python run_final_165seeds.py

# Especificar número de cores
python run_final_165seeds.py --cores 8

# Reanudar ejecución interrumpida (comportamiento por defecto)
python run_final_165seeds.py --resume

# Comenzar desde cero
python run_final_165seeds.py --no-resume
```

> El script guarda un checkpoint tras cada semilla. Si se interrumpe, se puede retomar sin perder progreso.

---

## Resultados

Los resultados finales se encuentran en `results/`:

- **`results/figures/`** — Frentes de Pareto y curvas de convergencia para NSGA-II y SPEA-II
- **`results/Solutions/`** — Soluciones Pareto en CSV (hourly y summary) para ambos algoritmos
- **`results/Tests/results_final_experiments/`** — Estadísticas de las 165 semillas (media, mediana, CV%, IC95 del Hypervolume)

### Interpretación del CV% del Hypervolume

| CV% | Robustez del algoritmo |
|-----|------------------------|
| < 1% | Muy robusto — una réplica es suficiente para comparar |
| 1–3% | Varianza moderada — usar ≥ 3 réplicas |
| > 3% | Alta varianza — revisar hiperparámetros |

---

## Trabajo futuro

La carpeta `GAs con LP/` contiene los desarrollos en curso para integrar los algoritmos evolutivos con modelos de **Programación Lineal (LP)** y **Control Predictivo (MPC)**. Estos modelos están en versión preliminar y no deben usarse como referencia de resultados.

---

## Datos de entrada

Los archivos `.inc` en `ventana_representativa/` y `ventana_completa/` contienen las series temporales del sistema (precios, generación FV, demanda) para Madrid, España (datos 2023). La ventana representativa corresponde a las semanas E3–E5 del año.
