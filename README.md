# 🏀 March Madness 2026 — Prediction Dashboard

**🇬🇧 [English](#english)** | **🇪🇸 [Español](#español)**

---

<a id="english"></a>
## 🇬🇧 English

Interactive Streamlit dashboard that predicts NCAA March Madness matchup outcomes using a
trained machine learning model. Built for the Kaggle "March Machine Learning Mania 2026"
competition, as the final deliverable of the *Machine Learning* course (Universidad Europea).

> **Scope of this repository:** this repo contains the **dashboard / model-serving app** only.
> Feature engineering and model training (forward-chaining cross-validation, model selection,
> etc.) were done in **Databricks notebooks**, which are not included here — see
> [Project background](#project-background) below.

### Features

- **Match prediction** — pick any two teams and get a live win-probability prediction, with the
  key statistical factors driving the model's decision.
- **Title favorites** — estimated championship probability for the top 20 contenders.
- **Model accuracy (past seasons)** — backtests the model against real 2023–2025 tournament
  results to show how well its predictions held up.
- **Model metrics** — log loss, AUC and a plain-English explanation of what each metric means.

Both the men's and women's tournaments are supported, each served from its own trained model.

### Screenshots

![Match prediction view](docs/screenshots/match_prediction.png)
![Title favorites view](docs/screenshots/title_favorites.png)

*(add your own screenshots to `docs/screenshots/` — see that folder for naming suggestions)*

### Project structure

```
.
├── dashboard.py           # Streamlit app entry point
├── requirements.txt
├── .streamlit/
│   └── config.toml        # app theme
├── models/                # trained model bundles (not tracked in git — see models/README.md)
│   └── README.md
├── data/                  # teams/seeds/results data (not tracked in git — see data/README.md)
│   └── README.md
└── docs/
    └── screenshots/
```

### Getting started

**1. Clone and install**

```bash
git clone https://github.com/clarakhma/march-madness-dashboard.git
cd march-madness-dashboard
pip install -r requirements.txt
```

**2. Add the model and data files**

This repo ships code only — the trained model bundles and data tables are too large for GitHub
(one file alone is ~240 MB). Place them locally as:

```
models/bundle_men.pkl
models/bundle_women.pkl
data/data_men.pkl
data/data_women.pkl
```

See [`models/README.md`](models/README.md) and [`data/README.md`](data/README.md) for details on
what each file contains and how to regenerate them.

**3. Run the app**

```bash
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Project background

This dashboard is the presentation layer of a larger pipeline:

1. **Data & feature engineering** (Databricks) — team stats, ratings, seeds and head-to-head
   history are assembled into model-ready features from the Kaggle competition data.
2. **Model training & selection** (Databricks) — several candidate models are evaluated with
   forward-only cross-validation (never trained on future seasons) and the best performer is
   exported as a `ModelBundle`.
3. **Serving (this repo)** — the Streamlit app loads the exported bundles and turns model output
   into an interactive, non-technical dashboard.

| | Men's tournament | Women's tournament |
|---|---|---|
| CV Log Loss | 0.5668 | 0.4414 |
| CV AUC | 0.7707 | n/a |
| Test Log Loss | 0.5504 | 0.4103 |
| Test AUC | 0.7997 | 0.8981 |

### Tech stack

- [Streamlit](https://streamlit.io/) — app framework
- [Plotly](https://plotly.com/python/) — charts
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling
- [scikit-learn](https://scikit-learn.org/) — trained model

### License

MIT — see [LICENSE](LICENSE).

---

<a id="español"></a>
## 🇪🇸 Español

Dashboard interactivo hecho con Streamlit que predice los resultados de enfrentamientos del
March Madness de la NCAA usando un modelo de machine learning entrenado. Desarrollado para la
competición de Kaggle "March Machine Learning Mania 2026", como entregable final de la
asignatura *Machine Learning* (Universidad Europea).

> **Alcance de este repositorio:** este repo contiene únicamente el **dashboard / la app que
> sirve el modelo**. La ingeniería de características y el entrenamiento del modelo
> (validación cruzada forward-chaining, selección de modelo, etc.) se hicieron en
> **notebooks de Databricks**, que no están incluidos aquí — ver
> [Contexto del proyecto](#contexto-del-proyecto) más abajo.

### Funcionalidades

- **Predicción de partido** — elige dos equipos cualquiera y obtén una predicción de
  probabilidad de victoria en vivo, junto con los factores estadísticos clave que influyen en
  la decisión del modelo.
- **Favoritos al título** — probabilidad estimada de ser campeón para los 20 mejores candidatos.
- **Precisión del modelo (temporadas pasadas)** — compara las predicciones del modelo contra
  resultados reales del torneo entre 2023 y 2025 para mostrar qué tan acertadas fueron.
- **Métricas del modelo** — log loss, AUC y una explicación en lenguaje sencillo de lo que
  significa cada métrica.

Están soportados tanto el torneo masculino como el femenino, cada uno servido desde su propio
modelo entrenado.

### Capturas de pantalla

![Vista de predicción de partido](docs/screenshots/match_prediction.png)
![Vista de favoritos al título](docs/screenshots/title_favorites.png)

*(añade tus propias capturas en `docs/screenshots/` — revisa esa carpeta para sugerencias de
nombres)*

### Estructura del proyecto

```
.
├── dashboard.py           # Punto de entrada de la app de Streamlit
├── requirements.txt
├── .streamlit/
│   └── config.toml        # tema visual de la app
├── models/                # bundles del modelo entrenado (no versionados — ver models/README.md)
│   └── README.md
├── data/                  # datos de equipos/seeds/resultados (no versionados — ver data/README.md)
│   └── README.md
└── docs/
    └── screenshots/
```

### Cómo empezar

**1. Clonar e instalar**

```bash
git clone https://github.com/clarakhma/march-madness-dashboard.git
cd march-madness-dashboard
pip install -r requirements.txt
```

**2. Añadir los archivos de modelo y datos**

Este repositorio solo incluye el código — los bundles del modelo entrenado y las tablas de
datos son demasiado grandes para GitHub (un solo archivo pesa ~240 MB). Colócalos localmente
así:

```
models/bundle_men.pkl
models/bundle_women.pkl
data/data_men.pkl
data/data_women.pkl
```

Consulta [`models/README.md`](models/README.md) y [`data/README.md`](data/README.md) para más
detalles sobre qué contiene cada archivo y cómo regenerarlos.

**3. Ejecutar la app**

```bash
streamlit run dashboard.py
```

Abre [http://localhost:8501](http://localhost:8501) en tu navegador.

### Contexto del proyecto

Este dashboard es la capa de presentación de un pipeline más amplio:

1. **Datos e ingeniería de características** (Databricks) — las estadísticas de equipos,
   ratings, seeds e historial de enfrentamientos se transforman en características listas para
   el modelo a partir de los datos de la competición de Kaggle.
2. **Entrenamiento y selección del modelo** (Databricks) — varios modelos candidatos se evalúan
   con validación cruzada forward-only (nunca se entrena con datos de temporadas futuras) y el
   mejor se exporta como un `ModelBundle`.
3. **Servido (este repositorio)** — la app de Streamlit carga los bundles exportados y
   convierte la salida del modelo en un dashboard interactivo y no técnico.

| | Torneo masculino | Torneo femenino |
|---|---|---|
| CV Log Loss | 0.5668 | 0.4414 |
| CV AUC | 0.7707 | n/a |
| Test Log Loss | 0.5504 | 0.4103 |
| Test AUC | 0.7997 | 0.8981 |

### Stack tecnológico

- [Streamlit](https://streamlit.io/) — framework de la app
- [Plotly](https://plotly.com/python/) — gráficos
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — manejo de datos
- [scikit-learn](https://scikit-learn.org/) — modelo entrenado

### Licencia

MIT — ver [LICENSE](LICENSE).
