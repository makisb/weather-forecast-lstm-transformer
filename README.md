# Explainable Multi-Step Weather Forecasting with LSTM & Transformer

> Multi-step temperature forecasting using ERA5-Land gridded data from the Achaia region, Greece.

---

## Description

A deep learning pipeline for **temperature forecasting** across **1 / 3 / 7 day horizons**, based on daily ERA5-Land gridded data from the Achaia region. The project compares LSTM and Transformer architectures against a persistence baseline, with SHAP-based explainability.

**Models:**
- **Persistence Baseline** — naive forecast (value from N days ago)
- **LSTM** — vanilla LSTM
- **Transformer** — Transformer encoder for time series

**Evaluation metrics:** RMSE · MAE · MAPE  
**Explainability:** SHAP (KernelExplainer) for feature importance analysis

---

## Project Structure

```
.
├── convlstm_transformer_experiment.ipynb   # Main notebook
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset

This project uses the **[Gridded Agro-Meteorological Data in Europe](https://data.jrc.ec.europa.eu/dataset/jrc-marsop4-7-weather_obs_grid_2019)** — JRC MARS Meteorological Database, European Commission Joint Research Centre.

- **DOI:** [10.2905/JRC.WG6925R](https://doi.org/10.2905/JRC.WG6925R)
- **Source:** Weather station observations interpolated on a 25×25 km grid, daily data from 1979
- **License:** European Commission reuse notice (free reuse with source acknowledgement)
- **Access:** Registration required at the [AGRI4CAST Data Portal](http://agri4cast.jrc.ec.europa.eu/DataPortal/Index.aspx)

> ⚠️ The dataset is **not included** in this repository due to file size.  
> Download the zip (`Dataset_Gridded_daily_agr.zip`) and place it in the project root.

**Citation:**
> Toreti, Andrea (2026): Gridded Agro-Meteorological Data in Europe. European Commission, Joint Research Centre [Dataset] doi: 10.2905/JRC.WG6925R

**Features (8 meteorological variables):**

| Variable | Description |
|---|---|
| `TEMPERATURE_MAX` | Maximum temperature (°C) |
| `TEMPERATURE_MIN` | Minimum temperature (°C) |
| `TEMPERATURE_AVG` | Mean temperature (°C) — **target** |
| `WINDSPEED` | Wind speed (m/s) |
| `VAPOURPRESSURE` | Vapour pressure (hPa) |
| `PRECIPITATION` | Precipitation (mm) |
| `ET0` | Reference evapotranspiration (mm) |
| `RADIATION` | Solar radiation (kJ/m²) |

**Engineered features:** `sin_day`, `cos_day`, `temp_lag1/3/7`, `temp_roll7/30`

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/convlstm-transformer-weather.git
cd convlstm-transformer-weather
pip install -r requirements.txt
```

Or open directly in **Google Colab** (recommended for GPU access).

---

## Usage

1. Open `convlstm_transformer_experiment.ipynb` in Colab or Jupyter
2. Upload the dataset zip when prompted
3. Run cells in order

**Key parameters:**

```python
SEQ_LEN    = 30        # Input window (days)
BATCH_SIZE = 64
HORIZONS   = [1, 3, 7] # Forecast horizons
```

---

## Results

| Model | Horizon | MAE (°C) | RMSE (°C) | MAPE (%) |
|---|---|---|---|---|
| Persistence | 1-day | — | — | — |
| LSTM | 1-day | — | — | — |
| Transformer | 1-day | — | — | — |
| Persistence | 3-day | — | — | — |
| LSTM | 3-day | — | — | — |
| Transformer | 3-day | — | — | — |
| Persistence | 7-day | — | — | — |
| LSTM | 7-day | — | — | — |
| Transformer | 7-day | — | — | — |

> Fill in after running the notebook.

---

## Visualizations

The following plots are saved automatically:

- `timeseries_overview.png` — Time series of all features
- `correlation_matrix.png` — Correlation heatmap
- `seasonality.png` — Seasonal patterns
- `predicted_vs_actual.png` — Predictions vs actual values
- `scatter_plot.png` — Scatter plots per model & horizon
- `error_comparison.png` — MAE/RMSE comparison
- `shap_comparison.png` — Feature importance (SHAP)

---

## Requirements

See `requirements.txt`. Main libraries:

- PyTorch ≥ 2.0
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- shap

---

## License

Apache License 2.0
