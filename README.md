# Indonesian SST Anomaly Prediction with LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-red)
![Oceanography](https://img.shields.io/badge/Domain-Oceanography-teal)

## Project Overview
Proyek ini memprediksi **Anomali Suhu Permukaan Laut (SST)** di perairan Indonesia menggunakan **Multivariate LSTM**. Model memanfaatkan **Niño 3.4 Index** sebagai prediktor eksternal untuk menangkap fenomena **El Niño-Southern Oscillation (ENSO)**.

---

## Scientific Background
| Dampak SST Anomaly | Penjelasan |
|-------------------|------------|
| 🪸 **Coral Bleaching** | Anomali positif ekstrem menyebabkan pemutihan karang |
| 🐟 **Perikanan** | Suhu mempengaruhi migrasi ikan dan upwelling |
| 🌏 **Telekoneksi Iklim** | El Niño/La Niña di Pasifik mempengaruhi Indonesia dengan time lag |

---

## Tech Stack
- **Deep Learning:** PyTorch (LSTM)
- **Data Processing:** Xarray, Pandas, NumPy
- **Visualization:** Matplotlib

---

## Data Sources
| Data | Sumber | Fungsi |
|------|--------|--------|
| Indonesian SST | NOAA OISST V2 | Target (Y) |
| Niño 3.4 Index | NOAA ERSSTv5 | Feature/Predictor (X) |

---

## Project Structure

```
enso-forecasting/
├── download_data.py         # Download NetCDF dari NOAA
├── preprocessing.py         # ETL: NetCDF → CSV (slice, resample, anomaly)
├── modeling.py              # Univariate LSTM (SST only)
├── multivariate_modeling.py # Multivariate LSTM (SST + Niño 3.4)
├── validation_2012.py       # Out-of-Sample Testing (Train: 2000-2011, Test: 2012)
├── sst_indo_clean.csv       # Processed SST data
└── nina34.anom.data.txt     # Niño 3.4 Index from NOAA
```

---

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Type | Multivariate LSTM |
| Input Features | 2 (SST Indo + Niño 3.4) |
| Lookback Window | 12 months |
| Hidden Size | 32 |
| Output | 1 (Indonesian SST Anomaly) |

---

## Results

### Out-of-Sample Validation (Year 2012)
Model trained on **2000-2011**, tested on **2012** (12 months never seen during training).

![Validation Results](validation_2012_results.png)

### Multivariate Prediction with El Niño Highlighting
![Multivariate Results](multivariate_lstm_results.png)

---

## How to Run

```bash
# 1. Clone repository
git clone https://github.com/lainx86/enso-forecasting.git
cd enso-forecasting

# 2. Install dependencies
pip install xarray netCDF4 pandas numpy torch matplotlib scikit-learn

# 3. Run preprocessing (if starting fresh)
python preprocessing.py

# 4. Train & evaluate (choose one)
python multivariate_modeling.py    # Standard 80/20 split
python validation_2012.py          # Out-of-sample 2012 validation
```

---

## Oceanographic Insight
- **El Niño Phase:** Niño 3.4 > 1.0°C → Indonesian waters typically **cool**
- **La Niña Phase:** Niño 3.4 < -1.0°C → Indonesian waters typically **warm**
- The model learns this **teleconnection** pattern from the Niño 3.4 predictor

---

*Project ini dibuat sebagai eksplorasi Data Science di bidang Oseanografi.*