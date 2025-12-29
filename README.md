# Indonesian SST Anomaly Prediction with LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-red)
![Oceanography](https://img.shields.io/badge/Domain-Oceanography-teal)

## Project Overview
Proyek ini memprediksi **Anomali Suhu Permukaan Laut (SST)** di perairan Indonesia menggunakan **Multivariate LSTM**. Model memanfaatkan **Niño 3.4 Index** sebagai prediktor eksternal untuk menangkap fenomena **El Niño-Southern Oscillation (ENSO)**.

---

## Project Structure

```
enso-forecasting/
├── data/
│   ├── raw/                    # Raw external data
│   │   └── nina34.anom.data.txt
│   └── processed/              # Processed data ready for modeling
│       └── sst_indo_clean.csv
├── data_sst/                   # Raw NetCDF files (gitignored)
├── output/
│   └── figures/                # Generated plots and visualizations
│       ├── lstm_results.png
│       ├── multivariate_lstm_results.png
│       ├── validation_2012_results.png
│       └── sst_anomaly_trend.png
├── download_data.py            # Download NetCDF from NOAA
├── preprocessing.py            # ETL: NetCDF → CSV
├── modeling.py                 # Univariate LSTM
├── multivariate_modeling.py    # Multivariate LSTM (SST + Niño 3.4)
└── validation_2012.py          # Out-of-Sample Testing
```

---

## Data Sources
| Data | Source | Location |
|------|--------|----------|
| Indonesian SST | NOAA OISST V2 | `data/processed/sst_indo_clean.csv` |
| Niño 3.4 Index | NOAA ERSSTv5 | `data/raw/nina34.anom.data.txt` |

### Raw NetCDF Data (Not Included)
Folder `data_sst/` berisi file NetCDF mentah dari NOAA (~500MB per file) yang **tidak di-upload ke GitHub** karena ukuran terlalu besar.

**Untuk menjalankan preprocessing dari awal:**
1. Download data dari NOAA OISST: https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html
2. Simpan file `sst.day.mean.YYYY.nc` di folder `data_sst/`
3. Jalankan `python preprocessing.py`

Atau jalankan:
```bash
python download_data.py  # Auto-download dari NOAA
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
![Validation Results](output/figures/validation_2012_results.png)

### Multivariate Prediction
![Multivariate Results](output/figures/multivariate_lstm_results.png)

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

# 4. Train & evaluate
python validation_2012.py          # Recommended: Out-of-sample validation
python multivariate_modeling.py    # Alternative: 80/20 split
```

---

*Project ini dibuat sebagai eksplorasi Data Science di bidang Oseanografi.*