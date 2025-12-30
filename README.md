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
│   ├── figures/                # Generated plots and visualizations
│   │   ├── validation_2024_results.png
│   │   └── sst_anomaly_trend.png
│   └── models/                 # Saved model checkpoints
│       └── best_model.pt
├── docs/                       # Documentation
│   └── TECHNICAL_DOCUMENTATION.md
├── download_data.py            # Download NetCDF from NOAA
├── preprocessing.py            # ETL: NetCDF → CSV
├── validation_2024.py          # Main Script: Out-of-Sample Testing (Train: 2000-2023, Test: 2024)
└── validation_2024_GRU.py      # Alternative: GRU Architecture
```

---

## Data Sources
| Data | Source | Location |
|------|--------|----------|
| Indonesian SST | NOAA OISST V2 | `data/processed/sst_indo_clean.csv` |
| Niño 3.4 Index | NOAA ERSSTv5 | `data/raw/nina34.anom.data.txt` |

### Raw NetCDF Data (Not Included)
Folder `data_sst/` berisi file NetCDF mentah dari NOAA (~500MB per file) yang **tidak di-upload ke GitHub** karena ukurannya terlalu besar.

**Workflow:**
1. `download_data.py` → Download data NetCDF dari NOAA ke folder `data_sst/`
2. `preprocessing.py` → Olah data NetCDF menjadi `data/processed/sst_indo_clean.csv`

---

## Training Approach

Script utama `validation_2024.py` menggunakan **TRUE out-of-sample validation**:
- **Training Data**: 2000-2023 (split internal 80% train / 20% validation untuk early stopping)
- **Test Data**: 2024 (diambil langsung dari raw NetCDF, tidak pernah dilihat saat training)
- **Early Stopping**: Training berhenti otomatis jika validation loss tidak membaik selama 15 epoch

> Pendekatan ini mensimulasikan skenario forecasting nyata dimana kita memprediksi masa depan yang benar-benar belum diketahui.

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

### Out-of-Sample Validation (Year 2024)
![Validation Results](output/figures/validation_2024_results.png)

---

## How to Run

```bash
# 1. Clone repository
git clone https://github.com/lainx86/enso-forecasting.git
cd enso-forecasting

# 2. Install dependencies
pip install -r requirements.txt
# 3. Download Data
python download_data.py

# 4. Run preprocessing (if starting fresh)
python preprocessing.py

# 5. Train & evaluate
python validation_2024.py          # LSTM with Early Stopping
```

---

*Project ini dibuat sebagai eksplorasi Data Science di bidang Oseanografi.*
