"""
validation_2024.py - Out-of-Sample Testing for Year 2024

This script performs TRUE out-of-sample validation:
1. Training on processed data from 2000-2023 (sst_indo_clean.csv)
2. Testing on 2024 data loaded DIRECTLY from raw NetCDF (never seen during training)

Author: Feby - For Data Science Portfolio Project
Date: December 2024
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Ensure reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================
SST_INDO_FILE = "data/processed/sst_indo_clean.csv"  # Training data (2000-2023)
NINO34_FILE = "data/raw/nina34.anom.data.txt"
SST_2024_NC = "data_sst/sst.day.mean.2024.nc"  # Raw 2024 data for testing

# Indonesian Maritime Region (same as preprocessing.py)
LAT_MIN, LAT_MAX = -11, 6
LON_MIN, LON_MAX = 95, 141

LOOKBACK = 12

# Model hyperparameters
INPUT_SIZE = 2
HIDDEN_SIZE = 32
NUM_LAYERS = 1
OUTPUT_SIZE = 1

# Training parameters
EPOCHS = 150
BATCH_SIZE = 4
LEARNING_RATE = 0.005

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data(sst_file: str, nino_file: str) -> pd.DataFrame:
    """Load training data (2000-2023) from processed CSV."""
    # Load Indonesian SST
    sst_df = pd.read_csv(sst_file)
    sst_df['date'] = pd.to_datetime(sst_df['date'])
    sst_df = sst_df.set_index('date')
    
    # Load Niño 3.4
    records = []
    with open(nino_file, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
            if year < 1900 or year > 2100:
                continue
        except ValueError:
            continue
        for month_idx, value_str in enumerate(parts[1:13]):
            try:
                value = float(value_str)
                if value < -90:
                    continue
                date = pd.Timestamp(year=year, month=month_idx + 1, day=1)
                records.append({'date': date, 'nino34': value})
            except ValueError:
                continue
    nino_df = pd.DataFrame(records).set_index('date').sort_index()
    
    # Merge
    merged = sst_df.join(nino_df, how='inner')[['sst_anomaly', 'nino34']].dropna()
    
    print(f"✓ Training data: {merged.index[0].strftime('%Y-%m')} to {merged.index[-1].strftime('%Y-%m')} ({len(merged)} records)")
    return merged, nino_df


def load_2024_from_netcdf(nc_file: str, nino_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load 2024 SST data directly from raw NetCDF file.
    This data was NEVER seen during training.
    """
    # Load NetCDF
    ds = xr.open_dataset(nc_file)
    
    # Slice to Indonesian region
    ds_indo = ds.sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX)
    )
    
    # Resample to monthly
    ds_monthly = ds_indo.resample(time='MS').mean(dim='time')
    
    # Spatial mean
    sst_mean = ds_monthly['sst'].mean(dim=['lat', 'lon'])
    
    # Calculate climatology from this year (or use simple mean)
    # For proper anomaly, we'd need long-term climatology
    # Here we use the mean of the training data as reference
    
    # Create DataFrame
    df_2024 = pd.DataFrame({
        'date': pd.to_datetime(sst_mean['time'].values),
        'sst_actual': sst_mean.values
    }).set_index('date')
    
    # We need to calculate anomaly using same climatology as training data
    # For simplicity, we'll compute it directly
    # Get Niño 3.4 for 2024
    nino_2024 = nino_df.loc[nino_df.index.year == 2024]
    
    df_2024 = df_2024.join(nino_2024, how='inner')
    
    ds.close()
    
    print(f"✓ Test data (2024): {df_2024.index[0].strftime('%Y-%m')} to {df_2024.index[-1].strftime('%Y-%m')} ({len(df_2024)} records)")
    
    return df_2024


def calculate_anomaly_for_2024(train_df: pd.DataFrame, test_2024: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 2024 anomaly using climatology from training data.
    """
    # Calculate monthly climatology from training data
    train_sst = pd.read_csv(SST_INDO_FILE)
    train_sst['date'] = pd.to_datetime(train_sst['date'])
    train_sst['month'] = train_sst['date'].dt.month
    
    climatology = train_sst.groupby('month')['sst_actual'].mean()
    
    # Apply to 2024
    test_2024 = test_2024.copy()
    test_2024['month'] = test_2024.index.month
    test_2024['climatology'] = test_2024['month'].map(climatology)
    test_2024['sst_anomaly'] = test_2024['sst_actual'] - test_2024['climatology']
    
    print(f"  2024 SST Anomaly range: {test_2024['sst_anomaly'].min():.2f}°C to {test_2024['sst_anomaly'].max():.2f}°C")
    
    return test_2024[['sst_anomaly', 'nino34']]


# ============================================================================
# MODEL & TRAINING (same as before)
# ============================================================================

class MultivariateLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1, output_size=1):
        super(MultivariateLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        return self.fc(lstm_out[:, -1, :])


def create_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y).reshape(-1, 1)


def train_model(model, train_loader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=False)
    
    train_losses = []
    print("\n" + "=" * 50)
    print("TRAINING (2000-2023 Data)")
    print("=" * 50)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | Loss: {avg_loss:.6f}")
    
    print("✓ Training Complete")
    return train_losses


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_2024_validation(actual, predicted, dates, nino34, rmse, train_losses):
    """Create visualization with 3 subplots: SST comparison, Niño 3.4, and training loss."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=100)
    
    month_labels = [d.strftime('%b %Y') for d in dates]
    x_pos = range(len(dates))
    
    # -------------------------------------------------------------------------
    # Plot 1: SST Actual vs Predicted
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(x_pos, actual, 'b-', linewidth=2.5, marker='o', markersize=8,
             label='Actual 2024 (from NetCDF)')
    ax1.plot(x_pos, predicted, 'r--', linewidth=2.5, marker='s', markersize=8,
             label='LSTM Prediction')
    
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(month_labels, rotation=45, ha='right')
    ax1.set_ylabel('SST Anomaly (°C)', fontsize=11)
    ax1.set_title(f'Out-of-Sample Validation: Indonesian SST Anomaly (Year 2024)\n'
                  f'RMSE = {rmse:.4f}°C', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Niño 3.4 Index (Predictor)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    
    # Color based on El Niño / La Niña threshold
    colors = ['coral' if v > 0.5 else 'steelblue' if v < -0.5 else 'gray' for v in nino34]
    ax2.bar(x_pos, nino34, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='El Niño threshold (+0.5)')
    ax2.axhline(y=-0.5, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='La Niña threshold (-0.5)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(month_labels, rotation=45, ha='right')
    ax2.set_ylabel('Niño 3.4 Index (°C)', fontsize=11)
    ax2.set_title('Niño 3.4 Index (Predictor from Pacific)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Plot 3: Training Loss
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    ax3.plot(range(1, len(train_losses)+1), train_losses, 'g-', linewidth=2)
    min_loss = min(train_losses)
    min_epoch = train_losses.index(min_loss) + 1
    ax3.scatter([min_epoch], [min_loss], color='red', s=100, zorder=5,
                label=f'Min Loss: {min_loss:.4f} (Epoch {min_epoch})')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('MSE Loss', fontsize=11)
    ax3.set_title('Training Loss Curve (2000-2023 Data)', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/figures/validation_2024_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: output/figures/validation_2024_results.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("TRUE OUT-OF-SAMPLE VALIDATION: Year 2024")
    print("Training: 2000-2023 (processed CSV)")
    print("Testing:  2024 (raw NetCDF - never seen by model)")
    print("=" * 70)
    
    # Load training data (2000-2023)
    print("\n[Step 1/6] Loading training data (2000-2023)...")
    train_df, nino_df = load_training_data(SST_INDO_FILE, NINO34_FILE)
    
    # Load 2024 test data from raw NetCDF
    print("\n[Step 2/6] Loading 2024 test data from raw NetCDF...")
    test_2024_raw = load_2024_from_netcdf(SST_2024_NC, nino_df)
    
    # Calculate anomaly for 2024 using training climatology
    print("\n[Step 3/6] Calculating 2024 anomaly using training climatology...")
    test_2024 = calculate_anomaly_for_2024(train_df, test_2024_raw)
    
    # Normalize using ONLY training data
    print("\n[Step 4/6] Normalizing (fit on training data only)...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train_df.values)
    test_scaled = scaler.transform(test_2024.values)
    
    # Combine for sequence creation (need lookback from end of training)
    full_scaled = np.vstack([train_scaled, test_scaled])
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, LOOKBACK)
    
    # For test: use last 12 months of training + test data
    X_test, y_test = [], []
    for i in range(len(test_scaled)):
        if i < LOOKBACK:
            # Use end of training data for lookback
            lookback_data = np.vstack([
                train_scaled[-(LOOKBACK-i):],
                test_scaled[:i]
            ]) if i > 0 else train_scaled[-LOOKBACK:]
        else:
            lookback_data = test_scaled[i-LOOKBACK:i]
        X_test.append(lookback_data)
        y_test.append(test_scaled[i, 0])
    
    X_test = np.array(X_test)
    y_test = np.array(y_test).reshape(-1, 1)
    
    print(f"  Train sequences: {X_train.shape}")
    print(f"  Test sequences: {X_test.shape}")
    
    # Create DataLoader
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.FloatTensor(y_test).to(DEVICE)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Build and train model
    print("\n[Step 5/6] Training model...")
    model = MultivariateLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(DEVICE)
    print(f"  Model: MultivariateLSTM(input={INPUT_SIZE}, hidden={HIDDEN_SIZE})")
    train_losses = train_model(model, train_loader, EPOCHS, LEARNING_RATE)
    
    # Evaluate on 2024
    print("\n[Step 6/6] Evaluating on Year 2024...")
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test_t).cpu().numpy()
    
    # Inverse transform
    n_features = scaler.n_features_in_
    pred_full = np.zeros((len(pred_scaled), n_features))
    pred_full[:, 0] = pred_scaled.flatten()
    pred_original = scaler.inverse_transform(pred_full)[:, 0]
    
    actual_full = np.zeros((len(y_test), n_features))
    actual_full[:, 0] = y_test.flatten()
    actual_original = scaler.inverse_transform(actual_full)[:, 0]
    
    # Metrics
    rmse = np.sqrt(np.mean((pred_original - actual_original) ** 2))
    mae = np.mean(np.abs(pred_original - actual_original))
    corr = np.corrcoef(actual_original, pred_original)[0, 1]
    
    print("\n" + "=" * 50)
    print("2024 OUT-OF-SAMPLE METRICS")
    print("=" * 50)
    print(f"RMSE:        {rmse:.4f} °C")
    print(f"MAE:         {mae:.4f} °C")
    print(f"Correlation: {corr:.4f}")
    
    # Plot with Niño 3.4
    nino34_2024 = test_2024['nino34'].values
    plot_2024_validation(actual_original, pred_original, test_2024.index, nino34_2024, rmse, train_losses)
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return model, scaler


if __name__ == "__main__":
    model, scaler = main()
