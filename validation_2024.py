"""
validation_2024.py - Pure Out-of-Sample Validation for Year 2024

RECURSIVE AUTOREGRESSIVE FORECASTING:
1. Model is trained for single-step prediction (1 month ahead)
2. For 2024 forecast: model predicts recursively using its own predictions
3. Niño 3.4 for 2024 uses "persistence" (last known value from Dec 2023)

This is the standard approach in operational climate forecasting.

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
import os

warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================
SST_INDO_FILE = "data/processed/sst_indo_clean.csv"
NINO34_FILE = "data/raw/nina34.anom.data.txt"
SST_2024_NC = "data_sst/sst.day.mean.2024.nc"

os.makedirs("output/models", exist_ok=True)
CHECKPOINT_PATH = "output/models/best_model.pt"

# Region bounds
LAT_MIN, LAT_MAX = -11, 6
LON_MIN, LON_MAX = 95, 141

# Model parameters
LOOKBACK = 12       # 12 months input
INPUT_SIZE = 2      # SST + Niño 3.4
HIDDEN_SIZE = 32
NUM_LAYERS = 1
OUTPUT_SIZE = 1     # Single-step prediction

# Training parameters
EPOCHS = 150
BATCH_SIZE = 4
LEARNING_RATE = 0.005
PATIENCE = 15

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# HELPER CLASSES
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'   EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'   Val loss improved ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data(sst_file, nino_file):
    """Load and merge training data (2000-2023)."""
    sst_df = pd.read_csv(sst_file)
    sst_df['date'] = pd.to_datetime(sst_df['date'])
    sst_df = sst_df.set_index('date')
    
    records = []
    with open(nino_file, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 13: continue
        try:
            year = int(parts[0])
            if year < 1900 or year > 2100: continue
        except ValueError: continue
        for month_idx, val in enumerate(parts[1:13]):
            try:
                v = float(val)
                if v < -90: continue
                records.append({'date': pd.Timestamp(year=year, month=month_idx+1, day=1), 'nino34': v})
            except ValueError: continue
    nino_df = pd.DataFrame(records).set_index('date').sort_index()
    
    merged = sst_df.join(nino_df, how='inner')[['sst_anomaly', 'nino34']].dropna()
    print(f"✓ Training data: {merged.index[0]:%Y-%m} to {merged.index[-1]:%Y-%m} ({len(merged)} records)")
    return merged, nino_df


def load_2024_from_netcdf(nc_file, nino_df):
    """Load 2024 SST from raw NetCDF."""
    ds = xr.open_dataset(nc_file)
    ds_indo = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
    ds_monthly = ds_indo.resample(time='MS').mean(dim='time')
    sst_mean = ds_monthly['sst'].mean(dim=['lat', 'lon'])
    
    df_2024 = pd.DataFrame({
        'date': pd.to_datetime(sst_mean['time'].values),
        'sst_actual': sst_mean.values
    }).set_index('date')
    
    nino_2024 = nino_df.loc[nino_df.index.year == 2024]
    df_2024 = df_2024.join(nino_2024, how='inner')
    ds.close()
    
    print(f"✓ Test data: {df_2024.index[0]:%Y-%m} to {df_2024.index[-1]:%Y-%m} ({len(df_2024)} records)")
    return df_2024


def calculate_anomaly_for_2024(train_df, test_2024):
    """Calculate 2024 anomaly using training climatology."""
    train_sst = pd.read_csv(SST_INDO_FILE)
    train_sst['date'] = pd.to_datetime(train_sst['date'])
    train_sst['month'] = train_sst['date'].dt.month
    climatology = train_sst.groupby('month')['sst_actual'].mean()
    
    test_2024 = test_2024.copy()
    test_2024['month'] = test_2024.index.month
    test_2024['climatology'] = test_2024['month'].map(climatology)
    test_2024['sst_anomaly'] = test_2024['sst_actual'] - test_2024['climatology']
    
    print(f"  2024 SST Anomaly: {test_2024['sst_anomaly'].min():.2f}°C to {test_2024['sst_anomaly'].max():.2f}°C")
    return test_2024[['sst_anomaly', 'nino34']]


# ============================================================================
# MODEL
# ============================================================================

class LSTMForecaster(nn.Module):
    """LSTM for single-step time series forecasting."""
    def __init__(self, input_size=2, hidden_size=32, num_layers=1, output_size=1):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def create_sequences(data, lookback):
    """Create sequences for single-step prediction."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :])  # All features
        y.append(data[i, 0])              # SST only
    return np.array(X), np.array(y).reshape(-1, 1)


def train_model(model, train_loader, val_loader, epochs, lr):
    """Train with early stopping."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=CHECKPOINT_PATH)
    
    train_losses, val_losses = [], []
    
    print("\n" + "=" * 50)
    print(f"TRAINING (Patience={PATIENCE})")
    print("=" * 50)
    
    for epoch in range(epochs):
        # Train
        model.train()
        batch_loss = []
        for X_b, y_b in train_loader:
            pred = model(X_b)
            loss = criterion(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_loss.append(loss.item())
        train_losses.append(np.mean(batch_loss))
        
        # Validate
        model.eval()
        batch_val_loss = []
        with torch.no_grad():
            for X_v, y_v in val_loader:
                pred = model(X_v)
                batch_val_loss.append(criterion(pred, y_v).item())
        val_losses.append(np.mean(batch_val_loss))
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] Train: {train_losses[-1]:.6f} | Val: {val_losses[-1]:.6f}")
        
        early_stopping(val_losses[-1], model)
        if early_stopping.early_stop:
            print(f"\n⚡ Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print("✓ Loaded best model")
    
    return train_losses, val_losses


# ============================================================================
# RECURSIVE FORECASTING
# ============================================================================

def recursive_forecast(model, initial_sequence, scaler, n_months=12):
    """
    Perform recursive autoregressive forecasting.
    
    Args:
        model: Trained LSTM model
        initial_sequence: Last 12 months of scaled data, shape (12, 2)
        scaler: Fitted MinMaxScaler
        n_months: Number of months to forecast
    
    Returns:
        predictions: Array of predictions in original scale
    """
    model.eval()
    
    # Copy to avoid modifying original
    current_seq = initial_sequence.copy()  # Shape: (12, 2)
    
    # Get last known Niño 3.4 value for persistence
    last_nino_scaled = current_seq[-1, 1]
    
    predictions_scaled = []
    
    print("\n  Recursive forecasting:")
    for month in range(n_months):
        # Prepare input
        X = torch.FloatTensor(current_seq).unsqueeze(0).to(DEVICE)  # Shape: (1, 12, 2)
        
        # Predict next SST
        with torch.no_grad():
            pred_sst = model(X).cpu().numpy().flatten()[0]
        
        predictions_scaled.append(pred_sst)
        
        # Shift window and update
        # New row: [predicted_sst, persisted_nino]
        new_row = np.array([pred_sst, last_nino_scaled])
        current_seq = np.vstack([current_seq[1:], new_row])
        
        month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month]
        print(f"    {month_name} 2024: predicted (scaled={pred_sst:.4f})")
    
    # Inverse transform
    predictions_scaled = np.array(predictions_scaled)
    pred_full = np.zeros((len(predictions_scaled), 2))
    pred_full[:, 0] = predictions_scaled
    pred_full[:, 1] = last_nino_scaled  # Dummy for inverse transform
    predictions = scaler.inverse_transform(pred_full)[:, 0]
    
    return predictions


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(actual, predicted, dates, rmse, train_losses, val_losses):
    """Plot results."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), dpi=100)
    
    months = [d.strftime('%b %Y') for d in dates]
    x = range(len(dates))
    
    # Plot 1: Predictions
    ax1 = axes[0]
    ax1.plot(x, actual, 'b-o', linewidth=2.5, markersize=8, label='Actual 2024')
    ax1.plot(x, predicted, 'r--s', linewidth=2.5, markersize=8, label='LSTM Recursive Forecast')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(months, rotation=45, ha='right')
    ax1.set_ylabel('SST Anomaly (°C)')
    ax1.set_title(f'Recursive Autoregressive Forecasting: Indonesian SST (2024)\n'
                  f'RMSE = {rmse:.4f}°C | Pure forecasting (no 2024 data used)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss
    ax2 = axes[1]
    ax2.plot(train_losses, 'g-', label='Training Loss')
    ax2.plot(val_losses, 'orange', linestyle='--', label='Validation Loss')
    min_val = min(val_losses)
    best_ep = val_losses.index(min_val)
    ax2.scatter([best_ep], [min_val], color='red', s=100, zorder=5, label=f'Best (Epoch {best_ep+1})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Training Progress', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/figures/validation_2024_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: output/figures/validation_2024_results.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("RECURSIVE AUTOREGRESSIVE FORECASTING: Year 2024")
    print("Model predicts 1 month at a time, using its own predictions as input")
    print("Niño 3.4 uses persistence (last known value from Dec 2023)")
    print("=" * 70)
    
    # 1. Load Data
    print("\n[Step 1/6] Loading data...")
    train_df, nino_df = load_training_data(SST_INDO_FILE, NINO34_FILE)
    test_2024_raw = load_2024_from_netcdf(SST_2024_NC, nino_df)
    test_2024 = calculate_anomaly_for_2024(train_df, test_2024_raw)
    
    # 2. Normalize
    print("\n[Step 2/6] Normalizing...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train_df.values)
    test_scaled = scaler.transform(test_2024.values)
    
    # 3. Create sequences (single-step)
    print("\n[Step 3/6] Creating sequences...")
    X_full, y_full = create_sequences(train_scaled, LOOKBACK)
    
    # 4. Train/Val split
    train_size = int(len(X_full) * 0.8)
    X_train, y_train = X_full[:train_size], y_full[:train_size]
    X_val, y_val = X_full[train_size:], y_full[train_size:]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Tensors & Loaders
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. Train
    print("\n[Step 4/6] Training...")
    model = LSTMForecaster(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(DEVICE)
    print(f"  Model: LSTMForecaster(in={INPUT_SIZE}, hidden={HIDDEN_SIZE}, out={OUTPUT_SIZE})")
    train_losses, val_losses = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE)
    
    # 6. Recursive Forecast
    print("\n[Step 5/6] Forecasting 2024 (recursive)...")
    print("  Input: Jan-Dec 2023 (last 12 months of training)")
    print("  Niño 3.4: Using persistence (Dec 2023 value repeated)")
    
    initial_seq = train_scaled[-LOOKBACK:]  # Last 12 months: Jan-Dec 2023
    predictions = recursive_forecast(model, initial_seq, scaler, n_months=12)
    
    # Actual values
    actual = test_2024['sst_anomaly'].values
    
    # Metrics
    rmse = np.sqrt(np.mean((predictions - actual) ** 2))
    mae = np.mean(np.abs(predictions - actual))
    corr = np.corrcoef(actual, predictions)[0, 1]
    
    print("\n" + "=" * 50)
    print("2024 PURE FORECASTING METRICS")
    print("=" * 50)
    print(f"RMSE:        {rmse:.4f} °C")
    print(f"MAE:         {mae:.4f} °C")
    print(f"Correlation: {corr:.4f}")
    
    # Monthly breakdown
    print("\n Monthly Results:")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for i, m in enumerate(months):
        if i < len(predictions):
            err = predictions[i] - actual[i]
            print(f"  {m}: Pred={predictions[i]:+.3f}°C | Actual={actual[i]:+.3f}°C | Error={err:+.3f}°C")
    
    # Plot
    print("\n[Step 6/6] Plotting...")
    plot_results(actual, predictions, test_2024.index, rmse, train_losses, val_losses)
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return model, scaler


if __name__ == "__main__":
    model, scaler = main()