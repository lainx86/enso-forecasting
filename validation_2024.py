"""
validation_2024.py - Out-of-Sample Testing for Year 2024

This script performs TRUE out-of-sample validation with EARLY STOPPING:
1. Training on processed data from 2000-2023 (sst_indo_clean.csv)
   - Split internally: 80% Train, 20% Validation (for Early Stopping)
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
import os

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

# Output directory for checkpoint
os.makedirs("output/models", exist_ok=True)
CHECKPOINT_PATH = "output/models/best_model.pt"

# Indonesian Maritime Region
LAT_MIN, LAT_MAX = -11, 6
LON_MIN, LON_MAX = 95, 141

LOOKBACK = 12       # Input: 12 months of history
FORECAST_HORIZON = 12  # Output: 12 months ahead (entire year)

# Model hyperparameters
INPUT_SIZE = 2
HIDDEN_SIZE = 64    # Increased for multi-output complexity
NUM_LAYERS = 2      # Deeper network for multi-step
OUTPUT_SIZE = FORECAST_HORIZON  # Predict 12 months at once

# Training parameters
EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 0.003
PATIENCE = 20  # Stop if no improvement after 20 epochs

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# HELPER CLASSES
# ============================================================================

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'   EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'   Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data(sst_file: str, nino_file: str) -> pd.DataFrame:
    """Load training data (2000-2023) from processed CSV."""
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
        for month_idx, value_str in enumerate(parts[1:13]):
            try:
                value = float(value_str)
                if value < -90: continue
                date = pd.Timestamp(year=year, month=month_idx + 1, day=1)
                records.append({'date': date, 'nino34': value})
            except ValueError: continue
    nino_df = pd.DataFrame(records).set_index('date').sort_index()
    
    merged = sst_df.join(nino_df, how='inner')[['sst_anomaly', 'nino34']].dropna()
    print(f"✓ Training data: {merged.index[0].strftime('%Y-%m')} to {merged.index[-1].strftime('%Y-%m')} ({len(merged)} records)")
    return merged, nino_df


def load_2024_from_netcdf(nc_file: str, nino_df: pd.DataFrame) -> pd.DataFrame:
    """Load 2024 SST data directly from raw NetCDF file."""
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
    
    print(f"✓ Test data (2024): {df_2024.index[0].strftime('%Y-%m')} to {df_2024.index[-1].strftime('%Y-%m')} ({len(df_2024)} records)")
    return df_2024


def calculate_anomaly_for_2024(train_df: pd.DataFrame, test_2024: pd.DataFrame) -> pd.DataFrame:
    """Calculate 2024 anomaly using climatology from training data."""
    train_sst = pd.read_csv(SST_INDO_FILE)
    train_sst['date'] = pd.to_datetime(train_sst['date'])
    train_sst['month'] = train_sst['date'].dt.month
    
    climatology = train_sst.groupby('month')['sst_actual'].mean()
    
    test_2024 = test_2024.copy()
    test_2024['month'] = test_2024.index.month
    test_2024['climatology'] = test_2024['month'].map(climatology)
    test_2024['sst_anomaly'] = test_2024['sst_actual'] - test_2024['climatology']
    
    print(f"  2024 SST Anomaly range: {test_2024['sst_anomaly'].min():.2f}°C to {test_2024['sst_anomaly'].max():.2f}°C")
    return test_2024[['sst_anomaly', 'nino34']]


# ============================================================================
# MODEL & TRAINING
# ============================================================================

class Seq2SeqLSTM(nn.Module):
    """
    Encoder-Decoder LSTM for Multi-Step Forecasting.
    Input: 12 months of (SST + Niño 3.4)
    Output: 12 months of SST predictions
    """
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=12):
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True, dropout=0.2)
        
        # Decoder layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output 12 values
    
    def forward(self, x):
        # Encoder: process input sequence
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        encoder_out, (hidden, cell) = self.encoder(x, (h0, c0))
        
        # Use the final hidden state to decode
        context = encoder_out[:, -1, :]  # Last timestep output
        
        # Decoder: produce multi-step forecast
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # Shape: (batch, 12)
        
        return out


def create_sequences_multistep(data: np.ndarray, lookback: int, horizon: int):
    """
    Create sequences for multi-step forecasting.
    
    Args:
        data: Array with shape (timesteps, features)
        lookback: Number of input timesteps (12)
        horizon: Number of output timesteps (12)
    
    Returns:
        X: (samples, lookback, features)
        y: (samples, horizon) - only SST column
    """
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i-lookback:i, :])           # Input: 12 months, all features
        y.append(data[i:i+horizon, 0])            # Target: next 12 months, SST only
    return np.array(X), np.array(y)


def train_model(model, train_loader, val_loader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=CHECKPOINT_PATH)
    
    train_losses = []
    val_losses = []
    
    print("\n" + "=" * 50)
    print(f"TRAINING (with Early Stopping, Patience={PATIENCE})")
    print("=" * 50)
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())
        
        avg_train_loss = np.mean(batch_losses)
        train_losses.append(avg_train_loss)
        
        # --- VALIDATION PHASE ---
        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                val_pred = model(X_val)
                v_loss = criterion(val_pred, y_val)
                batch_val_losses.append(v_loss.item())
        
        avg_val_loss = np.mean(batch_val_losses)
        val_losses.append(avg_val_loss)
        
        # Print status
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
        # Check Early Stopping
        early_stopping(avg_val_loss, model)
        
        if early_stopping.early_stop:
            print(f"\n!!! Early stopping triggered at epoch {epoch+1} !!!")
            break
    
    # Load the best model
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print("✓ Loaded best model weights")
    
    return train_losses, val_losses


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_2024_validation(actual, predicted, dates, rmse, train_losses, val_losses):
    """Create visualization with SST comparison and Loss Curves."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), dpi=100)
    
    month_labels = [d.strftime('%b %Y') for d in dates]
    x_pos = range(len(dates))
    
    # Plot 1: SST Actual vs Predicted
    ax1 = axes[0]
    ax1.plot(x_pos, actual, 'b-', linewidth=2.5, marker='o', markersize=8, label='Actual 2024')
    ax1.plot(x_pos, predicted, 'r--', linewidth=2.5, marker='s', markersize=8, label='LSTM Prediction')
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(month_labels, rotation=45, ha='right')
    ax1.set_ylabel('SST Anomaly (°C)')
    ax1.set_title(f'Out-of-Sample Validation: Indonesian SST Anomaly (2024)\nRMSE = {rmse:.4f}°C', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss Curves
    ax2 = axes[1]
    ax2.plot(train_losses, 'g-', label='Training Loss')
    ax2.plot(val_losses, 'orange', label='Validation Loss', linestyle='--')
    
    # Mark best epoch
    min_val_loss = min(val_losses)
    best_epoch = val_losses.index(min_val_loss)
    ax2.scatter([best_epoch], [min_val_loss], color='red', s=100, zorder=5, 
                label=f'Best Model (Epoch {best_epoch+1})')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Training vs Validation Loss (Early Stopping)', fontweight='bold')
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
    print("MULTI-STEP DIRECT FORECASTING: Year 2024")
    print("Model predicts ALL 12 months of 2024 from 12 months of 2023")
    print("=" * 70)
    
    # 1. Load Data
    print("\n[Step 1/6] Loading training data (2000-2023)...")
    train_df, nino_df = load_training_data(SST_INDO_FILE, NINO34_FILE)
    
    print("\n[Step 2/6] Loading 2024 test data from raw NetCDF...")
    test_2024_raw = load_2024_from_netcdf(SST_2024_NC, nino_df)
    
    print("\n[Step 3/6] Calculating 2024 anomaly...")
    test_2024 = calculate_anomaly_for_2024(train_df, test_2024_raw)
    
    # 2. Normalize
    print("\n[Step 4/6] Normalizing (fit on training data only)...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train_df.values)
    test_scaled = scaler.transform(test_2024.values)
    
    # 3. Create MULTI-STEP Sequences
    # Each sample: Input = 12 months, Output = next 12 months
    print(f"  Creating multi-step sequences (lookback={LOOKBACK}, horizon={FORECAST_HORIZON})...")
    X_full, y_full = create_sequences_multistep(train_scaled, LOOKBACK, FORECAST_HORIZON)
    
    print(f"  Total sequences: {len(X_full)}")
    print(f"  X shape: {X_full.shape} (samples, lookback, features)")
    print(f"  y shape: {y_full.shape} (samples, horizon)")
    
    # 4. SPLIT TRAIN vs VALIDATION (80/20)
    train_size = int(len(X_full) * 0.8)
    
    X_train = X_full[:train_size]
    y_train = y_full[:train_size]
    
    X_val = X_full[train_size:]
    y_val = y_full[train_size:]
    
    print(f"\n  Data Split:")
    print(f"   - Training   : {X_train.shape[0]} samples")
    print(f"   - Validation : {X_val.shape[0]} samples")
    
    # 5. Prepare Test Input (2024)
    # Use ONLY the last 12 months of 2023 (Jan-Dec 2023) as input
    # Model will output prediction for all 12 months of 2024
    X_test = train_scaled[-LOOKBACK:].reshape(1, LOOKBACK, -1)  # Shape: (1, 12, 2)
    y_test = test_scaled[:, 0]  # Actual SST anomaly for 2024, shape: (12,)
    
    print(f"\n  [PURE FORECASTING]")
    print(f"   Input: Jan-Dec 2023 (last 12 months of training)")
    print(f"   Output: Predict Jan-Dec 2024 (12 months at once)")
    print(f"   → No 2024 data seen during inference!")
    
    # Tensor Conversion
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t   = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t   = torch.FloatTensor(y_val).to(DEVICE)
    X_test_t  = torch.FloatTensor(X_test).to(DEVICE)
    
    # DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 6. Train
    print("\n[Step 5/6] Training Seq2Seq model...")
    model = Seq2SeqLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(DEVICE)
    print(f"  Model: Seq2SeqLSTM(input={INPUT_SIZE}, hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, output={OUTPUT_SIZE})")
    train_losses, val_losses = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE)
    
    # 7. Predict 2024
    print("\n[Step 6/6] Predicting Year 2024...")
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test_t).cpu().numpy().flatten()  # Shape: (12,)
    
    # Inverse Transform
    n_features = scaler.n_features_in_
    
    # Predictions
    pred_full = np.zeros((len(pred_scaled), n_features))
    pred_full[:, 0] = pred_scaled
    pred_original = scaler.inverse_transform(pred_full)[:, 0]
    
    # Actuals
    actual_full = np.zeros((len(y_test), n_features))
    actual_full[:, 0] = y_test
    actual_original = scaler.inverse_transform(actual_full)[:, 0]
    
    # Metrics
    rmse = np.sqrt(np.mean((pred_original - actual_original) ** 2))
    mae = np.mean(np.abs(pred_original - actual_original))
    corr = np.corrcoef(actual_original, pred_original)[0, 1]
    
    print("\n" + "=" * 50)
    print("2024 OUT-OF-SAMPLE METRICS (Pure Forecasting)")
    print("=" * 50)
    print(f"RMSE:        {rmse:.4f} °C")
    print(f"MAE:         {mae:.4f} °C")
    print(f"Correlation: {corr:.4f}")
    
    # Print monthly predictions
    print("\n Monthly Predictions vs Actual:")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for i, m in enumerate(months):
        if i < len(pred_original):
            diff = pred_original[i] - actual_original[i]
            print(f"  {m} 2024: Pred={pred_original[i]:+.3f}°C, Actual={actual_original[i]:+.3f}°C, Error={diff:+.3f}°C")
    
    # Plot
    plot_2024_validation(actual_original, pred_original, test_2024.index, rmse, train_losses, val_losses)
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = main()