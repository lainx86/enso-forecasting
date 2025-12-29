"""
validation_2012.py - Out-of-Sample Testing for Year 2012

This script performs rigorous out-of-sample validation by:
1. Training on data from 2000-2011
2. Testing exclusively on 2012 data (12 months)

This approach simulates a real-world forecasting scenario where the model
must predict future values it has never seen during training.

Author: Data Science Portfolio Project
Date: December 2024
"""

import numpy as np
import pandas as pd
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
SST_INDO_FILE = "sst_indo_clean.csv"
NINO34_FILE = "nina34.anom.data.txt"

LOOKBACK = 12  # 12 months lookback window

# Year-based split
TRAIN_END_YEAR = 2011   # Train on 2000-2011
TEST_YEAR = 2012        # Test on 2012

# Model hyperparameters
INPUT_SIZE = 2      # Indonesian SST + Niño 3.4
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

def load_indonesian_sst(filepath: str) -> pd.DataFrame:
    """Load Indonesian SST anomaly data."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    print(f"✓ Loaded Indonesian SST: {len(df)} records ({df.index[0].year}-{df.index[-1].year})")
    return df


def load_nino34_data(filepath: str) -> pd.DataFrame:
    """Parse NOAA Niño 3.4 anomaly data."""
    records = []
    
    with open(filepath, 'r') as f:
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
    
    df = pd.DataFrame(records).set_index('date').sort_index()
    print(f"✓ Loaded Niño 3.4: {len(df)} records ({df.index[0].year}-{df.index[-1].year})")
    return df


def merge_and_split_by_year(sst_df: pd.DataFrame, nino_df: pd.DataFrame, 
                            train_end_year: int, test_year: int) -> tuple:
    """
    Merge datasets and split chronologically by year.
    
    Returns:
        Tuple of (train_df, test_df)
    """
    merged = sst_df.join(nino_df, how='inner')[['sst_anomaly', 'nino34']].dropna()
    
    # Split by year
    train_df = merged[merged.index.year <= train_end_year]
    test_df = merged[merged.index.year == test_year]
    
    print(f"\n✓ Data Split by Year:")
    print(f"  Training: {train_df.index[0].strftime('%Y-%m')} to {train_df.index[-1].strftime('%Y-%m')} ({len(train_df)} records)")
    print(f"  Test:     {test_df.index[0].strftime('%Y-%m')} to {test_df.index[-1].strftime('%Y-%m')} ({len(test_df)} records)")
    
    return train_df, test_df, merged


# ============================================================================
# PREPROCESSING
# ============================================================================

def create_sequences_for_validation(full_data: np.ndarray, train_size: int, 
                                    test_size: int, lookback: int) -> tuple:
    """
    Create sequences ensuring test set uses lookback from training period.
    
    For Jan 2012 prediction, we need Dec 2010 - Nov 2011 (last 12 months of 2011).
    """
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    # Training sequences (within training data)
    for i in range(lookback, train_size):
        X_train.append(full_data[i-lookback:i, :])
        y_train.append(full_data[i, 0])  # Only predict SST
    
    # Test sequences (using end of training + test data)
    for i in range(train_size, train_size + test_size):
        X_test.append(full_data[i-lookback:i, :])
        y_test.append(full_data[i, 0])
    
    return (np.array(X_train), np.array(y_train).reshape(-1, 1),
            np.array(X_test), np.array(y_test).reshape(-1, 1))


# ============================================================================
# MODEL
# ============================================================================

class MultivariateLSTM(nn.Module):
    """Multivariate LSTM for time series forecasting."""
    
    def __init__(self, input_size=2, hidden_size=32, num_layers=1, output_size=1):
        super(MultivariateLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        return self.fc(lstm_out[:, -1, :])


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, epochs, lr):
    """Train the LSTM model."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=False
    )
    
    train_losses = []
    
    print("\n" + "=" * 50)
    print("TRAINING (2000-2011 Data)")
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

def plot_2012_validation(actual, predicted, dates, rmse, train_losses):
    """Create visualization for 2012 out-of-sample validation."""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=100)
    
    # -------------------------------------------------------------------------
    # Plot 1: 2012 Validation
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    
    # Format dates for plotting
    month_labels = [d.strftime('%b %Y') for d in dates]
    x_pos = range(len(dates))
    
    ax1.plot(x_pos, actual, 'b-', linewidth=2.5, marker='o', markersize=8,
             label='Actual (Ground Truth)')
    ax1.plot(x_pos, predicted, 'r--', linewidth=2.5, marker='s', markersize=8,
             label='LSTM Prediction')
    
    # Zero reference
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Styling
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(month_labels, rotation=45, ha='right')
    ax1.set_ylabel('SST Anomaly (°C)', fontsize=12)
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_title(f'Validation: Predicted vs Actual Indonesian SST Anomaly (Year 2012)\n'
                  f'Out-of-Sample RMSE = {rmse:.4f}°C', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Training Loss
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    
    epochs_range = range(1, len(train_losses) + 1)
    ax2.plot(epochs_range, train_losses, 'g-', linewidth=2)
    
    min_loss = min(train_losses)
    min_epoch = train_losses.index(min_loss) + 1
    ax2.scatter([min_epoch], [min_loss], color='red', s=100, zorder=5,
                label=f'Min Loss: {min_loss:.4f} (Epoch {min_epoch})')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MSE Loss', fontsize=12)
    ax2.set_title('Training Loss Curve (2000-2011 Data)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('validation_2012_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: validation_2012_results.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("OUT-OF-SAMPLE VALIDATION: Year 2012")
    print("Training on 2000-2011, Testing on 2012")
    print("=" * 70)
    
    # Load data
    print("\n[Step 1/5] Loading data...")
    sst_df = load_indonesian_sst(SST_INDO_FILE)
    nino_df = load_nino34_data(NINO34_FILE)
    
    # Merge and split by year
    print("\n[Step 2/5] Splitting data by year...")
    train_df, test_df, full_df = merge_and_split_by_year(
        sst_df, nino_df, TRAIN_END_YEAR, TEST_YEAR
    )
    
    # Normalize using ONLY training data (prevent data leakage)
    print("\n[Step 3/5] Normalizing (fit on training data only)...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    train_data = train_df.values
    test_data = test_df.values
    
    # Fit scaler on training data only
    scaler.fit(train_data)
    
    # Transform both train and test
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Combine for sequence creation (we need lookback from training for test)
    full_scaled = np.vstack([train_scaled, test_scaled])
    
    # Create sequences
    X_train, y_train, X_test, y_test = create_sequences_for_validation(
        full_scaled, len(train_scaled), len(test_scaled), LOOKBACK
    )
    
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
    print("\n[Step 4/5] Training model...")
    model = MultivariateLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE
    ).to(DEVICE)
    
    print(f"  Model: MultivariateLSTM(input={INPUT_SIZE}, hidden={HIDDEN_SIZE})")
    
    train_losses = train_model(model, train_loader, EPOCHS, LEARNING_RATE)
    
    # Evaluate on 2012
    print("\n[Step 5/5] Evaluating on Year 2012...")
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test_t).cpu().numpy()
    
    actual_scaled = y_test_t.cpu().numpy()
    
    # Inverse transform
    n_features = scaler.n_features_in_
    pred_full = np.zeros((len(pred_scaled), n_features))
    pred_full[:, 0] = pred_scaled.flatten()
    pred_original = scaler.inverse_transform(pred_full)[:, 0]
    
    actual_full = np.zeros((len(actual_scaled), n_features))
    actual_full[:, 0] = actual_scaled.flatten()
    actual_original = scaler.inverse_transform(actual_full)[:, 0]
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((pred_original - actual_original) ** 2))
    mae = np.mean(np.abs(pred_original - actual_original))
    corr = np.corrcoef(actual_original, pred_original)[0, 1]
    
    print("\n" + "=" * 50)
    print("2012 OUT-OF-SAMPLE METRICS")
    print("=" * 50)
    print(f"RMSE:        {rmse:.4f} °C")
    print(f"MAE:         {mae:.4f} °C")
    print(f"Correlation: {corr:.4f}")
    
    # Plot
    test_dates = test_df.index
    plot_2012_validation(actual_original, pred_original, test_dates, rmse, train_losses)
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return model, scaler


if __name__ == "__main__":
    model, scaler = main()
