"""
modeling.py - LSTM Model for Sea Surface Temperature Anomaly Forecasting

This script implements a lightweight Long Short-Term Memory (LSTM) neural network
to forecast SST anomalies in the Indonesian maritime region using PyTorch.

Oceanographic Context:
- SST anomalies in Indonesia are influenced by ENSO (El Niño-Southern Oscillation)
- ENSO operates on timescales of 2-7 years with peak intensity every 3-5 years
- The 12-month lookback window captures approximately one seasonal cycle
- This allows the model to learn patterns like: warm anomalies often persist for several months
- Forecasting SST anomalies can help predict coral bleaching, fisheries changes, and rainfall

Model Design Rationale:
- Small dataset (~84 months) requires a lightweight architecture to avoid overfitting
- 1 LSTM layer with small hidden size (16-32) is appropriate for this data volume
- MSE loss is standard for regression tasks (predicting continuous temperature values)
- We normalize to [-1, 1] range as LSTM activations (tanh) are bounded in this range

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
# Data parameters
INPUT_FILE = "sst_indo_clean.csv"
LOOKBACK = 12  # 12 months lookback (captures full seasonal cycle)
TRAIN_RATIO = 0.8  # 80% train, 20% test

# Model hyperparameters (tuned for small dataset)
HIDDEN_SIZE = 24  # Number of LSTM hidden units (16-32 range for small data)
NUM_LAYERS = 1    # Single LSTM layer to prevent overfitting
DROPOUT = 0.0     # No dropout for small dataset (already regularized by size)

# Training parameters
EPOCHS = 150      # Number of training epochs
BATCH_SIZE = 4    # Small batch size for small dataset (allows more gradient updates)
LEARNING_RATE = 0.005  # Moderate learning rate

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the preprocessed SST anomaly data.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with date and SST anomaly columns
    """
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} records from {filepath}")
    print(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    return df


def create_sequences(data: np.ndarray, lookback: int) -> tuple:
    """
    Create sliding window sequences for time series forecasting.
    
    For each window of 'lookback' months, we predict the next month's value.
    
    Example with lookback=12:
        Input X: [month_1, month_2, ..., month_12]
        Target y: month_13
    
    Oceanographic Rationale:
    - 12-month lookback captures one full seasonal/annual cycle
    - This allows the model to learn patterns like:
      "If anomalies have been positive for 6 months, they may continue positive"
    - ENSO events typically persist for 9-12 months
    
    Args:
        data: Normalized SST anomaly values (1D array)
        lookback: Number of past time steps to use as input
        
    Returns:
        Tuple of (X, y) where X has shape [samples, lookback, 1]
        and y has shape [samples, 1]
    """
    X, y = [], []
    
    for i in range(len(data) - lookback):
        # Input: lookback months of data
        X.append(data[i:(i + lookback)])
        # Target: the next month after the input window
        y.append(data[i + lookback])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to [samples, sequence_length, features]
    # LSTM expects 3D input: (batch, seq_len, input_size)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = y.reshape(-1, 1)
    
    return X, y


def prepare_dataloaders(X: np.ndarray, y: np.ndarray, 
                        train_ratio: float, batch_size: int) -> tuple:
    """
    Split data into train/test sets and create PyTorch DataLoaders.
    
    Note: For time series, we use a simple temporal split (not shuffled)
    to maintain temporal ordering and avoid lookahead bias.
    
    Args:
        X: Input sequences
        y: Target values
        train_ratio: Proportion of data for training
        batch_size: Batch size for DataLoader
        
    Returns:
        Tuple of (train_loader, test_loader, train_size)
    """
    train_size = int(len(X) * train_ratio)
    
    # Temporal split (no shuffling to preserve time order)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\nData Split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")
    
    # Handle edge case: if test set is too small
    if len(X_test) < 2:
        raise ValueError(f"Test set too small ({len(X_test)} samples). "
                        f"Need more data or lower train ratio.")
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.FloatTensor(y_test).to(DEVICE)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    # Adjust batch size if larger than dataset
    effective_batch_size = min(batch_size, len(train_dataset))
    
    # Create DataLoaders (shuffle only training data)
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, 
                              shuffle=True)  # Shuffle for training
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), 
                             shuffle=False)  # No shuffle for evaluation
    
    return train_loader, test_loader, train_size, X_test_t, y_test_t


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class LightweightLSTM(nn.Module):
    """
    Lightweight LSTM for small time series forecasting.
    
    Architecture:
    - Single LSTM layer (prevents overfitting on small data)
    - Small hidden size (16-32 units)
    - Linear output layer for regression
    
    Why single-layer LSTM?
    - With only ~70 training samples, deeper networks would overfit
    - LSTM's gating mechanism is sufficient to capture temporal dependencies
    - Simpler models generalize better on limited data
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 24, 
                 num_layers: int = 1, output_size: int = 1):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features (1 for univariate SST)
            hidden_size: Number of LSTM hidden units
            num_layers: Number of stacked LSTM layers
            output_size: Number of output features (1 for single-step forecast)
        """
        super(LightweightLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True means input shape is (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0 if num_layers == 1 else DROPOUT
        )
        
        # Linear layer to map LSTM output to prediction
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        LSTM returns:
        - output: (batch, seq_len, hidden_size) - output at each time step
        - (h_n, c_n): final hidden and cell states
        
        We only use the output from the last time step for prediction.
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take only the last time step's output
        # lstm_out shape: (batch, seq_len, hidden_size)
        # We want: (batch, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        prediction = self.fc(last_output)
        
        return prediction


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model: nn.Module, train_loader: DataLoader, 
                epochs: int, learning_rate: float) -> list:
    """
    Train the LSTM model.
    
    Uses:
    - MSE Loss: Standard for regression, measures squared error in °C
    - Adam Optimizer: Adaptive learning rate, works well for RNNs
    
    Args:
        model: LSTM model instance
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        List of training losses per epoch
    """
    # Loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (reduce LR when loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=False
    )
    
    train_losses = []
    
    print("\n" + "=" * 50)
    print("TRAINING STARTED")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {learning_rate}")
    print("-" * 50)
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients (common in RNNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Average loss for the epoch
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Print progress every 25 epochs
        if (epoch + 1) % 25 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1:3d}/{epochs}] | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
    
    print("-" * 50)
    print("TRAINING COMPLETE")
    
    return train_losses


# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor,
                   scaler: MinMaxScaler, train_size: int, 
                   dates: pd.Series) -> tuple:
    """
    Evaluate the model on test data and calculate metrics.
    
    The predictions are inverse-transformed back to original scale (°C)
    for interpretable evaluation.
    
    Args:
        model: Trained LSTM model
        X_test: Test input sequences
        y_test: Test target values
        scaler: Fitted MinMaxScaler for inverse transform
        train_size: Number of training samples (for date indexing)
        dates: Original date series
        
    Returns:
        Tuple of (actual values, predicted values, test dates)
    """
    model.eval()
    
    with torch.no_grad():
        predictions_scaled = model(X_test).cpu().numpy()
    
    # Get actual values
    actual_scaled = y_test.cpu().numpy()
    
    # Inverse transform to original scale (°C)
    predictions_original = scaler.inverse_transform(predictions_scaled)
    actual_original = scaler.inverse_transform(actual_scaled)
    
    # Calculate metrics
    mse = np.mean((predictions_original - actual_original) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_original - actual_original))
    
    # Correlation coefficient
    correlation = np.corrcoef(actual_original.flatten(), 
                               predictions_original.flatten())[0, 1]
    
    print("\n" + "=" * 50)
    print("EVALUATION METRICS (Test Set)")
    print("=" * 50)
    print(f"MSE:         {mse:.4f} °C²")
    print(f"RMSE:        {rmse:.4f} °C")
    print(f"MAE:         {mae:.4f} °C")
    print(f"Correlation: {correlation:.4f}")
    
    # Get corresponding dates for test set
    # Test data starts after training data + lookback
    test_start_idx = train_size + LOOKBACK
    test_dates = dates.iloc[test_start_idx:test_start_idx + len(actual_original)]
    
    return actual_original, predictions_original, test_dates


def plot_results(actual: np.ndarray, predicted: np.ndarray, 
                 dates: pd.Series, train_losses: list):
    """
    Create comprehensive visualization of model results.
    
    Generates two plots:
    1. Actual vs Predicted SST Anomaly
    2. Training Loss Curve
    
    Args:
        actual: Actual SST anomaly values
        predicted: Model predictions
        dates: Corresponding dates
        train_losses: List of training losses
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=100)
    
    # -------------------------------------------------------------------------
    # Plot 1: Actual vs Predicted
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    
    # Convert dates to datetime for proper plotting
    plot_dates = pd.to_datetime(dates.values)
    
    ax1.plot(plot_dates, actual, 'b-', linewidth=2, 
             label='Actual', marker='o', markersize=4)
    ax1.plot(plot_dates, predicted, 'r--', linewidth=2, 
             label='Predicted (LSTM)', marker='s', markersize=4)
    
    # Add zero reference line
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Styling
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('SST Anomaly (°C)', fontsize=11)
    ax1.set_title('LSTM SST Anomaly Forecast: Actual vs Predicted\n'
                  '(Test Set - Last 20% of Data)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # -------------------------------------------------------------------------
    # Plot 2: Training Loss Curve
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    
    epochs_range = range(1, len(train_losses) + 1)
    ax2.plot(epochs_range, train_losses, 'g-', linewidth=2)
    
    # Mark minimum loss
    min_loss = min(train_losses)
    min_epoch = train_losses.index(min_loss) + 1
    ax2.scatter([min_epoch], [min_loss], color='red', s=100, zorder=5,
                label=f'Min Loss: {min_loss:.4f} (Epoch {min_epoch})')
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('MSE Loss', fontsize=11)
    ax2.set_title('Training Loss Curve', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(train_losses))
    
    plt.tight_layout()
    plt.savefig('lstm_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Results plot saved to: lstm_results.png")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run the complete LSTM forecasting pipeline.
    
    Pipeline:
    1. Load preprocessed SST anomaly data
    2. Normalize data to [-1, 1] range
    3. Create sliding window sequences
    4. Build and train LSTM model
    5. Evaluate on test set
    6. Visualize results
    """
    print("=" * 70)
    print("LSTM SST ANOMALY FORECASTING - Indonesian Maritime Region")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Load Data
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Loading preprocessed data...")
    df = load_data(INPUT_FILE)
    
    # Extract anomaly column
    sst_anomaly = df['sst_anomaly'].values.reshape(-1, 1)
    dates = df['date']
    
    print(f"  SST Anomaly range: {sst_anomaly.min():.2f}°C to {sst_anomaly.max():.2f}°C")
    
    # -------------------------------------------------------------------------
    # Step 2: Normalize Data
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Normalizing data to [-1, 1] range...")
    print("  → Using MinMaxScaler for normalization")
    print("  → Range [-1, 1] matches LSTM tanh activation function")
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    sst_scaled = scaler.fit_transform(sst_anomaly)
    
    print(f"  Scaled data range: {sst_scaled.min():.2f} to {sst_scaled.max():.2f}")
    
    # -------------------------------------------------------------------------
    # Step 3: Create Sequences
    # -------------------------------------------------------------------------
    print(f"\n[Step 3/5] Creating sequences with {LOOKBACK}-month lookback window...")
    print(f"  → 12-month window captures full seasonal cycle")
    
    X, y = create_sequences(sst_scaled, LOOKBACK)
    
    print(f"  Total sequences created: {len(X)}")
    print(f"  Input shape (X): {X.shape} [samples, lookback, features]")
    print(f"  Target shape (y): {y.shape} [samples, 1]")
    
    # Check if we have enough data
    if len(X) < 5:
        raise ValueError(f"Not enough data! Only {len(X)} sequences created. "
                        f"Need at least 5 for meaningful train/test split.")
    
    # Create DataLoaders
    train_loader, test_loader, train_size, X_test, y_test = prepare_dataloaders(
        X, y, TRAIN_RATIO, BATCH_SIZE
    )
    
    # -------------------------------------------------------------------------
    # Step 4: Build and Train Model
    # -------------------------------------------------------------------------
    print(f"\n[Step 4/5] Building LSTM model...")
    
    model = LightweightLSTM(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1
    ).to(DEVICE)
    
    # Print model architecture
    print(f"\n  Model Architecture:")
    print(f"  {'-' * 40}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Input size:     1 (univariate)")
    print(f"  Hidden size:    {HIDDEN_SIZE}")
    print(f"  LSTM layers:    {NUM_LAYERS}")
    print(f"  Output size:    1 (single-step forecast)")
    print(f"  Total params:   {total_params:,}")
    print(f"  {'-' * 40}")
    
    # Train the model
    train_losses = train_model(model, train_loader, EPOCHS, LEARNING_RATE)
    
    # -------------------------------------------------------------------------
    # Step 5: Evaluate and Visualize
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Evaluating model on test set...")
    
    actual, predicted, test_dates = evaluate_model(
        model, X_test, y_test, scaler, train_size, dates
    )
    
    # Generate plots
    plot_results(actual, predicted, test_dates, train_losses)
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nFiles Generated:")
    print(f"  • lstm_results.png - Prediction plot and training curve")
    print(f"\nModel trained on {train_size} samples, tested on {len(actual)} samples")
    
    # Oceanographic interpretation
    print("\n" + "-" * 70)
    print("OCEANOGRAPHIC INTERPRETATION")
    print("-" * 70)
    print("""
The LSTM model has learned to forecast SST anomalies by capturing:
• Temporal persistence: Warm/cool anomalies often persist for multiple months
• Seasonal patterns: Anomalies may vary with monsoon seasons
• ENSO signatures: Multi-month positive anomalies suggest El Niño conditions

Caveats for this small dataset:
• Limited training data (~70 samples) may cause overfitting
• 7 years of data may not capture full ENSO variability (2-7 year cycle)
• Consider adding more historical data (1982-present) for robust forecasting
""")
    
    return model, scaler


if __name__ == "__main__":
    model, scaler = main()
