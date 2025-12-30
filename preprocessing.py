"""
preprocessing.py - ETL Pipeline for Sea Surface Temperature (SST) Anomaly Calculation

This script processes raw NetCDF SST data from NOAA OISST V2 High Resolution dataset
and creates a clean time series of SST anomalies for the Indonesian maritime region.

Oceanographic Context:
- Indonesia is located in the Indo-Pacific Warm Pool, the warmest oceanic region globally.
- SST anomalies in this region are strongly influenced by El Niño-Southern Oscillation (ENSO).
- Positive anomalies often correlate with El Niño events (warmer than normal).
- Negative anomalies correlate with La Niña events (cooler than normal).
- Removing the seasonal cycle (climatology) is crucial for isolating ENSO signals.

Author: Data Science Portfolio Project
Date: December 2024
"""

import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
START_YEAR = 2000
END_YEAR = 2024
DATA_DIR = "data_sst"
OUTPUT_FILE = "data/processed/sst_indo_clean.csv"

# Indonesian Maritime Region Boundaries
# This region captures the main Indonesian archipelago and surrounding waters
# Includes Java Sea, Banda Sea, Sulawesi Sea, and parts of the Pacific Warm Pool
LAT_MIN = -11  # Southern boundary (south of Java/Timor)
LAT_MAX = 6    # Northern boundary (north of Kalimantan)
LON_MIN = 95   # Western boundary (west of Sumatra)
LON_MAX = 141  # Eastern boundary (Papua/Irian Jaya)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_slice_nc_file(filepath: str) -> xr.Dataset:
    """
    Load a NetCDF file and slice it to the Indonesian region.
    
    The NOAA OISST data uses:
    - 'sst' as the variable name for sea surface temperature
    - 'lat' and 'lon' as coordinate names
    - 'time' as the temporal dimension
    
    Args:
        filepath: Path to the NetCDF file
        
    Returns:
        xarray Dataset sliced to Indonesian region
    """
    ds = xr.open_dataset(filepath)
    
    # Slice to Indonesian maritime region
    # Using slice() allows for automatic handling of coordinate ordering
    ds_indo = ds.sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX)
    )
    
    return ds_indo


def resample_to_monthly(ds: xr.Dataset) -> xr.Dataset:
    """
    Resample daily SST data to monthly means.
    
    Oceanographic Rationale:
    - Daily data contains high-frequency noise (weather events, tides, etc.)
    - Monthly means smooth out synoptic-scale variability
    - ENSO signals operate on seasonal to interannual timescales
    - Monthly data is the standard for ENSO analysis (e.g., Niño 3.4 index)
    
    Args:
        ds: xarray Dataset with daily 'sst' data
        
    Returns:
        xarray Dataset with monthly mean SST
    """
    # Resample to month-start ('MS') and compute mean
    # This aggregates all days within each month
    ds_monthly = ds.resample(time='MS').mean(dim='time')
    
    return ds_monthly


def spatial_aggregation(ds: xr.Dataset) -> xr.DataArray:
    """
    Calculate area-weighted mean SST over the Indonesian region.
    
    Oceanographic Rationale:
    - Spatial averaging creates a single representative index for the region
    - This is similar to how Niño indices are calculated (spatial mean over a box)
    - The resulting 1D time series captures the large-scale SST signal
    
    Note: For more accurate results, area-weighting by cos(latitude) could be applied,
    but for the small latitudinal extent of Indonesia, simple mean is acceptable.
    
    Args:
        ds: xarray Dataset with 'sst' variable
        
    Returns:
        xarray DataArray with 1D time series of mean SST
    """
    # Compute mean over both spatial dimensions
    sst_mean = ds['sst'].mean(dim=['lat', 'lon'])
    
    return sst_mean


def calculate_anomaly(sst_series: xr.DataArray) -> tuple:
    """
    Calculate SST anomalies by removing the monthly climatology.
    
    Oceanographic Rationale:
    - The "anomaly" removes the expected seasonal cycle
    - Climatology = long-term average for each month (Jan, Feb, ..., Dec)
    - Anomaly = observed SST - climatological SST for that month
    - This isolates interannual variability (ENSO, IOD, etc.) from the seasonal signal
    
    For example:
    - If July climatology is 28.5°C and observed July 2015 is 29.5°C
    - Then the anomaly is +1.0°C (warmer than normal → possible El Niño)
    
    Args:
        sst_series: xarray DataArray with monthly mean SST
        
    Returns:
        Tuple of (climatology, anomaly) as xarray DataArrays
    """
    # Group by month and compute the long-term mean for each calendar month
    # This creates a 12-value climatology (one per month)
    climatology = sst_series.groupby('time.month').mean('time')
    
    # Subtract climatology from observed values
    # xarray automatically aligns by month
    sst_anomaly = sst_series.groupby('time.month') - climatology
    
    return climatology, sst_anomaly


def plot_anomaly_trend(df: pd.DataFrame, output_path: str = "sst_anomaly_trend.png"):
    """
    Generate a publication-quality plot of SST anomaly time series.
    
    Args:
        df: DataFrame with 'date' and 'sst_anomaly' columns
        output_path: Filename for the saved figure
    """
    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
    
    # Convert date to datetime for proper plotting
    dates = pd.to_datetime(df['date'])
    anomaly = df['sst_anomaly']
    
    # Create filled areas for positive (red) and negative (blue) anomalies
    # This visualization style is standard in oceanography (e.g., NOAA Climate Dashboard)
    ax.fill_between(dates, anomaly, 0, 
                    where=(anomaly >= 0), color='coral', alpha=0.7, label='Warm Anomaly')
    ax.fill_between(dates, anomaly, 0, 
                    where=(anomaly < 0), color='steelblue', alpha=0.7, label='Cool Anomaly')
    
    # Add a trend line (zero reference)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Styling
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('SST Anomaly (°C)', fontsize=12)
    ax.set_title('Sea Surface Temperature Anomaly - Indonesian Maritime Region\n(2000-2006)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output/figures/sst_anomaly_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Anomaly trend plot saved to: output/figures/sst_anomaly_trend.png")


# ============================================================================
# MAIN ETL PIPELINE
# ============================================================================

def main():
    """
    Main ETL pipeline for SST data processing.
    
    Pipeline Steps:
    1. Iterate through yearly NetCDF files (2000-2006)
    2. Extract Indonesian region for each file
    3. Concatenate into a single dataset
    4. Resample from daily to monthly means
    5. Spatially aggregate to create 1D time series
    6. Calculate monthly climatology and anomalies
    7. Export to CSV and generate visualization
    """
    print("=" * 70)
    print("SST ANOMALY ETL PIPELINE - Indonesian Maritime Region")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Load and concatenate all yearly files
    # -------------------------------------------------------------------------
    print(f"\n[Step 1/6] Loading NetCDF files from {START_YEAR} to {END_YEAR}...")
    
    datasets = []
    for year in range(START_YEAR, END_YEAR + 1):
        filename = f"sst.day.mean.{year}.nc"
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"  ⚠ Warning: {filename} not found. Skipping...")
            continue
            
        print(f"  Loading: {filename}")
        ds = load_and_slice_nc_file(filepath)
        datasets.append(ds)
    
    if len(datasets) == 0:
        raise FileNotFoundError(f"No NetCDF files found in {DATA_DIR}!")
    
    # Concatenate all years along the time dimension
    print(f"\n[Step 2/6] Concatenating {len(datasets)} yearly datasets...")
    ds_combined = xr.concat(datasets, dim='time')
    
    # Close individual datasets to free memory
    for ds in datasets:
        ds.close()
    
    print(f"  Combined data shape: {ds_combined['sst'].shape}")
    print(f"  Time range: {ds_combined['time'].values[0]} to {ds_combined['time'].values[-1]}")
    
    # -------------------------------------------------------------------------
    # Step 2: Resample to monthly means
    # -------------------------------------------------------------------------
    print("\n[Step 3/6] Resampling daily data to monthly means...")
    ds_monthly = resample_to_monthly(ds_combined)
    print(f"  Monthly data points: {len(ds_monthly['time'])}")
    
    # -------------------------------------------------------------------------
    # Step 3: Spatial aggregation (mean over lat/lon)
    # -------------------------------------------------------------------------
    print("\n[Step 4/6] Computing spatial mean over Indonesian region...")
    sst_mean = spatial_aggregation(ds_monthly)
    print(f"  Time series length: {len(sst_mean)} months")
    
    # -------------------------------------------------------------------------
    # Step 4: Calculate anomaly (remove seasonal cycle)
    # -------------------------------------------------------------------------
    print("\n[Step 5/6] Calculating monthly climatology and anomalies...")
    print("  → Removing seasonal cycle to isolate interannual variability (ENSO signal)")
    climatology, sst_anomaly = calculate_anomaly(sst_mean)
    
    # Print climatology for reference
    print("\n  Monthly Climatology (°C):")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, name in enumerate(month_names, 1):
        clim_val = climatology.sel(month=month).values
        print(f"    {name}: {clim_val:.2f}°C")
    
    # -------------------------------------------------------------------------
    # Step 5: Create and export DataFrame
    # -------------------------------------------------------------------------
    print(f"\n[Step 6/6] Exporting data to {OUTPUT_FILE}...")
    
    # Create clean DataFrame
    df = pd.DataFrame({
        'date': pd.to_datetime(sst_mean['time'].values),
        'sst_actual': sst_mean.values,
        'sst_anomaly': sst_anomaly.values
    })
    
    # Format date column as YYYY-MM-DD for readability
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False, float_format='%.4f')
    print(f"  ✓ Saved {len(df)} monthly records to {OUTPUT_FILE}")
    
    # -------------------------------------------------------------------------
    # Step 6: Generate visualization
    # -------------------------------------------------------------------------
    print("\n[Bonus] Generating anomaly trend visualization...")
    plot_anomaly_trend(df)
    
    # -------------------------------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE - Summary Statistics")
    print("=" * 70)
    print(f"\nData Range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"Total Records: {len(df)} months")
    print(f"\nSST Actual (°C):")
    print(f"  Mean:  {df['sst_actual'].mean():.2f}")
    print(f"  Min:   {df['sst_actual'].min():.2f}")
    print(f"  Max:   {df['sst_actual'].max():.2f}")
    print(f"\nSST Anomaly (°C):")
    print(f"  Mean:  {df['sst_anomaly'].mean():.4f}")
    print(f"  Min:   {df['sst_anomaly'].min():.2f}")
    print(f"  Max:   {df['sst_anomaly'].max():.2f}")
    print(f"  Std:   {df['sst_anomaly'].std():.2f}")
    
    # Close combined dataset
    ds_combined.close()
    
    print("\n✓ All processing complete!")
    return df


if __name__ == "__main__":
    df = main()
