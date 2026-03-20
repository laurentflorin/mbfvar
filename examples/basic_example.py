"""
Basic MUFBVAR Example
=====================

This example demonstrates the core functionality of the MUFBVAR package:
1. Loading multi-frequency data
2. Preparing data for the model
3. Fitting a Bayesian VAR model
4. Generating forecasts
5. Creating visualizations

The example uses the sample dataset hist.xlsx which contains:
- Quarterly data (Q): 2 variables
- Monthly data (M): 3 variables
- Weekly data (W): 3 variables
"""

import MUFBVAR
import pandas as pd
import numpy as np
import os

# Change to examples directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("Loading data...")

io_data = "hist.xlsx"
frequencies = ["Q", "M", "W"]  # Lowest to highest frequency

# Load data for each frequency level
data = []
for freq in frequencies:
    data_temp = pd.read_excel(io_data, sheet_name=freq, index_col=0)
    data.append(data_temp)
    print(f"  {freq}: {data_temp.shape[0]} observations, {data_temp.shape[1]} variables")
    print(f"      Variables: {list(data_temp.columns)}")

# ============================================================================
# 2. SPECIFY TRANSFORMATIONS
# ============================================================================
print("\nSpecifying transformations...")

# Transformations for each variable:
# 0 = take natural log
# 1 = divide by 100 (for percentage/rate data)

trans = [
    np.array([1, 1]),        # Quarterly: 2 variables (both divide by 100)
    np.array([1, 1, 1]),     # Monthly: 3 variables (all divide by 100)
    np.array([1, 1, 1])      # Weekly: 3 variables (all divide by 100)
]

# ============================================================================
# 3. PREPARE DATA
# ============================================================================
print("\nPreparing data for MUFBVAR...")

data_in = MUFBVAR.mufbvar_data(data, trans, frequencies)
print("  Data prepared successfully!")

# ============================================================================
# 4. SPECIFY MODEL PARAMETERS
# ============================================================================
print("\nSetting model parameters...")

# Model specification
nsim = 1000         # Number of posterior draws
nburn = 0.5         # Proportion of draws to discard as burn-in (50%)
nlags = [6, 4]      # Lags: 6 for first frequency pair, 4 for second
thining = 1         # Keep every nth draw (1 = keep all)

# Minnesota prior hyperparameters [lambda1, lambda2, lambda3, lambda4, lambda5]
# - lambda1: Overall tightness (smaller = tighter)
# - lambda2: Cross-variable shrinkage
# - lambda3: Lag decay (fixed at 1)
# - lambda4: Exogenous variable tightness
# - lambda5: Intercept tightness

hyp = [
    [0.09, 4.3, 1, 2.7, 4.3],  # Hyperparameters for first frequency pair
    [0.09, 4.3, 1, 2.7, 4.3]   # Hyperparameters for second frequency pair
]

print(f"  Simulations: {nsim} (burn-in: {int(nsim*nburn)})")
print(f"  Lags: {nlags}")
print(f"  Hyperparameters: {hyp}")

# ============================================================================
# 5. INITIALIZE AND FIT MODEL
# ============================================================================
print("\nInitializing model...")

model = MUFBVAR.multifrequency_var(nsim, nburn, nlags, thining)

print("Fitting model (this may take a few moments)...")
model.fit(data_in, hyp=hyp)
print("  Model fitted successfully!")

# ============================================================================
# 6. GENERATE UNCONDITIONAL FORECAST
# ============================================================================
print("\nGenerating forecasts...")

H = 52  # Forecast horizon in highest frequency (52 weeks)

model.forecast(H)
print(f"  Generated {H}-period ahead forecast")

# ============================================================================
# 7. AGGREGATE FORECASTS
# ============================================================================
print("\nAggregating forecasts to quarterly frequency...")

model.aggregate(frequency="Q")
print("  Aggregation complete!")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\nSaving results...")

# Save high-frequency forecasts
model.to_excel("forecasts_weekly.xlsx", agg=False)
print("  Saved weekly forecasts to: forecasts_weekly.xlsx")

# Save aggregated (quarterly) forecasts
model.to_excel("forecasts_quarterly.xlsx", agg=True)
print("  Saved quarterly forecasts to: forecasts_quarterly.xlsx")

# ============================================================================
# 9. VISUALIZE RESULTS
# ============================================================================
print("\nGenerating visualizations...")

# Fan chart for all variables (aggregated to quarterly)
print("  Creating fan chart (quarterly)...")
model.fanchart(
    variables="all",
    save=True,
    show=False,
    agg=True,
    nhist=10,
    name="fanchart_quarterly"
)
print("    Saved: fanchart_quarterly.png")

# Fan chart for weekly data
print("  Creating fan chart (weekly)...")
model.fanchart(
    variables="all",
    save=True,
    show=False,
    agg=False,
    nhist=50,
    name="fanchart_weekly"
)
print("    Saved: fanchart_weekly.png")

# Mean forecast plot
print("  Creating mean forecast plot...")
model.mean_plot(
    variables="all",
    save=True,
    show=False,
    name="mean_forecast"
)
print("    Saved: mean_forecast.png")

print("\n" + "="*70)
print("EXAMPLE COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nGenerated files:")
print("  - forecasts_weekly.xlsx")
print("  - forecasts_quarterly.xlsx")
print("  - fanchart_quarterly.png")
print("  - fanchart_weekly.png")
print("  - mean_forecast.png")
