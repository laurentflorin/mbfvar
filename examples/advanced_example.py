"""
Advanced MUFBVAR Example
========================

This example demonstrates advanced features:
1. Conditional forecasting (imposing constraints)
2. Scenario analysis (comparing multiple forecast paths)
3. Hyperparameter optimization
4. Model comparison across time periods

Prerequisites: Run basic_example.py first to understand the fundamentals.
"""

import MBFVAR
import pandas as pd
import numpy as np
import os
from scipy.stats import uniform

# Change to examples directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# 1. SETUP: Load and prepare data
# ============================================================================
print("="*70)
print("ADVANCED MUFBVAR EXAMPLE")
print("="*70)

io_data = "hist.xlsx"
frequencies = ["Q", "M", "W"]

# Load data
data = []
for freq in frequencies:
    data_temp = pd.read_excel(io_data, sheet_name=freq, index_col=0)
    data.append(data_temp)

# Specify transformations
trans = [np.array([1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])]

# Prepare data
data_in = MBFVAR.mbfvar_data(data, trans, frequencies)

# Model parameters
nsim = 500  # Fewer simulations for speed in this example
nburn = 0.5
nlags = [6, 4]
thining = 1

# Initialize model
model = MBFVAR.MixedFrequencyBVAR(nsim, nburn, nlags, thining)

# ============================================================================
# 2. HYPERPARAMETER OPTIMIZATION
# ============================================================================
print("\n" + "="*70)
print("HYPERPARAMETER OPTIMIZATION")
print("="*70)

print("\nOptimizing hyperparameters using Bayesian optimization...")
print("This will take a few minutes...\n")

# Define parameter space for optimization
param_space = {
    'lambda1_1': uniform(0.001, 20),
    'lambda2_1': uniform(0.01, 10),
    'lambda4_1': uniform(0.01, 10),
    'lambda5_1': uniform(0.01, 10),
    'lambda1_2': uniform(0.001, 20),
    'lambda2_2': uniform(0.01, 10),
    'lambda4_2': uniform(0.01, 10),
    'lambda5_2': uniform(0.01, 10)
}

# Run optimization (using RMSE criterion)
try:
    hyp_optimal = model.update_hyperparameters_mango_rmse(
        data_in,
        param_space,
        H=2,                    # Forecast horizon for optimization
        init_points=2,          # Number of random initial points
        n_iter=3,               # Number of optimization iterations
        nsim=500,               # Simulations per evaluation
        njobs=1,                # Number of parallel jobs
        var_of_interest=["q_1"],  # Variable to optimize for
        temp_agg='mean',        # Temporal aggregation method
        save=False
    )
    print("\nOptimal hyperparameters found:")
    print(f"  {hyp_optimal}")
except Exception as e:
    print(f"\nOptimization encountered an issue: {e}")
    print("Using default hyperparameters instead...")
    hyp_optimal = [[0.09, 4.3, 1, 2.7, 4.3], [0.09, 4.3, 1, 2.7, 4.3]]

# ============================================================================
# 3. FIT MODEL WITH OPTIMIZED HYPERPARAMETERS
# ============================================================================
print("\n" + "="*70)
print("FITTING MODEL")
print("="*70)

print("\nFitting model with optimized hyperparameters...")
model.fit(data_in, hyp=hyp_optimal)
print("Model fitted successfully!")

# ============================================================================
# 4. CONDITIONAL FORECASTING
# ============================================================================
print("\n" + "="*70)
print("CONDITIONAL FORECASTING")
print("="*70)

print("\nGenerating conditional forecast...")
print("Imposing constraints on variables w_1 and m_2 for selected periods")

H = 24  # Forecast horizon

# Specify conditions (np.nan = no constraint)
conditionals = pd.DataFrame({
    'w_1': [0.018, 0.025, np.nan, np.nan, 0.0228, 0.05] + [np.nan] * (H-6),
    'm_2': [np.nan, 0.002, 0.01, 0.01, np.nan, np.nan] + [np.nan] * (H-6)
})

print(f"\nConstraints for first 6 periods:")
print(conditionals.head(6))

# Generate conditional forecast
model.forecast(H, conditionals)
print("\nConditional forecast generated!")

# Aggregate and save
model.aggregate(frequency="Q")
model.to_excel("forecast_conditional.xlsx", agg=True)
print("Saved: forecast_conditional.xlsx")

# ============================================================================
# 5. SCENARIO ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("SCENARIO ANALYSIS")
print("="*70)

print("\nComparing three scenarios: Optimistic, Pessimistic, and Baseline")

# Define scenarios with different constraints
scenario_optimistic = pd.DataFrame({
    'w_1': [0.02, 0.025, np.nan, np.nan, 0.03, 0.05] + [np.nan] * (H-6)
})

scenario_pessimistic = pd.DataFrame({
    'w_1': [-0.02, -0.025, np.nan, np.nan, -0.03, -0.05] + [np.nan] * (H-6)
})

# Baseline = unconditional forecast (None)
conditionals_scenarios = [scenario_optimistic, scenario_pessimistic, None]
scenario_names = ["Optimistic", "Pessimistic", "Baseline"]

print("\nGenerating scenario forecasts...")
out_scenarios = model.scenario_forecast(
    H,
    conditionals_scenarios,
    scenario_names,
    agg=True  # Aggregate to quarterly for comparison
)
print("Scenarios generated successfully!")

# Create scenario comparison plot
print("\nCreating scenario comparison plot...")
model.scenario_plot(
    scenario_dict=out_scenarios,
    variables="all",
    save=True,
    name="scenario_comparison",
    show=False,
    nhist=10
)
print("Saved: scenario_comparison.png")

# ============================================================================
# 6. FOCUSED FORECAST ON SPECIFIC VARIABLES
# ============================================================================
print("\n" + "="*70)
print("FOCUSED ANALYSIS")
print("="*70)

print("\nRe-fitting model with focus on specific variables...")

# Fit model focusing on quarterly variable q_1
model_focused = MBFVAR.MixedFrequencyBVAR(nsim, nburn, nlags, thining)
model_focused.fit(data_in, hyp=hyp_optimal, var_of_interest=["q_1"])
print("Focused model fitted!")

# Generate forecast
model_focused.forecast(H)
model_focused.aggregate(frequency="Q")

# Create fan chart for the variable of interest
print("\nCreating focused fan chart...")
model_focused.fanchart(
    variables=["q_1"],
    save=True,
    show=False,
    agg=True,
    nhist=10,
    name="fanchart_focused"
)
print("Saved: fanchart_focused.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("ADVANCED EXAMPLE COMPLETED SUCCESSFULLY!")
print("="*70)

print("\nGenerated files:")
print("  - forecast_conditional.xlsx    (Conditional forecast)")
print("  - scenario_comparison.png      (Multi-scenario comparison)")
print("  - fanchart_focused.png         (Focused variable analysis)")

print("\nKey techniques demonstrated:")
print("  ✓ Hyperparameter optimization with Bayesian methods")
print("  ✓ Conditional forecasting with constraints")
print("  ✓ Scenario analysis with multiple forecast paths")
print("  ✓ Focused modeling on specific variables")

print("\n" + "="*70)
