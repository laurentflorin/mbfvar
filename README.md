# MUFBVAR

**Multi-Frequency Bayesian Vector Autoregression**

A Python package for handling, disaggregating, and forecasting multi-frequency time series data using Bayesian VAR models. MUFBVAR allows you to work with data at different frequencies (e.g., quarterly, monthly, weekly) in a unified framework.

## Features

- **Multi-frequency data handling**: Seamlessly work with time series data at different frequencies (yearly, quarterly, monthly, weekly, daily)
- **Bayesian estimation**: Utilize Bayesian methods with Minnesota-type priors for robust parameter estimation
- **Flexible forecasting**: Generate unconditional and conditional forecasts in the highest frequency
- **Automatic aggregation**: Aggregate forecasts to lower frequencies
- **Hyperparameter optimization**: Built-in Bayesian optimization for hyperparameter tuning
- **Rich visualization**: Create fan charts, mean plots, and scenario comparisons
- **Scenario analysis**: Compare multiple forecast scenarios side-by-side
- **R integration**: Full compatibility with R through the `reticulate` package

## Installation

Install directly from GitHub using pip:

```bash
pip install git+https://github.com/laurentflorin/MBFVAR.git
```

For development installation:

```bash
git clone https://github.com/laurentflorin/MBFVAR.git
cd MBFVAR
pip install -e .
```

## Quick Start

Here's a minimal example to get you started:

```python
import MUFBVAR
import pandas as pd
import numpy as np

# Load your data (one DataFrame per frequency)
data_quarterly = pd.read_excel("data.xlsx", sheet_name="Q", index_col=0)
data_monthly = pd.read_excel("data.xlsx", sheet_name="M", index_col=0)
data_weekly = pd.read_excel("data.xlsx", sheet_name="W", index_col=0)

# Specify data transformations (0: log, 1: divide by 100)
trans_q = np.array([1, 1])  # 2 quarterly variables
trans_m = np.array([1, 1, 1])  # 3 monthly variables
trans_w = np.array([1, 1, 1])  # 3 weekly variables

# Prepare data
data = [data_quarterly, data_monthly, data_weekly]
trans = [trans_q, trans_m, trans_w]
frequencies = ["Q", "M", "W"]

data_in = MUFBVAR.mufbvar_data(data, trans, frequencies)

# Initialize and fit model
nsim = 1000  # Number of posterior draws
nburn = 0.5  # Burn-in proportion
nlags = [6, 4]  # Lags for each frequency level
thining = 1  # Keep every nth draw

model = MUFBVAR.multifrequency_var(nsim, nburn, nlags, thining)

# Hyperparameters for Minnesota prior
hyp = [[0.09, 4.3, 1, 2.7, 4.3], [0.09, 4.3, 1, 2.7, 4.3]]

model.fit(data_in, hyp=hyp)

# Generate forecast
H = 52  # Forecast horizon (in highest frequency)
model.forecast(H)

# Visualize results
model.fanchart(variables="all", save=False, show=True, nhist=10)
```

## Documentation

- [Full Documentation](./docs/_build/markdown/index.md)
- [Python Examples](./examples/python_example.py)
- [R Examples](./examples/r_example.R)
- [API Reference](./docs/_build/markdown/source/MUFBVAR.md)

## Key Concepts

### Data Structure

MUFBVAR expects data organized by frequency:
- Each frequency level has its own pandas DataFrame
- Data should have a datetime index
- Variables are organized in columns

### Transformations

Specify how each variable should be transformed:
- `0`: Take the natural logarithm
- `1`: Divide by 100 (for percentage data)

### Frequencies

Supported frequency combinations (from lowest to highest):
- Yearly (Y), Quarterly (Q), Monthly (M), Weekly (W), Daily (D)
- Example: ["Q", "M", "W"] means quarterly + monthly + weekly data

### Hyperparameters

The Minnesota-type prior uses 5 hyperparameters per frequency level:
1. `lambda1`: Overall tightness
2. `lambda2`: Cross-variable shrinkage
3. `lambda3`: Lag decay (fixed at 1 in current implementation)
4. `lambda4`: Exogenous variable tightness
5. `lambda5`: Intercept tightness

## Advanced Usage

### Conditional Forecasting

Impose constraints on specific variables:

```python
import pandas as pd
import numpy as np

# Specify conditions (use np.nan for unconstrained periods)
conditionals = pd.DataFrame({
    'w_1': [0.018, 0.025, np.nan, np.nan, 0.0228, 0.05],
    'm_2': [np.nan, 0.002, 0.01, 0.01, np.nan, np.nan]
})

model.forecast(H, conditionals)
```

### Hyperparameter Optimization

Automatically optimize hyperparameters using Bayesian optimization:

```python
from scipy.stats import uniform

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

hyp_optimal = model.update_hyperparameters_mango_rmse(
    data_in,
    param_space,
    H=2,
    init_points=3,
    n_iter=8,
    nsim=1000,
    njobs=1,
    var_of_interest=["q_1"],
    temp_agg='mean'
)
```

### Scenario Analysis

Compare multiple forecast scenarios:

```python
conditionals_good = pd.DataFrame({'w_1': [0.02, 0.025, np.nan, np.nan]})
conditionals_bad = pd.DataFrame({'w_1': [-0.02, -0.025, np.nan, np.nan]})

conditionals = [conditionals_good, conditionals_bad, None]
names = ["Optimistic", "Pessimistic", "Baseline"]

scenarios = model.scenario_forecast(H, conditionals, names, agg=True)
model.scenario_plot(scenario_dict=scenarios, variables="all", show=True)
```

### Frequency Aggregation

Aggregate high-frequency forecasts to lower frequencies:

```python
# Generate forecasts in highest frequency
model.forecast(H)

# Aggregate to quarterly
model.aggregate(frequency="Q")

# Export aggregated results
model.to_excel("forecasts_quarterly.xlsx", agg=True)
```

## Using MUFBVAR in R

MUFBVAR can be used in R through the `reticulate` package:

```r
library(reticulate)

# Import the package
mufbvar <- import("MUFBVAR")
pd <- import("pandas")
np <- import("numpy")

# Load data
io_data <- "hist.xlsx"
frequencies <- list("Q", "M", "W")

data <- list()
for (freq in frequencies) {
    data_temp <- pd$read_excel(io_data, sheet_name=freq, index_col=0L)
    data <- append(data, list(data_temp))
}

# Specify transformations
trans <- list(np$array(c(1, 1)), np$array(c(1, 1, 1)), np$array(c(1, 1, 1)))

# Initialize data
data_in <- mufbvar$mufbvar_data(data, trans, frequencies)

# Initialize and fit model
model <- mufbvar$multifrequency_var(1000L, 0.5, list(6L, 4L), 1L)
hyp <- list(list(0.09, 4.3, 1, 2.7, 4.3), list(0.09, 4.3, 1, 2.7, 4.3))
model$fit(data_in, hyp=hyp)

# Forecast
model$forecast(52L)
model$fanchart(variables="all", show=TRUE)
```

### Setting up R Environment

To use MUFBVAR in RStudio, you need to configure a Python virtual environment:

1. In RStudio, go to **Tools → Global Options**
2. Select **Python** from the left sidebar
3. Click **Select...** to choose your Python interpreter
4. Under the **Virtual Environments** tab, select a virtual environment with Python 3.8+
5. Install MUFBVAR in that environment

See [R Example](./examples/r_example.R) for a complete working example.

## Examples

The `examples/` directory contains:
- `python_example.py`: Comprehensive Python workflow
- `r_example.R`: Complete R integration example
- `generate_data.py`: Script to generate synthetic test data
- `hist.xlsx`: Sample multi-frequency dataset

## Requirements

- Python >= 3.8
- NumPy
- SciPy
- Pandas
- Matplotlib
- Plotly
- Seaborn
- tqdm
- fanchart
- bayesian-optimization
- arm-mango
- openpyxl
- xlsxwriter

## Citation

If you use MUFBVAR in your research, please cite:

```
Florin, Laurent. (2024). MUFBVAR: Multi-Frequency Bayesian Vector Autoregression.
https://github.com/laurentflorin/MBFVAR
```

## License

This project is licensed under the terms specified in the repository.

## Author

**Laurent Florin**
Email: laurent.florin@efv.admin.ch

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.