Examples
=============

MBFVAR provides comprehensive examples for both Python and R users. The examples demonstrate the core functionality of the package including data preparation, model fitting, forecasting, and visualization.

Quick Start
***********

The quickest way to get started is with the basic example:

.. code-block:: python

    import MBFVAR
    import pandas as pd
    import numpy as np

    # Load data (one DataFrame per frequency)
    data_q = pd.read_excel("data.xlsx", sheet_name="Q", index_col=0)
    data_m = pd.read_excel("data.xlsx", sheet_name="M", index_col=0)
    data_w = pd.read_excel("data.xlsx", sheet_name="W", index_col=0)

    # Prepare data
    data = [data_q, data_m, data_w]
    trans = [np.array([1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])]
    frequencies = ["Q", "M", "W"]

    data_in = MBFVAR.mbfvar_data(data, trans, frequencies)

    # Initialize and fit model
    model = MBFVAR.MixedFrequencyBVAR(nsim=1000, nburn=0.5, nlags=[6, 4], thining=1)
    hyp = [[0.09, 4.3, 1, 2.7, 4.3], [0.09, 4.3, 1, 2.7, 4.3]]
    model.fit(data_in, hyp=hyp)

    # Generate and visualize forecasts
    model.forecast(H=52)
    model.fanchart(variables="all", save=False, show=True)

Complete Python Example
************************

Below is a comprehensive example demonstrating all major features:

.. code-block:: python

    import MBFVAR
    import pandas as pd
    import numpy as np

    # ==========================================
    # 1. LOAD AND PREPARE DATA
    # ==========================================

    io_data = "hist.xlsx"
    frequencies = ["Q", "M", "W"]  # Quarterly, Monthly, Weekly

    # Load data for each frequency
    data = []
    for freq in frequencies:
        data_temp = pd.read_excel(io_data, sheet_name=freq, index_col=0)
        data.append(data_temp)

    # Specify transformations
    # 0 = take natural log, 1 = divide by 100
    trans = [
        np.array([1, 1]),        # Quarterly: 2 variables
        np.array([1, 1, 1]),     # Monthly: 3 variables
        np.array([1, 1, 1])      # Weekly: 3 variables
    ]

    # Initialize data object
    data_in = MBFVAR.mbfvar_data(data, trans, frequencies)

    # ==========================================
    # 2. MODEL SPECIFICATION
    # ==========================================

    nsim = 1000         # Number of posterior draws
    nburn = 0.5         # Burn-in proportion (50%)
    nlags = [6, 4]      # Lags for each frequency level
    thining = 1         # Keep every nth draw

    # Minnesota prior hyperparameters
    hyp = [
        [0.09, 4.3, 1, 2.7, 4.3],  # First frequency pair
        [0.09, 4.3, 1, 2.7, 4.3]   # Second frequency pair
    ]

    # ==========================================
    # 3. FIT MODEL
    # ==========================================

    model = MBFVAR.MixedFrequencyBVAR(nsim, nburn, nlags, thining)
    model.fit(data_in, hyp=hyp)

    # ==========================================
    # 4. UNCONDITIONAL FORECAST
    # ==========================================

    H = 52  # Forecast horizon (in highest frequency)
    model.forecast(H)

    # Aggregate to quarterly
    model.aggregate(frequency="Q")

    # Save results
    model.to_excel("forecasts_weekly.xlsx", agg=False)
    model.to_excel("forecasts_quarterly.xlsx", agg=True)

    # ==========================================
    # 5. VISUALIZATIONS
    # ==========================================

    # Fan chart
    model.fanchart(
        variables="all",
        save=True,
        show=False,
        agg=True,
        nhist=10,
        name="fanchart"
    )

    # Mean plot
    model.mean_plot(
        variables="all",
        save=True,
        show=False,
        name="mean_forecast"
    )

    # ==========================================
    # 6. CONDITIONAL FORECASTING
    # ==========================================

    # Impose constraints on specific variables
    conditionals = pd.DataFrame({
        'w_1': [0.018, 0.025, np.nan, np.nan, 0.0228, 0.05],
        'm_2': [np.nan, 0.002, 0.01, 0.01, np.nan, np.nan]
    })

    model.forecast(H, conditionals)
    model.aggregate(frequency="Q")
    model.to_excel("forecast_conditional.xlsx", agg=True)

    # ==========================================
    # 7. SCENARIO ANALYSIS
    # ==========================================

    # Define multiple scenarios
    conditionals = [
        pd.DataFrame({'w_1': [0.02, 0.025, np.nan, np.nan, 0.03, 0.05]}),
        pd.DataFrame({'w_1': [-0.02, -0.025, np.nan, np.nan, -0.03, -0.05]}),
        None  # Baseline (unconditional)
    ]
    names = ["Optimistic", "Pessimistic", "Baseline"]

    scenarios = model.scenario_forecast(H, conditionals, names, agg=True)

    # Plot scenarios
    model.scenario_plot(
        scenario_dict=scenarios,
        variables="all",
        save=True,
        name="scenario_comparison",
        show=False,
        nhist=10
    )

    # ==========================================
    # 8. HYPERPARAMETER OPTIMIZATION
    # ==========================================

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
        temp_agg='mean',
        save=False
    )

Complete R Example
******************

The module can be used in R through the reticulate package:

.. code-block:: r

    library(reticulate)

    # Import MBFVAR and dependencies
    mufbvar <- import("MBFVAR")
    pd <- import("pandas")
    np <- import("numpy")

    # ==========================================
    # 1. LOAD AND PREPARE DATA
    # ==========================================

    io_data <- "hist.xlsx"
    frequencies <- list("Q", "M", "W")

    # Load data for each frequency
    data <- list()
    for (freq in frequencies) {
        data_temp <- pd$read_excel(io_data, sheet_name=freq, index_col=0L)
        data <- append(data, list(data_temp))
    }

    # Specify transformations
    trans <- list(
        np$array(c(1L, 1L)),
        np$array(c(1L, 1L, 1L)),
        np$array(c(1L, 1L, 1L))
    )

    # Initialize data object
    data_in <- mufbvar$mbfvar_data(data, trans, frequencies)

    # ==========================================
    # 2. MODEL SPECIFICATION AND FITTING
    # ==========================================

    nsim <- 1000L
    nburn <- 0.5
    nlags <- list(6L, 4L)
    thining <- 1L

    hyp <- list(
        list(0.09, 4.3, 1, 2.7, 4.3),
        list(0.09, 4.3, 1, 2.7, 4.3)
    )

    model <- mufbvar$MixedFrequencyBVAR(nsim, nburn, nlags, thining)
    model$fit(data_in, hyp=hyp)

    # ==========================================
    # 3. FORECASTING
    # ==========================================

    H <- 52L
    model$forecast(H)
    model$aggregate(frequency="Q")

    # Save results
    model$to_excel("forecasts_r.xlsx", agg=TRUE)

    # ==========================================
    # 4. VISUALIZATIONS
    # ==========================================

    model$fanchart(
        variables="all",
        save=TRUE,
        show=FALSE,
        agg=TRUE,
        nhist=10L
    )

    model$mean_plot(
        variables="all",
        save=TRUE,
        show=FALSE
    )

    # ==========================================
    # 5. CONDITIONAL FORECASTING
    # ==========================================

    conditionals <- pd$DataFrame(list(
        'w_1' = c(0.018, 0.025, np$nan, np$nan, 0.0228, 0.05),
        'm_2' = c(np$nan, 0.002, 0.01, 0.01, np$nan, np$nan)
    ))

    model$forecast(H, conditionals)
    model$aggregate(frequency="Q")

    # ==========================================
    # 6. SCENARIO ANALYSIS
    # ==========================================

    scenario_good <- pd$DataFrame(list('w_1' = c(0.02, 0.025, rep(np$nan, H-2))))
    scenario_bad <- pd$DataFrame(list('w_1' = c(-0.02, -0.025, rep(np$nan, H-2))))

    conditionals_list <- list(scenario_good, scenario_bad, NULL)
    scenario_names <- list("Optimistic", "Pessimistic", "Baseline")

    scenarios <- model$scenario_forecast(H, conditionals_list, scenario_names, agg=TRUE)

    model$scenario_plot(
        scenario_dict=scenarios,
        variables="all",
        save=TRUE,
        name="scenarios",
        show=FALSE,
        nhist=10L
    )

Key Parameters
**************

Data Transformations
--------------------

The ``trans`` parameter specifies how each variable should be transformed:

- ``0``: Take the natural logarithm
- ``1``: Divide by 100 (for percentage or rate data)

Example:

.. code-block:: python

    # For 2 quarterly variables, 3 monthly, 3 weekly
    trans = [
        np.array([0, 1]),        # Q: log first, divide second by 100
        np.array([1, 1, 1]),     # M: divide all by 100
        np.array([1, 1, 1])      # W: divide all by 100
    ]

Hyperparameters
---------------

The Minnesota-type prior uses 5 hyperparameters per frequency level:

1. ``lambda1``: Overall tightness (smaller = tighter prior, more shrinkage)
2. ``lambda2``: Cross-variable shrinkage (larger = more independent variables)
3. ``lambda3``: Lag decay parameter (fixed at 1 in current implementation)
4. ``lambda4``: Exogenous variable tightness
5. ``lambda5``: Intercept tightness

Typical values: ``[0.09, 4.3, 1, 2.7, 4.3]``

Number of Lags
--------------

The ``nlags`` parameter must have one entry for each frequency transition:

- For ["Q", "M", "W"]: ``nlags = [nlags_QM, nlags_MW]``
- Each value must be at least as large as the frequency ratio
- Example: Q→M ratio is 3, M→W ratio is 4, so ``nlags = [6, 4]`` works

Forecast Horizon
----------------

The ``H`` parameter specifies forecast horizon in the **highest frequency**:

- For ["Q", "M", "W"] data with H=52: forecasts 52 weeks ahead
- After aggregation to "Q": results in 52/12 ≈ 4 quarters ahead

Additional Examples
*******************

The ``examples/`` directory contains complete, runnable examples:

- ``basic_example.py``: Step-by-step tutorial for Python users
- ``advanced_example.py``: Advanced features including hyperparameter optimization
- ``r_example.R``: Complete R workflow with reticulate
- ``test_basic.py``: Simple test suite to verify installation
- ``generate_data.py``: Script to create synthetic multi-frequency data

See Also
********

- :doc:`intro` - Introduction and installation
- :doc:`MBFVAR` - Complete API reference
- `GitHub Repository <https://github.com/laurentflorin/MBFVAR>`_ - Source code and latest updates
