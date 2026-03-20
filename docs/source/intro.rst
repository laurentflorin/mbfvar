Introduction
============

``MBFVAR`` (Multi-Frequency Bayesian Vector Autoregression) is a Python package for handling, disaggregating, and forecasting time series data at multiple frequencies using Bayesian VAR models.

Overview
********

Modern economic and financial data is often available at different frequencies:

- Economic indicators: GDP (quarterly), Industrial Production (monthly)
- Financial data: Stock prices (daily), Interest rates (daily/weekly)
- Survey data: Consumer confidence (monthly), Business surveys (quarterly)

MBFVAR allows you to:

- **Model jointly** variables observed at different frequencies
- **Disaggregate** lower-frequency data to match higher-frequency observations
- **Generate forecasts** at the highest frequency and aggregate as needed
- **Impose constraints** on forecasts (conditional forecasting)
- **Compare scenarios** with different assumptions
- **Optimize priors** using Bayesian hyperparameter tuning

Key Features
************

Multi-Frequency Modeling
------------------------

Work seamlessly with data at different temporal granularities:

- Yearly (Y), Quarterly (Q), Monthly (M), Weekly (W), Daily (D)
- Automatic handling of frequency ratios and temporal aggregation
- Efficient disaggregation following Schorfheide and Song (2015)

Bayesian Framework
------------------

The package implements a rigorous Bayesian approach:

- **Minnesota-type prior**: Shrinks coefficients toward a random walk
- **Gibbs sampling**: Efficient MCMC algorithm for posterior inference
- **Hyperparameter optimization**: Bayesian and Mango optimizers for prior tuning
- **Predictive distributions**: Full characterization of forecast uncertainty

Flexible Forecasting
--------------------

Generate various types of forecasts:

- **Unconditional forecasts**: Based solely on historical data
- **Conditional forecasts**: Impose constraints on specific variables/periods
- **Scenario analysis**: Compare multiple "what-if" scenarios
- **Model comparison**: Evaluate forecasts across different model vintages

Rich Visualization
------------------

Create publication-ready graphics:

- **Fan charts**: Visualize forecast uncertainty with probability bands
- **Mean forecast plots**: Show point forecasts and historical data
- **Scenario comparisons**: Side-by-side comparison of forecast paths
- **Customizable**: Save to file or display interactively

R Integration
-------------

Full compatibility with R through ``reticulate``:

- Call MBFVAR functions directly from R scripts
- Seamless data exchange between R and Python
- Leverage R's data manipulation with Python's modeling

Installation
************

Install from GitHub using pip:

.. code-block:: bash

    pip install git+https://github.com/laurentflorin/MBFVAR.git

For development or to run examples:

.. code-block:: bash

    git clone https://github.com/laurentflorin/MBFVAR.git
    cd MBFVAR
    pip install -e .

Requirements
************

- **Python**: >= 3.8
- **Core dependencies**: NumPy, SciPy, Pandas, Matplotlib
- **Visualization**: Plotly, Seaborn, fanchart
- **Optimization**: bayesian-optimization, arm-mango
- **I/O**: openpyxl, xlsxwriter

All dependencies are automatically installed when using pip.

Quick Start
***********

Here's a minimal working example:

.. code-block:: python

    import MBFVAR
    import pandas as pd
    import numpy as np

    # 1. Load your data
    data_q = pd.read_excel("data.xlsx", sheet_name="Q", index_col=0)  # Quarterly
    data_m = pd.read_excel("data.xlsx", sheet_name="M", index_col=0)  # Monthly
    data_w = pd.read_excel("data.xlsx", sheet_name="W", index_col=0)  # Weekly

    # 2. Prepare data
    data = [data_q, data_m, data_w]
    trans = [np.array([1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])]
    frequencies = ["Q", "M", "W"]

    data_in = MBFVAR.mbfvar_data(data, trans, frequencies)

    # 3. Fit model
    model = MBFVAR.MixedFrequencyBVAR(nsim=1000, nburn=0.5, nlags=[6, 4], thining=1)
    hyp = [[0.09, 4.3, 1, 2.7, 4.3], [0.09, 4.3, 1, 2.7, 4.3]]
    model.fit(data_in, hyp=hyp)

    # 4. Forecast and visualize
    model.forecast(H=52)  # 52 weeks ahead
    model.fanchart(variables="all", show=True)

Getting Help
************

- **Examples**: See the ``examples/`` directory for comprehensive tutorials
- **API Documentation**: Full reference at :doc:`MBFVAR`
- **Examples Guide**: Detailed examples at :doc:`examples`
- **Issues**: Report bugs at `GitHub Issues <https://github.com/laurentflorin/MBFVAR/issues>`_

Methodology
***********

The MBFVAR package implements the mixed-frequency VAR framework developed by
Schorfheide and Song (2015). Key methodological features:

State-Space Representation
--------------------------

The model uses a state-space form to handle mixed frequencies:

- **State equation**: VAR dynamics in the highest frequency
- **Observation equation**: Links observed data to latent high-frequency state
- **Kalman filter**: Efficiently handles missing observations

Minnesota Prior
---------------

The prior shrinks VAR coefficients toward a parsimonious benchmark:

- Own first lags shrink toward random walk (coefficient = 1)
- Other coefficients shrink toward zero
- Shrinkage strength controlled by hyperparameters
- Prevents overfitting in high-dimensional VARs

Gibbs Sampling
--------------

Posterior inference via MCMC:

1. Initialize parameters
2. Draw VAR coefficients | error covariance
3. Draw error covariance | VAR coefficients
4. Repeat steps 2-3 for nsim iterations
5. Discard burn-in period
6. Use remaining draws for inference

Citation
********

If you use MBFVAR in your research, please cite:

.. code-block:: text

    Florin, Laurent. (2024). MBFVAR: Multi-Frequency Bayesian Vector Autoregression.
    GitHub repository, https://github.com/laurentflorin/MBFVAR

References
**********

Schorfheide, F., & Song, D. (2015). Real-time forecasting with a mixed-frequency VAR.
*Journal of Business & Economic Statistics*, 33(3), 366-380.

Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for vector
autoregressions. *Review of Economics and Statistics*, 97(2), 436-451.

License
*******

This project is licensed under the terms specified in the repository.

Author
******

**Laurent Florin**

Email: laurent.florin@efv.admin.ch

