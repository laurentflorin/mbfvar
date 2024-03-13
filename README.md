# MUFBVAR

This Python class is designed for handling and forecasting multi-frequency data.

## Class Methods

### `__init__(self, frequencies, H, nsim, nburn_perc, nlags, thining)`

Initializes the multifrequency_var object.

- `frequencies`: List of the frequencies of the data, in order lowest to highest "Y", "Q", "M", "W", "D"
- `H`: Numeric. Forecast Horizon in the highest frequency
- `nsim`: Numeric. Number of simulations
- `nburn_perc`: Numeric. Between 0 and 1, proportion of simulations to throw away as burn in.
- `nlags`: Numeric. Number of lags in the highest frequency
- `thining`: Numeric. To save only every nth draw

### `fit(self, io_data, io_conditionals, io_trans, hyp)`

Fit the specified model to the data.

### `forecast(self)`

Generates a forecast.

### `aggregate(self, frequency)`

Aggregates the Mean, Median and quantiles in the highest frequency to the desired frequency.

- `frequency`: The frequency to which the data should be aggregated to

### `save(self, filename = "mufbvar_model.pkl")`

Saves the MFBVAR Object.

- `filename`: Path where to save the object. End must be .pkl

### `to_excel(self, filename, agg = False)`

Exports the data to an Excel file.

- `agg`: Boolean. Should the aggregated series be shown
- `filename`: The name of the output file.

### `mean_plot(self,frequency, variables = "all", save = True, name = "Output", show = True)`

Generates a mean plot for the specified variables.

- `frequency`: The frequency of the data
- `variables`: List of strings. Variables for which the plot should be generated, all if it should be generated for all
- `save`: Boolean. Whether the plots should be saved. The default is True.
- `name`: String. If the plots should be saved, path/name not including filetype. The default is None.
- `show`: Boolean. Whether the plots should be shown. Default is True.

### `fanchart(self, variables = "all", save = True, name = "Fancharts", show = True, agg = True, nhist = 5)`

Generates a fan chart for the specified variables.

- `variables`: List of strings. Variables for which the plot should be generated, all if it should be generated for all
- `save`: Boolean. Whether the plots should be saved. The default is True.
- `name`: String. If the plots should be saved, path/name not including filetype. The default is None.
- `show`: Boolean. Whether the plots should be shown. Default is True.
- `nhist`: Int. Number of historical periods that should be shown on the plot. Default is 5.

## Input Data

## Use in Python

``` python
from MUFBVAR.mufbvar import *

io_data = "hist.xlsx"
io_conditionals = "cond.xlsx"
io_trans = "trans.xlsx"

H = 96  #66      # forecast horizon
nsim     = 100  # number of draws from Posterior Density
nburn    = 0.5  # number of draws to discard
nlags = [6,4]
thining = 1

hyp = (0.09, 4.3, 1, 2.7, 4.3)

model =  multifrequency_var(["Q","M", "W"], H, nsim, nburn, nlags ,thining)

model.fit(io_data, io_conditionals, io_trans, hyp = hyp)

model.forecast()

model.mean_plot(1, variables = "all", save = False, show = True)

model.fanchart(variables = "all", save = False, show = True, agg = False, nhist = 150)

```
## Use in R