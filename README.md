# MUFBVAR

Module to handel, disaggregate and forecast Multiple frequency data using Bayesian VAR Models.

[Documentation PDF](./docs/_build/simplepdf/MUFBVAR.pdf)

[Documentation Markdown](./docs/_build/markdown/index.md)

[Documentation html](./docs/_build/html/index.html)

# Installation

Install via pip:
```console
foo@bar:~$ pip install git+https://gitea.efv.admin.ch/efv_fs/MUFBVAR.git
```



# Example code

The module can be used in python

```python
import MUFBVAR
import pandas as pd
import numpy as np


io_data = "hist.xlsx"

#Model Specification
H = 96        # forecast horizon
nsim     = 100  # number of draws from Posterior Density
nburn    = 0.5  # number of draws to discard
nlags = [6,4]
thining = 1

hyp = [[0.09, 4.3, 1, 2.7, 4.3], [0.09, 4.3, 1, 2.7, 4.3]]

frequencies = ["Q","M","W"]



# Load the data
data = []
for freq in range(len(frequencies)):
            freq = frequencies[freq]
            data_temp = pd.read_excel(io_data, sheet_name = freq, index_col = 0)
            data.append(data_temp)
#Transformations
trans = [np.array((1)), np.array((1,1,1)), np.array((1,1,1,1))]    
            

#Initialize data class            
data_in = MUFBVAR.mufbvar_data(data, trans, frequencies)


#Initialize model class    
model =  MUFBVAR.multifrequency_var(nsim, nburn, nlags ,thining)

#Estimate the model
model.fit(data_in, hyp = hyp)

#Conditional forecasts
conditionals = pd.DataFrame({'w_1' : [0.018, 0.025, np.nan, np.nan, 0.0228, 0.05],
                            'm_2' : [ np.nan, 0.002, 0.01 , 0.01, np.nan, np.nan]})   

#Create forecasts in highest frequency
model.forecast(H, conditionals)

#Aggregate
model.aggregate(frequency = "Q")

#Save results
#model.to_excel('out_test.xlsx', agg = True)

#Plots
model.mean_plot(variables = "all", save = False, show = True)

model.fanchart(variables = "all", save = False, show = True, agg = True, nhist = 10)

# Optimizing Hyperparameters

pbounds = {'lambda1_1': (0.001, 20), 'lambda2_1': (0.01, 10), 'lambda4_1': (0.01, 10), 'lambda5_1': (0.01, 10), 'lambda1_2': (0.001, 20), 'lambda2_2': (0.01, 10), 'lambda4_2': (0.01, 10), 'lambda5_2': (0.01, 10)}
init_points = 3
n_iter = 8
nsim = 100

hyp = model.update_hyperparameters(data_in, pbounds, init_points, n_iter, nsim, save = False, name = "hyp.txt")

# scenario analysis

conditionals = [pd.DataFrame({'w_1' : [0.018, 0.025, np.nan, np.nan, 0.0228, 0.05],
                            'm_2' : [ 0.3, 0.002, 0.01 , 0.01, np.nan, np.nan]}),
                    pd.DataFrame({'w_1' : [-0.02, -0.25, np.nan, np.nan, -0.228, 0.1],
                            'm_2' : [ -0.2, -0.012, 0 , 0.1, np.nan, np.nan]}), 
                            None]

names = ["good", "bad", "base"]

out_scenarios = model.scenario_forecast(H, conditionals, names, agg = True)

# Scenario Plot
model.scenario_plot(scenario_dict = out_scenarios, variables = "all", save = False, name = "Scenario", show = True, nhist = 10)

#compare with model from last quarter

data_old = []
idx = ['2001-07-01', '2001-09-01', '2001-09-28']
for i in range(len(frequencies)):
                freq = frequencies[i]
                data_temp = pd.read_excel(io_data, sheet_name = freq, index_col = 0)
                data_old.append(data_temp.loc[:idx[i]])


data_in_old = MUFBVAR.mufbvar_data(data_old, trans, frequencies)


#Initialize model class    
model_q1 =  MUFBVAR.multifrequency_var(nsim, nburn, nlags ,thining)

#Estimate the model
model_q1.fit(data_in_old, hyp = hyp)


#Create forecasts in highest frequency
model_q1.forecast(H = 96)

#Aggregate
model_q1.aggregate(frequency = "Q")

model_names = ["2001-Q4", "2001-Q3"]

multifrquency_var_models = [model_q1]

model.compare_models(multifrquency_var_models, model_names, agg = True, variables = "all", save = False, name = "Comparison", show = True, nhist = 5)

model.compare_models(multifrquency_var_models, model_names, agg = False, variables = ["q_1"], save = False, name = "Comparison", show = True, nhist = 20)

```

The module can also be used in r, using the reticulate package:

```r
library(reticulate)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

mufbvar <- import("MUFBVAR")
pd <- impoty("pandas")
np <- import("numpy")

io_data <- "hist.xlsx"


H <- 96L        # forecast horizon
nsim <- 20000L  # number of draws from Posterior Density
nburn <- 0.5  # number of draws to discard
nlags <- list(6L,4L)
thining = 1

hyp <- c(0.09, 4.3, 1, 2.7, 4.3)

frequencies <- list("Q", "M", "W")


data <- list()
for (freq in 1:length(frequencies)) {
    freq <- frequencies[[freq]]
    data_temp <- pd$read_excel(io_data, sheet_name = freq, index_col = 0)
    data <- append(data, list(data_temp))
}

#Transformations
trans <- list(np$array((1)), np$array((1,1,1)), np$array((1,1,1,1)))
            

#Initialize data class            
data_in <- mufbvar$mufbvar_data(data, trans, frequencies)


#Initialize model class    
model <-  mufbvar$multifrequency_var(nsim, nburn, nlags ,thining)

#Estimate the model
model$fit(data_in, hyp = hyp)

#Conditional forecasts
conditionals <- pd.DataFrame({'w_1' : [0.018, 0.025, np.nan, np.nan, 0.0228, 0.05],
                            'm_2' : [ np.nan, 0.002, 0.01 , 0.01, np.nan, np.nan]})   

#Create forecasts in highest frequency
model$forecast(H, conditionals)

#Aggregate
model$aggregate(frequency = "Q")

#Save results
#model$to_excel('out_test.xlsx', agg = True)

#Plots
model$mean_plot(variables = "all", save = False, show = True)

model$fanchart(variables = "all", save = False, show = True, agg = True, nhist = 10L)
```