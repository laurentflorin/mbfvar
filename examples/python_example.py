import MUFBVAR
import pandas as pd
import numpy as np
import pickle


# Preparations
#---------------------

io_data = "hist.xlsx"

#Model Specification
H = 96          # forecast horizon
nsim = 100      # number of draws from Posterior Density
nburn = 0.5     # number of draws to discard
nlags = [6,4]   # Number of lags
thining = 1     # Thining 

hyp = [[0.09, 4.3, 1, 2.7, 4.3], [0.09, 4.3, 1, 2.7, 4.3]] # Hyperparameters see documentation for details

frequencies = ["Q","M","W"] # Frequencies



# Load the data
# --------------
data = []
for freq in range(len(frequencies)):
        freq = frequencies[freq]
        data_temp = pd.read_excel(io_data, sheet_name = freq, index_col = 0)
        data.append(data_temp)

#Transformations
trans = [np.array((1)), np.array((1,1,1)), np.array((1,1,1,1))]    


# Initialize data class            
data_in = MUFBVAR.mufbvar_data(data, trans, frequencies)


# Fit and Forecast
#--------------------

# Initialize model class    
model =  MUFBVAR.multifrequency_var(nsim, nburn, nlags ,thining)

# Estimate the model
model.fit(data_in, hyp = hyp)

# Conditional forecasts

conditionals = pd.DataFrame({'w_1' : [0.018, 0.025, np.nan, np.nan, 0.0228, 0.05],
                        'm_2' : [ np.nan, 0.002, 0.01 , 0.01, np.nan, np.nan]})   

# Create forecasts in highest frequency
model.forecast(H, conditionals)

# Aggregate
model.aggregate(frequency = "Q")

# Save results
#------------
#model.to_excel('out_test.xlsx', agg = True)
#model.save("model_2002_Q4")

# Plots
#---------

model.mean_plot(variables = "all", save = False, show = True)

model.fanchart(variables = "all", save = False, show = True, agg = True, nhist = 10)

# Optimizing Hyperparameters
#------------------------------

# Define boundaries for each hyperparameter, see documentation for details
pbounds = {'lambda1_1': (0.001, 20), 'lambda2_1': (0.01, 10), 'lambda4_1': (0.01, 10), 'lambda5_1': (0.01, 10), 'lambda1_2': (0.001, 20), 'lambda2_2': (0.01, 10), 'lambda4_2': (0.01, 10), 'lambda5_2': (0.01, 10)}
init_points = 3 # number of random points
n_iter = 8 # number of baysian optimization steps
nsim = 100 # number of simulations 

hyp = model.update_hyperparameters(data_in, pbounds, init_points, n_iter, nsim, save = False, name = "hyp.txt")

# Scenario Analysis
#-------------------------------

# We can compare different scenarios

conditionals = [pd.DataFrame({'w_1' : [0.018, 0.025, np.nan, np.nan, 0.0228, 0.05],
                        'm_2' : [ 0.3, 0.002, 0.01 , 0.01, np.nan, np.nan]}),
                pd.DataFrame({'w_1' : [-0.02, -0.25, np.nan, np.nan, -0.228, 0.1],
                        'm_2' : [ -0.2, -0.012, 0 , 0.1, np.nan, np.nan]}), 
                None
                ]

names = ["good", "bad", "base"]

out_scenarios = model.scenario_forecast(H, conditionals, names, agg = True)

# Scenario Plot
model.scenario_plot(scenario_dict = out_scenarios, variables = "all", save = False, name = "Scenario", show = True, nhist = 10)

# Compare with model from last quarter
#--------------------------------------

#load previous quarter model:
file = open("model_2001_Q3.pkl",'rb')
model_2001_Q3 = pickle.load(file)

model_names = ["2001-Q4", "2001-Q3"]
multifrquency_var_models = [model_2001_Q3]

model.compare_models(multifrquency_var_models, model_names, agg = True, variables = "all", save = False, name = "Comparison", show = True, nhist = 5)

model.compare_models(multifrquency_var_models, model_names, agg = False, variables = ["q_1"], save = False, name = "Comparison", show = True, nhist = 20)