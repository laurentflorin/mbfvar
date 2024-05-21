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

out_scenarios = MUFBVAR.scenario_forecast(H, conditionals, names, agg = True)

# Scenario Plot
scenario_plot(out_scenarios, variables = "all", save = False, name = "Scenario", show = True, nhist = 10)