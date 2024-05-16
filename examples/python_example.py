
io_data = "hist.xlsx"


H = 96        # forecast horizon
nsim     = 100  # number of draws from Posterior Density
nburn    = 0.5  # number of draws to discard
nlags = [6,4]
thining = 1

hyp = (0.09, 4.3, 1, 2.7, 4.3)

frequencies = ["Q","M","W"]

data = []
trans = [np.array((1)), np.array((1,1,1)), np.array((1,1,1,1))]

for freq in range(len(frequencies)):
            freq = frequencies[freq]
            data_temp = pd.read_excel(io_data, sheet_name = freq, index_col = 0)
            data.append(data_temp)
            
            
conditionals = pd.DataFrame({'w_1' : [0.018, 0.025, np.nan, np.nan, 0.0228, 0.05],
                            'm_2' : [ np.nan, 0.002, 0.01 , 0.01, np.nan, np.nan]})            
            
            
data_in = mufbvar_data(data, trans, frequencies)



model =  multifrequency_var(nsim, nburn, nlags ,thining)

model.fit(data_in, hyp = hyp)

model.forecast()

model.aggregate(frequency = "Q")

#model.to_excel('out_test.xlsx', agg = True)

model.mean_plot(1, variables = "all", save = False, show = True)

model.fanchart(variables = "all", save = False, show = True, agg = True, nhist = 10)
