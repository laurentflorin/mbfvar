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

model.fanchart(variables = "all", save = False, show = False, agg = False, nhist = 150)
