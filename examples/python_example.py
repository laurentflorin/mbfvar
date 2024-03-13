import MUFBVAR

io_data = "hist.xlsx"
io_cond = "cond.xlsx"
io_trans = "trans.xlsx"

H = 96  #66      # forecast horizon
nsim     = 100  # number of draws from Posterior Density
nburn    = 0.5  # number of draws to discard
nlags = [6,4]
thining = 1

hyp = (0.09, 4.3, 1, 2.7, 4.3)

model =  multifrequency_var(["Q","M", "W"], H, nsim, nburn, nlags ,thining)
model.fit(io_data, io_cond, io_trans, hyp = hyp)