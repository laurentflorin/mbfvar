library(reticulate)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

mufbvar <- import("MUFBVAR.mufbvar")
mufbvar_data <- import("MUFBVAR.mufbvar_data")
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
data_in <- mufbvar_data$mufbvar_data(data, trans, frequencies)


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

model$fanchart(variables = "all", save = False, show = True, agg = True, nhist = 10)