
system("pip install git+https://gitea.efv.admin.ch/efv_fs/MUFBVAR.git")

library(reticulate)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

mufbvar <- import("MUFBVAR.mufbvar")

io_data <- "hist.xlsx"
io_conditionals <- "cond.xlsx"
io_trans <- "trans.xlsx"

H <- 96L        # forecast horizon
nsim <- 100L  # number of draws from Posterior Density
nburn <- 0.5  # number of draws to discard
nlags <- list(6L,4L)
thining = 1

hyp <- c(0.09, 4.3, 1, 2.7, 4.3)

model <- mufbvar$multifrequency_var(list("Q","M", "W"), H, nsim, nburn, nlags ,thining)

model$fit(io_data, io_conditionals, io_trans, hyp = hyp)

model$forecast()

model$mean_plot(1L, variables = "all", save = FALSE, show = TRUE)

model$fanchart(variables = "all", save = FALSE, show = TRUE, agg = FALSE, nhist = 150L)
