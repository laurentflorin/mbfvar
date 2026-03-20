# MUFBVAR R Example
# ===================
#
# This example demonstrates how to use MUFBVAR in R using the reticulate package.
# The reticulate package allows R to interface with Python packages.
#
# Prerequisites:
# 1. Install reticulate: install.packages("reticulate")
# 2. Set up a Python virtual environment with MUFBVAR installed
# 3. Have the hist.xlsx data file in the working directory

library(reticulate)

# Configure Python virtual environment
# IMPORTANT: Update this path to your actual virtual environment
# use_virtualenv("/path/to/your/virtualenv", required = TRUE)
# OR use a conda environment:
# use_condaenv("your-env-name", required = TRUE)

# Set working directory to the location of this script
# (Uncomment if running in RStudio)
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# ==============================================================================
# 1. IMPORT PYTHON MODULES
# ==============================================================================
cat("Importing MUFBVAR and dependencies...\n")

mufbvar <- import("MUFBVAR")
pd <- import("pandas")
np <- import("numpy")
pickle <- import("pickle")

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
cat("\nLoading data from hist.xlsx...\n")

io_data <- "hist.xlsx"
frequencies <- list("Q", "M", "W")

# Load data for each frequency
data <- list()
for (i in 1:length(frequencies)) {
    freq <- frequencies[[i]]
    data_temp <- pd$read_excel(io_data, sheet_name = freq, index_col = 0L)
    data <- append(data, list(data_temp))
    cat(sprintf("  Loaded %s data\n", freq))
}

# ==============================================================================
# 3. SPECIFY TRANSFORMATIONS
# ==============================================================================
cat("\nSpecifying transformations...\n")

# Transformations: 0 = log, 1 = divide by 100
# Note: Use c() to create vectors in R, then convert to numpy arrays
trans <- list(
    np$array(c(1L, 1L)),           # Quarterly: 2 variables
    np$array(c(1L, 1L, 1L)),       # Monthly: 3 variables
    np$array(c(1L, 1L, 1L))        # Weekly: 3 variables
)

# ==============================================================================
# 4. PREPARE DATA
# ==============================================================================
cat("\nPreparing data for MUFBVAR...\n")

data_in <- mufbvar$mufbvar_data(data, trans, frequencies)
cat("  Data prepared successfully!\n")

# ==============================================================================
# 5. MODEL SPECIFICATION
# ==============================================================================
cat("\nSetting up model parameters...\n")

H <- 52L               # Forecast horizon (in highest frequency)
nsim <- 1000L          # Number of posterior draws
nburn <- 0.5           # Burn-in proportion
nlags <- list(6L, 4L)  # Number of lags
thining <- 1L          # Thinning parameter

# Hyperparameters for Minnesota prior
hyp <- list(
    list(0.09, 4.3, 1, 2.7, 4.3),
    list(0.09, 4.3, 1, 2.7, 4.3)
)

cat(sprintf("  Simulations: %d (burn-in: %d)\n", nsim, as.integer(nsim * nburn)))
cat(sprintf("  Lags: [%d, %d]\n", nlags[[1]], nlags[[2]]))

# ==============================================================================
# 6. INITIALIZE AND FIT MODEL
# ==============================================================================
cat("\nInitializing model...\n")

model <- mufbvar$multifrequency_var(nsim, nburn, nlags, thining)

cat("Fitting model (this may take a few moments)...\n")
model$fit(data_in, hyp = hyp)
cat("  Model fitted successfully!\n")

# ==============================================================================
# 7. GENERATE FORECAST
# ==============================================================================
cat("\nGenerating unconditional forecast...\n")

model$forecast(H)
cat(sprintf("  Generated %d-period ahead forecast\n", H))

# ==============================================================================
# 8. AGGREGATE TO QUARTERLY
# ==============================================================================
cat("\nAggregating forecasts to quarterly frequency...\n")

model$aggregate(frequency = "Q")
cat("  Aggregation complete!\n")

# ==============================================================================
# 9. SAVE RESULTS
# ==============================================================================
cat("\nSaving results...\n")

model$to_excel("forecasts_r_weekly.xlsx", agg = FALSE)
cat("  Saved weekly forecasts to: forecasts_r_weekly.xlsx\n")

model$to_excel("forecasts_r_quarterly.xlsx", agg = TRUE)
cat("  Saved quarterly forecasts to: forecasts_r_quarterly.xlsx\n")

# ==============================================================================
# 10. VISUALIZE RESULTS
# ==============================================================================
cat("\nGenerating visualizations...\n")

# Fan chart (quarterly)
cat("  Creating fan chart (quarterly)...\n")
model$fanchart(
    variables = "all",
    save = TRUE,
    show = FALSE,
    agg = TRUE,
    nhist = 10L,
    name = "fanchart_r_quarterly"
)

# Mean plot
cat("  Creating mean forecast plot...\n")
model$mean_plot(
    variables = "all",
    save = TRUE,
    show = FALSE,
    name = "mean_forecast_r"
)

# ==============================================================================
# CONDITIONAL FORECASTING (OPTIONAL)
# ==============================================================================
cat("\n--- CONDITIONAL FORECASTING ---\n")

# Create a DataFrame with conditions (use pd$DataFrame with named list)
# Note: Use np$nan for unconstrained periods
conditionals <- pd$DataFrame(list(
    'w_1' = c(0.018, 0.025, np$nan, np$nan, 0.0228, 0.05),
    'm_2' = c(np$nan, 0.002, 0.01, 0.01, np$nan, np$nan)
))

cat("Generating conditional forecast...\n")
model$forecast(H, conditionals)

model$aggregate(frequency = "Q")
model$to_excel("forecasts_r_conditional.xlsx", agg = TRUE)
cat("  Saved conditional forecast\n")

# ==============================================================================
# SCENARIO ANALYSIS (OPTIONAL)
# ==============================================================================
cat("\n--- SCENARIO ANALYSIS ---\n")

# Define multiple scenarios
scenario_good <- pd$DataFrame(list(
    'w_1' = c(0.02, 0.025, rep(np$nan, H-2))
))

scenario_bad <- pd$DataFrame(list(
    'w_1' = c(-0.02, -0.025, rep(np$nan, H-2))
))

# NULL for baseline (unconditional forecast)
conditionals_list <- list(scenario_good, scenario_bad, NULL)
scenario_names <- list("Optimistic", "Pessimistic", "Baseline")

cat("Comparing scenarios...\n")
scenarios <- model$scenario_forecast(H, conditionals_list, scenario_names, agg = TRUE)

# Plot scenarios
model$scenario_plot(
    scenario_dict = scenarios,
    variables = "all",
    save = TRUE,
    name = "scenario_comparison_r",
    show = FALSE,
    nhist = 10L
)

# ==============================================================================
# HYPERPARAMETER OPTIMIZATION (OPTIONAL - ADVANCED)
# ==============================================================================
cat("\n--- HYPERPARAMETER OPTIMIZATION (COMMENTED OUT) ---\n")
cat("Uncomment the code below to run hyperparameter optimization\n")

# # This requires scipy
# scipy_stats <- import("scipy.stats")
#
# # Define parameter space
# param_space <- list(
#     lambda1_1 = scipy_stats$uniform(0.001, 20),
#     lambda2_1 = scipy_stats$uniform(0.01, 10),
#     lambda4_1 = scipy_stats$uniform(0.01, 10),
#     lambda5_1 = scipy_stats$uniform(0.01, 10),
#     lambda1_2 = scipy_stats$uniform(0.001, 20),
#     lambda2_2 = scipy_stats$uniform(0.01, 10),
#     lambda4_2 = scipy_stats$uniform(0.01, 10),
#     lambda5_2 = scipy_stats$uniform(0.01, 10)
# )
#
# cat("Optimizing hyperparameters...\n")
# hyp_optimal <- model$update_hyperparameters_mango_rmse(
#     data_in,
#     param_space,
#     H = 2L,
#     init_points = 3L,
#     n_iter = 8L,
#     nsim = 500L,
#     njobs = 1L,
#     var_of_interest = list("q_1"),
#     temp_agg = 'mean',
#     save = FALSE
# )

# ==============================================================================
# SUMMARY
# ==============================================================================
cat("\n========================================================================\n")
cat("R EXAMPLE COMPLETED SUCCESSFULLY!\n")
cat("========================================================================\n")

cat("\nGenerated files:\n")
cat("  - forecasts_r_weekly.xlsx\n")
cat("  - forecasts_r_quarterly.xlsx\n")
cat("  - forecasts_r_conditional.xlsx\n")
cat("  - fanchart_r_quarterly.png\n")
cat("  - mean_forecast_r.png\n")
cat("  - scenario_comparison_r.png\n")

cat("\n========================================================================\n")
