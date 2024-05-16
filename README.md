# MUFBVAR

Module to handel, disaggregate and forecast Multiple frequency data using Bayesian VAR Models.


[Documentation](./docs/_build/markdown/index.md)

# Installation

Install via pip:
```console
foo@bar:~$ pip install git+https://gitea.efv.admin.ch/efv_fs/MUFBVAR.git
```



# Example code

The module can be used in python

```python
from MUFBVAR.mufbvar import *

io_data = "hist.xlsx"
io_conditionals = "cond.xlsx"
io_trans = "trans.xlsx"

H = 96        # forecast horizon
nsim     = 100  # number of draws from Posterior Density
nburn    = 0.5  # number of draws to discard
nlags = [6,4]
thining = 1

hyp = (0.09, 4.3, 1, 2.7, 4.3)

model =  multifrequency_var(["Q","M","W"], H, nsim, nburn, nlags ,thining)

model.fit(io_data, io_conditionals, io_trans, hyp = hyp)

model.forecast()

model.aggregate(frequency = "Q")

#model.to_excel('out_test.xlsx', agg = True)

model.mean_plot(1, variables = "all", save = False, show = True)

model.fanchart(variables = "all", save = False, show = True, agg = True, nhist = 10)

```

The module can also be used in r, using the reticulate package:

```r
library(reticulate)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

mufbvar <- import("MUFBVAR.mufbvar")

io_data <- "hist_fctest.xlsx"
io_conditionals <- "cond_fctest.xlsx"
io_trans <- "trans.xlsx"

H <- 96L        # forecast horizon
nsim <- 20000L  # number of draws from Posterior Density
nburn <- 0.5  # number of draws to discard
nlags <- list(6L,4L)
thining = 1

hyp <- c(0.09, 4.3, 1, 2.7, 4.3)

model <- mufbvar$multifrequency_var(list("Q","M", "W"), H, nsim, nburn, nlags ,thining)

model$fit(io_data, io_conditionals, io_trans, hyp = hyp)

model$forecast()

model$mean_plot(1L, variables = "all", save = FALSE, show = TRUE)

model$fanchart(variables = "all", save = FALSE, show = TRUE, agg = FALSE, nhist = 150L)
```