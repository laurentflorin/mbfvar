# MUFBVAR package

## Subpackages

* [MUFBVAR.cholcov package](MUFBVAR.cholcov.md)
  * [Submodules](MUFBVAR.cholcov.md#submodules)
  * [MUFBVAR.cholcov.cholcov_module module](MUFBVAR.cholcov.md#module-MUFBVAR.cholcov.cholcov_module)
    * [`cholcovOrEigendecomp()`](MUFBVAR.cholcov.md#MUFBVAR.cholcov.cholcov_module.cholcovOrEigendecomp)
  * [Module contents](MUFBVAR.cholcov.md#module-MUFBVAR.cholcov)
* [MUFBVAR.inverse package](MUFBVAR.inverse.md)
  * [Submodules](MUFBVAR.inverse.md#submodules)
  * [MUFBVAR.inverse.matrix_inversion module](MUFBVAR.inverse.md#module-MUFBVAR.inverse.matrix_inversion)
    * [`invert_matrix()`](MUFBVAR.inverse.md#MUFBVAR.inverse.matrix_inversion.invert_matrix)
  * [Module contents](MUFBVAR.inverse.md#module-MUFBVAR.inverse)
* [MUFBVAR.pseudo_inverse package](MUFBVAR.pseudo_inverse.md)
  * [Submodules](MUFBVAR.pseudo_inverse.md#submodules)
  * [MUFBVAR.pseudo_inverse.pseudo_inverse module](MUFBVAR.pseudo_inverse.md#module-MUFBVAR.pseudo_inverse.pseudo_inverse)
    * [`calculate_pseudo_inverse()`](MUFBVAR.pseudo_inverse.md#MUFBVAR.pseudo_inverse.pseudo_inverse.calculate_pseudo_inverse)
  * [Module contents](MUFBVAR.pseudo_inverse.md#module-MUFBVAR.pseudo_inverse)
* [MUFBVAR.solve package](MUFBVAR.solve.md)
  * [Submodules](MUFBVAR.solve.md#submodules)
  * [MUFBVAR.solve.linalg_solve module](MUFBVAR.solve.md#module-MUFBVAR.solve.linalg_solve)
    * [`linalg_solve()`](MUFBVAR.solve.md#MUFBVAR.solve.linalg_solve.linalg_solve)
  * [Module contents](MUFBVAR.solve.md#module-MUFBVAR.solve)

## Submodules

## MUFBVAR.mfbvar_funcs module

This file contains functions used in mf_bvar_estim

@author: florinl

### MUFBVAR.mfbvar_funcs.calc_yyact(hyp, YY, spec)

### Parameters

hyp
: DESCRIPTION.

YY
: DESCRIPTION.

spec
: DESCRIPTION.

efficient
: DESCRIPTION.

### Returns

None.

### MUFBVAR.mfbvar_funcs.initialize(GAMMAs, GAMMAz, GAMMAc, GAMMAu, LAMBDAs, LAMBDAz, LAMBDAc, LAMBDAu, LAMBDAs_t, LAMBDAz_t, LAMBDAc_t, LAMBDAu_t, sig_qq, sig_mm, sig_qm, sig_mq, Zm, YDATA, init_mean, init_var, spec, Nm)

### MUFBVAR.mfbvar_funcs.mdd_(hyp, YY, spec)

### Parameters

hyp
: DESCRIPTION.

YY
: DESCRIPTION.

spec
: DESCRIPTION.

efficient
: DESCRIPTION.

### Returns

None.

### MUFBVAR.mfbvar_funcs.prior_init(hyp, YY, spec)

### Parameters

hyp
: DESCRIPTION.

YY
: DESCRIPTION.

spec
: DESCRIPTION.

### Returns

Phi_tilde

sigma

### MUFBVAR.mfbvar_funcs.prior_pdf(hyp, YY, spec, PHI, SIG)

### Parameters

hyp
: DESCRIPTION.

YY
: DESCRIPTION.

spec
: DESCRIPTION.

PHI
: DESCRIPTION.

SIG
: DESCRIPTION.

### Returns

None.

### MUFBVAR.mfbvar_funcs.varprior(nv, nlags, nex, hyp, premom)

### Parameters

nv
: numer of variables.

nlags
: number of lags.

nex
: number of exogenous variables inculding intercept.

hyp
: vector of hyperparameters.

premom
: pre-sample moments.

### Returns

None.

## Module contents

### *class* MUFBVAR.mufbvar_data(data, trans, frequencies)

Bases: `object`

Class to prepare the data that will be used in the MUFBVAR
…

### Parameters

data
: Data of each frequency stored in a pandas DataFrame, all stored in one list

trans
: A separate numpy array for each frequency all stored in a list. /n
  0: log is taken 
  1: divided by 100

frequencies
: “Y”, “Q”, “M”, “W”, “D”

### *class* MUFBVAR.multifrequency_var(nsim, nburn_perc, nlags, thining)

Bases: `object`

MUFBVAR class

### Parameters

nsim
: Number of simulations

nburn_perc
: Between 0 and 1, proportion of simulations to throw away as burn in.

nlags
: Number of lags in the highest frequency

thining
: To save only every nth draw

#### aggregate(frequency, reset_index=True)

Aggregates the Mean, Median and quantililes in the highest frequency to the desired frequency.

The Function ensures, that we start at the beginning of a Year or Quarter depending on the chosen frequency

#### Parameters

frequency
: The frequency to which the data should be aggregated to

reset_index
: Schould index be changed to period Index

#### fanchart(variables='all', save=True, name='Fancharts', show=True, agg=False, nhist=10)

Creates fan plots of the desired variables.

#### Parameters

variable
: variables for which the plot should be generated, all if it should be generated for all

save
: Whether the plots should be saved. The default is True.

name
: If the plots should be saved, path/name not including filetype. The default is None.

show
: Whether the plots should be shown. Default is True.

agg
: Whether the aggregated values should be shown

nhist
: number of historical periods that should be shown on the plot
  Default is 5

#### fit(mufbvar_data, hyp)

Estimates the model using the model parameter specified in the initialization.

And the data provided.

#### Parameters

mufbvar_data
: data in the form of a mufbvar_data class object

hyp
: list containing the hyperparameters
  <br/>
  1. overall tightness
  2. scaling down the variance for the coefficients of a distant lag
  3. number of observations used for obtaining the prior for the covariance matrix of error terms
  4. tuning parameter for coefficients for constant
  5. tuning parameter for the covariance between coefficients

#### forecast(H, conditionals=None)

Method to generate the forecasts in the highest frequency.

#### Parameters

H
: Forecast horizon in highest frequnecy

conditionals
: Conditional forecasts
  <br/>
  column names must be the variable names
  <br/>
  no index needed
  <br/>
  either values or np.nan

#### mean_plot(variables='all', save=True, name='Output', show=True)

Creates cumulative mean plots of the forecasts. If the model has converged the cumulative mean should be stable after burnin.

#### Parameters

variable
: variables for which the plot should be generated, all if it should be generated for all

save
: Whether the plots should be saved. The default is True.

name
: If the plots should be saved, path/name not including filetype. The default is None.

show
: Whether the plots should be shown. Default is True.

#### save(filename='mufbvar_model.pkl')

Saves the MFBVAR Object

#### Parameters

filename
: Path where to save the object. End must be .pkl

#### to_excel(filename, agg=False)

Writes the results to an excel

#### Parameters

agg
: Should the aggregated series be saved

filname
: file path.
