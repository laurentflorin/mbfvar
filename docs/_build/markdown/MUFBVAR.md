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

* **Parameters:**
  * **hyp** (*TYPE*) – DESCRIPTION.
  * **YY** (*TYPE*) – DESCRIPTION.
  * **spec** (*TYPE*) – DESCRIPTION.
  * **efficient** (*TYPE*) – DESCRIPTION.
* **Return type:**
  None.

### MUFBVAR.mfbvar_funcs.initialize(GAMMAs, GAMMAz, GAMMAc, GAMMAu, LAMBDAs, LAMBDAz, LAMBDAc, LAMBDAu, LAMBDAs_t, LAMBDAz_t, LAMBDAc_t, LAMBDAu_t, sig_qq, sig_mm, sig_qm, sig_mq, Zm, YDATA, init_mean, init_var, spec, Nm)

### MUFBVAR.mfbvar_funcs.mdd_(hyp, YY, spec)

* **Parameters:**
  * **hyp** (*TYPE*) – DESCRIPTION.
  * **YY** (*TYPE*) – DESCRIPTION.
  * **spec** (*TYPE*) – DESCRIPTION.
  * **efficient** (*TYPE*) – DESCRIPTION.
* **Return type:**
  None.

### MUFBVAR.mfbvar_funcs.prior_init(hyp, YY, spec)

* **Parameters:**
  * **hyp** (*TYPE*) – DESCRIPTION.
  * **YY** (*TYPE*) – DESCRIPTION.
  * **spec** (*TYPE*) – DESCRIPTION.
* **Returns:**
  * *Phi_tilde*
  * *sigma*

### MUFBVAR.mfbvar_funcs.prior_pdf(hyp, YY, spec, PHI, SIG)

* **Parameters:**
  * **hyp** (*TYPE*) – DESCRIPTION.
  * **YY** (*TYPE*) – DESCRIPTION.
  * **spec** (*TYPE*) – DESCRIPTION.
  * **PHI** (*TYPE*) – DESCRIPTION.
  * **SIG** (*TYPE*) – DESCRIPTION.
* **Return type:**
  None.

### MUFBVAR.mfbvar_funcs.varprior(nv, nlags, nex, hyp, premom)

* **Parameters:**
  * **nv** (*TYPE*) – numer of variables.
  * **nlags** (*TYPE*) – number of lags.
  * **nex** (*TYPE*) – number of exogenous variables inculding intercept.
  * **hyp** (*TYPE*) – vector of hyperparameters.
  * **premom** (*TYPE*) – pre-sample moments.
* **Return type:**
  None.

## MUFBVAR.mufbvar module

Created on Thu Nov 25 13:51:47 2021

@author: florinl

### *class* MUFBVAR.mufbvar.multifrequency_var(frequencies, H, nsim, nburn_perc, nlags, thining)

Bases: `object`

#### aggregate(frequency, reset_index=True)

Aggregates the Mean, Median and quantililes in the highest frequency to the desired frequency.

The Function ensures, that we start at the beginning of a Year or Quarter depending on the chosen frequency

* **Parameters:**
  * **frequency** (*str*) – The frequency to which the data should be aggregated to
  * **reset_index** (*boolean*) – Schould index be changed to period Index

#### fanchart(variables='all', save=True, name='Fancharts', show=True, agg=False, nhist=10)

Creates fan plots of the desired variables.

* **Parameters:**
  * **variable** (*list* *of* *strings*) – variables for which the plot should be generated, all if it should be generated for all
  * **save** (*boolean*) – Whether the plots should be saved. The default is True.
  * **name** (*string* *,* *optional*) – If the plots should be saved, path/name not including filetype. The default is None.
  * **show** (*boolean*) – Whether the plots should be shown. Default is True.
  * **agg** (*boolean*) – Whether the aggregated values should be shown
  * **nhist** (*int*) – number of historical periods that should be shown on the plot
    Default is 5

#### fit(io_data, io_conditionals, io_trans, hyp)

Estimates the model using the model parameter specified in the initialization.

Save data in excel with a sheet for data in each frequency. Name sheets after frequency: Y,Q,M,W,D

Save conditionals for the forecasts for each frequency in one excel. Name sheets after frequency: Y,Q,M,W,D

Save transformation in excel with a sheet for each frequency. Name sheets after frequency: Y,Q,M,W,D

* **Parameters:**
  * **io_data** (*str*) – path to excel containing the data
  * **io_conditionals** (*str*) – path to excel containing the conditional forecasts
  * **io_trans** (*str*) – path to excel containing the transformations
  * **hyp** (*list*) – list containing the hyperparameters

#### forecast()

Method to generate the forecasts in the highest frequency.

#### mean_plot(frequency, variables='all', save=True, name='Output', show=True)

Creates cumulative mean plots of the forecasts. If the model has converged the cumulative mean should be stable after burnin.

* **Parameters:**
  * **variable** (*list* *of* *strings*) – variables for which the plot should be generated, all if it should be generated for all
  * **save** (*boolean*) – Whether the plots should be saved. The default is True.
  * **name** (*string* *,* *optional*) – If the plots should be saved, path/name not including filetype. The default is None.
  * **show** (*boolean*) – Whether the plots should be shown. Default is True.

#### save(filename='mufbvar_model.pkl')

Saves the MFBVAR Object

* **Parameters:**
  **filename** (*str*) – Path where to save the object. End must be .pkl

#### to_excel(filename, agg=False)

Writes the results to an excel

* **Parameters:**
  * **agg** (*Boolean*) – Should the aggregated series be saved
  * **filname** (*Sting*) – file path.

## Module contents
