# MUFBVAR package

## Subpackages

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

## Module contents

### *class* MUFBVAR.mufbvar_data(data, trans, frequencies)

Bases: `object`

Class to prepare the data that will be used in the MUFBVAR
…

* **Parameters:**
  * **data** (*list* *of* *pandas DataFrames*) – Data of each frequency stored in a pandas DataFrame, all stored in one list
  * **trans** (*list* *of* *numpy arrays*) – A separate numpy array for each frequency all stored in a list. /n
    0: log is taken
    1: divided by 100
  * **frequencies** (*List* *of* *the frequencies* *of* *the data* *,* *in order lowest to highest*) – “Y”, “Q”, “M”, “W”, “D”

### *class* MUFBVAR.multifrequency_var(nsim, nburn_perc, nlags, thining)

Bases: `object`

MUFBVAR class

* **Parameters:**
  * **nsim** (*int*) – Number of simulations
  * **nburn_perc** (*numeric*) – Between 0 and 1, proportion of simulations to throw away as burn in.
  * **nlags** (*int*) – Number of lags in the highest frequency
  * **thining** (*int*) – To save only every nth draw

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

#### fit(mufbvar_data, hyp)

Estimates the model using the model parameter specified in the initialization.

And the data provided.

* **Parameters:**
  * **mufbvar_data** (*mufbvar_data class object*) – data in the form of a mufbvar_data class object
  * **hyp** (*list* *of* *list*) – 

    list containing list of the hyperparameters for each frequency step
    1. overall tightness
    2. scaling down the variance for the coefficients of a distant lag
    3. number of observations used for obtaining the prior for the covariance matrix of error terms
    4. tuning parameter for coefficients for constant
    5. tuning parameter for the covariance between coefficients

#### forecast(H, conditionals=None)

Method to generate the forecasts in the highest frequency.

* **Parameters:**
  * **H** (*int*) – Forecast horizon in highest frequnecy
  * **conditionals** (*pandas DataFrame* *or* *None*) – 

    Conditional forecasts

    column names must be the variable names

    no index needed

    either values or np.nan

#### mean_plot(variables='all', save=True, name='Output', show=True)

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

#### update_hyperparameters(mufbvar_data, pbounds, init_points, n_iter, nsim, save=False, name='hyp.txt')

This method uses bayesian optimization to find the hyperparameters with the highest mdd

lambda 1: overall tightness

lambda 2:  scaling down the variance for the coefficients of a distant lag

lambda 3:  number of observations used for obtaining the prior for the covariance matrix of error terms, fixed to 1

lambda 4: . tuning parameter for coefficients for constant

lambda 5:  tuning parameter for the covariance between coefficients

* **Parameters:**
  * **mufbvar_data** (*mufbvar_data class object*) – data in the form of a mufbvar_data class object
  * **pbound** (*dict*) – 

    boundries for each hyperparameter

    two frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1
    three frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2
    four frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2, lambda1_3, lambda2_3, lambda4_3, lambda5_3
  * **init_points** (*int*) – How many steps of random exploration you want to perform
  * **n_iter** (*int*) – How many steps of bayesian optimization you want to perform
  * **nsim** (*int*) – number of draws in each MUFBVAR estimation
  * **save** (*boolean*) – True if you want to save the hyperparameters as a txt
  * **name** (*str*) – path where you want to save the hyperparameters
