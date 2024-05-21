import os
import sys

import numpy as np
import math

from collections import deque

from scipy.stats import invwishart
import pandas as pd
from scipy.stats import multivariate_normal
from datetime import datetime
from pandas.tseries.offsets import Week , MonthBegin, QuarterBegin, Day

import itertools

from .mfbvar_funcs import calc_yyact
#plotting
import matplotlib.pyplot as plt

#for progressbar
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position = 0, leave=True) # this line does the magic

# for plots

import matplotlib.backends.backend_pdf
from matplotlib.pyplot import cm

#to save objects
import pickle
import copy



#from MUFBVAR.pseudo_inverse.pseudo_inverse import calculate_pseudo_inverse
from .cholcov.cholcov_module import cholcovOrEigendecomp
from .inverse.matrix_inversion import invert_matrix


def mean_plot(self, variables = "all", save = True, name = "Output", show = True):
    
    '''
    Creates cumulative mean plots of the forecasts. If the model has converged the cumulative mean should be stable after burnin.
    
    Parameters
    ----------
    variable : list of strings
        variables for which the plot should be generated, all if it should be generated for all
    save : boolean
        Whether the plots should be saved. The default is True.
    name : string, optional
        If the plots should be saved, path/name not including filetype. The default is None.
    show : boolean
        Whether the plots should be shown. Default is True.
        
    '''
    plt.ioff()
    
    
    if self.forecast_draws_list is None :
            sys.exit("Error: To generate traceplots, generate forecasts first")
    
    if isinstance(variables, str):
        if variables == "all":
            variables = self.varlist_list[-1]
        else:
            sys.exit("variables must be either a list of variables or all")
        
        
    check = set(variables)-set(self.varlist_list[-1])        
    if check and not variables == "all":
        sys.exit(print(check, " not in " , self.varlist_list[-1]))
    
    if save == True:
        pdf = matplotlib.backends.backend_pdf.PdfPages(name + ".pdf")
        
    for variable in variables:
        
        idx, = np.where(self.varlist_list[-1] == variables)
        lst = list(self.forecast_draws_list[-1].T)

        fig = plt.figure()           
        df = pd.DataFrame(lst[idx[0]].T).expanding().mean()
        plt.plot(df, linewidth=0.5)
        plt.axvline(x=self.nburn,  color='black', ls='--', lw=0.5)
        title = "Mean Plot of: " + variable
        plt.title(title)
        plt.xlabel('Draws')
        plt.ylabel('Value')
        if save == True:
            pdf.savefig( fig )
        if show == True:
            plt.show()
    if save == True:
        pdf.close() 
    plt.close("all")  
    
    

def fanchart(self, variables = "all", save = True, name = "Fancharts", show = True, agg = False, nhist = 10):
    
    '''
    Creates fan plots of the desired variables.
    
    
    Parameters
    ----------
    variable : list of strings
        variables for which the plot should be generated, all if it should be generated for all
    save : boolean
        Whether the plots should be saved. The default is True.
    name : string, optional
        If the plots should be saved, path/name not including filetype. The default is None.
    show : boolean
        Whether the plots should be shown. Default is True.
    agg : boolean
        Whether the aggregated values should be shown
    nhist : int
        number of historical periods that should be shown on the plot
        Default is 5

    '''
    
    plt.ioff()
    
    if self.forecast_draws_list is None :
            sys.exit("Error: To generate traceplots, generate forecasts first")
    
    if isinstance(variables, str):
        if variables == "all":
            variables = self.varlist_list[-1]
        else:
            sys.exit("variables must be either a list of variables or all")
    
    if agg == True and not hasattr(self, 'YY_095_agg'):
        sys.exit("Aggregate first")
        
        
    check = set(variables)-set(self.varlist_list[-1])        
    if check and not variables == "all":
        sys.exit(print(check, " not in " , self.varlist_list[-1]))
    
    if agg == True:
        
        # Set the frequency ratio        
        freq_lf = self.agg_freq
        freq_hf = self.frequencies[-1]
        if freq_hf == "Q" and freq_lf == "Y":
            freq_ratio = 4
        elif freq_hf == "M" and freq_lf == "Y":
            freq_ratio = 12
        elif freq_hf == "W" and freq_lf == "Y":
            freq_ratio = 48
        elif freq_hf == "D" and freq_lf == "Y":
            freq_ratio = 260
        elif freq_hf == "M" and freq_lf == "Q":
            freq_ratio = 3
        elif freq_hf == "W" and freq_lf == "Q":
            freq_ratio = 12
        elif freq_hf == "W" and freq_lf == "M":
            freq_ratio = 4
        elif freq_hf == "D" and freq_lf == "W":
            freq_ratio = 5
        elif freq_hf == "D" and freq_lf == "M":
            freq_ratio = 20
        
        forecast_start = self.YY_mean_agg.iloc[-int(self.H/freq_ratio),:].name
        
            
        if save == True:
            pdf = matplotlib.backends.backend_pdf.PdfPages(name + ".pdf")
            
        for variable in variables:
            
            idx, = np.where(self.varlist_list[-1] == variable)
            
            fig, ax = plt.subplots()
            x = self.YY_095_agg.iloc[-int(self.H/freq_ratio+nhist):,idx].index.to_timestamp()
            ax.fill_between(x, np.squeeze(np.array(self.YY_095_agg.iloc[-int(self.H/freq_ratio+nhist):,idx])), np.squeeze(np.array(self.YY_005_agg.iloc[-int(self.H/freq_ratio+nhist):,idx])), alpha = 0.5, color = "blue")
            ax.fill_between(x, np.squeeze(np.array(self.YY_084_agg.iloc[-int(self.H/freq_ratio+nhist):,idx])), np.squeeze(np.array(self.YY_016_agg.iloc[-int(self.H/freq_ratio+nhist):,idx])), alpha = 0.7, color = "blue")          
            ax.plot(x, np.squeeze(np.array(self.YY_mean_agg.iloc[-int(self.H/freq_ratio+nhist):,idx])), color = "red", linewidth = 0.5)
            
            ax.set_xticks(x)
            ax.set_xticklabels(self.YY_095_agg.iloc[-int(self.H/freq_ratio+nhist):,idx].index)
            plt.setp(ax.get_xticklabels(), rotation=40, horizontalalignment='right')

            #plt.axvline(x= forecast_start,  color='black', ls='--', lw=0.5)
            title = "Mean and 90% and 68% CI of: " + variable
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Value')
            if save == True:
                pdf.savefig( fig )
            if show == True:
                plt.show()
        if save == True:
            pdf.close() 

    else:
        forecast_start = self.YY_mean_pd.iloc[-self.H,:].name
        if save == True:
            pdf = matplotlib.backends.backend_pdf.PdfPages(name + ".pdf")
            
        for variable in variables:
            
            idx, = np.where(self.varlist_list[-1] == variable)
            
            fig, ax = plt.subplots()
            ax.fill_between(self.YY_095_pd.iloc[-(self.H+nhist):,idx].index, np.squeeze(np.array(self.YY_095_pd.iloc[-(self.H+nhist):,idx])), np.squeeze(np.array(self.YY_005_pd.iloc[-(self.H+nhist):,idx])), alpha = 0.5, color = "blue")
            ax.fill_between(self.YY_095_pd.iloc[-(self.H+nhist):,idx].index, np.squeeze(np.array(self.YY_084_pd.iloc[-(self.H+nhist):,idx])), np.squeeze(np.array(self.YY_016_pd.iloc[-(self.H+nhist):,idx])), alpha = 0.7, color = "blue")          
            ax.plot(self.YY_mean_pd.iloc[-(self.H+nhist):,idx].index, np.squeeze(np.array(self.YY_mean_pd.iloc[-(self.H+nhist):,idx])), color = "red", linewidth = 0.5)
            plt.axvline(x=forecast_start,  color='black', ls='--', lw=0.5)
            title = "Mean and 90% and 68% CI of: " + variable
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Value')
            if save == True:
                pdf.savefig( fig )
            if show == True:
                plt.show()
        if save == True:
            pdf.close()
    plt.close("all")
    

def scenario_plot(self, scenario_dict, variables = "all", save = True, name = "Scenario", show = True, nhist = 5):

    '''
    Creates a plot with the different scenarios
    
    
    Parameters
    ----------
    scenario_dict : dict
        output of self.scenario_forecast
    variable : list of strings
        variables for which the plot should be generated, all if it should be generated for all
    save : boolean
        Whether the plots should be saved. The default is True.
    name : string, optional
        If the plots should be saved, path/name not including filetype. The default is None.
    show : boolean
        Whether the plots should be shown. Default is True.
    agg : boolean
        Whether the aggregated values should be shown
    nhist : int
        number of historical periods that should be shown on the plot
        Default is 5

    '''
    #first get the history data. this is identical in all the scenarios
    names = list(scenario_dict.keys())
    
    hist = pd.merge(scenario_dict[names[0]], scenario_dict[names[1]], how = "inner")
    hist.index = scenario_dict[names[0]].index[:len(hist)]
    hist = hist.iloc[-nhist:,:]
    
    plt.ioff()
    
    if isinstance(variables, str):
        if variables == "all":
            variables = self.varlist_list[-1]
        else:
            sys.exit("variables must be either a list of variables or all")
        
        
    check = set(variables)-set(self.varlist_list[-1])        
    if check and not variables == "all":
        sys.exit(print(check, " not in " , self.varlist_list[-1]))
    
    if save == True:
        pdf = matplotlib.backends.backend_pdf.PdfPages(name + ".pdf")
    
    
    
    for variable in variables:
        color = iter(cm.tab10(np.linspace(0, 1, len(scenario_dict))))
        fig, ax = plt.subplots(dpi= 360)
        
        for i in range(len(scenario_dict)):
            c = next(color)
            temp = scenario_dict[names[i]].loc[hist.index[0]:,:]
            ax.plot(temp.index.to_timestamp(), temp[variable], color = c, linewidth = 0.9, linestyle = "dashed", label = names[i])
            ax.plot(hist.index.to_timestamp(), hist[variable], color = "black", linewidth = 1)
        
        ax.set_xticks(temp.index.to_timestamp())
        ax.set_xticklabels(temp.index)
        plt.setp(ax.get_xticklabels(), rotation=40, horizontalalignment='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.title("Scenario plot for " + variable)
        plt.grid(axis='both', alpha=.1)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        
        if save == True:
            pdf.savefig( fig )
        if show == True:
            plt.show()
        if save == True:
            pdf.close()
    plt.close("all")
