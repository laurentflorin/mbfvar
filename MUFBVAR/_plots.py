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


#for progressbar
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position = 0, leave=True) # this line does the magic

# for plots
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.pyplot import cm

#to save objects
import pickle
import copy



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
        
        idx, = np.where(self.varlist_list[-1] == variable)
        
        df = pd.DataFrame(self.Phip_list[-1][:,:,idx[0]]).expanding().mean()
        
        plt.style.use('seaborn-dark-palette')
        idx, = np.where(self.varlist_list[-1] == variable)

        fig, ax = plt.subplots(figsize = (14, 8.5))
        fig.patch.set_facecolor("#fdfdfd")
        ax.set_facecolor("#fdfdfd")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_lw(1)
        ax.spines["left"].set_color("black")
        ax.spines["left"].set_capstyle("butt")
        ax.spines["bottom"].set_lw(1.5)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_capstyle("butt")
        ax.grid(False)
        ax.grid(axis = "y", color="#A8BAC4", lw=0.4)
        ax.plot(df, linewidth=0.5)
        ax.axvline(x=self.nburn,  color='black', ls='--', lw=0.5)
        title = "Mean Plot of: " + variable
        plt.title(title)
        plt.xlabel('Draws')
        plt.ylabel('Value')
        plt.xlim(min(df.index), max(df.index))
        plt.gcf().set_dpi(320)
        
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
            
            plt.style.use('seaborn-dark-palette')
            
            idx, = np.where(self.varlist_list[-1] == variable)
            
            fig, ax = plt.subplots()
            x = self.YY_095_agg.iloc[-int(self.H/freq_ratio+nhist):,idx].index.to_timestamp()
            ax.fill_between(x, np.squeeze(np.array(self.YY_095_agg.iloc[-int(self.H/freq_ratio+nhist):,idx])), np.squeeze(np.array(self.YY_005_agg.iloc[-int(self.H/freq_ratio+nhist):,idx])), alpha = 0.5, color = "blue")
            ax.fill_between(x, np.squeeze(np.array(self.YY_084_agg.iloc[-int(self.H/freq_ratio+nhist):,idx])), np.squeeze(np.array(self.YY_016_agg.iloc[-int(self.H/freq_ratio+nhist):,idx])), alpha = 0.7, color = "blue")          
            ax.plot(x, np.squeeze(np.array(self.YY_mean_agg.iloc[-int(self.H/freq_ratio+nhist):,idx])), color = "red", linewidth = 0.5)
            
            ax.set_facecolor("#fdfdfd")
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_lw(1)
            ax.spines["left"].set_color("black")
            ax.spines["left"].set_capstyle("butt")
            ax.spines["bottom"].set_lw(1.5)
            ax.spines["bottom"].set_color("black")
            ax.spines["bottom"].set_capstyle("butt")
            ax.grid(False)
            ax.grid(axis = "y", color="#A8BAC4", lw=0.4)
            plt.gcf().set_dpi(320)
            
            ax.set_xticks(x)
            ax.set_xticklabels(self.YY_095_agg.iloc[-int(self.H/freq_ratio+nhist):,idx].index)
            plt.setp(ax.get_xticklabels(), rotation=40, horizontalalignment='right')

            #plt.axvline(x= forecast_start,  color='black', ls='--', lw=0.5)
            title = "Mean and 90% and 68% CI of: " + variable
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.xlim(min(self.YY_095_agg.iloc[-int(self.H/freq_ratio+nhist):,idx].index), max(self.YY_095_agg.iloc[-int(self.H/freq_ratio+nhist):,idx].index))
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
            
            plt.style.use('seaborn-dark-palette')
            
            idx, = np.where(self.varlist_list[-1] == variable)
            
            fig, ax = plt.subplots()
            ax.fill_between(self.YY_095_pd.iloc[-(self.H+nhist):,idx].index, np.squeeze(np.array(self.YY_095_pd.iloc[-(self.H+nhist):,idx])), np.squeeze(np.array(self.YY_005_pd.iloc[-(self.H+nhist):,idx])), alpha = 0.5, color = "blue")
            ax.fill_between(self.YY_095_pd.iloc[-(self.H+nhist):,idx].index, np.squeeze(np.array(self.YY_084_pd.iloc[-(self.H+nhist):,idx])), np.squeeze(np.array(self.YY_016_pd.iloc[-(self.H+nhist):,idx])), alpha = 0.7, color = "blue")          
            ax.plot(self.YY_mean_pd.iloc[-(self.H+nhist):,idx].index, np.squeeze(np.array(self.YY_mean_pd.iloc[-(self.H+nhist):,idx])), color = "red", linewidth = 0.5)
            fig.patch.set_facecolor("#fdfdfd")
            plt.axvline(x=forecast_start,  color='black', ls='--', lw=0.5)
            
            ax.set_facecolor("#fdfdfd")
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_lw(1)
            ax.spines["left"].set_color("black")
            ax.spines["left"].set_capstyle("butt")
            ax.spines["bottom"].set_lw(1.5)
            ax.spines["bottom"].set_color("black")
            ax.spines["bottom"].set_capstyle("butt")
            ax.grid(False)
            ax.grid(axis = "y", color="#A8BAC4", lw=0.4)
            plt.gcf().set_dpi(320)
            plt.setp(ax.get_xticklabels(), rotation=40, horizontalalignment='right')
            
            title = "Mean and 90% and 68% CI of: " + variable
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.xlim(min(self.YY_mean_pd.iloc[-(self.H+nhist):,idx].index), max(self.YY_mean_pd.iloc[-(self.H+nhist):,idx].index))
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
        
        plt.style.use('seaborn-dark-palette')
        
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
        
        ax.set_facecolor("#fdfdfd")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_lw(1)
        ax.spines["left"].set_color("black")
        ax.spines["left"].set_capstyle("butt")
        ax.spines["bottom"].set_lw(1.5)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_capstyle("butt")
        ax.grid(False)
        ax.grid(axis = "y", color="#A8BAC4", lw=0.4)
        plt.gcf().set_dpi(320)
        plt.xlim(min(temp.index), max(temp.index))
        
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
    
def compare_models(self, multifrquency_var_models, model_names, agg = True, variables = "all", save = True, name = "Comparison", show = True, nhist = 5):
    '''
    Creates a plot with the different scenarios
    
    
    Parameters
    ----------
    multifrquency_var_models : list
        list of multifrquency_var_models with forecasts
    model_names : list
        list of names of the models, including the current model at position 0
    agg : boolean
        should the aggregated time series be shown
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
    
    def week_of_month(dt):
        """ Returns the week of the month for the specified date. """
        first_day = dt.replace(day=1)
        dom = dt.day
        adjusted_dom = dom + first_day.weekday()
        out = int(adjusted_dom / 7) + 1
        if out == 5:
            out = 4
        return out
    
    plt.ioff()

    if isinstance(variables, str):
        if variables == "all":
            variables = self.varlist_list[-1]
        else:
            sys.exit("variables must be either a list of variables or all")
    
    if self.forecast_draws_list is None :
            sys.exit("Error: To generate comparison plots, generate forecasts first")
    
    
    if agg == True:
        
        freq_ratio = np.product(deque(itertools.islice(self.freq_ratio_list, 0, len(self.freq_ratio_list)-self.frequencies.index(self.agg_freq))))
        H = int(self.H/freq_ratio)
        
        if not hasattr(self, 'YY_mean_agg'):
                sys.exit("Error: To generate aggregated comparison plots, aggregate first")
                
            
        for variable in variables:
            
            plt.style.use('seaborn-dark-palette')
            
            color = iter(cm.tab10(np.linspace(0, 1, len(model_names))))
            fig, ax = plt.subplots(dpi= 360)
            
            c = next(color)
            current =  copy.deepcopy(self.YY_mean_agg.iloc[-(H+nhist):,:])
            ax.plot(current.index.to_timestamp(), current[variable], color = c, linewidth = 0.9, linestyle = "dashed", label = model_names[0])
            for i in range(1,len(model_names)):
                
                if multifrquency_var_models[i-1].forecast_draws_list is None:
                    sys.exit("Error: To generate comparison plots, generate forecasts of model" + model_names[i] + "first")
                if not hasattr(multifrquency_var_models[i-1], 'YY_mean_agg'):
                    sys.exit("Error: To generate aggregated comparison plots, aggregate model " + model_names[i] + " first")
                    
                temp = multifrquency_var_models[i-1].YY_mean_agg
                idx = current.index.intersection(temp.index, sort=False)
                
                c = next(color)
                ax.plot(temp.loc[idx].index.to_timestamp(), temp.loc[idx, variable], color = c, linewidth = 0.9, linestyle = "dashed", label = model_names[i])
                
            ax.plot(current.iloc[:-H,:].index.to_timestamp(), current.iloc[:-H][variable], color = "black", linewidth = 1)
            
            
            ax.set_facecolor("#fdfdfd")
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_lw(1)
            ax.spines["left"].set_color("black")
            ax.spines["left"].set_capstyle("butt")
            ax.spines["bottom"].set_lw(1.5)
            ax.spines["bottom"].set_color("black")
            ax.spines["bottom"].set_capstyle("butt")
            ax.grid(False)
            ax.grid(axis = "y", color="#A8BAC4", lw=0.4)
            plt.gcf().set_dpi(320)
            plt.xlim(min(current.index.to_timestamp()), max(current.index.to_timestamp()))
            
            ax.set_xticks(current.index.to_timestamp())
            ax.set_xticklabels(current.index)
            plt.setp(ax.get_xticklabels(), rotation=40, horizontalalignment='right')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.title("Comparison plot for " + variable)
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
        
    else:
        if not hasattr(self, 'YY_mean_pd'):
                sys.exit("Error: To generate aggregated comparison plots, aggregate first")
        
        diffs = []
        for i in range(len(multifrquency_var_models)):
            diffs.append(len(self.YY_mean_pd.index)- len(multifrquency_var_models[i].YY_mean_pd.index))
        diff = np.max(diffs)
        
        if nhist < diff:
            nhist = nhist + diff
        current = copy.deepcopy(self.YY_mean_pd.iloc[-(self.H+nhist):,:])
        
        if self.frequencies[-1] == "W":
            current['new_index'] = current.index.to_series().apply(lambda x: f"{x.year}-{x.month:02d}-W{week_of_month(x)}")
            current.set_index('new_index', inplace=True)
        
        for variable in variables:
            
            plt.style.use('seaborn-dark-palette')
            
            color = iter(cm.tab10(np.linspace(0, 1, len(model_names))))
            fig, ax = plt.subplots(dpi= 360)
            
            c = next(color)
            ax.plot(current.index, current[variable], color = c, linewidth = 0.9, linestyle = "dashed", label = model_names[0])
            for i in range(1,len(model_names)):
                
                if multifrquency_var_models[i-1].forecast_draws_list is None:
                    sys.exit("Error: To generate comparison plots, generate forecasts of model" + model_names[i] + "first")
                if not hasattr(multifrquency_var_models[i-1], 'YY_mean_pd'):
                    sys.exit("Error: To generate aggregated comparison plots, aggregate model " + model_names[i] + " first")
                    
                temp = copy.deepcopy(multifrquency_var_models[i-1].YY_mean_pd)
                if self.frequencies[-1] == "W":
                    temp['new_index'] = temp.index.to_series().apply(lambda x: f"{x.year}-{x.month:02d}-W{week_of_month(x)}")
                    temp.set_index('new_index', inplace=True)
                idx = temp.index[-self.H:]
                
                c = next(color)
                ax.plot(temp.loc[idx].index, temp.loc[idx, variable], color = c, linewidth = 0.9, linestyle = "dashed", label = model_names[i])
                
            ax.plot(current.iloc[:-self.H,:].index, current.iloc[:-self.H][variable], color = "black", linewidth = 1)
            
            ax.set_xticks(current.index[::5])
            ax.set_xticklabels(current.index[::5])
            plt.setp(ax.get_xticklabels(), rotation=40, horizontalalignment='right', fontsize=8)
            
            ax.set_facecolor("#fdfdfd")
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_lw(1)
            ax.spines["left"].set_color("black")
            ax.spines["left"].set_capstyle("butt")
            ax.spines["bottom"].set_lw(1.5)
            ax.spines["bottom"].set_color("black")
            ax.spines["bottom"].set_capstyle("butt")
            ax.grid(False)
            ax.grid(axis = "y", color="#A8BAC4", lw=0.4)
            plt.gcf().set_dpi(320)
            
            plt.title("Comparison plot for " + variable)
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
    