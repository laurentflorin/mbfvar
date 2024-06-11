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

import matplotlib.pyplot as plt

#for progressbar
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position = 0, leave=True) # this line does the magic

# for plots

import matplotlib.backends.backend_pdf

import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default='browser'
import plotly.express as px

#to save objects
import pickle
import copy


def save(self, filename = "mufbvar_model"):
        
        '''
        Saves the MFBVAR Object
        
        Parameters
        ----------
        filename : str
            Path where to save the object. End must be .pkl
        
        '''
        with open(filename + ".pkl", 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
            
            

def to_excel(self, filename, agg = False):
    
    '''
    Writes the results to an excel
    
    Parameters
    ----------
    agg : Boolean
        Should the aggregated series be saved
    filname : Sting
        file path.

    '''
    
    if not hasattr(self, 'forecast_draws_list'):
        
        if len(self.index_list[-1]) == self.lstate_list[-1][0,:,:].shape[1]:
        
            index = copy.deepcopy(self.index_list[-1])
        
        else:
            index = range(self.lstate_list[-1][0,:,:].shape[1])
        
        #mean
        YYnow_m = np.mean(self.YYactsim_list[-1][self.nburn:,1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]], axis = 0) # actual/nowcast monthlies
        if YYnow_m.size:
            YYnow_m[:, (self.select_m_list[-1] == 1)] = 100 * YYnow_m[:, (self.select_m_list[-1] == 1)]
            YYnow_m[:, (self.select_m_list[-1] == 0)] = np.exp(YYnow_m[:,(self.select_m_list[-1] == 0)])
        
        lstate_m = np.mean(self.lstate_list[-1][self.nburn:,:,:], axis = 0).T # hf obs for lf vars
        lstate_m[:, (self.select_q[-1] == 1)] = 100 * lstate_m[:, (self.select_q[-1] == 1)]
        lstate_m[:, (self.select_q[-1] == 0)] = np.exp(lstate_m[:, (self.select_q[-1]== 0)])
        
        YMh_list = copy.deepcopy(self.YMh_list)
        
        YMh_len_correction = int(YMh_list[-1].shape[0] - lstate_m[:-(self.freq_ratio_list[-1]),:].shape[0])
        
        if YMh_list[-1].size:
            YMh_list[-1][:, (self.select_m_list[-1] == 1)] = 100 * YMh_list[-1][:, (self.select_m_list[-1] == 1)]
            YMh_list[-1][:, (self.select_m_list[-1] == 0)] =  np.exp(YMh_list[-1][:, (self.select_m_list[-1] == 0)])
        
        YY_mean = np.vstack((np.hstack((YMh_list[-1][YMh_len_correction:,:], lstate_m[:-(self.freq_ratio_list[-1]),:])), np.hstack((YYnow_m, lstate_m[-self.freq_ratio_list[-1]:,:]))))
        
        #median
        
        YYnow_med = np.median(self.YYactsim_list[-1][self.nburn:,1:(self.freq_ratio+1),:self.Nm_list[-1]], axis = 0) # actual/nowcast monthlies
        
        if YYnow_med.size:
            YYnow_med[:, (self.select_m_list[-1] == 1)] = 100 * YYnow_med[:, (self.select_m_list[-1] == 1)]
            YYnow_med[:, (self.select_m_list[-1] == 0)] = np.exp(YYnow_med[:, (self.select_m_list[-1] == 0)])
        
        lstate_med = np.median(self.lstate_list[-1][self.nburn:,:,:], axis = 0).T # hf obs for lf vars
        lstate_med[:, (self.select_q[-1] == 1)] = 100 * lstate_med[:, (self.select_q[-1] == 1)]
        lstate_med[:, (self.select_q[-1] == 0)] = np.exp(lstate_med[:, (self.select_q[-1]== 0)])
        
        if YMh_list[-1].size:
            YY_med = np.vstack((np.hstack((YMh_list[-1][YMh_len_correction:,:], lstate_med[:-self.freq_ratio_list[-1],:])), np.hstack((YYnow_med, lstate_m[-self.freq_ratio_list[-1]:,:]))))
        else:
            YY_med = lstate_med 
            
        # safe uncertainty
        # 95%
        YYnow_095 = np.quantile(self.YYactsim_list[-1][self.nburn:,1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]], q = 0.95, axis = 0) # actual/nowcast monthlies
        if YYnow_095.size:
            YYnow_095[:, (self.select_m_list[-1] == 1)] = 100 * YYnow_095[:, (self.select_m_list[-1] == 1)]
            YYnow_095[:, (self.select_m_list[-1] == 0)] = np.exp(YYnow_095[:, (self.select_m_list[-1] == 0)])
        
        lstate_095 = np.quantile(self.lstate_list[-1][self.nburn:,:,:], q = 0.95, axis = 0).T # hf obs for lf vars
        lstate_095[:, (self.select_q[-1] == 1)] = 100 * lstate_095[:, (self.select_q[-1] == 1)]
        lstate_095[:, (self.select_q[-1] == 0)] = np.exp(lstate_095[:, (self.select_q[-1] == 0)])
        
        YMna = np.full(YMh_list[-1].shape, np.nan)
        
        if YMna.size:
            YY_095 = np.vstack((np.hstack((YMna[YMh_len_correction:,:],lstate_095[:-self.freq_ratio_list[-1],:])), np.hstack((YYnow_095, lstate_095[-self.freq_ratio_list[-1]:,:]))))
        else:
            YY_095 = lstate_095
        
        # 84%
        
        YYnow_084 = np.quantile(self.YYactsim_list[-1][self.nburn:,1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]], q = 0.84, axis = 0) # actual/nowcast monthlies
        if YYnow_084.size:
            YYnow_084[:, (self.select_m_list[-1] == 1)] = 100 * YYnow_084[:, (self.select_m_list[-1] == 1)]
            YYnow_084[:, (self.select_m_list[-1] == 0)] = np.exp(YYnow_084[:, (self.select_m_list[-1] == 0)])
        
        lstate_084 = np.quantile(self.lstate_list[-1][self.nburn:,:,:], q = 0.84, axis = 0).T # hf obs for lf vars
        lstate_084[:, (self.select_q[-1] == 1)] = 100 * lstate_084[:, (self.select_q[-1] == 1)]
        lstate_084[:, (self.select_q[-1] == 0)] = np.exp(lstate_084[:, (self.select_q[-1] == 0)])
        
        YMna = np.full(YMh_list[-1].shape, np.nan)
        
        if YMna.size:
            YY_084 = np.vstack((np.hstack((YMna[YMh_len_correction:,:],lstate_084[:-self.freq_ratio_list[-1],:])), np.hstack((YYnow_084, lstate_084[-self.freq_ratio_list[-1]:,:]))))
        else:
            YY_084 = lstate_084
            
        # 16%
        
        YYnow_016 = np.quantile(self.YYactsim_list[-1][self.nburn:,1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]], q = 0.16, axis = 0) # actual/nowcast monthlies
        if YYnow_016.size:
            YYnow_016[:, (self.select_m_list[-1] == 1)] = 100 * YYnow_016[:, (self.select_m_list[-1] == 1)]
            YYnow_016[:, (self.select_m_list[-1] == 0)] = np.exp(YYnow_016[:, (self.select_m_list[-1] == 0)])
        
        lstate_016 = np.quantile(self.lstate_list[-1][self.nburn:,:,:], q = 0.16, axis = 0).T # hf obs for lf vars
        lstate_016[:, (self.select_q[-1] == 1)] = 100 * lstate_016[:, (self.select_q[-1] == 1)]
        lstate_016[:, (self.select_q[-1] == 0)] = np.exp(lstate_016[:, (self.select_q[-1] == 0)])
        
        YMna = np.full(YMh_list[-1].shape, np.nan)
        
        if YMna.size:
            YY_016 = np.vstack((np.hstack((YMna[YMh_len_correction:,:],lstate_016[:-self.freq_ratio_list[-1],:])), np.hstack((YYnow_016, lstate_016[-self.freq_ratio_list[-1]:,:]))))
        else:
            YY_016 = lstate_016,YYftr_016
        
        
        # 5%    
        
        YYnow_005 = np.quantile(self.YYactsim_list[-1][self.nburn:,1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]], q = 0.05, axis = 0) # actual/nowcast monthlies
        
        if YYnow_005.size:   
            YYnow_005[:, (self.select_m_list[-1] == 1)] = 100 * YYnow_005[:, (self.select_m_list[-1] == 1)]
            YYnow_005[:, (self.select_m_list[-1] == 0)] = np.exp(YYnow_005[:, (self.select_m_list[-1] == 0)])
        
        lstate_005 = np.quantile(self.lstate_list[-1][self.nburn:,:,:], q = 0.05, axis = 0).T # hf obs for lf vars
        lstate_005[:, (self.select_q[-1] == 1)] = 100 * lstate_005[:, (self.select_q[-1] == 1)]
        lstate_005[:, (self.select_q[-1] == 0)] = np.exp(lstate_005[:, (self.select_q[-1] == 0)])
        
        if YMna.size:
            YY_005 = np.vstack((np.hstack((YMna[YMh_len_correction:,:], lstate_005[:-self.freq_ratio_list[-1],:])), np.hstack((YYnow_005, lstate_005[-self.freq_ratio_list[-1]:,:]))))
        else:
            YY_005 = lstate_005,YYftr_005
        
        index_start = self.YMX_list[-1].index[self.YMX_list[-1][self.YMX_list[-1].columns[0]] == self.YMh_list[-1][YMh_len_correction:,:][0,0]]
        
        YY_mean_pd = pd.DataFrame(YY_mean, columns = self.varlist_list[-1])
        YY_mean_pd.index = range(index_start[0], YY_mean_pd.shape[0]+index_start[0])
        
        
        YY_median_pd = pd.DataFrame(YY_med, columns = self.varlist_list[-1])
        YY_median_pd.index = range(index_start[0], YY_mean_pd.shape[0]+index_start[0])
        
        YY_095_pd = pd.DataFrame(YY_095, columns = self.varlist_list[-1])
        YY_095_pd.index = range(index_start[0], YY_mean_pd.shape[0]+index_start[0])
        
        YY_005_pd = pd.DataFrame(YY_005, columns = self.varlist_list[-1])
        YY_005_pd.index = range(index_start[0], YY_mean_pd.shape[0]+index_start[0])
        
        YY_084_pd = pd.DataFrame(YY_084, columns = self.varlist_list[-1])
        YY_084_pd.index = range(index_start[0], YY_mean_pd.shape[0]+index_start[0])
        
        YY_016_pd = pd.DataFrame(YY_016, columns = self.varlist_list[-1])
        YY_016_pd.index = range(index_start[0], YY_mean_pd.shape[0]+index_start[0])
        
            
        with pd.ExcelWriter(filename, engine = "xlsxwriter", datetime_format='yyyy-mm-dd') as writer:
            #writer = pd.ExcelWriter("sim_data.xlsx", engine="xlsxwriter")
                YY_mean_pd.to_excel(writer, sheet_name = "mean")
                YY_median_pd.to_excel(writer, sheet_name = "median")
                YY_095_pd.to_excel(writer, sheet_name = "95_quantile")
                YY_005_pd.to_excel(writer, sheet_name = "5_quantile")
                YY_084_pd.to_excel(writer, sheet_name = "84_quantile")
                YY_016_pd.to_excel(writer, sheet_name = "16_quantile")

    else:  
                
        if agg == True and not hasattr(self, 'YY_095_agg'):
            sys.exit("Aggregate first")
        
        if agg == False:
            
            with pd.ExcelWriter(filename, engine = "xlsxwriter", datetime_format='yyyy-mm-dd') as writer:
            #writer = pd.ExcelWriter("sim_data.xlsx", engine="xlsxwriter")
                self.YY_mean_pd.to_excel(writer, sheet_name = "mean")
                self.YY_median_pd.to_excel(writer, sheet_name = "median")
                self.YY_095_pd.to_excel(writer, sheet_name = "95_quantile")
                self.YY_005_pd.to_excel(writer, sheet_name = "5_quantile")
                self.YY_084_pd.to_excel(writer, sheet_name = "84_quantile")
                self.YY_016_pd.to_excel(writer, sheet_name = "16_quantile")
            #writer.close()
        else:
            if type(self.YY_mean_agg.index) == pd.DatetimeIndex:
                with pd.ExcelWriter(filename, engine = "xlsxwriter", datetime_format='yyyy-mm-dd') as writer:
                #writer = pd.ExcelWriter("sim_data.xlsx", engine="xlsxwriter")
                    self.YY_mean_agg.to_excel(writer, sheet_name = "mean")
                    self.YY_median_agg.to_excel(writer, sheet_name = "median")
                    self.YY_095_agg.to_excel(writer, sheet_name = "95_quantile")
                    self.YY_005_agg.to_excel(writer, sheet_name = "5_quantile")
                    self.YY_084_agg.to_excel(writer, sheet_name = "84_quantile")
                    self.YY_016_agg.to_excel(writer, sheet_name = "16_quantile")
            else:
                if self.agg_freq == "Q":
                    with pd.ExcelWriter(filename, engine = "xlsxwriter") as writer:
                        self.YY_mean_agg.assign(Index= self.YY_mean_agg.index.strftime('%YQ%q')).set_index('Index').to_excel(writer,  sheet_name = "mean")
                        self.YY_median_agg.assign(Index= self.YY_median_agg.index.strftime('%YQ%q')).set_index('Index').to_excel(writer,  sheet_name = "median")
                        self.YY_095_agg.assign(Index= self.YY_095_agg.index.strftime('%YQ%q')).set_index('Index').to_excel(writer,  sheet_name = "95_quantile")
                        self.YY_005_agg.assign(Index= self.YY_005_agg.index.strftime('%YQ%q')).set_index('Index').to_excel(writer,  sheet_name = "5_quantile")
                        self.YY_084_agg.assign(Index= self.YY_084_agg.index.strftime('%YQ%q')).set_index('Index').to_excel(writer,  sheet_name = "84_quantile")
                        self.YY_016_agg.assign(Index= self.YY_016_agg.index.strftime('%YQ%q')).set_index('Index').to_excel(writer,  sheet_name = "16_quantile")
                if self.agg_freq == "Y":
                    with pd.ExcelWriter(filename, engine = "xlsxwriter") as writer:
                        self.YY_mean_agg.assign(Index= self.YY_mean_agg.index.strftime('%Y')).set_index('Index').to_excel(writer,  sheet_name = "mean")
                        self.YY_median_agg.assign(Index= self.YY_median_agg.index.strftime('%Y')).set_index('Index').to_excel(writer,  sheet_name = "median")
                        self.YY_095_agg.assign(Index= self.YY_095_agg.index.strftime('%Y')).set_index('Index').to_excel(writer,  sheet_name = "95_quantile")
                        self.YY_005_agg.assign(Index= self.YY_005_agg.index.strftime('%Y')).set_index('Index').to_excel(writer,  sheet_name = "5_quantile")
                        self.YY_084_agg.assign(Index= self.YY_084_agg.index.strftime('%Y')).set_index('Index').to_excel(writer,  sheet_name = "84_quantile")
                        self.YY_016_agg.assign(Index= self.YY_016_agg.index.strftime('%Y')).set_index('Index').to_excel(writer,  sheet_name = "16_quantile")
                    
    