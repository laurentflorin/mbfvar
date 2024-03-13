# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:51:47 2021

@author: florinl
"""

# Wichtig: äussere schleife muss ziehungsschleife sien. innere schleife muss frequency schleife sein
#%%
import os
import sys

import numpy as np
import math

from collections import deque

from scipy.stats import invwishart
import pandas as pd
from scipy.stats import multivariate_normal
from datetime import datetime

import itertools

from .mfbvar_funcs import mdd_


#plotting
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

import fanchart

#to save objects
import pickle



#from MUFBVAR.pseudo_inverse.pseudo_inverse import calculate_pseudo_inverse
from .cholcov.cholcov_module import cholcovOrEigendecomp
from .inverse.matrix_inversion import invert_matrix

#%%

class multifrequency_var:
    
    def __init__(self, frequencies, H, nsim, nburn_perc, nlags, thining):
        '''
        

        Parameters
        ----------
        frequencies : List of the frequencies of the data, in order lowest to highest 
            "Y", "Q", "M", "W", "D"
        H : Numeric
            Forecast Horizon in the highest frequency
        nsim : Numeric
            Number of simulations
        nburn_perc : numeric
            Between 0 and 1, proportion of simulations to throw away as burn in.
        nlags : numeric
            Number of lags in the highest frequency
        forecast_frequencies : 
            "all" or "highest"
        thining : Numeric
            To save only every nth draw

        Returns
        -------
        None.

        '''
        self.frequencies = frequencies
        self.H = H
        self.nsim = nsim
        self.nburn_perc = nburn_perc
        self.nlags = nlags
        self.thining = thining
        
        
        
    
    
    def fit(self, io_data, io_conditionals, io_trans, hyp):
        '''
        Save data in excel with a sheet for data in each frequency. Name sheets
        after frequency: Y,Q,M,W,D
        Save conditionals for the forecasts for each frequency in one excel. Name sheets
        after frequency: Y,Q,M,W,D
        Save transformation in excel with a sheet for each frequency. Name sheets
        after frequency: Y,Q,M,W,D
        '''
        
        self.nex = 1
        self.hyp = hyp
        
        # We start by initializing all the matrices for all the frequencies
        # Then we performe one draw per MFBVAR (at each frequency), inclusive
        # the nowcast
        
        nburn = round((self.nburn_perc)*self.nsim)
        self.nburn = nburn
        
        
    
            
        
        # Load Data
        # create a list with the higher frequency data (frequency 2 to K)
        # these are the higher frequency data used in each MFBVAR
        
        YMX_list = deque()
        YM0_list = deque()
        YMC_list = deque()
        select_m_list = deque()
        vars_m_list = deque()
        YMh_list = deque()
        exc_list = deque()
        
        
        for freq in range(1,len(self.frequencies)):
            freq = self.frequencies[freq]
            YMX_temp = pd.read_excel(io_data, sheet_name = freq, index_col = 0)
            YMX_list.append(YMX_temp)
            YMC_list.append(pd.read_excel(io_conditionals, sheet_name = freq, index_col = 0).to_numpy())
            exc_list.append(YMC_list[-1] < np.exp(99))
            YM0_list.append(YMX_temp.to_numpy())
            select_m_list.append(pd.read_excel(io_trans, sheet_name = freq).to_numpy())
            vars_m_list.append(YMX_temp.columns[:])
            YMh_list.append(YMX_temp.to_numpy())
            
        del YMX_temp 
        
        input_data = YMX_list.copy()
        self.input_data = input_data
        
        # Data in the highest frequency: Data we want to temporally disaggregate and/or
        # forecast at the higher frequencies
        # We create a list but only one entry is generated here, the rest are generated later
        
        YQX_list = deque()
        YQ0_list = deque()
        select_q = deque()
        
        YQX_list.append(pd.read_excel(io_data, sheet_name = self.frequencies[0], index_col = 0))
        YQ0_list.append(YQX_list[-1].to_numpy())
        select_q.append(pd.read_excel(io_trans, sheet_name = self.frequencies[0]).to_numpy())
        vars_q = YQX_list[0].columns[:]
        
        input_data_Q =  YQX_list[0].copy()
        self.input_data_Q = input_data_Q
        
        varlist_list = deque()
        select_list = deque()
        select_c_list =deque()
        
        Nm_list = deque()
        nv_list = deque()
        Nq_list = deque()
        Nq_list.append(YQX_list[0].shape[1])
        for i in range(len(YMX_list)-1):
            Nq_list.append(YQX_list[0].shape[1] + YMX_list[i].shape[1]) 
            #Nq_list.append(np.shape(YMX_list[i])[1])
            if len(select_m_list[i]) == 0:
                select_q.append( select_q[0])
            else:
                select_q.append(np.hstack((select_m_list[i], select_q[0])))
        
        

        
        for i in range(len(YMX_list)):
            if i > 0:
                new_list = [item for item in list(itertools.islice(select_m_list, 0, i+1)) if len(item) > 0]
                select_list.append(np.hstack((np.hstack(new_list), select_q[0])))
                rev_vars_m = list(itertools.islice(vars_m_list, 0, i+1))
                rev_vars_m.reverse()
                varlist_list.append(np.squeeze(np.hstack((np.hstack(rev_vars_m), vars_q)))) 
            else:
                if select_m_list[i].size:
                    select_list.append(np.hstack((select_m_list[i], select_q[0])))
                    varlist_list.append(np.hstack((vars_m_list[i], vars_q)))
                else:
                    select_list.append(select_q[0])
                    varlist_list.append(np.squeeze(vars_q))
            select_c_list.append(-1*(select_list[i]-np.ones(np.shape(select_list[i])[1])))
            Nm_list.append(int(np.shape(YM0_list[i])[1]))
            nv_list.append(int(Nm_list[i] + Nq_list[i]))
        
        select_list_sep = list(select_q)
        select_list_sep.extend(select_m_list)
        
        freq_ratio_list = deque()
        
        for freq in range(1,len(self.frequencies)):
            freq_lf = self.frequencies[freq-1]
            freq_hf = self.frequencies[freq]
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
                
            else:
                print("Higher frequency: ", freq_hf, " Lower Frequency: ", freq_lf, end ='\n')
                freq_ratio = input("Please enter frequency ratio")
            
            freq_ratio_list.append(freq_ratio)
        
        #test if nlags for each step is at least frequency ratio
        for i in range(len(freq_ratio_list)):
            if deque(self.nlags)[i] < freq_ratio_list[i]:
                sys.exit("The number of lags at each step must be at least as long as the corresponding frequency ratio")
        
        # Data Transformations
        
        # TODO
        #YQ0_list[0] = YQ0_list[0]/freq_ratio_list[0] 
        # Divide by frequnecy ratio so the estimatet latent states
        # of the lower frequency data in the higher frequency approximately add up to the 
        # value in the lower frequency (this is a approximation because we work with geometric means)
        
        # perform data transformations
        for i in range(len(YM0_list)):
            if select_m_list[i].size:
                YM0_list[i][:,(select_m_list[i] == 1)[0]] = YM0_list[i][:,(select_m_list[i] == 1)[0]]/100
                YM0_list[i][:,(select_m_list[i] == 0)[0]] = np.log(YM0_list[i][:,(select_m_list[i] == 0)[0]])
        
        YQ0_list[0][:,(select_q[0] == 1)[0]] = YQ0_list[0][:,(select_q[0] == 1)[0]]/100
        YQ0_list[0][:,(select_q[0] == 0)[0]] = np.log(YQ0_list[0][:,(select_q[0] == 0)[0]])
        
        for i in range(len(YMC_list)):
            if select_list[i].size:
                YMC_list[i][:,(select_list[i] == 1)[0]] = YMC_list[i][:,(select_list[i] == 1)[0]]/100
                YMC_list[i][:,(select_list[i] == 0)[0]] = np.log(YMC_list[i][:,(select_list[i] == 0)[0]])
        
        YM_list = YM0_list
        YYcond_list = YMC_list
        
        YQ_list = deque()
        YQ_list.append(np.kron(YQ0_list[0], np.ones((freq_ratio_list[0],1))))
        
        #YQ = np.kron(YQ0, np.ones((freq_ratio,1))) #4 statt freq_ratio?
        Tstar_list = deque()
        T_list = deque()
        YDATA_list = deque()
        # TODO Take into account that YDATA gets shorter with each frequency
        
        if YM_list[0].size:
            Tstar_list.append(YM_list[0].shape[0])
            YDATA_list.append(np.full((Tstar_list[0],nv_list[0]), np.nan))
            YDATA_list[0][:,:Nm_list[0]] = YM_list[0]
        else:
            Tstar_list.append(YQ_list[0].shape[0])
            YDATA_list.append(np.full((0,nv_list[0]), np.nan))
        
    
        
        #for i in range(1,len(YM_list)):
        #    Tstar_list.append(YM_list[i][int(2*freq_ratio_list[i-1]*freq_ratio_list[i]):,:].shape[0])
        #    YDATA_list.append(np.full((Tstar_list[i],nv_list[i]), np.nan))
        #    YDATA_list[i][:,:Nm_list[i]] = YM_list[i][int(2*freq_ratio_list[i-1]*freq_ratio_list[i]):,:]
        
        T_list.append(YQ_list[0].shape[0])
        #for freq in range(1,len(self.frequencies)-1):
        #    T_list.append(np.kron(YMX_list[freq-1], np.ones((freq_ratio_list[freq],1))).shape[0])
        
        
        if YDATA_list[0].size:     
            YDATA_list[0][:T_list[0],Nm_list[0]:] = YQ_list[0]   
        else:
            YDATA_list[0] = YQ_list[0] 
        
        # The other entries of YDATA change with every draw and have to be filled 
        # at every iteration
        
        # fit the model
        ###############
        
        nlags_list_ = self.nlags
        
        # Initialize Matrices
        #########################
        
        # here we initialize all the matrixes for the sampling for all frequencies
        # we again create lists and append them, in the sample loop we can then
        # access these matrices 
        
        #create empty lists, so we can append to them
        T0_list = deque()
        p_list = deque()
        nlags_list = deque()
        kq_list = deque()
        nobs_list = deque()
        Tnew_list = deque()
        Tnobs_list = deque()
        
        varstxt_list = deque()
        smpltxt_list = deque()
        
        index_NY_list = deque()
        
        Sigmap_list    = deque()
        Phip_list      = deque()
        Cons_list      = deque()
        lstate_list    = deque()
        YYactsim_list  = deque()
        XXactsim_list  = deque()
        
        At_mat_list    = deque()
        Pt_mat_list    = deque()
        Atildemat_list = deque()
        Ptildemat_list = deque()
        loglh_list   = deque()
        counter_list = deque()
        
        phi_mm_list = deque()
        
        phi_mq_list = deque()
        phi_mc_list = deque()
        phi_qc_list = deque()
        
        Phi_list = deque()
        
        # Define Transition Equation Matrices in eq (10)
        GAMMAs_list = deque()
        
        
        GAMMAz_list = deque()
        GAMMAc_list = deque()
        GAMMAu_list = deque()

        LAMBDAs_list = deque()
        
        LAMBDAz_list = deque()
        LAMBDAc_list = deque()
        LAMBDAu_list = deque()
        
        # Define Covariance Terms sig_mm, sig_mq_sig_qm, siq_qq
        sigma_list = deque()
        sig_mm_list = deque()
        sig_mq_list = deque()
        sig_qm_list = deque()
        sig_qq_list = deque()
        
        # Define W matrix in eq (15) -- _t for tilde
        Wmatrix_list = deque()
        LAMBDAs_t_list = deque()
        LAMBDAz_t_list = deque()
        LAMBDAc_t_list = deque()
        LAMBDAu_t_list = deque()
        
        At_list = deque()
        At_draw_list = deque()
        Pt_list = deque()
        Pmean_list = deque()
        
        Zm_list = deque()
        
        Yq_list = deque()
        Ym_list = deque()
        
        Pt_last_list = deque()
        
        YY_list = deque()
        
        #for i in range(len(YMh_list)):
        T0_list.append(int(nlags_list_[0]))
        p_list.append(int(nlags_list_[0]))
        nlags_list.append(p_list[0])
        kq_list.append(Nq_list[0] * p_list[0])
        nobs_list.append(int(T_list[0]) - int(T0_list[0]))
        Tnew_list.append(Tstar_list[0]-T_list[0])
        Tnobs_list.append(Tstar_list[0]-T0_list[0])
        
    
        
        #for writing to a forecast w/ history file
        YMh_list[0] = YMh_list[0][int(T0_list[0]):-int(freq_ratio_list[0]),:]
        varstxt_list.append(np.hstack((YMX_list[0].columns, YQX_list[0].columns)))
        smpltxt_list.append(YMX_list[0].index[int(T0_list[0]):])
        
        index_NY_list.append(np.isnan(YDATA_list[0][nobs_list[0]+T0_list[0]:Tnobs_list[0]+T0_list[0],:]).T) # TODO CHECK
        
        # Parameter estimation
        # Matrices for collecting draws from Posterior Density
        Sigmap_list.append(np.zeros((round((self.nsim)/self.thining),nv_list[0],nv_list[0])))
        Phip_list.append(np.zeros((round((self.nsim)/self.thining),int(nv_list[0])*int(p_list[0])+1,int(nv_list[0]))))
        Cons_list.append(np.zeros((round((self.nsim)/self.thining),nv_list[0])))
        lstate_list.append(np.zeros((round((self.nsim)/self.thining),Nq_list[0],int(Tnobs_list[0]))))
        YYactsim_list.append(np.zeros((round((self.nsim)/self.thining),freq_ratio_list[0]+1,nv_list[0])))
        XXactsim_list.append(np.zeros((round((self.nsim)/self.thining),int(freq_ratio_list[0])+1,int(nv_list[0])*int(p_list[0])+1)))
        
        At_mat_list.append(np.zeros((int(Tnobs_list[0]), Nq_list[0]*(int(p_list[0])+1))))
        Pt_mat_list.append(np.zeros((int(Tnobs_list[0]), (Nq_list[0]*(int(p_list[0])+1))**2)))
        Atildemat_list.append(np.zeros((self.nsim, Nq_list[0]*(int(p_list[0])+1))))
        Ptildemat_list.append(np.zeros((self.nsim, Nq_list[0]*(int(p_list[0])+1),Nq_list[0]*(int(p_list[0])+1))))
        loglh_list.append(0)
        counter_list.append(0)
        
        # Define phi, phi(mm), phi(mq), phi(mc) used in alt ss rep -- eq (9)
        phi_mm_list.append(np.zeros((int(Nm_list[0])*int(p_list[0]),int(Nm_list[0]))))
        phi_mm_list[0][:Nm_list[0],:Nm_list[0]] = np.eye(Nm_list[0])
        
        phi_mq_list.append(np.zeros((Nq_list[0]*int(p_list[0]),int(Nm_list[0]))))
        phi_mc_list.append(np.zeros((1,Nm_list[0])))
        phi_qc_list.append(np.zeros((1,Nq_list[0])))
        
        Phi_list.append(np.vstack((0.95 * np.eye(int(Nm_list[0])+Nq_list[0]), np.zeros(((int(Nm_list[0])+Nq_list[0])*(int(p_list[0])-1)+1, int(Nm_list[0])+Nq_list[0])))))
        
        # Define Transition Equation Matrices in eq (10)
        GAMMAs_list.append(np.zeros((Nq_list[0]*(int(p_list[0])+1), Nq_list[0] * (int(p_list[0])+1))))
        IQ = np.eye(Nq_list[0])
        
        for j in range(int(p_list[0])):
            GAMMAs_list[0][(j+1)*Nq_list[0]:(j+2)*Nq_list[0], j*Nq_list[0]:(j+1)*Nq_list[0]] = IQ
            
        GAMMAs_list[0][:Nq_list[0],:Nq_list[0]] = 0.95 * np.eye(Nq_list[0])
        
        GAMMAz_list.append(np.zeros((Nq_list[0]* (int(p_list[0])+1), int(Nm_list[0])*p_list[0])))
        GAMMAc_list.append(np.zeros((Nq_list[0]*(p_list[0]+1), 1)))
        GAMMAu_list.append(np.vstack((np.eye(Nq_list[0]), np.zeros((p_list[0]*Nq_list[0],Nq_list[0])))))

        LAMBDAs_list.append(np.vstack((np.hstack((np.zeros((Nm_list[0],Nq_list[0])), np.transpose(phi_mq_list[0]))),1/freq_ratio_list[0]*np.hstack((np.tile(np.eye(Nq_list[0]), freq_ratio_list[0]), np.zeros((Nq_list[0],Nq_list[0]*(p_list[0]-(freq_ratio_list[0]-1)))))))))
        
        LAMBDAz_list.append(np.vstack((np.transpose(phi_mm_list[0]), np.zeros((Nq_list[0], p_list[0]*Nm_list[0])))))
        LAMBDAc_list.append(np.vstack((np.transpose(phi_mc_list[0]), np.zeros((Nq_list[0],1)))))
        LAMBDAu_list.append(np.vstack((np.eye(Nm_list[0]), np.zeros((Nq_list[0],Nm_list[0])))))
        
        # Define Covariance Terms sig_mm, sig_mq_sig_qm, siq_qq
        sigma_list.append((1e-4)*np.eye(Nm_list[0]+Nq_list[0]))
        sig_mm_list.append(sigma_list[0][:Nm_list[0],:Nm_list[0]])
        sig_mq_list.append(sigma_list[0][:Nm_list[0], Nm_list[0]:])
        sig_qm_list.append(sigma_list[0][Nm_list[0]:, :Nm_list[0]])
        sig_qq_list.append(sigma_list[0][Nm_list[0]:,Nm_list[0]:])
        
        # Define W matrix in eq (15) -- _t for tilde
        
        Wmatrix_list.append(np.hstack((np.eye(Nm_list[0]), np.zeros((Nm_list[0],Nq_list[0])))))
        LAMBDAs_t_list.append(Wmatrix_list[0] @ LAMBDAs_list[0])
        LAMBDAz_t_list.append(Wmatrix_list[0] @ LAMBDAz_list[0])
        LAMBDAc_t_list.append(Wmatrix_list[0] @ LAMBDAc_list[0])
        LAMBDAu_t_list.append(Wmatrix_list[0] @ LAMBDAu_list[0])
        
        At_list.append(np.zeros((Nq_list[0]*(p_list[0]+1))))
        Pt_list.append(np.zeros((Nq_list[0]*(p_list[0]+1), Nq_list[0]*(p_list[0]+1))))
        
        for kk in range(5):
            Pt_list[0] = GAMMAs_list[0] @ Pt_list[0] @ GAMMAs_list[0].T + GAMMAu_list[0] @ sig_qq_list[0] @ GAMMAu_list[0].T
        
        # Lagged HF observations
        Zm_list.append(np.zeros((nobs_list[0], Nm_list[0]*p_list[0])))
        
        if Zm_list[0].size:
            for j in range(p_list[0]):
                Zm_list[0][:, j * Nm_list[0]:(j+1)*Nm_list[0]] = YM_list[0][T0_list[0]-(j+1):T0_list[0]+nobs_list[0]-(j+1),:]
                
        # Observations in Monthly Freq
        
        Ym_list.append(YM_list[0][T0_list[0]:nobs_list[0]+T0_list[0],:])
            
        Yq_list.append(YQ_list[0][T0_list[0]:nobs_list[0]+T0_list[0],:])
        
        
        # Estimation
        #################
        print(" ", end = '\n')
        print("Multi Frequency BVAR: Estimation", end = '\n')
        print("Frequencies: ", self.frequencies, end = "\n")
        print("Total Number of Draws: ",self.nsim)
        
        #Here we start the sample loop, j is the current sample
        #inside the sample loop we need a loop for the MFBVARS
        for j in tqdm(range(self.nsim)):
            for m in range(len(YMh_list)):
                
                # initialization
                if j > 0:
                    At_list[m] = At_draw_list[m][0,:].T 
                    Pt_list[m] = Pmean_list[m]
                    
                # Kalman Filter Loop
                #########################
                
                
                for t in range(nobs_list[m]):  
                    
                    if Ym_list[m].size:
                        
                        if (t+1+T0_list[m])/freq_ratio_list[m] - np.floor((t+T0_list[m]+1)/freq_ratio_list[m]) == 0:
                            At1 = At_list[m]
                            Pt1 = Pt_list[m]
                            # Forecasting
                            alphahat = GAMMAs_list[m] @ At1 + GAMMAz_list[m] @ Zm_list[m][t,:] + GAMMAc_list[m][:,0]
                            Phat = GAMMAs_list[m] @ Pt1 @ GAMMAs_list[m].T + GAMMAu_list[m] @ sig_qq_list[m] @ GAMMAu_list[m].T
                            Phat = 0.5*(Phat + Phat.T)
                            yhat = LAMBDAs_list[m] @ alphahat + LAMBDAz_list[m] @ Zm_list[m][t,:] + LAMBDAc_list[m][:,0]
                            nut = np.concatenate((Ym_list[m][t,:], Yq_list[m][t,:])) - yhat  
                            Ft = (LAMBDAs_list[m] @ Phat @ LAMBDAs_list[m].T + LAMBDAu_list[m] @ sig_mm_list[m] @ LAMBDAu_list[m].T
                                + LAMBDAs_list[m] @ GAMMAu_list[m] @ sig_qm_list[m] @ LAMBDAu_list[m].T
                                + LAMBDAu_list[m] @ sig_mq_list[m] @ GAMMAu_list[m].T @ LAMBDAs_list[m].T)
                            Ft = 0.5*(Ft+Ft.T)
                            Xit = LAMBDAs_list[m] @ Phat + LAMBDAu_list[m] @ sig_mq_list[m] @ GAMMAu_list[m].T
                            sol = Xit.T @ invert_matrix(Ft)
                            At_list[m] = alphahat + sol @ nut
                            Pt_list[m] = Phat - sol @ Xit
                            At_mat_list[m][t,:] = At_list[m].T
                            Pt_mat_list[m][t,:] = Pt_list[m].reshape((1, (Nq_list[m]*(p_list[m]+1))**2), order = "F")
                        else:
                            At1 = At_list[m]
                            Pt1 = Pt_list[m]
                            # Forecasting
                            alphahat = GAMMAs_list[m] @ At1 + GAMMAz_list[m] @ Zm_list[m][t,:] + GAMMAc_list[m][:,0]
                            Phat = GAMMAs_list[m] @ Pt1 @ GAMMAs_list[m].T + GAMMAu_list[m] @ sig_qq_list[m] @ GAMMAu_list[m].T
                            Phat = 0.5*(Phat + Phat.T)
                            yhat = LAMBDAs_t_list[m] @ alphahat + LAMBDAz_t_list[m] @ Zm_list[m][t,:] + LAMBDAc_t_list[m][:,0]
                        
                            nut = Ym_list[m][t,:] - yhat
                            Ft = (LAMBDAs_t_list[m] @ Phat @ LAMBDAs_t_list[m].T + LAMBDAu_t_list[m] @ sig_mm_list[m] @ LAMBDAu_t_list[m].T
                                + LAMBDAs_t_list[m] @ GAMMAu_list[m] @ sig_qm_list[m] @ LAMBDAu_t_list[m].T
                                + LAMBDAu_t_list[m] @ sig_mq_list[m] @ GAMMAu_list[m].T @ LAMBDAs_t_list[m].T)
                            Ft = 0.5*(Ft+Ft.T)
                            Xit = LAMBDAs_t_list[m] @ Phat + LAMBDAu_t_list[m] @ sig_mq_list[m] @ GAMMAu_list[m].T
                            sol = Xit.T @ invert_matrix(Ft)
                            At_list[m] = alphahat + sol @ nut 
                            Pt_list[m] = Phat - sol @ Xit
                        
                            At_mat_list[m][t,:] = At_list[m].T
                            Pt_mat_list[m][t,:] = Pt_list[m].reshape((1, (Nq_list[m]*(p_list[m]+1))**2), order = "F")
                    else:
                        if (t+1+T0_list[m])/freq_ratio_list[m] - np.floor((t+T0_list[m]+1)/freq_ratio_list[m]) == 0:
                            At1 = At_list[m]
                            Pt1 = Pt_list[m]
                            # Forecasting
                            alphahat = GAMMAs_list[m] @ At1 + GAMMAz_list[m] @ Zm_list[m][t,:] + GAMMAc_list[m][:,0]
                            Phat = GAMMAs_list[m] @ Pt1 @ GAMMAs_list[m].T + GAMMAu_list[m] @ sig_qq_list[m] @ GAMMAu_list[m].T
                            Phat = 0.5*(Phat + Phat.T)
                            yhat = LAMBDAs_list[m] @ alphahat + LAMBDAz_list[m] @ Zm_list[m][t,:] + LAMBDAc_list[m][:,0]
                            nut = Yq_list[m][t,:] - yhat  
                            Ft = (LAMBDAs_list[m] @ Phat @ LAMBDAs_list[m].T + LAMBDAu_list[m] @ sig_mm_list[m] @ LAMBDAu_list[m].T
                                + LAMBDAs_list[m] @ GAMMAu_list[m] @ sig_qm_list[m] @ LAMBDAu_list[m].T
                                + LAMBDAu_list[m] @ sig_mq_list[m] @ GAMMAu_list[m].T @ LAMBDAs_list[m].T)
                            Ft = 0.5*(Ft+Ft.T)
                            Xit = LAMBDAs_list[m] @ Phat + LAMBDAu_list[m] @ sig_mq_list[m] @ GAMMAu_list[m].T
                            sol = Xit.T @ invert_matrix(Ft)
                            At_list[m] = alphahat + sol @ nut
                            Pt_list[m] = Phat - sol @ Xit
                            At_mat_list[m][t,:] = At_list[m].T
                            Pt_mat_list[m][t,:] = Pt_list[m].reshape((1, (Nq_list[m]*(p_list[m]+1))**2), order = "F")
                        else:
                            At1 = At_list[m]
                            Pt1 = Pt_list[m]
                            # Forecasting
                            alphahat = GAMMAs_list[m] @ At1 + GAMMAz_list[m] @ Zm_list[m][t,:] + GAMMAc_list[m][:,0]
                            Phat = GAMMAs_list[m] @ Pt1 @ GAMMAs_list[m].T + GAMMAu_list[m] @ sig_qq_list[m] @ GAMMAu_list[m].T
                            Phat = 0.5*(Phat + Phat.T)
                            At_list[m] = alphahat 
                            Pt_list[m] = Phat 
                            At_mat_list[m][t,:] = At_list[m].T
                            Pt_mat_list[m][t,:] = Pt_list[m].reshape((1, (Nq_list[m]*(p_list[m]+1))**2), order = "F")
                Atildemat_list[m][j,:] = At_mat_list[m][nobs_list[m]-1,:]
                
                if j == 0:
                    Pt_last_list.append(Pt_mat_list[m][nobs_list[m]-1,:].reshape((Nq_list[m]*(p_list[m]+1),Nq_list[m]*(p_list[m]+1)), order = "F"))
                else:
                    Pt_last_list[m] = Pt_mat_list[m][nobs_list[m]-1,:].reshape((Nq_list[m]*(p_list[m]+1),Nq_list[m]*(p_list[m]+1)), order = "F")
                Ptildemat_list[m][j,:,:] = Pt_last_list[m]
                
                ########## Unbalanced Dataset ###########
                
                kn = nv_list[m]*(p_list[m]+1)
                
                # Measurement Equation
                
                Z1 = np.zeros((Nm_list[m], kn))
                Z1[:,:Nm_list[m]] = np.eye(Nm_list[m])
                
                Z2 = np.zeros((Nq_list[m], kn))
                
                for bb in range(Nq_list[m]):
                    for ll in range(freq_ratio_list[m]):
                        Z2[bb, (ll+1)*Nm_list[m]+ll*Nq_list[m]+bb] = 1/freq_ratio_list[m]
                
                ZZ = np.vstack((Z1,Z2))
                
                if Ym_list[m].size:
                    BAt = (np.concatenate((Ym_list[m][-1,:], np.atleast_1d(np.squeeze(Atildemat_list[m][j,:Nq_list[m]])))))
                    for rr in range(1,p_list[m]+1):
                        BAt = np.concatenate((BAt,np.concatenate((Ym_list[m][-(rr+1),:], np.atleast_1d(np.squeeze(Atildemat_list[m][j,rr*Nq_list[m]:(rr+1)*Nq_list[m]]))))))
                
                else: 
                    BAt = np.atleast_1d(np.squeeze(Atildemat_list[m][j,:Nq_list[m]]))
                    for rr in range(1,p_list[m]+1):
                        BAt = np.concatenate((BAt,np.atleast_1d(np.squeeze(Atildemat_list[m][j,rr*Nq_list[m]:(rr+1)*Nq_list[m]]))))
                
                BPt = np.zeros((kn,kn))
                for rr in range(p_list[m]+1):
                    for vv in range(p_list[m]+1):
                        BPt[(rr+1)*Nm_list[m]+rr*Nq_list[m]:(rr+1)*(Nm_list[m]+Nq_list[m]), (vv+1)*Nm_list[m]+vv*Nq_list[m]:(vv+1)*(Nm_list[m]+Nq_list[m])] = np.squeeze(
                            Ptildemat_list[m][j,rr*Nq_list[m]:(rr+1)*Nq_list[m],vv*Nq_list[m]:(vv+1)*Nq_list[m]])
                        
                BAt_mat = np.zeros((Tnobs_list[m], kn))
                BPt_mat = np.zeros((Tnobs_list[m], kn**2))
                
                
                BAt_mat[nobs_list[m]-1,:] = BAt
                BPt_mat[nobs_list[m]-1,:] = BPt.reshape((1,kn**2), order = "F")
                
                # Define companion form matrix PHIF
                PHIF = np.zeros((kn,kn))
                IF = np.eye(nv_list[m])
                for i in range(p_list[m]):
                    PHIF[(i+1)*nv_list[m]:(i+2)*nv_list[m], i*nv_list[m]:(i+1)*nv_list[m]] = IF
                    
                PHIF[:nv_list[m], :nv_list[m]*p_list[m]] = Phi_list[m][:-1,:].T
                
                # Define constant term CONF
                CONF = np.hstack((Phi_list[m][-1,:].T, np.zeros((nv_list[m]*p_list[m]))))
                
                # Define covariance term SIGF
                SIGF = np.zeros((kn,kn))
                SIGF[:nv_list[m],:nv_list[m]] = sigma_list[m]
                
                # Filter Loop
                ###################
                
                for t in range(nobs_list[m], Tnobs_list[m]):
                    
                    #new indicator
                    kkk = t-nobs_list[m]
                    
                    # Define new data (ND) and new Z matrix (NZ)
                    ND = YDATA_list[m][nobs_list[m]+T0_list[m]+kkk,:][~np.isnan(YDATA_list[m][nobs_list[m]+T0_list[m]+kkk,:])]
                    NZ = ZZ[~index_NY_list[m][:,kkk],:]
                    
                    BAt1 = BAt
                    BPt1 = BPt
                    
                    # Forecasting
                    Balphahat = PHIF @ BAt1 + CONF
                    BPhat = PHIF @ BPt1 @ PHIF.T + SIGF
                    BPhat = 0.5 * (BPhat+BPhat.T)
                    
                    Byhat = NZ @ Balphahat
                    Bnut = ND - Byhat
                    
                    BFt = NZ @ BPhat @ NZ.T
                    BFt = 0.5*(BFt+BFt.T)
                    
                    #Updating
                    sol_1 = (BPhat @ NZ.T) @ invert_matrix(BFt)
                    BAt = Balphahat + sol_1 @ Bnut
                    BPt = BPhat - sol_1 @ (BPhat @ NZ.T).T
                    BAt_mat[t,:] = BAt
                    BPt_mat[t,:] = BPt.reshape((1,kn**2), order = "F")
                    
                    
                AT_draw = np.zeros((Tnew_list[m]+1, kn))
                
                # Draw from multivariate normal
                Pchol = cholcovOrEigendecomp(BPt_mat[Tnobs_list[m]-1,:].reshape((kn, kn), order = "F"))
                AT_draw[-1, :] = BAt_mat[Tnobs_list[m]-1,:]+np.transpose(Pchol @ np.random.standard_normal(kn))
                
                # Kalman Smoother
                #####################
                
                for i in range(Tnew_list[m]):
                    
                    BAtt = BAt_mat[Tnobs_list[m]-(i+2),:]
                    BPtt = BPt_mat[Tnobs_list[m]-(i+2),:].reshape((kn,kn), order = "F")
                    
                    BPhat = PHIF @ BPtt @ PHIF.T + SIGF
                    BPhat = 0.5*(BPhat+BPhat.T)
                    
                    inv_BPhat = invert_matrix(BPhat)
                    
                    Bnut = AT_draw[-(i+1),:]- PHIF @ BAtt - CONF
                    
                    Amean = BAtt + (BPtt @ PHIF.T) @ inv_BPhat @ Bnut
                    Pmean = BPtt - (BPtt @ PHIF.T) @ inv_BPhat @ np.transpose(BPtt @ PHIF.T)        
            
                    #Draw from multivariate normal
                    Pmchol = cholcovOrEigendecomp(Pmean)
                    AT_draw[-2-i,:] = np.transpose(Amean + Pmchol @ np.random.standard_normal(kn))
                    
                ########## Balanced Dataset ###########
                
                At_draw = np.zeros((nobs_list[m], Nq_list[m] * (p_list[m]+1)))
                for kk in range(p_list[m]+1):
                    At_draw[nobs_list[m]-1, kk * Nq_list[m]:(kk+1)*+Nq_list[m]] = AT_draw[0,(kk+1)*Nm_list[m] + kk*Nq_list[m]:(kk+1)*(Nm_list[m]+Nq_list[m])]
                    
            
                # Kalman Smoother
                #####################
                
                for i in range(nobs_list[m]-1):
                    Att = At_mat_list[m][nobs_list[m]-(i+2),:]#[:, np.newaxis]
                    Ptt = Pt_mat_list[m][nobs_list[m]-(i+2),:].reshape(Nq_list[m]*(p_list[m]+1), Nq_list[m]*(p_list[m]+1), order = "F")
                    
                    
                    Phat = GAMMAs_list[m] @ Ptt @ GAMMAs_list[m].T + GAMMAu_list[m] @ sig_qq_list[m] @ GAMMAu_list[m].T
                    
                    Phat = 0.5*(Phat + Phat.T)
                    
                    inv_Phat = invert_matrix(Phat)
                    
                    nut = At_draw[nobs_list[m]-(i+1), :] - GAMMAs_list[m] @ Att - GAMMAz_list[m] @ Zm_list[m][nobs_list[m]-1-(i+1)] - GAMMAc_list[m][:,0]

                    
                    temp = Ptt @ GAMMAs_list[m].T
                    Amean = Att + temp @ inv_Phat @ nut
                    Pmean = Ptt - temp @ inv_Phat @ np.transpose(temp)
                    
                    Pmchol = cholcovOrEigendecomp(Pmean)
                    At_draw[nobs_list[m]-1-(i+1), :] = np.transpose(Amean + Pmchol @ np.random.standard_normal(Nq_list[m]*(p_list[m]+1)))
                
            
            
                # Minesota Prior                            
                ########################
                if Ym_list[m].size:
                    YY = np.vstack((np.hstack((Ym_list[m], At_draw[:,:Nq_list[m]])), AT_draw[1:,:(Nm_list[m]+Nq_list[m])]))
                else:
                    YY = np.vstack((At_draw[:,:Nq_list[m]], AT_draw[1:,:(Nm_list[m]+Nq_list[m])]))
            
                
                #lstate = np.zeros((Nq_list[m], Tnobs_list[m]))
                #for hh in range(Nq_list[m]):
                #    lstate[hh, :nobs_list[m]] = At_draw[:, hh]
                #    lstate[hh, nobs_list[m]:] = AT_draw[1:, Nm_list[m]+hh]
            
                if (j%self.thining == 0):
                    for hh in range(Nq_list[m]):
                        lstate_list[m][int(int((j)/self.thining)), hh, :nobs_list[m]] = At_draw[:, hh]
                        lstate_list[m][int(int((j)/self.thining)), hh, nobs_list[m]:] = AT_draw[1:, Nm_list[m]+hh]
        
                nobs_ = np.shape(YY)[0] - T0_list[m]
                spec = np.hstack((nlags_list_[m], T0_list[m], self.nex, nv_list[m], nobs_))
            
            
                # dummy observations and actual observations
                mdd, YYact, YYdum, XXact, XXdum = mdd_(self.hyp, YY, spec)
                
                if (j%self.thining == 0):
                    YYactsim_list[m][int(int((j)/self.thining)),:,:] = YYact[-(freq_ratio_list[m]+1):,:] #TODO
                    XXactsim_list[m][int(int((j)/self.thining)),:,:] = XXact[-(freq_ratio_list[m]+1):,:]
                
                
                # Draws from posterior distribution
                
                Tdummy, n = YYdum.shape
                n = int(n)
                Tdummy = int(Tdummy)
                Tobs, n = YYact.shape
                X = np.vstack((XXact, XXdum))
                Y = np.vstack((YYact, YYdum))
                p = int(spec[0])
                T = Tobs + Tdummy
                F = np.zeros((int(n*p), int(n*p)))
                I = np.eye(n)
                
                for i in range(p-1):
                    F[(i+1)*n:(i+2)*n, i*n:(i+1)*n] = I
        
                vl, d, vr = np.linalg.svd(X, full_matrices=False)
                vr = vr.T
                di = 1/d
                B = vl.T @ Y
                xxi = (vr * np.tile(di.T,(n*p+1,1))) 
                inv_x = xxi @ xxi.T
                Phi_tilde = xxi @ B
        
                Sigma = (Y - X @ Phi_tilde).T @ (Y - X @ Phi_tilde)
                
                # Draws from the density Sigma | Y 
                sigma = invwishart.rvs(scale = Sigma, df = T-n*p-1)
                
                # Draws from density vec(Phi)|Sigma(j), Y
                sigma_chol = cholcovOrEigendecomp(np.kron(sigma, inv_x))
                phi_new =  np.squeeze(Phi_tilde.reshape(n*(n*p+1),1,order = "F")) + sigma_chol @ np.random.standard_normal(sigma_chol.shape[0])
                
                #phi_new = np.random.default_rng().multivariate_normal(mean = np.squeeze(Phi_tilde.reshape(n*(n*p+1),1,order = "F")), cov = np.kron(sigma, inv_x), method = "cholesky")
                
                Phi = phi_new.reshape(n*p+1,n,order = "F")
                
                if j > 0:
                    Phi_list[m] = Phi
                elif j == 0 & m == 0:
                    Phi_list[m] = Phi
                
                if (j % self.thining == 0):
                    j_temp = int(j/self.thining)
                    Sigmap_list[m][j_temp,:,:] = sigma
                    Phip_list[m][j_temp,:,:]   = Phi
                    Cons_list[m][j_temp,:]     = Phi[-1,:]
                
                # Define phi(qm), phi(qq), phi(qc)
                phi_qm = np.zeros((Nm_list[m]*p,Nq_list[m]))
                for i in range(p_list[m]):
                    phi_qm[Nm_list[m]*i:Nm_list[m]*(i+1), :] = Phi[i*(Nm_list[m]+Nq_list[m]):i*(Nm_list[m]+Nq_list[m])+Nm_list[m],Nm_list[m]:]
                    
                phi_qq = np.zeros((Nq_list[m]*p,Nq_list[m]))
                for i in range(p):
                    phi_qq[Nq_list[m]*i:Nq_list[m]*(i+1), :] = Phi[i*(Nm_list[m]+Nq_list[m])+Nm_list[m]:(i+1)*(Nm_list[m]+Nq_list[m]), Nm_list[m]:]
                    
                phi_qc = Phi[-1,Nm_list[m]:, np.newaxis]
                
                # Define  phi(mm), phi(mq), phi(mc)
                phi_mm = np.zeros((Nm_list[m]*p, Nm_list[m]))
                for i in range(p):
                    phi_mm[Nm_list[m]*i:Nm_list[m]*(i+1),:] = Phi[i*(Nm_list[m]+Nq_list[m]):i*(Nm_list[m]+Nq_list[m])+Nm_list[m],:Nm_list[m]]
            
                phi_mq = np.zeros((Nq_list[m]*p, Nm_list[m]))
                for i in range(p):
                    phi_mq[Nq_list[m]*i:Nq_list[m]*(i+1),:] = Phi[i*(Nm_list[m]+Nq_list[m])+Nm_list[m]:(i+1)*(Nm_list[m]+Nq_list[m]),:Nm_list[m]]      
                
                phi_mc = Phi[-1,:Nm_list[m], np.newaxis]
            
                # Define Covariance Term sig_mm, sig_mq, sig_qm, sig_qq
                if Nm_list[m]:
                    sig_mm_list[m] = sigma[:Nm_list[m],:Nm_list[m]]  
                    sig_mq_list[m] = 0.5*(sigma[:Nm_list[m],Nm_list[m]:]+np.transpose(sigma[Nm_list[m]:,:Nm_list[m]]))
                    sig_qm_list[m] = 0.5*(sigma[Nm_list[m]:,:Nm_list[m]]+np.transpose(sigma[:Nm_list[m],Nm_list[m]:]))
                    sig_qq_list[m] = sigma[Nm_list[m]:,Nm_list[m]:]
                else:
                    sig_qq_list[m] =  np.atleast_2d(sigma)
                
                
                # Define Transition Equation Matrices
                GAMMAs_list[m] = np.vstack((np.hstack((np.transpose(phi_qq), np.zeros((Nq_list[m],Nq_list[m])))), np.hstack((np.eye(p*Nq_list[m]), np.zeros((p*Nq_list[m],Nq_list[m]))))))
                GAMMAz_list[m] = np.vstack((np.transpose(phi_qm), np.zeros((p*Nq_list[m], p*Nm_list[m]))))
                GAMMAc_list[m] = np.vstack((phi_qc, np.zeros((p*Nq_list[m],1))))
                GAMMAu_list[m] = np.vstack((np.eye(Nq_list[m]), np.zeros((p*Nq_list[m],Nq_list[m]))))
                
                # Define Measurment equation Matrices
                LAMBDAs_list[m] = np.vstack((np.hstack((np.zeros((Nm_list[m],Nq_list[m])), np.transpose(phi_mq))),1/freq_ratio*np.hstack((np.tile(np.eye(Nq_list[m]), freq_ratio), np.zeros((Nq_list[m],Nq_list[m]*(p-(freq_ratio-1))))))))
                
            
                LAMBDAz_list[m] = np.vstack((np.transpose(phi_mm), np.zeros((Nq_list[m], p*Nm_list[m]))))
                LAMBDAc_list[m] = np.vstack((phi_mc, np.zeros((Nq_list[m],1))))
                LAMBDAu_list[m] = np.vstack((np.eye(Nm_list[m]), np.zeros((Nq_list[m],Nm_list[m]))))
                
                # Define W matrix
                Wmatrix_list[m] = np.hstack((np.eye(Nm_list[m]), np.zeros((Nm_list[m], Nq_list[m]))))
                LAMBDAs_t_list[m] = np.matmul(Wmatrix_list[m],LAMBDAs_list[m])
                LAMBDAz_t_list[m] = np.matmul(Wmatrix_list[m], LAMBDAz_list[m])
                LAMBDAc_t_list[m] = np.matmul(Wmatrix_list[m], LAMBDAc_list[m])
                LAMBDAu_t_list[m] = np.matmul(Wmatrix_list[m], LAMBDAu_list[m])
        
                # now we need to define the new low frequency data as the temporally disaggregated
                # low frequency data of the current iteration
                
                if m < (len(YMh_list)-1):
                    if j == 0:
                        #Yq_list.append((np.kron(lstate, np.ones((1,freq_ratio_list[m+1])))).T)
                        YQ0_list.append(YYact)#YQ0_list.append(YYact[:,-Nq_list[m+1]:])
                        YQ_list.append(np.kron(YQ0_list[m+1], np.ones((freq_ratio_list[m+1],1)))[np.product(np.array(nlags_list_[:(m+2)])):,:])
                        #Yq_list.append(YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:])
                        T_list.append(YQ_list[m+1].shape[0])
                        
                        Tstar_list.append(YM_list[m+1][2*np.product(np.array(nlags_list_[:(m+2)])):].shape[0])

                        Tnew_list.append(Tstar_list[m+1]-T_list[m+1])

                        
                        YDATA_list.append(np.full((Tstar_list[m+1],nv_list[m+1]), np.nan))
                        YDATA_list[m+1][:,:Nm_list[m+1]] = YM_list[m+1][-Tstar_list[m+1]:,:]
                        
                        T0_list.append(int(nlags_list_[m+1]))
                        p_list.append(int(nlags_list_[m+1]))
                        nlags_list.append(p_list[m+1])
                        kq_list.append(Nq_list[m+1] * p_list[m+1])
                        nobs_list.append(int(T_list[m+1]) - int(T0_list[m+1]))
                        Tnobs_list.append(Tstar_list[m+1]-T0_list[m+1])
                        
                    
                        
                        #for writing to a forecast w/ history file
                        #YMh_list[m+1] = YMh_list[m+1][int(T0_list[m+1]):-int(freq_ratio_list[m+1]),:]
                        YMh_list[m+1] = YMh_list[m+1][2*np.product(np.array(nlags_list_[:(m+2)]))+int(T0_list[m+1]):-int(freq_ratio_list[m+1]),:]
                        varstxt_list.append(np.hstack((YMX_list[m+1].columns, YQX_list[0].columns)))
                        smpltxt_list.append(YMX_list[m+1].index[int(T0_list[m+1]):])
                        
                        index_NY_list.append(np.isnan(YDATA_list[m+1][nobs_list[m+1]+T0_list[m+1]:Tnobs_list[m+1]+T0_list[m+1],:]).T) # TODO CHECK
                        
                        # Parameter estimation
                        # Matrices for collecting draws from Posterior Density
                        Sigmap_list.append(np.zeros((round((self.nsim)/self.thining),nv_list[m+1],nv_list[m+1])))
                        Phip_list.append(np.zeros((round((self.nsim)/self.thining),int(nv_list[m+1])*int(p_list[m+1])+1,int(nv_list[m+1]))))
                        Cons_list.append(np.zeros((round((self.nsim)/self.thining),nv_list[m+1])))
                        lstate_list.append(np.zeros((round((self.nsim)/self.thining),Nq_list[m+1],int(Tnobs_list[m+1]))))
                        YYactsim_list.append(np.zeros((round((self.nsim)/self.thining),freq_ratio_list[m+1]+1,nv_list[m+1])))
                        XXactsim_list.append(np.zeros((round((self.nsim)/self.thining),int(freq_ratio_list[m+1])+1,int(nv_list[m+1])*int(p_list[m+1])+1)))
                        
                        At_mat_list.append(np.zeros((int(Tnobs_list[m+1]), Nq_list[m+1]*(int(p_list[m+1])+1))))
                        Pt_mat_list.append(np.zeros((int(Tnobs_list[m+1]), (Nq_list[m+1]*(int(p_list[m+1])+1))**2)))
                        Atildemat_list.append(np.zeros((self.nsim, Nq_list[m+1]*(int(p_list[m+1])+1))))
                        Ptildemat_list.append(np.zeros((self.nsim, Nq_list[m+1]*(int(p_list[m+1])+1),Nq_list[m+1]*(int(p_list[m+1])+1))))
                        loglh_list.append(0)
                        counter_list.append(0)
                        
                        # Define phi, phi(mm), phi(mq), phi(mc) used in alt ss rep -- eq (9)
                        phi_mm_list.append(np.zeros((int(Nm_list[m+1])*int(p_list[m+1]),int(Nm_list[m+1]))))
                        phi_mm_list[m+1][:Nm_list[m+1],:Nm_list[m+1]] = np.eye(Nm_list[m+1])
                        
                        phi_mq_list.append(np.zeros((Nq_list[m+1]*int(p_list[m+1]),int(Nm_list[m+1]))))
                        phi_mc_list.append(np.zeros((1,Nm_list[m+1])))
                        phi_qc_list.append(np.zeros((1,Nq_list[m+1])))
                        
                        Phi_list.append(np.vstack((0.95 * np.eye(int(Nm_list[m+1])+Nq_list[m+1]), np.zeros(((int(Nm_list[m+1])+Nq_list[m+1])*(int(p_list[m+1])-1)+1, int(Nm_list[m+1])+Nq_list[m+1])))))
                        
                        # Define Transition Equation Matrices in eq (10)
                        GAMMAs_list.append(np.zeros((Nq_list[m+1]*(int(p_list[m+1])+1), Nq_list[m+1] * (int(p_list[m+1])+1))))
                        IQ = np.eye(Nq_list[m+1])
                        
                        for k in range(int(p_list[m+1])):
                            GAMMAs_list[m+1][(k+1)*Nq_list[m+1]:(k+2)*Nq_list[m+1], k*Nq_list[m+1]:(k+1)*Nq_list[m+1]] = IQ
                            
                        GAMMAs_list[m+1][:Nq_list[m+1],:Nq_list[m+1]] = 0.95 * np.eye(Nq_list[m+1])
                        
                        GAMMAz_list.append(np.zeros((Nq_list[m+1]* (int(p_list[m+1])+1), int(Nm_list[m+1])*p_list[m+1])))
                        GAMMAc_list.append(np.zeros((Nq_list[m+1]*(p_list[m+1]+1), 1)))
                        GAMMAu_list.append(np.vstack((np.eye(Nq_list[m+1]), np.zeros((p_list[m+1]*Nq_list[m+1],Nq_list[m+1])))))

                        LAMBDAs_list.append(np.vstack((np.hstack((np.zeros((Nm_list[m+1],Nq_list[m+1])), np.transpose(phi_mq_list[m+1]))),1/freq_ratio_list[m+1]*np.hstack((np.tile(np.eye(Nq_list[m+1]), freq_ratio_list[m+1]), np.zeros((Nq_list[m+1],Nq_list[m+1]*(p_list[m+1]-(freq_ratio_list[m+1]-1)))))))))
                        
                        LAMBDAz_list.append(np.vstack((np.transpose(phi_mm_list[m+1]), np.zeros((Nq_list[m+1], p_list[m+1]*Nm_list[m+1])))))
                        LAMBDAc_list.append(np.vstack((np.transpose(phi_mc_list[m+1]), np.zeros((Nq_list[m+1],1)))))
                        LAMBDAu_list.append(np.vstack((np.eye(Nm_list[m+1]), np.zeros((Nq_list[m+1],Nm_list[m+1])))))
                        
                        # Define Covariance Terms sig_mm, sig_mq_sig_qm, siq_qq
                        sigma_list.append((1e-4)*np.eye(Nm_list[m+1]+Nq_list[m+1]))
                        sig_mm_list.append(sigma_list[m+1][:Nm_list[m+1],:Nm_list[m+1]])
                        sig_mq_list.append(sigma_list[m+1][:Nm_list[m+1], Nm_list[m+1]:])
                        sig_qm_list.append(sigma_list[m+1][Nm_list[m+1]:, :Nm_list[m+1]])
                        sig_qq_list.append(sigma_list[m+1][Nm_list[m+1]:,Nm_list[m+1]:])
                        
                        # Define W matrix in eq (15) -- _t for tilde
                        Wmatrix_list.append(np.hstack((np.eye(Nm_list[m+1]), np.zeros((Nm_list[m+1],Nq_list[m+1])))))
                        LAMBDAs_t_list.append(Wmatrix_list[m+1] @ LAMBDAs_list[m+1])
                        LAMBDAz_t_list.append(Wmatrix_list[m+1] @ LAMBDAz_list[m+1])
                        LAMBDAc_t_list.append(Wmatrix_list[m+1] @ LAMBDAc_list[m+1])
                        LAMBDAu_t_list.append(Wmatrix_list[m+1] @ LAMBDAu_list[m+1])
                        
                        At_list.append(np.zeros((Nq_list[m+1]*(p_list[m+1]+1))))
                        Pt_list.append(np.zeros((Nq_list[m+1]*(p_list[m+1]+1), Nq_list[m+1]*(p_list[m+1]+1))))
                        
                        for kk in range(5):
                            Pt_list[m+1] = GAMMAs_list[m+1] @ Pt_list[m+1] @ GAMMAs_list[m+1].T + GAMMAu_list[m+1] @ sig_qq_list[m+1] @ GAMMAu_list[m+1].T
                        
                        YM_short = YM_list[m+1][2*np.product(np.array(nlags_list_[:(m+2)])):,:]
                        
                        # Lagged HF observations
                        Zm_list.append(np.zeros((nobs_list[m+1], Nm_list[m+1]*p_list[m+1])))
                        for k in range(p_list[m+1]):
                            Zm_list[m+1][:, k * Nm_list[m+1]:(k+1)*Nm_list[m+1]] = YM_short[T0_list[m+1]-(k+1):T0_list[m+1]+nobs_list[m+1]-(k+1),:]
                        
                        # Observations in Monthly Freq
                        
                        #Ym_list.append(YM_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:])
                        Ym_list.append(YM_short[T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:])    
                        Yq_list.append(YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:])
                        
                    else:
                        #Yq_list[m+1] = (np.kron(lstate, np.ones((1,freq_ratio_list[m+1])))).T
                        YQ0_list[m+1] = YYact
                        YQ_list[m+1] = np.kron(YQ0_list[m+1], np.ones((freq_ratio_list[m+1],1)))[np.product(np.array(nlags_list_[:(m+2)])):,:]
                        Yq_list[m+1] = YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:]
                        T_list[m+1] = YQ_list[m+1].shape[0]
                        Tnew_list[m+1] = Tstar_list[m+1]-T_list[m+1]
                        
                    YDATA_list[m+1][:T_list[m+1],Nm_list[m+1]:] = YQ_list[m+1]    
                
                if j == 0:
                    At_draw_list.append(At_draw)
                else:
                    At_draw_list[m] = At_draw
                
                if j == 0:
                    Pmean_list.append(Pmean)
                else:
                    Pmean_list[m] = Pmean
                        
                        
                #TODO
                    
                
            self.YYactsim = YYactsim_list[-1]
            self.XXactsim = XXactsim_list[-1]
            self.Phip = Phip_list[-1]
            self.Sigmap = Sigmap_list[-1]
            #self.lstate = lstate
            #self.latent_draws = lstate_list[-1]
            
            self.YDATA = YDATA_list[-1]
            self.select_q = select_q
            self.select_m = select_m_list[-1]
            self.select = select_list[-1]
            self.varlist = varlist_list[-1]
            self.Nm = Nm_list[-1]
            #self.Nq_list[-1] = Nq_list[-1]
            self.nv = nv_list[-1]
            self.YYcond = YYcond_list[-1]
            self.freq_ratio = freq_ratio_list[-1]
            self.YMC = YMC_list[-1]
            self.T = T_list[-1]
            self.Tstar = Tstar_list[-1]
            self.YQX = YQX_list[-1]
            self.YMX = YMX_list[-1]
            self.YQ = YQ_list[-1]
            self.YM = YM_list[-1]
            self.YMh = YMh_list[-1]
            self.exc = exc_list[-1]
            
            self.Ym = Ym_list[-1]
            self.Yq = Yq_list[-1]
            self.T0 = T0_list[-1]
            self.Tnew = Tnew_list[-1]
            
        #save lists to self
        self.YMh_list = YMh_list
        self.T0_list = T0_list
        self.freq_ratio_list = freq_ratio_list
        self.varstxt_list = varstxt_list
        self.Nm_list = Nm_list
        self.Nq_list = Nq_list
        self.nv_list = nv_list        
        self.YYactsim_list = YYactsim_list
        self.XXactsim_list = XXactsim_list
        self.Phip_list = Phip_list
        self.Sigmap_list = Sigmap_list
        self.exc_list = exc_list
        self.YYcond_list = YYcond_list
        self.select_list = select_list
        self.Tnew_list = Tnew_list
        self.Ym_list = Ym_list
        self.Yq_list = Yq_list
        self.select_m_list = select_m_list
        self.select_q = select_q
        self.lstate_list = lstate_list
        self.nlags_list = nlags_list
        self.varlist_list = varlist_list
        self.YMX_list =YMX_list
        
        
        
    def forecast(self):
        
        
        YY_m_list = deque()
        YY_med_list = deque()
        YY_095_list = deque()
        YY_005_list = deque()
        YY_084_list = deque()
        YY_016_list = deque()
        forecast_draws_list = deque()
        
        for m in range(len(self.YMh_list)):
            
            # Define forecast horizon in the current frequency m
            if (m < len(self.YMh_list)-1):
                H_ = int(self.H/int(np.product(list(itertools.islice(self.freq_ratio_list, m+1, len(self.freq_ratio_list))))))
            else:
                H_= int(self.H)
        
            #for writing to a forecast w/ history file
            #self.YMh_list[m] = self.YMh_list[m][self.T0_list[m]:-self.freq_ratio_list[m],:]
            #self.varstxt_list = np.hstack((self.YMX.columns, self.YQX.columns))
            #self.smpltxt =self.YMX.index[self.T0:]
            
            ###############  
            # Forecasting #
            ###############
            
            # store forecasts in monthly frequency
            YYvector_ml  = np.zeros((round((self.nsim)/self.thining),H_,self.Nm_list[m]+self.Nq_list[m]))     # collects now/forecast      
            YYvector_mg  = np.zeros((round((self.nsim)/self.thining),H_,self.Nm_list[m]+self.Nq_list[m]))
            YYvector_m0  = np.zeros((round((self.nsim)/self.thining),H_,self.nv_list[m]))
            
            # store forecasts in quarterly frequency
            YYvector_ql  = np.zeros((round((self.nsim)/self.thining),int(self.H/self.freq_ratio_list[m]),self.Nm_list[m]+self.Nq_list[m]))   
            YYvector_qg  = np.zeros((round((self.nsim)/self.thining),int(self.H/self.freq_ratio_list[m]),self.Nm_list[m]+self.Nq_list[m]))
            
            print(" ", end = '\n')
            print("Mixed Frequency BVAR: Forecasting", end = "\n")
            print("Forecast Horizon: ", H_, end = "\n")
            print("Total Draws: ", self.nsim)
            print("Current Frequency: ", self.frequencies[m+1])
            
        
            
            for jj in tqdm(range(round((self.nsim)/self.thining))):
                
                YYact = np.squeeze(self.YYactsim_list[m][jj, -1, :])
                XXact = np.squeeze(self.XXactsim_list[m][jj, -1, :])
                post_phi = np.squeeze(self.Phip_list[m][jj,:,:])
                post_sig = np.squeeze(self.Sigmap_list[m][jj,:,:])
                

                # Bayesian Estimation Forecasting 
                ###################################

                
                YYpred = np.zeros((H_+1, self.nv_list[m])) # forecasts from VAR
                YYpred[0,:] = YYact
                XXpred = np.zeros((H_+1, self.nv_list[m]*self.nlags_list[m]+1))
                XXpred[:,-1] = np.full((H_+1), fill_value = 1)
                XXpred[0,:] = XXact
                
                # given posterior draw, draw number (H+1) random sequence
        
                error_pred = np.zeros((H_+1, self.nv_list[m]))
                
                for h in range(H_+1):
                    if post_sig.size > 1:
                        error_pred[h,:] = np.random.default_rng().multivariate_normal(mean = np.zeros(self.nv_list[m]), cov = post_sig, method = "cholesky")
                    else:
                        error_pred[h,:] = np.random.default_rng().normal(loc = 0, scale = post_sig)
                # given posterior draw, iterate forward to construct forecasts
                
                for h in range(1,H_+1):
                    
                    XXpred[h,self.nv_list[m]:-1] = XXpred[h-1, :-self.nv_list[m]-1]
                    XXpred[h, :self.nv_list[m]] = YYpred[h-1, :]
                    #YYpred[h,:] = (XXpred[h,:] @ post_phi + error_pred[h,:])
                    # TODO add conditional forecasts
                    YYpred[h,:] = (1-self.exc_list[m][h-1,:]) * (XXpred[h,:] @ post_phi + error_pred[h,:]) + self.exc_list[m][h-1,:] * self.YYcond_list[m][h-1,:]
                
                YYpred1 = YYpred
                YYpred = YYpred[1:,:]
                

                
                
                # Now-/Forecasts
                # Store in hf
                YYvector_ml[jj,:,:] = YYpred
                YYvector_mg[jj,:,:] = 100*(YYpred1[1:,:]-YYpred1[:-1,:])
                YYvector_m0[jj,:,(self.select_list[m] == 1)[0]] = (100* YYpred[:, (self.select_list[m] == 1)[0]]).T
                YYvector_m0[jj,:,(self.select_list[m] == 0)[0]] = np.exp(YYpred[:, (self.select_list[m] == 0)[0]]).T
                
                    
                '''       
                # store forecasts in low frequency
                
                for ll in range(int(self.H/self.freq_ratio_list[m]-1)):
                    YYvector_ql[jj, ll+1,:] = np.mean(YYvector_ml[jj,self.freq_ratio_list[m]*(ll+1)-self.Tnew_list[m]:self.freq_ratio_list[m]*(ll+2)-self.Tnew_list[m],:], axis = 0)
                    
                YYnow = self.YYactsim[jj,-self.Tnew_list[m]:,:]
                
                if YYnow.shape[0] > YYnow.shape[1]:
                    YYnow = YYnow.T
                    
                if YYvector_ml[jj,:self.freq_ratio_list[m]-self.Tnew_list[m],:].shape[0] > YYvector_ml[jj,:self.freq_ratio_list[m]-self.Tnew_list[m],:].shape[1]:
                    YYfuture = YYvector_ml[jj,:self.freq_ratio_list[m]-self.Tnew_list[m],:].T
                else:
                    YYfuture = YYvector_ml[jj,:self.freq_ratio_list[m]-self.Tnew_list[m],:]
                
                
                if self.Tnew_list[m] == self.freq_ratio_list[m]:
                    YYvector_ql[jj,0,:] = np.mean(YYnow, axis = 0)
                else:
                    YYvector_ql[jj,0,:] = np.mean(np.vstack((YYnow, YYfuture)), axis = 0)
                
                YYvector_qg[jj,0,:] = (100 * YYvector_ql[jj, 0, :] - 
                                    np.mean(np.hstack((self.Ym_list[m][-(self.freq_ratio_list[m]-1):,:], self.Yq_list[m][-(self.freq_ratio_list[m]-1):,:])), axis = 0)) #np.mean(np.hstack((Ym[-2:,:], Yq[-2:,:])), axis = 0))
                
                for bb in range(1,int(self.H/self.freq_ratio_list[m])):
                    YYvector_qg[jj, bb, :] = 100*YYvector_ql[jj, bb, :] - YYvector_ql[jj, bb - 1,:]
                '''
            forecast_draws_list.append(YYvector_ml)
            
            #mean
            YYftr_m = np.nanmean(YYvector_ml[self.nburn:,:,:], axis = 0)
            YYftr_m[:, (self.select_list[m] == 1)[0]] = 100 * YYftr_m[:, (self.select_list[m] == 1)[0]]
            YYftr_m[:, (self.select_list[m] == 0)[0]] = np.exp(YYftr_m[:, (self.select_list[m] == 0)[0]])
            
            YYnow_m = np.mean(self.YYactsim_list[m][self.nburn:,1:(self.freq_ratio_list[m]+1),:self.Nm_list[m]], axis = 0) # actual/nowcast monthlies
            if YYnow_m.size:
                YYnow_m[:, (self.select_m_list[m] == 1)[0]] = 100 * YYnow_m[:, (self.select_m_list[m] == 1)[0]]
                YYnow_m[:, (self.select_m_list[m] == 0)[0]] = np.exp(YYnow_m[:,(self.select_m_list[m] == 0)[0]])
            
            lstate_m = np.mean(self.lstate_list[m][self.nburn:,:,:], axis = 0).T # hf obs for lf vars
            lstate_m[:, (self.select_q[m] == 1)[0]] = 100 * lstate_m[:, (self.select_q[m] == 1)[0]]
            lstate_m[:, (self.select_q[m] == 0)[0]] = np.exp(lstate_m[:, (self.select_q[m]== 0)[0]])
            

            
            YMh_len_correction = int(self.YMh_list[m].shape[0] - lstate_m[:-(self.freq_ratio_list[m]),:].shape[0])
            
            if self.YMh_list[m].size:
                self.YMh_list[m][:, (self.select_m_list[m] == 1)[0]] = 100 * self.YMh_list[m][:, (self.select_m_list[m] == 1)[0]]
                self.YMh_list[m][:, (self.select_m_list[m] == 0)[0]] =  np.exp(self.YMh_list[m][:, (self.select_m_list[m] == 0)[0]])
            
            if self.YMh_list[m].size:
                YY_m_list.append(np.vstack((np.vstack((np.hstack((self.YMh_list[m][YMh_len_correction:,:], lstate_m[:-(self.freq_ratio_list[m]),:])), np.hstack((YYnow_m, lstate_m[-self.freq_ratio_list[m]:,:])))), YYftr_m)))
                #YY_m_list.append(np.vstack((np.vstack((np.hstack((self.YMh_list[m], lstate_m[:-(self.freq_ratio_list[m]),:])), np.hstack((YYnow_m, lstate_m[-self.freq_ratio_list[m]:,:])))), YYftr_m)))
            else:
                YY_m_list.append(np.vstack((lstate_m,YYftr_m)))
            
            
            
            
            #self.mean_phi = np.mean(self.Phip, axis = 0)
            
            #median
            YYftr_med = np.nanmedian(YYvector_ml[self.nburn:,:,:], axis = 0)
            YYftr_med[:, (self.select_list[m] == 1)[0]] = 100 * YYftr_med[:, (self.select_list[m] == 1)[0]]
            YYftr_med[:, (self.select_list[m] == 0)[0]] = np.exp(YYftr_med[:, (self.select_list[m] == 0)[0]])
            
            YYnow_med = np.median(self.YYactsim_list[m][self.nburn:,1:(self.freq_ratio+1),:self.Nm_list[m]], axis = 0) # actual/nowcast monthlies
            
            if YYnow_med.size:
                YYnow_med[:, (self.select_m_list[m] == 1)[0]] = 100 * YYnow_med[:, (self.select_m_list[m] == 1)[0]]
                YYnow_med[:, (self.select_m_list[m] == 0)[0]] = np.exp(YYnow_med[:, (self.select_m_list[m] == 0)[0]])
            
            lstate_med = np.median(self.lstate_list[m][self.nburn:,:,:], axis = 0).T # hf obs for lf vars
            lstate_med[:, (self.select_q[m] == 1)[0]] = 100 * lstate_med[:, (self.select_q[m] == 1)[0]]
            lstate_med[:, (self.select_q[m] == 0)[0]] = np.exp(lstate_med[:, (self.select_q[m]== 0)[0]])
            
            if self.YMh_list[m].size:
                YY_med_list.append(np.vstack((np.vstack((np.hstack((self.YMh_list[m][YMh_len_correction:,:], lstate_med[:-self.freq_ratio_list[m],:])), np.hstack((YYnow_med, lstate_m[-self.freq_ratio_list[m]:,:])))), YYftr_med)))
            else:
                YY_med_list.append(np.vstack((lstate_med,YYftr_med)))
                
            # safe uncertainty
            # 95%
            YYftr_095 = np.nanquantile(YYvector_ml[self.nburn:,:,:], q = 0.95 ,axis = 0)
            YYftr_095[:, (self.select_list[m] == 1)[0]] = 100 * YYftr_095[:, (self.select_list[m] == 1)[0]]
            YYftr_095[:, (self.select_list[m] == 0)[0]] = np.exp(YYftr_095[:, (self.select_list[m] == 0)[0]])
            
            YYnow_095 = np.quantile(self.YYactsim_list[m][self.nburn:,1:(self.freq_ratio_list[m]+1),:self.Nm_list[m]], q = 0.95, axis = 0) # actual/nowcast monthlies
            if YYnow_095.size:
                YYnow_095[:, (self.select_m_list[m] == 1)[0]] = 100 * YYnow_095[:, (self.select_m_list[m] == 1)[0]]
                YYnow_095[:, (self.select_m_list[m] == 0)[0]] = np.exp(YYnow_095[:, (self.select_m_list[m] == 0)[0]])
            
            lstate_095 = np.quantile(self.lstate_list[m][self.nburn:,:,:], q = 0.95, axis = 0).T # hf obs for lf vars
            lstate_095[:, (self.select_q[m] == 1)[0]] = 100 * lstate_095[:, (self.select_q[m] == 1)[0]]
            lstate_095[:, (self.select_q[m] == 0)[0]] = np.exp(lstate_095[:, (self.select_q[m] == 0)[0]])
            
            YMna = np.full(self.YMh_list[m].shape, np.nan)
            
            if YMna.size:
                YY_095_list.append(np.vstack((np.vstack((np.hstack((YMna[YMh_len_correction:,:],lstate_095[:-self.freq_ratio_list[m],:])), np.hstack((YYnow_095, lstate_095[-self.freq_ratio_list[m]:,:])))), YYftr_095)))
            else:
                YY_095_list.append(np.vstack((lstate_095,YYftr_095)))
            
            # 84%
            YYftr_084 = np.nanquantile(YYvector_ml[self.nburn:,:,:], q = 0.84 ,axis = 0)
            YYftr_084[:, (self.select_list[m] == 1)[0]] = 100 * YYftr_084[:, (self.select_list[m] == 1)[0]]
            YYftr_084[:, (self.select_list[m] == 0)[0]] = np.exp(YYftr_084[:, (self.select_list[m] == 0)[0]])
            
            YYnow_084 = np.quantile(self.YYactsim_list[m][self.nburn:,1:(self.freq_ratio_list[m]+1),:self.Nm_list[m]], q = 0.84, axis = 0) # actual/nowcast monthlies
            if YYnow_084.size:
                YYnow_084[:, (self.select_m_list[m] == 1)[0]] = 100 * YYnow_084[:, (self.select_m_list[m] == 1)[0]]
                YYnow_084[:, (self.select_m_list[m] == 0)[0]] = np.exp(YYnow_084[:, (self.select_m_list[m] == 0)[0]])
            
            lstate_084 = np.quantile(self.lstate_list[m][self.nburn:,:,:], q = 0.84, axis = 0).T # hf obs for lf vars
            lstate_084[:, (self.select_q[m] == 1)[0]] = 100 * lstate_084[:, (self.select_q[m] == 1)[0]]
            lstate_084[:, (self.select_q[m] == 0)[0]] = np.exp(lstate_084[:, (self.select_q[m] == 0)[0]])
            
            YMna = np.full(self.YMh_list[m].shape, np.nan)
            
            if YMna.size:
                YY_084_list.append(np.vstack((np.vstack((np.hstack((YMna[YMh_len_correction:,:],lstate_084[:-self.freq_ratio_list[m],:])), np.hstack((YYnow_084, lstate_084[-self.freq_ratio_list[m]:,:])))), YYftr_084)))
            else:
                YY_084_list.append(np.vstack((lstate_084,YYftr_084)))
                
            # 16%
            
            YYftr_016 = np.nanquantile(YYvector_ml[self.nburn:,:,:], q = 0.16 ,axis = 0)
            YYftr_016[:, (self.select_list[m] == 1)[0]] = 100 * YYftr_016[:, (self.select_list[m] == 1)[0]]
            YYftr_016[:, (self.select_list[m] == 0)[0]] = np.exp(YYftr_016[:, (self.select_list[m] == 0)[0]])
            
            YYnow_016 = np.quantile(self.YYactsim_list[m][self.nburn:,1:(self.freq_ratio_list[m]+1),:self.Nm_list[m]], q = 0.16, axis = 0) # actual/nowcast monthlies
            if YYnow_016.size:
                YYnow_016[:, (self.select_m_list[m] == 1)[0]] = 100 * YYnow_016[:, (self.select_m_list[m] == 1)[0]]
                YYnow_016[:, (self.select_m_list[m] == 0)[0]] = np.exp(YYnow_016[:, (self.select_m_list[m] == 0)[0]])
            
            lstate_016 = np.quantile(self.lstate_list[m][self.nburn:,:,:], q = 0.16, axis = 0).T # hf obs for lf vars
            lstate_016[:, (self.select_q[m] == 1)[0]] = 100 * lstate_016[:, (self.select_q[m] == 1)[0]]
            lstate_016[:, (self.select_q[m] == 0)[0]] = np.exp(lstate_016[:, (self.select_q[m] == 0)[0]])
            
            YMna = np.full(self.YMh_list[m].shape, np.nan)
            
            if YMna.size:
                YY_016_list.append(np.vstack((np.vstack((np.hstack((YMna[YMh_len_correction:,:],lstate_016[:-self.freq_ratio_list[m],:])), np.hstack((YYnow_016, lstate_016[-self.freq_ratio_list[m]:,:])))), YYftr_016)))
            else:
                YY_016_list.append(np.vstack((lstate_016,YYftr_016)))
            
            
            # 5%    
            
            YYftr_005 = np.nanquantile(YYvector_ml[self.nburn:,:,:], q = 0.05 ,axis = 0)
            YYftr_005[:, (self.select_list[m] == 1)[0]] = 100 * YYftr_005[:, (self.select_list[m] == 1)[0]]
            YYftr_005[:, (self.select_list[m] == 0)[0]] = np.exp(YYftr_005[:, (self.select_list[m] == 0)[0]])
            
            YYnow_005 = np.quantile(self.YYactsim_list[m][self.nburn:,1:(self.freq_ratio_list[m]+1),:self.Nm_list[m]], q = 0.05, axis = 0) # actual/nowcast monthlies
            
            if YYnow_005.size:   
                YYnow_005[:, (self.select_m_list[m] == 1)[0]] = 100 * YYnow_005[:, (self.select_m_list[m] == 1)[0]]
                YYnow_005[:, (self.select_m_list[m] == 0)[0]] = np.exp(YYnow_005[:, (self.select_m_list[m] == 0)[0]])
            
            lstate_005 = np.quantile(self.lstate_list[m][self.nburn:,:,:], q = 0.05, axis = 0).T # hf obs for lf vars
            lstate_005[:, (self.select_q[m] == 1)[0]] = 100 * lstate_005[:, (self.select_q[m] == 1)[0]]
            lstate_005[:, (self.select_q[m] == 0)[0]] = np.exp(lstate_005[:, (self.select_q[m] == 0)[0]])
            
            if YMna.size:
                YY_005_list.append(np.vstack((np.vstack((np.hstack((YMna[YMh_len_correction:,:], lstate_005[:-self.freq_ratio_list[m],:])), np.hstack((YYnow_005, lstate_005[-self.freq_ratio_list[m]:,:])))), YYftr_005)))
            else:
                YY_005_list.append(np.vstack((lstate_005,YYftr_005))) 
        
        self.YY_095 = YY_095_list
        self.YY_084 = YY_084_list
        self.YY_016 = YY_016_list
        self.YY_005 = YY_005_list
        self.YY_mean = YY_m_list
        self.YY_median = YY_med_list
        self.forecast_draws_list = forecast_draws_list
        
    def aggregate(self, frequency):
        """
        Aggregates the Mean, Median and quantililes in the highest frequency to the desired frequency
        
        Parameters
        
        frequency : str
            The frequency to which the data should be aggregated to
        ----------
        Returns
        -------
        None.

        """
        
        if self.forecast_draws_list is None :
                sys.exit("Error: To gaggregate generate forecasts first")
                
        
        # Set the frequency ratio        
        freq_lf = frequency
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
                
                
        diff = self.YMX.shape[0]-(self.YY_005[-1].shape[0]- self.H)
        start = freq_ratio * math.ceil(diff/freq_ratio) - diff
        self.YY_095_agg = np.array(pd.DataFrame(self.YY_095[-1][start:]).groupby(pd.DataFrame(self.YY_095[-1][start:]).index // freq_ratio).mean())
        self.YY_084_agg = np.array(pd.DataFrame(self.YY_084[-1][start:]).groupby(pd.DataFrame(self.YY_084[-1][start:]).index // freq_ratio).mean())
        self.YY_016_agg = np.array(pd.DataFrame(self.YY_016[-1][start:]).groupby(pd.DataFrame(self.YY_016[-1][start:]).index // freq_ratio).mean())
        self.YY_005_agg = np.array(pd.DataFrame(self.YY_005[-1][start:]).groupby(pd.DataFrame(self.YY_005[-1][start:]).index // freq_ratio).mean())
        self.YY_mean_agg = np.array(pd.DataFrame(self.YY_mean[-1][start:]).groupby(pd.DataFrame(self.YY_mean[-1][start:]).index // freq_ratio).mean())
        self.YY_median_agg = np.array(pd.DataFrame(self.YY_median[-1])[start:].groupby(pd.DataFrame(self.YY_median[-1][start:]).index // freq_ratio).mean())
        self.agg_freq = frequency
    
    def save(self, filename = "mufbvar_model.pkl"):
        """
        
        Parameters
        ----------
        filename : Path where to save the object. End must be .pkl
        Returns
        -------
        None.
        Saves the MFBVAR Object
        """
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
            
            
    
    def to_excel(self, filename, agg = False):
        """
        

        Parameters
        ----------
        agg : Boolean
            Should the aggregated series be shown
        filname : TYPE
            DESCRIPTION.
        

        Returns
        -------
        None.

        """
        
        if self.forecast_draws_list is None :
                sys.exit("Error: To generate traceplots, generate forecasts first")
                
        if agg == True and not hasattr(self, 'YY_095_agg'):
            sys.exit("Aggregate first")
        
        if agg == False:
            YY_mean_pd = pd.DataFrame(self.YY_mean[-1], columns = self.varlist_list[-1])
            YY_median_pd = pd.DataFrame(self.YY_median[-1], columns = self.varlist_list[-1])
            
            YY_095_pd = pd.DataFrame(self.YY_095[-1], columns = self.varlist_list[-1])
            YY_005_pd = pd.DataFrame(self.YY_005[-1], columns = self.varlist_list[-1])
            
            YY_084_pd = pd.DataFrame(self.YY_084[-1], columns = self.varlist_list[-1])
            YY_016_pd = pd.DataFrame(self.YY_016[-1], columns = self.varlist_list[-1])
            
            with pd.ExcelWriter(filename) as writer:
            #writer = pd.ExcelWriter("sim_data.xlsx", engine="xlsxwriter")
                YY_mean_pd.to_excel(writer, sheet_name = "mean")
                YY_median_pd.to_excel(writer, sheet_name = "median")
                YY_095_pd.to_excel(writer, sheet_name = "95_quantile")
                YY_005_pd.to_excel(writer, sheet_name = "5_quantile")
                YY_084_pd.to_excel(writer, sheet_name = "84_quantile")
                YY_016_pd.to_excel(writer, sheet_name = "16_quantile")
            #writer.close()
        else:
            YY_mean_pd = pd.DataFrame(self.YY_mean_agg, columns = self.varlist_list[-1])
            YY_median_pd = pd.DataFrame(self.YY_median_agg, columns = self.varlist_list[-1])
            
            YY_095_pd = pd.DataFrame(self.YY_095_agg, columns = self.varlist_list[-1])
            YY_005_pd = pd.DataFrame(self.YY_005_agg,  columns = self.varlist_list[-1])
            
            YY_084_pd = pd.DataFrame(self.YY_084_agg, columns = self.varlist_list[-1])
            YY_016_pd = pd.DataFrame(self.YY_016_agg,  columns = self.varlist_list[-1])
            
            with pd.ExcelWriter(filename) as writer:
            #writer = pd.ExcelWriter("sim_data.xlsx", engine="xlsxwriter")
                YY_mean_pd.to_excel(writer, sheet_name = "mean")
                YY_median_pd.to_excel(writer, sheet_name = "median")
                YY_095_pd.to_excel(writer, sheet_name = "95_quantile")
                YY_005_pd.to_excel(writer, sheet_name = "5_quantile")
                YY_084_pd.to_excel(writer, sheet_name = "84_quantile")
                YY_016_pd.to_excel(writer, sheet_name = "16_quantile")
        
    def mean_plot(self,frequency, variables = "all", save = True, name = "Output", show = True):
        
        """
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
        Returns
        -------
        None.
        """
        plt.ioff()
        
        if self.forecast_draws_list is None :
                sys.exit("Error: To generate traceplots, generate forecasts first")
        
        if isinstance(variables, str):
            if variables == "all":
                variables = self.varlist_list[frequency]
            else:
                sys.exit("variables must be either a list of variables or all")
            
            
        check = set(variables)-set(self.varlist_list[frequency])        
        if check and not variables == "all":
            sys.exit(print(check, " not in " , self.varlist_list[frequency]))
        
        if save == True:
            pdf = matplotlib.backends.backend_pdf.PdfPages(name + ".pdf")
            
        for variable in variables:
            
            idx, = np.where(self.varlist_list[frequency] == variable)
            lst = list(self.forecast_draws_list[frequency].T)

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
        
        plt.close(fig)

    def fanchart(self, variables = "all", save = True, name = "Fancharts", show = True, agg = True, nhist = 5):
        """
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
        nhist : int
            number of historical periods that should be shown on the plot
            Default is 5
        Returns
        -------
        None.
        """
        
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
        
        if agg == True:
            if self.frequencies.index(self.agg_freq) == 0:
                history = np.array(self.input_data_Q)
            else:
                history = self.input_data[self.frequencies.index(self.agg_freq)-1] 
            for i in range(self.frequencies.index(self.agg_freq), len(self.freq_ratio_list)):
                if self.input_data[i].size:
                    ratio = int(np.prod(list(itertools.islice(self.freq_ratio_list,self.frequencies.index(self.agg_freq) , i+1))))
                    end = self.input_data[i].shape[0] - self.input_data[i].shape[0]%ratio -1
                    history = np.hstack((np.array(self.input_data[i].iloc[:end,:].groupby(self.input_data[i].iloc[:end,:].index // ratio).mean()), history))
            
            if self.frequencies.index(self.agg_freq) == 0:    
                YY_mean_agg = np.vstack((history, self.YY_mean_agg[history.shape[0]-self.nlags[self.frequencies.index(self.agg_freq)]:,:]))
            else:
                YY_mean_agg = np.vstack((history, self.YY_mean_agg[history.shape[0]-self.nlags[self.frequencies.index(self.agg_freq)]:,:history.shape[1]]))
            
            '''
            history = pd.DataFrame(history)
            history.columns = list(self.varlist_list[-1])
            history = np.array(history[list(variables)])
            '''    
            if save == True:
                pdf = matplotlib.backends.backend_pdf.PdfPages(name + ".pdf")
                
            for variable in variables:
                idx, = np.where(self.varlist_list[-1] == variable)
                
                if idx < YY_mean_agg.shape[1]:
                
                    fig, ax = plt.subplots()
                    ax.fill_between(range(len(YY_mean_agg[-(int(self.H/ratio+nhist)):,idx])-len(self.YY_095_agg[history.shape[0]-self.nlags[0]:,idx]) ,len(YY_mean_agg[-(int(self.H/ratio+nhist)):,idx])), np.squeeze(self.YY_095_agg[history.shape[0]-self.nlags[0]:,idx]), np.squeeze(self.YY_005_agg[history.shape[0]-self.nlags[0]:,idx]), alpha = 0.5, color = "blue")          
                    ax.fill_between(range(len(YY_mean_agg[-(int(self.H/ratio+nhist)):,idx])-len(self.YY_084_agg[history.shape[0]-self.nlags[0]:,idx]) ,len(YY_mean_agg[-(int(self.H/ratio+nhist)):,idx])), np.squeeze(self.YY_084_agg[history.shape[0]-self.nlags[0]:,idx]), np.squeeze(self.YY_016_agg[history.shape[0]-self.nlags[0]:,idx]), alpha = 0.7, color = "blue")  
                    ax.plot(range(len(YY_mean_agg[-(int(self.H/ratio+nhist)):,idx])), np.squeeze(YY_mean_agg[-(int(self.H/ratio+nhist)):,idx]), color = "black", linewidth = 0.5)
                    ax.plot(range(len(YY_mean_agg[-(int(self.H/ratio+nhist)):,idx])-len(self.YY_095_agg[history.shape[0]-self.nlags[0]:,idx]) ,len(YY_mean_agg[-(int(self.H/ratio+nhist)):,idx])), self.YY_mean_agg[history.shape[0]-self.nlags[0]:,idx], color = "red", linewidth = 0.5)
                    #plt.axvline(x=forecast_start,  color='black', ls='--', lw=0.5)
                    title = "Mean and 90% and 68% CI of Forecasts of: " + variable
                    plt.title(title)
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    if save == True:
                        pdf.savefig( fig )
                    if show == True:
                        plt.show()
                else:
                    forecast_start = nhist #self.YY_mean[-1].shape[0] - int(self.H/ratio)
                    YY_mean = np.array(pd.DataFrame(self.YY_mean[-1][-(self.H/ratio+nhist*ratio):,idx]).groupby(pd.DataFrame(self.YY_mean[-1][-(self.H/ratio+nhist*ratio):,idx]).index // ratio).mean())
                    YY_095 = np.array(pd.DataFrame(self.YY_095[-1][-(self.H/ratio+nhist*ratio),idx]).groupby(pd.DataFrame(self.YY_095[-1][-(self.H/ratio+nhist*ratio):,idx]).index // ratio).mean())
                    YY_005 = np.array(pd.DataFrame(self.YY_005[-1][-(self.H/ratio+nhist*ratio):,idx]).groupby(pd.DataFrame(self.YY_005[-1][-(self.H/ratio+nhist*ratio):,idx]).index // ratio).mean())
                    YY_084 = np.array(pd.DataFrame(self.YY_084[-1][-(self.H/ratio+nhist*ratio):,idx]).groupby(pd.DataFrame(self.YY_084[-1][-(self.H/ratio+nhist*ratio):,idx]).index // ratio).mean())
                    YY_016 = np.array(pd.DataFrame(self.YY_016[-1][-(self.H/ratio+nhist*ratio):,idx]).groupby(pd.DataFrame(self.YY_016[-1][-(self.H/ratio+nhist*ratio):,idx]).index // ratio).mean())
                    
                    fig, ax = plt.subplots()
                    ax.fill_between(range(len(np.squeeze(YY_mean[-1][:,idx]))), np.squeeze(YY_095[-1][:,idx]), np.squeeze(YY_005[-1][:,idx]), alpha = 0.5, color = "blue")
                    ax.fill_between(range(len(np.squeeze(YY_mean[-1][:,idx]))), np.squeeze(YY_084[-1][:,idx]), np.squeeze(YY_016[-1][:,idx]), alpha = 0.7, color = "blue")          
                    ax.plot(range(len(np.squeeze(YY_mean[-1][:,idx]))),np.squeeze(YY_mean[-1][:,idx]), color = "red", linewidth = 0.5)
                    plt.axvline(x=forecast_start,  color='black', ls='--', lw=0.5)
                    title = "Mean and 90% and 68% CI of: " + variable
                    plt.title(title)
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    if save == True:
                        pdf.savefig( fig )
                    if show == True:
                        plt.show()
            if save == True:        
                pdf.close()   

        else:
            forecast_start = nhist #self.YY_mean[-1].shape[0] - self.H
            if save == True:
                pdf = matplotlib.backends.backend_pdf.PdfPages(name + ".pdf")
                
            for variable in variables:
                
                idx, = np.where(self.varlist_list[-1] == variable)
                
                fig, ax = plt.subplots()
                ax.fill_between(range(len(np.squeeze(self.YY_mean[-1][-(self.H+nhist):,idx]))), np.squeeze(self.YY_095[-1][-(self.H+nhist):,idx]), np.squeeze(self.YY_005[-1][-(self.H+nhist):,idx]), alpha = 0.5, color = "blue")
                ax.fill_between(range(len(np.squeeze(self.YY_mean[-1][-(self.H+nhist):,idx]))), np.squeeze(self.YY_084[-1][-(self.H+nhist):,idx]), np.squeeze(self.YY_016[-1][-(self.H+nhist):,idx]), alpha = 0.7, color = "blue")          
                ax.plot(range(len(np.squeeze(self.YY_mean[-1][-(self.H+nhist):,idx]))),np.squeeze(self.YY_mean[-1][-(self.H+nhist):,idx]), color = "red", linewidth = 0.5)
                plt.axvline(x=forecast_start,  color='black', ls='--', lw=0.5)
                title = "Mean and 90% and 68% CI of: " + variable
                plt.title(title)
                plt.xlabel('Time')
                plt.ylabel('Value')
                if save == True:
                    pdf.savefig( fig )
                if show == True:
                    plt.show()
            if save == True:
                pdf.close()   
        plt.close(fig)

# %%
