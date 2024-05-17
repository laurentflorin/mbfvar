# -*- coding: utf-8 -*-
'''
Created on Thu Nov 25 13:51:47 2021

@author: florinl
'''

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

import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default='browser'
import plotly.express as px

#to save objects
import pickle
import copy



#from MUFBVAR.pseudo_inverse.pseudo_inverse import calculate_pseudo_inverse
from .cholcov.cholcov_module import cholcovOrEigendecomp
from .inverse.matrix_inversion import invert_matrix



#%%

class multifrequency_var:
    
    def __init__(self, nsim, nburn_perc, nlags, thining):
        
        '''
        Used to initiate the model.
        
        Parameters
        ----------
        nsim : Numeric
            Number of simulations
        nburn_perc : numeric
            Between 0 and 1, proportion of simulations to throw away as burn in.
        nlags : numeric
            Number of lags in the highest frequency
        thining : Numeric
            To save only every nth draw


        '''
        self.nsim = nsim
        self.nburn_perc = nburn_perc
        self.nlags = nlags
        self.thining = thining
        
        
        
    
    
    def fit(self, mufbvar_data, hyp):
        
        '''
        Estimates the model using the model parameter specified in the initialization. \n
        And the data provided. 
        
        Parameters
        ----------
        mufbvar_data : mufbvar_data class object 
            data in the form of a mufbvar_data class object
        hyp : list
            list containing the hyperparameters\n
            1. overall tightness\n
            2. scaling down the variance for the coefficients of a distant lag\n
            3. number of observations used for obtaining the prior for the covariance matrix of error terms\n
            4. tuning parameter for coefficients for constant\n
            5. tuning parameter for the covariance between coefficients\n

        '''
        
        self.nex = 1
        self.hyp = hyp
        
        # data from mufbvar_data
        
        YMX_list = mufbvar_data.YMX_list
        YM0_list = mufbvar_data.YM0_list
        select_m_list = mufbvar_data.select_m_list
        vars_m_list = mufbvar_data.vars_m_list
        YMh_list = mufbvar_data.YMh_list
        index_list = mufbvar_data.index_list
        frequencies = mufbvar_data.frequencies
        self.frequencies = frequencies
        YQX_list = mufbvar_data.YQX_list
        YQ0_list = mufbvar_data.YQ0_list
        select_q = mufbvar_data.select_q
        input_data_Q =  mufbvar_data.input_data_Q
        self.input_data_Q = input_data_Q
        varlist_list = mufbvar_data.varlist_list
        select_list = mufbvar_data.select_list
        select_c_list = mufbvar_data.select_c_list
        Nm_list = mufbvar_data.Nm_list
        nv_list = mufbvar_data.nv_list
        Nq_list = mufbvar_data.Nq_list
        select_list_sep = mufbvar_data.select_list_sep
        freq_ratio_list = mufbvar_data.freq_ratio_list
        YQ_list = mufbvar_data.YQ_list
        Tstar_list = mufbvar_data.Tstar_list
        T_list = mufbvar_data.T_list
        YDATA_list = mufbvar_data.YDATA_list
        YM_list = mufbvar_data.YM_list
        input_data = mufbvar_data.input_data
        self.input_data = input_data
        
        nburn = round((self.nburn_perc)*self.nsim)
        self.nburn = nburn
        
        #test if nlags for each step is at least frequency ratio
        for i in range(len(freq_ratio_list)):
            if deque(self.nlags)[i] < freq_ratio_list[i]:
                sys.exit("The number of lags at each step must be at least as long as the corresponding frequency ratio")
        
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
        #inside the sample loop we need a loop for the MFBVARS: m
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
                #mdd, YYact, YYdum, XXact, XXdum = mdd_(self.hyp, YY, spec)
                YYact, YYdum, XXact, XXdum = calc_yyact(self.hyp, YY, spec)
                
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
                LAMBDAs_list[m] = np.vstack((np.hstack((np.zeros((Nm_list[m],Nq_list[m])), np.transpose(phi_mq))),1/freq_ratio_list[m]*np.hstack((np.tile(np.eye(Nq_list[m]), freq_ratio_list[m]), np.zeros((Nq_list[m],Nq_list[m]*(p-(freq_ratio_list[m]-1))))))))
                
            
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
                        YQ_list.append(np.kron(YQ0_list[m+1], np.ones((freq_ratio_list[m+1],1))))#[np.product(np.array(nlags_list_[:(m+2)])):,:])
                        #Yq_list.append(YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:])
                        YM_list[m+1] = YM_list[m+1][2*np.product(np.array(nlags_list_[:(m+2)])):,:]
                        
                        T_list.append(YQ_list[m+1].shape[0])
                        
                        Tstar_list.append(YM_list[m+1].shape[0])

                        Tnew_list.append(Tstar_list[m+1]-T_list[m+1])

                        
                        YDATA_list.append(np.full((Tstar_list[m+1],nv_list[m+1]), np.nan))
                        YDATA_list[m+1][:,:Nm_list[m+1]] = YM_list[m+1]#[-Tstar_list[m+1]:,:]
                        
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
                        
                        #YM_short = YM_list[m+1][2*np.product(np.array(nlags_list_[:(m+2)])):,:]
                        
                        # Lagged HF observations
                        Zm_list.append(np.zeros((nobs_list[m+1], Nm_list[m+1]*p_list[m+1])))
                        for k in range(p_list[m+1]):
                            Zm_list[m+1][:, k * Nm_list[m+1]:(k+1)*Nm_list[m+1]] = YM_list[m+1][T0_list[m+1]-(k+1):T0_list[m+1]+nobs_list[m+1]-(k+1),:]
                        # Observations in Monthly Freq
                        
                        Ym_list.append(YM_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:])
                        #Ym_list.append(YM_list[m+1][:nobs_list[m+1]+T0_list[m+1],:])    
                        #Yq_list.append(YQ_list[m+1])#
                        Yq_list.append(YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:])
                        
                        
                        
                    else:
                        #Yq_list[m+1] = (np.kron(lstate, np.ones((1,freq_ratio_list[m+1])))).T
                        YQ0_list[m+1] = YYact
                        YQ_list[m+1] = np.kron(YQ0_list[m+1], np.ones((freq_ratio_list[m+1],1)))#[np.product(np.array(nlags_list_[:(m+2)])):,:]
                        Yq_list[m+1] = YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:]#YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:]
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
            self.YDATA = YDATA_list[-1]
            self.select_q = select_q
            self.select_m = select_m_list[-1]
            self.select = select_list[-1]
            self.varlist = varlist_list[-1]
            self.Nm = Nm_list[-1]
            self.nv = nv_list[-1]
            self.freq_ratio = freq_ratio_list[-1]
            self.T = T_list[-1]
            self.Tstar = Tstar_list[-1]
            self.YQX = YQX_list[-1]
            self.YMX = YMX_list[-1]
            self.YQ = YQ_list[-1]
            self.YM = YM_list[-1]
            self.YMh = YMh_list[-1]
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
        self.index_list = index_list
        
        
        
        
        
    def forecast(self, H, conditionals = None):
        
        '''
        Method to generate the forecasts in the highest frequency.\n
        
        Parameters
        ----------
        H : int
            Forecast horizon in highest frequnecy
        conditionals : pandas DataFrame or None
            Conditional forecasts\n
            column names must be the variable names\n
            no index needed\n
            either values or np.nan
        
        '''
        self.H = H
        # First we need to extend the index
        # depending on the highest frequencies the approach differs
        
        if self.frequencies[-1] == "D":
            
            self.index_list[-1] = self.index_list[-1].append(pd.date_range(start=self.index_list[-1][-1] + Day(), periods=H, freq='D'))

            # Function to check if a month has more than 20 days
            def has_more_than_20_days(month, dti):
                return sum(dti.to_period('M') == month) > 20

            # Function to remove the last day of a month
            def remove_last_day_of_month(month, dti):
                return dti[~((dti.to_period('M') == month) & (dti.day > 20))]

            # Check each month in extended_dti
            for month in self.index_list[-1].to_period('M').unique():
                # If a month has more than 20 days
                while has_more_than_20_days(month, self.index_list[-1]):
                    # Remove the last day of that month
                    self.index_list[-1] = remove_last_day_of_month(month, self.index_list[-1])
                    # Add an additional day at the end
                    self.index_list[-1] = self.index_list[-1].append(pd.DatetimeIndex([self.index_list[-1][-1] + Day()]))
        
        if self.frequencies[-1] == 'W':
            self.index_list[-1] = self.index_list[-1].append(pd.date_range(start=self.index_list[-1][-1] + Week(), periods=H, freq='W-MON'))

            # Function to check if a month has more than 4 weeks
            def has_more_than_4_weeks(month, dti):
                return sum(dti.to_period('M') == month) > 4

            # Function to remove the last week of a month
            def remove_last_week_of_month(month, dti):
                return dti[~((dti.to_period('M') == month) & (dti.day > 28))]

            # Check each month in extended_dti
            for month in self.index_list[-1].to_period('M').unique():
                # If a month has more than 4 weeks
                while has_more_than_4_weeks(month, self.index_list[-1]):
                    # Remove the last week of that month
                    self.index_list[-1] = remove_last_week_of_month(month, self.index_list[-1])
                    # Add an additional week at the end
                    self.index_list[-1] = self.index_list[-1].append(pd.DatetimeIndex([self.index_list[-1][-1] + Week()]))
                    
        if self.frequencies[-1] == 'M':
            self.index_list[-1] = self.index_list[-1].append(pd.date_range(start=self.index_list[-1][-1] + MonthBegin(), periods=12, freq='MS'))
        
        if self.frequencies[-1] == 'Q':
            self.index_list[-1] = self.index_list[-1].append(pd.date_range(start=self.index_list[-1][-1] + QuarterBegin(), periods=H, freq='QS'))
            
        
        # Now we need to look at the conditional forecasts:
        
        YYcond = pd.DataFrame(np.nan, index =  self.index_list[-1][-H:], columns= self.varlist_list[-1])
        
        if conditionals is not None:
            conditionals.index = YYcond.index[:len(conditionals.index)]
            
            YYcond.update(conditionals)
            YYcond = np.array(YYcond) 
            YYcond[:,(self.select_list[-1] == 1)] = YYcond[:,(self.select_list[-1] == 1)]/100
            YYcond[:,(self.select_list[-1] == 0)] = np.log(YYcond[:,(self.select_list[-1] == 0)])
        exc = ~np.isnan(YYcond)
            
        YY_m_list = deque()
        YY_med_list = deque()
        YY_095_list = deque()
        YY_005_list = deque()
        YY_084_list = deque()
        YY_016_list = deque()
        forecast_draws_list = deque()
        
        
        H_= int(self.H)
                
        #Prepare index for output
        self.index_list[-1] = self.index_list[-1][self.index_list[-1].shape[0]-(self.lstate_list[-1].shape[2]+H_):]
    
        ###############  
        # Forecasting #
        ###############
        
        # store forecasts in monthly frequency
        YYvector_ml  = np.zeros((round((self.nsim)/self.thining),H_,self.Nm_list[-1]+self.Nq_list[-1]))     # collects now/forecast      
        
        # store forecasts in quarterly frequency
        YYvector_ql  = np.zeros((round((self.nsim)/self.thining),int(self.H/self.freq_ratio_list[-1]),self.Nm_list[-1]+self.Nq_list[-1]))   
        YYvector_qg  = np.zeros((round((self.nsim)/self.thining),int(self.H/self.freq_ratio_list[-1]),self.Nm_list[-1]+self.Nq_list[-1]))
        
        print(" ", end = '\n')
        print("Multiple Frequency BVAR: Forecasting", end = "\n")
        print("Forecast Horizon: ", H_, end = "\n")
        print("Total Draws: ", self.nsim)
        
        
        for jj in tqdm(range(round((self.nsim)/self.thining))):
            
            YYact = np.squeeze(self.YYactsim_list[-1][jj, -1, :])
            XXact = np.squeeze(self.XXactsim_list[-1][jj, -1, :])
            post_phi = np.squeeze(self.Phip_list[-1][jj,:,:])
            post_sig = np.squeeze(self.Sigmap_list[-1][jj,:,:])
            

            # Bayesian Estimation Forecasting 
            ###################################

            
            YYpred = np.zeros((H_+1, self.nv_list[-1])) # forecasts from VAR
            YYpred[0,:] = YYact
            XXpred = np.zeros((H_+1, self.nv_list[-1]*self.nlags_list[-1]+1))
            XXpred[:,-1] = np.full((H_+1), fill_value = 1)
            XXpred[0,:] = XXact
            
            # given posterior draw, draw number (H+1) random sequence
    
            error_pred = np.zeros((H_+1, self.nv_list[-1]))
            
            for h in range(H_+1):
                if post_sig.size > 1:
                    error_pred[h,:] = np.random.default_rng().multivariate_normal(mean = np.zeros(self.nv_list[-1]), cov = post_sig, method = "cholesky")
                else:
                    error_pred[h,:] = np.random.default_rng().normal(loc = 0, scale = post_sig)
            # given posterior draw, iterate forward to construct forecasts
            
            for h in range(1,H_+1):
                
                XXpred[h,self.nv_list[-1]:-1] = XXpred[h-1, :-self.nv_list[-1]-1]
                XXpred[h, :self.nv_list[-1]] = YYpred[h-1, :]
                #YYpred[h,:] = (XXpred[h,:] @ post_phi + error_pred[h,:])
                YYpred[h,:] = (1-exc[h-1,:]) * (XXpred[h,:] @ post_phi + error_pred[h,:]) + exc[h-1,:] * np.nan_to_num(YYcond[h-1,:])
            
            YYpred1 = copy.deepcopy(YYpred)
            YYpred = YYpred[1:,:]
            

            
            
            # Now-/Forecasts
            # Store in hf
            YYvector_ml[jj,:,:] = YYpred
            
        forecast_draws_list.append(YYvector_ml)
        
        #mean
        YYftr_m = np.nanmean(YYvector_ml[self.nburn:,:,:], axis = 0)
        YYftr_m[:, (self.select_list[-1] == 1)[0]] = 100 * YYftr_m[:, (self.select_list[-1] == 1)[0]]
        YYftr_m[:, (self.select_list[-1] == 0)[0]] = np.exp(YYftr_m[:, (self.select_list[-1] == 0)[0]])
        
        YYnow_m = np.mean(self.YYactsim_list[-1][self.nburn:,1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]], axis = 0) # actual/nowcast monthlies
        if YYnow_m.size:
            YYnow_m[:, (self.select_m_list[-1] == 1)[0]] = 100 * YYnow_m[:, (self.select_m_list[-1] == 1)[0]]
            YYnow_m[:, (self.select_m_list[-1] == 0)[0]] = np.exp(YYnow_m[:,(self.select_m_list[-1] == 0)[0]])
        
        lstate_m = np.mean(self.lstate_list[-1][self.nburn:,:,:], axis = 0).T # hf obs for lf vars
        lstate_m[:, (self.select_q[-1] == 1)[0]] = 100 * lstate_m[:, (self.select_q[-1] == 1)[0]]
        lstate_m[:, (self.select_q[-1] == 0)[0]] = np.exp(lstate_m[:, (self.select_q[-1]== 0)[0]])
        

        
        YMh_len_correction = int(self.YMh_list[-1].shape[0] - lstate_m[:-(self.freq_ratio_list[-1]),:].shape[0])
        
        if self.YMh_list[-1].size:
            self.YMh_list[-1][:, (self.select_m_list[-1] == 1)[0]] = 100 * self.YMh_list[-1][:, (self.select_m_list[-1] == 1)[0]]
            self.YMh_list[-1][:, (self.select_m_list[-1] == 0)[0]] =  np.exp(self.YMh_list[-1][:, (self.select_m_list[-1] == 0)[0]])
        
        if self.YMh_list[-1].size:
            YY_m_list.append(np.vstack((np.vstack((np.hstack((self.YMh_list[-1][YMh_len_correction:,:], lstate_m[:-(self.freq_ratio_list[-1]),:])), np.hstack((YYnow_m, lstate_m[-self.freq_ratio_list[-1]:,:])))), YYftr_m)))
            #YY_m_list.append(np.vstack((np.vstack((np.hstack((self.YMh_list[-1], lstate_m[:-(self.freq_ratio_list[-1]),:])), np.hstack((YYnow_m, lstate_m[-self.freq_ratio_list[-1]:,:])))), YYftr_m)))
        else:
            YY_m_list.append(np.vstack((lstate_m,YYftr_m)))
        
        
        
        
        #self.mean_phi = np.mean(self.Phip, axis = 0)
        
        #median
        YYftr_med = np.nanmedian(YYvector_ml[self.nburn:,:,:], axis = 0)
        YYftr_med[:, (self.select_list[-1] == 1)[0]] = 100 * YYftr_med[:, (self.select_list[-1] == 1)[0]]
        YYftr_med[:, (self.select_list[-1] == 0)[0]] = np.exp(YYftr_med[:, (self.select_list[-1] == 0)[0]])
        
        YYnow_med = np.median(self.YYactsim_list[-1][self.nburn:,1:(self.freq_ratio+1),:self.Nm_list[-1]], axis = 0) # actual/nowcast monthlies
        
        if YYnow_med.size:
            YYnow_med[:, (self.select_m_list[-1] == 1)[0]] = 100 * YYnow_med[:, (self.select_m_list[-1] == 1)[0]]
            YYnow_med[:, (self.select_m_list[-1] == 0)[0]] = np.exp(YYnow_med[:, (self.select_m_list[-1] == 0)[0]])
        
        lstate_med = np.median(self.lstate_list[-1][self.nburn:,:,:], axis = 0).T # hf obs for lf vars
        lstate_med[:, (self.select_q[-1] == 1)[0]] = 100 * lstate_med[:, (self.select_q[-1] == 1)[0]]
        lstate_med[:, (self.select_q[-1] == 0)[0]] = np.exp(lstate_med[:, (self.select_q[-1]== 0)[0]])
        
        if self.YMh_list[-1].size:
            YY_med_list.append(np.vstack((np.vstack((np.hstack((self.YMh_list[-1][YMh_len_correction:,:], lstate_med[:-self.freq_ratio_list[-1],:])), np.hstack((YYnow_med, lstate_m[-self.freq_ratio_list[-1]:,:])))), YYftr_med)))
        else:
            YY_med_list.append(np.vstack((lstate_med,YYftr_med)))
            
        # safe uncertainty
        # 95%
        YYftr_095 = np.nanquantile(YYvector_ml[self.nburn:,:,:], q = 0.95 ,axis = 0)
        YYftr_095[:, (self.select_list[-1] == 1)[0]] = 100 * YYftr_095[:, (self.select_list[-1] == 1)[0]]
        YYftr_095[:, (self.select_list[-1] == 0)[0]] = np.exp(YYftr_095[:, (self.select_list[-1] == 0)[0]])
        
        YYnow_095 = np.quantile(self.YYactsim_list[-1][self.nburn:,1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]], q = 0.95, axis = 0) # actual/nowcast monthlies
        if YYnow_095.size:
            YYnow_095[:, (self.select_m_list[-1] == 1)[0]] = 100 * YYnow_095[:, (self.select_m_list[-1] == 1)[0]]
            YYnow_095[:, (self.select_m_list[-1] == 0)[0]] = np.exp(YYnow_095[:, (self.select_m_list[-1] == 0)[0]])
        
        lstate_095 = np.quantile(self.lstate_list[-1][self.nburn:,:,:], q = 0.95, axis = 0).T # hf obs for lf vars
        lstate_095[:, (self.select_q[-1] == 1)[0]] = 100 * lstate_095[:, (self.select_q[-1] == 1)[0]]
        lstate_095[:, (self.select_q[-1] == 0)[0]] = np.exp(lstate_095[:, (self.select_q[-1] == 0)[0]])
        
        YMna = np.full(self.YMh_list[-1].shape, np.nan)
        
        if YMna.size:
            YY_095_list.append(np.vstack((np.vstack((np.hstack((YMna[YMh_len_correction:,:],lstate_095[:-self.freq_ratio_list[-1],:])), np.hstack((YYnow_095, lstate_095[-self.freq_ratio_list[-1]:,:])))), YYftr_095)))
        else:
            YY_095_list.append(np.vstack((lstate_095,YYftr_095)))
        
        # 84%
        YYftr_084 = np.nanquantile(YYvector_ml[self.nburn:,:,:], q = 0.84 ,axis = 0)
        YYftr_084[:, (self.select_list[-1] == 1)[0]] = 100 * YYftr_084[:, (self.select_list[-1] == 1)[0]]
        YYftr_084[:, (self.select_list[-1] == 0)[0]] = np.exp(YYftr_084[:, (self.select_list[-1] == 0)[0]])
        
        YYnow_084 = np.quantile(self.YYactsim_list[-1][self.nburn:,1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]], q = 0.84, axis = 0) # actual/nowcast monthlies
        if YYnow_084.size:
            YYnow_084[:, (self.select_m_list[-1] == 1)[0]] = 100 * YYnow_084[:, (self.select_m_list[-1] == 1)[0]]
            YYnow_084[:, (self.select_m_list[-1] == 0)[0]] = np.exp(YYnow_084[:, (self.select_m_list[-1] == 0)[0]])
        
        lstate_084 = np.quantile(self.lstate_list[-1][self.nburn:,:,:], q = 0.84, axis = 0).T # hf obs for lf vars
        lstate_084[:, (self.select_q[-1] == 1)[0]] = 100 * lstate_084[:, (self.select_q[-1] == 1)[0]]
        lstate_084[:, (self.select_q[-1] == 0)[0]] = np.exp(lstate_084[:, (self.select_q[-1] == 0)[0]])
        
        YMna = np.full(self.YMh_list[-1].shape, np.nan)
        
        if YMna.size:
            YY_084_list.append(np.vstack((np.vstack((np.hstack((YMna[YMh_len_correction:,:],lstate_084[:-self.freq_ratio_list[-1],:])), np.hstack((YYnow_084, lstate_084[-self.freq_ratio_list[-1]:,:])))), YYftr_084)))
        else:
            YY_084_list.append(np.vstack((lstate_084,YYftr_084)))
            
        # 16%
        
        YYftr_016 = np.nanquantile(YYvector_ml[self.nburn:,:,:], q = 0.16 ,axis = 0)
        YYftr_016[:, (self.select_list[-1] == 1)[0]] = 100 * YYftr_016[:, (self.select_list[-1] == 1)[0]]
        YYftr_016[:, (self.select_list[-1] == 0)[0]] = np.exp(YYftr_016[:, (self.select_list[-1] == 0)[0]])
        
        YYnow_016 = np.quantile(self.YYactsim_list[-1][self.nburn:,1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]], q = 0.16, axis = 0) # actual/nowcast monthlies
        if YYnow_016.size:
            YYnow_016[:, (self.select_m_list[-1] == 1)[0]] = 100 * YYnow_016[:, (self.select_m_list[-1] == 1)[0]]
            YYnow_016[:, (self.select_m_list[-1] == 0)[0]] = np.exp(YYnow_016[:, (self.select_m_list[-1] == 0)[0]])
        
        lstate_016 = np.quantile(self.lstate_list[-1][self.nburn:,:,:], q = 0.16, axis = 0).T # hf obs for lf vars
        lstate_016[:, (self.select_q[-1] == 1)[0]] = 100 * lstate_016[:, (self.select_q[-1] == 1)[0]]
        lstate_016[:, (self.select_q[-1] == 0)[0]] = np.exp(lstate_016[:, (self.select_q[-1] == 0)[0]])
        
        YMna = np.full(self.YMh_list[-1].shape, np.nan)
        
        if YMna.size:
            YY_016_list.append(np.vstack((np.vstack((np.hstack((YMna[YMh_len_correction:,:],lstate_016[:-self.freq_ratio_list[-1],:])), np.hstack((YYnow_016, lstate_016[-self.freq_ratio_list[-1]:,:])))), YYftr_016)))
        else:
            YY_016_list.append(np.vstack((lstate_016,YYftr_016)))
        
        
        # 5%    
        
        YYftr_005 = np.nanquantile(YYvector_ml[self.nburn:,:,:], q = 0.05 ,axis = 0)
        YYftr_005[:, (self.select_list[-1] == 1)[0]] = 100 * YYftr_005[:, (self.select_list[-1] == 1)[0]]
        YYftr_005[:, (self.select_list[-1] == 0)[0]] = np.exp(YYftr_005[:, (self.select_list[-1] == 0)[0]])
        
        YYnow_005 = np.quantile(self.YYactsim_list[-1][self.nburn:,1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]], q = 0.05, axis = 0) # actual/nowcast monthlies
        
        if YYnow_005.size:   
            YYnow_005[:, (self.select_m_list[-1] == 1)[0]] = 100 * YYnow_005[:, (self.select_m_list[-1] == 1)[0]]
            YYnow_005[:, (self.select_m_list[-1] == 0)[0]] = np.exp(YYnow_005[:, (self.select_m_list[-1] == 0)[0]])
        
        lstate_005 = np.quantile(self.lstate_list[-1][self.nburn:,:,:], q = 0.05, axis = 0).T # hf obs for lf vars
        lstate_005[:, (self.select_q[-1] == 1)[0]] = 100 * lstate_005[:, (self.select_q[-1] == 1)[0]]
        lstate_005[:, (self.select_q[-1] == 0)[0]] = np.exp(lstate_005[:, (self.select_q[-1] == 0)[0]])
        
        if YMna.size:
            YY_005_list.append(np.vstack((np.vstack((np.hstack((YMna[YMh_len_correction:,:], lstate_005[:-self.freq_ratio_list[-1],:])), np.hstack((YYnow_005, lstate_005[-self.freq_ratio_list[-1]:,:])))), YYftr_005)))
        else:
            YY_005_list.append(np.vstack((lstate_005,YYftr_005)))
                
        YY_mean_pd = pd.DataFrame(YY_m_list[-1], columns = self.varlist_list[-1])
        YY_mean_pd.index = self.index_list[-1]
        
        YY_median_pd = pd.DataFrame(YY_med_list[-1], columns = self.varlist_list[-1])
        YY_median_pd.index = self.index_list[-1]
        
        YY_095_pd = pd.DataFrame(YY_095_list[-1], columns = self.varlist_list[-1])
        YY_095_pd.index = self.index_list[-1]
        
        YY_005_pd = pd.DataFrame(YY_005_list[-1], columns = self.varlist_list[-1])
        YY_005_pd.index = self.index_list[-1]
        
        YY_084_pd = pd.DataFrame(YY_084_list[-1], columns = self.varlist_list[-1])
        YY_084_pd.index = self.index_list[-1]
        
        YY_016_pd = pd.DataFrame(YY_016_list[-1], columns = self.varlist_list[-1])
        YY_016_pd.index = self.index_list[-1]
    
        self.YY_095 = YY_095_list
        self.YY_084 = YY_084_list
        self.YY_016 = YY_016_list
        self.YY_005 = YY_005_list
        self.YY_mean = YY_m_list
        self.YY_median = YY_med_list
        self.forecast_draws_list = forecast_draws_list
        
        self.YY_mean_pd = YY_mean_pd
        self.YY_median_pd = YY_median_pd
        self.YY_095_pd = YY_095_pd
        self.YY_005_pd = YY_005_pd
        self.YY_084_pd = YY_084_pd
        self.YY_016_pd = YY_016_pd
        
        
    def aggregate(self, frequency, reset_index = True):
        
        '''
        Aggregates the Mean, Median and quantililes in the highest frequency to the desired frequency. \n
        The Function ensures, that we start at the beginning of a Year or Quarter depending on the chosen frequency \n
        
        Parameters
        ----------
        frequency : str
            The frequency to which the data should be aggregated to
        reset_index : boolean
            Schould index be changed to period Index

        '''
        if self.forecast_draws_list is None :
                sys.exit("Error: To gaggregate generate forecasts first")
                
        if frequency not in ["Y","Q"] :
                sys.exit("Error: Aggregation currently only implemented for aggregation to yearly and quarterly frequency")
        freq_lf = frequency
        freq_hf = self.frequencies[-1]
        
        YY_full_list = deque()
        YY_full_list_agg = deque()
        
        YYnow = copy.copy(self.YYactsim_list[-1])# actual/nowcast
        lstate = copy.copy(self.lstate_list[-1]) # hf obs for lf vars
        
        
        YMh_len_correction = int(self.YMh_list[-1].shape[0] - lstate[0][:,:-(self.freq_ratio_list[-1])].shape[1])
        
        if self.YMh_list[-1].size:
            for i in range(self.nsim):
                lstate_temp = lstate[i].T
                lstate_temp[:, (self.select_q[-1] == 1)[0]] = 100 * lstate_temp[:, (self.select_q[-1] == 1)[0]]
                lstate_temp[:, (self.select_q[-1] == 0)[0]] = np.exp(lstate_temp[:, (self.select_q[-1]== 0)[0]])
                YYnow_temp = YYnow[i][1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]]
                YYnow_temp[:, (self.select_m_list[-1] == 1)[0]] = 100 * YYnow_temp[:, (self.select_m_list[-1] == 1)[0]]
                YYnow_temp[:, (self.select_m_list[-1] == 0)[0]] = np.exp(YYnow_temp[:, (self.select_m_list[-1] == 0)[0]])
                forecast_draws_temp = copy.copy(self.forecast_draws_list[-1][i,:,:])
                forecast_draws_temp[:, (self.select_list[-1] == 1)[0]] = 100 * forecast_draws_temp[:, (self.select_list[-1] == 1)[0]]
                forecast_draws_temp[:, (self.select_list[-1] == 0)[0]] = np.exp(forecast_draws_temp[:, (self.select_list[-1] == 0)[0]])

                temp = np.vstack((np.vstack((np.hstack((self.YMh_list[-1][YMh_len_correction:,:], lstate_temp[:-(self.freq_ratio_list[-1]), :])), np.hstack((YYnow_temp, lstate_temp[-self.freq_ratio_list[-1]:,:])))), forecast_draws_temp))
                temp = pd.DataFrame(temp, columns = self.varlist_list[-1])
                temp.index = self.index_list[-1]
                YY_full_list.append(temp)
        else:
            for i in range(self.nsim):
                temp = np.vstack((lstate[i,:,:],self.forecast_draws_list[i,:,:]))
                temp = pd.DataFrame(temp, columns = self.varlist_list[-1])
                temp.index = self.index_list[-1]
                YY_full_list.append(temp)
        
        
        def find_first_position(arr, numbers, count):
            for i in range(len(arr) - count + 1):
                if arr[i] in numbers and all(arr[i] == arr[j] for j in range(i+1, i+count)):
                    return i
        
        
                
        def agg_helper(freq_lf, freq_hf, df):
            # Set the frequency ratio        
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
                    
            if frequency == 'Q':
                if freq_hf == 'W':
                    start = find_first_position(df.index.month, [1, 4, 7, 10], 4 )
                elif freq_hf == 'M':
                    start = find_first_position(df.index.month, [1, 4, 7, 10], 1)
                elif freq_hf == 'D':
                    start = find_first_position(df.index.month, [1, 4, 7, 10], 20)
            elif frequency == 'Y':
                if freq_hf == 'Q':
                    start = find_first_position(df.index.month, [1,3], 1 )
                elif freq_hf == 'M':
                    start = find_first_position(df.index.month, [1], 1)
                elif freq_hf == 'W':
                    start = find_first_position(df.index.month, [1], 4)
                elif freq_hf == 'D':
                    start = find_first_position(df.index.month, [1], 20)
            return freq_ratio, start
        
        
        freq_ratio, start = agg_helper(freq_lf, freq_hf, YY_full_list[0])
        print("Aggregating for each draw")
        for i in tqdm(range(self.nburn, self.nsim)):
            temp = YY_full_list[i].iloc[start:,].groupby(YY_full_list[i].iloc[start:,].reset_index().index // freq_ratio).filter(lambda x: len(x) == freq_ratio)
            temp = temp.groupby(temp.reset_index().index // freq_ratio).mean()
            temp.index = YY_full_list[i].iloc[start:,].index[::freq_ratio][:temp.shape[0]]
            temp.index = temp.index.map(lambda x: x.replace(day=1))
            YY_full_list_agg.append(temp)
            
        self.YY_mean_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).mean()
        self.YY_median_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).median()
        self.YY_095_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).quantile(0.95)
        self.YY_005_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).quantile(0.05)
        self.YY_084_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).quantile(0.84)
        self.YY_016_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).quantile(0.16)
        
        hist = copy.copy(self.input_data)
        hist.appendleft(self.input_data_Q)
        i = 0
        for m in self.frequencies:
            if self.frequencies.index(m) == self.frequencies.index(frequency):
                # history ersetzen durch originaldaten 
                # interval durch NA
                idx = self.YY_mean_agg.index.intersection(hist[i].index, sort=False)
                self.YY_mean_agg.loc[idx, hist[i].columns] = hist[i].loc[idx, hist[i].columns]
                self.YY_median_agg.loc[idx, hist[i].columns] = hist[i].loc[idx, hist[i].columns]
                self.YY_095_agg.loc[idx, hist[i].columns] = np.nan
                self.YY_005_agg.loc[idx, hist[i].columns] = np.nan
                self.YY_084_agg.loc[idx, hist[i].columns] = np.nan
                self.YY_016_agg.loc[idx, hist[i].columns] = np.nan
                
            if self.frequencies.index(m) > self.frequencies.index(frequency) & self.frequencies.index(m) < len(self.frequencies)-1:
                freq_ratio_temp, start_temp = agg_helper(frequency, m, hist[i])
                hist_agg = hist[i].iloc[start_temp:,].groupby(hist[i].iloc[start_temp:,].reset_index().index // freq_ratio_temp).filter(lambda x: len(x) == freq_ratio_temp)
                hist_agg = hist_agg.groupby(hist_agg.reset_index().index // freq_ratio_temp).mean()
                hist_agg.index = hist[i].iloc[start_temp:,].index[::freq_ratio_temp][:hist_agg.shape[0]]
                hist_agg.index = hist_agg.index.map(lambda x: x.replace(day=1))
                
                idx = hist_agg.index.intersection(self.YY_mean_agg.index, sort=False)
                self.YY_mean_agg.loc[idx, hist[i].columns] = hist_agg.loc[idx, hist[i].columns]
                self.YY_median_agg.loc[idx, hist[i].columns] = hist_agg.loc[idx, hist[i].columns]
                self.YY_095_agg.loc[idx, hist[i].columns] = np.nan
                self.YY_005_agg.loc[idx, hist[i].columns] = np.nan
                self.YY_084_agg.loc[idx, hist[i].columns] = np.nan
                self.YY_016_agg.loc[idx, hist[i].columns] = np.nan
            
            i = i+1
        
        if reset_index == True:
            index_new = pd.PeriodIndex(self.YY_mean_agg.index, freq= frequency)
            self.YY_mean_agg.index = index_new
            self.YY_median_agg.index = index_new
            self.YY_095_agg.index = index_new
            self.YY_005_agg.index = index_new
            self.YY_084_agg.index = index_new
            self.YY_016_agg.index = index_new
        
        self.agg_freq = frequency
    
    def save(self, filename = "mufbvar_model.pkl"):
        
        '''
        Saves the MFBVAR Object
        
        Parameters
        ----------
        filename : str
            Path where to save the object. End must be .pkl
        
        '''
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
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
        
        if self.forecast_draws_list is None :
                sys.exit("Error: To generate traceplots, generate forecasts first")
                
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
            
            idx, = np.where(self.varlist_list[-1] == variableS)
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
                
                self.YY_095_pd
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


# %%
