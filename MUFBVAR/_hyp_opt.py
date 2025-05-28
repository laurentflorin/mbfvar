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

import copy

# for hyperparameter tuning
from bayes_opt import BayesianOptimization

from mango import scheduler, Tuner

from .mufbvar_data import mufbvar_data

#from MUFBVAR.pseudo_inverse.pseudo_inverse import calculate_pseudo_inverse
from .mfbvar_funcs import mdd_, is_explosive
from .cholcov.cholcov_module import cholcovOrEigendecomp
from .inverse.matrix_inversion import invert_matrix



def update_hyperparameters(self, mufbvar_data, pbounds, init_points, n_iter, nsim, var_of_interest = None, temp_agg = 'mean', save = False, name = "hyp.txt"):
    
    '''
    This method uses bayesian optimization to find the hyperparameters with the highest mdd\n
    lambda 1: overall tightness\n
    lambda 2:  scaling down the variance for the coefficients of a distant lag\n
    lambda 3:  number of observations used for obtaining the prior for the covariance matrix of error terms, fixed to 1\n
    lambda 4: . tuning parameter for coefficients for constant\n
    lambda 5:  tuning parameter for the covariance between coefficients\n

    
    Parameters
    ----------
    mufbvar_data : mufbvar_data class object 
        data in the form of a mufbvar_data class object
    pbound : dict
        boundaries for each hyperparameter:\n
        - two frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1\n
        - three frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2\n
        - four frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2, lambda1_3, lambda2_3, lambda4_3, lambda5_3
    init_points : int
        How many steps of random exploration you want to perform
    n_iter : int
        How many steps of bayesian optimization you want to perform
    nsim : int
        number of draws in each MUFBVAR estimation
    var_of_interest: list of names of variables that we are interested in or None
        Only the variables that are in this list get used in every bi frequency var.
        If None all variables get taken into each higher frequency bi frequency var.
    temp_agg : str
        `mean` or `sum` defines the measurement equation
    save : boolean
        True if you want to save the hyperparameters as a txt
    name : str
        path where you want to save the hyperparameters
        
    Returns
    ----------
    
    hyp : list
        list containing the optimized hyperparameters
        

    '''
    
    def estim(mufbvar_data, hyp_list, nsim, var_of_interest, temp_agg):
        
        explosive_counter = 0
        valid_draws = []        
        
        self.nex = 1
        mdd_list = [np.nan] * (len(mufbvar_data.frequencies)-1)
        self.temp_agg = temp_agg
        # data from mufbvar_data

        YMX_list = copy.deepcopy(mufbvar_data.YMX_list)
        YM0_list = copy.deepcopy(mufbvar_data.YM0_list)
        select_m_list = copy.deepcopy(mufbvar_data.select_m_list)
        vars_m_list = copy.deepcopy(mufbvar_data.vars_m_list)
        YMh_list = copy.deepcopy(mufbvar_data.YMh_list)
        index_list = copy.deepcopy(mufbvar_data.index_list)
        frequencies = copy.deepcopy(mufbvar_data.frequencies)
        self.frequencies = frequencies
        YQX_list = copy.deepcopy(mufbvar_data.YQX_list)
        YQ0_list = copy.deepcopy(mufbvar_data.YQ0_list)
        select_q = copy.deepcopy(mufbvar_data.select_q)
        input_data_Q =  copy.deepcopy(mufbvar_data.input_data_Q)
        self.input_data_Q = input_data_Q
        varlist_list = copy.deepcopy(mufbvar_data.varlist_list)
        select_list = copy.deepcopy(mufbvar_data.select_list)
        select_c_list = copy.deepcopy(mufbvar_data.select_c_list)
        Nm_list = copy.deepcopy(mufbvar_data.Nm_list)
        nv_list = copy.deepcopy(mufbvar_data.nv_list)
        Nq_list = copy.deepcopy(mufbvar_data.Nq_list)
        select_list_sep = copy.deepcopy(mufbvar_data.select_list_sep)
        freq_ratio_list = copy.deepcopy(mufbvar_data.freq_ratio_list)
        YQ_list = copy.deepcopy(mufbvar_data.YQ_list)
        Tstar_list = copy.deepcopy(mufbvar_data.Tstar_list)
        T_list = copy.deepcopy(mufbvar_data.T_list)
        YDATA_list = copy.deepcopy(mufbvar_data.YDATA_list)
        YM_list = copy.deepcopy(mufbvar_data.YM_list)
        input_data = copy.deepcopy(mufbvar_data.input_data)
        self.input_data = input_data

        if not(var_of_interest is None):
            idx_var_of_interest = list(filter(
                lambda x: YQX_list[0].columns.tolist()[x] in var_of_interest,
                range(len(YQX_list[0].columns.tolist()))))
    
        nburn = round((self.nburn_perc)*math.ceil(nsim/self.thining))
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
        Sigmap_list.append(np.zeros((math.ceil((nsim)/self.thining),nv_list[0],nv_list[0])))
        Phip_list.append(np.zeros((math.ceil((nsim)/self.thining),int(nv_list[0])*int(p_list[0])+1,int(nv_list[0]))))
        Cons_list.append(np.zeros((math.ceil((nsim)/self.thining),nv_list[0])))
        #lstate_list.append(np.zeros((round((self.nsim)/self.thining),Nq_list[0],int(Tnobs_list[0]))))
        #YYactsim_list.append(np.zeros((round((self.nsim)/self.thining),freq_ratio_list[0]+1,nv_list[0])))
        #XXactsim_list.append(np.zeros((round((self.nsim)/self.thining),int(freq_ratio_list[0])+1,int(nv_list[0])*int(p_list[0])+1)))
        
        At_mat_list.append(np.zeros((int(Tnobs_list[0]), Nq_list[0]*(int(p_list[0])+1))))
        Pt_mat_list.append(np.zeros((int(Tnobs_list[0]), (Nq_list[0]*(int(p_list[0])+1))**2)))
        Atildemat_list.append(np.zeros((nsim, Nq_list[0]*(int(p_list[0])+1))))
        Ptildemat_list.append(np.zeros((nsim, Nq_list[0]*(int(p_list[0])+1),Nq_list[0]*(int(p_list[0])+1))))
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

        if self.temp_agg == "mean":
            LAMBDAs_list.append(np.vstack((np.hstack((np.zeros((Nm_list[0],Nq_list[0])), np.transpose(phi_mq_list[0]))),1/freq_ratio_list[0]*np.hstack((np.tile(np.eye(Nq_list[0]), freq_ratio_list[0]), np.zeros((Nq_list[0],Nq_list[0]*(p_list[0]-(freq_ratio_list[0]-1)))))))))
        
        if self.temp_agg == "sum":
            LAMBDAs_list.append(np.vstack((np.hstack((np.zeros((Nm_list[0],Nq_list[0])), np.transpose(phi_mq_list[0]))),np.hstack((np.tile(np.eye(Nq_list[0]), freq_ratio_list[0]), np.zeros((Nq_list[0],Nq_list[0]*(p_list[0]-(freq_ratio_list[0]-1)))))))))
            
        
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
                
        # Observations in HF
        
        Ym_list.append(YM_list[0][T0_list[0]:nobs_list[0]+T0_list[0],:])
        
        Yq_list.append(YQ_list[0][T0_list[0]:nobs_list[0]+T0_list[0],:])

        # Estimation
        #################
        print(" ", end = '\n')
        print("Multi Frequency BVAR: Estimation", end = '\n')
        print("Frequencies: ", self.frequencies, end = "\n")
        print("Total Number of Draws: ",nsim)

        #Here we start the sample loop, j is the current sample
        #inside the sample loop we need a loop for the MFBVARS: m
        for j in tqdm(range(nsim)):
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
                        if self.temp_agg == "mean":
                            Z2[bb, (ll+1)*Nm_list[m]+ll*Nq_list[m]+bb] = 1/freq_ratio_list[m]
                        if self.temp_agg == "sum":
                            Z2[bb, (ll+1)*Nm_list[m]+ll*Nq_list[m]+bb] = 1
                
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
            
                if (j%self.thining == 0 and m == (len(YMh_list)-1)):
                    if lstate_list:
                        for hh in range(Nq_list[m]):
                            lstate_list[0][int(int((j)/self.thining)), hh, :nobs_list[m]] = At_draw[:, hh]
                            lstate_list[0][int(int((j)/self.thining)), hh, nobs_list[m]:] = AT_draw[1:, Nm_list[m]+hh]
                    else:
                        lstate_list.append(np.zeros((math.ceil((nsim)/self.thining),Nq_list[0],int(Tnobs_list[0]))))
                        for hh in range(Nq_list[m]):
                            lstate_list[0][int(int((j)/self.thining)), hh, :nobs_list[m]] = At_draw[:, hh]
                            lstate_list[0][int(int((j)/self.thining)), hh, nobs_list[m]:] = AT_draw[1:, Nm_list[m]+hh]
                
                nobs_ = np.shape(YY)[0] - T0_list[m]
                spec = np.hstack((nlags_list_[m], T0_list[m], self.nex, nv_list[m], nobs_))
            
            
                # dummy observations and actual observations
                #mdd, YYact, YYdum, XXact, XXdum = mdd_(self.hyp, YY, spec)
                mdd_list[m], YYact, YYdum, XXact, XXdum = mdd_(hyp_list[m], YY, spec)
                
                if (j%self.thining == 0 and m == (len(YMh_list)-1)):
                    if YYactsim_list:
                        YYactsim_list[0][int(int((j)/self.thining)),:,:] = YYact[-(freq_ratio_list[m]+1):,:] 
                        XXactsim_list[0][int(int((j)/self.thining)),:,:] = XXact[-(freq_ratio_list[m]+1):,:]
                    else:
                        YYactsim_list.append(np.zeros((math.ceil((nsim)/self.thining),freq_ratio_list[0]+1,nv_list[0])))
                        XXactsim_list.append(np.zeros((math.ceil((nsim)/self.thining),int(freq_ratio_list[0])+1,int(nv_list[0])*int(p_list[0])+1)))
                        YYactsim_list[0][int(int((j)/self.thining)),:,:] = YYact[-(freq_ratio_list[m]+1):,:] 
                        XXactsim_list[0][int(int((j)/self.thining)),:,:] = XXact[-(freq_ratio_list[m]+1):,:]
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
                
                sigma = invwishart.rvs(scale = Sigma, df = T-n*p-1)
                # Draws from the density Sigma | Y 
                attempts = 0
                while attempts < 1000:
                    sigma_chol = cholcovOrEigendecomp(np.kron(sigma, inv_x))
                    phi_new = np.squeeze(Phi_tilde.reshape(n*(n*p+1), 1, order="F")) + sigma_chol @ np.random.standard_normal(sigma_chol.shape[0])
                    Phi = phi_new.reshape(n*p+1, n, order="F")
                    if not is_explosive(Phi, n, p):
                        break
                    attempts += 1
                if attempts == 1000:
                    explosive_counter += 1
                    print(f"Explosive VAR detected {explosive_counter} times.")
                    m = 0
                    continue
                    
                #while loop bis hier
                
                if j > 0:
                    Phi_list[m] = Phi
                elif j == 0 and m == 0:
                    Phi_list[m] = Phi
                
                if (j % self.thining == 0):
                    j_temp = int(j/self.thining)
                    Sigmap_list[m][j_temp,:,:] = sigma
                    Phip_list[m][j_temp,:,:]   = Phi
                    Cons_list[m][j_temp,:]     = Phi[-1,:]
                    if (m == len(YMh_list)-1):
                        valid_draws.append(j_temp)
                
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
                if self.temp_agg == "mean":
                    LAMBDAs_list[m] = np.vstack((np.hstack((np.zeros((Nm_list[m],Nq_list[m])), np.transpose(phi_mq))),1/freq_ratio_list[m]*np.hstack((np.tile(np.eye(Nq_list[m]), freq_ratio_list[m]), np.zeros((Nq_list[m],Nq_list[m]*(p-(freq_ratio_list[m]-1))))))))
                if self.temp_agg == "sum":
                    LAMBDAs_list[m] = np.vstack((np.hstack((np.zeros((Nm_list[m],Nq_list[m])), np.transpose(phi_mq))),np.hstack((np.tile(np.eye(Nq_list[m]), freq_ratio_list[m]), np.zeros((Nq_list[m],Nq_list[m]*(p-(freq_ratio_list[m]-1))))))))
                    
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
                
                #get relevant high frequency variables, so that the get used in the next var
                
                if not(var_of_interest is None):
                    idx_var_of_interest_m = list(filter(lambda x: YMX_list[m].columns.tolist()[x] in var_of_interest, range(len(YMX_list[m].columns.tolist()))))
                
                if m < (len(YMh_list)-1):
                    if j == 0:
                        #Yq_list.append((np.kron(lstate, np.ones((1,freq_ratio_list[m+1])))).T)
                        #YQ0_list.append(YYact)#YQ0_list.append(YYact[:,-Nq_list[m+1]:])
                        if var_of_interest is None:
                            YQ0_list.append(YYact)
                        else:
                            idx_vars = np.concatenate((np.array(idx_var_of_interest_m) , (YM_list[m].shape[1]+np.array(idx_var_of_interest))))
                            YQ0_list.append(YYact[:,np.int_(idx_vars)].reshape(-1, len(var_of_interest)))
                            #we also need to update nv_list and Nq_lsit
                            nv_list[m + 1] = len(idx_vars) + YM0_list[m+1].shape[1]
                            Nq_list[m + 1] = len(idx_vars)
                        YQ_list.append(np.kron(YQ0_list[m+1], np.ones((freq_ratio_list[m+1],1))))#[np.product(np.array(nlags_list_[:(m+2)])):,:])
                        #Yq_list.append(YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:])
                        if YM_list[m].size:
                            YM_list[m+1] = YM_list[m+1][2*np.product(np.array(nlags_list_[:(m+2)])):,:]
                        else:
                            YM_list[m+1] = YM_list[m+1][2*nlags_list_[(m+1)]:,:]
                        
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
                        if YM_list[m].size:
                            YMh_list[m+1] = YMh_list[m+1][2*np.product(np.array(nlags_list_[:(m+2)]))+int(T0_list[m+1]):-int(freq_ratio_list[m+1]),:]
                        varstxt_list.append(np.hstack((YMX_list[m+1].columns, YQX_list[0].columns)))
                        smpltxt_list.append(YMX_list[m+1].index[int(T0_list[m+1]):])
                        
                        index_NY_list.append(np.isnan(YDATA_list[m+1][nobs_list[m+1]+T0_list[m+1]:Tnobs_list[m+1]+T0_list[m+1],:]).T) # TODO CHECK
                        
                        # Parameter estimation
                        # Matrices for collecting draws from Posterior Density
                        Sigmap_list.append(np.zeros((math.ceil((nsim)/self.thining),nv_list[m+1],nv_list[m+1])))
                        Phip_list.append(np.zeros((math.ceil((nsim)/self.thining),int(nv_list[m+1])*int(p_list[m+1])+1,int(nv_list[m+1]))))
                        Cons_list.append(np.zeros((math.ceil((nsim)/self.thining),nv_list[m+1])))
                        if m == (len(YMh_list)-2):
                            lstate_list.append(np.zeros((math.ceil((nsim)/self.thining),Nq_list[m+1],int(Tnobs_list[m+1]))))
                            YYactsim_list.append(np.zeros((math.ceil((nsim)/self.thining),freq_ratio_list[m+1]+1,nv_list[m+1])))
                            XXactsim_list.append(np.zeros((math.ceil((nsim)/self.thining),int(freq_ratio_list[m+1])+1,int(nv_list[m+1])*int(p_list[m+1])+1)))
                        
                        At_mat_list.append(np.zeros((int(Tnobs_list[m+1]), Nq_list[m+1]*(int(p_list[m+1])+1))))
                        Pt_mat_list.append(np.zeros((int(Tnobs_list[m+1]), (Nq_list[m+1]*(int(p_list[m+1])+1))**2)))
                        Atildemat_list.append(np.zeros((nsim, Nq_list[m+1]*(int(p_list[m+1])+1))))
                        Ptildemat_list.append(np.zeros((nsim, Nq_list[m+1]*(int(p_list[m+1])+1),Nq_list[m+1]*(int(p_list[m+1])+1))))
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

                        
                        if self.temp_agg == "mean":
                            LAMBDAs_list.append(np.vstack((np.hstack((np.zeros((Nm_list[m+1],Nq_list[m+1])), np.transpose(phi_mq_list[m+1]))),1/freq_ratio_list[m+1]*np.hstack((np.tile(np.eye(Nq_list[m+1]), freq_ratio_list[m+1]), np.zeros((Nq_list[m+1],Nq_list[m+1]*(p_list[m+1]-(freq_ratio_list[m+1]-1)))))))))
                        if self.temp_agg == "sum":
                            LAMBDAs_list.append(np.vstack((np.hstack((np.zeros((Nm_list[m+1],Nq_list[m+1])), np.transpose(phi_mq_list[m+1]))),np.hstack((np.tile(np.eye(Nq_list[m+1]), freq_ratio_list[m+1]), np.zeros((Nq_list[m+1],Nq_list[m+1]*(p_list[m+1]-(freq_ratio_list[m+1]-1)))))))))
                            
                                
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
                        Yq_list.append(YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:])
                        
                        
                    else:
                        #Yq_list[m+1] = (np.kron(lstate, np.ones((1,freq_ratio_list[m+1])))).T
                        if var_of_interest is None:
                            YQ0_list[m+1] = YYact
                        else:
                            idx_vars = np.concatenate((np.array(idx_var_of_interest_m) , (YM_list[m].shape[1]+np.array(idx_var_of_interest))))
                            YQ0_list[m+1] = YYact[:,np.int_(idx_vars)].reshape(-1, len(var_of_interest))
                            
                        YQ_list[m+1] =np.kron(YQ0_list[m+1], np.ones((freq_ratio_list[m+1],1)))#[np.product(np.array(nlags_list_[:(m+2)])):,:]
                        
                        Yq_list[m+1] = YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:]#YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:]
                        
                        
                        T_list[m+1] = YQ_list[m+1].shape[0]
                        Tnew_list[m+1] = Tstar_list[m+1]-T_list[m+1]
                        
                    #if self.temp_agg == "mean":    
                    #    YDATA_list[m+1][:T_list[m+1],Nm_list[m+1]:] = YQ_list[m+1]
                    #if sself.temp_agg == "sum":    
                    #    YDATA_list[m+1][:T_list[m+1],Nm_list[m+1]:] = 1/freq_ratio_list[m+1]* YQ_list[m+1]
                    YDATA_list[m+1][:T_list[m+1],Nm_list[m+1]:] = YQ_list[m+1]    
                
                if j == 0:
                    At_draw_list.append(At_draw)
                else:
                    At_draw_list[m] = At_draw
                
                if j == 0:
                    Pmean_list.append(Pmean)
                else:
                    Pmean_list[m] = Pmean
                    
        return mdd_list[-1]
        
    def calc_mdd_1(lambda1_1, lambda2_1, lambda4_1, lambda5_1):
        
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1]]
        mdd = estim(mufbvar_data, hyp_list, nsim, var_of_interest, temp_agg)
        
        return mdd
    
    def calc_mdd_2(lambda1_1, lambda2_1, lambda4_1,
                lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2):
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1],
                    [lambda1_2, lambda2_2, 1, lambda4_2, lambda5_2]]
        mdd = estim(mufbvar_data, hyp_list, nsim, var_of_interest, temp_agg)
        
        return mdd
    
    def calc_mdd_3(lambda1_1, lambda2_1, lambda4_1,
                lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2,
                lambda1_3, lambda2_3, lambda4_3, lambda5_3):
        
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1],
                    [lambda1_2, lambda2_2, 1, lambda4_2, lambda5_2],
                    [lambda1_3, lambda2_3, 1, lambda4_3, lambda5_3]]
        mdd = estim(mufbvar_data, hyp_list, nsim, var_of_interest, temp_agg)
        
        return mdd
        
    if len(mufbvar_data.frequencies)-1 == 1:
        optimizer = BayesianOptimization(
        f= calc_mdd_1,
        pbounds=pbounds,
        verbose= 2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        )
        
        
    if len(mufbvar_data.frequencies)-1 == 2:
        optimizer = BayesianOptimization(
        f= calc_mdd_2,
        pbounds=pbounds,
        verbose= 2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        )
        
    if len(mufbvar_data.frequencies)-1 == 3:
        optimizer = BayesianOptimization(
        f = calc_mdd_3,
        pbounds=pbounds,
        verbose= 2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        )
        
    optimizer.maximize(
    init_points = init_points,
    n_iter = n_iter,
    )
    
    hyp_opt = optimizer.max
    
    sublists = [list(hyp_opt["params"].values())[i:i+4] for i in range(0, len(list(hyp_opt["params"].values())), 4)] 
    hyp = []
    for i in sublists:
        i.insert(2,1)
        hyp.append(i)
    
    
    if save == True:
        with open(name, 'w') as f:
            print(hyp, file=f)
            
    return hyp


def update_hyperparameters_mango(self, mufbvar_data, param_space, init_points, n_iter, nsim, njobs, var_of_interest = None, temp_agg = 'mean', save = False, name = "hyp.txt"):
    
    '''
    This method uses bayesian optimization to find the hyperparameters with the highest mdd\n
    lambda 1: overall tightness\n
    lambda 2:  scaling down the variance for the coefficients of a distant lag\n
    lambda 3:  number of observations used for obtaining the prior for the covariance matrix of error terms, fixed to 1\n
    lambda 4: . tuning parameter for coefficients for constant\n
    lambda 5:  tuning parameter for the covariance between coefficients\n

    
    Parameters
    ----------
    mufbvar_data : mufbvar_data class object 
        data in the form of a mufbvar_data class object
    param_space : dict
        boundaries for each hyperparameter:\n
        - two frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1\n
        - three frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2\n
        - four frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2, lambda1_3, lambda2_3, lambda4_3, lambda5_3
    init_points : int
        How many steps of random exploration you want to perform
    n_iter : int
        How many steps of bayesian optimization you want to perform
    nsim : int
        number of draws in each MUFBVAR estimation
    njobs : int
        number of parallel jobs
    var_of_interest: list of names of variables that we are interested in or None
        Only the variables that are in this list get used in every bi frequency var.
        If None all variables get taken into each higher frequency bi frequency var.
    temp_agg : str
        `mean` or `sum` defines the measurement equation
    save : boolean
        True if you want to save the hyperparameters as a txt
    name : str
        path where you want to save the hyperparameters
        
    Returns
    ----------
    
    hyp : list
        list containing the optimized hyperparameters
        

    '''
    
    def estim(mufbvar_data, hyp_list, nsim, var_of_interest, temp_agg):
        
        explosive_counter = 0
        valid_draws = []        
        
        self.nex = 1
        mdd_list = [np.nan] * (len(mufbvar_data.frequencies)-1)
        self.temp_agg = temp_agg
        # data from mufbvar_data

        YMX_list = copy.deepcopy(mufbvar_data.YMX_list)
        YM0_list = copy.deepcopy(mufbvar_data.YM0_list)
        select_m_list = copy.deepcopy(mufbvar_data.select_m_list)
        vars_m_list = copy.deepcopy(mufbvar_data.vars_m_list)
        YMh_list = copy.deepcopy(mufbvar_data.YMh_list)
        index_list = copy.deepcopy(mufbvar_data.index_list)
        frequencies = copy.deepcopy(mufbvar_data.frequencies)
        self.frequencies = frequencies
        YQX_list = copy.deepcopy(mufbvar_data.YQX_list)
        YQ0_list = copy.deepcopy(mufbvar_data.YQ0_list)
        select_q = copy.deepcopy(mufbvar_data.select_q)
        input_data_Q =  copy.deepcopy(mufbvar_data.input_data_Q)
        self.input_data_Q = input_data_Q
        varlist_list = copy.deepcopy(mufbvar_data.varlist_list)
        select_list = copy.deepcopy(mufbvar_data.select_list)
        select_c_list = copy.deepcopy(mufbvar_data.select_c_list)
        Nm_list = copy.deepcopy(mufbvar_data.Nm_list)
        nv_list = copy.deepcopy(mufbvar_data.nv_list)
        Nq_list = copy.deepcopy(mufbvar_data.Nq_list)
        select_list_sep = copy.deepcopy(mufbvar_data.select_list_sep)
        freq_ratio_list = copy.deepcopy(mufbvar_data.freq_ratio_list)
        YQ_list = copy.deepcopy(mufbvar_data.YQ_list)
        Tstar_list = copy.deepcopy(mufbvar_data.Tstar_list)
        T_list = copy.deepcopy(mufbvar_data.T_list)
        YDATA_list = copy.deepcopy(mufbvar_data.YDATA_list)
        YM_list = copy.deepcopy(mufbvar_data.YM_list)
        input_data = copy.deepcopy(mufbvar_data.input_data)
        self.input_data = input_data

        if not(var_of_interest is None):
            idx_var_of_interest = list(filter(
                lambda x: YQX_list[0].columns.tolist()[x] in var_of_interest,
                range(len(YQX_list[0].columns.tolist()))))
    
        nburn = round((self.nburn_perc)*math.ceil(nsim/self.thining))
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
        Sigmap_list.append(np.zeros((math.ceil((nsim)/self.thining),nv_list[0],nv_list[0])))
        Phip_list.append(np.zeros((math.ceil((nsim)/self.thining),int(nv_list[0])*int(p_list[0])+1,int(nv_list[0]))))
        Cons_list.append(np.zeros((math.ceil((nsim)/self.thining),nv_list[0])))
        #lstate_list.append(np.zeros((round((self.nsim)/self.thining),Nq_list[0],int(Tnobs_list[0]))))
        #YYactsim_list.append(np.zeros((round((self.nsim)/self.thining),freq_ratio_list[0]+1,nv_list[0])))
        #XXactsim_list.append(np.zeros((round((self.nsim)/self.thining),int(freq_ratio_list[0])+1,int(nv_list[0])*int(p_list[0])+1)))
        
        At_mat_list.append(np.zeros((int(Tnobs_list[0]), Nq_list[0]*(int(p_list[0])+1))))
        Pt_mat_list.append(np.zeros((int(Tnobs_list[0]), (Nq_list[0]*(int(p_list[0])+1))**2)))
        Atildemat_list.append(np.zeros((nsim, Nq_list[0]*(int(p_list[0])+1))))
        Ptildemat_list.append(np.zeros((nsim, Nq_list[0]*(int(p_list[0])+1),Nq_list[0]*(int(p_list[0])+1))))
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

        if self.temp_agg == "mean":
            LAMBDAs_list.append(np.vstack((np.hstack((np.zeros((Nm_list[0],Nq_list[0])), np.transpose(phi_mq_list[0]))),1/freq_ratio_list[0]*np.hstack((np.tile(np.eye(Nq_list[0]), freq_ratio_list[0]), np.zeros((Nq_list[0],Nq_list[0]*(p_list[0]-(freq_ratio_list[0]-1)))))))))
        
        if self.temp_agg == "sum":
            LAMBDAs_list.append(np.vstack((np.hstack((np.zeros((Nm_list[0],Nq_list[0])), np.transpose(phi_mq_list[0]))),np.hstack((np.tile(np.eye(Nq_list[0]), freq_ratio_list[0]), np.zeros((Nq_list[0],Nq_list[0]*(p_list[0]-(freq_ratio_list[0]-1)))))))))
            
        
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
                
        # Observations in HF
        
        Ym_list.append(YM_list[0][T0_list[0]:nobs_list[0]+T0_list[0],:])
        
        Yq_list.append(YQ_list[0][T0_list[0]:nobs_list[0]+T0_list[0],:])

        # Estimation
        #################
        print(" ", end = '\n')
        print("Multi Frequency BVAR: Estimation", end = '\n')
        print("Frequencies: ", self.frequencies, end = "\n")
        print("Total Number of Draws: ",nsim)

        #Here we start the sample loop, j is the current sample
        #inside the sample loop we need a loop for the MFBVARS: m
        for j in tqdm(range(nsim)):
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
                        if self.temp_agg == "mean":
                            Z2[bb, (ll+1)*Nm_list[m]+ll*Nq_list[m]+bb] = 1/freq_ratio_list[m]
                        if self.temp_agg == "sum":
                            Z2[bb, (ll+1)*Nm_list[m]+ll*Nq_list[m]+bb] = 1
                
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
            
                if (j%self.thining == 0 and m == (len(YMh_list)-1)):
                    if lstate_list:
                        for hh in range(Nq_list[m]):
                            lstate_list[0][int(int((j)/self.thining)), hh, :nobs_list[m]] = At_draw[:, hh]
                            lstate_list[0][int(int((j)/self.thining)), hh, nobs_list[m]:] = AT_draw[1:, Nm_list[m]+hh]
                    else:
                        lstate_list.append(np.zeros((math.ceil((nsim)/self.thining),Nq_list[0],int(Tnobs_list[0]))))
                        for hh in range(Nq_list[m]):
                            lstate_list[0][int(int((j)/self.thining)), hh, :nobs_list[m]] = At_draw[:, hh]
                            lstate_list[0][int(int((j)/self.thining)), hh, nobs_list[m]:] = AT_draw[1:, Nm_list[m]+hh]
                
                nobs_ = np.shape(YY)[0] - T0_list[m]
                spec = np.hstack((nlags_list_[m], T0_list[m], self.nex, nv_list[m], nobs_))
            
            
                # dummy observations and actual observations
                #mdd, YYact, YYdum, XXact, XXdum = mdd_(self.hyp, YY, spec)
                mdd_list[m], YYact, YYdum, XXact, XXdum = mdd_(hyp_list[m], YY, spec)
                
                if (j%self.thining == 0 and m == (len(YMh_list)-1)):
                    if YYactsim_list:
                        YYactsim_list[0][int(int((j)/self.thining)),:,:] = YYact[-(freq_ratio_list[m]+1):,:] 
                        XXactsim_list[0][int(int((j)/self.thining)),:,:] = XXact[-(freq_ratio_list[m]+1):,:]
                    else:
                        YYactsim_list.append(np.zeros((math.ceil((nsim)/self.thining),freq_ratio_list[0]+1,nv_list[0])))
                        XXactsim_list.append(np.zeros((math.ceil((nsim)/self.thining),int(freq_ratio_list[0])+1,int(nv_list[0])*int(p_list[0])+1)))
                        YYactsim_list[0][int(int((j)/self.thining)),:,:] = YYact[-(freq_ratio_list[m]+1):,:] 
                        XXactsim_list[0][int(int((j)/self.thining)),:,:] = XXact[-(freq_ratio_list[m]+1):,:]
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
                
                sigma = invwishart.rvs(scale = Sigma, df = T-n*p-1)
                # Draws from the density Sigma | Y 
                attempts = 0
                while attempts < 1000:
                    sigma_chol = cholcovOrEigendecomp(np.kron(sigma, inv_x))
                    phi_new = np.squeeze(Phi_tilde.reshape(n*(n*p+1), 1, order="F")) + sigma_chol @ np.random.standard_normal(sigma_chol.shape[0])
                    Phi = phi_new.reshape(n*p+1, n, order="F")
                    if not is_explosive(Phi, n, p):
                        break
                    attempts += 1
                if attempts == 1000:
                    explosive_counter += 1
                    print(f"Explosive VAR detected {explosive_counter} times.")
                    m = 0
                    continue
                    
                #while loop bis hier
                
                if j > 0:
                    Phi_list[m] = Phi
                elif j == 0 and m == 0:
                    Phi_list[m] = Phi
                
                if (j % self.thining == 0):
                    j_temp = int(j/self.thining)
                    Sigmap_list[m][j_temp,:,:] = sigma
                    Phip_list[m][j_temp,:,:]   = Phi
                    Cons_list[m][j_temp,:]     = Phi[-1,:]
                    if (m == len(YMh_list)-1):
                        valid_draws.append(j_temp)
                
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
                if self.temp_agg == "mean":
                    LAMBDAs_list[m] = np.vstack((np.hstack((np.zeros((Nm_list[m],Nq_list[m])), np.transpose(phi_mq))),1/freq_ratio_list[m]*np.hstack((np.tile(np.eye(Nq_list[m]), freq_ratio_list[m]), np.zeros((Nq_list[m],Nq_list[m]*(p-(freq_ratio_list[m]-1))))))))
                if self.temp_agg == "sum":
                    LAMBDAs_list[m] = np.vstack((np.hstack((np.zeros((Nm_list[m],Nq_list[m])), np.transpose(phi_mq))),np.hstack((np.tile(np.eye(Nq_list[m]), freq_ratio_list[m]), np.zeros((Nq_list[m],Nq_list[m]*(p-(freq_ratio_list[m]-1))))))))
                    
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
                
                #get relevant high frequency variables, so that the get used in the next var
                
                if not(var_of_interest is None):
                    idx_var_of_interest_m = list(filter(lambda x: YMX_list[m].columns.tolist()[x] in var_of_interest, range(len(YMX_list[m].columns.tolist()))))
                
                if m < (len(YMh_list)-1):
                    if j == 0:
                        #Yq_list.append((np.kron(lstate, np.ones((1,freq_ratio_list[m+1])))).T)
                        #YQ0_list.append(YYact)#YQ0_list.append(YYact[:,-Nq_list[m+1]:])
                        if var_of_interest is None:
                            YQ0_list.append(YYact)
                        else:
                            idx_vars = np.concatenate((np.array(idx_var_of_interest_m) , (YM_list[m].shape[1]+np.array(idx_var_of_interest))))
                            YQ0_list.append(YYact[:,np.int_(idx_vars)].reshape(-1, len(var_of_interest)))
                            #we also need to update nv_list and Nq_lsit
                            nv_list[m + 1] = len(idx_vars) + YM0_list[m+1].shape[1]
                            Nq_list[m + 1] = len(idx_vars)
                        YQ_list.append(np.kron(YQ0_list[m+1], np.ones((freq_ratio_list[m+1],1))))#[np.product(np.array(nlags_list_[:(m+2)])):,:])
                        #Yq_list.append(YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:])
                        if YM_list[m].size:
                            YM_list[m+1] = YM_list[m+1][2*np.product(np.array(nlags_list_[:(m+2)])):,:]
                        else:
                            YM_list[m+1] = YM_list[m+1][2*nlags_list_[(m+1)]:,:]
                        
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
                        if YM_list[m].size:
                            YMh_list[m+1] = YMh_list[m+1][2*np.product(np.array(nlags_list_[:(m+2)]))+int(T0_list[m+1]):-int(freq_ratio_list[m+1]),:]
                        varstxt_list.append(np.hstack((YMX_list[m+1].columns, YQX_list[0].columns)))
                        smpltxt_list.append(YMX_list[m+1].index[int(T0_list[m+1]):])
                        
                        index_NY_list.append(np.isnan(YDATA_list[m+1][nobs_list[m+1]+T0_list[m+1]:Tnobs_list[m+1]+T0_list[m+1],:]).T) # TODO CHECK
                        
                        # Parameter estimation
                        # Matrices for collecting draws from Posterior Density
                        Sigmap_list.append(np.zeros((math.ceil((nsim)/self.thining),nv_list[m+1],nv_list[m+1])))
                        Phip_list.append(np.zeros((math.ceil((nsim)/self.thining),int(nv_list[m+1])*int(p_list[m+1])+1,int(nv_list[m+1]))))
                        Cons_list.append(np.zeros((math.ceil((nsim)/self.thining),nv_list[m+1])))
                        if m == (len(YMh_list)-2):
                            lstate_list.append(np.zeros((math.ceil((nsim)/self.thining),Nq_list[m+1],int(Tnobs_list[m+1]))))
                            YYactsim_list.append(np.zeros((math.ceil((nsim)/self.thining),freq_ratio_list[m+1]+1,nv_list[m+1])))
                            XXactsim_list.append(np.zeros((math.ceil((nsim)/self.thining),int(freq_ratio_list[m+1])+1,int(nv_list[m+1])*int(p_list[m+1])+1)))
                        
                        At_mat_list.append(np.zeros((int(Tnobs_list[m+1]), Nq_list[m+1]*(int(p_list[m+1])+1))))
                        Pt_mat_list.append(np.zeros((int(Tnobs_list[m+1]), (Nq_list[m+1]*(int(p_list[m+1])+1))**2)))
                        Atildemat_list.append(np.zeros((nsim, Nq_list[m+1]*(int(p_list[m+1])+1))))
                        Ptildemat_list.append(np.zeros((nsim, Nq_list[m+1]*(int(p_list[m+1])+1),Nq_list[m+1]*(int(p_list[m+1])+1))))
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

                        
                        if self.temp_agg == "mean":
                            LAMBDAs_list.append(np.vstack((np.hstack((np.zeros((Nm_list[m+1],Nq_list[m+1])), np.transpose(phi_mq_list[m+1]))),1/freq_ratio_list[m+1]*np.hstack((np.tile(np.eye(Nq_list[m+1]), freq_ratio_list[m+1]), np.zeros((Nq_list[m+1],Nq_list[m+1]*(p_list[m+1]-(freq_ratio_list[m+1]-1)))))))))
                        if self.temp_agg == "sum":
                            LAMBDAs_list.append(np.vstack((np.hstack((np.zeros((Nm_list[m+1],Nq_list[m+1])), np.transpose(phi_mq_list[m+1]))),np.hstack((np.tile(np.eye(Nq_list[m+1]), freq_ratio_list[m+1]), np.zeros((Nq_list[m+1],Nq_list[m+1]*(p_list[m+1]-(freq_ratio_list[m+1]-1)))))))))
                            
                                
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
                        Yq_list.append(YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:])
                        
                        
                    else:
                        #Yq_list[m+1] = (np.kron(lstate, np.ones((1,freq_ratio_list[m+1])))).T
                        if var_of_interest is None:
                            YQ0_list[m+1] = YYact
                        else:
                            idx_vars = np.concatenate((np.array(idx_var_of_interest_m) , (YM_list[m].shape[1]+np.array(idx_var_of_interest))))
                            YQ0_list[m+1] = YYact[:,np.int_(idx_vars)].reshape(-1, len(var_of_interest))
                            
                        YQ_list[m+1] =np.kron(YQ0_list[m+1], np.ones((freq_ratio_list[m+1],1)))#[np.product(np.array(nlags_list_[:(m+2)])):,:]
                        
                        Yq_list[m+1] = YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:]#YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:]
                        
                        
                        T_list[m+1] = YQ_list[m+1].shape[0]
                        Tnew_list[m+1] = Tstar_list[m+1]-T_list[m+1]
                        
                    #if self.temp_agg == "mean":    
                    #    YDATA_list[m+1][:T_list[m+1],Nm_list[m+1]:] = YQ_list[m+1]
                    #if sself.temp_agg == "sum":    
                    #    YDATA_list[m+1][:T_list[m+1],Nm_list[m+1]:] = 1/freq_ratio_list[m+1]* YQ_list[m+1]
                    YDATA_list[m+1][:T_list[m+1],Nm_list[m+1]:] = YQ_list[m+1]    
                
                if j == 0:
                    At_draw_list.append(At_draw)
                else:
                    At_draw_list[m] = At_draw
                
                if j == 0:
                    Pmean_list.append(Pmean)
                else:
                    Pmean_list[m] = Pmean
                    
        return mdd_list[-1]
    
    @scheduler.parallel(n_jobs = njobs)   
    def calc_mdd_1(lambda1_1, lambda2_1, lambda4_1, lambda5_1):
        
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1]]
        mdd = estim(mufbvar_data, hyp_list, nsim, var_of_interest, temp_agg)
        
        return mdd
    
    @scheduler.parallel(n_jobs = njobs)
    def calc_mdd_2(lambda1_1, lambda2_1, lambda4_1,
                lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2):
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1],
                    [lambda1_2, lambda2_2, 1, lambda4_2, lambda5_2]]
        mdd = estim(mufbvar_data, hyp_list, nsim, var_of_interest, temp_agg)
        
        return mdd
    
    @scheduler.parallel(n_jobs = njobs)
    def calc_mdd_3(lambda1_1, lambda2_1, lambda4_1,
                lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2,
                lambda1_3, lambda2_3, lambda4_3, lambda5_3):
        
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1],
                    [lambda1_2, lambda2_2, 1, lambda4_2, lambda5_2],
                    [lambda1_3, lambda2_3, 1, lambda4_3, lambda5_3]]
        mdd = estim(mufbvar_data, hyp_list, nsim, var_of_interest, temp_agg)
        
        return mdd
    
    conf_dict = dict(num_iteration = n_iter, initial_random = init_points)
    
    
    if len(mufbvar_data.frequencies)-1 == 1:
        tuner = Tuner(param_space, calc_mdd_1, conf_dict)
        
    if len(mufbvar_data.frequencies)-1 == 2:
        tuner = Tuner(param_space, calc_mdd_2, conf_dict)
        
        
    if len(mufbvar_data.frequencies)-1 == 3:
        tuner = Tuner(param_space, calc_mdd_3, conf_dict)
        
        
    results = tuner.maximize()
    best_params = results["best_params"]
    if save == True:
        with open(name, 'w') as f:
            print(best_params, file=f)
            
    return best_params


    
def update_hyperparameters_mango_rmse(self, mufbvar_data_in, param_space, H, init_points, n_iter, nsim, njobs, var_of_interest = None, temp_agg = 'mean', save = False, name = "hyp.txt"):
    """
    Use Bayesian optimization to select hyperparameters minimizing out-of-sample RMSE for MUFBVAR models.

    This method tunes hyperparameters (lambdas) for the multifrequency VAR (MUFBVAR) model using Bayesian optimization (via Mango).
    It runs the model over a rolling forecast to evaluate the out-of-sample RMSE for each hyperparameter set, and returns the set with the lowest RMSE.

    Hyperparameters:
        - lambda1: overall tightness
        - lambda2: scaling factor for the variance of distant lags
        - lambda3: number of observations for the prior on the error covariance (fixed to 1)
        - lambda4: tuning for coefficients of the constant
        - lambda5: tuning for covariance between coefficients

    Parameters
    ----------
    mufbvar_data : MUFBVAR.mufbvar_data object
        Holds the input data for multifrequency VAR estimation.
    param_space : dict
        Dictionary with bounds for each hyperparameter, structured according to the number of frequencies:
            - Two frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1
            - Three frequencies: add lambda1_2, lambda2_2, lambda4_2, lambda5_2
            - Four frequencies: add lambda1_3, lambda2_3, lambda4_3, lambda5_3
    H : int
        Forecast horizon in the lowest frequency.
    init_points : int
        Number of initial random exploration steps for Bayesian optimization.
    n_iter : int
        Number of optimization iterations.
    nsim : int
        Number of simulation draws in MUFBVAR estimation.
    njobs : int
        Number of parallel jobs.
    var_of_interest : list of str or None, default None
        List of variable names to consider. If None, all variables are used.
    temp_agg : str, default 'mean'
        Temporal aggregation method ('mean' or 'sum'), defining the measurement equation.
    save : bool, default False
        If True, saves the best hyperparameters to a file.
    name : str, default "hyp.txt"
        Path to file for saving hyperparameters if `save` is True.

    Returns
    -------
    hyp : list
        List of optimized hyperparameters (best set found).
    """
    
    nburn_perc =  self.nburn_perc
    nlags = self.nlags
    thining = self.thining
    
    
    

    def calc_rmse(hyp_list, mufbvar_data_in, H, nsim, var_of_interest, temp_agg, nlags, nburn_perc, thining):
        
        mufbvar_data_temp = copy.deepcopy(mufbvar_data_in)
        horizon_mapping = {f'{mufbvar_data_temp.frequencies[0]}' : H}
        for i, freq  in enumerate(mufbvar_data_temp.frequencies[1:]):
            horizon_mapping.update({f'{freq}' : math.prod(itertools.islice(mufbvar_data_temp.freq_ratio_list,0 ,i+1))})
        
        
        mufbvar_data_temp.input_data.appendleft(mufbvar_data_temp.input_data_Q)
        data_list = list(mufbvar_data_temp.input_data)

        result_in_sample = []
        result_out_sample = []
        for df, freq in zip(data_list, mufbvar_data_temp.frequencies):
            horizon = horizon_mapping.get(freq)
            if len(df) <= horizon:
                raise ValueError(f"DataFrame with frequency {freq} has fewer rows than the required horizon")
            
            # Split the data
            in_sample = df.iloc[:-horizon].copy()
            out_sample = df.iloc[-horizon:].copy()
            
            result_in_sample.append((in_sample))
            result_out_sample.append((out_sample))
            
        data_in = mufbvar_data(result_in_sample, mufbvar_data_temp.trans, mufbvar_data_temp.frequencies)    
        
        model_temp = self.__class__(nsim, nburn_perc, nlags, thining)
        model_temp.fit(data_in, hyp = hyp_list, var_of_interest = var_of_interest,  temp_agg = temp_agg)
        model_temp.forecast(H)
        model_temp.aggregate(frequency = mufbvar_data.frequencies[0])
        
        
        out_sample = result_out_sample[0]
        if (mufbvar_data.frequencies[0] == "Q"):
            out_sample = out_sample.assign(Index = pd.DatetimeIndex(out_sample.index).to_period('Q')).set_index('Index')
            out_sample = out_sample.add_suffix('_out_sample')
        
        df = model_temp.YY_mean_agg[var_of_interest].join(out_sample, how = "inner")
        
        suffix = '_out_sample'
        rmse_results = []

        for col in df.columns:
            if col.endswith(suffix):
                pred_col = col.replace(suffix, '')
                if pred_col in df.columns:
                    rmse = np.sqrt(((df[pred_col] - df[col]) ** 2).mean())
                    rmse_results.append(rmse)
                    
        mean_rmse = np.mean(rmse_results)
        
        return mean_rmse


    #@scheduler.parallel(n_jobs = njobs)   
    def calc_rmse_1(**params):     
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1]]
        rmse = calc_rmse(hyp_list, mufbvar_data_in, H, nsim, var_of_interest, temp_agg, nlags, nburn_perc, thining)
        
        return rmse

    #@scheduler.parallel(n_jobs = njobs)
    def calc_rmse_2(**params):
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1],
                    [lambda1_2, lambda2_2, 1, lambda4_2, lambda5_2]]
        rmse = calc_rmse(hyp_list, mufbvar_data_in, H, nsim, var_of_interest, temp_agg, nlags, nburn_perc, thining)
        
        return rmse

    #@scheduler.parallel(n_jobs = njobs)
    def calc_rmse_3(**params):      
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1],
                    [lambda1_2, lambda2_2, 1, lambda4_2, lambda5_2],
                    [lambda1_3, lambda2_3, 1, lambda4_3, lambda5_3]]
        rmse = calc_rmse(hyp_list, mufbvar_data_in, H, nsim, var_of_interest, temp_agg, nlags, nburn_perc, thining)
        
        return rmse
    
    conf_dict = dict(num_iteration = n_iter, initial_random = init_points)
    
    
    if len(mufbvar_data_in.frequencies)-1 == 1:
        tuner = Tuner(param_space, calc_rmse_1, conf_dict)
        
    if len(mufbvar_data_in.frequencies)-1 == 2:
        tuner = Tuner(param_space, calc_rmse_2, conf_dict)
        
    if len(mufbvar_data_in.frequencies)-1 == 3:
        tuner = Tuner(param_space, calc_rmse_3, conf_dict)
        
        
    results = tuner.minimize()
    best_params = results["best_params"]
    if save == True:
        with open(name, 'w') as f:
            print(best_params, file=f)