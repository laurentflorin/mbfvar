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

# for hyperparameter tuning
from bayes_opt import BayesianOptimization

#%%

class multifrequency_var:
    
    '''
    MUFBVAR class
    
    Parameters
    ----------
    nsim : int
        Number of simulations
    nburn_perc : numeric
        Between 0 and 1, proportion of simulations to throw away as burn in.
    nlags : int
        Number of lags in the highest frequency
    thining : int
        To save only every nth draw


    '''
    
    def __init__(self, nsim, nburn_perc, nlags, thining):
        
        self.nsim = nsim
        self.nburn_perc = nburn_perc
        self.nlags = nlags
        self.thining = thining
        
    # Imported methods
    import ._estimation 
    import ._plots 
    import ._save 