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
    from ._estimation import fit, forecast, aggregate
    from ._plots import fanchart, mean_plot
    from ._save import to_excel, save 