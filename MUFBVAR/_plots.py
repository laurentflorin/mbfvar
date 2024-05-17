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