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
                    
    