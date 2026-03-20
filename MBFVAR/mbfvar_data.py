import os
import sys

import pandas as pd
import numpy as np

from collections import deque
import copy
import itertools

    
class mbfvar_data:
    """
    Prepare multi-frequency time series data for MBFVAR estimation.

    This class handles the preprocessing and organization of time series data at
    different frequencies (e.g., quarterly, monthly, weekly) into a format suitable
    for the multi-frequency Bayesian VAR model.

    Parameters
    ----------
    data : list of pandas.DataFrame
        List of DataFrames, one for each frequency level, ordered from lowest to
        highest frequency. Each DataFrame should have:
        - Time series data with variables as columns
        - A datetime index or index that can be converted to datetime
        - Variables named consistently for identification

        Example: [quarterly_df, monthly_df, weekly_df]

    trans : list of numpy.ndarray
        List of transformation arrays, one for each frequency level. Each array
        specifies the transformation for each variable in that frequency:
        - 0: Take natural logarithm of the variable
        - 1: Divide the variable by 100 (for percentages/rates)

        Length of each array must match the number of variables in the corresponding
        DataFrame in `data`.

        Example: [np.array([1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])]

    frequencies : list of str
        List of frequency identifiers ordered from lowest to highest.
        Supported frequencies:
        - "Y": Yearly
        - "Q": Quarterly
        - "M": Monthly
        - "W": Weekly
        - "D": Daily

        The frequency ratios are automatically calculated based on standard
        calendar conventions (e.g., Q to M is 3, M to W is 4).

        Example: ["Q", "M", "W"]

    freq_offsets : list of int, optional
        Starting offsets for each frequency in base frequency units.
        - Length must equal len(frequencies)
        - First element must be 0 (base frequency has no offset)
        - Subsequent elements specify how many base frequency periods later
          each higher frequency starts
        - Default: None (equivalent to all zeros - all frequencies start together)

        Example: For frequencies=["Q", "M", "W"], freq_offsets=[0, 20, 32]
                means monthly data starts 20 quarters later, weekly starts 32 quarters later.

    Attributes
    ----------
    YMX_list : collections.deque
        Original high-frequency data (all except lowest frequency)
    YM0_list : collections.deque
        Transformed high-frequency data as numpy arrays
    YQX_list : collections.deque
        Original low-frequency data (lowest frequency only)
    YQ0_list : collections.deque
        Transformed low-frequency data as numpy arrays
    frequencies : list of str
        The frequency identifiers passed during initialization
    freq_ratio_list : collections.deque
        Calculated frequency ratios between consecutive frequency levels
    freq_offsets : list of int
        Starting offsets for each frequency in base frequency units
    freq_offsets_hf : list of int
        Starting offsets converted to highest frequency units
    varlist_list : collections.deque
        Variable names for each frequency combination
    YDATA_list : collections.deque
        Combined data matrices ready for model estimation

    Notes
    -----
    - The data transformations are applied automatically during initialization
    - For unsupported frequency combinations, you will be prompted to enter the
      frequency ratio manually
    - The class prepares the data for disaggregation from low to high frequency
      following the methodology in Schorfheide and Song (2015)

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import MBFVAR
    >>>
    >>> # Load data at different frequencies
    >>> data_q = pd.read_excel("data.xlsx", sheet_name="Q", index_col=0)
    >>> data_m = pd.read_excel("data.xlsx", sheet_name="M", index_col=0)
    >>> data_w = pd.read_excel("data.xlsx", sheet_name="W", index_col=0)
    >>>
    >>> # Specify transformations (1 = divide by 100 for all variables)
    >>> trans = [
    ...     np.array([1, 1]),       # 2 quarterly variables
    ...     np.array([1, 1, 1]),    # 3 monthly variables
    ...     np.array([1, 1, 1])     # 3 weekly variables
    ... ]
    >>>
    >>> # Create data object
    >>> data_in = MBFVAR.mbfvar_data(
    ...     data=[data_q, data_m, data_w],
    ...     trans=trans,
    ...     frequencies=["Q", "M", "W"]
    ... )

    References
    ----------
    Schorfheide, F., & Song, D. (2015). Real-time forecasting with a mixed-frequency
    VAR. Journal of Business & Economic Statistics, 33(3), 366-380.
    """
    
    def __init__(self, data, trans, frequencies, freq_offsets=None):
        """
        Initialize the mbfvar_data object.

        Parameters
        ----------
        freq_offsets : list of int, optional
            Starting offsets for each frequency in base frequency units.
            - Length must equal len(frequencies)
            - First element should be 0 (base frequency has no offset)
            - Subsequent elements specify how many base frequency periods later
              each higher frequency starts
            - Default: None (equivalent to all zeros - all frequencies start together)

            Example: For frequencies=["Q", "M", "W"], freq_offsets=[0, 20, 32]
                    means monthly data starts 20 quarters later, weekly starts 32 quarters later.
        """

        # Initialize and validate freq_offsets
        if freq_offsets is None:
            # Default behavior: all frequencies start at the same time
            freq_offsets = [0] * len(frequencies)
        else:
            # Validate freq_offsets
            if len(freq_offsets) != len(frequencies):
                raise ValueError(
                    f"freq_offsets must have length {len(frequencies)} to match frequencies, "
                    f"but got length {len(freq_offsets)}"
                )
            if freq_offsets[0] != 0:
                raise ValueError(
                    "First element of freq_offsets must be 0 (base frequency cannot have offset)"
                )
            if any(offset < 0 for offset in freq_offsets):
                raise ValueError("All offsets in freq_offsets must be non-negative")
            if not all(isinstance(offset, (int, np.integer)) for offset in freq_offsets):
                raise ValueError("All offsets in freq_offsets must be integers")

        self.freq_offsets = list(freq_offsets)  # Store as list

        # Creating lists of highfrequency data
        
        YMX_list = deque()
        YM0_list = deque()
        select_m_list = deque()
        vars_m_list = deque()
        YMh_list = deque()
        exc_list = deque()
        index_list = deque()
        
        
        for i in range(1, len(frequencies)):
            YMX_list.append(data[i])
            YM0_list.append(data[i].to_numpy(copy=True))
            select_m_list.append(trans[i])
            vars_m_list.append(data[i].columns[:])
            YMh_list.append(data[i].to_numpy())
            index_list.append(data[i].index)
        
        if not isinstance(index_list[-1], pd.DatetimeIndex):
            try:
                index_list[-1] = pd.to_datetime(index_list[-1])
            except:
                print("Index must be of the form 'YYYY-MM-DD'")
        
        
        input_data = copy.deepcopy(YMX_list)
        
        # Creating list of low frequency data
        # Here only the first entry with the lowest frequency data is generated
        # The other entries are generated during the estimation
        
        YQX_list = deque()
        YQ0_list = deque()
        select_q = deque()
        
        YQX_list.append(data[0])
        # copy=True ensures array is writable for in-place transformations
        YQ0_list.append(YQX_list[-1].to_numpy(copy=True))
        select_q.append(trans[0])
        vars_q = YQX_list[0].columns[:]
        
        input_data_Q =  copy.deepcopy(YQX_list[0])
        
        
        varlist_list = deque()
        select_list = deque()
        select_c_list =deque()
        
        Nm_list = deque()
        nv_list = deque()
        Nq_list = deque()
        
        Nq_list.append(YQX_list[0].shape[1])
        
        def safe_len(arr):
            arr = np.asarray(arr)
            if arr.shape == ():  # scalar
                return 0
            return len(arr)

        for i in range(len(YMX_list)-1):
            Nq_list.append(YQX_list[0].shape[1] + YMX_list[i].shape[1]) 
            if safe_len(select_m_list[i]) == 0:
                select_q.append( select_q[0])
            else:
                select_q.append(np.hstack((select_m_list[i], select_q[0])))
                
                
        for i in range(len(YMX_list)):
            if i > 0:
                new_list = [item for item in list(itertools.islice(select_m_list, 0, i+1)) if safe_len(item) > 0]
                select_list.append(np.hstack((np.hstack(list(reversed(new_list))), select_q[0])))
                rev_vars_m = list(itertools.islice(vars_m_list, 0, i+1))
                rev_vars_m.reverse()
                varlist_list.append(np.squeeze(np.hstack((np.hstack(rev_vars_m), vars_q)))) 
            else:
                if select_m_list[i].size:
                    select_list.append(np.hstack((select_m_list[i], select_q[0])))
                    varlist_list.append(np.hstack((vars_m_list[i], vars_q)))
                else:
                    select_list.append(select_q[0])
                    if len(vars_q.shape) > 1:
                        varlist_list.append(np.squeeze(vars_q))
                    else:
                        varlist_list.append(vars_q)
            Nm_list.append(int(np.shape(YM0_list[i])[1]))
            nv_list.append(int(Nm_list[i] + Nq_list[i]))
        
        select_list_sep = list(select_q)
        select_list_sep.extend(select_m_list)
        
        
        # Calculate the frequency ratios
        
        freq_ratio_list = deque()
        
        for freq in range(1,len(frequencies)):
            freq_lf = frequencies[freq-1]
            freq_hf = frequencies[freq]
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


        # Calculate offsets in highest frequency units
        # freq_offsets are in base frequency (lowest frequency) units
        # We need to convert them to the highest frequency units

        freq_offsets_hf = [0] * len(frequencies)  # Initialize

        # Calculate cumulative frequency ratios from base to each level
        cumulative_ratios = [1]  # Base frequency ratio to itself is 1
        for i in range(len(freq_ratio_list)):
            cumulative_ratios.append(cumulative_ratios[-1] * freq_ratio_list[i])

        # Convert offsets to highest frequency units
        # The highest frequency is the last one, so its cumulative ratio is the product of all ratios
        total_ratio_to_highest = cumulative_ratios[-1]

        for i in range(len(frequencies)):
            # offset in base frequency units * ratio from base to highest frequency
            freq_offsets_hf[i] = self.freq_offsets[i] * total_ratio_to_highest

        # Validate that offsets don't exceed data availability
        for i in range(len(frequencies)):
            available_obs = data[i].shape[0]
            # Convert offset from highest frequency back to current frequency
            offset_in_current_freq = self.freq_offsets[i] * cumulative_ratios[i]

            if offset_in_current_freq >= available_obs:
                raise ValueError(
                    f"Offset for frequency {frequencies[i]} (position {i}) requires "
                    f"{offset_in_current_freq} observations in that frequency, "
                    f"but only {available_obs} observations are available. "
                    f"Reduce the offset or provide more data."
                )

        # Store the calculated values
        self.freq_offsets_hf = freq_offsets_hf
        self.cumulative_ratios = cumulative_ratios


        #performe data transformations
        
        for i in range(len(YM0_list)):
            if select_m_list[i].size:
                YM0_list[i][:,(select_m_list[i] == 1)] = YM0_list[i][:,(select_m_list[i] == 1)]/100
                YM0_list[i][:,(select_m_list[i] == 0)] = np.log(YM0_list[i][:,(select_m_list[i] == 0)])
        
        YQ0_list[0][:,(select_q[0] == 1)] = YQ0_list[0][:,(select_q[0] == 1)]/100
        YQ0_list[0][:,(select_q[0] == 0)] = np.log(YQ0_list[0][:,(select_q[0] == 0)])
        
        
        YM_list = copy.deepcopy(YM0_list)

        # Low frequency data in higher frequency
        # With offsets, we need to handle the case where LF data starts later than HF data
        YQ_list = deque()

        # Expand low-frequency data using Kronecker product
        YQ_expanded = np.kron(YQ0_list[0], np.ones((freq_ratio_list[0], 1)))

        # Apply offset if higher frequency starts later than base frequency
        # Note: In the typical case, higher frequencies start LATER, so offset_hf[1] > 0
        # means the first HF data (monthly if base is quarterly) starts offset_hf[1] periods later
        # in the highest frequency time grid

        if len(freq_offsets_hf) > 1 and freq_offsets_hf[1] > 0:
            # The first higher frequency (YM_list[0]) starts later
            # We need to pad the beginning of YQ_expanded with NaN
            offset_hf = int(freq_offsets_hf[1])
            padding = np.full((offset_hf, YQ_expanded.shape[1]), np.nan)
            YQ_list.append(np.vstack((padding, YQ_expanded)))
        else:
            YQ_list.append(YQ_expanded)
        
        Tstar_list = deque()
        T_list = deque()
        YDATA_list = deque()

        # Apply offset padding to YM_list if the first higher frequency starts later
        if len(freq_offsets_hf) > 1 and freq_offsets_hf[1] > 0:
            offset_hf = int(freq_offsets_hf[1])
            if YM_list[0].size:
                # Pad the beginning of YM_list[0] with NaN rows
                padding = np.full((offset_hf, YM_list[0].shape[1]), np.nan)
                YM_list[0] = np.vstack((padding, YM_list[0]))

        if YM_list[0].size:
            Tstar_list.append(YM_list[0].shape[0])
            # YDATA should be the size of the longest series (max of YM and YQ)
            max_T = max(YM_list[0].shape[0], YQ_list[0].shape[0])
            YDATA_list.append(np.full((max_T, nv_list[0]), np.nan))
            # Assign YM data
            YDATA_list[0][:YM_list[0].shape[0], :Nm_list[0]] = YM_list[0]
        else:
            Tstar_list.append(YQ_list[0].shape[0])
            YDATA_list.append(np.full((YQ_list[0].shape[0], nv_list[0]), np.nan))

        T_list.append(YQ_list[0].shape[0])

        # Assign YQ data - handle different lengths
        if YDATA_list[0].size:
            yq_rows = min(YQ_list[0].shape[0], YDATA_list[0].shape[0])
            YDATA_list[0][:yq_rows, Nm_list[0]:] = YQ_list[0][:yq_rows, :]
        else:
            YDATA_list[0] = YQ_list[0] 
            
        
        #attach variables to instance
        
        self.YMX_list = YMX_list
        self.YM0_list = YM0_list
        self.select_m_list = select_m_list
        self.vars_m_list = vars_m_list
        self.YMh_list = YMh_list
        self.exc_list = exc_list
        self.index_list = index_list
        self.frequencies = frequencies
        self.YQX_list = YQX_list
        self.YQ0_list = YQ0_list
        self.select_q = select_q
        self.input_data_Q =  input_data_Q
        self.varlist_list = varlist_list
        self.select_list = select_list
        self.select_c_list = select_c_list
        self.Nm_list = Nm_list
        self.nv_list = nv_list
        self.Nq_list = Nq_list
        self.select_list_sep = select_list_sep
        self.freq_ratio_list = freq_ratio_list
        self.YQ_list = YQ_list
        self.Tstar_list = Tstar_list
        self.T_list = T_list
        self.YDATA_list = YDATA_list
        self.YM_list = YM_list
        self.input_data = input_data
        self.trans = trans