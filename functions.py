from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy import special as sc 

import inspect
from skimage.restoration import denoise_tv_chambolle

def conv_eng_si(value,
               unit:str
):
  #si/eng
  factor={'psi': 6894.76,
          'psi-1': 1/6894.76,
          'acres': 4046.86,
          'ft': 0.3048,
          'h': 3600,
          'min':60,
          'cP': 0.001,
          'RB/STB':1,
          'STB/D': 0.0000018401,
          'vol_fraction':1,
          'md':9.8692326671601E-16
  }

  unit_c={'psi': 'Pa',
          'psi-1': 'Pa-1',
          'acres': 'm**2',
          'ft': 'm',
          'h': 's',
          'min':'s',
          'cP': 'Pa*s',
          'RB/STB': ' ',
          'STB/D': 'm**3/s',
          'vol_fraction':'vol_fraction',
          'md':'m**2'
  }

  try:
    conversion_factor = factor[unit]
    converted_value = value * conversion_factor
        
    return converted_value,unit_c[unit]
        
  except KeyError:
    print(f"Error: Unit '{unit}' not found in conversion dictionary.")


#-----------------------------------------------------------------------------
def pwd_ps(t, ddict, k, r):
    t = np.atleast_1d(t).reshape(-1, 1)
    k = np.atleast_1d(k).flatten()[None, :]
    r = np.atleast_1d(r).flatten()[None, :]

    mu = ddict['mu']
    por = ddict['por']
    c_t = ddict['c_t']
    r_w = ddict['r_w'] 

    t_safe = np.maximum(t, 1e-12) #avoiding zero division
    td = (k * t_safe) / (por * mu * c_t * (r_w**2))
    rd = r / r_w  
    
    pwd = (1 / rd) * sc.erfc(rd / (2 * np.sqrt(td)))
    pwd = np.where(t <= 0, 0.0, pwd)
    
    if pwd.shape[1] == 1:
        return pwd.ravel() #flatten 1D arrays
    return pwd

#----------------------------------------------------------------------------
def step_rate_s(func, delta_t, tp, q_array, rd_dict, k_val, r_val, *args):

    #--------------------------
    required_params = list(inspect.signature(func).parameters.values())
    num_kernel_requires = len(required_params)
    num_provided = 4 + len(args) 

    if num_provided < num_kernel_requires:
        missing = [p.name for p in required_params[num_provided:]]
        raise ValueError(f"Kernel '{func.__name__}' needs more parameters: {missing}. "
                         f"Check your *args in step_rate_pws.")
    # ---------------------------

    pi, mu, B = rd_dict['p_i'], rd_dict['mu'], rd_dict['B']
    r_w = rd_dict['r_w']
    
    k_arr = np.atleast_1d(k_val).flatten()[None, :]
    r_arr = np.atleast_1d(r_val).flatten()[None, :]
    num_scenarios = max(k_arr.size, r_arr.size)
    
    dt_matrix = delta_t[:, None] - tp[None, :] #delta_t[:, None] is (N, 1), tp[None, :] is (1, M)
    mask = dt_matrix > 1e-12 #1e-12 as the epsilon for T=0 issues
    dt_safe = np.where(mask, dt_matrix, 0.0)
    
    p_drop_raw = func(dt_safe.ravel(), rd_dict, k_arr, r_arr, *args)
    
    num_times, num_events = dt_matrix.shape
    p_drop_2d = np.reshape(p_drop_raw, (-1, num_scenarios))
    p_drop_matrix = p_drop_2d.reshape(num_times, num_events, num_scenarios)

    dq = np.diff(q_array, prepend=0)

    summation = np.einsum('j,ijk->ik', dq, p_drop_matrix) #Einstein Summation

    C = (mu * B) / (4 * np.pi * k_arr * r_w)
    p_ws = pi - (C * summation)

    return p_ws.ravel() if num_scenarios == 1 else p_ws

#----------------------------------------------------------------------------

def scale_and_smooth(series, w=0.1):
    """Smothing a signal with scaling and function
    arg. 
    series: data series
    w: weight of the smoothing
    Re.
    Smoothed curve in original scale
    """
    q_min = series.min()
    q_max = series.max()
    q_range = q_max - q_min
    
    if q_range == 0: return series # Avoid division by zero
    
    scaled = (series - q_min) / q_range
    
    smoothed = denoise_tv_chambolle(scaled.values, weight=w)
    
    return (smoothed * q_range) + q_min

#----------------------------------------------------------------------------

def pickings(series, window=20, sensitivity=5):
    series_vals = series.values
    series_idx = series.index
    
    rows=[]
   
    rolling_std = series.rolling(window=window).std().fillna(0).values #function of moving standard deviation
    
    last_val = series_vals[0]
    i = window
    
    while i < len(series_vals) - 1:
        current_val = series_vals[i]
        local_noise = rolling_std[i]

        dynamic_threshold = (local_noise * sensitivity) + 0.05 #0.05 for not triggering in silence
        
        if abs(current_val - last_val) > dynamic_threshold:
            rows.append([series_idx[i], current_val]) 
            last_val = current_val
            i += window
        else:
            i += 1

    return   np.array(rows)#transpose the shape


#----------------------------------------------------------------------------