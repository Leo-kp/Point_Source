import numpy as np
from scipy import special as sc 
import inspect

#-----------------------------------------------------------------------------
def pd_ps(t, ddict, k, r):
    """
    pd_ point-source spherical flow
    Supports ONE vectorized parameter at a time (k or r).
    Dimensionless variables (tD, rD) are calculated relative to r_w.
    Inputs.
        t: time
        ddict: dictionary with parameters/properties (mu, por, ct, rw)
        k: permeability (scalar or array)
        k: radius (scalar or array)
    Ouputs.
        pd_: dimensionless pressure (array)
    """
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
    
    pd_ = (1 / rd) * sc.erfc(rd / (2 * np.sqrt(td)))
    pd_ = np.where(t <= 0, 0.0, pd_)
    
    if pd_.shape[1] == 1:
        return pd_.squeeze() #flatten 1D arrays
    return pd_

#----------------------------------------------------------------------------
def step_rate_s(func, delta_t, tp, q_array, rd_dict, k_val, r_val, *args):
    """
    2D Optimized Step Rate for spherical-flow family. 
    Supports ONE vectorized parameter (k, r, or an arg).
    Inputs.
        func: function-call to evaluate the steps 
        delta_t: time value of test
        tp: T time of step protocol change
        q_array: flowrate of test in every step T
        rd_dict: dictionary with parameters/properties (pi,mu,b,h)
        k_val: permeability value (scalar or array)
        r_val: radii of evaluation (scalar or array)
        *arg: ordered argument of function kernel used
    Ouput:
        p_ws: pressure of the step rate in consistent units. 
    """

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

    return p_ws.squeeze() if num_scenarios == 1 else p_ws

#----------------------------------------------------------------------------
