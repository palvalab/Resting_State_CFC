# -*- coding: utf-8 -*-
"""
Created by Felix Siebenhühner
adapted from astropy module
"""
import numpy as np

def CI_from_bootstrap(data, bootnum=1000, low=2.5, high=97.5, N_array=None, samples=None, bootfunc=None):
         
    '''
       data can have any number ND of dimensions, bootstrapping will be done for the 0th dimension
       N_array can be used for weighing, can have 1 or ND-1 dimensions 
       e.g. shape(data) = 5x30x60   --> shape(N_array)=30 or shape(N_array)=30x60        
       
    '''
    
    if samples is None:
        samples = data.shape[0]
    
    # make sure the input is sane
    if samples < 1 or bootnum < 1:
        raise ValueError("neither 'samples' nor 'bootnum' can be less than 1.")
    
    if bootfunc is None:
        resultdims = (bootnum,) + (samples,) + data.shape[1:]
    else:
    # test number of outputs from bootfunc, avoid single outputs which are
    # array-like
        try:
            resultdims = (bootnum, len(bootfunc(data)))
        except TypeError:
            resultdims = (bootnum,)
        
    # create empty boot array
    boot = np.empty(resultdims)
    
    for i in range(bootnum):
        bootarr = np.random.randint(low=0, high=data.shape[0], size=samples)
        if bootfunc is None:
            boot[i] = data[bootarr]
        else:
            boot[i] = bootfunc(data[bootarr])
        if N_array is not None:
            N = np.nansum(N_array[bootarr],0)
            N.astype(float)
            boot[i] = boot[i]/N
    
    if N_array is not None:
        mean      = np.nansum(data,0)/np.nansum(N_array,0)
        means     = np.nansum(boot,1)
    else: 
        mean      = np.nanmean(data,0)
        means     = np.nanmean(boot,1)
        
    mean_boot = np.nanmean(means,0)
    lower     = np.percentile(means,low,0)
    upper     = np.percentile(means,high,0)
    
    return mean, lower, upper, mean_boot