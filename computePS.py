# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:21:10 2017

@author: felix

input: 
    data:      size N_CH x N_S, channels times samples
    temp_mask: size N_S, where 0 marks flagged (bad) time points
    metric:    either 'cPLV', 'wPLI', or 'dwPLI'
   
       

output: 
    values:           size N_CH x N_CH, complex for cPLV, float for wPLI
    values_surr:      size N_CH x N_CH, surrogate values from time-shifting     
"""
   
import numpy as np
         


def computePS(data,temp_mask=None,metric='cPLV',win=False,winsize=100,windist=20):  # data in ch x t (where t is samples or samples concat. over trials)
    
    output=[]                                                         
   
    if temp_mask == None:
        temp_mask = np.ones(np.shape(data,1))
    
    del_idx    = np.where(temp_mask==0)[0]                                           # cut spiky windows for static cPLV
    data_cut   = np.delete(data,del_idx,1)  
    N_S_cut    = len(data_cut[0])
    N_CH       = len(data_cut)
    
    shifts     = np.random.uniform(0,N_S_cut,N_CH)
    shifts     = shifts.astype(int) 
    data_shift = np.zeros([N_CH,N_S_cut],'complex64')  
  
    
    for j in range(N_CH):                                                                   # shift data for surrogates
        s=shifts[j]
        data_shift[j,:]=np.append(data_cut[j,s:],(data_cut[j,:s]))
   
    if metric == 'cPLV':
        data_cutN     = data_cut/abs(data_cut)                                             # normalize 
        data_shiftN   = data_shift/abs(data_shift)
        values        = np.inner(data_cutN,np.conj(data_cutN))/N_S_cut                      # compute static cPLV
        values_surr   = np.inner(data_cutN,np.conj(data_shiftN))/N_S_cut
    if metric == 'wPLI': 
        data_tp       = np.transpose(data_cut)                                              # transpose to t x ch
        data_shift_tp = np.transpose(data_shift)
        cs            = np.einsum(data_tp,[Ellipsis,0],np.conj(data_tp),[Ellipsis,1])      
        cs_shift      = np.einsum(data_tp,[Ellipsis,0],np.conj(data_shift_tp),[Ellipsis,1])       
        values        = compute_wpli(cs)
        values_surr   = compute_wpli(cs_shift)
    if metric == 'dwPLI': 
        data_tp       = np.transpose(data_cut)        
        data_shift_tp = np.transpose(data_shift)
        cs            = np.einsum(data_tp,[Ellipsis,0],np.conj(data_tp),[Ellipsis,1])      # compute cross-spectrum
        cs_shift      = np.einsum(data_tp,[Ellipsis,0],np.conj(data_shift_tp),[Ellipsis,1])
        values        = compute_wpli(cs,True)
        values_surr   = compute_wpli(cs_shift,True)    
       
    output.append([values,values_surr])
                    
    return output
   

             
                
                
def compute_wpli(cs, debias=False):        #where cs = repetitions x channel x channel 


    csi     = np.imag(cs)
    outsum  = np.nansum(csi,0)
    outsumW = np.nansum(abs(csi),0)
    if debias:
        outssq = np.nansum(csi**2,0)
        wpli   = (outsum**2 - outssq) / (outsumW**2 - outssq) 
    else:
        wpli   = outsum/outsumW 
    return wpli 





























#### alt imp for cs
# else:                                                                                  # compute cross-spectrum as one
#            cs=np.zeros([nS_cut,nCH,nCH],'complex')
#            cs_shift=np.zeros([nS_cut,nCH,nCH],'complex')        
#            s = nS_cut/10
#            for i in range(10):            
#                cs[i*s:(i+1)*s,:,:]            = np.einsum(data_tp[i*s:(i+1)*s] ,[Ellipsis,0],np.conj(data_tp[i*s:(i+1)*s] ),[Ellipsis,1])
#                cs_shift[i*s:(i+1)*s,:,:]      = np.einsum(data_shift_tp[i*s:(i+1)*s] ,[Ellipsis,0],np.conj(data_shift_tp[i*s:(i+1)*s] ),[Ellipsis,1])




#def compute_ampl_env_metrics(data,temp_mask,freq,dt):                               ### WiP
#    
#    output=[]
#                                             
#    del_idx    = np.where(temp_mask==0)[0]                                           # cut spiky windows 
#    del_idx2   = del_idx.astype(int)
#    nS_uncut   = len(data[0])
#    data       = np.delete(data,del_idx,1)  
#    nS         = len(data[0])
#    nCH        = len(data)
#    
#    
#    #orthogonalize data
#    
#    data_pinv   = np.linalg.pinv(data)
#    data_orth_x = np.empty([nCH])
#    data_orth_y = np.empty([nCH])
#    for x in range(nCH):
#        for y in range(nCH):            
#            data_orth_y = data[y]-data[x]*data_pinv[:,x]*data[y,:]
#            data_orth_x = data[x]-data[y]*data_pinv[:,y]*data[x,:]
#            
#            Ax  = ampl_env(data[x])
#            Ay  = ampl_env(data[])
#            Axy = ampl_env(data_orth_x)
#            Ayx = ampl_env(data_orth_y)
#            
#            s1  = np.mean( (Ax  - np.mean(Ax))  * (Ayx - np.mean(Ayx)) ) / (np.std(Ax)  * np.std(Ayx))
#            s2  = np.mean( (Axy - np.mean(Axy)) * (Ay  - np.mean(Ay) ) ) / (np.std(Axy) * np.std(Ay))
#            cc  = 0.5 * (s1 + s2)
#            
#            cplv = np.inner(Axy,Ayx)/nS                      # compute static cPLV
#                
#
#
#



