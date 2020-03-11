# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:23:53 2019

@author: Localadmin_fesieben
"""


source_directory    = 'E:\\HY-data\\FESIEBEN\\OL2000\\OL2015\\source\\'
import numpy as np
import sys
sys.path.append(source_directory + 'Python27\\Utilities')
import CF_functions as cffun
import plot_functions as plots
import matplotlib.pyplot as plt
from scipy import stats as stat
import statsmodels.sandbox.stats.multicomp as multicomp

my_cmap  = plots.make_cmap([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.8, 0.0, 1.0)])



# set basic parameters
PS_metric = 'wPLI'
CF_type   = 'CFS'
SEEG_directory = 'K:\\palva\\resting_state\\RS_CF_SEEG\\'
MEG_directory  = 'K:\\palva\\resting_state\\RS_CF_MEG\\'

N_ratios=6
ratios  = ['1:'+str(i+2) for i in range(N_ratios)]    
ratios2 = ['1-'+str(i+2) for i in range(N_ratios)]  

CFM_filename_MEG = MEG_directory + '_settings\\CF_matrix_MEG.csv'
CFM_MEG          = np.genfromtxt(CFM_filename_MEG, delimiter=';')
cutoff_LF        = 100
cutoff_HF        = 315
LFs_MEG          = CFM_MEG[:,0][CFM_MEG[:,0]<cutoff_LF]         
freqs_MEG        = CFM_MEG[:,0]
HFs_MEG          = []     
for f in range(len(LFs_MEG)):
    x   = CFM_MEG[f,1:N_ratios+1]
    HFs_MEG.append(x[np.where(x<cutoff_HF)])
    
    
freqs_filename_SEEG    = SEEG_directory + '_settings\\all_frequencies_SEEG.csv'
freqs_SEEG             = np.genfromtxt(freqs_filename_SEEG, delimiter=';')                                                
CFM_filename_SEEG      = SEEG_directory + '_settings\\CF_matrix_SEEG.csv'
CFM_SEEG               = np.genfromtxt(CFM_filename_SEEG, delimiter=';')    
LFs_SEEG               = CFM_SEEG[:,0]

MEG_indices  = [1,2,4,5,6,7,8,9,10,12,13,14,15,18,19,20,21,23,24,26,27,28,30,31,33,34,35,36,38,40]
SEEG_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,20,22,23,24,25,27,29,30,31,32,34,35]    # for PS

LF_mix = np.array([LFs_MEG[MEG_indices],LFs_SEEG])

freqs=LFs_SEEG
N_freq = len(MEG_indices)


parcels = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 36, 37, 38, 39, 40, 41, 42,
 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147]



''' ######################   degree analysis   ###################### '''

#temporary cause some freqs missing in MEG?
if CF_type == 'CFS':
  #  MEG_indices = MEG_indices[:25]
    N_freq=27

parc = 'parc2009'    
morph_sources = cffun.get_morph_sources(MEG_directory + '\\_settings\\morphing sources 200 to 148.csv')
layer_int = ['superf-surf', 'deep-deep', 'superf-deep', 'deep-superf']

#define directories    
SEEG_gm_dir = SEEG_directory + '_results\\graph metrics 2018-05\\'
MEG_gm_dir  = MEG_directory  + '_results\\graph metrics 2018-05\\'   
SEEG_deg_dir0 = SEEG_gm_dir + CF_type + ' degrees ' + parc + '\\'
SEEG_deg_dirs = [SEEG_gm_dir + CF_type + ' degrees ' + parc + ' ' + layer_int[a] +'\\' for a in range(4)]
MEG_deg_dir   = MEG_gm_dir + CF_type + ' degrees ' + 'parc2009_200AFS\\'    
    

# load MEG CF degrees    
MEG_degrees_CF = np.array([np.genfromtxt(MEG_deg_dir + 'CF ' + r + '.csv',delimiter=';')   for r in ratios2])
# morph and prune freqs
MEG_degrees_CF = np.array([[[ np.mean(d1[f][ms]) for ms in morph_sources]  for f in MEG_indices] for d1 in MEG_degrees_CF])
# prune parcels
MEG_degrees_CF = MEG_degrees_CF[np.ix_(range(6),range(N_freq),parcels)]
# load MEG PS degrees    
MEG_degrees_PS = np.genfromtxt(MEG_deg_dir + 'PS.csv', delimiter=';')   
# morph and prune freqs
MEG_degrees_PS = np.array([[ np.mean(MEG_degrees_PS[f][ms]) for ms in morph_sources]  for f in MEG_indices]) 
# prune parcels
MEG_degrees_PS = MEG_degrees_PS[np.ix_(range(N_freq),parcels)]



# load SEEG CF degrees    
SEEG_degrees_CF     = np.array([np.genfromtxt(SEEG_deg_dir0 + 'CF ' + r + '.csv',delimiter=';')   for r in ratios2])
# load SEEG CF degrees in layers  
SEEG_degrees_CF_lay = np.array([[np.genfromtxt(SEEG_deg_dirs[l] + 'CF ' + r + '.csv',delimiter=';')   for r in ratios2] for l in range(4)])
# load SEEG PS degrees    
SEEG_degrees_PS     = np.array(np.genfromtxt(SEEG_deg_dir0 + 'PS.csv', delimiter=';')   )
# load SEEG PS degrees in layers
SEEG_degrees_PS_lay = np.array([np.genfromtxt(SEEG_deg_dirs[l] + 'PS.csv', delimiter=';')   for l in range(4)])
# freq and parcel pruning
SEEG_degrees_PS     = SEEG_degrees_PS[np.ix_(SEEG_indices,parcels)]
# freq and parcel pruning
SEEG_degrees_PS_lay = SEEG_degrees_PS_lay[np.ix_(range(4),SEEG_indices,parcels)]
# parcel pruning
SEEG_degrees_CF     = SEEG_degrees_CF[np.ix_(range(6),range(N_freq),parcels)]
# parcel pruning
SEEG_degrees_CF_lay = SEEG_degrees_CF_lay[np.ix_(range(4),range(6),range(N_freq),parcels)]









# compute CF correlations in layers and bands
freq_bands = [range(1,6),range(3,10),range(9,16),range(16,22),range(22,N_freq)]

N_ratios = 3

corr_CF_layPB = np.zeros([4,N_ratios,5])
corr_CF_laySB = np.zeros([4,N_ratios,5])
p_CF_layPB    = np.zeros([4,N_ratios,5])
p_CF_laySB    = np.zeros([4,N_ratios,5])

for l in range(4):
    for r in range(N_ratios):
        for f in range(5):
            a = np.sum(MEG_degrees_CF[r,freq_bands[f]],0)
            b = np.sum(SEEG_degrees_CF_lay[l,r,freq_bands[f]],0)
            idx = np.intersect1d(np.where(np.isfinite(a))[0],np.where(np.isfinite(b))[0])
            corr_CF_layPB[l,r,f], p_CF_layPB[l,r,f] = stat.pearsonr(a[idx],b[idx])
            corr_CF_laySB[l,r,f], p_CF_laySB[l,r,f] = stat.spearmanr(a[idx],b[idx])



my_cmap8 = plots.make_cmap([(0, 0, 1), (1, 1, 1), (1, 0, 0)]) 

z_CF_laySB = 0.5*np.log( (corr_CF_laySB+1) / (1-corr_CF_laySB))
z_compB = (z_CF_laySB[1] - z_CF_laySB[0]) / np.sqrt( (1./130) + (1./130))
np.abs(z_compB[:,2]) >1.96

p_vals = p_CF_laySB[:,:,2]
corrp  = (multicomp.multipletests(np.reshape(p_vals,np.prod(np.shape(p_vals))),method='fdr_bh')[1])    
p_vals<0.05
p_vals<0.01
p_vals<0.001

np.reshape(corrp<0.05,np.shape(p_vals))

plots.simple_CF_plot(corr_CF_laySB[:,:,2],[8,6],'ratio',CF_type,np.arange(0.5,6,1), 
                     np.arange(0.5,4,1),ratios2,layer_int,zmax=0.3,zmin=-0.3,ztix=[-0.3,0,0.3],cmap=my_cmap8)
















