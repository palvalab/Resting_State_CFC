# -*- coding: utf-8 -*-
"""
@author: Felix Siebenhühner
"""


source_directory    = 'point to source directory here'
directory           = 'point to MEG data directory here'

import sys
sys.path.append(source_directory)
import numpy as np
import cPickle as pick
import CF_functions as cffun
import plot_functions as plots
import matplotlib as mpl
import time
import bootstrap as bst
from scipy import stats as stat
import statsmodels.sandbox.stats.multicomp as multicomp

# initialize settings

sign_z_CFC   = 2.42
PS_metric    = 'wPLI'                                                       # 'PLV' or 'wPLI'
CF_type      = 'PAC'                                                        # 'PAC' or 'CFS'                                                  

if PS_metric == 'wPLI':                                                     # for readout
    metric2 = 'wpli'
    sign_z_PS    = 2 
else:
    metric2 = 'cPLV'
    sign_z_PS    = 2.42

# frequencies
CFM_filename = directory + '_support_files\\CF_matrix_MEG.csv'
CFM          = np.genfromtxt(CFM_filename, delimiter=';')
cutoff       = 316   
cutoff_LF    = 100
                   
HFs_env = [CFM[:41,i+1] for i in range(6)]


LFs          = CFM[:,0][CFM[:,0]<cutoff_LF]         
freqs        = CFM[:,0][CFM[:,0]<cutoff]           
masks        = False
mask_type    = ''
N_CH         = 200

subject_sets=['S0006 set01','S0008 set01','S0035 set01','S0038 set02','S0039 set01','S0049 set01',             
              'S0113 set02','S0116 set01','S0116 set02','S0116 set03','S0117 set01','S0118 set01',
              'S0118 set02','S0118 set03','S0119 set01','S0119 set02','S0120 set01',
              'S0121 set01','S0123 set01','S0123 set02','S0124 set01','S0124 set02',
              'S0126 set01','S0127 set01','S0128 set01','S0130 set01','S0130 set02']

subjects=[s[:5] for s in subject_sets]                                  

N_sets   = len(subject_sets)
N_freq   = len(freqs)
N_LF     = len(LFs)

N_ratios = 6
ratios  = ['1:'+str(i+2) for i in range(N_ratios)]    
ratios2 = ['1-'+str(i+2) for i in range(N_ratios)] 

xlims_PS = [1,cutoff]
xlims_CF = [1,cutoff_LF]

cutoff_HF = 350
HFs       = []     
for f in range(len(LFs)):
    x   = CFM[f,1:N_ratios+1]
    HFs.append(x[np.where(x<cutoff_HF)])


parcellation = 'parc2009_200AFS'
N_parc = 200

CP_PLV    = [None for s in subjects]
fidelity  = [None for s in subjects]

# get cross-patch PLV
for s,subj in enumerate(subjects):
    filename_base = directory + '_support_files\\' + subj + '\\Cross-Patch PLV ' + parcellation
    CP_PLV[s] = cffun.read_complex_data_from_csv(filename_base, delimiter=';')

# get patch fidelity
for s,subj in enumerate(subjects):
    filename = directory + '_support_files\\' + subj + '\\Patch Fidelity ' + parcellation + '.csv'
    fidelity[s]   = np.genfromtxt(filename, delimiter=';')
   

# get networks for parcel
filename          = directory + '\\_support_files\\networks.csv'
network_indices   = np.array(np.genfromtxt(filename, delimiter=';'),'int')
networks          = [np.where(network_indices==i)[0] for i in range(7)]
network_names     = ['C','DM','DA','Lim','VA','SM','Vis']
N_networks        = len(network_names)

mean_CP_PLV   = np.mean(abs(np.array(CP_PLV)),0)
mean_fidelity = np.mean(np.array(fidelity),0)

fidelity_threshold = 0.1
if PS_metric == 'wPLI':                                                     
    CP_PLV_threshold = 1         
else:
    CP_PLV_threshold = 0.2143

fidelity_mask   = np.outer((mean_fidelity>fidelity_threshold),(mean_fidelity>fidelity_threshold))    # create a nice mask from 
CP_PLV_mask     = mean_CP_PLV < CP_PLV_threshold                         # CP-PLV is 1 on diagonal, so the mask diagonal will be 0 - good!
mask            = fidelity_mask*CP_PLV_mask

edges_retained = np.sum(mask)/float(200*199)

np.fill_diagonal(mask,1)                            # for local!

# get distances   
dist      = [None for s in subjects]
dist_thresholds = [0.0]    
all_dist = np.empty(0)

for s,subj in enumerate(subjects):
    filename  = directory + '_support_files\\' + subj + '\\Patch Distances ' + parcellation + '.csv'
    dist[s]   = np.genfromtxt(filename, delimiter=';')
    d1        = dist[s]*mask    
    all_dist  = np.append(all_dist,d1.reshape(len(dist[s])**2))

all_dist = all_dist[np.where(all_dist>0)]  
dist_thresholds.extend(np.percentile(all_dist,[33.3333,66.66667]))
dist_max = max(all_dist)
dist_thresholds.extend([dist_max])

dist_strs = ['{:.1f}'.format(dd*100) for dd in dist_thresholds]  
N_dist_bins = len(dist_thresholds)-1   
distances = [dist_strs[i]+'-'+dist_strs[i+1]+'cm' for i in range(N_dist_bins)]


##### colormaps for plotting
my_cmap  = plots.make_cmap([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.8, 0.0, 1.0)])
my_cmap2 = plots.make_cmap([(0.0, 0.0, 0.0), (0.5, 0.5, 1.0), (0.6, 0.6, 1.0), (0.7, 0.7, 1.0), (0.8, 0.8, 1.0),(0.9, 0.9, 1.0), (1, 1, 1)])
my_cmap3 = plots.make_cmap([(1.0, 0.0, 0.0), (0.0, 0.6, 0.0), (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (0.6, 0.4, 0.4)]) 
my_cmap4 = plots.make_cmap([(0.8, 0.6, 0.0), (1.0, 0.0, 0.0), (0.0, 0.8, 0.0), (0.1, 0.1, 0.1), (1.0, 0.4, 0.9), (0.0, 0.0, 1.0), (0.8, 0.0, 0.9)])
my_cmap5 = plots.make_cmap([(1,0,0), (0,1,0), (0,0,1)])
my_cmap6 = plots.make_cmap([(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.6, 0.0), (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (0.6, 0.4, 0.4)]) 

   
# set matplotlib parameters    
mpl.rcParams['pdf.fonttype'] = 42           # for PDF compatibility with Illustrator
mpl.rcParams.update({'font.size': 8})
mpl.rcParams.update({'axes.titlesize': 8})
mpl.rcParams.update({'axes.labelsize': 8})
mpl.rcParams.update({'legend.fontsize': 6})
mpl.rcParams.update({'xtick.labelsize': 7})
mpl.rcParams.update({'ytick.labelsize': 7})





# initialize lists
PS          =  [[   None for i in range(N_freq)] for j in range(N_sets)]       
PS_dist     =  [[[  None for b in range(N_dist_bins)] for i in range(N_freq)] for j in range(N_sets)] 
 
PS_ENV      =  [[[  None for i in range(len(HFs[j]))] for j in range(N_LF)] for k in range(N_sets)]
PS_ENV_dist =  [[[[ None for b in range(N_dist_bins)] for i in range(len(HFs[j]))] for j in range(N_LF)] for k in range(N_sets)]

CFC         =  [[[  None for i in range(len(HFs[j]))] for j in range(N_LF)] for k in range(N_sets)]
CFC_dist    =  [[[[ None for b in range(N_dist_bins)] for i in range(len(HFs[j]))] for j in range(N_LF)] for k in range(N_sets)]


#### analysis of PS  ####   
for s,sset in enumerate(subject_sets): 
    for f,F in enumerate(freqs):     
        F_str       = '{:.2f}'.format(F)
        if PS_metric == 'wPLI':
            file1       = directory + '_data\\_PS_wPLI\\'  + sset + ' f=' + F_str + '.csv'
            file2       = directory + '_data\\_PS_wPLI\\'  + sset + ' f=' + F_str + '_surr.csv'
        else:                
            file1       = directory + '_data\\_PS_PLV\\'  + sset + ' f=' + F_str + '.csv'
            file2       = directory + '_data\\_PS_PLV\\'  + sset + ' f=' + F_str + '_surr.csv'
        masked_data = mask*np.genfromtxt(file1, delimiter=';') 
        surr_data   = mask*np.genfromtxt(file2, delimiter=';') 
        stats       = cffun.K_stats_PS_2(masked_data,surr_data,sign_z_PS,PS_metric)
        PS[s][f]    = stats 
        
        for d in range(N_dist_bins):
            dist_mask        = mask * ( ( (dist[s]>dist_thresholds[d]) * (dist[s]<=dist_thresholds[d+1]) )>0)
            masked_dataD     = masked_data * dist_mask
            N_pot            = np.sum(dist_mask!=0)                                       
            if N_pot>0:
                stats        = cffun.K_stats_PS_2(masked_dataD,surr_data,sign_z_PS,PS_metric)           
            else:
                stats        = cffun.stats_PS(np.nan)    
            PS_dist[s][f][d] = stats    
    print sset


#### analysis of LF-envelope filtered HF amplitude correlations #####   
for s,sset in enumerate(subject_sets):
    for lf,LF in enumerate(LFs):
        for hf,HF in enumerate(HFs[lf]):  
            np.fill_diagonal(mask,1)      
            path        = directory + '_data\\_ENV\\'
            LF_str      = '{:.2f}'.format(LF)                
            HF_str      = '{:.2f}'.format(HF)  
            file1       = path + sset + ' LF= ' + LF_str + ' HF= ' + HF_str + '.csv'            
            file2       = path + sset + ' LF= ' + LF_str + ' HF= ' + HF_str + '_surr.csv'
            N_pot       = np.nansum(mask)                
            data        = mask*np.genfromtxt(file1, delimiter=';') 
            data_surr   = mask*np.genfromtxt(file2, delimiter=';') 
            stats       = cffun.K_stats_PS_2(data,data_surr,sign_z_PS,'PLV')
            PS_ENV[s][lf][hf]= stats                 
            for d in range(N_dist_bins):
                dist_mask        = mask * ( ( (dist[s]>dist_thresholds[d]) * (dist[s]<=dist_thresholds[d+1]) )>0)
                N_potD           = np.nansum(dist_mask)                                       
                dataD            = data*dist_mask       
                if N_potD>0:
                    stats      = cffun.K_stats_PS_2(dataD,data_surr,sign_z_PS,PS_metric)    
                else:
                    stats = cffun.stats_PS(np.nan)    
                PS_ENV_dist[s][lf][hf][d] = stats                              
    print(time.strftime("%Y-%m-%d %H:%M") + '          ' + sset)
 
     
#### analysis of CFC ####
for s,sset in enumerate(subject_sets):
    for lf,LF in enumerate(LFs):  
        for hf,HF in enumerate(HFs[lf]): 
            LF_str      = '{:.2f}'.format(LF)                
            HF_str      = '{:.2f}'.format(HF) 
            LF_PS       = PS[s][lf].data_sign
            HF_idx      = np.where(freqs==HF)[0][0] 
            if CF_type == 'CFS':
                HF_PS     = PS[s][HF_idx].data_sign
            else:
                HF_PS     = PS_ENV[s][lf][hf].data_sign
            path        = directory + '_data\\_' + CF_type + '\\'
            file0       = path + sset + ' LF= ' + LF_str + ' HF= ' + HF_str + '.csv'            
            file_surr   = path + sset + ' LF= ' + LF_str + ' HF= ' + HF_str + '_surr.csv'
            masked_data = np.genfromtxt(file0,  delimiter=';') * mask   
            surr_data   = np.genfromtxt(file_surr, delimiter=';') * mask            
          
            N_pot       = np.sum(mask)   
            stats       = cffun.K_stats_CFC_2(masked_data,surr_data,sign_z_CFC,LF_PS,HF_PS)
            CFC[s][lf][hf] = stats
       
            for d in range(N_dist_bins):
                dist_mask        = mask * ( ( (dist[s]>dist_thresholds[d]) * (dist[s]<=dist_thresholds[d+1]) )>0)
                np.fill_diagonal(dist_mask,1)
                masked_dataD     = masked_data*dist_mask
                mask             = np.ones([N_CH,N_CH])                 
                N_pot            = np.sum(dist_mask!=0)                                       
                if N_pot>0:
                    stats       = cffun.K_stats_CFC_2(masked_dataD,surr_data,sign_z_CFC,LF_PS,HF_PS)
                else:
                    stats = cffun.stats_CFC(np.nan)    
                CFC_dist[s][lf][hf][d] = stats       

    print sset





##############################################    
##########    PLOT PS  - S5 Figure     #######
   
# get means and create plots for PS
PLV_PS_ps        = np.zeros([N_sets,N_freq])                           # PS = "Phase Synch", ps = "per subject"
K_PS_ps          = np.zeros([N_sets,N_freq])
N_PS_ps          = np.zeros([N_sets,N_freq])
N_pot_PS_ps      = np.zeros([N_sets,N_freq])
PLV_PS_dist_ps   = np.zeros([N_sets,N_freq,N_dist_bins])
K_PS_dist_ps     = np.zeros([N_sets,N_freq,N_dist_bins])
N_PS_dist_ps     = np.zeros([N_sets,N_freq,N_dist_bins])
N_pot_PS_dist_ps = np.zeros([N_sets,N_freq,N_dist_bins])

for f,F in enumerate(freqs): 
   for s,ss in enumerate(subject_sets):            
       PLV_PS_ps[s,f]        = PS[s][f].mean_masked                            
       K_PS_ps[s,f]          = 100*PS[s][f].K

       for d in range(N_dist_bins):   
           PLV_PS_dist_ps[s,f,d]    = PS_dist[s][f][d].mean_masked 
           K_PS_dist_ps[s,f,d]      = 100*PS_dist[s][f][d].K
                              
mean_K_PS           =  [np.transpose(np.nanmean(K_PS_ps,0))]
mean_PLV_PS         =  [np.transpose(np.nanmean(PLV_PS_ps,0))] 
mean_K_PS_dist      =   np.transpose(np.nanmean(K_PS_dist_ps,0))
mean_PLV_PS_dist    =   np.transpose(np.nanmean(PLV_PS_dist_ps,0))     

K_PS_stats          = [bst.CI_from_bootstrap(K_PS_ps)]              # returns [mean, mean_boot, lower, upper] x freq x ratio
PLV_PS_stats        = [bst.CI_from_bootstrap(PLV_PS_ps)]            # returns [mean, mean_boot, lower, upper] x freq x ratio
K_PS_stats_dist     = [bst.CI_from_bootstrap(K_PS_dist_ps[:,:,i]) for i in range(N_dist_bins)] 
PLV_PS_stats_dist   = [bst.CI_from_bootstrap(PLV_PS_dist_ps[:,:,i]) for i in range(N_dist_bins)] 
  

# PLOT PS with distances
o67 = directory + '_results\\MEG PS\\MEG ' + PS_metric + '.pdf'

figsize     = [4.3,2.3]
rows        = 2
cols        = 2
dataL       = [PLV_PS_stats,PLV_PS_stats_dist,K_PS_stats,K_PS_stats_dist]
xlimA       = [xlims_PS for i in range(4)]
if PS_metric == 'wPLI':
    ylimA       = [[-0.005,0.12],[-0.005,0.12],[-5,100],[-5,100]]
else:
    ylimA       = [[-0.005,0.26],[-0.005,0.26],[-5,100],[-5,100]]

titlesA     = ['' for i in range(4)]                          #'mean '+PS_metric,'mean '+PS_metric+' per distance','mean '+PS_metric+' per subject','mean K','mean K per distance','mean K per subject']
legendA     = [None, distances, None, distances]
ylabA       = [PS_metric,'','K','']
cmapA       = ['winter','brg','winter','brg']
CI          = [0.3,0.3,0.3,0.3]
legend_posA = [None,'ur',None,None]
xlab        = [0,0,1,1]
Ryt         = [1,1,0,0]
plots.semi_log_plot_multi(figsize,rows,cols,dataL,freqs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt)   

# save pdf
plots.semi_log_plot_multi(figsize,rows,cols,dataL,freqs,xlimA,ylabA,titlesA,cmapA,legendA,o67,legend_posA,ylimA,False,1,CI)   





##################################################
######               PLOT CFC               ######
   
# init CFC arrays
K_CFC_ps               = np.full([N_sets,N_LF,N_ratios,],np.nan)
PLV_CFC_ps             = np.full([N_sets,N_LF,N_ratios],np.nan)
K_ENV_ps               = np.full([N_sets,N_LF,N_ratios,],np.nan)
PLV_ENV_ps             = np.full([N_sets,N_LF,N_ratios],np.nan)
K_CFC_local_ps         = np.full([N_sets,N_LF,N_ratios],np.nan)
PLV_CFC_local_ps       = np.full([N_sets,N_LF,N_ratios],np.nan)
K_CFC_dist_ps          = np.full([N_sets,N_LF,N_ratios,N_dist_bins],np.nan)
PLV_CFC_dist_ps        = np.full([N_sets,N_LF,N_ratios,N_dist_bins],np.nan) 
K_ENV_dist_ps          = np.full([N_sets,N_LF,N_ratios,N_dist_bins],np.nan)
PLV_ENV_dist_ps        = np.full([N_sets,N_LF,N_ratios,N_dist_bins],np.nan) 
K_CFC_ps_mod           = np.full([N_sets,N_LF,N_ratios],np.nan)
K_CFC_ps_mod_w         = np.full([N_sets,N_LF,N_ratios],np.nan)
PLV_CFC_ps_mod         = np.full([N_sets,N_LF,N_ratios],np.nan)
K_CFC_dist_ps_mod      = np.full([N_sets,N_LF,N_ratios,N_dist_bins],np.nan)
PLV_CFC_dist_ps_mod    = np.full([N_sets,N_LF,N_ratios,N_dist_bins],np.nan)
K_CFC_dist_ps_mod_w    = np.full([N_sets,N_LF,N_ratios,N_dist_bins],np.nan)
N_pot_mod_subj         = np.full([N_sets,N_LF,N_ratios],np.nan)

   
# get CFC values
for lf,LF in enumerate(LFs): 
   for hf,HF in enumerate(HFs[lf]):
       if HF<cutoff_HF:
           for s,ss in enumerate(subject_sets):                    
               K_CFC_local_ps [s,lf,hf]        = 100*CFC[s][lf][hf].K_local                   
               K_CFC_ps       [s,lf,hf]        = 100*CFC[s][lf][hf].K
               PLV_CFC_ps     [s,lf,hf]        = CFC[s][lf][hf].mean_masked
               K_ENV_ps       [s,lf,hf]        = 100*PS_ENV[s][lf][hf].K
               PLV_ENV_ps     [s,lf,hf]        = PS_ENV[s][lf][hf].mean_masked
               K_CFC_ps_mod   [s,lf,hf]        = 100*CFC[s][lf][hf].K_mod
               K_CFC_ps_mod_w [s,lf,hf]        = 100*CFC[s][lf][hf].K_mod * CFC[s][lf][hf].N_pot_mod    #weighting by subject's N_pot_mod
               PLV_CFC_ps_mod [s,lf,hf]        = CFC[s][lf][hf].mean_mod                    
               N_pot_mod_subj [s,lf,hf]        = CFC[s][lf][hf].N_pot_mod
               PLV_CFC_local_ps[s,lf,hf]  = CFC[s][lf][hf].mean_local

               for d in range(N_dist_bins):                                      
                   PLV_CFC_dist_ps [s,lf,hf,d] = CFC_dist[s][lf][hf][d].mean_masked
                   K_CFC_dist_ps   [s,lf,hf,d] = 100*CFC_dist[s][lf][hf][d].K
                   PLV_ENV_dist_ps [s,lf,hf,d] = PS_ENV_dist[s][lf][hf][d].mean_masked
                   K_ENV_dist_ps   [s,lf,hf,d] = 100*PS_ENV_dist[s][lf][hf][d].K
                   PLV_CFC_dist_ps_mod [s,lf,hf,d] = CFC_dist[s][lf][hf][d].mean_mod
                   K_CFC_dist_ps_mod   [s,lf,hf,d] = 100*CFC_dist[s][lf][hf][d].K_mod
                   K_CFC_dist_ps_mod_w [s,lf,hf,d] = 100*CFC_dist[s][lf][hf][d].K_mod * CFC[s][lf][hf].N_pot_mod 
   
# get CFC means and 95% confidence intervals
N_boot=1000
    
PLV_CFC_stats       = [np.array(bst.CI_from_bootstrap(PLV_CFC_ps[:,:,i]))       for i in range(N_ratios)] # returns [mean, mean_boot, lower, upper] x freq x ratio
K_CFC_stats         = [np.array(bst.CI_from_bootstrap(K_CFC_ps[:,:,i])) -1      for i in range(N_ratios)]  
K_CFC_stats_mod     = [np.array(bst.CI_from_bootstrap(K_CFC_ps_mod[:,:,i]))-1   for i in range(N_ratios)] 
K_CFC_stats_mod_w   = [np.array(bst.CI_from_bootstrap(K_CFC_ps_mod_w[:,:,i],N_boot,2.5,97.5,N_pot_mod_subj[:,:,i]))-1  for i in range(N_ratios)] 
PLV_CFC_local_stats = [np.array(bst.CI_from_bootstrap(PLV_CFC_local_ps[:,:,i])) for i in range(N_ratios)]
K_CFC_local_stats   = [np.array(bst.CI_from_bootstrap(K_CFC_local_ps[:,:,i]))-1 for i in range(N_ratios)]  
PLV_ENV_stats       = [np.array(bst.CI_from_bootstrap(PLV_ENV_ps[:,:,i]))       for i in range(N_ratios)] # returns [mean, mean_boot, lower, upper] x freq x ratio
K_ENV_stats         = [np.array(bst.CI_from_bootstrap(K_ENV_ps[:,:,i])) -1      for i in range(N_ratios)]  

 # get dist stats 
PLV_CFC_dist_12_stats     = [bst.CI_from_bootstrap(PLV_CFC_dist_ps[:,:,0,i]) for i in range(N_dist_bins)]           # returns [mean, mean_boot, lower, upper] x freq x dist
PLV_CFC_dist_13_stats     = [bst.CI_from_bootstrap(PLV_CFC_dist_ps[:,:,1,i]) for i in range(N_dist_bins)]           
K_CFC_dist_12_stats       = [np.array(bst.CI_from_bootstrap(K_CFC_dist_ps[:,:,0,i]))-1 for i in range(N_dist_bins)] 
K_CFC_dist_13_stats       = [np.array(bst.CI_from_bootstrap(K_CFC_dist_ps[:,:,1,i]))-1 for i in range(N_dist_bins)] 
K_CFC_dist_12_stats_mod   = [np.array(bst.CI_from_bootstrap(K_CFC_dist_ps_mod[:,:,0,i]))-1 for i in range(N_dist_bins)] 
K_CFC_dist_13_stats_mod   = [np.array(bst.CI_from_bootstrap(K_CFC_dist_ps_mod[:,:,1,i]))-1 for i in range(N_dist_bins)] 
K_CFC_dist_12_stats_mod_w = [np.array(bst.CI_from_bootstrap(K_CFC_dist_ps_mod_w[:,:,0,i],N_boot,2.5,97.5,N_pot_mod_subj[:,:,i]))-1 for i in range(N_dist_bins)] 
K_CFC_dist_13_stats_mod_w = [np.array(bst.CI_from_bootstrap(K_CFC_dist_ps_mod_w[:,:,1,i],N_boot,2.5,97.5,N_pot_mod_subj[:,:,i]))-1 for i in range(N_dist_bins)] 
PLV_ENV_dist_12_stats     = [bst.CI_from_bootstrap(PLV_ENV_dist_ps[:,:,0,i]) for i in range(N_dist_bins)]           # returns [mean, mean_boot, lower, upper] x freq x dist
PLV_ENV_dist_13_stats     = [bst.CI_from_bootstrap(PLV_ENV_dist_ps[:,:,1,i]) for i in range(N_dist_bins)]           
K_ENV_dist_12_stats       = [np.array(bst.CI_from_bootstrap(K_ENV_dist_ps[:,:,0,i]))-1 for i in range(N_dist_bins)] 
K_ENV_dist_13_stats       = [np.array(bst.CI_from_bootstrap(K_ENV_dist_ps[:,:,1,i]))-1 for i in range(N_dist_bins)] 
                

mean_K_CFC               = np.transpose(np.nanmean(K_CFC_ps,0))-1
mean_PLV_CFC             = np.transpose(np.nanmean(PLV_CFC_ps,0))  
mean_K_CFC_local         = np.transpose(np.nanmean(K_CFC_local_ps,0))-1
mean_PLV_CFC_local       = np.transpose(np.nanmean(PLV_CFC_local_ps,0))    
mean_K_CFC_dist          = np.nanmean(K_CFC_dist_ps,0)-1
mean_PLV_CFC_dist        = np.nanmean(PLV_CFC_dist_ps,0)
mean_K_CFC_mod           = np.transpose(np.nanmean(K_CFC_ps_mod,0))-1
mean_PLV_CFC_mod         = np.transpose(np.nanmean(PLV_CFC_ps_mod,0)) 
mean_K_CFC_dist_mod      = np.nanmean(K_CFC_dist_ps_mod,0)-1
mean_PLV_CFC_dist_mod    = np.nanmean(PLV_CFC_dist_ps_mod,0)
mean_PLV_CFC_ps_12       = (PLV_CFC_ps[:,:,0])
mean_PLV_CFC_ps_13       = (PLV_CFC_ps[:,:,1])    
mean_K_CFC_ps_12         = (K_CFC_ps[:,:,0])
mean_K_CFC_ps_13         = (K_CFC_ps[:,:,1])     
mean_PLV_CFC_dist_12     = np.transpose(mean_PLV_CFC_dist[:,0,:])
mean_PLV_CFC_dist_13     = np.transpose(mean_PLV_CFC_dist[:,1,:])    
mean_K_CFC_dist_12       = np.transpose(mean_K_CFC_dist[:,0,:])
mean_K_CFC_dist_13       = np.transpose(mean_K_CFC_dist[:,1,:])
mean_PLV_CFC_ps_12_mod   = (PLV_CFC_ps_mod[:,:,0])
mean_PLV_CFC_ps_13_mod   = (PLV_CFC_ps_mod[:,:,1])    
mean_K_CFC_ps_12_mod     = (K_CFC_ps_mod[:,:,0])
mean_K_CFC_ps_13_mod     = (K_CFC_ps_mod[:,:,1])       
mean_PLV_CFC_dist_12_mod = np.transpose(mean_PLV_CFC_dist_mod[:,0,:])
mean_PLV_CFC_dist_13_mod = np.transpose(mean_PLV_CFC_dist_mod[:,1,:])    
mean_K_CFC_dist_12_mod   = np.transpose(mean_K_CFC_dist_mod[:,0,:])
mean_K_CFC_dist_13_mod   = np.transpose(mean_K_CFC_dist_mod[:,1,:])

 
 

# remove 0 entries in higher ratios
#mean_K_CFC    = list(mean_K_CFC)
#mean_K_CFC    = [np.array(filter(lambda a: a != 0, i)) for i in mean_K_CFC]
mean_PLV_CFC  = list(mean_PLV_CFC)
mean_PLV_CFC  = [np.array(filter(lambda a: a != 0, i)) for i in mean_PLV_CFC]
mean_PLV_ENV  = list(mean_PLV_CFC)
mean_PLV_ENV  = [np.array(filter(lambda a: a != 0, i)) for i in mean_PLV_CFC]
#mean_K_CFC_local     = list(mean_K_CFC_local )
#mean_K_CFC_local     = [np.array(filter(lambda a: a != 0, i)) for i in mean_K_CFC_local]
mean_PLV_CFC_local   = list(mean_PLV_CFC_local )
mean_PLV_CFC_local   = [np.array(filter(lambda a: a != 0, i)) for i in mean_PLV_CFC_local]
mean_K_CFC_mod    = list(mean_K_CFC_mod)
mean_K_CFC_mod    = [np.array(filter(lambda a: a != 0, i)) for i in mean_K_CFC_mod]
mean_PLV_CFC_mod  = list(mean_PLV_CFC_mod)
mean_PLV_CFC_mod  = [np.array(filter(lambda a: a != 0, i)) for i in mean_PLV_CFC_mod]
    
K_CFC_stats_mod_w   = [[np.array(i[~(i==-1)]) for i in j] for j in K_CFC_stats_mod_w]
K_CFC_stats_mod_w   = [[np.array(i[~np.isnan(i)]) for i in j] for j in K_CFC_stats_mod_w]
K_CFC_dist_13_stats = [[np.array(filter(lambda a: a != -1, i)) for i in j] for j in K_CFC_dist_13_stats]
K_ENV_dist_13_stats = [[np.array(filter(lambda a: a != -1, i)) for i in j] for j in K_CFC_dist_13_stats]
K_CFC_dist_13_stats_mod = [[np.array(filter(lambda a: a != -1, i)) for i in j] for j in K_CFC_dist_13_stats_mod]
K_CFC_dist_13_stats_mod_w = [[np.array(filter(lambda a: a != -1, i)) for i in j] for j in K_CFC_dist_13_stats_mod_w]
K_CFC_dist_13_stats     = [[np.array(i[~np.isnan(i)]) for i in j] for j in K_CFC_dist_13_stats]
K_CFC_dist_13_stats_mod = [[np.array(i[~np.isnan(i)]) for i in j] for j in K_CFC_dist_13_stats_mod]
K_CFC_dist_13_stats_mod_w = [[np.array(i[~np.isnan(i)]) for i in j] for j in K_CFC_dist_13_stats_mod_w]



##############################################
####        plot CFC - Figures 2, S7,S8 

o80 = directory + '_results\\MEG CF\\MEG ' + CF_type + ' controlled with '+PS_metric+'.pdf'

figsize = [6.3,2.3]   
rows    = 2
cols    = 3
dataL   = [PLV_CFC_stats[:1],K_CFC_stats[:1],K_CFC_stats_mod_w[:1],
           PLV_CFC_stats[1:],K_CFC_stats[1:],K_CFC_stats_mod_w[1:]]    
xlimA   = [xlims_CF for i in range(6)]
titlesA = ['' for i in range(6)]        #['mean PLV','mean K','mean K (controlled)','','','']
if CF_type == 'CFS':
    ylimA   = [[-0.002,0.045], [-1, 11],   [-1, 11],  [-0.002,0.038], [-0.17,1.56], [-0.17,1.56]]  
else:
    ylimA   = [[-0.002,0.068], [-1, 13.5], [-1,  9],  [-0.002,0.058], [-1,   13.5],[-1,   9]]  
legendA = [ratios[:1],ratios[:1],ratios[:1],
           ratios[1:],ratios[1:],ratios[1:],]    
ylabA   = ['PLV','K [%]','K [%]',
           'PLV','K [%]','K [%]']
cmapA   = ['brg','brg','brg',my_cmap3,my_cmap3,my_cmap3]
legend_posA = ['ur',None,None,None,None,None]
CI      = [0.2 for i in range(6)]
xlab    = [0,0,0,1,1,1]
Ryt     = [1,1,1,1,1,1]
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt)   

#export PDF
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,o80,legend_posA,ylimA,False,1,CI,xlab,Ryt)   


# plot heatmap
o90  = directory + '_results\\MEG CF\\MEG ' + CF_type + ' heatmap, uncontrolled.pdf'
o90a = directory + '_results\\MEG CF\\MEG ' + CF_type + ' heatmap, controlled with '+PS_metric+'.pdf'

data1 = np.transpose(mean_K_CFC)
data2 = np.transpose(mean_K_CFC_mod)
figsize = [1.6,1.9]
if CF_type == 'CFS':
    zmax1  = 8   # CFC
    zmax2  = 8
    ztix1  = [0, 2, 4, 6, 8] # CFC
    ztix2  = [0, 2, 4, 6, 8] # CFC

else:
    zmax1 = 9    # PAC
    zmax2 = 6
    ztix1 = [0,3,6,9]
    ztix2 = [0,2,4,6]
    
LF_ics = [0,4,8,12,16,20,24,28,32,36,40]
LF_map = ['1.1', '2.2', '3.7', '5.9', '9.0', '13.1', '19.7', '28.7', '42.5', '65.3', '95.6']
       
plots.simple_CF_plot(data1,figsize,'ratio','Low Frequency [Hz]',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax1,ztix=ztix1,outfile=None)             
plots.simple_CF_plot(data2,figsize,'ratio','Low Frequency [Hz]',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax2,ztix=ztix2,outfile=None)             
  
# export PDFs 
plots.simple_CF_plot(data1,figsize,'ratio','Low Frequency [Hz]',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax1,ztix=ztix1,outfile=o90)             
plots.simple_CF_plot(data2,figsize,'ratio','Low Frequency [Hz]',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax2,ztix=ztix2,outfile=o90a)             
   







################################################################
####        plot CFC in distance bins - Figure 3
o81 = directory + '_results\\MEG CF\\MEG ' + CF_type + ' per distance, controlled with '+PS_metric+'.pdf'

figsize = [6.3,2.3]   
rows    = 2
cols    = 3
dataL   = [PLV_CFC_dist_12_stats, K_CFC_dist_12_stats, K_CFC_dist_12_stats_mod_w,
           PLV_CFC_dist_13_stats, K_CFC_dist_13_stats, K_CFC_dist_13_stats_mod_w]        
xlimA   = [xlims_CF for i in range(6)]
titlesA = ['mean PLV per distance (1:2)', 'mean K per distance (1:2)', 'mean K per dist. (1:2, contr.)',
           'mean PLV per distance (1:3)', 'mean K per distance (1:3)', 'mean K per dist. (1:3, contr.)']
if CF_type =='CFS':
    ylimA   = [[-0.003,0.045], [-0.5,18],  [-0.5,18], [-0.003,0.045], [-0.2,3.4], [-0.2,3.4]]  ### CFC    
else:
    ylimA   = [[-0.005,0.059], [-1,17],  [-0.5,10], [-0.005,0.054], [-1,17],   [-0.5,10]]  ### PAC             
legendA = [distances for i in range(6)]
ylabA   = ['PLV','K','K','PLV', 'K','K']
cmapA   = ['brg','brg','brg','brg','brg','brg']
legend_posA = [None,None,None,None,None,None]
CI      = [0.3 for i in range(6)]
xlab    = [0,0,0,1,1,1]
Ryt     = [1,1,1,1,1,1]
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt)   

#export PDF
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,o81,legend_posA,ylimA,False,1,CI,xlab,Ryt)   





#####################################
#####    plot ENV  - Figure S9

o80e = directory + '_results\\MEG CF\\MEG Envelopes LFx.pdf'
o80f = directory + '_results\\MEG CF\\MEG Envelopes HFx.pdf'

figsize = [5.3,2.3]  
#figsize = [12.7,3.6]   
 
rows    = 2
cols    = 2
dataL   = [PLV_ENV_stats,K_ENV_stats,
           PLV_ENV_stats,K_ENV_stats]    
xlimA   = [[1,330] for i in range(4)]
titlesA = ['' for i in range(4)] #['mean PLV','mean K','mean K (controlled)','','','']
ylimA   = [[-0.009,0.095],  [-5, 59],[-0.009,0.095],  [-5,59], ] 
legendA = [ratios[:1],ratios[:1],
           ratios[1:],ratios[1:],]    
ylabA   = ['PLV','K [%]', 'PLV','K [%]',]
cmapA   = [my_cmap6,my_cmap6,my_cmap6,my_cmap6]
legend_posA = [None,None,None,None,]
CI      = [0.2 for i in range(4)]
xlab    = [0,0,1,1,]
Ryt     = [1,1,1,1,]

plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,     xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt)   
plots.semi_log_plot_multi2(figsize,rows,cols,dataL,HFs_env,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt)   

## export PDF
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,     xlimA,ylabA,titlesA,cmapA,legendA,o80e,legend_posA,ylimA,False,1,CI,xlab,Ryt)   
plots.semi_log_plot_multi2(figsize,rows,cols,dataL,HFs_env,xlimA,ylabA,titlesA,cmapA,legendA,o80f,legend_posA,ylimA,False,1,CI,xlab,Ryt)   




##################################################
#####        plot local CF - Figure S6  
o82 = directory + '_results\\MEG CF\\MEG ' + CF_type + ' local .pdf'

figsize = [4.5,2.3]   
rows    = 2
cols    = 2
dataL   = [PLV_CFC_local_stats[:1], K_CFC_local_stats[:1], 
           PLV_CFC_local_stats[1:], K_CFC_local_stats[1:]]        
xlimA   = [xlims_CF for i in range(4)]
titlesA = ['' for i in range(4)]  #'mean PLV', 'mean K', '', '']
if CF_type == 'CFS':
    ylimA   = [[-0.004,0.049], [-3,31], [-0.004,0.0491], [-0.5,6.9]]    # CFC    
else:
    ylimA   = [[-0.004,0.069],  [-2,34], [-0.004,0.069],  [-2,34]]     # PAC       
legendA = [ratios[:1],ratios[:1],
           ratios[1:],ratios[1:],] 
ylabA   = ['PLV','K [%]','PLV','K [%]']
cmapA   = ['brg','brg',my_cmap3,my_cmap3]
legend_posA = [None,None,None,None]
CI      = [0.2,0.2,0.2,0.2]
xlab    = [0,0,1,1]
Ryt     = [1,1,1,1]
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt)   

plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,o82,legend_posA,ylimA,False,1,CI,xlab,Ryt)   



# plot heatmap
o93 = directory + '_results\\MEG CF\\MEG ' + CF_type + ' local heatmap.pdf'
data = np.transpose(mean_K_CFC_local)
figsize_hm = [1.6,1.9]

LF_ics = [0,4,8,12,16,20,24,28,32,36,40]
LF_map = ['1.1', '2.2', '3.7', '5.9', '9.0', '13.1', '19.7', '28.7', '42.5', '65.3', '95.6']
zmax = 24
ztix=[0,6,12,18,24]


plots.simple_CF_plot(data,figsize_hm,'ratio','LF',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax,ztix=ztix,outfile=None)             
   
# export PDF 
plots.simple_CF_plot(data,figsize_hm,'ratio','LF',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax,ztix=ztix,outfile=o93)          









###############################################################################
##############################  analysis of location  #########################

    


# analyze local CFC

N_sign_local_pp    = np.zeros([N_LF,N_ratios,N_parc])
PLV_local_pp       = np.zeros([N_LF,N_ratios,N_parc])

for s,ss in enumerate(subject_sets):   
    for lf,LF in enumerate(LFs):                                               #### get sign CFC edges per parcel pair ####
        for hf,HF in enumerate(HFs[lf]):   
            data_local      = np.diagonal(CFC[s][lf][hf].data_masked)
            data_local_sign = np.diagonal(CFC[s][lf][hf].data_sign)
            for ch in range(200):
                N_sign_local_pp[lf][hf][ch] += data_local_sign[ch]>0
                PLV_local_pp[lf][hf][ch]      += data_local[ch]

E = cffun.Bunch()
E.mean_PLV_local_pp  = PLV_local_pp     /N_sets
E.K_local_pp         = N_sign_local_pp  /N_sets  
E.mean_PLV_local_pn  = np.array([np.nanmean(E.mean_PLV_local_pp [:,:,networks[i]],2) for i in range(7)])
E.K_local_pn         = np.array([np.nanmean(E.K_local_pp   [:,:,networks[i]],2) for i in range(7)])

cffun.write_csv_local(directory,E,ratios2,'parc2009_200AFS',CF_type)



freq_bands    = [range(0,9),range(9,15),range(15,22),range(22,29),range(29,34),range(34,41)]
freq_clusters = [range(14,22)]        # only alpha


######## compare long and short distance bins
    
# get CFC values
for lf,LF in enumerate(LFs): 
   for hf,HF in enumerate(HFs[lf]):
           for s in range(N_sets):                                      
               for d in range(N_dist_bins): 
                       PLV_CFC_dist_ps[s,lf,hf,d]      = CFC_dist[s][lf][hf][d].mean_masked
                       K_CFC_dist_ps[s,lf,hf,d]        = CFC_dist[s][lf][hf][d].K
                       K_CFC_dist_ps_mod[s,lf,hf,d]    = CFC_dist[s][lf][hf][d].K_mod
                      # K_CFC_dist_ps_excl[s,lf,hf,d]   = CFC_dist[s][lf][hf][d].K_excl
                                              
wilc_pm     = np.zeros([N_LF,2])
wilc_p      = np.zeros([N_LF,2])
wilc_p_mod  = np.zeros([N_LF,2])

for lf,LF in enumerate(LFs): 
   for hf in range(2):
      aaa, wilc_pm[lf,hf]   = stat.wilcoxon(PLV_CFC_dist_ps[:,lf,hf,0],  PLV_CFC_dist_ps[:,lf,hf,2])
      aaa, wilc_p[lf,hf]     = stat.wilcoxon(K_CFC_dist_ps[:,lf,hf,0],    K_CFC_dist_ps[:,lf,hf,2])
      aaa, wilc_p_mod[lf,hf] = stat.wilcoxon(K_CFC_dist_ps_mod[:,lf,hf,0],K_CFC_dist_ps_mod[:,lf,hf,2])
  
s_12_ps  = 1.*multicomp.multipletests(wilc_pm[2:,0]     ,method ='fdr_bh')[0]
s_13_ps  = 1.*multicomp.multipletests(wilc_pm[2:24,1]   ,method ='fdr_bh')[0]
s_12     = 1.*multicomp.multipletests(wilc_p[2:,0]       ,method ='fdr_bh')[0]
s_13     = 1.*multicomp.multipletests(wilc_p[2:24,1]     ,method ='fdr_bh')[0]
s_12_mod = 1.*multicomp.multipletests(wilc_p_mod[2:,0]   ,method ='fdr_bh')[0]
s_13_mod = 1.*multicomp.multipletests(wilc_p_mod[2:24,1] ,method ='fdr_bh')[0]


o83 = directory + '_results\\MEG CF\\MEG ' + CF_type + ' with ' + PS_metric + ' control, distance comparison.pdf'
dataA = [[s_12_ps],[s_12],[s_12_mod],[s_13_ps],[s_13],[s_13_mod]]
cmapA = ['brg','brg','brg','brg','brg','brg']
xlimA   = [xlims_CF for i in range(6)]
plots.semi_log_plot_multi([7.7,3],2,3,dataA,LFs,xlimA,['','','','','',''],['1-2','1-2','1-2 c','1-3','1-3','1-3 c'],cmapA,None,None,None,None,True,1,None,None,None,8,3)   

# save pdf
plots.semi_log_plot_multi([7.7,3],2,3,dataA,LFs,xlimA,['','','','','',''],['1-2','1-2','1-2 c','1-3','1-3','1-3 c'],cmapA,None,o83,None,None,0,1,None,None,None,8,3)   













##################################################################################
################ count edges and compare low-high strength across parcels - for Figure 6
    
diag_zero_mask = np.ones([N_parc,N_parc])
np.fill_diagonal(diag_zero_mask,0) 

edges = cffun.Bunch()
edges.PLV_CF_pp_all      = [[[[[] for b in range(N_parc)] for i in range(N_parc)] for j in range(N_ratios)] for k in range(N_LF)]
edges.PLV_CF_pp_all_TP   = [[[[[] for b in range(N_parc)] for i in range(N_parc)] for j in range(N_ratios)] for k in range(N_LF)]
edges.PLV_CF_pp_all_pn   = [[[[[] for b in range(N_networks)] for i in range(N_parc)] for j in range(N_ratios)] for k in range(N_LF)]


N_rat = 2      
for s,subject in enumerate(subjects):                #### get CFC edges per parcel pair ####
    for lf,LF in enumerate(LFs):                                              
        for hf,HF in enumerate(HFs[lf]):   
            data_masked = CFC[s][lf][hf].data_masked * diag_zero_mask
            for p1 in range(N_parc):
                for p2 in range(N_parc):
                    val = data_masked[p1,p2]
                    n1  = network_indices[p1]
                    n2  = network_indices[p2]
                    edges.PLV_CF_pp_all   [lf][hf][p1][p2].append(val)            
                    edges.PLV_CF_pp_all_TP[lf][hf][p2][p1].append(val)
                    edges.PLV_CF_pp_all_pn[lf][hf][n1][n2].append(val)  
    print subject       
N,lh = cffun.low_to_high_analysis_MEG(edges,LFs,HFs,networks,network_indices,alpha=0.05,N_perm=1000,N_rat=N_rat)
cffun.write_csv_low_to_high(directory,lh,ratios2,'parc2009_200AFS',CF_type,add_inf='')

fileout41 = directory + '_results\\_pickle dump\\Low-high ' + CF_type + ', ' + PS_metric + ', N_rat=' + str(N_rat) + ', ' + parcellation + ' ' + time.strftime("%Y-%m-%d") + '.dat'  # save with pickle
pick.dump(lh,open(fileout41,'wb'))



for s,subject in enumerate(subjects):                #### get sign mod CFC edges per parcel pair ####
    for lf,LF in enumerate(LFs):                                              
        for hf,HF in enumerate(HFs[lf]):   
            data_masked = CFC[s][lf][hf].data_sign_mod * diag_zero_mask
            for p1 in range(N_parc):
                for p2 in range(N_parc):
                    val = data_masked[p1,p2]
                    n1  = network_indices[p1]
                    n2  = network_indices[p2]
                    edges.PLV_CF_pp_all   [lf][hf][p1][p2].append(val)            
                    edges.PLV_CF_pp_all_TP[lf][hf][p2][p1].append(val)
                    edges.PLV_CF_pp_all_pn[lf][hf][n1][n2].append(val)  
    print subject
N,lh = cffun.low_to_high_analysis_MEG(edges,LFs,HFs,networks,network_indices,alpha=0.05,N_perm=1000,N_rat=N_rat)
cffun.write_csv_low_to_high(directory,lh,ratios2,'parc2009_200AFS',CF_type,add_inf='sign_mod')

fileout42 = directory + '_results\\_pickle dump\\Low-high ' + CF_type + ', ' + PS_metric + ', N_rat=' + str(N_rat) + ', ' + parcellation + ' ' + time.strftime("%Y-%m-%d") + '.dat'  # save with pickle
pick.dump(lh,open(fileout42,'wb'))


for s,subject in enumerate(subjects):                #### get sign CFC edges per parcel pair ####
    for lf,LF in enumerate(LFs):                                              
        for hf,HF in enumerate(HFs[lf]):   
            data_masked = CFC[s][lf][hf].data_sign * diag_zero_mask
            for p1 in range(N_parc):
                for p2 in range(N_parc):
                    val = data_masked[p1,p2]
                    n1  = network_indices[p1]
                    n2  = network_indices[p2]
                    edges.PLV_CF_pp_all   [lf][hf][p1][p2].append(val)            
                    edges.PLV_CF_pp_all_TP[lf][hf][p2][p1].append(val)
                    edges.PLV_CF_pp_all_pn[lf][hf][n1][n2].append(val)  
    print subject
N,lh = cffun.low_to_high_analysis_MEG(edges,LFs,HFs,networks,network_indices,alpha=0.05,N_perm=1000,N_rat=N_rat)
cffun.write_csv_low_to_high(directory,lh,ratios2,'parc2009_200AFS',CF_type,add_inf='sign')

fileout43 = directory + '_results\\_pickle dump\\Low-high ' + CF_type + ', ' + PS_metric + ', N_rat=' + str(N_rat) + ', ' + parcellation + ' ' + time.strftime("%Y-%m-%d") + '.dat'  # save with pickle
pick.dump(lh,open(fileout43,'wb'))









##################################################################################
##########################  get degrees - for Figure 5


degrees_subj             = np.zeros([N_sets,N_freq,N_parc])
degrees_subj_LF          = np.zeros([N_sets,N_LF,N_ratios,N_parc])
degrees_subj_HF          = np.zeros([N_sets,N_LF,N_ratios,N_parc])
degrees_subj_LF_mod      = np.zeros([N_sets,N_LF,N_ratios,N_parc])
degrees_subj_HF_mod      = np.zeros([N_sets,N_LF,N_ratios,N_parc])

for s,sset in enumerate(subject_sets):     
    for f,freq in enumerate(freqs):
        degrees_subj[s,f,:]        =  PS[s][f].degree     
    for lf,LF in enumerate(LFs):
        for hf,HF in enumerate(HFs[lf]):
            degrees_subj_LF[s,lf,hf,:]         = CFC[s][lf][hf].degree_LF
            degrees_subj_HF[s,lf,hf,:]         = CFC[s][lf][hf].degree_HF            
            degrees_subj_LF_mod[s,lf,hf,:]     = CFC[s][lf][hf].degree_LF_mod
            degrees_subj_HF_mod[s,lf,hf,:]     = CFC[s][lf][hf].degree_HF_mod

degrees_all         = np.mean(degrees_subj,     0)
degrees_all_LF      = np.mean(degrees_subj_LF,  0)
degrees_all_HF      = np.mean(degrees_subj_HF,  0)
degrees_all_LF_mod  = np.mean(degrees_subj_LF_mod,  0)
degrees_all_HF_mod  = np.mean(degrees_subj_HF_mod,  0)
degrees_all_CF      = degrees_all_LF + degrees_all_HF
degrees_all_CF_mod  = degrees_all_LF_mod + degrees_all_HF_mod

file_deg_PS      =  directory + '_results\\graph metrics 2018-05\\'+ CF_type + ' degrees ' + parcellation + '\\PS ' + PS_metric + '.csv'
file_deg_LF      = [directory + '_results\\graph metrics 2018-05\\'+ CF_type + ' degrees ' + parcellation + '\\CF LF ' + i + '.csv' for i in ratios2]
file_deg_HF      = [directory + '_results\\graph metrics 2018-05\\'+ CF_type + ' degrees ' + parcellation + '\\CF HF ' + i + '.csv' for i in ratios2]
file_deg_CF      = [directory + '_results\\graph metrics 2018-05\\'+ CF_type + ' degrees ' + parcellation + '\\CF '     + i + '.csv' for i in ratios2]
file_deg_LF_mod  = [directory + '_results\\graph metrics 2018-05\\'+ CF_type + ' degrees ' + parcellation + '\\CF LF mod ' + i + '.csv' for i in ratios2]
file_deg_HF_mod  = [directory + '_results\\graph metrics 2018-05\\'+ CF_type + ' degrees ' + parcellation + '\\CF HF mod ' + i + '.csv' for i in ratios2]
file_deg_CF_mod  = [directory + '_results\\graph metrics 2018-05\\'+ CF_type + ' degrees ' + parcellation + '\\CF mod '     + i + '.csv' for i in ratios2]

np.savetxt(file_deg_PS, degrees_all, delimiter=";")
for i in range(N_ratios):        
    np.savetxt(file_deg_LF[i], degrees_all_LF  [:,i,:], delimiter=";")
    np.savetxt(file_deg_HF[i], degrees_all_HF  [:,i,:], delimiter=";")
    np.savetxt(file_deg_CF[i], degrees_all_CF  [:,i,:], delimiter=";")    
    np.savetxt(file_deg_LF_mod[i], degrees_all_LF_mod  [:,i,:], delimiter=";")
    np.savetxt(file_deg_HF_mod[i], degrees_all_HF_mod  [:,i,:], delimiter=";")
    np.savetxt(file_deg_CF_mod[i], degrees_all_CF_mod  [:,i,:], delimiter=";")
   













