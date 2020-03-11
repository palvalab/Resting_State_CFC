# -*- coding: utf-8 -*-
"""
@author: Felix Siebenhühner
"""


source_directory     = 'point to source directory here'
projects_directory   = 'point to SEEG data directory here'

import numpy as np
import csv
import cPickle as pick
import sys
import copy
sys.path.append(source_directory + 'Python27\\Utilities')
import CF_functions as cffun
import plot_functions as plots
import matplotlib.pyplot as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import bootstrap as bst
from scipy import stats as stat
import statsmodels.sandbox.stats.multicomp as multicomp


### CF type and PS metric

CF_type     = 'CFS'                             # 'PAC' or 'CFS'
PS_metric   = 'wPLI'                            # 'PLV' or 'wPLI'

sign_z_CFC  = 2.42
if PS_metric == 'PLV':
    sign_z_PS   = 2.42                                
else:
    sign_z_PS = 2.33



### subject, frequency and parcellation settings
subjects = ['S' + str(i+1).zfill(3) for i in range(59)]
   
directory     = projects_directory+'resting_state\\RS_CF_SEEG\\'


CFM_filename  = directory + '_support_files\\CF_matrix_SEEG.csv'
freqs_filename= directory + '_support_files\\all_frequencies_SEEG.csv'
CFM           = np.genfromtxt(CFM_filename, delimiter=';')    
LFs           = CFM[:,0]
freqs         = np.genfromtxt(freqs_filename, delimiter=';')                                                

parc      = 'parc2009'       
N_freq    = len(freqs)
N_LF      = len(LFs)
N_subj    = len(subjects)
N_ratios  = 6  
cutoff_PS = 100      
cutoff_CF = 350    
xlims_PS  = [1,100]
xlims_CFC = [1,100]  

HFs       = []     
for f in range(len(LFs)):
    x   = CFM[f,1:N_ratios+1]
    xx = x[np.intersect1d(np.where(x<cutoff_CF),np.where(x>0)) ] 
    if len(xx) >0:
        HFs.append(xx)    
HFs_env = [CFM[:30,i+1] for i in range(6)] 

freq_bands    = [range(0,5),range(5,11),range(11,18),range(18,25),range(25,N_freq)]
freq_bands_LF = [range(1,6),range(3,10),range(9,16),range(16,22),range(22,N_LF)]
   
ratios  = ['1:'+str(i+2) for i in range(N_ratios)]    
ratios2 = ['1-'+str(i+2) for i in range(N_ratios)]   


### get epileptic-contact-masks and distance masks

dist_thresholds = [0.02]    
all_dist = np.empty(0)
dist           = [None for i in subjects]
masks          = [None for i in subjects]
ch_per_subject = [None for i in subjects]
for s,subject in enumerate(subjects):
     dist_filename = directory + '_support_files\\distances\\' + subject + '.csv'
     dist[s]       = np.genfromtxt(dist_filename, delimiter=';') 
     mask_filename = directory + '_support_files\\masks\\' + subject + '.csv'
     masks[s]      = np.genfromtxt(mask_filename, delimiter=';') 
     d2            = dist[s]*masks[s]
     all_dist      = np.append(all_dist,d2.reshape(len(dist[s])**2))                      
     ch_per_subject[s] = len(list(masks[s]))
all_dist = all_dist[np.where(all_dist>0)]  
dist_thresholds.extend(np.percentile(all_dist,[33.3333,66.66667]))
dist_max = max(all_dist)
dist_thresholds.extend([dist_max])
dist_strs = ['{:.1f}'.format(d*100) for d in dist_thresholds]  
N_dist_bins = len(dist_thresholds)-1   
distances = [dist_strs[i]+'-'+dist_strs[i+1]+'cm' for i in range(N_dist_bins)]

dists_short = ['short','mid','long']




### get GMPI info

GMPI_vals     = [[] for i in subjects]
GMPI_list     = [[] for i in subjects]
GMPI_vals_all = []

for s,subject in enumerate(subjects):
     gmpi_filename = directory + '_support_files\\gmpi\\' + subject + '.csv'
     with open(gmpi_filename, 'rb') as csvfile:
         reader = csv.reader(csvfile, delimiter = ';')
         for row in reader:
             GMPI_list[s].append(row)   
             GMPI_vals[s].append(float(row[1]))
             GMPI_vals_all.append(float(row[1]))
             
             
### get layer interaction masks   
             
GMPI_vals_all=filter(lambda v: v==v, GMPI_vals_all)             # remove nans       
N_layer_int       = 4   
N_layer           = 3      
layer_int_masks   = [[None for j in range(N_layer_int)] for i in subjects]        # 0: deep-to-deep, 1: sup-to-sup, 2: deep-to-sup
N_pairs_layer_int = [[None for j in range(N_layer_int)] for i in subjects]
layer_int         = ['superf-surf','deep-deep','superf-deep','deep-superf']
layers            = ['superf','interm','deep']
channel_layers    = [None for s in subjects]
N_ch_layer_s      = np.zeros([N_subj,N_layer])

for s,subject in enumerate(subjects):
     channel_layers[s] = np.full(ch_per_subject[s],np.nan)
     for l in range(N_layer_int):   
        layer_int_masks[s][l] = np.zeros([ch_per_subject[s],ch_per_subject[s]])     
     for ch1,g1 in enumerate(GMPI_vals[s]):
        if ( 0.5 < g1 < 1.2):
             channel_layers[s][ch1] = 0              # surface
        if ( 0   < g1 < 0.5):
             channel_layers[s][ch1] = 1              # intermed.
        if (-0.3 < g1 < 0 ): 
             channel_layers[s][ch1] = 2              # deep
     for ch1,g1 in enumerate(GMPI_vals[s]):             
        for ch2,g2 in enumerate(GMPI_vals[s]):
            if (0.5 < g1 < 1.2 and 0.5 < g2 < 1.2 ):           # surface to surface
                layer_int_masks[s][0][ch1,ch2]=1
            if (-0.3 < g1 < 0 and -0.3 < g2 < 0 ):             # deep to deep
                layer_int_masks[s][1][ch1,ch2]=1 
            if (0.5 < g1 < 1.2 and -0.3  < g2 < 0):            # surf to deep
                layer_int_masks[s][3][ch1,ch2]=1            
            if (-0.3 < g1 < 0 and 0.5 < g2 < 1.2 ):            # deep to surface
                layer_int_masks[s][2][ch1,ch2]=1
     
     for l in range(N_layer_int):   
        mask1 = copy.copy(masks[s])
        layer_int_masks[s][l]   = layer_int_masks[s][l]*mask1        
        N_pairs_layer_int[s][l] = int(np.sum(layer_int_masks[s][l])) - np.sum(np.diag(layer_int_masks[s][l])>0)   
     for l in range(N_layer):           
        N_ch_layer_s[s,l] = np.sum(channel_layers[s]==l)

N_ch_layer   = np.nansum(N_ch_layer_s,0)
   


### colormaps for plotting
my_cmap  = plots.make_cmap([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.8, 0.0, 1.0)])
my_cmap2 = plots.make_cmap([(0.0, 0.0, 0.0), (0.5, 0.5, 1.0), (0.6, 0.6, 1.0), (0.7, 0.7, 1.0), (0.8, 0.8, 1.0),(0.9, 0.9, 1.0), (1, 1, 1)])
my_cmap3 = plots.make_cmap([(1.0, 0.0, 0.0), (0.0, 0.6, 0.0), (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (0.6, 0.4, 0.4)]) 
my_cmap4 = plots.make_cmap([(0.8, 0.6, 0.0), (1.0, 0.0, 0.0), (0.0, 0.8, 0.0), (0.1, 0.1, 0.1), (1.0, 0.4, 0.9), (0.0, 0.0, 1.0), (0.8, 0.0, 0.9)])
my_cmap5 = plots.make_cmap([(1,0,0), (0,1,0), (0,0,1)])
my_cmap6 = plots.make_cmap([(1,0,0), (0,0.7,0), (0,0,1), (1, 0.4, 0.4), (0.4,1,0.4), (0.4,0.4,1) ])
my_cmap7 = plots.make_cmap([(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.6, 0.0), (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (0.6, 0.4, 0.4)]) 



### set matplotlib parameters    
mpl.rcParams['pdf.fonttype'] = 42                 # for PDF compatibility with Illustrator
mpl.rcParams.update({'font.size': 8})
mpl.rcParams.update({'axes.titlesize': 8})
mpl.rcParams.update({'axes.labelsize': 8})
mpl.rcParams.update({'legend.fontsize': 6})
mpl.rcParams.update({'xtick.labelsize': 7})
mpl.rcParams.update({'ytick.labelsize': 7})




### initialize lists
PS           =  [[   None for i in range(N_freq)] for j in range(N_subj)]    
PS_dist      =  [[[  None for b in range(N_dist_bins)] for i in range(N_freq)] for j in range(N_subj)]  
PS_layer     =  [[[  None for b in range(N_layer_int)] for i in range(N_freq)] for j in range(N_subj)] 
CFC          =  [[[  None for i in range(N_ratios)] for j in range(N_LF)] for k in range(N_subj)]
CFC_dist     =  [[[[ None for b in range(N_dist_bins)] for i in range(N_ratios)] for j in range(N_LF)] for k in range(N_subj)]
CFC_layer    =  [[[[ None for b in range(N_layer_int)] for i in range(N_ratios)] for j in range(N_LF)] for k in range(N_subj)]
PS_ENV       =  [[[  None for i in range(N_ratios)] for j in range(N_LF)] for k in range(N_subj)]
PS_ENV_dist  =  [[[[ None for b in range(N_dist_bins)] for i in range(N_ratios)] for j in range(N_LF)] for k in range(N_subj)]
PS_ENV_layer =  [[[[ None for b in range(N_layer_int)] for i in range(N_ratios)] for j in range(N_LF)] for k in range(N_subj)]


      
    

#### compute PS single-subject stats    
for s,subject in enumerate(subjects): 
    for f,F in enumerate(freqs):  
        F_str     = '{:.2f}'.format(F)             
        mask      = copy.copy(masks[s])
        N_pot     = np.nansum(mask)     
        if PS_metric == 'wPLI':                                  
            file1     = directory + '_data\\_PS_wPLI\\' + subject + ' f=' + F_str + '.csv'
            file2     = directory + '_data\\_PS_wPLI\\' + subject + ' f=' + F_str + '_surr.csv'
        else:  
            file1     = directory + '_data\\_PS_PLV\\'  + subject + ' f=' + F_str + '.csv'
            file2     = directory + '_data\\_PS_PLV\\'  + subject + ' f=' + F_str + '_surr.csv'
        data      = mask*np.genfromtxt(file1, delimiter=';') 
        data_surr = mask*np.genfromtxt(file2, delimiter=';') 
        stats = cffun.K_stats_PS_2(data,data_surr,sign_z_PS,PS_metric)

        PS[s][f] = stats 
        
        for d in range(N_dist_bins):
            dist_mask        = mask * ( ( (dist[s]>dist_thresholds[d]) * (dist[s]<=dist_thresholds[d+1]) )>0)
            N_potD           = np.nansum(dist_mask)                                       
            dataD            = data*dist_mask       
            if N_potD>0:
                stats      = cffun.K_stats_PS_2(dataD,data_surr,sign_z_PS,PS_metric)    
            else:
                stats = cffun.Stats_PS(np.nan)    
            PS_dist[s][f][d] = stats

        for l in range(N_layer_int):
            layer_mask       = mask * layer_int_masks[s][l]
            N_potL           = np.nansum(layer_mask)                                        
            dataL            = data*layer_mask
            if N_potL>0:
                stats      = cffun.K_stats_PS_2(dataL,data_surr,sign_z_PS,PS_metric)  
            else:
                stats = cffun.Stats_PS(np.nan)    
            PS_layer[s][f][l] = stats
                      
    print(time.strftime("%Y-%m-%d %H:%M") + '          ' + subject)
 

#### for PAC: compute PLV single-subject stats of LF-envelope filtered HF data 
if CF_type == 'PAC':
    for s,subject in enumerate(subjects): 
        mask=masks[s]  
        for lf,LF in enumerate(LFs):
            for hf,HF in enumerate(HFs[lf]):  
                np.fill_diagonal(mask,1)      
                path        = directory + '_data\\_ENV\\'
                LF_str      = '{:.2f}'.format(LF)                
                HF_str      = '{:.2f}'.format(HF)  
                file1       = path + subject + ' LF= ' + LF_str + ' HF= ' + HF_str + '.csv'            
                file2       = path + subject + ' LF= ' + LF_str + ' HF= ' + HF_str + '_surr.csv'
                N_pot       = np.nansum(mask)                
                data        = mask*np.genfromtxt(file1, delimiter=';') 
                data_surr   = mask*np.genfromtxt(file2, delimiter=';') 
                stats       = cffun.K_stats_PS_2(data,data_surr,sign_z_PS,PS_metric)
                PS_ENV[s][lf][hf]= stats                 
                for d in range(N_dist_bins):
                    dist_mask        = mask * ( ( (dist[s]>dist_thresholds[d]) * (dist[s]<=dist_thresholds[d+1]) )>0)
                    N_potD           = np.nansum(dist_mask)                                       
                    dataD            = data*dist_mask       
                    if N_potD>0:
                        stats      = cffun.K_stats_PS_2(dataD,data_surr,sign_z_PS,PS_metric)    
                    else:
                        stats = cffun.Stats_PS(np.nan)    
                    PS_ENV_dist[s][lf][hf][d] = stats
    
                for l in range(N_layer_int):
                    layer_mask       = mask * layer_int_masks[s][l]
                    N_potL           = np.nansum(layer_mask)                                         
                    dataL            = data*layer_mask
                    if N_potL>0:
                        stats      = cffun.K_stats_PS_2(dataL,data_surr,sign_z_PS,PS_metric)   
                    else:
                        stats = cffun.Stats_PS(np.nan)    
                    PS_ENV_layer[s][lf][hf][l] = stats
                              
        print(time.strftime("%Y-%m-%d %H:%M") + '          ' + subject)
 

############################################################################################################################



############ SEEG: compute CFC or PAC stats ############                                    
for s,subject in enumerate(subjects):    
    mask=copy.copy(masks[s])
    for lf,LF in enumerate(LFs):
        for hf,HF in enumerate(HFs[lf]):  
                np.fill_diagonal(mask,1)                            # for local!
                LF_str       = '{:.2f}'.format(LF)                
                HF_str       = '{:.2f}'.format(HF)  
                LF_idx       = np.where(freqs==LF)[0][0]                    
                LF_PS        = PS[s][LF_idx].data_sign              
                if CF_type == 'CFS':                
                    HF_idx       = np.where(freqs==HF)[0][0]                 
                    HF_PS    = PS[s][HF_idx].data_sign
                    path     = directory + '_data\\_CFS\\'  
                    file0       = path + subject + ' LF=' + LF_str + ' HF=' + HF_str + '.csv'  
                    file_surr   = path + subject + ' LF=' + LF_str + ' HF=' + HF_str + '_surr.csv'
                else: 
                    HF_PS   = PS_ENV[s][lf][hf].data_sign
                    path    = directory + '_data\\_PAC\\'              
                    file0       = path + subject + ' LF= ' + LF_str + ' HF= ' + HF_str + '.csv'  
                    file_surr   = path + subject + ' LF= ' + LF_str + ' HF= ' + HF_str + '_surr.csv'
                masked_data   = np.genfromtxt(file0,  delimiter=';') * mask
                surr_data   = np.genfromtxt(file_surr, delimiter=';') * mask
                np.fill_diagonal(mask,0)                            
                N_CH        = len(mask)
                N_pot       = np.nansum(masked_data>0) - np.trace(masked_data>0)  
                if N_pot > 0 and np.nansum(masked_data)-np.trace(masked_data)>0:    
                   stats   = cffun.K_stats_CFC_2(masked_data,surr_data,sign_z_CFC,LF_PS,HF_PS) 
                else:
                    stats       = cffun.Stats_CFC(np.nan)
                CFC[s][lf][hf] = stats 
                
                for d in range(len(distances)):
                    dist_mask        = mask * ((dist[s]>dist_thresholds[d]) * (dist[s]<=dist_thresholds[d+1])) >0
                    np.fill_diagonal(dist_mask,1)                                                           # 1 so that local CFC is preserved in data
                    masked_dataD     = masked_data * dist_mask                        
                    np.fill_diagonal(dist_mask,0)                                                           # 0 so that local CFC is removed in subfunction
                    N_potD = np.sum(masked_dataD>0)-np.trace(masked_dataD>0)
                    if N_potD>0:                     
                        upper_idx     = np.triu_indices(N_CH)
                        statsD   = cffun.K_stats_CFC_2(masked_dataD,surr_data,sign_z_CFC,LF_PS,HF_PS) 
                    else:
                        statsD        = cffun.Stats_CFC(np.nan)          # set K to 0 if no edges in dist mask
                    CFC_dist[s][lf][hf][d] = statsD   
                    
                for l in range(N_layer_int):
                    layer_mask        = mask * layer_int_masks[s][l]
                    np.fill_diagonal(layer_mask,1)                                                           # 1 so that local CFC is preserved in data
                    masked_dataL     = masked_data * layer_mask    
                    N_potL = np.sum(masked_dataL>0)-np.trace(masked_dataL>0)
                    if N_potL>0:                     
                        upper_idx     = np.triu_indices(N_CH)
                        statsL   = cffun.K_stats_CFC_2(masked_dataL,surr_data,sign_z_CFC,LF_PS,HF_PS)  
                    else:
                        statsL        = cffun.Stats_CFC(np.nan)          # set K to 0 if no edges in dist mask
                    CFC_layer[s][lf][hf][l] = statsL   
                    
    print(time.strftime("%Y-%m-%d %H:%M") + '          ' + subject)





















###############################################################################
#################            SAVE DATA WITH PICKLE            #################


save=1
if save:
    fileout1 = directory + '_results\\_pickle dump\\SEEG ' + PS_metric + ', z=' + '{:.2f}'.format(sign_z_PS) + ', ' + time.strftime("%Y-%m-%d") + '.dat'
    fileout2 = directory + '_results\\_pickle dump\\SEEG ' + CF_type + ' corrected with ' + PS_metric + ', z=' + '{:.2f}'.format(sign_z_PS) + ', '  + time.strftime("%Y-%m-%d") + '.dat'
    fileout3 = directory + '_results\\_pickle dump\\SEEG ' + CF_type + ' by layer interaction corrected with ' + PS_metric +', z=' + '{:.2f}'.format(sign_z_PS) + ', '  + time.strftime("%Y-%m-%d") + '.dat'
    fileout4 = directory + '_results\\_pickle dump\\SEEG PLV Envelope ' + time.strftime("%Y-%m-%d") + '.dat'
    pick.dump([PS,PS_dist],open(fileout1,'wb'))
    pick.dump([CFC,CFC_dist],open(fileout2,'wb'))
    pick.dump([PS_layer,CFC_layer],open(fileout3,'wb'))   
    if CF_type=='PAC':
            pick.dump([PS_ENV,PS_ENV_dist,PS_ENV_layer],open(fileout4,'wb'))   




###############################################################################    
##############                     LOAD DATA                  #################
    
use_IMs = False                                   # only need to be loaded for degree and directionality analyses
    
if use_IMs:
    file_in1 = directory + '_results\\_pickle dump\\SEEG ' + PS_metric + '.dat'
    file_in2 = directory + '_results\\_pickle dump\\SEEG ' + CF_type + ' corrected with ' + PS_metric + '.dat'    
    file_in3 = directory + '_results\\_pickle dump\\SEEG ' + CF_type + ' by layer interaction corrected with ' + PS_metric + '.dat'    
else:  
    file_in1 = directory + '_results\\_pickle dump\\SEEG ' + PS_metric + '_no_IMS' + '.dat'
    file_in2 = directory + '_results\\_pickle dump\\SEEG ' + CF_type + ' corrected with ' + PS_metric + '_no_IMS'  + '.dat'    
    file_in3 = directory + '_results\\_pickle dump\\SEEG ' + CF_type + ' by layer interaction corrected with ' + PS_metric + '_no_IMS'  + '.dat'


PS,PS_dist            = pick.load(open(file_in1,'rb'))
CFC,CFC_dist          = pick.load(open(file_in2,'rb'))
PS_layer,CFC_layer    = pick.load(open(file_in3,'rb'))

file_in4 = directory + '_results\\_pickle dump\\SEEG PLV Envelope.dat'
PS_ENV,PS_ENV_dist,PS_ENV_layer    = pick.load(open(file_in4,'rb'))






###############################################################################
###############################################################################
#########                         PLOT PS                  
                
    

### get numbers of edges for PS 
N_edges                  = 0
N_edges_dist             = np.zeros([N_dist_bins])
N_edges_layer            = np.zeros([N_layer_int])
N_edges_subj             = np.zeros(N_subj)
N_edges_dist_subj        = np.zeros([N_subj,N_dist_bins])
N_edges_layer_subj       = np.zeros([N_subj,N_layer_int])
N_CH_subj                = np.zeros(N_subj)
N_subj_contr             = 0                        # number of contributing subjects
N_subj_contr_dist        = np.zeros(N_dist_bins)
N_layer_contr_dist       = np.zeros(N_layer_int)

for s in range(N_subj):    
    N_edges                 += np.nan_to_num(PS[s][0].N_pot)        
    N_subj_contr            += np.nan_to_num(int(PS[s][0].N_pot>0))
    N_edges_subj[s]          = PS[s][0].N_pot

    for d in range(N_dist_bins):
        N_edges_dist[d]                += np.nan_to_num(PS_dist[s][0][d].N_pot)
        N_edges_dist_subj[s,d]          = np.nan_to_num(PS_dist[s][0][d].N_pot)
        N_subj_contr_dist[d]           += np.nan_to_num(int(PS_dist[s][0][d].N_pot>0))
        
    for l in range(N_layer_int):
        N_edges_layer[l]                += np.nan_to_num(PS_layer[s][0][l].N_pot)
        N_edges_layer_subj[s,l]          = np.nan_to_num(PS_layer[s][0][l].N_pot)
        N_layer_contr_dist[l]           += np.nan_to_num(int(PS_layer[s][0][l].N_pot>0))
        
         
### set divisors for mean calculation         
div1 = N_edges
div2 = N_edges_dist       
div3 = N_edges_subj  
div4 = N_edges_layer
      
### init PS arrays
PLV_PS_ps        = np.zeros([N_subj,N_freq])                           # PS = "Phase Synch", ps = "per subject"
K_PS_ps          = np.zeros([N_subj,N_freq])
PLV_PS_dist_ps   = np.zeros([N_subj,N_freq,N_dist_bins])
K_PS_dist_ps     = np.zeros([N_subj,N_freq,N_dist_bins])
PLV_PS_layer_ps  = np.zeros([N_subj,N_freq,N_layer_int])
K_PS_layer_ps    = np.zeros([N_subj,N_freq,N_layer_int])

### get PS values
for f,F in enumerate(freqs): 
   for s in range(N_subj):    
       PLV_PS_ps[s,f]        = PS[s][f].mean_masked * PS[s][f].N_pot                             
       K_PS_ps[s,f]          = 100*PS[s][f].K           * PS[s][f].N_pot           
       for d in range(N_dist_bins):   
           PLV_PS_dist_ps[s,f,d]   = PS_dist[s][f][d].mean_masked * PS_dist[s][f][d].N_pot
           K_PS_dist_ps[s,f,d]     = 100*PS_dist[s][f][d].K           * PS_dist[s][f][d].N_pot    
       for l in range(N_layer_int):                             
           PLV_PS_layer_ps[s,f,l]   = PS_layer[s][f][l].mean_masked * PS_layer[s][f][l].N_pot
           K_PS_layer_ps[s,f,l]     = 100*PS_layer[s][f][l].K           * PS_layer[s][f][l].N_pot                   
  
                            
### get bootstrap stats for PS
N_boot = 1000    
K_PS_stats         = [np.array(bst.CI_from_bootstrap(K_PS_ps,N_boot,  2.5,97.5,N_edges_subj))-1] 
PLV_PS_stats       = [bst.CI_from_bootstrap(PLV_PS_ps,N_boot,2.5,97.5,N_edges_subj)] 
K_PS_stats_dist    = [np.array(bst.CI_from_bootstrap(K_PS_dist_ps[:,:,i],   N_boot,2.5,97.5, N_edges_dist_subj[:,i]))-1 for i in range(N_dist_bins)] 
PLV_PS_stats_dist  = [bst.CI_from_bootstrap(PLV_PS_dist_ps[:,:,i], N_boot,2.5,97.5, N_edges_dist_subj[:,i]) for i in range(N_dist_bins)] 
K_PS_stats_layer   = [np.array(bst.CI_from_bootstrap(K_PS_layer_ps[:,:,i],  N_boot,2.5,97.5, N_edges_layer_subj[:,i]))-1 for i in range(N_layer_int)] 
PLV_PS_stats_layer = [bst.CI_from_bootstrap(PLV_PS_layer_ps[:,:,i],N_boot,2.5,97.5, N_edges_layer_subj[:,i]) for i in range(N_layer_int)] 

### get PS means            
mean_K_PS           =  [(np.nansum(K_PS_ps,0)/div1)-1]
mean_PLV_PS         =  [(np.nansum(PLV_PS_ps,0)/div1)]    
mean_K_PS_dist      =   np.transpose(np.nansum(K_PS_dist_ps,0)/div2)-1
mean_PLV_PS_dist    =   np.transpose(np.nansum(PLV_PS_dist_ps,0)/div2)  
mean_K_PS_layer     =   np.transpose(np.nansum(K_PS_layer_ps,0)/div4)-1
mean_PLV_PS_layer   =   np.transpose(np.nansum(PLV_PS_layer_ps,0)/div4)   
K_PS_ps             =   (K_PS_ps/div3[:,np.newaxis])-1
PLV_PS_ps           =   (PLV_PS_ps/div3[:,np.newaxis])   
    

### PLOT PS 
    
figsize = [6.1,2.3]
rows    = 2
cols    = 3
dataL   = [PLV_PS_stats,PLV_PS_stats_dist,PLV_PS_ps,K_PS_stats,K_PS_stats_dist,K_PS_ps]
xlimA   = [xlims_PS for i in range(6)]
ylimA   = [[0,0.2],[0,0.2],[0,0.3],[0,100],[0,100],[0,100]]
titlesA = ['' for i in range(6)]   ###['mean '+PS_metric,'mean '+PS_metric+' per distance','mean '+PS_metric+' per subject','mean K','mean K per distance','mean K per subject']
legendA = [None, distances, None, None, distances, None]
ylabA   = [PS_metric,PS_metric,PS_metric,'K [%]','K [%]','K [%]']
cmapA   = ['brg','brg','brg','brg','brg','brg']
CI      = [0.2,0.2,None,0.2,0.2,None]
legend_posA = [None,'ur',None,None,None,None]
plots.semi_log_plot_multi(figsize,rows,cols,dataL,freqs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI)   

## export .pdf
o66 = directory + '_results\\SEEG PS new\\SEEG ' + PS_metric + '.pdf'
plots.semi_log_plot_multi(figsize,rows,cols,dataL,freqs,xlimA,ylabA,titlesA,cmapA,legendA,o66,legend_posA,ylimA,False,1,CI)   


### PLOT PS with distances and layers  

figsize = [7.7,3]
rows    = 2
cols    = 3
dataL   = [PLV_PS_stats,PLV_PS_stats_dist,PLV_PS_stats_layer[:3],K_PS_stats,K_PS_stats_dist,K_PS_stats_layer[:3]]
xlimA   = [xlims_PS for i in range(6)]
ylimA   = [[0,0.2],[0,0.2],[0,0.2],[0,100],[0,100],[0,100]]
titlesA = ['mean '+PS_metric, 'mean '+PS_metric+' per distance','mean '+PS_metric+' per layer int.',
           'mean K','mean K per distance', 'mean K per layer int.']
legendA = [None, distances, layer_int[:3], None, distances, layer_int[:3]]
ylabA   = [PS_metric,'','','K','','']
cmapA   = ['brg','brg',my_cmap,'brg','brg',my_cmap,]
CI      = [0.2 for i in range(6)]
legend_posA = [None,'ur','ur',None,'ur','ur']
xlab    = [0,0,0,1,1,1]
plots.semi_log_plot_multi(figsize,rows,cols,dataL,freqs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab)   

## export .pdf
o67 = directory + '_results\\SEEG PS new\\SEEG '+PS_metric + '_with layers.pdf'    
plots.semi_log_plot_multi(figsize,rows,cols,dataL,freqs,xlimA,ylabA,titlesA,cmapA,legendA,o67,legend_posA,ylimA,False,1,CI,xlab)   


#### export plot data as csv files

o31  =  directory + '_results\\_plot_data_new\\PS\\SEEG\\SEEG ' + PS_metric + ' K.csv'
o32  =  directory + '_results\\_plot_data_new\\PS\\SEEG\\SEEG ' + PS_metric + ' GS.csv'
o33a = [directory + '_results\\_plot_data_new\\PS\\SEEG\\SEEG ' + PS_metric + ' K, ' + dists_short[i] + '.csv' for i in range(3)] 
o34a = [directory + '_results\\_plot_data_new\\PS\\SEEG\\SEEG ' + PS_metric + ' GS, ' + dists_short[i] + '.csv' for i in range(3)] 

np.savetxt(o31,K_PS_stats[0][:3],delimiter=';')
np.savetxt(o32,PLV_PS_stats[0][:3],delimiter=';')
for i in range(3):
    np.savetxt(o33a[i],K_PS_stats_dist[i][:3],delimiter=';')
    np.savetxt(o34a[i],PLV_PS_stats_dist[i][:3],delimiter=';')





###############################################################################
###############################################################################






###############################################################################
###############################################################################

##############         PLOT  CFC              


### get numbers of edges for CFC

N_CH_subj                = np.zeros(N_subj)
N_pot_subj_CF            = np.zeros(N_subj)
N_pot_subj_CF_mod        = np.zeros([N_subj,N_LF,N_ratios])
N_pot_subj_CF_excl       = np.zeros([N_subj,N_LF,N_ratios])
N_pot_dist_subj_CF       = np.zeros([N_subj,N_dist_bins])
N_pot_dist_subj_CF_mod   = np.zeros([N_subj,N_dist_bins,N_LF,N_ratios])
N_pot_dist_subj_CF_excl  = np.zeros([N_subj,N_dist_bins,N_LF,N_ratios])
N_pot_layer_subj_CF      = np.zeros([N_subj,N_layer_int])
N_pot_layer_subj_CF_mod  = np.zeros([N_subj,N_layer_int,N_LF,N_ratios])
N_pot_layer_subj_CF_excl = np.zeros([N_subj,N_layer_int,N_LF,N_ratios])


N_subj_contr             = 0                        # number of contributing subjects
N_subj_contr_dist        = np.zeros(N_dist_bins)
N_subj_contr_layer       = np.zeros(N_layer_int)

for s in range(N_subj):     
      N_pot_subj_CF[s]       = CFC[s][0][0].N_pot                        # these are dependent on mask only
      N_CH_subj[s]           = CFC[s][0][0].N_CH
      for lf in range(N_LF):
         for hf in range(N_ratios):    
             try:
                N_pot_subj_CF_mod[s,lf,hf]   = CFC[s][lf][hf].N_pot_mod
                N_pot_subj_CF_excl[s,lf,hf]  = CFC[s][lf][hf].N_pot_excl              
                N_subj_contr                += np.int(N_pot_subj_CF[s]>0) 
             except:
                pass    
        
for s in range(N_subj):  
    for d in range(N_dist_bins):   
        N_pot_dist_subj_CF[s,d]       = np.nan_to_num(CFC_dist[s][0][0][d].N_pot)
        N_subj_contr_dist[d]         += np.int(N_pot_dist_subj_CF[s,d]>0)
        for lf in range(N_LF):
            for hf in range(N_ratios):    
                try:        
                    N_pot_dist_subj_CF_mod[s,d,lf,hf]   = np.nan_to_num(CFC_dist[s][lf][hf][d].N_pot_mod)
                    N_pot_dist_subj_CF_excl[s,d,lf,hf]  = np.nan_to_num(CFC_dist[s][lf][hf][d].N_pot_excl)
                except:
                    pass
                
    for l in range(N_layer_int):   
        N_pot_layer_subj_CF[s,l]       = np.nan_to_num(CFC_layer[s][0][0][l].N_pot)
        N_subj_contr_layer[l]         += np.int(N_pot_layer_subj_CF[s,l]>0)
        for lf in range(N_LF):
            for hf in range(N_ratios):    
                try:        
                    N_pot_layer_subj_CF_mod[s,l,lf,hf]   = np.nan_to_num(CFC_layer[s][lf][hf][l].N_pot_mod)
                    N_pot_layer_subj_CF_excl[s,l,lf,hf]  = np.nan_to_num(CFC_layer[s][lf][hf][l].N_pot_excl)
                except:
                    pass
  
N_pot_CF            = np.nansum(N_pot_subj_CF)     
N_pot_CF_mod        = np.nansum(N_pot_subj_CF_mod,0)  
N_pot_CF_excl       = np.nansum(N_pot_subj_CF_excl,0)  

N_pot_dist_CF       = np.nansum(N_pot_dist_subj_CF,0)
N_pot_dist_CF_mod   = np.nansum(N_pot_dist_subj_CF_mod,0)
N_pot_dist_CF_excl  = np.nansum(N_pot_dist_subj_CF_excl,0)
N_pot_dist_CF_mod2  = np.moveaxis(N_pot_dist_CF_mod,0,-1)

N_pot_layer_CF      = np.nansum(N_pot_layer_subj_CF,0)
N_pot_layer_CF_mod  = np.nansum(N_pot_layer_subj_CF_mod,0)
N_pot_layer_CF_excl = np.nansum(N_pot_layer_subj_CF_excl,0)
N_pot_layer_CF_mod2 = np.moveaxis(N_pot_layer_CF_mod,0,-1)
                
       


## initialize arrays    
PLV_CFC_ps             = np.full([N_subj,N_LF,N_ratios],np.nan)
PLV_CFC_local_ps       = np.full([N_subj,N_LF,N_ratios],np.nan)
PLV_CFC_ps_mod         = np.full([N_subj,N_LF,N_ratios],np.nan)
PLV_CFC_ps_excl        = np.full([N_subj,N_LF,N_ratios],np.nan)
PLV_CFC_dist_ps        = np.full([N_subj,N_LF,N_ratios,N_dist_bins],np.nan)    
PLV_CFC_dist_ps_mod    = np.full([N_subj,N_LF,N_ratios,N_dist_bins],np.nan)
PLV_CFC_dist_ps_excl   = np.full([N_subj,N_LF,N_ratios,N_dist_bins],np.nan)
PLV_CFC_layer_ps       = np.full([N_subj,N_LF,N_ratios,N_layer_int],np.nan)    
PLV_CFC_layer_ps_mod   = np.full([N_subj,N_LF,N_ratios,N_layer_int],np.nan)
PLV_CFC_layer_ps_excl  = np.full([N_subj,N_LF,N_ratios,N_layer_int],np.nan)
nPLV_CFC_dist_ps       = np.full([N_subj,N_LF,N_ratios,N_dist_bins],np.nan)    
N_CFC_ps               = np.full([N_subj,N_LF,N_ratios],np.nan)
N_CFC_local_ps         = np.full([N_subj,N_LF,N_ratios],np.nan)
N_CFC_ps_mod           = np.full([N_subj,N_LF,N_ratios],np.nan)
N_CFC_ps_excl          = np.full([N_subj,N_LF,N_ratios],np.nan)
N_CFC_dist_ps          = np.full([N_subj,N_LF,N_ratios,N_dist_bins],np.nan)    
N_CFC_dist_ps_mod      = np.full([N_subj,N_LF,N_ratios,N_dist_bins],np.nan)
N_CFC_layer_ps         = np.full([N_subj,N_LF,N_ratios,N_layer_int],np.nan)    
N_CFC_layer_ps_mod     = np.full([N_subj,N_LF,N_ratios,N_layer_int],np.nan)
N_CFC_layer_ps_excl    = np.full([N_subj,N_LF,N_ratios,N_layer_int],np.nan)


   
# get CFC values
for lf,LF in enumerate(LFs): 
   for hf,HF in enumerate(HFs[lf]):
       if HF<cutoff_CF:
           for s in range(N_subj):                    
               
               PLV_CFC_ps[s,lf,hf]        = CFC[s][lf][hf].N_pot * CFC[s][lf][hf].mean_masked   
               PLV_CFC_ps_mod[s,lf,hf]    = CFC[s][lf][hf].N_pot * CFC[s][lf][hf].mean_mod                       
               PLV_CFC_ps_excl[s,lf,hf]   = CFC[s][lf][hf].N_pot * CFC[s][lf][hf].mean_excl
               PLV_CFC_local_ps[s,lf,hf]  = CFC[s][lf][hf].N_CH  * CFC[s][lf][hf].mean_local  
               N_CFC_ps[s,lf,hf]          = 100*CFC[s][lf][hf].N                     
               N_CFC_ps_mod[s,lf,hf]      = 100*CFC[s][lf][hf].N_mod            
               N_CFC_ps_excl[s,lf,hf]     = 100*CFC[s][lf][hf].N_excl       
               N_CFC_local_ps[s,lf,hf]    = 100*CFC[s][lf][hf].N_local       
               
               for d in range(N_dist_bins):                   
                   PLV_CFC_dist_ps[s,lf,hf,d]       = CFC_dist[s][lf][hf][d].N_pot * CFC_dist[s][lf][hf][d].mean_masked
                   PLV_CFC_dist_ps_mod[s,lf,hf,d]   = CFC_dist[s][lf][hf][d].N_pot * CFC_dist[s][lf][hf][d].mean_mod                           
                   PLV_CFC_dist_ps_excl[s,lf,hf,d]  = CFC_dist[s][lf][hf][d].N_pot * CFC_dist[s][lf][hf][d].mean_excl     
                   try:
                       nPLV_CFC_dist_ps[s,lf,hf,d]      = CFC_dist[s][lf][hf][d].N_pot * CFC_dist[s][lf][hf][d].mean_masked / CFC_dist[s][lf][hf][d].mean_surr_masked
                   except:
                       pass
                       
                   N_CFC_dist_ps[s,lf,hf,d]         = 100*CFC_dist[s][lf][hf][d].N
                   N_CFC_dist_ps_mod[s,lf,hf,d]     = 100*CFC_dist[s][lf][hf][d].N_mod
              
                       
               for l in range(N_layer_int):                   
                   PLV_CFC_layer_ps[s,lf,hf,l]      = CFC_layer[s][lf][hf][l].N_pot * CFC_layer[s][lf][hf][l].mean_masked
                   PLV_CFC_layer_ps_mod[s,lf,hf,l]  = CFC_layer[s][lf][hf][l].N_pot * CFC_layer[s][lf][hf][l].mean_mod                           
                   PLV_CFC_layer_ps_excl[s,lf,hf,l] = CFC_layer[s][lf][hf][l].N_pot * CFC_layer[s][lf][hf][l].mean_excl                               
                   N_CFC_layer_ps[s,lf,hf,l]        = 100*CFC_layer[s][lf][hf][l].N
                   N_CFC_layer_ps_mod[s,lf,hf,l]    = 100*CFC_layer[s][lf][hf][l].N_mod
                   N_CFC_layer_ps_excl[s,lf,hf,l]   = 100*CFC_layer[s][lf][hf][l].N_excl



# get CFC stats for Figure 3 with 95% confidence intervals
N_boot = 1000

PLV_CFC_stats        = [bst.CI_from_bootstrap(PLV_CFC_ps      [:,:,i],N_boot,2.5,97.5,N_pot_subj_CF) for i in range(N_ratios)] # returns [mean, mean_boot, lower, upper] x freq x ratio
PLV_CFC_local_stats  = [bst.CI_from_bootstrap(PLV_CFC_local_ps[:,:,i],N_boot,2.5,97.5,N_CH_subj) for i in range(N_ratios)]
K_CFC_stats          = [np.array(bst.CI_from_bootstrap(N_CFC_ps      [:,:,i],N_boot,2.5,97.5,N_pot_subj_CF            )) -1 for i in range(N_ratios)]  
K_CFC_stats_mod      = [np.array(bst.CI_from_bootstrap(N_CFC_ps_mod  [:,:,i],N_boot,2.5,97.5,N_pot_subj_CF_mod[:,:,i] )) -1 for i in range(N_ratios)] 
K_CFC_stats_excl     = [np.array(bst.CI_from_bootstrap(N_CFC_ps_excl [:,:,i],N_boot,2.5,97.5,N_pot_subj_CF_excl[:,:,i])) -1 for i in range(N_ratios)] 
K_CFC_local_stats    = [np.array(bst.CI_from_bootstrap(N_CFC_local_ps[:,:,i],N_boot,2.5,97.5,N_CH_subj                )) -1 for i in range(N_ratios)]  

  
# get stats for Figure 5 (dist)
PLV_CFC_dist_12_stats     = [bst.CI_from_bootstrap(PLV_CFC_dist_ps[:,:,0,i],N_boot,2.5,97.5,N_pot_dist_subj_CF[:,i]) for i in range(N_dist_bins)] # returns [mean, lower, upper, mean_boot] x freq x dist
PLV_CFC_dist_13_stats     = [bst.CI_from_bootstrap(PLV_CFC_dist_ps[:,:,1,i],N_boot,2.5,97.5,N_pot_dist_subj_CF[:,i]) for i in range(N_dist_bins)] 
K_CFC_dist_12_stats       = [np.array(bst.CI_from_bootstrap(N_CFC_dist_ps[:,:,0,i],N_boot,2.5,97.5,N_pot_dist_subj_CF[:,i]))-1 for i in range(N_dist_bins)] 
K_CFC_dist_13_stats       = [np.array(bst.CI_from_bootstrap(N_CFC_dist_ps[:,:,1,i],N_boot,2.5,97.5,N_pot_dist_subj_CF[:,i]))-1 for i in range(N_dist_bins)]
K_CFC_dist_12_stats_mod   = [np.array(bst.CI_from_bootstrap(N_CFC_dist_ps_mod[:,:,0,i],N_boot,2.5,97.5,N_pot_dist_subj_CF_mod[:,i,:,0]))-1 for i in range(N_dist_bins)] 
K_CFC_dist_13_stats_mod   = [np.array(bst.CI_from_bootstrap(N_CFC_dist_ps_mod[:,:,1,i],N_boot,2.5,97.5,N_pot_dist_subj_CF_mod[:,i,:,1]))-1 for i in range(N_dist_bins)] 

# get stats for Figure 6 (layer)
PLV_CFC_layer_12_stats    = [bst.CI_from_bootstrap(PLV_CFC_layer_ps[:,:,0,i],N_boot,2.5,97.5,N_pot_layer_subj_CF[:,i]) for i in range(N_layer_int)] # returns [mean, lower, upper, mean_boot] x freq x layer
PLV_CFC_layer_13_stats    = [bst.CI_from_bootstrap(PLV_CFC_layer_ps[:,:,1,i],N_boot,2.5,97.5,N_pot_layer_subj_CF[:,i]) for i in range(N_layer_int)] 
K_CFC_layer_12_stats      = [np.array(bst.CI_from_bootstrap(N_CFC_layer_ps[:,:,0,i],N_boot,2.5,97.5,N_pot_layer_subj_CF[:,i]))-1 for i in range(N_layer_int)] 
K_CFC_layer_13_stats      = [np.array(bst.CI_from_bootstrap(N_CFC_layer_ps[:,:,1,i],N_boot,2.5,97.5,N_pot_layer_subj_CF[:,i]))-1 for i in range(N_layer_int)]
K_CFC_layer_12_stats_mod  = [np.array(bst.CI_from_bootstrap(N_CFC_layer_ps_mod[:,:,0,i],N_boot,2.5,97.5,N_pot_layer_subj_CF_mod[:,i,:,0]))-1 for i in range(N_layer_int)] 
K_CFC_layer_13_stats_mod  = [np.array(bst.CI_from_bootstrap(N_CFC_layer_ps_mod[:,:,1,i],N_boot,2.5,97.5,N_pot_layer_subj_CF_mod[:,i,:,1]))-1 for i in range(N_layer_int)] 

# get CFC means                              
mean_K_CFC             = np.transpose(np.nansum(N_CFC_ps,0)  /N_pot_CF) -1
mean_PLV_CFC           = np.transpose(np.nansum(PLV_CFC_ps,0)/N_pot_CF)    
mean_K_CFC_dist        = np.nansum(N_CFC_dist_ps,0)  /N_pot_dist_CF -1
mean_PLV_CFC_dist      = np.nansum(PLV_CFC_dist_ps,0)/N_pot_dist_CF 
mean_K_CFC_mod         = np.transpose(np.nansum(N_CFC_ps_mod,0)  /N_pot_CF_mod) -1
mean_PLV_CFC_mod       = np.transpose(np.nansum(PLV_CFC_ps_mod,0)/N_pot_CF_mod) 
mean_K_CFC_dist_mod    = np.nansum(N_CFC_dist_ps_mod,0)  /N_pot_dist_CF_mod2 -1
mean_PLV_CFC_dist_mod  = np.nansum(PLV_CFC_dist_ps_mod,0)/N_pot_dist_CF_mod2    

mean_K_CFC_layer       = np.nansum(N_CFC_layer_ps,0)  /N_pot_layer_CF -1
mean_PLV_CFC_layer     = np.nansum(PLV_CFC_layer_ps,0)/N_pot_layer_CF 
mean_K_CFC_layer_mod   = np.nansum(N_CFC_layer_ps_mod,0)  /N_pot_layer_CF_mod2 -1
mean_PLV_CFC_layer_mod = np.nansum(PLV_CFC_layer_ps_mod,0)/N_pot_layer_CF_mod2     

# divide by edge number to get get individual K and PLVs

K_CFC_ps_2            = N_CFC_ps*(1/div3[:,np.newaxis,np.newaxis])
PLV_CFC_ps_2          = PLV_CFC_ps*(1/div3[:,np.newaxis,np.newaxis])
K_CFC_ps_mod_2        = N_CFC_ps_mod*(1/div3[:,np.newaxis,np.newaxis])
PLV_CFC_ps_mod_2      = PLV_CFC_ps_mod*(1/div3[:,np.newaxis,np.newaxis])    
mean_K_CFC_local      = np.transpose(np.nansum(N_CFC_local_ps,0)/sum(N_CH_subj))-1
mean_PLV_CFC_local    = np.transpose(np.nansum(PLV_CFC_local_ps,0)/sum(N_CH_subj))     
    
# get means per subject
mean_PLV_CFC_ps_12        = (PLV_CFC_ps_2[:,:,0])
mean_PLV_CFC_ps_13        = (PLV_CFC_ps_2[:,:,1])    
mean_K_CFC_ps_12          = (K_CFC_ps_2[:,:,0])
mean_K_CFC_ps_13          = (K_CFC_ps_2[:,:,1])    
mean_PLV_CFC_dist_12      = np.transpose(mean_PLV_CFC_dist[:,0,:])
mean_PLV_CFC_dist_13      = np.transpose(mean_PLV_CFC_dist[:,1,:])    
mean_K_CFC_dist_12        = np.transpose(mean_K_CFC_dist[:,0,:])
mean_K_CFC_dist_13        = np.transpose(mean_K_CFC_dist[:,1,:])
mean_PLV_CFC_layer_12     = np.transpose(mean_PLV_CFC_layer[:,0,:])
mean_PLV_CFC_layer_13     = np.transpose(mean_PLV_CFC_layer[:,1,:])    
mean_K_CFC_layer_12       = np.transpose(mean_K_CFC_layer[:,0,:])
mean_K_CFC_layer_13       = np.transpose(mean_K_CFC_layer[:,1,:])
mean_PLV_CFC_ps_12_mod    = (PLV_CFC_ps_mod[:,:,0])
mean_PLV_CFC_ps_13_mod    = (PLV_CFC_ps_mod[:,:,1])    
mean_K_CFC_ps_12_mod      = (N_CFC_ps_mod[:,:,0])
mean_K_CFC_ps_13_mod      = (N_CFC_ps_mod[:,:,1])       
mean_PLV_CFC_dist_12_mod  = np.transpose(mean_PLV_CFC_dist_mod[:,0,:])
mean_PLV_CFC_dist_13_mod  = np.transpose(mean_PLV_CFC_dist_mod[:,1,:])    
mean_K_CFC_dist_12_mod    = np.transpose(mean_K_CFC_dist_mod[:,0,:])
mean_K_CFC_dist_13_mod    = np.transpose(mean_K_CFC_dist_mod[:,1,:])
mean_PLV_CFC_layer_12_mod = np.transpose(mean_PLV_CFC_layer_mod[:,0,:])
mean_PLV_CFC_layer_13_mod = np.transpose(mean_PLV_CFC_layer_mod[:,1,:])    
mean_K_CFC_layer_12_mod   = np.transpose(mean_K_CFC_layer_mod[:,0,:])
mean_K_CFC_layer_13_mod   = np.transpose(mean_K_CFC_layer_mod[:,1,:])
max_PLV_CFC               = np.nanmax(mean_PLV_CFC)
max_K_CFC                 = np.nanmax(mean_K_CFC)   
max_PLV_CFC_local         = np.nanmax(mean_PLV_CFC_local)
max_K_CFC_local           = np.nanmax(mean_K_CFC_local)   
max_PLV_CFC_dist_13       = np.nanmax(mean_PLV_CFC_dist_13)
max_K_CFC_dist_13         = np.nanmax(mean_K_CFC_dist_13)

### remove 0 entries in higher ratios
mean_K_CFC    = list(mean_K_CFC)
mean_K_CFC    = [np.array(filter(lambda a: a != -0.01, i)) for i in mean_K_CFC]
#mean_PLV_CFC  = list(mean_PLV_CFC)
#mean_PLV_CFC  = [np.array(filter(lambda a: a != 0, i)) for i in mean_PLV_CFC]
#mean_K_CFC_mod    = list(mean_K_CFC_mod)
#mean_K_CFC_mod    = [np.array(filter(lambda a: a != -0.01, i)) for i in mean_K_CFC_mod]
#mean_PLV_CFC_mod  = list(mean_PLV_CFC_mod)
#mean_PLV_CFC_mod  = [np.array(filter(lambda a: a != 0, i)) for i in mean_PLV_CFC_mod]    
#mean_K_CFC_local     = list(mean_K_CFC_local )
#mean_K_CFC_local     = [np.array(filter(lambda a: a != -1, i)) for i in mean_K_CFC_local]
mean_PLV_CFC_local     = list(mean_PLV_CFC_local )
mean_PLV_CFC_local     = [np.array(filter(lambda a: a != 0, i)) for i in mean_PLV_CFC_local]  
mean_K_CFC_dist_13     = list(mean_K_CFC_dist_13 )
mean_K_CFC_dist_13     = [np.array(filter(lambda a: a != -1, i)) for i in mean_K_CFC_dist_13]
mean_K_CFC_dist_13_mod = list(mean_K_CFC_dist_13_mod )
mean_K_CFC_dist_13_mod = [np.array(filter(lambda a: a != -1, i)) for i in mean_K_CFC_dist_13_mod]
mean_PLV_CFC_dist_13   = list(mean_PLV_CFC_dist_13 )
mean_PLV_CFC_dist_13   = [np.array(filter(lambda a: a != 0, i)) for i in mean_PLV_CFC_dist_13]

PLV_CFC_stats                = [[np.array(filter(lambda a: a !=  0   , i)) for i in j] for j in PLV_CFC_stats] 
PLV_CFC_local_stats          = [[np.array(filter(lambda a: a !=  0,    i)) for i in j] for j in PLV_CFC_local_stats]
PLV_CFC_dist_13_stats        = [[np.array(filter(lambda a: a != -0   , i)) for i in j] for j in PLV_CFC_dist_13_stats] 
K_CFC_stats                  = [[np.array(filter(lambda a: a != -1, i)) for i in j] for j in K_CFC_stats]
K_CFC_stats_mod              = [[np.array(filter(lambda a: a != -1, i)) for i in j] for j in K_CFC_stats_mod]
K_CFC_local_stats            = [[np.array(filter(lambda a: a != -1, i)) for i in j] for j in K_CFC_local_stats]  

K_CFC_stats                  = [[K_CFC_stats[rat][i]*(K_CFC_stats[rat][i]>=0) for i in range(4)] for rat in range(6)]
K_CFC_stats_mod              = [[K_CFC_stats_mod[rat][i]*(K_CFC_stats_mod[rat][i]>=0) for i in range(4)] for rat in range(6)]
K_CFC_dist_12_stats          = [[K_CFC_dist_12_stats[dist][i]*(K_CFC_dist_12_stats[dist][i]>=0) for i in range(4)] for dist in range(3)]
K_CFC_dist_12_stats_mod      = [[K_CFC_dist_12_stats_mod[dist][i]*(K_CFC_dist_12_stats_mod[dist][i]>=0) for i in range(4)] for dist in range(3)]
K_CFC_dist_13_stats          = [[K_CFC_dist_13_stats[dist][i]*(K_CFC_dist_13_stats[dist][i]>=0) for i in range(4)] for dist in range(3)]
K_CFC_dist_13_stats_mod      = [[K_CFC_dist_13_stats_mod[dist][i]*(K_CFC_dist_13_stats_mod[dist][i]>=0) for i in range(4)] for dist in range(3)]
K_CFC_layer_12_stats         = [[K_CFC_layer_12_stats[layer][i]*(K_CFC_layer_12_stats[layer][i]>=0) for i in range(4)] for layer in range(4)]
K_CFC_layer_12_stats_mod     = [[K_CFC_layer_12_stats_mod[layer][i]*(K_CFC_layer_12_stats_mod[layer][i]>=0) for i in range(4)] for layer in range(4)]
K_CFC_layer_13_stats         = [[K_CFC_layer_13_stats[layer][i]*(K_CFC_layer_13_stats[layer][i]>=0) for i in range(4)] for layer in range(4)]
K_CFC_layer_13_stats_mod     = [[K_CFC_layer_13_stats_mod[layer][i]*(K_CFC_layer_13_stats_mod[layer][i]>=0) for i in range(4)] for layer in range(4)]

PLV_CFC_layer_13_stats       = [[np.array(filter(lambda a: a != -0   , i)) for i in j] for j in PLV_CFC_layer_13_stats] 
PLV_CFC_local_layer_13_stats = [[np.array(filter(lambda a: a != -0   , i)) for i in j] for j in PLV_CFC_local_layer_13_stats] 
K_CFC_dist_13_stats          = [[np.array(filter(lambda a: a != -1, i)) for i in j] for j in K_CFC_dist_13_stats]
K_CFC_dist_13_stats_mod      = [[np.array(filter(lambda a: a != -1, i)) for i in j] for j in K_CFC_dist_13_stats_mod]
K_CFC_layer_13_stats         = [[np.array(filter(lambda a: a != -1, i)) for i in j] for j in K_CFC_layer_13_stats]
K_CFC_layer_13_stats_mod     = [[np.array(filter(lambda a: a != -1, i)) for i in j] for j in K_CFC_layer_13_stats_mod]
K_CFC_local_layer_13_stats   = [[np.array(filter(lambda a: a != -1, i)) for i in j] for j in K_CFC_local_layer_13_stats]

PLV_CFC_stats                = [[np.array(filter(lambda a: a != np.nan, i)) for i in j] for j in PLV_CFC_stats] 
PLV_CFC_local_stats          = [[np.array(filter(lambda a: a != np.nan, i)) for i in j] for j in PLV_CFC_local_stats]
K_CFC_stats                  = [[np.array(filter(lambda a: a != np.nan, i)) for i in j] for j in K_CFC_stats]
K_CFC_stats_mod              = [[np.array(i[~np.isnan(i)]) for i in j] for j in K_CFC_stats_mod]
K_CFC_local_stats            = [[np.array(i[~np.isnan(i)]) for i in j] for j in K_CFC_local_stats]    
K_CFC_dist_13_stats          = [[np.array(i[~np.isnan(i)]) for i in j] for j in K_CFC_dist_13_stats]
K_CFC_dist_13_stats_mod      = [[np.array(i[~np.isnan(i)]) for i in j] for j in K_CFC_dist_13_stats_mod]
K_CFC_layer_13_stats         = [[np.array(i[~np.isnan(i)]) for i in j] for j in K_CFC_layer_13_stats]
K_CFC_layer_13_stats_mod     = [[np.array(i[~np.isnan(i)]) for i in j] for j in K_CFC_layer_13_stats_mod]


###############################################################################
##########              plot CFC     


figsize = [6.3,2.3]  
#figsize = [12.7,3.6]   
 
rows    = 2
cols    = 3
dataL   = [PLV_CFC_stats[:1],K_CFC_stats[:1],K_CFC_stats_mod[:1],
           PLV_CFC_stats[1:],K_CFC_stats[1:],K_CFC_stats_mod[1:]]    
xlimA   = [xlims_CFC for i in range(6)]
titlesA = ['' for i in range(6)] #['mean PLV','mean K','mean K (controlled)','','','']
if CF_type == 'CFS':
    ylimA   = [[-0.005,0.049], [-1.4, 14], [-1.4,14], [-0.005,0.049], [-0.25,2.5], [-0.25,2.5]]  
else:
    ylimA   = [[-0.007,0.069],  [-2.9, 29], [-2.9,29],  [-0.007,0.069], [-2.9,29],  [-2.9,29]] 
legendA = [ratios[:1],ratios[:1],ratios[:1],
           ratios[1:],ratios[1:],ratios[1:],]    
ylabA   = ['GS','K [%]','K [%]', 'GS','K [%]','K [%]']
cmapA   = ['brg','brg','brg',my_cmap3,my_cmap3,my_cmap3]
legend_posA = ['ur',None,None,'ur',None,None]
CI      = [0.2 for i in range(6)]
xlab    = [0,0,0,1,1,1]
Ryt     = [1,1,1,1,1,1]
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt)   

## export PDF
o80 = directory + '_results\\SEEG CF new\\SEEG ' + CF_type + ', controlled with ' + PS_metric + '.pdf'
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,o80,legend_posA,ylimA,False,1,CI,xlab,Ryt)   



#### plot as heatmap

data1 = np.transpose(mean_K_CFC)
data2 = np.transpose(mean_K_CFC_mod)
figsize_hm = [1.6,1.9]

if CF_type == 'CFS':
    zmax1  = 12
    zmax2  = 4
    ztix1  = [0,3,6,9,12] 
    ztix2  = [0,1,2,3,4]    
else:
    zmax1  = 24
    zmax2  = 8
    ztix1  = [0,6,12,18,24] 
    ztix2  = [0,2,4,6,8]  
    

LF_ics = [0,3,6,9,12,15,18,21,24,27,29]    
LF_map = ['1.2', '2.4', '3.7', '5.9', '8.6', '13.2', '19.5', '29.5', '47.3', '68.1', '94.5']
   
plots.simple_CF_plot(data1,figsize_hm,'ratio','Low Frequency [Hz]',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax1,ztix=ztix1,outfile=None)             
plots.simple_CF_plot(data2,figsize_hm,'ratio','Low Frequency [Hz]',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax2,ztix=ztix2,outfile=None)             
  
# export PDFs 
o90  = directory + '_results\\SEEG CF new\\SEEG ' + CF_type + ' heatmap.pdf'
o90a = directory + '_results\\SEEG CF new\\SEEG ' + CF_type + ' heatmap, controlled with ' + PS_metric + '.pdf'
plots.simple_CF_plot(data1,figsize_hm,'ratio','Low Frequency [Hz]',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax1,ztix=ztix1,outfile=o90)             
plots.simple_CF_plot(data2,figsize_hm,'ratio','Low Frequency [Hz]',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax2,ztix=ztix2,outfile=o90a)             
   




###############################################################################
#####              GROUPS STATS AND PLOTS FOR ENVELOPE    

# init ENV arrays
K_ENV_ps               = np.full([N_subj,N_LF,N_ratios,],np.nan)
PLV_ENV_ps             = np.full([N_subj,N_LF,N_ratios],np.nan)
K_ENV_local_ps         = np.full([N_subj,N_LF,N_ratios],np.nan)
PLV_ENV_local_ps       = np.full([N_subj,N_LF,N_ratios],np.nan)
K_ENV_dist_ps          = np.full([N_subj,N_LF,N_ratios,N_dist_bins],np.nan)
PLV_ENV_dist_ps        = np.full([N_subj,N_LF,N_ratios,N_dist_bins],np.nan)    

### get ENV values
for lf,LF in enumerate(LFs): 
   for hf,HF in enumerate(HFs[lf]):
       for s,ss in enumerate(subjects):             
           K_ENV_ps       [s,lf,hf]        = 100*PS_ENV[s][lf][hf].K
           PLV_ENV_ps     [s,lf,hf]        = PS_ENV[s][lf][hf].mean_masked
           for d in range(N_dist_bins):                                  
               PLV_ENV_dist_ps [s,lf,hf,d] = PS_ENV_dist[s][lf][hf][d].mean_masked
               K_ENV_dist_ps   [s,lf,hf,d] = 100*PS_ENV_dist[s][lf][hf][d].K


### get ENV means and 95% confidence intervals
N_boot=1000    
PLV_ENV_stats          = [np.array(bst.CI_from_bootstrap(PLV_ENV_ps[:,:,i]))       for i in range(N_ratios)] # returns [mean, mean_boot, lower, upper] x freq x ratio
K_ENV_stats            = [np.array(bst.CI_from_bootstrap(K_ENV_ps[:,:,i])) -1      for i in range(N_ratios)]  
PLV_ENV_dist_12_stats  = [bst.CI_from_bootstrap(PLV_ENV_dist_ps[:,:,0,i]) for i in range(N_dist_bins)] # returns [mean, mean_boot, lower, upper] x freq x dist
PLV_ENV_dist_13_stats  = [bst.CI_from_bootstrap(PLV_ENV_dist_ps[:,:,1,i]) for i in range(N_dist_bins)] # returns [mean, mean_boot, lower, upper] x freq x dist 
K_ENV_dist_12_stats    = [np.array(bst.CI_from_bootstrap(K_ENV_dist_ps[:,:,0,i]))-1 for i in range(N_dist_bins)] # returns [mean, mean_boot, lower, upper] x freq x dist
K_ENV_dist_13_stats    = [np.array(bst.CI_from_bootstrap(K_ENV_dist_ps[:,:,1,i]))-1 for i in range(N_dist_bins)] # returns [mean, mean_boot, lower, upper] x freq x dist


###############################################################################
###########    plot amplitude envelope

figsize = [5.3,2.3]  
rows    = 2
cols    = 2
dataL   = [PLV_ENV_stats,K_ENV_stats,
           PLV_ENV_stats,K_ENV_stats,]    
xlimA   = [[1,330] for i in range(4)]
titlesA = ['' for i in range(4)] #['mean PLV','mean K','mean K (controlled)','','','']
ylimA   = [[-0.007,0.07],  [-2, 22],[-0.007,0.07],  [-2,22], ] 
legendA = [ratios,ratios,
           ratios,ratios,]    
ylabA   = ['GS','K [%]', 'GS','K [%]',]
cmapA   = [my_cmap7,my_cmap7,my_cmap7,my_cmap7]
legend_posA = [None,None,None,None,]
CI      = [0.2 for i in range(4)]
xlab    = [0,0,1,1,]
Ryt     = [1,1,1,1,]

plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,     xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt)   
plots.semi_log_plot_multi(figsize,rows,cols,dataL,HFs_env,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt)   


## export PDF
o80e = directory + '_results\\SEEG CF new\\SEEG Envelopes LFx.pdf'
o80f = directory + '_results\\SEEG CF new\\SEEG Envelopes HFx.pdf'
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,    xlimA,ylabA,titlesA,cmapA,legendA,o80e,legend_posA,ylimA,False,1,CI,xlab,Ryt)   
plots.semi_log_plot_multi(figsize,rows,cols,dataL,HFs_env,xlimA,ylabA,titlesA,cmapA,legendA,o80f,legend_posA,ylimA,False,1,CI,xlab,Ryt)   


#### plot as heatmap
data1 = np.transpose(mean_K_CFC)
figsize_hm = [1.6,1.9]
zmax1  = 20
zmax2  = 4
ztix1  = [0,5,10,15,20] 
ztix2  = [0,1,2,3,4] 
LF_ics = [0,3,6,9,12,15,18,21,24,27,29]    
LF_map = ['1.2', '2.4', '3.7', '5.9', '8.6', '13.2', '19.5', '29.5', '47.3', '68.1', '94.5'] 
plots.simple_CF_plot(data1,figsize_hm,'ratio','Low Frequency [Hz]',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax1,ztix=ztix1,outfile=None)             

## export pdf
o99  = directory + '_results\\SEEG CF new\\SEEG Envelope heatmap.pdf'
plots.simple_CF_plot(data1,figsize_hm,'ratio','Low Frequency [Hz]',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax1,ztix=ztix1,outfile=o99)             










###############################################################################
#####      plot CFC in distance bins 


figsize = [6.3,2.3]  
rows    = 2
cols    = 3
dataL   = [PLV_CFC_dist_12_stats, K_CFC_dist_12_stats, K_CFC_dist_12_stats_mod,
           PLV_CFC_dist_13_stats, K_CFC_dist_13_stats, K_CFC_dist_13_stats_mod]        
xlimA   = [xlims_CFC for i in range(6)]
titlesA = ['' for i in range(6)] 
if CF_type =='CFS':
    ylimA   = [[-0.005,0.049], [-2, 19], [-2,19], [-0.005,0.049], [-0.35,3.4], [-0.35,3.4]]      
else:
    ylimA   = [[-0.007,0.067],  [-3.8, 38],  [-3.8,38], [-0.007,0.067],  [-3.8, 38],  [-3.8,38]]                  # PAC           
legendA = [distances for i in range(6)]
ylabA   = ['GS','K [%]','K [%]', 'GS','K [%]','K [%]']
cmapA   = ['brg','brg','brg','brg','brg','brg']
legend_posA = ['ur']+[None for i in range(5)]
CI      = [0.2 for i in range(6)]
xlab    = [0,0,0,1,1,1]
Ryt     = [1,1,1,1,1,1]
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt)   

## export PDF
o81 = directory + '_results\\SEEG CF new\\SEEG ' + CF_type + ' controlled with ' + PS_metric + ', distance bins.pdf'
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,o81,legend_posA,ylimA,False,1,CI,xlab,Ryt)   










###############################################################################
#####             plot CF by layer combination

figsize = [6.3,2.3]  
rows    = 2
cols    = 3
dataL   = [PLV_CFC_layer_12_stats, K_CFC_layer_12_stats, K_CFC_layer_12_stats_mod,
           PLV_CFC_layer_13_stats, K_CFC_layer_13_stats, K_CFC_layer_13_stats_mod]        
xlimA   = [xlims_CFC for i in range(6)]
titlesA = ['' for i in range(6)]       
if CF_type =='CFS':
    ylimA   = [[-0.005,0.049], [-2, 19], [-2,19], [-0.005,0.049], [-0.44,4.6], [-0.44,4.6]]      
else:
    ylimA   = [[-0.007,0.067],  [-3.8, 38],  [-3.8,38], [-0.007,0.067],  [-3.8, 38],  [-3.8,38]]                   # PAC         
legendA = [layer_int for i in range(6)]
ylabA   = ['GS','K [%]','K [%]', 'GS','K [%]','K [%]']
cmapA   = [my_cmap for i in range(6)]
legend_posA = [None,None,'ur',None,None,None ]
CI      = [0.2 for i in range(6)]
xlab    = [0,0,0,1,1,1]
Ryt     = [1,0,1,1,1,1]
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt)   

## export PDF
o82 = directory + '_results\\SEEG CF new\\SEEG ' + CF_type + ' controlled with ' + PS_metric + ', layer interation.pdf'
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,o82,legend_posA,ylimA,False,1,CI,xlab,Ryt)   







###############################################################################
#####              plot local CF 

figsize = [4.1,2.3] 
figsize = [4.5,2.3]  
rows    = 2
cols    = 2
dataL   = [PLV_CFC_local_stats[:1], K_CFC_local_stats[:1], 
           PLV_CFC_local_stats[1:], K_CFC_local_stats[1:]]        
xlimA   = [xlims_CFC for i in range(4)]
titlesA = ['', '', '', '']
if CF_type == 'CFS':
    ylimA   = [[-0.01,0.13], [-0.01,100], [-0.005,0.044],  [-0.01,30]]     
else:
    ylimA   = [[-0.02,0.2], [-0.01,100],    [-0.02,0.2], [-0.01,100]]          
legendA = [ratios[:1],ratios[:1],
           ratios[1:],ratios[1:],] 
ylabA   = ['GS','K','GS','K']
cmapA   = ['brg','brg',my_cmap3,my_cmap3]
legend_posA = [None,None,None,None]
CI      = [0.2,0.2,0.2,0.2]
xlab    = [0,0,1,1]
Ryt     = [0,0,0,0]
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt)   

# export PDF
o83 = directory + '_results\\SEEG CF new\\SEEG local ' + CF_type + '.pdf'
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,o83,legend_posA,ylimA,False,1,CI,xlab,Ryt)   


### plot heatmap
data = np.transpose(np.array(mean_K_CFC_local))
figsize_hm = [1.6,1.9]
zmax   = 80 
ztix   = [0,20,40,60,80] 
LF_ics = [0,3,6,9,12,15,18,21,24,27,29]    
LF_map = ['1.2', '2.4', '3.7', '5.9', '8.6', '13.2', '19.5', '29.5', '47.3', '68.1', '94.5']    
plots.simple_CF_plot(data,figsize_hm,'ratio','LF',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax,ztix=ztix,outfile=None)             
   
# export PDF 
o93 = directory + '_results\\SEEG CF new\\SEEG local ' + CF_type + ' heatmap.pdf'
plots.simple_CF_plot(data,figsize_hm,'ratio','LF',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax,ztix=ztix,outfile=o93)             






###### save plot data as csv files

o41a = [directory + '_results\\_plot_data_new\\' + CF_type + '\\SEEG\\SEEG ' + CF_type  + ' K ' + r + '.csv' for r in ratios2 ]
o41b = [directory + '_results\\_plot_data_new\\' + CF_type + '\\SEEG\\SEEG ' + CF_type  + ' K_mod using ' + PS_metric + ' ' + r + '.csv' for r in ratios2 ]
o41c = [directory + '_results\\_plot_data_new\\' + CF_type + '\\SEEG\\SEEG ' + CF_type  + ' GS ' + r + '.csv' for r in ratios2 ]
o42a = [directory + '_results\\_plot_data_new\\' + CF_type + '_dist\\SEEG\\SEEG ' + CF_type + ' 1-2 K, ' + d + ' .csv' for d in dists_short ]
o42b = [directory + '_results\\_plot_data_new\\' + CF_type + '_dist\\SEEG\\SEEG ' + CF_type + ' 1-2 K_mod using ' + PS_metric + ', ' + d + '.csv' for d in dists_short ]
o42c = [directory + '_results\\_plot_data_new\\' + CF_type + '_dist\\SEEG\\SEEG ' + CF_type + ' 1-2 GS, ' + d + ' .csv' for d in dists_short ]
o43a = [directory + '_results\\_plot_data_new\\' + CF_type + '_dist\\SEEG\\SEEG ' + CF_type + ' 1-3 K, ' + d + '.csv' for d in dists_short ]
o43b = [directory + '_results\\_plot_data_new\\' + CF_type + '_dist\\SEEG\\SEEG ' + CF_type + ' 1-3 K_mod using ' + PS_metric + ', ' + d + '.csv' for d in dists_short ]
o43c = [directory + '_results\\_plot_data_new\\' + CF_type + '_dist\\SEEG\\SEEG ' + CF_type + ' 1-3 GS, ' + d + '.csv' for d in dists_short ]
o44a = [directory + '_results\\_plot_data_new\\' + CF_type + '_local\\SEEG\\SEEG local ' + CF_type  + ' K ' + r + '.csv' for r in ratios2 ]
o44b = [directory + '_results\\_plot_data_new\\' + CF_type + '_local\\SEEG\\SEEG local ' + CF_type  + ' GS ' + r + '.csv' for r in ratios2 ]       
o45a = [directory + '_results\\_plot_data_new\\ENV\\SEEG\\SEEG ENV K ' + r + '.csv' for r in ratios2 ]
o45b = [directory + '_results\\_plot_data_new\\ENV\\SEEG\\SEEG ENV GS ' + r + '.csv' for r in ratios2 ]
o46a = [directory + '_results\\_plot_data_new\\' + CF_type + '_layer\\SEEG\\SEEG ' + CF_type + ' 1-2 K, ' + l + ' .csv' for l in layer_int ]
o46b = [directory + '_results\\_plot_data_new\\' + CF_type + '_layer\\SEEG\\SEEG ' + CF_type + ' 1-2 K_mod using ' + PS_metric + ', ' + l + '.csv' for l in layer_int ]
o46c = [directory + '_results\\_plot_data_new\\' + CF_type + '_layer\\SEEG\\SEEG ' + CF_type + ' 1-2 GS, ' + l + ' .csv' for l in layer_int ]
o47a = [directory + '_results\\_plot_data_new\\' + CF_type + '_layer\\SEEG\\SEEG ' + CF_type + ' 1-3 K, ' + l + '.csv' for l in layer_int ]
o47b = [directory + '_results\\_plot_data_new\\' + CF_type + '_layer\\SEEG\\SEEG ' + CF_type + ' 1-3 K_mod using ' + PS_metric + ', ' + l + '.csv' for l in layer_int ]
o47c = [directory + '_results\\_plot_data_new\\' + CF_type + '_layer\\SEEG\\SEEG ' + CF_type + ' 1-3 GS, ' + l + '.csv' for l in layer_int ]
for r in range(6):
    np.savetxt(o41a[r],K_CFC_stats      [r][:3],delimiter=';')
    np.savetxt(o41b[r],K_CFC_stats_mod  [r][:3],delimiter=';')
    np.savetxt(o41c[r],PLV_CFC_stats    [r][:3],delimiter=';')
    for d in range(3):
        np.savetxt(o42a[d],K_CFC_dist_12_stats      [d][:3],delimiter=';')
        np.savetxt(o42b[d],K_CFC_dist_12_stats_mod  [d][:3],delimiter=';')
        np.savetxt(o42c[d],PLV_CFC_dist_12_stats    [d][:3],delimiter=';')
        np.savetxt(o43a[d],K_CFC_dist_13_stats      [d][:3],delimiter=';')
        np.savetxt(o43b[d],K_CFC_dist_13_stats_mod  [d][:3],delimiter=';')
        np.savetxt(o43c[d],PLV_CFC_dist_13_stats    [d][:3],delimiter=';')   
    np.savetxt(o44a[r],K_CFC_local_stats[r][:3],delimiter=';')
    np.savetxt(o44b[r],PLV_CFC_local_stats[r][:3],delimiter=';')
    if CF_type == 'PAC':
        np.savetxt(o45a[r],K_ENV_stats[r][:3],delimiter=';')
        np.savetxt(o45b[r],PLV_ENV_stats[r][:3],delimiter=';')
    for l in range(4):
        np.savetxt(o46a[l],K_CFC_layer_12_stats      [l][:3],delimiter=';')
        np.savetxt(o46b[l],K_CFC_layer_12_stats_mod  [l][:3],delimiter=';')
        np.savetxt(o46c[l],PLV_CFC_layer_12_stats    [l][:3],delimiter=';')
        np.savetxt(o47a[l],K_CFC_layer_13_stats      [l][:3],delimiter=';')
        np.savetxt(o47b[l],K_CFC_layer_13_stats_mod  [l][:3],delimiter=';')
        np.savetxt(o47c[l],PLV_CFC_layer_13_stats    [l][:3],delimiter=';')





###############################################################################
#######     compare long distance vs short dist. with Wilcoxon

K_CFC_dist_ps     = np.full([N_subj,N_LF,N_ratios,N_dist_bins],np.nan)
K_CFC_dist_ps_mod = np.full([N_subj,N_LF,N_ratios,N_dist_bins],np.nan)
   
# get CFC values
for lf,LF in enumerate(LFs): 
   for hf,HF in enumerate(HFs[lf]):
           for s in range(N_subj):                                      
               for d in range(N_dist_bins): 
                       PLV_CFC_dist_ps[s,lf,hf,d]      = CFC_dist[s][lf][hf][d].mean_masked
                       K_CFC_dist_ps[s,lf,hf,d]        = CFC_dist[s][lf][hf][d].K
                       K_CFC_dist_ps_mod[s,lf,hf,d]    = CFC_dist[s][lf][hf][d].K_mod
                                
wilc_pm     = np.zeros([N_LF,2,3])
wilc_p      = np.zeros([N_LF,2,3])
wilc_p_mod  = np.zeros([N_LF,2,3])

combo1      = [0,0,1]
combo2      = [1,2,2]

for lf,LF in enumerate(LFs): 
   for rat in range(2):
      for co in range(3):
          c1 = combo1[co]
          c2 = combo2[co]
          aaa, wilc_pm[lf,rat,co]    = stat.wilcoxon(PLV_CFC_dist_ps  [:,lf,rat,c1], PLV_CFC_dist_ps  [:,lf,rat,c2])
          aaa, wilc_p[lf,rat,co]     = stat.wilcoxon(K_CFC_dist_ps    [:,lf,rat,c1], K_CFC_dist_ps    [:,lf,rat,c2])
          aaa, wilc_p_mod[lf,rat,co] = stat.wilcoxon(K_CFC_dist_ps_mod[:,lf,rat,c1], K_CFC_dist_ps_mod[:,lf,rat,c2])
      
      
s_12_ps  = np.reshape( (1.*multicomp.multipletests(np.reshape(wilc_pm   [:,0],N_LF*3), method ='fdr_bh')[0]),[N_LF,3])
s_13_ps  = np.reshape( (1.*multicomp.multipletests(np.reshape(wilc_pm   [:,1],N_LF*3), method ='fdr_bh')[0]),[N_LF,3])
s_12     = np.reshape( (1.*multicomp.multipletests(np.reshape(wilc_p    [:,0],N_LF*3), method ='fdr_bh')[0]),[N_LF,3])
s_13     = np.reshape( (1.*multicomp.multipletests(np.reshape(wilc_p    [:,1],N_LF*3), method ='fdr_bh')[0]),[N_LF,3])
s_12_mod = np.reshape( (1.*multicomp.multipletests(np.reshape(wilc_p_mod[:,0],N_LF*3), method ='fdr_bh')[0]),[N_LF,3])
s_13_mod = np.reshape( (1.*multicomp.multipletests(np.reshape(wilc_p_mod[:,1],N_LF*3), method ='fdr_bh')[0]),[N_LF,3])

dists_short = ['short', 'mid', 'long']

### plot long vs short bin results
for co in range(3):
    combo_str = dists_short[combo1[co]] + '-' + dists_short[combo2[co]] 
    dataA = [[s_12_ps[:,co]],[s_12[:,co]],[s_12_mod[:,co]],[s_13_ps[:,co]],[s_13[:,co]],[s_13_mod[:,co]]]
    cmapA = ['brg','brg','brg','brg','brg','brg']
    xlimA = [xlims_CFC for i in range(6)]
    ylabA = ['','','','','','','']
    ylimA = [1,1,1,1,1,1] 
    titlesA = ['1-2','1-2','1-2 c','1-3','1-3','1-3 c']
    plots.semi_log_plot_multi([7.7,3],2,3,dataA,LFs,xlimA,ylabA,titlesA,cmapA,None,None,None,None,True,1,None,None,None,'auto',8,3)   
    
    # save pdf
    o85 = directory + '_results\\SEEG CF new\\SEEG ' + CF_type + ', controlled with ' + PS_metric + ', distance comparison ' + combo_str + '.pdf'
    plots.semi_log_plot_multi([7.7,3],2,3,dataA,LFs,xlimA,ylabA,titlesA,cmapA,None,o85,None,ylimA,True,1,None,None,None,'auto',8,3)   







###############################################################################
########    compare superficial vs. deep layer int with Wilcoxon

K_CFC_layer_ps     = np.full([N_subj,N_LF,N_ratios,N_layer_int],np.nan)
K_CFC_layer_ps_mod = np.full([N_subj,N_LF,N_ratios,N_layer_int],np.nan)

# get CFC values
for lf,LF in enumerate(LFs): 
   for hf,HF in enumerate(HFs[lf]):
           for s in range(N_subj):                                      
               for d in range(N_layer_int): 
                       PLV_CFC_layer_ps[s,lf,hf,d]      = CFC_layer[s][lf][hf][d].mean_masked
                       K_CFC_layer_ps[s,lf,hf,d]        = CFC_layer[s][lf][hf][d].K
                       K_CFC_layer_ps_mod[s,lf,hf,d]    = CFC_layer[s][lf][hf][d].K_mod
              #         K_CFC_layer_ps_excl[s,lf,hf,d]   = CFC_layer[s][lf][hf][d].K_excl                       
                       
wilc_pps    = np.zeros([N_LF,2,6])
wilc_p      = np.zeros([N_LF,2,6])
wilc_p_mod  = np.zeros([N_LF,2,6])

combo1      = [0,0,0,1,1,2]
combo2      = [1,2,3,2,3,3]

for lf,LF in enumerate(LFs): 
   for rat in range(2):
       for co in range(6):
           c1 = combo1[co]
           c2 = combo2[co]           
           aaa, wilc_pps  [lf,rat,co] = stat.wilcoxon(PLV_CFC_layer_ps  [:,lf,rat,c1], PLV_CFC_layer_ps  [:,lf,rat,c2])
           aaa, wilc_p    [lf,rat,co] = stat.wilcoxon(K_CFC_layer_ps    [:,lf,rat,c1], K_CFC_layer_ps    [:,lf,rat,c2])
           aaa, wilc_p_mod[lf,rat,co] = stat.wilcoxon(K_CFC_layer_ps_mod[:,lf,rat,c1], K_CFC_layer_ps_mod[:,lf,rat,c2])
  

s_12_ps  = np.reshape( (1.*multicomp.multipletests(np.reshape(wilc_pps  [:,0],N_LF*6), method ='fdr_bh')[0]),[N_LF,6])
s_13_ps  = np.reshape( (1.*multicomp.multipletests(np.reshape(wilc_pps  [:,1],N_LF*6), method ='fdr_bh')[0]),[N_LF,6])
s_12     = np.reshape( (1.*multicomp.multipletests(np.reshape(wilc_p    [:,0],N_LF*6), method ='fdr_bh')[0]),[N_LF,6])
s_13     = np.reshape( (1.*multicomp.multipletests(np.reshape(wilc_p    [:,1],N_LF*6), method ='fdr_bh')[0]),[N_LF,6])
s_12_mod = np.reshape( (1.*multicomp.multipletests(np.reshape(wilc_p_mod[:,0],N_LF*6), method ='fdr_bh')[0]),[N_LF,6])
s_13_mod = np.reshape( (1.*multicomp.multipletests(np.reshape(wilc_p_mod[:,1],N_LF*6), method ='fdr_bh')[0]),[N_LF,6])



s_12_ps[s_12_ps==0]=np.nan
s_12[s_12==0]=np.nan
s_12_mod[s_12_mod==0]=0.01
s_13_ps[s_13_ps==0]=0.01
s_13[s_13==0]=0.01
s_13_mod[s_13_mod==0]=0.01

lay_sh = ['SS','DD','SD','DS']


for co in range(6):
    dataA = [[s_12_ps[:,co]],[s_12[:,co]],[s_12_mod[:,co]],[s_13_ps[:,co]],[s_13[:,co]],[s_13_mod[:,co]]]
    cmapA = ['brg','brg','brg','brg','brg','brg']
    xlimA   = [xlims_CFC for i in range(6)]
    ylimA = [[0,1.1] for i in range(6)]
    ylabA = ['','','','','','']
    ylabA = ['','','','','','']
    titlesA = ['1-2','1-2','1-2 c','1-3','1-3','1-3 c']
    plots.semi_log_plot_multi([7.7,3],2,3,dataA,LFs,xlimA,ylabA,titlesA,cmapA,None,None,None,ylimA,True,1,None,None,None,'auto',8,3)   


    combo_str_l = lay_sh[combo1[co]] + '-' + lay_sh[combo2[co]]

    ## save pdf
    o86 = directory + '_results\\SEEG CF new\\SEEG ' + CF_type + ', controlled with ' + PS_metric + ', layer comparison ' + combo_str_l + '.pdf'
    plots.semi_log_plot_multi([7.7,3],2,3,dataA,LFs,xlimA,ylabA,titlesA,cmapA,None,o86,None,ylimA,True,1,None,None,None,'auto',8,3)   
    








###############################################################################
########  initialize morphing ops and networks, get degrees and strengths  


file_in1 = directory + '_results\\_pickle dump\\SEEG ' + PS_metric  + '.dat'
file_in2 = directory + '_results\\_pickle dump\\SEEG ' + CF_type + ' corrected with ' + PS_metric + '.dat'    
file_in3 = directory + '_results\\_pickle dump\\SEEG ' + CF_type + ' by layer interaction corrected with ' + PS_metric  + '.dat'
PS,PS_dist            = pick.load(open(file_in1,'rb'))
CFC,CFC_dist          = pick.load(open(file_in2,'rb'))
PS_layer,CFC_layer    = pick.load(open(file_in3,'rb'))

### initialize networks 
network_names   = ['C','DM','DA','Lim','VA','SM','Vis']
N_network       = 7

file_networks   = 'M:\\SEEG_Morlet\\_RAW_line_filtered\\_support_files\\networks parc2009.csv'
network_indices = np.array(np.genfromtxt(file_networks, delimiter=';'),'int')


networks        = [np.where(network_indices==i)[0] for i in range(7)] 



### do edge counting    
N, edges        = cffun.edge_counting(directory,subjects,ch_per_subject,freqs[:37],LFs,HFs,PS,CFC,CFC_dist,CFC_layer,'parc2009',channel_layers)

### do edge counting in layers

N_layer     = [[] for i in range(4)]
edges_layer = [[] for i in range(4)]
CFC_dummy   = [[[  None for i in range(N_ratios)] for j in range(N_LF)] for k in range(N_subj)]



reload(cffun)
for l in range(4):    
    for c1 in range(58):
        for c2 in range(30):
            for c3 in range(6):
                CFC_dummy[c1][c2][c3] = CFC_layer[c1][c2][c3][l]    
    N_layer[l], edges_layer[l]        = cffun.edge_counting(directory,subjects_red,ch_per_subject,freqs[:37],LFs,HFs,PS,CFC_dummy,'parc2009',channel_layers)
    print l

# save layer edges to pickle dump
fileout37 = directory + '_results\\_pickle dump\\Edges in layers ' + CF_type + ', ' + PS_metric + ', ' + parc + ', '  + time.strftime("%Y-%m-%d") + '.dat'  # save with pickle
pick.dump([N_layer,edges_layer],open(fileout37,'wb'))


for l in range(4):
    D = cffun.degree_analysis(edges_layer[l],N_layer[l],networks)
    cffun.write_degrees(directory,D,ratios2,'parc2009',CF_type, add_inf=layer_int) 


### analyze local CF and write to .csv
N, edges        = cffun.analyze_local(N,edges,networks,N_ch_layer)
cffun.write_csv_local(directory,edges,ratios2,parc,CF_type, add_inf='')

### analyze PS per parcel and network ####################
N, edges =  cffun.analyze_PS(N,edges,networks,N_ch_layer)

### save edges to pickle dump
fileout24 = directory + '_results\\_pickle dump\\Edges ' + CF_type + ', ' + PS_metric + ', ' + parc + ', '  + time.strftime("%Y-%m-%d") + '.dat'  # save with pickle
pick.dump([N,edges],open(fileout24,'wb'))


### load edges
if CF_type == 'CFS':
    filein24 = 'M:\\SEEG_Morlet\\_RAW_line_filtered\\_results\\_pickle dump\\Edges CFC, wPLI, parc2009.dat'

else:        
    filein24 = 'M:\\SEEG_Morlet\\_RAW_line_filtered\\_results\\_pickle dump\\Edges PAC, wPLI, parc2009.dat'

[N,edges] = pick.load(open(filein24,'rb'))



xlimA   = [xlims_CFC for i in range(6)]



########     plot local CFC per network    

dataL   = [edges.mean_PLV_local_pn[:,:,0],edges.K_local_pn [:,:,0],edges.mean_PLV_local_pn[:,:,1],edges.K_local_pn[:,:,1]]
titlesA = ['local str 1:2','local K [%] 1:2','local str 1:3','local K [%] 1:3' ]
ylabA   = ['Str','K [%]','GS','K [%]']
cmapA   = [my_cmap2 for i in dataL]
legendA = [network_names for i in dataL]
figsize = [7,4]
plots.semi_log_plot_multi(figsize,2,2,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,None,['ur',None,None,None],None,1,1,None,None,Nyt=[1,1,1,1],'auto',fontsize=8,markersize=0)

o15 = directory + '_results\\results by system\\Local ' + CF_type + ' by system.pdf'
plots.semi_log_plot_multi(figsize,2,2,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,o15,['ur',None,None,None],None,1,1,None,None,Nyt=[1,1,1,1],'auto',fontsize=8,markersize=0)




##################################################################
#### get regular and relative degrees, in-degrees and out-degrees

D = cffun.degree_analysis(edges,N,networks)
cffun.write_degrees(directory,D,ratios2,'parc2009',CF_type, add_inf='') 






###############################################################################
##########                       low-high analysis 

### run low-high analysis with Wilcoxon and permutation tests
alpha = 0.05
N_perm = 1000
N_rat  = 6
N, lh = cffun.low_to_high_analysis(edges,N,LFs,HFs,alpha,N_perm,parc,directory,networks,N_rat=N_rat)                        
N_min = 8
lh_thr    = cffun.low_to_high_threshold(lh,N,N_min,networks)           # apply N_min threshold 
lh = lh_thr
   
### save results with pickle
fileout4 = directory + '_results\\_pickle dump\\Low-high ' + CF_type + ', ' + PS_metric + ', ' + parc + ' ' + time.strftime("%Y-%m-%d") + '.dat'  # save with pickle
pick.dump(lh,open(fileout4,'wb'))

### load results
filein4 =  directory + '_results\\_pickle dump\\Low-high ' + CF_type + ', ' + PS_metric + ', parc2009, N_rat=2.dat'
lh = pick.load(open(filein4,'rb'))


  
### write in&out values csv
cffun.write_csv_low_to_high(directory,lh_thr,ratios2,parc, CF_type,add_inf=' corr')

### plot out-in
plots.semi_log_plot([10,5],lh.out_minus_in_degree_pn[:,:,0],LFs, [1,50], 'degree', network_names,None,'ur',None,True,cmap=my_cmap4,ncols=2,CI=False)   
plots.semi_log_plot([10,5],lh.out_minus_in_degree_pn[:,:,1],LFs, [1,50], 'degree', network_names,None,'ur',None,True,cmap=my_cmap4,ncols=2,CI=False)   


### plot results of difference tests: Wilc and perm.
dataL = [[lh.K_LTH_wilc[:,0]],[lh.K_LTH_wilc[:24,1]],[lh.K_LH[:,0]],[lh.K_LH[:24,1]]]    
ylimA = [[0,0.3] for i in range(4)]
plots.semi_log_plot_multi([7,4],2,2,dataL,LFs,[[0,50] for i in range(4)],['K','K','K','K'],['wilc 1:2','wilc 1:3','perm 1:2','perm 1:3'],['brg','brg','brg','brg'],ylimA=ylimA,show=True,xlab=[0,0,1,1],fontsize=12)



