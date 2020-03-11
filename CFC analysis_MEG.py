# -*- coding: utf-8 -*-
"""
@author: Felix Siebenhühner
"""


source_directory   = 'point to source directory here'
project_directory  = 'point to MEG data directory here'

import sys
sys.path.append(source_directory + 'Python27\\Utilities')
sys.path.append(source_directory + 'Python27\\Data Analysis')
sys.path.append(source_directory + 'Python27\\scientific colormaps')
import numpy as np
import cPickle as pick
import CF_functions as cffun
import plot_functions as plots
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import read_tdms as rt
import bootstrap as bst
from scipy import stats as stat
import read_values as rv
import statsmodels.sandbox.stats.multicomp as multicomp




### choose CF type and PS metric
sign_z_CFC   = 2.42
PS_metric    = 'wPLI'                                                       # 'PLV' or 'wPLI'
CF_type      = 'CFS'                                                        # 'PAC' or 'CFS'                                                  

if PS_metric == 'wPLI':                                                     
    metric2 = 'wpli'
    sign_z_PS    = 2.33 
else:
    metric2 = 'cPLV'
    sign_z_PS    = 2.42

### frequency settings
directory    = project_directory 
CFM_filename = directory + '_settings\\CF_matrix_MEG.csv'
CFM          = np.genfromtxt(CFM_filename, delimiter=';')
cutoff       = 316
cutoff_HF    = 316   
cutoff_LF    = 100                   
HFs_env = [CFM[:41,i+1] for i in range(6)]
LFs          = CFM[:,0][CFM[:,0]<cutoff_LF]         
freqs        = CFM[:,0][CFM[:,0]<cutoff]           
masks        = False
mask_type    = ''
N_freq       = len(freqs)
N_LF         = len(LFs)
N_CH         = 200
N_ratios     = 6
ratios       = ['1:'+str(i+2) for i in range(N_ratios)]    
ratios2      = ['1-'+str(i+2) for i in range(N_ratios)] 

xlims_PS = [1,cutoff]
xlims_CF = [1,cutoff_LF]

HFs       = []     
for f in range(len(LFs)):
    x   = CFM[f,1:N_ratios+1]
    HFs.append(x[np.where(x<cutoff_HF)])
N_rat_f = [len(HFs[i]) for i in range(len(LFs))]



#### choose cohort

data_directory = directory
subject_sets = ['S0006 set01','S0008 set01','S0035 set01','S0038 set02','S0039 set01','S0049 set01',             
                'S0113 set02','S0116 set01','S0116 set02','S0116 set03','S0117 set01','S0118 set01',
                'S0118 set02','S0118 set03','S0119 set01','S0119 set02','S0120 set01',
                'S0121 set01','S0123 set01','S0123 set02','S0124 set01','S0124 set02',
                'S0126 set01','S0127 set01','S0128 set01','S0130 set01','S0130 set02']
parcellation = 'parc2009_200AFS'
N_parc = 200
cutoff_HF = 350


subjects     = [s[:5] for s in subject_sets]                                  
subjects2    = list(set(subjects))
subjects2.sort()

N_sets   = len(subject_sets)
N_subj   = len(subjects2)


CP_PLV    = [None for s in subjects]
fidelity  = [None for s in subjects]


### get cross-patch PLV
for s,subj in enumerate(subjects):
    filename_base = directory + '_support_files\\' +  subj + '\\Cross-Patch PLV ' + parcellation
    CP_PLV[s]     = rt.read_complex_data_from_csv(filename_base, delimiter=';')


### get patch fidelity
for s,subj in enumerate(subjects):
    filename    = directory  + '_support_files\\_EOEC\\' + subj + '\\Patch Fidelity ' + parcellation + '.csv'
    fidelity[s] = np.genfromtxt(filename, delimiter=';') [:N_parc]
   
    
    

### get network assignments for parcels
filename          = directory + '\\_settings\\networks.csv'
network_indices   = np.array(np.genfromtxt(filename, delimiter=';'),'int')
networks          = [np.where(network_indices==i)[0] for i in range(7)]
network_names     = ['C','DM','DA','Lim','VA','SM','Vis']
N_networks        = len(network_names)

mean_CP_PLV   = np.mean(abs(np.array(CP_PLV)),0)
mean_fidelity = np.mean(np.array(fidelity),0)


fidelity_threshold = 0.05

if PS_metric == 'wPLI':                                                     
    CP_PLV_threshold = 1         
else:
    CP_PLV_threshold = 0.2143

fidelity_mask   = np.outer((mean_fidelity>fidelity_threshold),(mean_fidelity>fidelity_threshold))    # create a nice mask from 
CP_PLV_mask     = mean_CP_PLV < CP_PLV_threshold                         # CP-PLV is 1 on diagonal, so the mask diagonal will be 0 - good!
mask            = fidelity_mask*CP_PLV_mask

edges_retained = np.sum(mask)/float(N_parc*N_parc-N_parc)
np.fill_diagonal(mask,1)                            

### get distances   
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

dist_strs   = ['{:.1f}'.format(dd*100) for dd in dist_thresholds]  
N_dist_bins = len(dist_thresholds)-1   
distances   = [dist_strs[i]+'-'+dist_strs[i+1]+'cm' for i in range(N_dist_bins)]
dists_short = ['short','mid','long']

### colormaps for plotting
my_cmap  = plots.make_cmap([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.8, 0.0, 1.0)])
my_cmap2 = plots.make_cmap([(0.0, 0.0, 0.0), (0.5, 0.5, 1.0), (0.6, 0.6, 1.0), (0.7, 0.7, 1.0), (0.8, 0.8, 1.0),(0.9, 0.9, 1.0), (1, 1, 1)])
my_cmap3 = plots.make_cmap([(1.0, 0.0, 0.0), (0.0, 0.6, 0.0), (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (0.6, 0.4, 0.4)]) 
my_cmap4 = plots.make_cmap([(0.8, 0.6, 0.0), (1.0, 0.0, 0.0), (0.0, 0.8, 0.0), (0.1, 0.1, 0.1), (1.0, 0.4, 0.9), (0.0, 0.0, 1.0), (0.8, 0.0, 0.9)])
my_cmap5 = plots.make_cmap([(1,0,0), (0,1,0), (0,0,1)])
my_cmap6 = plots.make_cmap([(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.6, 0.0), (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (0.6, 0.4, 0.4)]) 

   
### set matplotlib parameters    
mpl.rcParams['pdf.fonttype'] = 42      # for PDF compatibility with Illustrator
mpl.rcParams.update({'font.size': 8})
mpl.rcParams.update({'axes.titlesize': 8})
mpl.rcParams.update({'axes.labelsize': 8})
mpl.rcParams.update({'legend.fontsize': 6})
mpl.rcParams.update({'xtick.labelsize': 7})
mpl.rcParams.update({'ytick.labelsize': 7})





###############################################################################
################                  COMPUTE PS AND CFC

# initialize lists
PS          =  [[   None for i in range(N_freq)] for j in range(N_sets)]       
PS_dist     =  [[[  None for b in range(N_dist_bins)] for i in range(N_freq)] for j in range(N_sets)] 
 
PS_ENV      =  [[[  None for i in range(len(HFs[j]))] for j in range(N_LF)] for k in range(N_sets)]
PS_ENV_dist =  [[[[ None for b in range(N_dist_bins)] for i in range(len(HFs[j]))] for j in range(N_LF)] for k in range(N_sets)]

CFC         =  [[[  None for i in range(len(HFs[j]))] for j in range(N_LF)] for k in range(N_sets)]
CFC_dist    =  [[[[ None for b in range(N_dist_bins)] for i in range(len(HFs[j]))] for j in range(N_LF)] for k in range(N_sets)]


#### compute PS  ####   
for s,sset in enumerate(subject_sets[:1]): 
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
                stats        = cffun.Stats_PS(np.nan)    
            PS_dist[s][f][d] = stats    
    print sset


#### for PAC: compute PLV stats of LF-envelope filtered HF data    
if CF_type == 'PAC':
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
                        stats = cffun.Stats_PS(np.nan)    
                    PS_ENV_dist[s][lf][hf][d] = stats                              
        print(time.strftime("%Y-%m-%d %H:%M") + '          ' + sset)
 

     
#### compute CFC  
for s,sset in enumerate(subject_sets):
    
    for lf,LF in enumerate(LFs):  
        for rat,HF in enumerate(HFs[lf]): 
         #   try:  
                LF_str      = '{:.2f}'.format(LF)                
                HF_str      = '{:.2f}'.format(HF) 
                LF_PS       = PS[s][lf].data_sign
                HF_idx      = np.where(freqs==HF)[0][0] 
                if CF_type == 'CFS':
                    HF_PS     = PS[s][HF_idx].data_sign
                else:
                    HF_PS     = PS_ENV[s][lf][rat].data_sign
                path        = directory + '_data\\_' + CF_type + '\\'
    
                file0       = path + sset + ' LF= ' + LF_str + ' HF= ' + HF_str + '.csv'            
                file_surr   = path + sset + ' LF= ' + LF_str + ' HF= ' + HF_str + '_surr.csv'
                masked_data = np.genfromtxt(file0,  delimiter=';') * mask   
                surr_data   = np.genfromtxt(file_surr, delimiter=';') * mask
                              
                stats       = cffun.K_stats_CFC_2(masked_data,surr_data,sign_z_CFC,LF_PS,HF_PS)
                CFC[s][lf][rat] = stats        
        
                for d in range(N_dist_bins):
                    dist_mask        = mask * ( ( (dist[s]>dist_thresholds[d]) * (dist[s]<=dist_thresholds[d+1]) )>0)
                    np.fill_diagonal(dist_mask,1)
                    masked_dataD     = masked_data*dist_mask
                    mask             = np.ones([N_CH,N_CH])                  
                    N_pot            = np.sum(dist_mask!=0)                                     
                    if N_pot>0:
                        stats       = cffun.K_stats_CFC_2(masked_dataD,surr_data,sign_z_CFC,LF_PS,HF_PS)
                    else:
                        stats = cffun.Stats_CFC(np.nan)    
                    CFC_dist[s][lf][rat][d] = stats        
#                
#            except:
#                print('error for ' + ratios[rat] + ' ' + file0 ) 
        print(time.strftime("%Y-%m-%d %H:%M") + '          ' + sset)
    









       


###############################################################################
##################      save data with Pickle

fileout1 = directory + '_results\\_pickle dump\\' + PS_metric + ',  z=' + '{:.2f}'.format(sign_z_PS) + ', '  +  time.strftime("%Y-%m-%d") + '.dat'
fileout2 = directory + '_results\\_pickle dump\\' + PS_metric + '_dist, z=' + '{:.2f}'.format(sign_z_PS) + ', ' +  time.strftime("%Y-%m-%d") + '.dat'

fileout3 = directory + '_results\\_pickle dump\\PS_envelope ' +  time.strftime("%Y-%m-%d") + '.dat'
fileout4 = directory + '_results\\_pickle dump\\PS_envelope_dist ' +  time.strftime("%Y-%m-%d") + '.dat'

fileout5 = directory + '_results\\_pickle dump\\' + CF_type + ' with '       + PS_metric + ' control, z=' + '{:.2f}'.format(sign_z_PS) + ', ' +  time.strftime("%Y-%m-%d") + '.dat'
fileout6 = directory + '_results\\_pickle dump\\' + CF_type + '_dist with '  + PS_metric + ' control, z=' + '{:.2f}'.format(sign_z_PS) + ', ' +  time.strftime("%Y-%m-%d") + '.dat'

pick.dump(PS,open(fileout1,'wb'))
pick.dump(PS_dist,open(fileout2,'wb'))

if CF_type == 'PAC':
    pick.dump(PS_ENV,open(fileout3,'wb'))
    pick.dump(PS_ENV_dist,open(fileout4,'wb'))

pick.dump(CFC,open(fileout5,'wb'))
pick.dump(CFC_dist,open(fileout6,'wb'))

###############################################################################




###############################################################################
###########        load data

use_IMs = False


if use_IMs:
    filein1 = directory + '_results\\_pickle dump\\' + PS_metric     + '.dat'
    filein2 = directory + '_results\\_pickle dump\\' + PS_metric     + '_dist.dat'    
    filein3 = directory + '_results\\_pickle dump\\PS_envelope.dat'
    filein4 = directory + '_results\\_pickle dump\\PS_envelope_dist.dat'    
    filein5 = directory + '_results\\_pickle dump\\' + CF_type + ' with '      + PS_metric + ' control.dat'
    filein6 = directory + '_results\\_pickle dump\\' + CF_type + '_dist with ' + PS_metric + ' control.dat'
else:
    filein1 = directory + '_results\\_pickle dump\\' + PS_metric     + '_no_IMs.dat'
    filein2 = directory + '_results\\_pickle dump\\' + PS_metric     + '_dist_no_IMs.dat'
    filein3 = directory + '_results\\_pickle dump\\PS_envelope_no_IMs.dat'
    filein4 = directory + '_results\\_pickle dump\\PS_envelope_dist_no_IMs.dat'
    filein5 = directory + '_results\\_pickle dump\\' + CF_type + ' with '      + PS_metric + ' control_no_IMs.dat'
    filein6 = directory + '_results\\_pickle dump\\' + CF_type + '_dist with ' + PS_metric + ' control_no_IMs.dat'


PS          = pick.load(open(filein1,'rb'))
PS_dist     = pick.load(open(filein2,'rb'))

CFC         = pick.load(open(filein5,'rb'))
CFC_dist    = pick.load(open(filein6,'rb'))


if CF_type == 'PAC':
    PS_ENV      = pick.load(open(filein3,'rb'))
    PS_ENV_dist = pick.load(open(filein4,'rb'))
    
    
###############################################################################






############################################################################### 
###########              GROUP STATS AND PLOTS FOR PS 
   
### get means and create plots for PS
PLV_PS_ps        = np.zeros([N_sets,N_freq])             # PS = "Phase Synch", ps = "per subject"
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
           
                              
mean_K_PS          =  [np.transpose(np.nanmean(K_PS_ps,0))]
mean_PLV_PS        =  [np.transpose(np.nanmean(PLV_PS_ps,0))] 
mean_K_PS_dist     =   np.transpose(np.nanmean(K_PS_dist_ps,0))
mean_PLV_PS_dist   =   np.transpose(np.nanmean(PLV_PS_dist_ps,0))     

K_PS_stats         = [bst.CI_from_bootstrap(K_PS_ps)] # returns [mean, mean_boot, lower, upper] x freq x ratio
PLV_PS_stats       = [bst.CI_from_bootstrap(PLV_PS_ps)] # returns [mean, mean_boot, lower, upper] x freq x ratio
K_PS_stats_dist    = [bst.CI_from_bootstrap(K_PS_dist_ps[:,:,i]) for i in range(N_dist_bins)] 
PLV_PS_stats_dist  = [bst.CI_from_bootstrap(PLV_PS_dist_ps[:,:,i]) for i in range(N_dist_bins)] 
 




##### PLOT PS with dists, indiv subjects

figsize     = [12.3,4.3]
rows        = 2
cols        = 3
dataL       = [PLV_PS_stats,PLV_PS_stats_dist,PLV_PS_ps,K_PS_stats,K_PS_stats_dist,K_PS_ps]
xlimA       = [xlims_PS for i in range(6)]
if PS_metric == 'wPLI':
    ylimA       = [[-0.005,0.12],[-0.005,0.12],[-0.01,0.5],[-5,100],[-5,100],[-5,100]]
else:
    ylimA       = [[-0.005,0.26],[-0.005,0.26],[-0.01,0.5],[-5,100],[-5,100],[-5,100]]

titlesA     = ['' for i in range(6)]                          #'mean '+PS_metric,'mean '+PS_metric+' per distance','mean '+PS_metric+' per subject','mean K','mean K per distance','mean K per subject']
legendA     = [None, distances, None, None, distances, None]
ylabA       = [PS_metric,'','','K','','']
cmapA       = ['winter','brg',my_cmap,'winter','brg',my_cmap]
CI          = [0.3,0.3,None,0.3,0.3,None]
legend_posA = [None,None,None,None,'ur',None]
xlab        = [0,0,0,1,1,1]
Ryt         = [1,1,0,0,0,0]
plots.semi_log_plot_multi(figsize,rows,cols,dataL,freqs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt)   

# save pdf
o67 = directory + '_results\\MEG PS\\MEG ' + PS_metric + '.pdf'
plots.semi_log_plot_multi(figsize,rows,cols,dataL,freqs,xlimA,ylabA,titlesA,cmapA,legendA,o67,legend_posA,ylimA,False,1,CI)   





#### save plot data as .csv files

o31  =  directory + '_results\\_plot_data_new\\PS\\MEG\\MEG ' + PS_metric + ' K.csv'
o32  =  directory + '_results\\_plot_data_new\\PS\\MEG\\MEG ' + PS_metric + ' GS.csv'
o33a = [directory + '_results\\_plot_data_new\\PS\\MEG\\MEG ' + PS_metric + ' K, ' + dists_short[i] + '.csv' for i in range(3)] 
o34a = [directory + '_results\\_plot_data_new\\PS\\MEG\\MEG ' + PS_metric + ' GS, ' + dists_short[i] + '.csv' for i in range(3)] 

np.savetxt(o31,K_PS_stats[0][:3],delimiter=';')
np.savetxt(o32,PLV_PS_stats[0][:3],delimiter=';')
for i in range(3):
    np.savetxt(o33a[i],K_PS_stats_dist[i][:3],delimiter=';')
    np.savetxt(o34a[i],PLV_PS_stats_dist[i][:3],delimiter=';')






###############################################################################
######################     GROUP STATS AND PLOTS FOR CFC        



   
### init CFC arrays
K_CFC_ps               = np.full([N_sets,N_LF,N_ratios],np.nan)
PLV_CFC_ps             = np.full([N_sets,N_LF,N_ratios],np.nan)
PLV_CFC_ps_sig         = np.full([N_sets,N_LF,N_ratios],np.nan)
K_CFC_local_ps         = np.full([N_sets,N_LF,N_ratios],np.nan)
PLV_CFC_local_ps       = np.full([N_sets,N_LF,N_ratios],np.nan)    
K_CFC_ps_mod           = np.full([N_sets,N_LF,N_ratios],np.nan)
K_CFC_ps_mod_w         = np.full([N_sets,N_LF,N_ratios],np.nan)
PLV_CFC_ps_mod         = np.full([N_sets,N_LF,N_ratios],np.nan)
N_pot_mod_subj         = np.full([N_sets,N_LF,N_ratios],np.nan)
K_CFC_dist_ps          = np.full([N_sets,N_LF,N_ratios,N_dist_bins],np.nan)
PLV_CFC_dist_ps        = np.full([N_sets,N_LF,N_ratios,N_dist_bins],np.nan)
K_CFC_dist_ps_mod      = np.full([N_sets,N_LF,N_ratios,N_dist_bins],np.nan)
PLV_CFC_dist_ps_mod    = np.full([N_sets,N_LF,N_ratios,N_dist_bins],np.nan)
K_CFC_dist_ps_mod_w    = np.full([N_sets,N_LF,N_ratios,N_dist_bins],np.nan)
   
### get CFC values
for lf,LF in enumerate(LFs): 
   for hf,HF in enumerate(HFs[lf]):
       if HF<cutoff_HF:
           for s,ss in enumerate(subject_sets):   
               try:                 
                   K_CFC_local_ps [s,lf,hf]        = 100*CFC[s][lf][hf].K_local                   
                   K_CFC_ps       [s,lf,hf]        = 100*CFC[s][lf][hf].K
                   PLV_CFC_ps     [s,lf,hf]        = CFC[s][lf][hf].mean_masked
                   PLV_CFC_ps_sig [s,lf,hf]        = CFC[s][lf][hf].mean_sign        
                   K_CFC_ps_mod   [s,lf,hf]        = 100*CFC[s][lf][hf].K_mod
                   K_CFC_ps_mod_w [s,lf,hf]        = 100*CFC[s][lf][hf].K_mod * CFC[s][lf][hf].N_pot_mod    #weighting by subject's N_pot_mod
                   PLV_CFC_ps_mod [s,lf,hf]        = CFC[s][lf][hf].mean_mod                    
                   N_pot_mod_subj [s,lf,hf]        = CFC[s][lf][hf].N_pot_mod
               except:
                   pass
               
               try:
                   PLV_CFC_local_ps[s,lf,hf]  = CFC[s][lf][hf].mean_local
               except: 
                   PLV_CFC_local_ps[s,lf,hf]  = np.nan

               for d in range(N_dist_bins):
                                      
                   PLV_CFC_dist_ps [s,lf,hf,d] = CFC_dist[s][lf][hf][d].mean_masked
                   K_CFC_dist_ps   [s,lf,hf,d] = 100*CFC_dist[s][lf][hf][d].K
                   try:
                       PLV_CFC_dist_ps_mod [s,lf,hf,d] = CFC_dist[s][lf][hf][d].mean_mod
                       K_CFC_dist_ps_mod   [s,lf,hf,d] = 100*CFC_dist[s][lf][hf][d].K_mod
                       K_CFC_dist_ps_mod_w [s,lf,hf,d] = 100*CFC_dist[s][lf][hf][d].K_mod * CFC[s][lf][hf].N_pot_mod 
                   except: 
                       PLV_CFC_dist_ps_mod [s,lf,hf,d] = np.nan
                       K_CFC_dist_ps_mod   [s,lf,hf,d] = np.nan
                       K_CFC_dist_ps_mod_w [s,lf,hf,d] = np.nan * CFC[s][lf][hf].N_pot_mod 




### get CFC group means and 95% confidence intervals
N_boot=1000
    
PLV_CFC_stats       = [np.array(bst.CI_from_bootstrap(PLV_CFC_ps[:,:,i]))       for i in range(N_ratios)] # returns [mean, mean_boot, lower, upper] x freq x ratio
K_CFC_stats         = [np.array(bst.CI_from_bootstrap(K_CFC_ps[:,:,i])) -1      for i in range(N_ratios)]  
K_CFC_stats_mod     = [np.array(bst.CI_from_bootstrap(K_CFC_ps_mod[:,:,i]))-1   for i in range(N_ratios)] 
K_CFC_stats_mod_w   = [np.array(bst.CI_from_bootstrap(K_CFC_ps_mod_w[:,:,i],N_boot,2.5,97.5,N_pot_mod_subj[:,:,i]))-1  for i in range(N_ratios)] 
PLV_CFC_local_stats = [np.array(bst.CI_from_bootstrap(PLV_CFC_local_ps[:,:,i])) for i in range(N_ratios)]
K_CFC_local_stats   = [np.array(bst.CI_from_bootstrap(K_CFC_local_ps[:,:,i]))-1 for i in range(N_ratios)]  

PLV_CFC_dist_12_stats     = [bst.CI_from_bootstrap(PLV_CFC_dist_ps[:,:,0,i]) for i in range(N_dist_bins)] # returns [mean, mean_boot, lower, upper] x freq x dist
PLV_CFC_dist_13_stats     = [bst.CI_from_bootstrap(PLV_CFC_dist_ps[:,:,1,i]) for i in range(N_dist_bins)] # returns [mean, mean_boot, lower, upper] x freq x dist 
K_CFC_dist_12_stats       = [np.array(bst.CI_from_bootstrap(K_CFC_dist_ps[:,:,0,i]))-1 for i in range(N_dist_bins)] # returns [mean, mean_boot, lower, upper] x freq x dist
K_CFC_dist_13_stats       = [np.array(bst.CI_from_bootstrap(K_CFC_dist_ps[:,:,1,i]))-1 for i in range(N_dist_bins)] # returns [mean, mean_boot, lower, upper] x freq x dist
K_CFC_dist_12_stats_mod   = [np.array(bst.CI_from_bootstrap(K_CFC_dist_ps_mod[:,:,0,i]))-1 for i in range(N_dist_bins)] # returns [mean, mean_boot, lower, upper] x freq x dist
K_CFC_dist_13_stats_mod   = [np.array(bst.CI_from_bootstrap(K_CFC_dist_ps_mod[:,:,1,i]))-1 for i in range(N_dist_bins)] # returns [mean, mean_boot, lower, upper] x freq x dist
K_CFC_dist_12_stats_mod_w = [np.array(bst.CI_from_bootstrap(K_CFC_dist_ps_mod_w[:,:,0,i],N_boot,2.5,97.5,N_pot_mod_subj[:,:,i]))-1 for i in range(N_dist_bins)] # returns [mean, mean_boot, lower, upper] x freq x dist
K_CFC_dist_13_stats_mod_w = [np.array(bst.CI_from_bootstrap(K_CFC_dist_ps_mod_w[:,:,1,i],N_boot,2.5,97.5,N_pot_mod_subj[:,:,i]))-1 for i in range(N_dist_bins)] # returns [mean, mean_boot, lower, upper] x freq x dist
 
K_CFC_stats                  = [[K_CFC_stats[rat][i]*(K_CFC_stats[rat][i]>=0) for i in range(4)] for rat in range(6)]
K_CFC_stats_mod_w            = [[K_CFC_stats_mod[rat][i]*(K_CFC_stats_mod[rat][i]>=0) for i in range(4)] for rat in range(6)]
K_CFC_dist_12_stats          = [[K_CFC_dist_12_stats[di][i]*(K_CFC_dist_12_stats[di][i]>=0) for i in range(4)] for di in range(3)]
K_CFC_dist_12_stats_mod_w    = [[K_CFC_dist_12_stats_mod[di][i]*(K_CFC_dist_12_stats_mod[di][i]>=0) for i in range(4)] for di in range(3)]
K_CFC_dist_13_stats          = [[K_CFC_dist_13_stats[di][i]*(K_CFC_dist_13_stats[di][i]>=0) for i in range(4)] for di in range(3)]
K_CFC_dist_13_stats_mod_w    = [[K_CFC_dist_13_stats_mod[di][i]*(K_CFC_dist_13_stats_mod[di][i]>=0) for i in range(4)] for di in range(3)]



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

max_PLV_CFC       = np.nanmax(np.array(mean_PLV_CFC)[:,6:])
max_K_CFC         = np.nanmax(mean_K_CFC)   
max_PLV_CFC_local = np.nanmax(np.array(mean_PLV_CFC_local)[:,6:])
max_K_CFC_local   = np.nanmax(mean_K_CFC_local)   
 




### remove 0 entries in higher ratios

#mean_K_CFC    = list(mean_K_CFC)
#mean_K_CFC    = [np.array(filter(lambda a: a != 0, i)) for i in mean_K_CFC]
mean_PLV_CFC  = list(mean_PLV_CFC)
mean_PLV_CFC  = [np.array(filter(lambda a: a != 0, i)) for i in mean_PLV_CFC]

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
K_CFC_dist_13_stats_mod = [[np.array(filter(lambda a: a != -1, i)) for i in j] for j in K_CFC_dist_13_stats_mod]

K_CFC_dist_13_stats_mod_w = [[np.array(filter(lambda a: a != -1, i)) for i in j] for j in K_CFC_dist_13_stats_mod_w]

K_CFC_dist_13_stats     = [[np.array(i[~np.isnan(i)]) for i in j] for j in K_CFC_dist_13_stats]
K_CFC_dist_13_stats_mod = [[np.array(i[~np.isnan(i)]) for i in j] for j in K_CFC_dist_13_stats_mod]

K_CFC_dist_13_stats_mod_w = [[np.array(i[~np.isnan(i)]) for i in j] for j in K_CFC_dist_13_stats_mod_w]





###############################################################################
#############         plot CFC 

figsize1 = [10,4]  
figsize2 = [6.3,2.3]   
rows    = 2
cols    = 3
dataL   = [PLV_CFC_stats[:1],K_CFC_stats[:1],K_CFC_stats_mod_w[:1],
           PLV_CFC_stats[1:],K_CFC_stats[1:],K_CFC_stats_mod_w[1:]]    
xlimA   = [xlims_CF for i in range(6)]
titlesA = ['' for i in range(6)]        #['mean PLV','mean K','mean K (controlled)','','','']
if CF_type == 'CFS':
    ylimA   = [[-0.0045,0.045], [-1, 11],   [-1, 11],  [-0.0038,0.038], [-0.17,1.56], [-0.17,1.56]]  
else:
    ylimA   = [[-0.002,0.068], [-1, 13.5], [-1.3,  13.5],  [-0.002,0.058], [-1.3,   13.5],[-1.3,   13.5]]  
legendA = [ratios[:1],ratios[:1],ratios[:1],
           ratios[1:],ratios[1:],ratios[1:],]    
ylabA   = ['PLV','K [%]','K [%]',
           'PLV','K [%]','K [%]']
cmapA   = ['brg','brg','brg',my_cmap3,my_cmap3,my_cmap3]
legend_posA = ['ur',None,None,None,None,None]
CI      = [0.2 for i in range(6)]
xlabA   = [0,0,0,1,1,1]
Ryt     = [1,1,1,1,1,1]
plots.semi_log_plot_multi(figsize1,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlabA,Ryt,fontsize=10)   

## export PDF
o80 = directory + '_results\\MEG CF NEW\\MEG ' + CF_type + ', controlled with ' + PS_metric + '.pdf'
plots.semi_log_plot_multi(figsize2,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,o80,legend_posA,ylimA,False,1,CI,xlab,Ryt,fontsize=7)   


#### plot heatmap

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
o90  = directory + '_results\\MEG CF NEW\\MEG ' + CF_type + ' heatmap, uncontrolled.pdf'
o90a = directory + '_results\\MEG CF NEW\\MEG ' + CF_type + ' heatmap, controlled with '+PS_metric+'.pdf'
plots.simple_CF_plot(data1,figsize,'ratio','Low Frequency [Hz]',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax1,ztix=ztix1,outfile=o90)             
plots.simple_CF_plot(data2,figsize,'ratio','Low Frequency [Hz]',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax2,ztix=ztix2,outfile=o90a)             
   






###############################################################################
########                  plot CFC in distance bins

figsize1 = [10,4]  
figsize2 = [6.3,2.3] 
rows    = 2
cols    = 3
dataL   = [PLV_CFC_dist_12_stats, K_CFC_dist_12_stats, K_CFC_dist_12_stats_mod_w,
           PLV_CFC_dist_13_stats, K_CFC_dist_13_stats, K_CFC_dist_13_stats_mod_w]        
xlimA   = [xlims_CF for i in range(6)]
titlesA = ['mean PLV per distance (1:2)', 'mean K per distance (1:2)', 'mean K per dist. (1:2, contr.)',
           'mean PLV per distance (1:3)', 'mean K per distance (1:3)', 'mean K per dist. (1:3, contr.)']
if CF_type =='CFS':
    ylimA   = [[-0.0045,0.045], [-1.8,18],  [-1.8,18], [-0.0045,0.045], [-0.32,3.2], [-0.32,3.2]]  ### CFC    
else:
    ylimA   = [[-0.006,0.059], [-1.8,18],  [-1.8,18], [-0.006,0.059], [-1.8,18],   [-1.8,18]]  ### PAC             
legendA = [distances for i in range(6)]
ylabA   = ['PLV','K','K','PLV', 'K','K']
cmapA   = ['brg','brg','brg','brg','brg','brg']
legend_posA = [None,None,None,None,None,None]
CI      = [0.3 for i in range(6)]
xlab    = [0,0,0,1,1,1]
Ryt     = [1,1,1,1,1,1]
plots.semi_log_plot_multi(figsize1,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt,fontsize=10)   

#export PDF
o81 = directory + '_results\\MEG CF NEW\\MEG ' + CF_type + ', controlled with ' + PS_metric + ', distance bins.pdf'
plots.semi_log_plot_multi(figsize2,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,o81,legend_posA,ylimA,False,1,CI,xlab,Ryt,fontsize=7)   




###############################################################################
#####              GROUPS STATS AND PLOTS FOR ENVELOPE    

# init ENV arrays
K_ENV_ps               = np.full([N_sets,N_LF,N_ratios,],np.nan)
PLV_ENV_ps             = np.full([N_sets,N_LF,N_ratios],np.nan)
K_ENV_local_ps         = np.full([N_sets,N_LF,N_ratios],np.nan)
PLV_ENV_local_ps       = np.full([N_sets,N_LF,N_ratios],np.nan)
K_ENV_dist_ps          = np.full([N_sets,N_LF,N_ratios,N_dist_bins],np.nan)
PLV_ENV_dist_ps        = np.full([N_sets,N_LF,N_ratios,N_dist_bins],np.nan)    

### get ENV values
for lf,LF in enumerate(LFs): 
   for hf,HF in enumerate(HFs[lf]):
       if HF<cutoff_HF:
           for s,ss in enumerate(subject_sets):             
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





################## plot ENV
figsize1 = [9.7,3.6]   
figsize2 = [5.3,2.3]  
rows    = 2
cols    = 2
dataL   = [PLV_ENV_stats,K_ENV_stats,
           PLV_ENV_stats,K_ENV_stats]    
xlimA1  = [[1,100] for i in range(4)]
xlimA2  = [[1,330] for i in range(4)]
titlesA = ['' for i in range(4)] 
ylimA   = [[-0.009,0.095],  [-5, 59],[-0.009,0.095],  [-5,59], ] 
legendA = [ratios[:1],ratios[:1],
           ratios[1:],ratios[1:],]    
ylabA   = ['PLV','K [%]', 'PLV','K [%]',]
cmapA   = [my_cmap6,my_cmap6,my_cmap6,my_cmap6]
legend_posA = [None,None,None,None,]
CI      = [0.2 for i in range(4)]
xlab    = [0,0,1,1,]
Ryt     = [1,1,1,1,]
plots.semi_log_plot_multi(figsize1,rows,cols,dataL,LFs,    xlimA1,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt,fontsize=10)   
plots.semi_log_plot_multi(figsize1,rows,cols,dataL,HFs_env,xlimA2,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt,fontsize=10)   

## export PDF
o80e = directory + '_results\\MEG CF NEW\\MEG Envelopes LFx.pdf'
o80f = directory + '_results\\MEG CF NEW\\MEG Envelopes HFx.pdf'
plots.semi_log_plot_multi(figsize,rows,cols,dataL,LFs,    xlimA1,ylabA,titlesA,cmapA,legendA,o80e,legend_posA,ylimA,False,1,CI,xlab,Ryt,fontsize=7)   
plots.semi_log_plot_multi(figsize,rows,cols,dataL,HFs_env,xlimA2,ylabA,titlesA,cmapA,legendA,o80f,legend_posA,ylimA,False,1,CI,xlab,Ryt,fontsize=7)   



###### plot ENV heatmap
data = np.transpose(mean_K_CFC)
figsize_hm = [1.6,1.9]
LF_ics = [0,4,8,12,16,20,24,28,32,36,40]
LF_map = ['1.1', '2.2', '3.7', '5.9', '9.0', '13.1', '19.7', '28.7', '42.5', '65.3', '95.6']
zmax = 45
ztix=[0,15,30,45]
plots.simple_CF_plot(data,figsize_hm,'ratio','LF',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax,ztix=ztix,outfile=None)             
   
# export PDF 
o99 = directory + '_results\\MEG CF NEW\\MEG Envelopes heatmap.pdf'
plots.simple_CF_plot(data,figsize_hm,'ratio','LF',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax,ztix=ztix,outfile=o99)          





###############################################################################
##########            plot local CF 

figsize1 = [8.9,4.3]  
figsize2 = [4.5,2.3]   
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
plots.semi_log_plot_multi(figsize1,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,1,CI,xlab,Ryt,fontsize=10)   

## export PDF
o82 = directory + '_results\\MEG CF NEW\\MEG ' + CF_type + ' local .pdf'
plots.semi_log_plot_multi(figsize2,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,o82,legend_posA,ylimA,False,1,CI,xlab,Ryt,fontsize=7)   



#### plot heatmap
data = np.transpose(mean_K_CFC_local)
figsize_hm = [1.6,1.9]
LF_ics = [0,4,8,12,16,20,24,28,32,36,40]
LF_map = ['1.1', '2.2', '3.7', '5.9', '9.0', '13.1', '19.7', '28.7', '42.5', '65.3', '95.6']
zmax = 24
ztix=[0,6,12,18,24]
plots.simple_CF_plot(data,figsize_hm,'ratio','LF',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax,ztix=ztix,outfile=None)             
   
## export PDF 
o93 = directory + '_results\\MEG CF NEW\\MEG ' + CF_type + ' local heatmap.pdf'
plots.simple_CF_plot(data,figsize_hm,'ratio','LF',np.arange(0.5,5.6,1),LF_ics,ratios,LF_map,zmax=zmax,ztix=ztix,outfile=o93)          



###### save plot data as csv files

o41a = [directory + '_results\\_plot_data_new\\' + CF_type + '\\MEG\\MEG ' + CF_type  + ' K ' + r + '.csv' for r in ratios2 ]
o41b = [directory + '_results\\_plot_data_new\\' + CF_type + '\\MEG\\MEG ' + CF_type  + ' K_mod using ' + PS_metric + ' ' + r + '.csv' for r in ratios2 ]
o41c = [directory + '_results\\_plot_data_new\\' + CF_type + '\\MEG\\MEG ' + CF_type  + ' GS ' + r + '.csv' for r in ratios2 ]
o42a = [directory + '_results\\_plot_data_new\\' + CF_type + '_dist\\MEG\\MEG ' + CF_type + ' 1-2 K ' + d + ' .csv' for d in dists_short ]
o42b = [directory + '_results\\_plot_data_new\\' + CF_type + '_dist\\MEG\\MEG ' + CF_type + ' 1-2 K_mod using ' + PS_metric + ' ' + d + '.csv' for d in dists_short ]
o42c = [directory + '_results\\_plot_data_new\\' + CF_type + '_dist\\MEG\\MEG ' + CF_type + ' 1-2 GS ' + d + ' .csv' for d in dists_short ]
o43a = [directory + '_results\\_plot_data_new\\' + CF_type + '_dist\\MEG\\MEG ' + CF_type + ' 1-3 K ' + d + '.csv' for d in dists_short ]
o43b = [directory + '_results\\_plot_data_new\\' + CF_type + '_dist\\MEG\\MEG ' + CF_type + ' 1-3 K_mod using ' + PS_metric + ' ' + d + '.csv' for d in dists_short ]
o43c = [directory + '_results\\_plot_data_new\\' + CF_type + '_dist\\MEG\\MEG ' + CF_type + ' 1-3 GS ' + d + '.csv' for d in dists_short ]
o44a = [directory + '_results\\_plot_data_new\\' + CF_type + '_local\\MEG\\MEG local ' + CF_type  + ' K ' + r + '.csv' for r in ratios2 ]
o44b = [directory + '_results\\_plot_data_new\\' + CF_type + '_local\\MEG\\MEG local ' + CF_type  + ' GS ' + r + '.csv' for r in ratios2 ]       
o45a = [directory + '_results\\_plot_data_new\\ENV\\MEG\\MEG ENV K ' + r + '.csv' for r in ratios2 ]
o45b = [directory + '_results\\_plot_data_new\\ENV\\MEG\\MEG ENV GS ' + r + '.csv' for r in ratios2 ]
for r in range(6):
    np.savetxt(o41a[r],K_CFC_stats      [r][:3],delimiter=';')
    np.savetxt(o41b[r],K_CFC_stats_mod_w[r][:3],delimiter=';')
    np.savetxt(o41c[r],PLV_CFC_stats    [r][:3],delimiter=';')
    for d in range(3):
        np.savetxt(o42a[d],K_CFC_dist_12_stats      [d][:3],delimiter=';')
        np.savetxt(o42b[d],K_CFC_dist_12_stats_mod_w[d][:3],delimiter=';')
        np.savetxt(o42c[d],PLV_CFC_dist_12_stats    [d][:3],delimiter=';')
        np.savetxt(o43a[d],K_CFC_dist_13_stats      [d][:3],delimiter=';')
        np.savetxt(o43b[d],K_CFC_dist_13_stats_mod_w[d][:3],delimiter=';')
        np.savetxt(o43c[d],PLV_CFC_dist_13_stats    [d][:3],delimiter=';')
    np.savetxt(o44a[r],K_CFC_local_stats[r][:3],delimiter=';')
    np.savetxt(o44b[r],PLV_CFC_local_stats[r][:3],delimiter=';')
    if CF_type == 'PAC':
        np.savetxt(o45a[r],K_ENV_stats[r][:3],delimiter=';')
        np.savetxt(o45b[r],PLV_ENV_stats[r][:3],delimiter=';')






###############################################################################
######## compare long and short distance bins


### get CFC values
for lf,LF in enumerate(LFs): 
   for rat,HF in enumerate(HFs[lf]):
           for s in range(N_sets):                                      
               for d in range(N_dist_bins): 
                       PLV_CFC_dist_ps  [s,lf,rat,d]    = CFC_dist[s][lf][rat][d].mean_masked
                       K_CFC_dist_ps    [s,lf,rat,d]    = CFC_dist[s][lf][rat][d].K
                       K_CFC_dist_ps_mod[s,lf,rat,d]    = CFC_dist[s][lf][rat][d].K_mod
                                              
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


### plot long vs short bin results
for co in range(3):
    dataA = [[s_12_ps[:,co]],[s_12[:,co]],[s_12_mod[:,co]],[s_13_ps[:,co]],[s_13[:,co]],[s_13_mod[:,co]]]
    cmapA = ['brg','brg','brg','brg','brg','brg']
    xlimA = [xlims_CF for i in range(6)]
    plots.semi_log_plot_multi([7.7,3],2,3,dataA,LFs,xlimA,['','','','','',''],['1-2','1-2','1-2 c','1-3','1-3','1-3 c'],cmapA,None,None,None,None,True,1,None,None,None,'auto',8,3)   
    
    # save pdf
    combo_str = dists_short[combo1[co]] + '-' + dists_short[combo2[co]] 
    o83 = directory + '_results\\MEG CF NEW\\MEG ' + CF_type + ', controlled with ' + PS_metric + ', distance comparison ' + combo_str + '.pdf'
    plots.semi_log_plot_multi([7.7,3],2,3,dataA,LFs,xlimA,['','','','','',''],['1-2','1-2','1-2 c','1-3','1-3','1-3 c'],cmapA,None,o83,None,None,0,1,None,None,None,'auto',8,3)   






  

###############################################################################
################### analyze local CFC

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







##################################################################################
################ count edges and compare low-high strength across parcels
    
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
##########################  get degrees and strengths ############################


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
   








###############################################################################
######################  correlation with scores  ##############################

# colormap for plots
import vik as col10                                                       # substitute berlin with color of your choice
colordata10 = col10.cm_data
my_cmap10   = plots.make_cmap(colordata10)


subj_indices = np.array([0, 1, 2, 3, 4, 5, 6,  7, 7,  7,  8 ,  9,  9,  9, 10, 10, 11, 12, 13, 13, 14, 14, 15, 16, 17, 18, 18])
subj_weights = np.array([1, 1, 1, 1, 1, 1, 1,.33,.33,.33, 1 ,.33,.33,.33, .5, .5,  1,  1, .5, .5, .5, .5,  1,  1,  1, .5, .5])


PLV_CFC_ps_new = np.zeros([19,41,6])

# mean strength (all)
for s in range(19):
    PLV_CFC_ps_new[s] = np.mean([PLV_CFC_ps[i] for i in np.where(subj_indices==s)[0]],0)

# OR: mean strength (sign)
for s in range(19):
    PLV_CFC_ps_new[s] = np.mean([PLV_CFC_ps_sig[i] for i in np.where(subj_indices==s)[0]],0)

# OR: K
for s in range(19):
    PLV_CFC_ps_new[s] = np.mean([K_CFC_ps[i] for i in np.where(subj_indices==s)[0]],0)






freq_str = [str(f) for f in freqs]
#####   load scores
testtype = 'nepsy'

if testtype =='clinical':
    # select scores clinical
    score_vars = [ 'STAI-trait', 'ASRS1', 'ASRS2', 'SCL anx', 'BDI-II',  'SCL depr', 'SCL_GSI','Audit' ]
    dict_list, nest_dict = rv.get_dict('M:\\Resting_state_data\\_scores\\np_clinical.csv')
else:   
    # select scores neuropsychological tests
    score_vars = [ 'ForwDigits', 'BackDigits', 'Digit symbol coding_test','LNS',
                  'TMT-A','TMT-B','Zoo map_plan','Zoo map_time']
    dict_list, nest_dict = rv.get_dict(directory + '_scores\\np_nepsy.csv')



# get scores
dicts = []
N_scores = len(score_vars)

for ss, sset in enumerate(subjects2):
    for di in dict_list:
        if di.get('subj') == subjects2[ss]:
          dicts.append(di)

score_mat = np.zeros([N_subj,N_scores])
for s in range(N_subj):
    for v, var in enumerate(score_vars):
        score_mat[s,v] = dicts[s].get(var)

N_entries = np.sum((score_mat!=-1),0)

null_entries = (score_mat==-1)
null_entries.astype('int')


###############################################################################
#########       compute correlations for individual CF ratios         #########

ct=0
corr_types = ['spearman','pearson']
corr_type  = corr_types[ct]

# compute CF correlations voxel-wise
corrs  = np.zeros([41,N_ratios,N_scores])
p_vals = np.ones([41,N_ratios,N_scores])

for v,var in enumerate(score_vars):
    for f in range(41):
        for r in range(N_ratios):
            inds = np.where(null_entries[:,v]==0)[0]
            CF_vals  = PLV_CFC_ps_new[inds,f,r]
            sc_vals  = score_mat[inds,v]
            try:
                if corr_type == 'spearman':
                    corrs[f,r,v], p_vals[f,r,v] = stat.spearmanr(CF_vals,sc_vals)
                else:
                    corrs[f,r,v], p_vals[f,r,v] = stat.pearsonr(CF_vals,sc_vals)

            except:
                pass

corrs      = np.nan_to_num(corrs)
corrs_sig  = corrs * (p_vals<0.05)

corr_method = 'fdr_bh'
for i in range(8):
    corrv = (multicomp.multipletests(np.reshape(p_vals[:,:,i],41*6),method=corr_method)[1])    
    print np.sum(corrv<0.05)


# plot CF correlations voxel-wise
fontsize=7

y_indices = np.append(np.arange(0,39,3),[])
y_indices = np.arange(0,40,3)

fig1,axes=plt.subplots(2,4,figsize=[10,7])             
for v,var in enumerate(score_vars):
    ax = axes[v/4,v%4]        
    dataP = corrs[:,:,v]
    if (testtype=='nepsy') & (v>3):
        dataP = dataP * -1.
    dataP[np.isnan(dataP)]=0
    vm = .6
    im = ax.imshow(dataP,origin='bottom',aspect=.5,interpolation='none',cmap=my_cmap10,vmin=-vm,vmax=vm)
    if (v+1)%4==0:
        cbar = fig1.colorbar(im, ax=ax)
        cbar.ax.tick_params(axis='y', direction='out',labelsize=fontsize) 
    ax.set_title(var,fontsize=fontsize)
    ax.set_yticks(y_indices)
    ax.set_xticks(np.arange(0,6,1))
    ax.set_yticklabels(np.array(freq_str)[y_indices],fontsize=fontsize)
    ax.set_xticklabels(ratios,rotation=45,fontsize=fontsize)
    ax.tick_params(direction='out',top=False,right=False) 
    if v%4!=0:
        ax.tick_params(labelleft=False)
    outfile = 'K:\\palva\\resting_state\\RS_CF_MEG\\_results\\_plot_data_new\\nepsy_corr\\' + CF_type + ' ' + var + '.csv'
    np.savetxt(outfile,corrs[:,:,v],delimiter=';')



fig2,axes=plt.subplots(2,4,figsize=[10,7])             
for v,var in enumerate(score_vars):
    ax = axes[v/4,v%4]        
    dataP = corrs_sig[:,:,v]
    if (testtype=='nepsy') & (v>3):
        dataP = dataP * -1.
    dataP[np.isnan(dataP)]=0
    vm = .6
    im = ax.imshow(dataP,origin='bottom',aspect=.5,interpolation='none',cmap=my_cmap10,vmin=-vm,vmax=vm)
    if (v+1)%4==0:
        cbar = fig2.colorbar(im, ax=ax)
        cbar.ax.tick_params(axis='y', direction='out',labelsize=fontsize) 
    ax.set_title(var,fontsize=fontsize)
    ax.set_yticks(y_indices)
    ax.set_xticks(np.arange(0,6,1))
    ax.set_yticklabels(np.array(freq_str)[y_indices],fontsize=fontsize)
    ax.set_xticklabels(ratios,rotation=45,fontsize=fontsize)
    ax.tick_params(direction='out',top=False,right=False) 
    if v%4!=0:
        ax.tick_params(labelleft=False)


outfile1 = directory + '_results\\_score_corr\\' + testtype + ' ' + corr_type + ' ' + CF_type + '.pdf'
outfile2 = directory + '_results\\_score_corr\\' + testtype + ' ' + corr_type + ' ' + CF_type + ' sig.pdf'

fig1.savefig(outfile1)
fig2.savefig(outfile2)


