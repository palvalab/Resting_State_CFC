# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:53:14 2019

@author: Localadmin_fesieben
"""


projects_directory  =   'K:\\palva\\'
#source_directory    =   'D:\\felix\\OL2015\source\\'
source_directory    = 'E:\\HY-data\\FESIEBEN\\OL2000\\OL2015\\source\\'

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
import bootstrap as bst
from scipy import stats as stat
from scipy import sparse
import read_values as rv
import statsmodels.sandbox.stats.multicomp as multicomp

sys.path.append(source_directory + 'Python27\\scientific colormaps')
import berlin as col1                                                       # substitute berlin with color of your choice
colordata1 = col1.cm_data
my_cmapS1   = plots.make_cmap(colordata1)
my_cmap4 = plots.make_cmap([(0.8, 0.6, 0.0), (1.0, 0.0, 0.0), (0.0, 0.8, 0.0), (0.1, 0.1, 0.1), (1.0, 0.4, 0.9), (0.0, 0.0, 1.0), (0.8, 0.0, 0.9)])



### set matplotlib parameters    
mpl.rcParams['pdf.fonttype'] = 42      # for PDF compatibility with Illustrator
mpl.rcParams.update({'font.size': 8})
mpl.rcParams.update({'axes.titlesize': 8})
mpl.rcParams.update({'axes.labelsize': 8})
mpl.rcParams.update({'legend.fontsize': 7})
mpl.rcParams.update({'xtick.labelsize': 8})
mpl.rcParams.update({'ytick.labelsize': 8})








# MEG-specific
directory    = projects_directory + 'Resting_state\\'
data_directory = 'K:\\palva\\resting_state\\controls\__CFC\parc2009_csv\CFS\_MEG_ICA-ML_w\\'
CFM_filename = directory + '_settings\\CF_matrix_MEG.csv'
CFM          = np.genfromtxt(CFM_filename, delimiter=';')
cutoff       = 100
cutoff_LF    = 100
                   
LFs          = CFM[:,0][CFM[:,0]<cutoff_LF]         
freqs        = CFM[:,0]           
masks        = False
mask_type    = ''

parcellation = 'parc2009'
N_parc = 148
N_freq = len(freqs)

PS_metric  = 'wPLI'
sign_z_PS  = 2.42
CF_type    = 'CFS'
sign_z_CFS = 2.42
 
subject_sets   = ['S0001 set08_','S0001 set08c','S0006 set12_','S0006 set12c',
                  'S0008 set08_','S0008 set08c',
                  'S0113 set10_','S0113 set10c','S0116 set08_','S0116 set08c',
                  'S0156 set02_','S0156 set02c','S0166 set02_','S0166 set02c',
                  'S0172 set01_','S0172 set01c','S0173 set01_','S0173 set01c',
                  'S0174 set01_','S0174 set01c',]


subjects     = [s[:5] for s in subject_sets]                                  
subjects2    = list(set(subjects))
subjects2.sort()

N_LF     = len(LFs)
N_sets   = len(subject_sets)
N_sets   = len(subject_sets)
N_subj   = len(subjects2)

N_ratios = 3
ratios  = ['1:'+str(i+2) for i in range(N_ratios)]    
ratios2 = ['1-'+str(i+2) for i in range(N_ratios)] 


xlims_PS = [1,cutoff]
xlims_CF = [1,cutoff_LF]

HFs       = []     
for f in range(len(LFs)):
    x   = CFM[f,1:N_ratios+1]
    HFs.append(x)
N_rat_f = [len(HFs[i]) for i in range(len(LFs))]




CP_PLV    = [None for s in subjects]
fidelity  = [None for s in subjects]

support_dir = 'K:\\palva\\resting_state\\RS_CF_MEG\\_support_files\\_EOEC\\'


# get cross-patch PLV
for s,subj in enumerate(subjects):
    filename      = support_dir + subj + '\\Cross-Patch PLV ' + parcellation + '.csv'
    CP_PLV[s]     = np.genfromtxt(filename, delimiter=';')

# get patch fidelity
for s,subj in enumerate(subjects):
    filename    = support_dir + subj + '\\Patch Fidelity ' + parcellation + '.csv'
    fidelity[s] = np.genfromtxt(filename, delimiter=';')
   
     
    

# get networks for parcel
filename          = directory + '\\RS_CF_MEG\\_settings\\networks.csv'
network_indices   = np.array(np.genfromtxt(filename, delimiter=';'),'int')
networks          = [np.where(network_indices==i)[0] for i in range(7)]
network_names     = ['C','DM','DA','Lim','VA','SM','Vis']
N_networks        = len(network_names)

mean_CP_PLV   = np.mean(abs(np.array(CP_PLV)),0)
mean_fidelity = np.mean(np.array(fidelity),0)


fidelity_threshold = 0.08

if PS_metric == 'wPLI':                                                     
    CP_PLV_threshold = 1         
else:
    CP_PLV_threshold = 0.2143

number_channels = np.sum(mean_fidelity>fidelity_threshold)
fidelity_mask   = np.outer((mean_fidelity>fidelity_threshold),(mean_fidelity>fidelity_threshold))    # create a nice mask from 
CP_PLV_mask     = mean_CP_PLV < CP_PLV_threshold                         # CP-PLV is 1 on diagonal, so the mask diagonal will be 0 - good!
mask            = fidelity_mask*CP_PLV_mask

mask = mask[:N_parc,:N_parc]

edges_retained = np.sum(mask)/float(N_parc*N_parc-N_parc)
print edges_retained


np.fill_diagonal(mask,1)       


groups = ['eyes open','eyes closed']





'''    ##########        compute subject statistics          ##########     '''

### initialize lists
PS          =  [[   None for i in range(N_freq)] for j in range(N_sets)]       
CFC         =  [[[  None for i in range(len(HFs[j]))] for j in range(N_LF)] for k in range(N_sets)]


#### compute PS per-subject statistics ####   
for s,sset in enumerate(subject_sets): 
    for f,F in enumerate(freqs):
        
        F_str       = '{:.2f}'.format(F)
        file1       = directory + 'controls\\__PS\\' + parcellation + '_csv\\' + PS_metric + '\\_MEG_ICA-ML_w\\' + subjects[s] + '\\' + sset + ' f=' + F_str + '.csv'
        file2       = directory + 'controls\\__PS\\' + parcellation + '_csv\\' + PS_metric + '\\_MEG_ICA-ML_w\\' + subjects[s] + '\\' + sset + ' f=' + F_str + '_surr.csv'

        masked_data = mask*np.genfromtxt(file1, delimiter=';') 
        surr_data   = mask*np.genfromtxt(file2, delimiter=';') 
        stats       = cffun.K_stats_PS_2(masked_data,surr_data,sign_z_PS,PS_metric)
        PS[s][f]    = stats 

    print sset


#### compute CF per-subject statistics  ####

errors = []
for s,sset in enumerate(subject_sets):
    for lf,LF in enumerate(LFs):  
        for rat,HF in enumerate(HFs[lf]): 
          try: 
           if HF>100:   
            LF_str      = '{:.2f}'.format(LF)                
            HF_str      = '{:.2f}'.format(HF)   
            
            LF_PS       = PS[s][lf].data_sign     
            
            HF_idx      = np.where(freqs==HF)[0][0] 
            HF_PS     = PS[s][HF_idx].data_sign

            path        = data_directory 

            file0       = path + sset + ' LF=' + LF_str + ' HF=' + HF_str + '.csv'            
            file_surr   = path + sset + ' LF=' + LF_str + ' HF=' + HF_str + '_surr.csv'
            masked_data = np.genfromtxt(file0,  delimiter=';') * mask   
            surr_data   = np.genfromtxt(file_surr, delimiter=';') * mask           
          
            N_pot       = np.sum(mask)  
            stats         = cffun.K_stats_CFC_2(masked_data,surr_data,sign_z_CFS,LF_PS,HF_PS)

            CFC[s][lf][rat] = stats
          except:
              str1 = 'error for ' + ratios[rat] + ' ' + file0 
              print str1
              errors.append(str1)
    print sset


outfile1 = data_directory + 'pickle_dump\\' + PS_metric + ' ' + str(int(len(PS)/2)) + ' sets.dat'
outfile2 = data_directory + 'pickle_dump\\' + CF_type + ' with ' + PS_metric + ' corr ' + str(int(len(PS)/2)) + ' sets.dat'

pick.dump(PS,open(outfile1,'wb'))
pick.dump(CFC,open(outfile2,'wb'))
   




'''    ######     get group statistics and create plots for PS     ######   '''
    
# initialize lists
PLV_PS_ps        = np.zeros([2,N_sets/2,N_freq])                           # PS = "Phase Synch", ps = "per subject"
K_PS_ps          = np.zeros([2,N_sets/2,N_freq])
N_PS_ps          = np.zeros([2,N_sets/2,N_freq])
N_pot_PS_ps      = np.zeros([2,N_sets/2,N_freq])

for f,F in enumerate(freqs): 
   for s,ss in enumerate(subject_sets):            
       PLV_PS_ps[s%2,s/2,f]        = PS[s][f].mean_masked                            
       K_PS_ps[s%2,s/2,f]          = 100*PS[s][f].K


# select group and compute group stats
g = 0          
K_PS_stats         = [bst.CI_from_bootstrap(K_PS_ps[g])] # returns [mean, mean_boot, lower, upper] x freq x ratio
PLV_PS_stats       = [bst.CI_from_bootstrap(PLV_PS_ps[g])] # returns [mean, mean_boot, lower, upper] x freq x ratio
  

# PLOT PS with dists, indiv subjects

figsize     = [12.3,4.3]
rows        = 2
cols        = 2
dataL       = [PLV_PS_stats,(PLV_PS_ps[g]),K_PS_stats,(K_PS_ps[g])]
xlimA       = [[0,315] for i in range(6)]

if PS_metric == 'PLV':
    ylimA       = [[-0.005,0.22],[-0.005,0.22],[-5,100],[-5,100]]
else: 
    ylimA       = [[-0.005,0.15],[-0.005,0.15],[-5,100],[-5,100]]

titlesA     = ['' for i in range(6)]                          #'mean '+PS_metric,'mean '+PS_metric+' per distance','mean '+PS_metric+' per subject','mean K','mean K per distance','mean K per subject']
legendA     = [None, subjects[::2], None, None]
ylabA       = [PS_metric,'','K','','']
cmapA       = ['winter',my_cmap4,'winter',my_cmap4]
CI          = [0.3,None,0.3,None]
legend_posA = [None,'ur',None,None]
xlab        = [0,0,1,1,1]
Ryt         = [1,1,0,0,0]
plots.semi_log_plot_multi(figsize,rows,cols,dataL,freqs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,2,CI,xlab,Ryt)   

# save pdf
o67 = outfile1  = 'M:\\Resting_state_data\\controls\\New folder\\_results\\' + PS_metric + ' ' + groups[g] + '.png'
plots.semi_log_plot_multi([9,4],rows,cols,dataL,freqs,xlimA,ylabA,titlesA,cmapA,legendA,o67,legend_posA,ylimA,False,2,CI)   




'''       get group statistics and create plots for CF          '''

### init CFS arrays
K_CFC_ps               = np.full([2,N_sets/2,N_LF,N_ratios],np.nan)
PLV_CFC_ps             = np.full([2,N_sets/2,N_LF,N_ratios],np.nan)
PLV_CFC_ps_sig         = np.full([2,N_sets/2,N_LF,N_ratios],np.nan)
K_CFC_local_ps         = np.full([2,N_sets/2,N_LF,N_ratios],np.nan)
PLV_CFC_local_ps       = np.full([2,N_sets/2,N_LF,N_ratios],np.nan)    
K_CFC_ps_mod           = np.full([2,N_sets/2,N_LF,N_ratios],np.nan)
K_CFC_ps_mod_w         = np.full([2,N_sets/2,N_LF,N_ratios],np.nan)
PLV_CFC_ps_mod         = np.full([2,N_sets/2,N_LF,N_ratios],np.nan)
N_pot_mod_subj         = np.full([2,N_sets/2,N_LF,N_ratios],np.nan)

### get CFS values
for lf,LF in enumerate(LFs): 
   for hf,HF in enumerate(HFs[lf]):
       for s,ss in enumerate(subject_sets): 
           
         try:                 
           K_CFC_local_ps [s%2,s/2,lf,hf]        = 100*CFC[s][lf][hf].K_local                   
           K_CFC_ps       [s%2,s/2,lf,hf]        = 100*CFC[s][lf][hf].K
           PLV_CFC_ps     [s%2,s/2,lf,hf]        = CFC[s][lf][hf].mean_masked
           PLV_CFC_ps_sig [s%2,s/2,lf,hf]        = CFC[s][lf][hf].mean_sign        
           K_CFC_ps_mod   [s%2,s/2,lf,hf]        = 100*CFC[s][lf][hf].K_mod
           K_CFC_ps_mod_w [s%2,s/2,lf,hf]        = 100*CFC[s][lf][hf].K_mod * CFC[s][lf][hf].N_pot_mod    #weighting by subject's N_pot_mod
           PLV_CFC_ps_mod [s%2,s/2,lf,hf]        = CFC[s][lf][hf].mean_mod                    
           N_pot_mod_subj [s%2,s/2,lf,hf]        = CFC[s][lf][hf].N_pot_mod
         except:
             pass
           
         try:
               PLV_CFC_local_ps[s%2,s/2,lf,hf]  = CFC[s][lf][hf].mean_local
         except: 
               PLV_CFC_local_ps[s%2,s/2,lf,hf]  = np.nan


### select group and compute group stats
g=01
N_boot=1000

PLV_CFC_stats       = [np.array(bst.CI_from_bootstrap(PLV_CFC_ps[g,:,:,i]))       for i in range(N_ratios)] # returns [mean, mean_boot, lower, upper] x freq x ratio
K_CFC_stats         = [np.array(bst.CI_from_bootstrap(K_CFC_ps[g,:,:,i])) -1      for i in range(N_ratios)] 
K_CFC_stats_mod     = [np.array(bst.CI_from_bootstrap(K_CFC_ps_mod[g,:,:,i]))-1   for i in range(N_ratios)] 
PLV_CFC_local_stats = [np.array(bst.CI_from_bootstrap(PLV_CFC_local_ps[g,:,:,i])) for i in range(N_ratios)] 
K_CFC_local_stats   = [np.array(bst.CI_from_bootstrap(K_CFC_local_ps[g,:,:,i]))-1 for i in range(N_ratios)]


### plot CFC results with individual results
figsize1 = [16,6]  
figsize2 = [6.3,2.3]   
rows    = 2
cols    = 3
dataL   = [PLV_CFC_stats[0:3],    K_CFC_stats  [0:3], K_CFC_stats_mod[0:3],
           (PLV_CFC_ps[g,:,:,0]),(K_CFC_ps[g,:,:,0]),(K_CFC_ps_mod[g,:,:,0]) ] 
xlimA   = [xlims_CF for i in range(8)]
titlesA = ['' for i in range(8)]      
ylimA   = [[-0.002,0.05], [-2, 17], [-2, 17],[-0.002,0.05], [-2,17], [-2,17]]  
legendA = [ratios[:3],ratios[:3],ratios[:3],
           subjects[::2],subjects[::2],subjects[::2]]    
ylabA   = [PS_metric,'K [%]','K [%]',
           PS_metric,'PLV','K [%]','K [%]']
cmapA   = ['brg','brg','brg',my_cmap4,my_cmap4,my_cmap4]
legend_posA = ['ur',None,None,'ur',None,None]
CI      = [0.2,0.2,0.2,None,None,None]
xlabA   = [0,0,0,1,1,1]
Ryt     = [1,1,1,1,1,1]
plots.semi_log_plot_multi(figsize1,rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,None,legend_posA,ylimA,True,2,CI,xlabA,Ryt,fontsize=10)   


# export
outfile1  = directory + 'RS_CF_MEG\\_results\\CFS\\CFS ' + groups[g] + ', with subject values.pdf'
plots.semi_log_plot_multi([7.3,3.3],rows,cols,dataL,LFs,xlimA,ylabA,titlesA,cmapA,legendA,outfile1,legend_posA,ylimA,True,2,CI,xlabA,Ryt,fontsize=8)   





o41a = [directory + 'RS_CF_MEG\\_results\\_plot_data_new\\' + CF_type + '\\MEG\\MEG ' + CF_type  + ' ' + groups[g] + ' K ' + r + '.csv' for r in ratios2 ]
o41b = [directory + 'RS_CF_MEG\\_results\\_plot_data_new\\' + CF_type + '\\MEG\\MEG ' + CF_type  + ' ' + groups[g] +' K_mod using ' + PS_metric + ' ' + r + '.csv' for r in ratios2 ]
o41c = [directory + 'RS_CF_MEG\\_results\\_plot_data_new\\' + CF_type + '\\MEG\\MEG ' + CF_type  + ' ' + groups[g] +' GS ' + r + '.csv' for r in ratios2 ]
for r in range(3):
    np.savetxt(o41a[r],K_CFC_stats    [r][:3],delimiter=';')
    np.savetxt(o41b[r],K_CFC_stats_mod[r][:3],delimiter=';')
    np.savetxt(o41c[r],PLV_CFC_stats  [r][:3],delimiter=';')





















