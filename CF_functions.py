
"""
Intensity Plots for CF
"""


import numpy as np
from numpy import genfromtxt
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
from numpy.ma import masked_invalid
from matplotlib.colors import LogNorm
#import mne.connectivity.spectral as spec
import scipy
from scipy import stats as stat
from scipy import sparse
import statsmodels.sandbox.stats.multicomp as multicomp
import time

def process_K(LF_file, datafolder, datafile, N_LF, N_HF,zmax=0.04,                      
              CM=plt.cm.YlOrRd,figsize=[18,8]):
    
    LF_matrix = genfromtxt(LF_file, delimiter=';')
    freqs = list(LF_matrix[1:,1])
    freqs = [round(f,3) for f in freqs]    
    LF_matrix = LF_matrix[1:,2:]
    LF_matrix[np.isnan(LF_matrix)]=0
    
    N_ratio = LF_matrix.shape[1]
    ratios=['1:'+str(i+2) for i in range(N_ratio)]
    
    N_freq  = [sum(LF_matrix[:,i]!=0) for i in range(N_ratio)]
    N_lowfreqs = N_LF
    N_highfreqs = N_HF
    lowfreqs=freqs[:N_lowfreqs]
    highfreqs=freqs[-N_highfreqs:]
    shift=[len(freqs)-N_freq[z] for z in range(N_ratio)]
    
    os.chdir(datafolder)
    data=np.transpose(genfromtxt(datafile, delimiter=';'))
    
    intlist=[Interaction() for x in range(N_ratio*N_lowfreqs)]
    data2=np.zeros([max(N_freq)+2,N_ratio])       # LF x ratio
    data2.fill('NaN')
    data3=np.zeros([len(freqs),len(freqs)])     # LF x HF
    data3.fill('NaN')
    
    i=0
    for r in range(N_ratio):
        for f in range(N_freq[r]):
            
            value=data[f,r]            
            LF = LF_matrix[f,r]
            HF = freqs[shift[r]+f]
            LFind = lowfreqs.index(LF)
            HFind = highfreqs.index(HF)
            data2[LFind,r]=value
            data3[HFind,LFind]=value
            
            if data[f,r]!=0:
                print
                intlist[i].r_ind = r+2
                intlist[i].ratio = ratios[r]
                intlist[i].LF    = LF
                intlist[i].HF    = HF
                intlist[i].value = value
                intlist[i].LFind = LFind
                intlist[i].HFind = HFind                
                i=i+1    
    
    intlist=intlist[:i]  

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 20)
    
    fig = plt.plot_heatmap(fig,gs[:,:5],data2,'Ratio','Low freq. [Hz]',
                 ratios,[round(f,1) for f in lowfreqs],zmax=zmax,cbar=0)   
    fig = plt.plot_heatmap(fig,gs[:,7:],data3,'Low freq. [Hz]','High freq. [Hz]',
                 [round(f,1) for f in lowfreqs],[round(f,1) for f in highfreqs],
                  zmax=zmax)

    fileout = datafile[:-4]+'.png'
    fig.savefig(fileout)







def K_stats_PS(masked_data,surr_data,sign_z,metric='PLV'):                 # metric either 'PLV' or 'iPLV', or 'wPLI' 

    stats  = stats_PS(0)
    data   = Bunch()
    N_CH   = len(masked_data)      
    
    if metric=='iPLV':                                                             # else: iPLV
        data.data_masked          = abs(np.imag(masked_data))
        data.data_surr_masked     = abs(np.imag(surr_data))
    else: 
        data.data_masked          = abs(masked_data)  
        data.data_surr_masked     = abs(surr_data)  
    
    np.fill_diagonal(data.data_masked,0)                                    # necessary????
    np.fill_diagonal(data.data_surr_masked,0)  
    
    N_pot      = np.nansum(data.data_masked>0)
    N_pot_surr = np.nansum(data.data_surr_masked>0)
    
    stats.mean_masked       = np.nansum(data.data_masked)/N_pot
    stats.mean_surr_masked  = np.nansum(data.data_surr_masked)/N_pot_surr
    
    if metric=='wPLI':
        stats.std_surr_masked = np.nanstd(data.data_surr_masked)
        stats.z_score         = (data.data_masked - stats.mean_surr_masked)/stats.std_surr_masked
        data.data_sign        = data.data_masked*(stats.z_score>sign_z).astype(int)
        
    else:       
        stats.threshold         = sign_z*stats.mean_surr_masked 
        data.data_sign          = data.data_masked*(data.data_masked > stats.threshold).astype(int) 
        
    stats.data_masked       = np.float32(data.data_masked)                    # needed for CFC local test!
    stats.data_sign         = np.float32(data.data_sign)                    # needed for CFC local test!
    stats.mean_sign         = np.nansum(data.data_sign)/N_pot
    stats.N                 = len(np.transpose(np.where(data.data_sign > 0)))
    stats.K                 = stats.N/float(N_pot)
    stats.N_pot             = N_pot
    stats.N_pot_surr        = N_pot_surr
    stats.N_CH              = N_CH
    stats.degree            = np.nansum(stats.data_sign>0,0) + np.nansum(stats.data_sign>0,1)
    stats.strength          = np.nansum(stats.data_sign,0) + np.nansum(stats.data_sign,1)
    np.fill_diagonal(data.data_masked,1)
    np.fill_diagonal(data.data_sign,1)
    data.graph              = nx.to_networkx_graph(data.data_masked,create_using=nx.DiGraph())
    data.graph_sign         = nx.to_networkx_graph(data.data_sign,create_using=nx.DiGraph())    
    stats.degree_centrality = nx.degree_centrality(data.graph_sign)
    stats.betw_centrality   = nx.betweenness_centrality(data.graph_sign)

    try:
        stats.ev_centrality   = nx.eigenvector_centrality(data.graph_sign)
    except:
        stats.ev_centrality   = np.full(N_CH,np.nan)

    return stats                                                   

  



def K_stats_CFC(masked_data,surr_data,sign_z,LF_PS=None,HF_PS=None):    # for a single ratio. no masks here!

    stats  = stats_CFC(0)
    data   = Bunch()
    N_CH   = len(masked_data)  
     
    data.data_masked      = abs(masked_data)                                    # mask must have diagonal 0
    data.data_surr_masked = abs(surr_data)     
   
    data.data_local         = abs(masked_data.diagonal())                                  # local CFC is on the diagonal
    data.data_surr_local    = abs(surr_data.diagonal())
    
    np.fill_diagonal(data.data_masked,0)
    np.fill_diagonal(data.data_surr_masked,0)  
    
    N_pot      = np.nansum(data.data_masked>0)
    N_pot_surr = np.nansum(data.data_surr_masked>0)
    
    stats.mean_masked       = np.sum(data.data_masked)/N_pot
    stats.mean_surr_masked  = np.sum(data.data_surr_masked)/N_pot_surr
    stats.mean_local        = np.sum(data.data_local)/N_CH
    stats.mean_surr_local   = np.sum(data.data_surr_local)/N_CH    
    stats.threshold         = sign_z*stats.mean_surr_masked            
    stats.threshold_local   = sign_z*stats.mean_surr_local     
    
    data.data_sign          = data.data_masked*(data.data_masked > stats.threshold)               #correct?
    data.data_sign_local    = data.data_local*(data.data_local > stats.threshold_local)               #correct? 
    stats.mean_sign         = np.sum(data.data_sign)/N_pot
    stats.N                 = len(np.transpose(np.where(data.data_masked  > stats.threshold)))
    stats.N_local           = len(np.transpose(np.where(data.data_local   > stats.threshold_local)))    
    stats.K                 = stats.N/float(N_pot)
    stats.K_local           = stats.N_local/float(N_CH)
    stats.N_pot             = N_pot
    stats.N_CH              = N_CH
    stats.degree_LF         = np.nansum(data.data_sign>0,0)
    stats.degree_HF         = np.nansum(data.data_sign>0,1)
    stats.strength_LF       = np.nansum(data.data_sign,0)
    stats.strength_HF       = np.nansum(data.data_sign,1)
    
    for i in range(N_CH):
        data.data_sign[i,i] = data.data_sign_local[i]
    data.graph_sign         = nx.to_networkx_graph(data.data_sign,create_using=nx.DiGraph())

    stats.degree_centrality = nx.degree_centrality(data.graph_sign)
    stats.betw_centrality   = nx.betweenness_centrality(data.graph_sign)
    
    for i in range(N_CH):
        data.data_sign[i,i] = 0
           
    try:
        stats.ev_centrality     = nx.eigenvector_centrality(data.graph_sign)
    except:
        stats.ev_centrality     = np.full(N_CH,np.nan)
    
    if LF_PS is not None:                                                 # if PS sign matrix is given, do test for triangles 
        HF_local               = np.array([data.data_sign_local]*N_CH)
        LF_local               = np.transpose(np.array([data.data_sign_local]*N_CH))         
        tri_mask               = ((LF_PS==0) + (HF_local==0)) * ((HF_PS==0) + (LF_local==0))
        control_mask           = tri_mask      * (data.data_masked>0)            # last term added as hacky fix
        excl_mask              = (tri_mask==0) * (data.data_masked>0) 
        data.data_sign_mod     = data.data_sign * control_mask
        data.data_sign_excl    = data.data_sign * excl_mask
        stats.mean_mod         = np.float64(np.sum(data.data_sign_mod))/np.sum(data.data_sign_mod>0)
        stats.mean_excl        = np.float64(np.sum(data.data_sign_mod))/np.sum(data.data_sign_mod>0)
        
        stats.N_mod            = np.sum(data.data_sign_mod >0)
        stats.N_pot_mod        = np.sum(control_mask>0)
        stats.K_mod            = stats.N_mod/float(stats.N_pot_mod)
        stats.degree_LF_mod    = np.nansum(data.data_sign_mod>0,0)
        stats.degree_HF_mod    = np.nansum(data.data_sign_mod>0,1)  
        stats.strength_LF_mod  = np.nansum(data.data_sign_mod,0)
        stats.strength_HF_mod  = np.nansum(data.data_sign_mod,1) 
        
        stats.N_excl           = np.sum(data.data_sign_excl >0)
        stats.N_pot_excl       = np.sum(excl_mask>0)
        stats.K_excl           = stats.N_excl/float(stats.N_pot_excl)
        stats.degree_LF_excl   = np.nansum(data.data_sign_excl>0,0)
        stats.degree_HF_excl   = np.nansum(data.data_sign_excl>0,1)
        stats.strength_LF_excl = np.nansum(data.data_sign_excl,0)
        stats.strength_HF_excl = np.nansum(data.data_sign_excl,1)
        
        for i in range(N_CH):
            data.data_sign_mod[i,i] = data.data_sign_local[i]
       
        data.graph_mod         = nx.to_networkx_graph(data.data_sign_mod,create_using=nx.DiGraph())

        stats.degree_centrality_mod = nx.degree_centrality(data.graph_mod)
        stats.betw_centrality_mod   = nx.betweenness_centrality(data.graph_mod)
        try:
            stats.ev_centrality_mod     = nx.eigenvector_centrality(data.graph_mod)
        except:
            stats.ev_centrality_mod     = np.full(N_CH,np.nan)
                       
        stats.data_sign_mod   = np.float32(data.data_sign_mod)
    
    for i in range(N_CH):
        data.data_sign[i,i]   = data.data_sign_local[i] 
        data.data_masked[i,i] = data.data_local[i] 
            
    stats.data_sign   = np.float32(data.data_sign)
    stats.data_masked = np.float32(data.data_masked)  
    
    return stats
                


def combined_matrix(freqs,LFs,HFs,subject,ratio,freq_idx,PS,CFC,f_HF=1,f_CF=1,binary=0):
    

    LF       = freqs[freq_idx]
    LF_ind   = (list(LFs)).index(LF)
    HF       = HFs[LF_ind][ratio]
    HF_in_freqs_ind  = (list(freqs)).index(HF)   
    
    
    PS_low  = PS[subject][freq_idx]
    PS_high = PS[subject][HF_in_freqs_ind]
    
    N_CH    = int(PS_low.N_CH)
       
    matrix = np.zeros([N_CH*2,N_CH*2])
    
    matrix[0:N_CH,0:N_CH] = PS_low.data_sign
    
    matrix[N_CH:N_CH*2,N_CH:N_CH*2] = PS_high.data_sign*f_HF
    
    # one of these need flipping - which one?
    
    matrix[N_CH:N_CH*2,0:N_CH] = CFC[subject][LF_ind][ratio].data_sign*f_CF
    matrix[0:N_CH,N_CH:N_CH*2] = np.transpose(CFC[subject][LF_ind][ratio].data_sign*f_CF)
    
    print('LF = ' + str(LF) + ', HF = ' + str(HF))
    
    if binary:
        matrix = (matrix >0)
    
    plt.imshow(matrix,origin='bottom')


    return matrix




def edge_counting(directory,subjects,ch_per_subject,freqs,LFs,HFs,PS,CFC,CFC_dist,CFC_layer,parc,channel_layers):
    N = Bunch()    
    edges = Bunch()
    if parc == 'parc2009':
        N.parcel=149
    elif parc =='parc2018yeo7_200':
        N.parcel = 201
    else:
        print 'not a valid parcellation!'
        return
    N.LF    = len(LFs)
    N.freq  = len(freqs) 
    N.subj  = len(subjects)
    N.ratios = len(CFC[0][0])
    N.dist_bins = len(CFC_dist[0][0][0])
    N.layer_int = len(CFC_layer[0][0][0])
    N.layer     = 3
    N.network   = 7
    N.chan_per_parc_subj       = np.zeros([N.subj,N.parcel],'int')                                    # +1 to have a bin for masked channels     
    edges.N_pot_pp_subj      = np.zeros([N.subj,N.parcel,N.parcel],'int') 
    edges.N_pot_ppd_subj     = np.zeros([N.dist_bins,N.subj,N.parcel,N.parcel],'int') 
    edges.N_pot_ppl_subj     = np.zeros([N.layer_int,N.subj,N.parcel,N.parcel],'int')     
    edges.metric_PS_pp       = np.zeros([N.freq,N.parcel,N.parcel])  
    edges.PLV_CF_pp_all      = [[[[[] for b in range(N.parcel)] for i in range(N.parcel)] for j in range(N.ratios)] for k in range(N.LF)]
    edges.PLV_CF_pp_all_TP   = [[[[[] for b in range(N.parcel)] for i in range(N.parcel)] for j in range(N.ratios)] for k in range(N.LF)]
    edges.N_sign_PS_pp       = np.zeros([N.freq,N.parcel,N.parcel],'int')
    edges.N_sign_CF_pp       = np.zeros([N.LF,N.ratios,N.parcel,N.parcel],'int')   
    edges.N_sign_CF_pp_mod   = np.zeros([N.LF,N.ratios,N.parcel,N.parcel],'int') 
    edges.N_sign_local_pp    = np.zeros([N.LF,N.ratios,N.parcel])
    edges.N_sign_local_pl    = np.zeros([N.LF,N.ratios,N.layer])
    edges.PLV_local_pp       = np.zeros([N.LF,N.ratios,N.parcel])
    edges.PLV_local_pl       = np.zeros([N.LF,N.ratios,N.layer])

    diag_zero_mask_parc = np.ones([N.parcel,N.parcel])
    np.fill_diagonal(diag_zero_mask_parc,0)    
    diag_zero_mask = [None for i in subjects]
    
    morph_filename0   = directory + '_settings\\morph 256 to 200.csv'                                       # get the morphing from 254 parcels to 200 parcels
    morph_no_dummies  = np.genfromtxt(morph_filename0, delimiter=';')[:,1].astype('int')    
    
    for s,subject in enumerate(subjects): 
        diag_zero_mask[s] = np.ones([ch_per_subject[s],ch_per_subject[s]])
        np.fill_diagonal(diag_zero_mask[s],0)
        
        mask_filename    = directory + '_settings\\masks new\\CC all, 0.2\\' + subject + '.csv'
        mask             = np.genfromtxt(mask_filename, delimiter=';')   
        ch_mask          = np.sum(mask,0)>0    
        morph_filename1  = directory + '_settings\\morphing OPs 2018-05\\' + subject + ' ' + parc + ' val1.csv'
        morphing_targ_T  = np.genfromtxt(morph_filename1, delimiter=';')[:,0].astype('int')
        
        if parc   == 'parc2018yeo7_200':
            morphing_targ    = [morph_no_dummies[i] for i in morphing_targ_T]  ### multipl. terms to remove unassigned + medial wall
        else:
            morphing_targ    = morphing_targ_T   
            
        morphing_targ2   = np.array(N.parcel+1)                                                                  # for counting
        morphing_targ2   = [morphing_targ[i]+1 if (ch_mask[i]==1) else 0 for i in range(len(mask))]     
        N.chan_per_parc_subj[s]  =  np.bincount(morphing_targ2,minlength=N.parcel+1)[1:]
        
        MM  = np.zeros([N.parcel,ch_per_subject[s]],'int')                    #### morphing matrix (int)
        MMF = np.zeros([N.parcel,ch_per_subject[s]])                          #### morphing matrix (float)
        
        for i in range(ch_per_subject[s]):
            MM[morphing_targ[i],i] = 1
            MMF[morphing_targ[i],i] = 1
                
        ME  = CFC[s][0][0].data_masked * diag_zero_mask[s]                         #### count possible edges per parcel pair ####
        MT  = np.matmul(MM,(ME>0))
        edges.N_pot_pp_subj[s] += np.transpose(np.matmul(MM,np.transpose(MT)))
        
        for d in range(N.dist_bins):                                               #### count possible edges per parcel pair in dist. bin ####
            ME  = CFC_dist[s][0][0][d].data_masked * diag_zero_mask[s]             
            MT  = np.matmul(MM,(ME>0))
            edges.N_pot_ppd_subj[d,s] += np.transpose(np.matmul(MM,np.transpose(MT)))    
           
        for l in range(N.layer_int):                                               #### count possible edges per parcel pair in layer interaction ####
            try:
                ME  = CFC_layer[s][0][0][l].data_masked * diag_zero_mask[s]            
                MT  = np.matmul(MM,(ME>0))
                edges.N_pot_ppl_subj[l,s] += np.transpose(np.matmul(MM,np.transpose(MT)))
            except: 
                print('empty layer in subject ' + subject)   
                   
        for f,freq in enumerate(freqs):                                            #### PS: get PLV & sign. edges per parcel pair ####
            data_masked    = PS[s][f].data_masked * diag_zero_mask[s]
            MT             = np.matmul(MM,data_masked)
            edges.metric_PS_pp[f,:,:] += np.transpose(np.matmul(MM,np.transpose(MT)))
                
            data_sign    = PS[s][f].data_sign * diag_zero_mask[s]
            MT           = np.matmul(MM,(data_sign>0))
            edges.N_sign_PS_pp[f,:,:] += np.transpose(np.matmul(MM,np.transpose(MT)))        
                    
        for lf,LF in enumerate(LFs):                                               #### get sign CFC edges per parcel pair ####
            for hf,HF in enumerate(HFs[lf]):   
                data_local      = np.diagonal(CFC[s][lf][hf].data_masked)
                data_local_sign = np.diagonal(CFC[s][lf][hf].data_sign)
                for ch in range(len(data_local)):
                    edges.N_sign_local_pp  [lf][hf][morphing_targ[ch]] += data_local_sign[ch]>0
                    edges.PLV_local_pp[lf][hf][morphing_targ[ch]]      += data_local[ch]

                    if channel_layers[s][ch] >=0:
                        edges.N_sign_local_pl[lf][hf][channel_layers[s][ch]] += data_local_sign[ch]>0
                        edges.PLV_local_pl   [lf][hf][channel_layers[s][ch]] += data_local[ch]

                
                data_masked = CFC[s][lf][hf].data_masked * diag_zero_mask[s]
                SM     = sparse.find(data_masked)
                for cc in range(len(SM[0])): 
                    p1  = morphing_targ[SM[0][cc]]
                    p2  = morphing_targ[SM[1][cc]]
                    val = SM[2][cc]
                    edges.PLV_CF_pp_all   [lf][hf][p1][p2].append(val)            
                    edges.PLV_CF_pp_all_TP[lf][hf][p2][p1].append(val)
        
                data_sign  = CFC[s][lf][hf].data_sign * diag_zero_mask[s]
                MT         = np.matmul(MM,(data_sign>0))
                edges.N_sign_CF_pp[lf,hf,:,:] += np.transpose(np.matmul(MM,np.transpose(MT))) 
                
                data_sign_mod  = CFC[s][lf][hf].data_sign_mod * diag_zero_mask[s]
                MT             = np.matmul(MM,(data_sign_mod>0))
                edges.N_sign_CF_pp_mod[lf,hf,:,:] += np.transpose(np.matmul(MM,np.transpose(MT)))    
          
#                for d in range(N.dist_bins):
#                    dataMD = CFC_dist[s][lf][hf][d].data_masked * diag_zero_mask[s]
#                    MT = np.matmul(MM,dataMD)  
#                    #edges.PLV_CF_ppd[lf,hf,d,:,:] += np.transpose(np.matmul(MM,np.transpose(MT))) /edges.N_pot_ppd_subj[d,s]
#                    
#                    dataSD = CFC_dist[s][lf][hf][d].data_sign * diag_zero_mask[s]
#                    MT = np.matmul(MM,(dataSD>0))  
#                    edges.N_sign_CF_ppd[lf,hf,d,:,:] += np.transpose(np.matmul(MM,np.transpose(MT)))
#                try:      
#                    for l in range(N.layer_int):
#                        dataML = CFC_layer[s][lf][hf][l].data_masked * diag_zero_mask[s]                  
#                        MT = np.matmul(MM,dataML>0)  
#                        edges.PLV_CF_ppl_subj[lf,hf,l,s,:,:] = np.transpose(np.matmul(MM,np.transpose(MT))) /edges.N_pot_ppl_subj[l,s]
#                        
#                        dataSL = CFC_layer[s][lf][hf][l].data_sign * diag_zero_mask[s]                  
#                        MT = np.matmul(MM,(dataSL>0))  
#                        edges.N_sign_CF_ppl[lf,hf,l,:,:] += np.transpose(np.matmul(MM,np.transpose(MT)))                                                 
#                except:
                   # pass 
        
        print(time.strftime("%Y-%m-%d %H:%M") + '          ' + subject)
    
    if parc =='parc2018yeo7_200':
        N.parcel = 200
    if parc == 'parc2009':
        N.parcel=148
        
    edges.N_pot_pp_subj      = edges.N_pot_pp_subj[:,:N.parcel,:N.parcel] 
    edges.N_pot_ppd_subj     = edges.N_pot_ppd_subj[:,:,:N.parcel,:N.parcel]
    edges.N_pot_ppl_subj     = edges.N_pot_ppl_subj[:,:,:N.parcel,:N.parcel]    
    edges.metric_PS_pp       = edges.metric_PS_pp[:,:N.parcel,:N.parcel]   
    edges.N_sign_PS_pp       = edges.N_sign_PS_pp[:,:N.parcel,:N.parcel]
    edges.N_sign_CF_pp       = edges.N_sign_CF_pp[:,:,:N.parcel,:N.parcel]        
    edges.N_sign_CF_pp_mod   = edges.N_sign_CF_pp_mod[:,:,:N.parcel,:N.parcel]
    edges.PLV_local_pp       = edges.PLV_local_pp[:,:,:N.parcel]
    edges.N_sign_local_pp    = edges.N_sign_local_pp[:,:,:N.parcel]
    N.chan_per_parc_subj     = [a[:N.parcel] for a in N.chan_per_parc_subj]    
    
    edges.N_pot_pp  = np.sum(edges.N_pot_pp_subj,0)
    edges.N_pot_ppd = np.sum(edges.N_pot_ppd_subj,0)
    edges.N_pot_ppl = np.sum(edges.N_pot_ppl_subj,0)

    return N, edges    
       
 
### for SEEG ###
def low_to_high_analysis(edges,N,LFs,HFs,alpha,N_perm,parc,directory,networks,N_rat=2):                   
  
    if parc == 'parc2009':
        N.parcel = 148
        file_networks   = 'M:\\SEEG_Morlet\\_RAW_line_filtered\\_settings\\networks parc2009.csv'
    if parc =='parc2018yeo7_200':
        N.parcel = 200
        file_networks   = 'M:\\SEEG_Morlet\\_RAW_line_filtered\\_settings\\networks parc2018yeo7_200.csv'
    network_indices = np.array(np.genfromtxt(file_networks, delimiter=';'),'int')
    
    lh = Bunch()
    lh.N_LTH_wilc          = np.zeros([N.LF,N_rat])
    lh.N_pot_LTH_wilc      = np.zeros([N.LF,N_rat])
    lh.diffs               = np.zeros([N.LF,N_rat,N.parcel,N.parcel])  
    lh.N_el_list  = [[[]for i in range(N_rat)]for j in range(N.LF)]
    lh.N_el_array = np.zeros([N.LF,N.ratios,10000]) 
    
    for lf,LF in enumerate(LFs):                                               #### get sign CFC edges per parcel pair ####
        for hf,HF in enumerate(HFs[lf][:N_rat]):
          if hf>1:  
            for p1 in range(N.parcel):
                for p2 in range(N.parcel):  
                    l1        = len(edges.PLV_CF_pp_all[0][0][p1][p2])
                    l2        = len(edges.PLV_CF_pp_all_TP[0][0][p1][p2])              
                    if (l1>0):
                        lh.N_el_array[lf,hf,l1+l2] +=1  
                        lh.N_el_list[lf][hf].append(l1+l2)
                        

    
    #### check differences by Wilcoxon method    
    for lf,LF in enumerate(LFs):                                               #### get sign CFC edges per parcel pair ####
        for hf,HF in enumerate(HFs[lf][:N_rat]):
          if hf>1:  
            for p1 in range(N.parcel):
                for p2 in range(N.parcel):
                    a1 = edges.PLV_CF_pp_all   [lf][hf][p1][p2]
                    a2 = edges.PLV_CF_pp_all_TP[lf][hf][p1][p2]
                    lh.diffs[lf][hf][p1][p2] = np.sum(a1) - np.sum(a2)
                    if ((len(a1) == len(a2)) and (len(a1)+len(a2) >0)):
                        lh.N_pot_LTH_wilc[lf,hf]+=1
                        s,pwil = stat.wilcoxon(edges.PLV_CF_pp_all[lf][hf][p1][p2],edges.PLV_CF_pp_all_TP[lf][hf][p1][p2])
                        if pwil<alpha:
                            lh.N_LTH_wilc[lf,hf] += 1
        print lf           
    lh.K_LTH_wilc = lh.N_LTH_wilc/lh.N_pot_LTH_wilc 
    
    #### check differences using the permutation method    
    lh.N_pot_LH       = np.zeros([N.LF,N_rat])
    lh.N_pot_LH_pn    = np.zeros([N.LF,N_rat,N.network,N.network])
    lh.N_pot_LH_pp    = np.zeros([N.LF,N_rat,N.parcel,N.parcel])
    lh.N_el_pot_LH    = np.zeros([N.LF,N_rat])
    lh.N_el_pot_LH_pn = np.zeros([N.LF,N_rat,N.network,N.network])
    lh.N_el_pot_LH_pp = np.zeros([N.LF,N_rat,N.parcel,N.parcel])
    lh.V_LH_sig_pp    = np.zeros([N.LF,N_rat,N.parcel,N.parcel])
    lh.V_LH_pp        = np.zeros([N.LF,N_rat,N.parcel,N.parcel])
    lh.N_LH_sig_pn    = np.zeros([N.LF,N_rat,N.network,N.network])
    lh.V_LH_sig_pn    = np.zeros([N.LF,N_rat,N.network,N.network]) 
    lh.V_LH_pn        = np.zeros([N.LF,N_rat,N.network,N.network]) 
    
    pctile         = 1 - alpha
    lh.L_sig       = np.zeros([N.LF,N.ratios,10000])  
    lh.L_sig_list  = [[[]for i in range(N_rat)]for j in range(N.LF)]            #### significant edges per length    
    
    for lf,LF in enumerate(LFs):                                               #### get sign CFC edges per parcel pair ####
        for hf,HF in enumerate(HFs[lf][:N_rat]):
          if hf>1:  
            for p1 in range(N.parcel):
                for p2 in range(N.parcel):                
                    diff_perm = np.zeros(N_perm)                 
                    l1        = len(edges.PLV_CF_pp_all[lf][hf][p1][p2])
                    l2        = len(edges.PLV_CF_pp_all_TP[lf][hf][p1][p2])
                    concatA   = np.array(edges.PLV_CF_pp_all[lf][hf][p1][p2] + edges.PLV_CF_pp_all_TP[lf][hf][p1][p2])
                    indices   = range(l1+l2)  
                    diff      = np.sum(edges.PLV_CF_pp_all[lf][hf][p1][p2]) - np.sum(edges.PLV_CF_pp_all_TP[lf][hf][p1][p2])
                    if ((l1>0 and l2>0) and (l1==l2)):
                        lh.N_pot_LH   [lf,hf] += 1
                        lh.N_pot_LH_pp[lf,hf,p1,p2] += 1                                       # are there electrodes connecting this parcel pair? 0 or 1
                        lh.N_pot_LH_pn[lf,hf,network_indices[p1],network_indices[p2]] += 1     # number of parcel pairs projecting to this netw. pair 
                        lh.N_el_pot_LH   [lf,hf] += l1
                        lh.N_el_pot_LH_pp[lf,hf,p1,p2] += l1                                      # number of el pairs projecting to this parcel pair
                        lh.N_el_pot_LH_pn[lf,hf,network_indices[p1],network_indices[p2]] += l1    # number of el pairs projecting to this netw. pair
                        for n in range(N_perm): 
                            ind_rand     = np.random.permutation(indices)
                            diff_perm[n] = np.sum(concatA[ind_rand[:l1]]) - np.sum(concatA[ind_rand[l1:]])  
                        pctile_p = np.sum(diff > diff_perm)/np.float(N_perm)
                        
                        lh.V_LH_pp[lf,hf,p1,p2] = diff                                      # value of  difference in LH 1->2 and LH 2->1 at this parcel pair
                        lh.V_LH_pn[lf,hf,network_indices[p1],network_indices[p2]] += diff   # sum of  differences of all parcels pairs projecting to this network pair
    
                        if pctile_p > pctile: 
                            lh.L_sig_list[lf][hf].append(l1+l2)
                            lh.L_sig     [lf,hf,l1+l2] +=1 
                            lh.V_LH_sig_pp[lf,hf,p1,p2] = diff                          # value of sign. difference in LH 1->2 and LH 2->1 at this parcel pair
                            lh.N_LH_sig_pn[lf,hf,network_indices[p1],network_indices[p2]] += 1      # number sign. different parcels pairs projecting to this network pair
                            lh.V_LH_sig_pn[lf,hf,network_indices[p1],network_indices[p2]] += diff   # sum of sign. differences of all parcels pairs projecting to this network pair
            print(time.strftime("%Y-%m-%d %H:%M") + ' lf= '+ str(lf) + ', hf= ' + str(hf))      
    return N, lh   


### for SEEG ###
def low_to_high_threshold(lh,N,N_min,networks):                         
    
    # threshold by N_min:    
    lh.V_LH_sig_pp = lh.V_LH_sig_pp * (lh.N_el_pot_LH_pp>N_min)  
    
    lh.N_LH_sig    = np.nansum((lh.V_LH_sig_pp>0),(2,3))              ### 
    lh.N_pot_LH_am = lh.N_el_array*(lh.N_el_array>N_min)           ### ????
    lh.N_pot_LH    = np.sum(lh.N_pot_LH_pp*(lh.N_el_pot_LH_pp>N_min)  ,(2,3))
    lh.K_LH        = lh.N_LH_sig/lh.N_pot_LH    
    lh.N_LH_sig    = np.sum(lh.V_LH_sig_pp>0,(2,3))
    lh.K_LH_pn     = lh.N_LH_sig_pn/lh.N_pot_LH_pn     
    lh.out_degree = np.nansum(lh.V_LH_sig_pp>0,2)/np.nansum(lh.N_el_pot_LH_pp,2)
    lh.in_degree  = np.nansum(lh.V_LH_sig_pp>0,3)/np.nansum(lh.N_el_pot_LH_pp,3)

   
    lh.out_strength      = np.nanmean(lh.V_LH_pp/lh.N_el_pot_LH_pp,2)        ##### ?????
    lh.in_strength       = np.nanmean(lh.V_LH_pp/lh.N_el_pot_LH_pp,3) 
    lh.out_degree_pn     = np.array([np.nansum(lh.out_degree[:,:,networks[i]],2) for i in range(7)])
    lh.in_degree_pn      = np.array([np.nansum(lh.in_degree [:,:,networks[i]],2) for i in range(7)])
    lh.out_strength_pn   = np.array([np.nansum(lh.out_strength[:,:,networks[i]],2) for i in range(7)])
    lh.in_strength_pn    = np.array([np.nansum(lh.in_strength [:,:,networks[i]],2) for i in range(7)])
    lh.out_minus_in_degree      = lh.out_degree - lh.in_degree
    lh.out_minus_in_strength    = lh.out_strength - lh.in_strength
    lh.out_minus_in_degree_pn   = lh.out_degree_pn - lh.in_degree_pn
    lh.out_minus_in_strength_pn = lh.out_strength_pn - lh.in_strength_pn
    return lh
    

def low_to_high_analysis_MEG(edges,LFs,HFs,networks,network_indices,alpha,N_perm,N_rat=2):
  
    N = Bunch()
    N.LF = len(LFs)
    N.ratios = len(HFs[0])
    N.parcel = 200
    N.network = 7    
    lh = Bunch()
    lh.diffs               = np.zeros([N.LF,N_rat,N.parcel,N.parcel])  
    lh.N_LTH_wilc          = np.zeros([N.LF,N_rat])

    
    #### check differences by Wilcoxon method    
    for lf,LF in enumerate(LFs):                                               #### get sign CFC edges per parcel pair ####
        for hf,HF in enumerate(HFs[lf][:2]):
            for p1 in range(N.parcel):
                for p2 in range(N.parcel):
                    a1 = edges.PLV_CF_pp_all   [lf][hf][p1][p2]
                    a2 = edges.PLV_CF_pp_all_TP[lf][hf][p1][p2]
                    lh.diffs[lf][hf][p1][p2] = np.sum(a1) - np.sum(a2)
                    s,pwil = stat.wilcoxon(edges.PLV_CF_pp_all[lf][hf][p1][p2],edges.PLV_CF_pp_all_TP[lf][hf][p1][p2])
                    if pwil<alpha:
                        lh.N_LTH_wilc[lf,hf] += 1
        print lf           
    lh.K_LTH_wilc = lh.N_LTH_wilc/200 
    
    #### check differences using the permutation method    
    lh.N_pot_LH       = np.zeros([N.LF,N_rat])
    lh.N_pot_LH_pn    = np.zeros([N.LF,N_rat,N.network,N.network])
    lh.N_pot_LH_pp    = np.zeros([N.LF,N_rat,N.parcel,N.parcel])
    lh.N_el_pot_LH    = np.zeros([N.LF,N_rat])
    lh.N_el_pot_LH_pn = np.zeros([N.LF,N_rat,N.network,N.network])
    lh.N_el_pot_LH_pp = np.zeros([N.LF,N_rat,N.parcel,N.parcel])
    lh.V_LH_sig_pp    = np.zeros([N.LF,N_rat,N.parcel,N.parcel])
    lh.V_LH_pp        = np.zeros([N.LF,N_rat,N.parcel,N.parcel])
    lh.N_LH_sig_pn    = np.zeros([N.LF,N_rat,N.network,N.network])
    lh.V_LH_sig_pn    = np.zeros([N.LF,N_rat,N.network,N.network]) 
    lh.V_LH_pn        = np.zeros([N.LF,N_rat,N.network,N.network]) 
    
    pctile         = 1 - alpha
    lh.L_sig       = np.zeros([N.LF,N.ratios,10000])  
    lh.L_sig_list  = [[[]for i in range(N_rat)]for j in range(N.LF)]            #### significant edges per length    
    
    for lf,LF in enumerate(LFs):                                               #### get sign CFC edges per parcel pair ####
        for hf,HF in enumerate(HFs[lf][:N_rat]):
            for p1 in range(N.parcel):
                for p2 in range(N.parcel):                
                    diff_perm = np.zeros(N_perm)                 
                    l1        = len(edges.PLV_CF_pp_all[lf][hf][p1][p2])
                    l2        = len(edges.PLV_CF_pp_all_TP[lf][hf][p1][p2])
                    concatA   = np.array(edges.PLV_CF_pp_all[lf][hf][p1][p2] + edges.PLV_CF_pp_all_TP[lf][hf][p1][p2])
                    indices   = range(l1+l2)  
                    diff      = np.sum(edges.PLV_CF_pp_all[lf][hf][p1][p2]) - np.sum(edges.PLV_CF_pp_all_TP[lf][hf][p1][p2])
                    lh.N_pot_LH   [lf,hf]       += 1
                    lh.N_pot_LH_pp[lf,hf,p1,p2] += 1                                       # are there electrodes connecting this parcel pair? 0 or 1
                    lh.N_pot_LH_pn[lf,hf,network_indices[p1],network_indices[p2]] += 1     # number of parcel pairs projecting to this netw. pair 
                    for n in range(N_perm): 
                        ind_rand     = np.random.permutation(indices)
                        diff_perm[n] = np.sum(concatA[ind_rand[:l1]]) - np.sum(concatA[ind_rand[l1:]])  
                    pctile_p = np.sum(diff > diff_perm)/np.float(N_perm)
                    
                    lh.V_LH_pp[lf,hf,p1,p2] = diff                                      # value of  difference in LH 1->2 and LH 2->1 at this parcel pair
                    lh.V_LH_pn[lf,hf,network_indices[p1],network_indices[p2]] += diff   # sum of  differences of all parcels pairs projecting to this network pair

                    if pctile_p > pctile: 
                        lh.V_LH_sig_pp[lf,hf,p1,p2] = diff                          # value of sign. difference in LH 1->2 and LH 2->1 at this parcel pair
                        lh.N_LH_sig_pn[lf,hf,network_indices[p1],network_indices[p2]] += 1      # number sign. different parcels pairs projecting to this network pair
                        lh.V_LH_sig_pn[lf,hf,network_indices[p1],network_indices[p2]] += diff   # sum of sign. differences of all parcels pairs projecting to this network pair
            print(time.strftime("%Y-%m-%d %H:%M") + ' lf= '+ str(lf) + ', hf= ' + str(hf))      
    
    lh.out_degree               = np.nanmean(lh.V_LH_sig_pp>0, 2)
    lh.in_degree                = np.nanmean(lh.V_LH_sig_pp>0, 3)
    lh.out_strength             = np.nanmean(lh.V_LH_pp,2)        ##### ?????
    lh.in_strength              = np.nanmean(lh.V_LH_pp,3) 
    
    
    lh.out_degree_pn            = np.array([np.nansum(lh.out_degree  [:,:,networks[i]],2) for i in range(7)])
    lh.in_degree_pn             = np.array([np.nansum(lh.in_degree   [:,:,networks[i]],2) for i in range(7)])
    lh.out_strength_pn          = np.array([np.nansum(lh.out_strength[:,:,networks[i]],2) for i in range(7)])
    lh.in_strength_pn           = np.array([np.nansum(lh.in_strength [:,:,networks[i]],2) for i in range(7)])
    lh.out_minus_in_degree      = lh.out_degree - lh.in_degree
    lh.out_minus_in_strength    = lh.out_strength - lh.in_strength
    lh.out_minus_in_degree_pn   = lh.out_degree_pn - lh.in_degree_pn
    lh.out_minus_in_strength_pn = lh.out_strength_pn - lh.in_strength_pn
    
    return N, lh  




def peak_finder(freqs,values,fmin=2,fmax=20):

    N_rat       = len(values)                         #values in format ratios x values
    curves      = np.zeros([N_rat,2,(fmax-fmin)*10])
    peak_freqs  = np.zeros(N_rat)
    
    for r in range(N_rat):
        f = scipy.interpolate.interp1d(freqs,values[r],kind='cubic')
        xnew = np.arange(fmin,fmax,0.1)
        ynew = f(xnew)
        curves[r] = [xnew,ynew]
        peak_freqs[r]=xnew[np.argmax(ynew)]
        
    return peak_freqs, curves         



def write_csv_low_to_high(directory,lh,ratios2,parc,CF_type,add_inf=''):
    
    write_dir = directory + '_results\\graph metrics 2018-05\\' + CF_type + ' LH '+ parc + add_inf 
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
        
    file_LH_out_deg      =  [write_dir + '\\CFC out-deg ' + i + '.csv' for i in ratios2[:2]]
    file_LH_in_deg       =  [write_dir + '\\CFC in-deg '  + i + '.csv' for i in ratios2[:2]]
    file_LH_diff_deg     =  [write_dir + '\\CFC out-minus_in-deg ' + i + '.csv' for i in ratios2[:2]]

    file_LH_out_str      =  [write_dir + '\\CFC out-str ' + i + '.csv' for i in ratios2[:2]]
    file_LH_in_str       =  [write_dir + '\\CFC in-str '  + i + '.csv' for i in ratios2[:2]]    
    file_LH_diff_str     =  [write_dir + '\\CFC out-minus_in-str ' + i + '.csv' for i in ratios2[:2]]

    for i in range(2):
        np.savetxt(file_LH_out_deg[i],  lh.out_degree  [:,i,:], delimiter=";")
        np.savetxt(file_LH_in_deg[i],   lh.in_degree   [:,i,:], delimiter=";")        
        np.savetxt(file_LH_diff_deg[i], lh.out_minus_in_degree   [:,i,:], delimiter=";")
        np.savetxt(file_LH_out_str[i],  lh.out_strength[:,i,:], delimiter=";")
        np.savetxt(file_LH_in_str[i],   lh.in_strength [:,i,:], delimiter=";")
        np.savetxt(file_LH_diff_str[i], lh.out_minus_in_strength   [:,i,:], delimiter=";")

def analyze_local(N,edges,networks,N_ch_layer):
        
    N.chan_per_parc          = np.nansum(N.chan_per_parc_subj,0)
    
    edges.mean_PLV_local_pp  = edges.PLV_local_pp     /N.chan_per_parc
    edges.K_local_pp         = edges.N_sign_local_pp  /N.chan_per_parc 
    edges.mean_PLV_local_pl  = edges.PLV_local_pl     /N_ch_layer
    edges.K_local_pl         = edges.N_sign_local_pl  /N_ch_layer    
    edges.mean_PLV_local_pn  = np.array([np.nanmean(edges.mean_PLV_local_pp [:,:,networks[i]],2) for i in range(7)])
    edges.K_local_pn         = np.array([np.nanmean(edges.K_local_pp   [:,:,networks[i]],2) for i in range(7)])
    
    return N,edges


def analyze_PS(N,edges,networks,N_ch_layer):
        
    N.chan_per_parc       = np.sum(N.chan_per_parc_subj,0)
    
    edges.mean_PLV_ps_pp  = edges.metric_PS_pp  /N.chan_per_parc
    edges.K_ps_pp         = edges.N_sign_PS_pp  /N.chan_per_parc 
    edges.mean_PLV_ps_pn  = np.array([np.nanmean(edges.mean_PLV_ps_pp  [:,:,networks[i]],(1,2)) for i in range(7)])
    edges.K_ps_pn         = np.array([np.nanmean(edges.K_ps_pp         [:,:,networks[i]],(1,2)) for i in range(7)])
 #   edges.mean_PLV_ps_pnn = np.array([[np.nanmean(edges.mean_PLV_ps_pp [:,networks[i],networks[j]],(1,2)) for i in range(7)]for j in range(7)])
 #   edges.K_ps_pnn        = np.array([[np.nanmean(edges.K_ps_pp        [:,networks[i],networks[j]],(1,2)) for i in range(7)]for j in range(7)])
    
    return N,edges







def write_csv_local(directory,edges,ratios2,parc,CF_type,add_inf=''):
    
    write_dir = directory + '_results\\graph metrics 2018-05\\' + CF_type + ' local '+ parc + add_inf   
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
        
    file_PLV_local     =  [write_dir + '\\PLV local ' + i + '.csv' for i in ratios2]
    file_K_local       =  [write_dir + '\\K local '  + i + '.csv' for i in ratios2]

    for i in range(6):
        np.savetxt(file_PLV_local[i],  edges.mean_PLV_local_pp [:,i,:], delimiter=";")
        np.savetxt(file_K_local[i],    edges.K_local_pp   [:,i,:], delimiter=";")        
        

def degree_analysis(edges,N,networks):
    D = Bunch()
    
    D.degree_PS_pp         = np.sum(edges.N_sign_PS_pp,1)    
    D.degree_CF_pp         = np.sum(edges.N_sign_CF_pp,2)    + np.sum(edges.N_sign_CF_pp,3)
    D.degree_LF_pp         = np.sum(edges.N_sign_CF_pp,2)   
    D.degree_HF_pp         = np.sum(edges.N_sign_CF_pp,3)
    D.degree_CF_pp_mod     = np.sum(edges.N_sign_CF_pp_mod,2) + np.sum(edges.N_sign_CF_pp_mod,3)
    D.degree_LF_pp_mod     = np.sum(edges.N_sign_CF_pp_mod,2) 
    D.degree_HF_pp_mod     = np.sum(edges.N_sign_CF_pp_mod,3)
    
    D.rel_degree_PS_pp     = np.nansum(edges.N_sign_PS_pp,1)/(1.*np.nansum(edges.N_pot_pp,0))     
    D.rel_degree_CF_pp     = np.nansum(edges.N_sign_CF_pp,2)/(1.*np.nansum(edges.N_pot_pp,0))     + np.nansum(edges.N_sign_CF_pp,3)/(1.*np.nansum(edges.N_pot_pp,1))
    D.rel_degree_LF_pp     = np.nansum(edges.N_sign_CF_pp,2)/(1.*np.nansum(edges.N_pot_pp,0))     
    D.rel_degree_HF_pp     = np.nansum(edges.N_sign_CF_pp,3)/(1.*np.nansum(edges.N_pot_pp,1))
    D.rel_degree_CF_pp_mod = np.nansum(edges.N_sign_CF_pp_mod,2)/(1.*np.nansum(edges.N_pot_pp,0)) + np.nansum(edges.N_sign_CF_pp_mod,3)/(1.*np.nansum(edges.N_pot_pp,1))
    D.rel_degree_LF_pp_mod = np.nansum(edges.N_sign_CF_pp_mod,2)/(1.*np.nansum(edges.N_pot_pp,0)) 
    D.rel_degree_HF_pp_mod = np.nansum(edges.N_sign_CF_pp_mod,3)/(1.*np.nansum(edges.N_pot_pp,1))
    
    D.degree_PS_pn         = np.array([np.nanmean(D.degree_PS_pp[:,networks[i]],1) for i in range(7)])
    D.degree_CF_pn         = np.array([np.nanmean(D.degree_CF_pp[:,:,networks[i]],2) for i in range(7)])
    D.degree_LF_pn         = np.array([np.nanmean(D.degree_LF_pp[:,:,networks[i]],2) for i in range(7)])   
    D.degree_HF_pn         = np.array([np.nanmean(D.degree_HF_pp[:,:,networks[i]],2) for i in range(7)])
    D.degree_CF_pn_mod     = np.array([np.nanmean(D.degree_CF_pp_mod[:,:,networks[i]],2) for i in range(7)])
    D.degree_LF_pn_mod     = np.array([np.nanmean(D.degree_LF_pp_mod[:,:,networks[i]],2) for i in range(7)])
    D.degree_HF_pn_mod     = np.array([np.nanmean(D.degree_HF_pp_mod[:,:,networks[i]],2) for i in range(7)])
    
    D.rel_degree_PS_pn     = np.array([np.nanmean(D.rel_degree_PS_pp[:,networks[i]],1) for i in range(7)])
    D.rel_degree_LF_pn     = np.array([np.nanmean(D.rel_degree_LF_pp[:,:,networks[i]],2) for i in range(7)])   
    D.rel_degree_HF_pn     = np.array([np.nanmean(D.rel_degree_HF_pp[:,:,networks[i]],2) for i in range(7)])
    D.rel_degree_CF_pn     = np.array([np.nanmean(D.rel_degree_CF_pp[:,:,networks[i]],2) for i in range(7)])
    D.rel_degree_CF_pn_mod = np.array([np.nanmean(D.rel_degree_CF_pp_mod[:,:,networks[i]],2) for i in range(7)])
    D.rel_degree_LF_pn_mod = np.array([np.nanmean(D.rel_degree_LF_pp_mod[:,:,networks[i]],2) for i in range(7)])
    D.rel_degree_HF_pn_mod = np.array([np.nanmean(D.rel_degree_HF_pp_mod[:,:,networks[i]],2) for i in range(7)])

    return D

def write_degrees(directory,D,ratios2,parc, CF_type, add_inf=''):    

    write_dir = directory + '_results\\graph metrics 2018-05\\' + CF_type + ' degrees '+ parc + add_inf  
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
        
    file_deg_PS      =  write_dir + '\\PS.csv'
    file_deg_CF      = [write_dir + '\\CFC CF '      + i + '.csv' for i in ratios2]
    file_deg_LF      = [write_dir + '\\CFC LF '      + i + '.csv' for i in ratios2]
    file_deg_HF      = [write_dir + '\\CFC HF '      + i + '.csv' for i in ratios2]
    file_deg_CF_mod  = [write_dir + '\\CFC CF_mod '  + i + '.csv' for i in ratios2]
    file_deg_LF_mod  = [write_dir + '\\CFC LF_mod '  + i + '.csv' for i in ratios2]
    file_deg_HF_mod  = [write_dir + '\\CFC HF_mod '  + i + '.csv' for i in ratios2]
    
    np.savetxt(file_deg_PS,      D.rel_degree_PS_pp, delimiter=";")
    for i in range(len(ratios2)):   
        np.savetxt(file_deg_CF[i],      D.rel_degree_CF_pp[:,i,:], delimiter=";")    
        np.savetxt(file_deg_LF[i],      D.rel_degree_LF_pp[:,i,:], delimiter=";")
        np.savetxt(file_deg_HF[i],      D.rel_degree_HF_pp[:,i,:], delimiter=";")
        np.savetxt(file_deg_CF_mod[i],  D.rel_degree_CF_pp_mod  [:,i,:], delimiter=";")
        np.savetxt(file_deg_LF_mod[i],  D.rel_degree_LF_pp_mod  [:,i,:], delimiter=";")
        np.savetxt(file_deg_HF_mod[i],  D.rel_degree_HF_pp_mod  [:,i,:], delimiter=";")
 





def read_complex_data_from_csv(filename_base,delimiter=';'):

    filename_RE = filename_base + '_re.csv'
    filename_IM = filename_base + '_im.csv'
    data_re     = np.genfromtxt(filename_RE, delimiter=';')
    data_im     = np.genfromtxt(filename_IM, delimiter=';')
    data        = data_re + 1j*data_im
    return data
                      









                
class Interaction:                                                              # unused????
    pass 

class Bunch:
    __init__ = lambda self, **kw: setattr(self, '__dict__', kw)

class DataB:
    def __init__(self,val,N_CH):
        self.data_masked = np.full([N_CH,N_CH],val)
        self.data_sign   = np.full([N_CH,N_CH],val)  
                    
class stats_PS:
    def __init__(self,val):
        self.mean_masked = val
        self.mean_sign   = val
        self.N           = val
        self.K           = val
        self.N_pot       = val
        self.degree      = val        
                
class stats_CFC:
    def __init__(self,val):
        self.mean_masked   = val
        self.mean_sign     = val
        self.mean_mod      = val 
        self.mean_excl     = val
        self.mean_local    = val

        self.N             = val       
        self.N_mod         = val
        self.N_excl        = val
        self.N_local       = val

        self.K             = val
        self.K_mod         = val
        self.K_excl        = val 
        self.K_local       = val        

        self.N_pot         = val  
        self.N_pot_mod     = val   
        self.N_pot_excl    = val   
        self.N_CH          = val

        self.degree_LF      = val        
        self.degree_HF      = val
        self.degree_LF_mod  = val        
        self.degree_HF_mod  = val
        self.degree_LF_excl = val
        self.degree_HF_excl = val        

        self.strength         = val
        self.strength         = val
        self.strength_LF_mod  = val
        self.strength_HF_mod  = val
        self.strength_LF_excl = val
        self.strength_HF_excl = val
                
                
                
                