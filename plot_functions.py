# -*- coding: utf-8 -*-
"""
@author: Felix Siebenhühner
"""


import numpy as np
import matplotlib as mp
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
from numpy.ma import masked_invalid


def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            print("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            print("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap





   
def semi_log_plot(figsize,data,freqs,xlim,ylab,legend=None,outfile=None,legend_pos=None,ylim=None,show=False,cmap='gist_rainbow',ncols=3,CI=False):   
    fig,ax=plt.subplots(figsize=figsize) 
    ax.hold(True)
    if type(cmap) is mpc.LinearSegmentedColormap:
        colorspace = [cmap(i) for i in np.linspace(0, 1, len(data))]            # get colorspace from colormap
    else:
        colorspace = [plt.get_cmap(cmap)(i) for i in np.linspace(0, 1, len(data))]   
    colorspace_CI = np.array(colorspace)*np.array([1,1,1,0.3])                              # colors for confidence intervals
    ax.set_color_cycle(colorspace)                                                          # set different colors for different plots
    for i in range(len(data)):                                                              # for each plot i
        if CI==True: 
            ax.plot(freqs[:len(data[i][0])],data[i][0])                                     # if CI, data for each plot i comes as [mean,CI_low, CI_high]
            ax.fill_between(freqs,data[i][1],data[i][2],color=colorspace_CI[i])             # fill between CI_low and CI_high
        else:
            ax.plot(freqs[:len(data[i])],data[i])          
        
    ax.set_xscale('log')
    ax.set_xticks([1,2,3,5,10,20,30,50,100,200,300])
    ax.get_xaxis().set_major_formatter(mp.ticker.ScalarFormatter())
    ax.set_xlim(xlim) 
    ax.set_ylabel(ylab,fontsize=14)
    ax.set_xlabel('Frequency [Hz]',fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if ylim!=None:
        ax.set_ylim(ylim) 
    if legend_pos=='uc':
        plt.legend(legend,loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=ncols)
    if legend_pos=='ur':
        plt.legend(legend, loc='upper right', ncol=ncols)    
    if outfile!=None:    
        plt.savefig(outfile) 
    if show:
        plt.show()
    plt.clf()
    
    
    

def semi_log_plot_multi(figsize,rows,cols,dataL,freqs,xlimA,ylabA,titlesA,cmapA,legendA=None,
                        outfile=None,legend_posA=None,ylimA=None,show=False,ncols=3,CI=None,xlab=None,Nyt=None,fontsize=8,markersize=0):   
    fig,axes=plt.subplots(rows,cols,figsize=figsize)
    if CI == None:
        CI = [None for i in range(len(dataL))]  
    if ylimA==None:
        ylimA = [False for i in range(len(dataL))]
    if xlab==None:
        xlab = [False for i in range(len(dataL))]
    if legend_posA==None:
        legend_posA = ['' for i in range(len(dataL))]      
        
    for d,data in enumerate(dataL):
        if (rows==1) or (cols ==1):
            ax = axes[d]
        else:
            ax = axes[d/cols,d%cols]
        ax.hold(True)
        ax.set_title(titlesA[d])
        
        if type(cmapA[d]) is mpc.LinearSegmentedColormap:
            colorspace = [cmapA[d](i) for i in np.linspace(0, 1, len(data))]
        else:
            colorspace = [plt.get_cmap(cmapA[d])(i) for i in np.linspace(0, 1, len(data))]
        if CI[d]!= None:
            colorspace_CI = np.array(colorspace)*np.array([1,1,1,CI[d]])
            ax.set_color_cycle(colorspace)
        for i in range(len(data)):            
            if CI[d]!=None:
                fr = freqs[:len(data[i][0])]
                ax.plot(fr,data[i][0],'*-',markersize=markersize)
                ax.fill_between(fr,data[i][1],data[i][2],color=colorspace_CI[i])    
            else:
                fr = freqs[:len(data[i])]
                ax.plot(fr,data[i],'*-',markersize=markersize)
        if Nyt != None:
            if Nyt[d] ==1:
                for label in ax.get_yticklabels()[::2]:
                    label.set_visible(False)  
                 
        ax.set_xscale('log')
        ax.get_xaxis().set_major_formatter(mp.ticker.ScalarFormatter()) 
        ax.set_xticks([1,2,3,5,10,20,30,50,100,200,300])      
        if xlab[d]==True:
            ax.set_xlabel('Low frequency [Hz]',fontsize=fontsize)       
        ax.set_xlim(xlimA[d]) 
        ax.set_ylabel(ylabA[d],fontsize=fontsize)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if ylimA[d]!=None:
            ax.set_ylim(ylimA[d]) 
        if legend_posA[d]=='uc':
            ax.legend(legendA[d],loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=ncols)
        if legend_posA[d]=='ur':
            ax.legend(legendA[d], loc='upper right', ncol=ncols, frameon=0)  
        if legend_posA[d]=='ul':
            ax.legend(legendA[d],loc='upper left', ncol=ncols, frameon=0)
    plt.tight_layout()   
    if outfile!=None:    
        plt.savefig(outfile)
    if show:
        plt.show()
  
    plt.clf()
    

def semi_log_plot_multi2(figsize,rows,cols,dataL,freqs,xlimA,ylabA,titlesA,cmapA,legendA=None,
                        outfile=None,legend_posA=None,ylimA=None,show=False,ncols=3,CI=None,xlab=None,Nyt=None,fontsize=8,markersize=0):   
    fig,axes=plt.subplots(rows,cols,figsize=figsize)
    if CI == None:
        CI = [None for i in range(len(dataL))]  
    if ylimA==None:
        ylimA = [False for i in range(len(dataL))]
    if xlab==None:
        xlab = [False for i in range(len(dataL))]
    if legend_posA==None:
        legend_posA = ['' for i in range(len(dataL))]      
        
    for d,data in enumerate(dataL):
        if (rows==1) or (cols ==1):
            ax = axes[d]
        else:
            ax = axes[d/cols,d%cols]
        ax.hold(True)
        ax.set_title(titlesA[d])
        
        if type(cmapA[d]) is mpc.LinearSegmentedColormap:
            colorspace = [cmapA[d](i) for i in np.linspace(0, 1, len(data))]
        else:
            colorspace = [plt.get_cmap(cmapA[d])(i) for i in np.linspace(0, 1, len(data))]
        if CI[d]!= None:
            colorspace_CI = np.array(colorspace)*np.array([1,1,1,CI[d]])
            ax.set_color_cycle(colorspace)
        for i in range(len(data)):            
            if CI[d]!=None:
                fr = freqs[i][:len(data[i][0])]
                ax.plot(fr,data[i][0],'*-',markersize=markersize)
                ax.fill_between(fr,data[i][1],data[i][2],color=colorspace_CI[i])    
            else:
                fr = freqs[i][:len(data[i])]
                ax.plot(fr,data[i],'*-',markersize=markersize)
        if Nyt != None:
            if Nyt[d] ==1:
                for label in ax.get_yticklabels()[::2]:
                    label.set_visible(False)  
                 
        ax.set_xscale('log')
        ax.get_xaxis().set_major_formatter(mp.ticker.ScalarFormatter()) 
        ax.set_xticks([1,2,3,5,10,20,30,50,100,200,300])      
        if xlab[d]==True:
            ax.set_xlabel('High frequency [Hz]',fontsize=fontsize)       
        ax.set_xlim(xlimA[d]) 
        ax.set_ylabel(ylabA[d],fontsize=fontsize)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if ylimA[d]!=None:
            ax.set_ylim(ylimA[d]) 
        if legend_posA[d]=='uc':
            ax.legend(legendA[d],loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=ncols)
        if legend_posA[d]=='ur':
            ax.legend(legendA[d], loc='upper right', ncol=ncols, frameon=0)  
        if legend_posA[d]=='ul':
            ax.legend(legendA[d],loc='upper left', ncol=ncols, frameon=0)
    plt.tight_layout()   
    if outfile!=None:    
        plt.savefig(outfile)
    if show:
        plt.show()
  
    plt.clf()

def plot_PLV_distribution(figsize,data,freqs,outfile):
    
    plt.figure(figsize=[10,5], dpi=600, facecolor='w', edgecolor='k')
    p, axes = plt.subplots(nrows=5, ncols=10,figsize=[40,60])
    pos = range(np.size(data,1))
    k=0
    for i in range(5):
        for j in range(10):
                k+=1
                if k<len(data):
                    ax=axes[i,j]
                    parts = ax.violinplot(data[k],pos,widths=0.6,
                          showmeans=True, showextrema=False, showmedians=True)
                    for pc in parts['bodies']:
                        pc.set_facecolor('#D43F3A')
                        pc.set_edgecolor('black')
                        pc.set_alpha(1)
                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(20)  
                    ax.set_xticks([0,2,4,6,8,10])    
                    
                    print k
    plt.savefig(outfile)
    plt.close()



def plot_histogram(data, N_bins=20):
    import matplotlib.pyplot as plt
    import numpy as np
    
    hist, bins = np.histogram(data, bins=N_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


    
        
def log_log_plot(figsize,data,freqs,xlim,ylab,legend=None,outfile=None,legend_pos=None,ylim=None,show=False,cmap='gist_rainbow',ncols=3):   
    plt.figure(figsize=figsize, dpi=600, facecolor='w', edgecolor='k')
    p,ax=plt.subplots() 
    plt.hold(True)
    ax.set_color_cycle([plt.get_cmap(cmap)(i) for i in np.linspace(0, 0.9, len(data))])
    for i in range(len(data)):
        ax.plot(freqs[:len(data[i])],data[i])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([1,2,3,5,10,20,30,50,100,200,300])
    ax.get_xaxis().set_major_formatter(mp.ticker.ScalarFormatter())
    ax.set_xlim(xlim) 
    ax.set_ylabel(ylab,fontsize=14)
    ax.set_xlabel('Frequency [Hz]',fontsize=14)
    if ylim!=None:
        ax.set_ylim(ylim) 
    if legend_pos=='uc':
        plt.legend(legend,loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=ncols)
    if legend_pos=='ur':
        plt.legend(legend, loc='upper right', ncol=ncols)    
    if outfile!=None:    
        plt.savefig(outfile)
    if show:
        plt.show()
    plt.clf()
        
    return p
    
   

def simple_CF_plot(data,figsize,xname,yname,xtix,ytix,xlabels,ylabels,zmax=1,ztix=[0,0.2,0.4,0.6,0.8,1],outfile=None):            
    
    #data in shape freq x ratio
    eps = np.spacing(0.0)                                            # an epsilon
    CM=plt.cm.YlOrRd 
    CM.set_under('None')
    fig=plt.figure(figsize=figsize, facecolor='w', edgecolor='k')
    ax=fig.add_axes([0.2,0.2,.7,.7])
    mesh=ax.pcolormesh(data,vmin=eps,vmax=zmax,cmap=CM)    
    ax.set_xticks(xtix)
    ax.set_xticklabels(xlabels,rotation=45)
    ax.set_xlim([0,len(data[0])])      
    ax.set_yticks(ytix)
    ax.set_yticklabels(ylabels)
    ax.set_ylim([0,len(data)])
    ax.set_xlabel(xname)
    cbar = fig.colorbar(mesh, ticks=ztix)
    cbar.ax.tick_params(axis='y', direction='out')
    ax.tick_params(axis='both',which='both',length=0)
    ax.set_ylabel(yname)    
    if outfile!=None:
        fig.savefig(outfile)



    
 
    
    
    
    
    
    
    
    
    
    