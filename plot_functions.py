# -*- coding: utf-8 -*-
"""
@author: felix
please don't modify existing functions without consulting me!
"""


import numpy as np
import matplotlib as mp
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
from numpy.ma import masked_invalid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

''' list of functions: 
    
    make_cmap: creates colormap from list of RGB values
    
    plot_lines: simple line plot with easy customization
    
    plot_heatmap: plots 2D data as a "heatmap"
    
    plot_heatmaps: plots multiple heatmaps into one figure
    
    semi_log_plot: plots data series over log-spaced frequencies 
                   optionally with confidence intervals/standard deviation
                   
    semi_log_plot_multi: plots data series over log-freqs. in subplots 
                   optionally with confidence intervals/standard deviation
                    
    simple_CF_plot: heatmap plot specifically for the visualization of 
                multi-ratio cross-frequency interactions
                    
'''





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

    cmap = mp.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap




def plot_lines(data, names=None, figsize=[13,2], cmap='jet', colors = None, 
               xlabel ='', ylabel = '', xticks = None, yticks = None, xticklabels = None, 
               less_spines = True, outfile = None, xlim = None, ylim = None, fontsize=12):
    ''' INPUT:
        data:    1D array or list, or 2D array or list
        names:   list of names of data series for legend
        figsize: Figure size
        cmap:    Colormap. Can be:
                 - the name of a library cmap
                 - an instance of mpc.LinearSegmentedColormap    
                 - a list of colors as characters or RGB tuples (in 0-1 range)
        xlabel, ylabel: axis labels
        xticks, yticks: axis ticks, list of int
        less_spines: no axes on right and top 
        outfile: file to save the plot to 
        xlim, ylim: x and y limits, e.g. xlim=[-4,4]
    '''  
    try:                            # if 1D, force to 2D
        d54 = data[0][0]
    except IndexError:
        data = [data]
    
    if type(cmap) is list:
        colors = cmap
    elif type(cmap) is mpc.LinearSegmentedColormap:
        colors = [cmap(i) for i in np.linspace(0, 1, len(data))] 
    else:
        colors = [plt.get_cmap(cmap)(i) for i in np.linspace(0, 1, len(data))]         
                
    fig = plt.figure(figsize=figsize)
    ax  = plt.subplot(111)    
    for i in range(len(data)):
        ax.plot(data[i],color=colors[i])

    ax.tick_params(labelsize=fontsize-2)
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_ylabel(ylabel,fontsize=fontsize)
    if names != None:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(names,loc='center left', bbox_to_anchor=(1, 0.5),fontsize=fontsize)

    if np.all(xticks != None):        
        ax.set_xticks(xticks)  
    if np.all(xticklabels != None):
        ax.set_xticklabels(xticklabels,fontsize=fontsize) 
    if yticks != None:        
        ax.set_yticks(yticks)
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
        
    if less_spines:        
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    
    if outfile != None:
        plt.savefig(outfile) 

    



        
    
def plot_heatmap(data,figsize=[9,7],
                 cbar='right',zmax=None,zmin=None,cmap=plt.cm.YlOrRd,  
                 xlabel='',ylabel='',zlabel='',fontsize = 18,
                 xticks=None,yticks=None,zticks=None,
                 xticklabels=None, yticklabels=None,
                 zticklabels=None, xticklab_rot=45,
                 masking='False',bad='white',under=None, topaxis=False):
    ''' INPUT:
        data:                 2D array
        figsize:              figure size             
        cbar:                 can be 'right', 'bottom' or None
        zmax:                 max value for z axis (auto if None)
        zmin:                 min value for z axis (epsilon if None)
        cmap:                 colormap for colorbar; either:
                                - Instance of mpc.LinearSegmentedColormap 
                                - name of library cmap                            
        xlabel,ylabel,zlabel: labels for x, y, z axes
        fontsize:             fontsize for major labels
        xticks,yticks,zticks: tick values for x, y, z axes
        xticklabels:          labels for x ticks
        yticklabels:          labels for y ticks
        xticklab_rot:         degrees by which to rotate xticklabels
        masking:              whether to mask invalid values (inf,nan)
        bad:                  color for masked values
        under:                color for low out-of range values
        topaxis:              whether to have additional x axis on top
    
    '''
    
    eps = np.spacing(0.0)
    fig, ax = plt.subplots(1,figsize=figsize)

    if type(cmap) != mpc.LinearSegmentedColormap:
        cmap = mp.cm.get_cmap(cmap)
    cmap.set_bad(bad)
        
    if under !=None:
        cmap.set_under(under)        
    if masking:
        data = masked_invalid(data)
    if topaxis:
        ax.tick_params(labeltop=True)        
    if zmax == None:
        zmax = max(np.reshape(data,-1))     
        
    PCM = ax.pcolormesh(data,vmin=eps,vmax=zmax,cmap=cmap)   
    
    ax.set_xlim([0,len(data[0])])
    ax.set_ylim([0,len(data)])
    
    if np.all(xticks != None): 
        if xticks == 'auto':    
            Nx=len(xticklabels)
            ax.set_xticks(np.arange(Nx)+0.5)
            ax.set_xlim([0,Nx]) 
        else:
            ax.set_xticks(xticks)
        if xticklabels !=None:
            ax.set_xticklabels(xticklabels,rotation=xticklab_rot)
            
    if np.all(yticks != None): 
        if yticks == 'auto':    
            Ny=len(yticks)
            ax.set_yticks(np.arange(Ny)+0.5)
            ax.set_ylim([0,Ny]) 
        else:
            ax.set_yticks(yticks)
        if yticklabels !=None:
            ax.set_yticklabels(yticklabels)

        
    ax.set_xlabel(xlabel,fontsize=18)
    ax.set_ylabel(ylabel,fontsize=18)
    ax.tick_params(axis='both',which='both',length=0,labelsize=fontsize-2)
    if cbar == 'bottom':
        orient = 'horizontal'
    else:
        orient = 'vertical'
    if cbar != None:    
        if zticks !=None:
            cb  = plt.colorbar(PCM, ax=ax, ticks = zticks, orientation = orient)
            if zticklabels  != None:
                if orient == 'vertical':
                    cb.ax.set_yticklabels(zticklabels)
                else:
                    cb.ax.set_xticklabels(zticklabels)
        else:
            cb  = plt.colorbar(PCM, ax=ax, orientation = orient)
        cb.set_label(zlabel,fontsize=18)
        cb.ax.tick_params(labelsize=14) 




def plot_heatmaps(data, titles=None, N_cols=3, figsize=None, fontsizeT=13, fontsizeL=11, 
                  ylabel=None, xlabel=None, zlabel= None, cmap='jet',zmax=None, zmin=0,
                  xticks = None, yticks = None, zticks = None,
                  xticklabels = None, yticklabels=None, zticklabels=None,xticklab_rot=0):
    ''' Input:
        data:      3D array or list of 2D arrays
        titles:    array of titles, empty by default 
        N_cols:    number of columns, default 3
        figsize:   fixed figure size, will be determined automatically if None
        fontsizeT: fontsize for title
        fontsizeL: fontsize for labels
        xlab,ylab,clab: axis labels, empty by default, can be single string or list of strings 
        cmap:      name of a library cmap, or instance of mpc.LinearSegmentedColormap, or a list of either of these
    '''    
    

    N_plots = len(data)
    N_rows  = int(np.ceil(1.*N_plots/N_cols) )
    
    if figsize==None:
        figsize =[N_cols*4.8,N_rows*3.5]

        
    cmaps       = _repeat_if_needed(cmap, N_plots, 1)    
    zmax        = _repeat_if_needed(zmax, N_plots, 1)   
    zmin        = _repeat_if_needed(zmin, N_plots, 1)   
    xticks      = _repeat_if_needed(xticks, N_plots, 2)   
    yticks      = _repeat_if_needed(yticks, N_plots, 2) 
    zticks      = _repeat_if_needed(zticks, N_plots, 2)   
    xticklabels = _repeat_if_needed(xticklabels, N_plots, 2)   
    yticklabels = _repeat_if_needed(yticklabels, N_plots, 2)  
    zticklabels = _repeat_if_needed(zticklabels, N_plots, 2)  

    fig,axes=plt.subplots(N_rows,N_cols,figsize=figsize)     
    plt.subplots_adjust(wspace=.2,hspace=.3)
    
    if type(xlabel) == str:
        xlabel = [xlabel] * N_plots
    if type(ylabel) == str:
        ylabel = [ylabel] * N_plots    
    if type(zlabel) == str:
        zlabel = [zlabel] * N_plots  

    
    for i in range(N_plots):    
        if (N_rows==1) or (N_cols ==1):
            ax = axes[i]
        else:
            ax = axes[i/N_cols,i%N_cols]            
        
        ax.hold(True)
        ax.grid(False)
        if zmax[i] == None:
            zmax[i] = np.max(data[i])
        p = ax.imshow(data[i],origin='bottom',interpolation='none',cmap=cmaps[i],vmax=zmax[i],vmin=zmin[i])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        if np.all(zticks[i] != None):
            cb = plt.colorbar(p, cax=cax, ticks = zticks[i])  
            if zticklabels[i] != None:
                    cb.ax.set_yticklabels(zticklabels)
        else:
            zticks[i] = [zmax[i]*j for j in [0,1./4,1./2,3./4,1.]]
            cb = plt.colorbar(p, cax=cax, ticks = zticks[i])  
          
                    
        if np.all(titles!=None):
            ax.set_title(titles[i],fontsize=fontsizeT)
        if np.all(xlabel!=None):    
            ax.set_xlabel(xlabel[i], fontsize=fontsizeL)    
        if np.all(ylabel!=None):    
            ax.set_ylabel(ylabel[i], fontsize=fontsizeL)   
        if np.all(zlabel!=None):    
            cb.set_label(zlabel[i], fontsize=fontsizeL)  
        if np.all(xticks[i] !=None):
            ax.set_xticks(xticks[i])
        if np.all(yticks[i] !=None):
            ax.set_yticks(yticks[i])
        if np.all(xticklabels[i] !=None):
            ax.set_xticklabels(xticklabels[i],rotation=xticklab_rot)
        if np.all(yticklabels[i] !=None):
            ax.set_yticklabels(yticklabels[i])

   
def semi_log_plot(figsize,data,freqs,xlim,ylabel,legend=None,outfile=None,
                  legend_pos='ur',ylim=None,show=True,cmap='gist_rainbow',
                  ncols=3,CI=False,xticks=None,bgcolor=None,
                  fontsize=11, markersize=0,
                  sig_id=[None],sig_fac=1.05,sig_style='*'):   
    '''
    plots data over log-spaced frequencies; 
    with or without confidence intervals/standard deviation    
    
    figsize:       figure size [width, height]
    data:          2D or 3D array/list (without or with CI/SD resp.)
                   1st dim: groups
                   optional dim = [mean, lower bound, upper bound] for CI or SD
                   last dim: frequencies
    freqs:         1D or 2D array/list of frequencies (float)
                   use 2D if different frequencies for different groups                         
    xlim:          [xmin,xmax]
    ylabel:        label for the y axis
    legend:        array of strings, 1 for each group
    outfile:       if not None, figure will be exported to this file
    legend_pos:    position of legend ('uc','br' or 'ur'); no legend if None
    ylim:          [ymin,ymax]
    show:          if False, the figure is not shown in console/window
    cmap:          either name of a standard colormap 
                   or an instance of matplotlib.colors.LinearSegmentedColormap
    ncols:         number of columns in the plot legend
    CI:            None if no CI/SDs, else alpha value (0 to 1) for CI/SD areas
    xticks:        custom values for xticks. if None, standard value are used
    bgcolor:       background color
    fontsize:      fontsize
    markersize:    size of data point marker, default = 0
    sig_id:        indices where significance is to be indicated
    sig_fac:       controls the height at which significance indicators shown
    sig_style:     can be e.g. '*' or '-' or '-*'    
    '''
    depth = lambda L: isinstance(L, list) and max(map(depth, L))+1
    fig,ax=plt.subplots(figsize=figsize) 
    ax.hold(True)
    if type(cmap) is mpc.LinearSegmentedColormap:
        colorspace = [cmap(i) for i in np.linspace(0, 1, len(data))]            # get colorspace from colormap
    else:
        colorspace = [plt.get_cmap(cmap)(i) for i in np.linspace(0, 1, len(data))]   
    if CI != None:
        colorspace_CI = np.array(colorspace)*np.array([1,1,1,CI])                              # colors for confidence intervals
    ax.set_color_cycle(colorspace)                                                          # set different colors for different plots
    for i in range(len(data)):                                                              # for each plot i
        if depth(freqs)==1:    
            freqs2=freqs[i]
        else:
            freqs2=freqs
        if CI != False: 
            N_F = len(data[i][0])
            ax.plot(freqs2[:N_F],data[i][0],'o-',markersize=markersize)                                     # if CI, data for each plot i comes as [mean,CI_low, CI_high]
            ax.fill_between(freqs2,data[i][1],data[i][2],color=colorspace_CI[i])             # fill between CI_low and CI_high
        else:
            N_F = len(data[i])
            ax.plot(freqs2[:N_F],data[i],'o-',markersize=markersize)          
    if xticks==None:
        xticks=[1,2,3,5,10,20,30,50,100,200,300]
    ax.set_xscale('log')
    ax.set_xticks(xticks)
    if bgcolor != None:
        ax.set_axis_bgcolor(bgcolor)
    ax.get_xaxis().set_major_formatter(mp.ticker.ScalarFormatter())
    ax.set_xlim(xlim) 
    ax.axis('on')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel('Frequency [Hz]', fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    if sig_id[0]!=None:
        plt.plot(freqs[:len(sig_id)],sig_id*np.nanmax(data)*sig_fac,
                       sig_style,color='k')
    if ylim!=None:
        ax.set_ylim(ylim) 
        
    loc_dict = {'uc': 'upper center', 'ur': 'upper center', 
                'br': 'lower right', 'best': 'best'}  
    if legend!=None and legend_pos!=None:
        plt.legend(legend, loc=loc_dict.get(legend_pos), ncol=ncols, fontsize=fontsize) 
    if outfile!=None:    
        plt.savefig(outfile) 
    if show:
        plt.show()
        
        
        
        

    
    
    

def semi_log_plot_multi(figsize,rows,cols,dataL,freqs,xlimA,ylabA,titlesA,
                        cmapA,legendA=None,outfile=None,legend_posA=None,
                        ylimA=None,show=False,ncols=3,CI=None,
                        xlabA=None,Ryt=None,xticks='auto',
                        fontsize=8,markersize=0): 
    '''
    multiple subplots of data over log-spaced frequencies
    with or without confidence intervals/standard deviation 
    
    figsize:       figure size [width, height]
    rows:          number of rows for subplots
    cols:          number of columns for subplots
    dataL:         3D or 4D array/list (without or with CI/SD resp.)
                   1st dim: datasets, 1 per subplot
                   2nd dim: groups within subplot 
                   optional dim: [mean, lower bound, upper bound] for CI or SD
                   last dim: frequencies   
                   The numbers of groups and frequencies can vary between 
                   subplots, if your dataL object is a list on the 1st dim.
    freqs:         1D, 2D or 3D array/list of frequencies (float) 
                   2D if every group uses different frequencies
                   3D if every dataset and every group uses different freqs 
                   Dimensions must match the data!
    xlimA:         2D array of [xmin,xmax] for each subplot
    ylabA:         2D array of labels for the y axis in each subplot
    titlesA:       2D array of subplots titles    
    cmapA:         array of colormaps, either names of standard colormaps 
                   or instances of matplotlib.colors.LinearSegmentedColormap
    legendA:       2D array of legends (strings); or None for no legends 
    outfile:       if not None, figure will be exported to this file
    legend_posA:   position of the legend ('uc' or 'ur') in each subplot; 
                   or None for no legends
    ylimA:         2D array of [ymin,ymax] for each subplot; or None for auto
    show:          if False, the figure is not shown in console/window

    ncols:         number of columns in the plot legend
    CI:            None if no CI/SDs, else alpha value (0 to 1) for CI/SD areas 
    xticks:        custom values for xticks. If auto, standard values are used
    xlabA:         array of booleans, whether to show the x label; 
                   or None for all True
    Ryt:           if not None, reduces the number of y ticks
    fontsize:      fontsize in plot
    markersize:    size of markers in plot
    '''
    
    depth = lambda L: isinstance(L, list) and max(map(depth, L))+1
    fig,axes=plt.subplots(rows,cols,figsize=figsize)
    if CI == None:
        CI = [None for i in range(len(dataL))]  
    if ylimA==None:
        ylimA = [False for i in range(len(dataL))]
    if xlabA==None:
        xlabA = [True for i in range(len(dataL))]
    if legend_posA==None:
        legend_posA = [None for i in range(len(dataL))]  


    for d,data in enumerate(dataL):         # each dataset in one subplot 
        if (rows==1) or (cols ==1):
            ax = axes[d]
        else:
            ax = axes[d/cols,d%cols]
        ax.hold(True)
        ax.set_title(titlesA[d],fontsize=fontsize)
        
        if type(cmapA[d]) is mpc.LinearSegmentedColormap:
            colorspace = [cmapA[d](i) for i in np.linspace(0, 1, len(data))]
        else:
            colorspace = [plt.get_cmap(cmapA[d])(i) for i in np.linspace(0, 1, len(data))]
        if CI[d]!= None:
            colorspace_CI = np.array(colorspace)*np.array([1,1,1,CI[d]])
        ax.set_color_cycle(colorspace)
        
        for i in range(len(data)):
            if depth(freqs)==2:
                freqs2=freqs[d][i]
            elif depth(freqs)==1:    
                freqs2=freqs[i]
            else:
                freqs2=freqs
                
            if CI[d]!=None:
                fr = freqs2[:len(data[i][0])]
                ax.plot(fr,data[i][0],'o-',markersize=markersize)
                ax.fill_between(fr,data[i][1],data[i][2],color=colorspace_CI[i])    
            else:
                fr = freqs2[:len(data[i])]
                ax.plot(fr,data[i],'o-',markersize=markersize,color=colorspace[i])
        if Ryt != None:
            if Ryt[d] ==1:
                for label in ax.get_yticklabels()[::2]:
                    label.set_visible(False)  
                 
        ax.set_xscale('log')
        ax.get_xaxis().set_major_formatter(mp.ticker.ScalarFormatter()) 
        if xticks=='auto':
            xticks=[1,2,3,5,10,20,30,50,100,200,300]
        ax.set_xticks(xticks)
        xticklabels = [str(i) for i in xticks]
        ax.set_xticklabels(xticklabels,fontsize=fontsize-2)
        if xlabA[d]==True:
            ax.set_xlabel('Frequency [Hz]',fontsize=fontsize)       
        ax.set_xlim(xlimA[d]) 
        ax.set_ylabel(ylabA[d],fontsize=fontsize)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if ylimA[d]!=None:
            ax.set_ylim(ylimA[d]) 
            
        loc_dict = {'uc': 'upper center', 'ur': 'upper center', 
                'br': 'lower right', 'best': 'best'}  
        if legend_posA[d] !=None:
            if legendA[d] != None:
                ax.legend(legendA[d],loc=loc_dict.get(legend_posA[d]), 
                          bbox_to_anchor=(0.5, 1.05), ncol=ncols)
        if legend_posA[d]=='ur':
            if legendA[d] != None:
                ax.legend(legendA[d], loc='upper right', ncol=ncols, frameon=0,fontsize=fontsize)  
        if legend_posA[d]=='ul':
            if legendA[d] != None:
                ax.legend(legendA[d],loc='upper left', ncol=ncols, frameon=0,fontsize=fontsize)
    plt.tight_layout()   
    if outfile!=None:    
        plt.savefig(outfile)
    if show:
        plt.show()
  
    plt.clf()
    


def plot_histogram(data, N_bins=20, width=0.7):
    import matplotlib.pyplot as plt
    import numpy as np
    
    hist, bins = np.histogram(data, bins=N_bins)
    width = width * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()
    
    
    
    
    
    



def simple_CF_plot(data,figsize,xname,yname,xtix,ytix,xlabels,ylabels,zmax=1,ztix=[0,0.2,0.4,0.6,0.8,1],
                   outfile=None,cmap='std',zmin=None,fontsize=10):            
    
    #data in shape freq x ratio
    # example of use: 
    # LF_indices = [0,4,8,12,16,20,24,28,32,36,40]
    # LF_str     = ['1.1', '2.2', '3.7', '5.9', '9.0', '13.1', '19.7', '28.7', '42.5', '65.3', '95.6']
    # plots.simple_CF_plot(data,figsize,'ratio','LF',np.arange(0.5,5.6,1),LF_indices,ratios,LF_str,zmax=zmax,ztix=ztix,outfile=None)             


    #eps  = np.spacing(0.0)                                            # an epsilon
    if zmin==None:
        vmin = 1e-20
    else:
        vmin = zmin
    if cmap == 'std':
        CM = plt.cm.YlOrRd 
        CM.set_under('None')
    else:
        CM = cmap
    
    fig=plt.figure(figsize=figsize, facecolor='w', edgecolor='k')
    ax=fig.add_axes([0.2,0.2,.7,.7])
    mesh=ax.pcolormesh(data,vmin=vmin,vmax=zmax,cmap=CM)    
    ax.set_xticks(xtix)
    ax.set_xticklabels(xlabels,rotation=45,fontsize=fontsize)
    ax.set_xlim([0,len(data[0])])      
    ax.set_yticks(ytix)
    ax.set_yticklabels(ylabels,fontsize=fontsize)
    ax.set_ylim([0,len(data)])
    ax.set_xlabel(xname,fontsize=fontsize)
    cbar = fig.colorbar(mesh, ticks=ztix)
    cbar.ax.tick_params(axis='y', direction='out',labelsize=fontsize) 
    ax.tick_params(axis='both',which='both',length=0)
    ax.set_ylabel(yname,fontsize=fontsize)    
    if outfile!=None:
        fig.savefig(outfile)




def _repeat_if_needed(param,N_plots,depth):   
    if _depth(param)==0:
        if param==None and depth>0:
            depth=1     
    if _depth(param) >= depth:
        param = param
    else:
        param = [param for n in range(N_plots)]
        param = _repeat_if_needed(param,N_plots,depth)
    return param
    
    
def _depth(obj,c=0):
     try:
         if type(obj) != str:
             obj1 = obj[0]
             c = _depth(obj1,c+1)
         else:
             return c
     except:
         return c
     return c
 
    
