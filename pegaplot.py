#!/usr/bin/env python
# coding: utf-8
# ver. 2020.04.01.
# Jehyun Lee (jehyun.lee@gmail.com)

"""
Pegaplot: A Visulization module wrapping Matplotlib and Seaborn

Default Settings
----------------
STYLE      : style for plots.
             plt.style.use(STYLE) is called.
             default = "seaborn-whitegrid" of matplotlib, equivalent to "whitegrid" of seaborn
CONTEXT    : set of font sizes.
             default = "talk" of seaborn
COLOR      : plot colors.             
             default = "green"
PALETTE    : sequence of plot colors.
             sns.set_palette(PALETTE) is called.
             default = "bright" of seaborn.

FONTMATH   : font for mathmatic expression, ex. MathJax.
             default = "cm"

FONTKR     : font for Korean characters
             default = "NanumGothic"

NEGSUPFONT : alternative font for superscription, due to the missing U+2212 in Korean fonts.
             default = "Liberation Sans"

DPI        : quality of images on screen.
             default = 72
PRINTDPI   : quality of image files.
             default = 200
"""

### Import libraries

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import os, copy, sys, io, json
import numpy as np
import pandas as pd
import scipy.stats as stats

from IPython.core.display import HTML

from fontkr import add_FONTKR, set_FONTKR
from fn import show, chk_type, chk_len, chk_nan, cmp_dfs, set_labels, set_xylim, fit_1D

get_ipython().run_line_magic("matplotlib", "inline")


### Default settings

with open('./config/settings.json', 'r') as f:
    settings = json.load(f)

params = settings['params']    
params_def = copy.deepcopy(params)


### Korean Font Setting

FONTKR = set_FONTKR(params['fontkr'])    # set Korean font from installed fonts
add_FONTKR(FONTKR)                       # add Korean font to matplotlib


### Plot Style Setting

def set(**kwargs):
    global params
    
    # 1. parameter settings 
    flag_any = 0
    
    if len(kwargs.keys()) == 0:    # recover defaults
        print('# Setting Default Values:')
        params = copy.deepcopy(params_def)
        for key, value in params.items():
            print(f'  {key:10s} = {value}')
        
    else:    # edit parameter as required
        for key, value in kwargs.items():
            if key in params.keys():
                flag_any += 1
                params[key] = value
                print(f'params[{key}]={value}')
            if key == 'palette':
                with sns.axes_style("darkgrid"):
                    sns.palplot(sns.color_palette(params['palette'])) # show palette
        
        if flag_any == 0:
            sys.exit(f'NotValidKey: "{key}" is not a member of param')
    
    # 2. apply settings
    plt.style.use(params['style'])        # style
    sns.set_palette(params['palette'])    # palette
    sns.set_context(params['context'])    # context
    plt.rcParams['mathtext.fontset'] = params['fontmath']  # math font
    plt.rcParams['figure.figsize'] = params['figsize']     # figure size
    plt.rcParams['figure.dpi'] = params['showdpi']  # image dpi on notebook
    plt.rcParams['savefig.dpi'] = params['filedpi'] # image dpi on savefile
    
    # 3. setting for Korean font
    add_FONTKR(FONTKR)

set()    


### Font Settings

fonts = settings['fonts']

# fontdicts
font_label = fonts['label']     # xlabel and ylabel
font_title = fonts['title']     # title
font_negsup = fonts['negsub']   # negative(-) on superscript 
font_math = fonts['math']       # math
font_text = fonts['text']       # text

# fontproperties:suptitle
font_suptitle = mpl.font_manager.FontProperties()
font_suptitle_ = fonts['suptitle']
font_suptitle.set_family(font_suptitle_['family'])
font_suptitle.set_size(font_suptitle_['fontsize'])
font_suptitle.set_weight(font_suptitle_['fontweight'])


### Plot Settings
#- subplots
def subplots(ncols=1, nrows=1, 
             figsize=None, 
             axsize=None,
             sharex=False, 
             sharey=False,
             **kwargs):
    """
    plot: create a figure and a set of subplots.
    
    Parameters
    ------------------------
    nrows, ncols   : (int) default=1
                     number of rows, columns of the subplots.

    figsize        : (float, float) default=None.
                     width, height in inches.
                     figsize overrides axsize
    axsize         : (float, float) default=None.
                     unlike figsize of matplotlib.pyplot.figure, the figsize is applied for EACH subplots.
                     if figsize==None and axsize==None, axsize = params['figsize']

    sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default=False
                     When subplots have a shared x-axis along a column, only the x tick labels of the bottom subplot are created. Similarly, when subplots have a shared y-axis along a row, only the y tick labels of the first column subplot are created. To later turn other subplots' ticklabels on, use tick_params.
    
    
    Returns
    ------------------------
    Figure                : matplotlib.figure.Figure
    Axes or array of Axes : matplotlib.axes.Axes
    
    """
    if figsize==None:
        if axsize == None:
            figsize = params['figsize'] 
        else:
            assert chk_type(axsize, 'array-like') and len(axsize)==2
            figsize = [axsize[0]*ncols, axsize[1]*nrows] 
    else:
        assert chk_type(figsize, 'array-like') and len(figsize)==2
        figsize = figsize
        
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, 
                           sharex=sharex, sharey=sharey, **kwargs)
        
    if nrows == 1 and ncols == 1:
        ax = np.array(ax)
    ax = ax.ravel()
        
    for i, axi in enumerate(ax):
        axi.set_xlabel(' ', fontdict=font_label, labelpad=12)
        axi.set_ylabel(' ', fontdict=font_label, labelpad=12)
        axi.set_title(' ', fontdict=font_title, pad=12)
    
    fig.suptitle(' ', y=1.05, fontproperties=font_suptitle)
    
    if nrows == 1 and ncols == 1:
        return fig, ax[0]
    else:
        return fig, ax


#- plot_missings
def missing(df, labels=None, figname=None, figsize=None, color='lightcoral', text_y=None):
    """
    plot: numbers and types of missing data as bar type
    
    Parameters
    ------------------------
    df : (pd.DataFrame)
    labels : (dictionary) default=None: column names
             key: value = column name: label
    figname : (str) default=None: no file output.
               file name to save plot.
    figsize : (float, float) default=None: (ncols*0.5, 10)
              figure size in inch.
    color : (str) default='lightcoral'
            a member of matplots color list.
            https://matplotlib.org/3.2.1/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
    text_y : (float) default=None: 10% of minimum, or 1e-3
             offset of texts on the figure.
    """
    columns = list(df.columns)
    counts = df.count()
    size = df.shape[0]
    types = df.dtypes.tolist()
    
    # text y position
    if text_y == None:
        text_ys = np.array(100 - counts/size*100)
        try:
            text_y = text_ys[np.where(text_ys >0)].min() * 0.1
        except:
            text_y = 1e-3
    
    # labels
    labels_str = set_labels(df, columns, labels)
        
    # figure size
    if figsize == None:
        figsize = (df.shape[1]*0.5, 10)

    # plot
    fig, ax = subplots(figsize=figsize)

    ax.bar(columns, (1-np.array(counts)/size)*100, color=color)

    ax.set_title(f'no. of total data = {size}', fontdict=font_title)
    ax.set_xlim(-0.5, len(columns)-0.5)
    ax.set_yscale('log')
    ax.set_ylim(text_y*0.5,)
    ax.set_xticklabels(labels_str, rotation=45, horizontalalignment='right')
    ax.set_xlabel(' ', fontdict=font_label)
    ax.set_ylabel(' ', fontdict=font_label)
    
    for i, count in enumerate(counts):
        ax.text(i, text_y, f'{size-count} ({(1-count/size)*100:0.2f}%) {types[i]}', 
                fontsize=12, color='k', rotation=90,
               horizontalalignment='center')

    fig.tight_layout()
    
    if figname != None:
        fig.savefig(f'./images/missings_{figname:s}.png')
    
    plt.show()


#- add 1D regression
def add_reg(x, y, ax, margin=0.05, compare=False, ci=True):
    """
    plot: add 1-dimensional polynomial regression
    
    Parameters
    ------------------------
    x, y    : (numpy.ndarray)
    ax      : (matplotlib.pyplot.Axes)
    margin  : (float) default=0.05
              x-, y-directional margin
    compare : (Boolean) default=False
              1:1 comparison between x and y
    ci      : (Boolean) default=True    
              if True, calculate confidential interval
    
    
    Returns
    ------------------------
    (matplotlib.pyplot.Axes) with regression
    
    """
    # set mins and maxes 
    xmin, xmax, ymin, ymax = set_xylim(x, y, margin, compare=compare)    
    
    # 1D fit
    p, cov, y_fit, x_regminmax, y_regminmax, xmin, xmax, ymin, ymax = \
    fit_1D(x, y, xmin, xmax, ymin, ymax)
    
    dof = y.size - p.size                    # degree of freedom
    t = stats.t.ppf(0.975, dof)  # used for CI and PI bands
    
    ax.plot(x_regminmax, y_regminmax, c='b')

    # confidence interval
    if ci == True:
        # error estimation in data/model
        resid = y - y_fit                      
        chi2 = np.sum((resid/y_fit)**2)        # chi-squared; estimates error in data
        chi2_red = chi2/dof                    # reduced chi-squared; measures goodness of fit
        s_err = np.sqrt(np.sum(resid**2)/dof)  # standard deviation of the error

        # plot confidential interval
        confint = t * s_err * np.sqrt(1/y.size + (x_regminmax - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        ax.fill_between(x_regminmax, y_regminmax + confint, y_regminmax - confint, color='#AAAAFF')
        
    return ax


#- plot scatter with histograms
# kde plot : https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python
# ci plot : https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
def scatter(x, y, c=params['color'], 
            xlabel=None, ylabel=None, clabel=None, labels=None, 
            cmap='summer', lut=5, vmin=None, vmax=None, 
            extend=None, cmap_over='magenta', cmap_under='darkgray',           
            s=10, edgecolors='face', alpha=1, 
            kde=False, kde_cmap='Greys', # parameters for kde
            reg=True, ci=True, margin=0.05,      # parameters for regression
            xhist_binnum=20, xhist_linewidth=1, xhist_edgecolors='k', xhist_scale=None,
            yhist_binnum=20, yhist_linewidth=1, yhist_edgecolors='k', yhist_scale=None,
            hist_pad=0.4, figsize=(11, 10), figname=None, compare=False
           ):
    """
    plot: scatter plot with histograms, 2D KDE, and regression.
    
    Parameters
    ------------------------
    x, y    : array-like data (list, numpy.ndarray, pd.Series), shape (n,)
              the data positions.
    c       : (str) default=pegaplot.params['color']
              a member of matplots color list, 
              or array-like data (list, numpy.ndarray, pd.Series)
              https://matplotlib.org/3.2.1/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
              
    xlabel  : (str) default=None.
              label for x-axis
    ylabel  : (str) default=None. 
              label for y-axis
    clabel  : (str) default=None. 
              label for colormap
    labels  : (dict) default=None: xlabel, ylabel, clabel is used.
              labels for input data. if not None, xlabel, ylabel, clabel is overriden.
    
    cmap       : (Colormap) default='summer'.
                 A Colormap instance or registered colormap name. cmap is only used if c is an array of floats.    
    lut        : (int or None) default=5.
                 if name is not already a Colormap instance and lut is not None, the colormap will be resampled to have lut entries in the lookup table.
    vmin, vmax : (scalar) default='None'.
                 lower and upper bound for cmap. If None, minimum and maximum values are set.
    extend     : {'neither', 'both', 'min', 'max'} default=None
                 direction of colorbar extension
                 if not 'neither', automatically sets extend by cmap, vmin and vmax.
    cmap_over  : (str) default='magenta'
                 color for data over 'vmax'.
    cmap_under : (str) default='darkgray'
                 color for data under 'vmin'.
    
    s          : (scalar or array-like with shape (n,))
                 the marker size in points**2. Default is rcParams['lines.markersize'] ** 2.
    edgecolors : (str) default='face'.
                 the edge color of the marker. Possible values:
                 - 'face': The edge color will always be the same as the face color.
                 - 'none': No patch boundary will be drawn.
                 - A Matplotlib color or sequence of color.
    alpha      : (scalar) default=1.
                 the alpha blending value, between 0 (transparent) and 1 (opaque). 
                 
    kde      : (Boolean) default=False.
    kde_cmap : (Colormap) default='Grays'.      
    
    reg      : (Boolean) default=True.
               if True, 1st order regression is displayed.
    ci       : (Boolean) default=True.
               works only if reg == True.
               if True, confidential interval is displayed.
    margin   : (scalar) default=0.05.
               x- and y-directional margin of regression

    x(y)hist_binnum     : (int) default=20.
                          number of bins for histogram of x.                 
    x(y)hist_linewidth  : (scalar) default=1.
                          linewidth for histogram of x.
    x(y)hist_edgecolors : (str) default='k'.
                          a member of matplots color list.
    x(y)hist_scale      : {'linear', 'log', 'symlog', 'logit'} default=None(='linear').
   
    
    Returns
    ------------------------
    None
    
    """    
    # labels
    if chk_type(x, 'series') and xlabel == None:
        if labels == None:
            xlabel = x.name
        else:
            xlabel = labels[x.name]
    if chk_type(y, 'series') and ylabel == None:
        if labels == None:
            ylabel = y.name
        else:
            ylabel = labels[y.name]
    if chk_type(c, 'series') and clabel == None:
        if labels == None:
            clabel = c.name
        else:
            clabel = labels[c.name]
    
    # check null data 
    if chk_type(c, 'list') or chk_type(c, 'series'):
        x, y, c = chk_nan(x, y, c)
    else:
        x, y = chk_nan(x, y)
        
    # canvas preparation
    fig = plt.figure(figsize=figsize)
    
    # grid preparation
    gridsize = (4, 4)
    ax_xhist = plt.subplot2grid(gridsize, (0, 0), colspan=3)
    ax_yhist = plt.subplot2grid(gridsize, (1, 3), rowspan=3)
    ax_main = plt.subplot2grid(gridsize, (1, 0), colspan=3, rowspan=3)
    
    # scatter & kde & regression plot with confidence interval

    #- range
    xmin, xmax, ymin, ymax = set_xylim(x, y, margin=margin, compare=compare)
    
    #- bins for histogram
    xhist_bin = np.linspace(xmin, xmax, xhist_binnum)
    yhist_bin = np.linspace(ymin, ymax, yhist_binnum)
    
    #- classes for stacked histogram
    xdata = []
    ydata = []
    eps = 1e-9
    if lut == None or chk_type(c, 'str'):
        xdata.append(x)
        ydata.append(y)
        
    elif chk_type(c, 'list'):
        if vmax == None:
            vmax = c.max()        
        if vmin == None:
            vmin = c.min()

        if extend != 'neither':
            if (vmax < c.max()) and (vmin > c.min()):
                extend = 'both'
            elif (vmax < c.max()) and (vmin <= c.min()):
                extend = 'max'
            elif (vmax >= c.max()) and (vmin > c.min()):
                extend = 'min'
            else:
                extend = 'neither'
            
        cmins = np.linspace(vmin, vmax + eps, int(lut+1))[:-1]
        cmaxs = np.linspace(vmin, vmax + eps, int(lut+1))[1:]
        clrvals = np.linspace(0, 1, 2*lut + 1)
        xhist_color = []
        yhist_color = []
        
        if chk_type(c, 'series'):
            cval = c.values
        elif chk_type(c, 'list'):
            cval = np.array(c)

        if chk_type(x, 'series'):
            xval = x.values
        elif chk_type(x, 'list'):
            xval = np.array(x)

        if chk_type(y, 'series'):
            yval = y.values
        elif chk_type(y, 'list'):
            yval = np.array(y)
        
        for i in range(len(cmins)):
            idx_range = np.where((cval >= cmins[i]) & (cval < cmaxs[i]))[0]
            xdata.append(x[idx_range])
            ydata.append(y[idx_range])
            xhist_color.append(mpl.cm.get_cmap(cmap)(clrvals[i*2 + 1]))
            yhist_color.append(mpl.cm.get_cmap(cmap)(clrvals[i*2 + 1]))
            

    #- plot scatter
    cmap = mpl.cm.get_cmap(cmap, lut)
    cmap.set_over(cmap_over)
    cmap.set_under(cmap_under)
    main = ax_main.scatter(x, y, 
                           s=s, c=c, edgecolors=edgecolors,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           alpha=alpha)

    if reg == True:
        add_reg(x, y, ax_main, compare=compare, ci=True)
    
    if kde == True:
        # perform the kernel density estimate
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        
        # contourf plot
        # cfset = ax_main.contourf(xx, yy, f, cmap=cmap, vmin=0)
        # contour plot
        cset = ax_main.contour(xx, yy, f, cmap=kde_cmap, linewidths=1, vmin=-0.3)
        # ax_main.clabel(cset, inline=1, fontsize=10)
        
    ax_main.set_xlabel(xlabel, fontdict=font_label)
    ax_main.set_ylabel(ylabel, fontdict=font_label)

    ax_main.set_xlim(xmin, xmax)
    ax_main.set_ylim(ymin, ymax)
    
    #- plot compare
    if compare == True:
        ax_main.plot([xmin, xmax], [ymin, ymax], c='gray')    
    
    # hist plot (x)
    if xhist_color == None:
        if chk_type(c, 'str'):
            xhist_color = c
        elif chk_type(c, 'list') or chk_type(c, 'series'):
            xhist_color = mpl.cm.get_cmap(cmap)(0.5)
            
    ax_xhist.hist(xdata, histtype='barstacked', orientation='vertical', 
                  linewidth=xhist_linewidth, edgecolor=xhist_edgecolors, stacked=True,
                  bins=xhist_bin, color=xhist_color, alpha=max(xhist_alpha, 0.2))
    ax_xhist.set_xticks([])
    if xhist_scale != None:
        ax_xhist.set_yscale(xhist_scale)
    
    # hist plot (y)
    if yhist_color == None:
        if chk_type(c, 'str'):
            yhist_color = c
        elif chk_type(c, 'list') or chk_type(c, 'series'):
            yhist_color = mpl.cm.get_cmap(cmap)(0.5)
            
    ax_yhist.hist(ydata, histtype='barstacked', orientation='horizontal', 
                  linewidth=yhist_linewidth, edgecolor=yhist_edgecolors, stacked=True,
                  bins=yhist_bin, color=yhist_color, alpha=max(yhist_alpha, 0.2))
    ax_yhist.set_yticks([])
    if yhist_scale != None:
        ax_yhist.set_xscale(yhist_scale)
    
    # layout tuning
    fig.tight_layout(h_pad=hist_pad, w_pad=hist_pad)
    
    # colorbar
    if chk_type(c, 'list') or chk_type(c, 'series'):
            ax_main.set_title(f'color: {clabel} ({c.min():1.2f} ~ {c.max():1.2f})', 
                              fontdict=font_label, fontsize=12)       
            cbar = fig.colorbar(main, ax=ax_yhist, extend=extend)
            cbar.set_label(clabel, fontdict=font_label, labelpad=12)
        
    # file save
    if figname != None:
        fig.savefig(f'./images/scatter_{figname:s}.png')
    
    plt.show()
    