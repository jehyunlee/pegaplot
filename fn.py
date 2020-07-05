#!/usr/bin/env python
# coding: utf-8
# ver. 2020.04.07.
# Jehyun Lee (jehyun.lee@gmail.com)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import os, copy, sys, io, json
import numpy as np
import pandas as pd

from functools import reduce
from IPython.core.display import HTML

from fontkr import add_FONTKR  

#- print settings
def show(df):
    display(HTML(df.to_html()))


# check type of a variable
def chk_type(var, typename):
    """
    check type of variable
    
    Parameters
    ------------------------
    var : variable
    typename : (str)type name 
               'array' is regarded as 'list' as well.    
    
    
    Returns
    ------------------------
    True, if type of variable is typename
    False, if not.    
               
    """
    typename_tol = ['nullnullnull']   # dummy letters
    typename_tol_arraylike = ['array', 'list', 'series']
    typename_tol_scalar = ['float', 'int']
    
    # 'list' and 'array' are regarded as same type
    if typename == 'list': typename_tol = typename_tol_arraylike
    if typename == 'array': typename_tol = typename_tol_arraylike
    if typename == 'array-like': typename_tol = typename_tol_arraylike
    if typename == 'scalar': typename_tol = typename_tol_scalar
    
    typename_tol.append(typename)
    
    if any([(typetol in str(type(var))) for typetol in typename_tol]):
        return True
    else:
        return False


# check numbers of x and y.
def chk_len(x, y):
    """
    check if lengths of two array-like variable are same.
    
    Parameters
    ------------------------
    x, y : (list or array)
    
    
    Returns
    ------------------------
    None. 
    pass if lengths of x and y are same, otherwise AssertError arises.
    
    """
    assert chk_type(x, 'list')
    assert chk_type(y, 'list')
    
    lenx, leny = len(x), len(y)
    assert lenx == leny


# check null values
def chk_nan(*args):
    """
    returns null-free dataset.
    
    Parameters
    ------------------------
    *args : array-like data (list, numpy.ndarray, pd.Series)
    
    
    Returns
    ------------------------
    data without null rows, if any.
    
    """
    data_names = []
    data = []
    data_nanidx = []
    data_new = []
    
    for i, arg in enumerate(args):
        # data should be array-like
        assert chk_type(arg, 'list') or chk_type(arg, 'array') or chk_type(arg, 'series'), "Only array-like data is acceptable"
        
        # data names
        if chk_type(arg, 'series'):
            data_names.append(arg.name)
        else: 
            data_names.append(f'data_{i}')
        
        # data
        data.append(np.array(arg))
            
        # null values
        data_nanidx.append(np.where(np.isnan(arg))[0])
        if len(data_nanidx[i]) > 0:
            print(f'# WARNING {data_names[i]}: Number of null values={len(data_nanidx[i])} of {len(data[i])}.')

    # union indices of null values 
    data_nanidx_all = reduce(lambda a, b: list(set(a) | set(b)), data_nanidx)
    print(f'# Total number of missing data: {len(data_nanidx_all)}')
    
    for i, datum in enumerate(data):
        data_new.append(np.delete(data[i], data_nanidx_all))
    
    return data_new
    

# compare dataframes' relation 
def cmp_dfs(df1, df2, key1, key2):
    """
    check type of variable
    
    Parameters
    ------------------------
    df1, df2   : (pandas.DataFrame)
    key1, key2 : (str) key column name
    
    
    Returns
    ------------------------
    indices of (df1-df2), (df2-df1), (df1 & df2), (df1 | df2)
    
    """
    assert key1 in df1.columns
    assert key2 in df1.columns
    
    dfkey1 = df1[key1].tolist()
    dfkey2 = df2[key2].tolist()

    df1_2 = list(set(dfkey1) - set(dfkey2))
    df2_1 = list(set(dfkey2) - set(dfkey1))
    df1n2 = list(set(dfkey1) & set(dfkey2))
    df1u2 = list(set(dfkey1) | set(dfkey2))
    
    print(f'# No. of df1= {df1.shape[0]}, df2= {df2.shape[0]}')
    print(f'- No. of [df1 - df2] = {len(df1_2)}')
    print(f'- No. of [df2 - df1] = {len(df2_1)}')
    print(f'- No. of [df1 & df2] = {len(df1n2)}')
    print(f'- No. of [df1 | df2] = {len(df1u2)}')
    
    return df1_2, df2_1, df1n2, df1u2


# labels preparation
def set_labels(df, columns, labels):
    """
    builds a list of labels for given dataframe.
    
    Parameters
    ------------------------
    df : (pandas DataFrame)
    columns : (list of str) variable names
    labels : (dict) {column: label}
    
    
    Returns
    ------------------------
    (list) labels for given columns
    
    """    
    assert chk_type(columns, 'list')
    assert chk_type(labels, 'dict')
    
    # if columns not None
    if labels == None:
        labels = dict(zip(df.columns, df.columns))
    
    labels_not_exist = []
    for column in columns:
        if column.lower() not in labels.keys():
            labels_not_exist.append(column)
            labels[column.lower()] = column
        
    labels_str = [labels[column.lower()] for column in columns]
    if len(labels_not_exist) > 0:
        print(f'# following columns do not have label: {labels_not_exist}')
    
    return labels_str


# setting limitations of x and y variables, with given margin
def set_xylim(x, y, margin=0.05, compare=False):
    """
    x and y limitations of given data
    
    Parameters
    ------------------------
    x, y    : (numpy.ndarray)
    margin  : (float) x-, y-directional margin
    compare : (Boolean) 1:1 comparison between x and y
    
    Returns
    ------------------------
    (float, float, float, float): xmin, xmax, ymin, ymax
    
    """
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    xmin = x.min() - x_range*margin
    xmax = x.max() + x_range*margin
    ymin = y.min() - y_range*margin
    ymax = y.max() + y_range*margin
    
    if compare == True:
        minval = min(xmin, ymin)
        maxval = max(xmax, ymax)
        xmin, xmax, ymin, ymax = minval, maxval, minval, maxval
    
    return xmin, xmax, ymin, ymax
    

# 1st order fitting
def fit_1D(x, y, xmin=None, xmax=None, ymin=None, ymax=None):
    """
    linear fitting for given (x, y)
    
    Parameters
    ------------------------
    x, y : array-like data (list, numpy.ndarray, pd.Series). shape (M,)
    xmin, xmax, ymin, ymax : (float) range of fittings
   
    
    Returns
    ------------------------
    p     : (numpy.ndarray) shape (2)
    cov   : (numpy.ndarray) shape (M, M)
    y_fit : (numpy.ndarray) shape (M)
    x_regminmax, y_regminmax : (numpy.ndarray) fitted line (x, y)
    xmin, xmax, ymin, ymax   : (float) range of fittings
    
    """
    x, y = chk_nan(x, y)
    p, cov = np.polyfit(x, y, 1, cov=True)   # parameters and covariance from of the fit of 1-D polynom.
    y_fit = np.polyval(p, x)                 # model using the fit parameters; NOTE: parameters here are coefficients
    
    if xmin == None: 
        xmin = x.min()
    if xmax == None:
        xmax = x.max()
    if ymin == None:
        ymin = y.min()
    if ymax == None:
        ymax = y.max()
    
    x_regminmax = np.linspace(xmin, xmax, 100)
    y_regminmax = np.polyval(p, x_regminmax)
    ymax = max(ymax, y_regminmax.max())
    ymin = max(ymin, y_regminmax.min())
    
    return p, cov, y_fit, x_regminmax, y_regminmax, xmin, xmax, ymin, ymax


# get x and y ranges of interest
# https://jehyunlee.github.io/2020/04/30/Python-DS-12-auto_focus/
def get_focus(x, y, res=1000, yth=1, xmargin=0.1, ymargin=0.1):
    """
    automatically adjust range to of interest.
    
    Parameters
    ------------------------
    x, y       : (numpy.ndarray)
    res        : (int) default=1000.
                 resolution of y range
    yth        : (int) default=1.
                 select data bins if its size equals to or larger than yth
    x(y)margin : (float) default= 0.1 (10%).
                 expand x(y)-directional limits to x(y) range * xmargin   
    
    Returns
    ------------------------
    xmin, xmax, ymin, ymax : (floats)
    
    """
    ymin, ymax = y.min(), y.max()
    xmin, xmax = x.min(), x.max()
    
    yhist, ybins = np.histogram(y, bins=res)
    
    yfoc = ybins[np.where(yhist > yth)[0]]
    
    if len(yfoc) >= 2:
        ymin, ymax = yfoc.min(), yfoc.max() 
        xfoc = x[(y>=ymin) & (y<=ymax)]    # y 범위를 만족하는 x만 선택합니다.
        xmin, xmax = min(xfoc), max(xfoc)
        
    xrange = xmax-xmin    
    yrange = ymax-ymin
    
    xmin -=  xrange*xmargin
    xmax += xrange*xmargin
    ymin -= yrange*ymargin
    ymax += yrange*ymargin

    return xmin, xmax, ymin, ymax