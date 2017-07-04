#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:17:07 2017

@author: zengxiang
"""
import pandas as pd
def data_summary(df_x_org:pd.DataFrame):
    variable_type = df_x_org.dtypes
    Missing_count = df_x_org.isnull().sum()/len(df_x_org)
    data_info = pd.DataFrame(dict(type = variable_type,missing = Missing_count))
    print(data_info)

def outlier_remove(series,scalar: int = 3):
    if len(series) == 0:
        return Null
    
    if len(series)  ==1:
        return [True]
    mean = series.mean()
    oldstd = 0
    newstd = series.std()
    index = (series-mean).abs()<=scalar*newstd
    while oldstd!=newstd:
        oldstd = newstd
        newstd = series[index].std()
        mean = series[index].mean()
        index = (series-mean).abs()<=scalar*newstd
    return index