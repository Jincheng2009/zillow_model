#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:08:43 2017

@author: zengxiang
"""
import function 
import pandas as pd
import numpy as np
from derive_variable import derive_variable

df_x_path = "./data/properties_2016.csv"
df_y_path = "./data/train_2016.csv"

df_x_org = pd.read_csv(df_x_path)
df_y_org = pd.read_csv(df_y_path)
function.data_summary(df_x_org)


df_y = df_y_org
df_y["month"]= pd.DatetimeIndex(df_y_org["transactiondate"]).month
del df_y["transactiondate"]
df_y["month"] = df_y["month"].astype('category')


Na_remove = ['parcelid', 'longitude','latitude']
Na_to_mean=[]
Na_to_0=['garagetotalsqft','poolcnt','numberofstories']

df_x = df_x_org[Na_remove+ Na_to_mean+ Na_to_0].copy()
df_x.dropna(subset = Na_remove,inplace = True)
df_x[Na_to_0] = df_x[Na_to_0].fillna(0)
function.data_summary(df_x)

derive_variable(df_x,df_y,[10,11,12])