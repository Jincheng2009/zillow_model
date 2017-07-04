#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:47:30 2017

@author: zengxiang
"""
import pandas as pd
import numpy as np
import timeit
from function import outlier_remove
import sys
def derive_variable(df_x,df_y,month: list=[10,11,12],mile: int = 1):
    position_unique = df_x[['latitude','longitude','parcelid']]
    position_unique=position_unique.drop_duplicates()
    position_unique["latindex"] =  np.floor(position_unique["latitude"]/(mile*15000))
    position_unique["lonindex"] = np.floor(position_unique["longitude"]/(mile*17000))
    position_unique.sort_values(by = ["latindex","lonindex"],inplace = True)
    position_unique["rownum"] = range(0,len(position_unique.parcelid))
    position_unique.index = range(0,len(position_unique.parcelid))
    group = position_unique.groupby(["latindex","lonindex"])
    position_index_table=pd.DataFrame(data=dict(latindex=group.latindex.first(), lonindex=group.lonindex.first(),min=group.rownum.min(),max=group.rownum.max()))
    position_index_table.index = range(0,len(position_index_table.latindex))
    
    transaction_data = pd.merge(left = df_x,right =df_y,on = 'parcelid',how ='inner')[["latitude","longitude","month","logerror"]]
    transaction_data["latindex"] =  np.floor(transaction_data["latitude"]/(mile*15000))
    transaction_data["lonindex"] = np.floor(transaction_data["longitude"]/(mile*17000))
    transaction_data.sort_values(by = ["latindex","lonindex"],inplace = True)
    transaction_data.index = range(0,len(transaction_data))
    transaction_data["rownum"] = range(0,len(transaction_data.latitude))

    data_return = pd.DataFrame()
    house_total = len(position_unique)
    current_house_num = 0
    last_print_time= timeit.default_timer()
    start_time = timeit.default_timer()
    
    for i in range(0,len(position_index_table)):
        startposition = position_index_table["min"][i]
        endposition = position_index_table["max"][i]+1
        latindex = position_index_table["latindex"][i]
        lonindex = position_index_table["lonindex"][i]
    
        current_area=position_unique.iloc[startposition:endposition]
        current_area.index = range(0,len(current_area))
        
        
        
        position_index_table_index=(position_index_table["latindex"]>=latindex-1)&(position_index_table["latindex"]<=latindex+1) \
        &((position_index_table["lonindex"]>=lonindex-1))&(position_index_table["lonindex"]<=lonindex+1)
        position_index_table_neighbor =  position_index_table[position_index_table_index]
    
        position_neighbor = position_unique[0:0]
        position_index_table_neighbor.index = range(0,len(position_index_table_neighbor))
        for i in range(0,len(position_index_table_neighbor)):
            startrow = position_index_table_neighbor["min"][i]
            endrow = position_index_table_neighbor["max"][i]+1
            position_neighbor = pd.concat([position_neighbor,position_unique.iloc[startrow:endrow]],ignore_index =True)
            
        transaction_neighbor =  transaction_data[((transaction_data["latindex"]-latindex).abs()<=1)& \
                                                 ((transaction_data["lonindex"]-lonindex).abs()<=1)]
        transaction_neighbor.index = range(0,len(transaction_neighbor.index))
        
        for j in range(0,len(current_area)):
            time_now = timeit.default_timer()
            if time_now - last_print_time >= 180:
                print("running time:%d mins" % ((time_now-start_time)/60 ))
                print("Complete number:%d/%d" % (current_house_num,house_total))
                print("estimated remaining time: %d mins"  % ((time_now-start_time)/current_house_num*(house_total-current_house_num)/60 ))
                last_print_time = time_now
                sys.stdout.flush()

                
            current_house = current_area.iloc[j]
            lat = current_house["latitude"]
            lon = current_house["longitude"]
            position_table =  position_neighbor[((position_neighbor["latitude"]-lat).abs()<=mile*15000)& \
                                                ((position_neighbor["longitude"]-lon).abs()<=mile*17000)]            
            transaction_table =  transaction_neighbor[((transaction_neighbor["latitude"]-lat).abs()<=mile*15000)& \
                                                      ((transaction_neighbor["longitude"]-lon).abs()<=mile*17000)]
            
            data_rowi = dict()
            neighbor_num = len(position_table)
            obs = len(transaction_table)
            if obs<=10:
                year_logerror = 0
            else:
                index = outlier_remove(transaction_table.logerror)
                obs = sum(index)
                if obs<=5:
                    year_logerror = 0
                else:
                    year_logerror = transaction_table.logerror[index].mean()
    

            data_rowi.update({"percelid":current_house.parcelid,"neighbor_num":neighbor_num,"year_logerror":year_logerror})


            for monthi in month:
                tran_table_month = transaction_neighbor[transaction_neighbor.month==monthi]
                freq_num= len(tran_table_month)
                if freq_num<=6:
                    month_logerror = 0
                else:
                    index = outlier_remove(tran_table_month.logerror)
                    obs = sum(index)
                    if obs<=4:
                        month_logerror = 0
                    else:
                        month_logerror = tran_table_month.logerror[index].mean()
                data_rowi.update({"freq_num":freq_num,"month":monthi,"month_logerror":month_logerror})
                data_return=data_return.append(data_rowi, ignore_index =True)
            current_house_num +=1
    return data_return