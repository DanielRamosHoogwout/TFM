# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:46:35 2022

@author: Daniel

This script adds yahoo finance index
"""

import pandas as pd

df1 = pd.read_csv("C:/Users/Daniel/Desktop/TFM/Datos/BTC_test_mar2020.csv")
df2 = pd.read_csv("C:/Users/Daniel/Desktop/TFM/Datos/IBEX_test_mar2020.csv")

#Only need the open date of both

df1 = df1[["Date", "Open"]]
df2 = df2[["Date", "Open"]]

left_join = pd.merge(df1, 
                      df2, 
                      on ='Date', 
                      how ='left')
left_join = left_join.rename(columns={"Open_x": "BTC", "Open_y": "IBEX"})

df_final = left_join.fillna(method="ffill")

df_final.to_csv('C:/Users/Daniel/Desktop/TFM/Datos/BTC_IBEX_test_mar2020.csv')
