
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:44:03 2017

@author: YuQian
"""

import os
os.chdir('/Users/YuQian/Desktop/Connexin/data_download')
import pandas as pd
import numpy as np

#data processing
df1 = pd.read_csv('LoanStats3a.csv',delimiter=',',skiprows=0,header=1)
df2 = pd.read_csv('LoanStats3b.csv',delimiter=',',skiprows=0,header=1)
data_total = pd.concat([df1,df2])

#drop the useless columns
#data_total = data_total.drop(data_total.columns[53:136],axis=1)
data_total = data_total.drop(['id','member_id','url','last_pymnt_d'],axis=1)

#drop the features for which greater than 10% of the loans were missing data for
num_rows = data_total.count(axis=1)
for i in data_total:
   if len(data_total.columns[i]):
   if num_not_null[i] <= 0.9*(num_rows + 1):
        data_total=data_total.drop(data_total.columns[i])
    
#drop the loans that were missing data for any field
data_total=data_total.dropna(axis=0)


#dealing with categorial features
#data_total_new=data_total['addr_state','application_type','emp_length','grade','home_ownership','initial_list_status','pymnt_plan','sub_grade','term','verification_status'].astype(category)


#dealing with zipcode
zip_data = pd.read_csv('ZIP.csv')
zip_data['Zip']=zip_data['Zip'].astype(str)
for i in zip_data.Zip:
    if len(zip_data.Zip[i])==4:
        zip_data.Zip[i] = '0'+zip_data.Zip[i]
zip_data['Zip'] = (zip_data.Zip.values/100)).astype(int)

zip_data = zip_data.groupby(['Zip']).sum()
zip_data['weighted_median']=zip_data['Median'].apply(lambda x:x)
        
     

zip_data['weighted_mean'] = zip_data.weight*zip_data.Mean_new.astype(float)
zip_data['weighted_median'] = zip_data.weight*zip_data.Median_new.astype(float)
new_zip=zip_data.groupby(['zip_3'])['weighted_mean','weighted_median'].sum()




                  
#label the dataset
