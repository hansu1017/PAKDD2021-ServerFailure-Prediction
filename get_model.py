# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:24:30 2021

@author: hp
"""

import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import warnings 
warnings.filterwarnings("ignore") 
import random
from datetime import timedelta

PARENT_FOLDER = '../data'
kernel_train_path = 'memory_sample_kernel_log_round1_a_train.csv'
failure_train_path = 'memory_sample_failure_tag_round1_a_train.csv'
adress_train_path = 'memory_sample_address_log_round1_a_train.csv'

kernel_test_path = 'memory_sample_kernel_log_round1_b_test.csv'
failure_test_path = 'memory_sample_failure_tag_round1_b_test.csv'
adress_test_path = 'memory_sample_address_log_round1_b_test.csv'

def label(x):
    if x<=60*60*24*7:
        return 1
    else:
        return 0
    
kernel_var = ['1_hwerr_f', '1_hwerr_e', '2_hwerr_c', '2_sel', '3_hwerr_n', '2_hwerr_s', '3_hwerr_m', '1_hwerr_st',
       '1_hw_mem_c', '3_hwerr_p', '2_hwerr_ce', '3_hwerr_as', '1_ke', '2_hwerr_p', '3_hwerr_kp', '1_hwerr_fl', '3_hwerr_r', '_hwerr_cd',
       '3_sup_mce_note', '3_cmci_sub', '3_cmci_det', '3_hwerr_pi', '3_hwerr_o', '3_hwerr_mce_l']

def etl_kernel(path, agg_time,time):
    data = pd.read_csv(os.path.join(PARENT_FOLDER, path))
    data['collect_time'] = pd.to_datetime(data['collect_time']).dt.floor(agg_time)
    data['count'+time] = 1
    
    if 'tag' in data.columns:
        del data['tag']
        del data['failure_time']
    
    group_data = data.groupby(['serial_number','manufacturer','vendor','collect_time'],as_index=False).agg('sum')

    return group_data

def getLabel(label_path, kernel_df):
    failure_tag = pd.read_csv(os.path.join(PARENT_FOLDER,label_path))
    failure_tag['failure_time']= pd.to_datetime(failure_tag['failure_time'])
    merged_kernel = pd.merge(kernel_df,failure_tag[['serial_number',
                                               'manufacturer','vendor','failure_time']],how='left',on=['serial_number',
                                                                                                       'manufacturer','vendor'])
    merged_kernel['diff_seconds'] = (merged_kernel['failure_time'] - merged_kernel['collect_time']).dt.days*24*60*60 \
                                + ((merged_kernel['failure_time']-merged_kernel['collect_time']).dt.seconds)
    merged_kernel['label'] = merged_kernel['diff_seconds'].map(label)
    return merged_kernel

def sta_repeat(x):
    n = 0
    for i in set(x):
        if x.count(i) > 1:
            n += 1
    return n

def sta_max(x):
    ls = []
    for i in set(x):
        ls.append(x.count(i))
    return max(ls)

def get_Feature(address):
    address['collect_time'] = pd.to_datetime(address['collect_time']).dt.floor('2min')
    address_row = address[['serial_number','collect_time','row']].groupby(['serial_number','collect_time'],as_index=False).agg(list)
    address_col = address[['serial_number','collect_time','col']].groupby(['serial_number','collect_time'],as_index=False).agg(list)
    address_row['row_repeat'] = address_row['row'].map(sta_repeat)
    address_col['col_repeat'] = address_col['col'].map(sta_repeat)
    address_row['row_max'] = address_row['row'].map(sta_max)
    address_col['col_max'] = address_col['col'].map(sta_max)
    address_row['row_num'] = address_row['row'].apply(lambda x: len(set(x)))
    address_col['col_num'] = address_col['col'].apply(lambda x: len(set(x)))
    address_sta = pd.merge(address_col,address_row[['serial_number','collect_time','row_num','row_max','row_repeat']],how='left',on=['serial_number','collect_time'])
    return address_sta

failure_tag = pd.read_csv('../data/memory_sample_failure_tag_round1_a_train.csv')
failure_tag['failure_time']= pd.to_datetime(failure_tag['failure_time'])

failure_data = pd.read_csv('../data/memory_sample_failure_tag_round1_b_test.csv')
failure_data['failure_time'] = pd.to_datetime(failure_data['failure_time'])

kernel_train_2min = etl_kernel(kernel_train_path,'2min','2min')
kernel_test_2min = etl_kernel(kernel_test_path,'2min','2min')

mce_a = pd.read_csv("../data/memory_sample_mce_log_round1_a_train.csv")
mce_b = pd.read_csv("../data/memory_sample_mce_log_round1_b_test.csv")

def get_mce_data(df_mce,agg_time,time):
    df = df_mce.copy()
    for i in ["Z", "AP", "G", "F", "BB", "E", "CC", "AF", "AE"]:
        df['mca_'+i+time] = (df.mca_id == i).astype("float")
    for i in [0, 1, 2, 3]:
        df['trans_'+str(i)+time] = (df.transaction == i).astype("float")
    df['collect_time'] = pd.to_datetime(df['collect_time']).dt.floor(agg_time)

    df['count_mce_'+time] = 1
    mce_new = df.groupby(['serial_number','manufacturer','vendor','collect_time'],as_index=False).agg('sum')
    return mce_new

mce_train_2min = get_mce_data(mce_a,'2min','2min')
mce_test_2min = get_mce_data(mce_b,'2min','2min')

train = pd.merge(kernel_train_2min, mce_train_2min,how='left',on=['serial_number','manufacturer','vendor','collect_time'])
test = pd.merge(kernel_test_2min, mce_test_2min,how='left',on=['serial_number','manufacturer','vendor','collect_time'])

kernel_train_2min_sta = kernel_train_2min[['serial_number','count2min']].groupby(['serial_number'],as_index=False).agg(list)
kernel_train_2min_sta['mean_2min'] = kernel_train_2min_sta['count2min'].apply(lambda x: np.mean(x))
kernel_train_2min_sta['median_2min'] = kernel_train_2min_sta['count2min'].apply(lambda x: np.median(x))
kernel_train_2min_sta['sum_2min'] = kernel_train_2min_sta['count2min'].apply(lambda x: sum(x))
kernel_train_2min_sta['max_2min'] = kernel_train_2min_sta['count2min'].apply(lambda x: max(x))
train = pd.merge(train,kernel_train_2min_sta[['serial_number','mean_2min','median_2min','sum_2min','max_2min']],how='left',on=['serial_number'])

kernel_test_2min_sta = kernel_test_2min[['serial_number','count2min']].groupby(['serial_number'],as_index=False).agg(list)
kernel_test_2min_sta['mean_2min'] = kernel_test_2min_sta['count2min'].apply(lambda x: np.mean(x))
kernel_test_2min_sta['median_2min'] = kernel_test_2min_sta['count2min'].apply(lambda x: np.median(x))
kernel_test_2min_sta['sum_2min'] = kernel_test_2min_sta['count2min'].apply(lambda x: sum(x))
kernel_test_2min_sta['max_2min'] = kernel_test_2min_sta['count2min'].apply(lambda x: max(x))
test = pd.merge(test,kernel_test_2min_sta[['serial_number','mean_2min','median_2min','sum_2min','max_2min']],how='left',on=['serial_number'])

address_a = pd.read_csv('../data/memory_sample_address_log_round1_a_test.csv')
address_b = pd.read_csv('../data/memory_sample_address_log_round1_b_test.csv')
address_sta_train = get_Feature(address_a)
address_sta_test = get_Feature(address_b)

train_new = pd.merge(train,address_sta_train[['serial_number', 'collect_time',
       'row_num', 'row_max', 'row_repeat', 'col_num', 'col_max', 'col_repeat',
       ]],how='left',on = ['serial_number','collect_time']) 
test_new = pd.merge(test,address_sta_test[['serial_number', 'collect_time',
       'row_num', 'row_max', 'row_repeat', 'col_num', 'col_max', 'col_repeat',
       ]],how='left',on = ['serial_number','collect_time']) 

train_new = getLabel(failure_train_path, train_new)

feats = ['manufacturer', 'vendor', '1_hwerr_f','1_hwerr_e','2_hwerr_c', '3_hwerr_as','1_ke','3_hwerr_kp','3_sup_mce_note','2_hwerr_p', 'count2min',
       'mca_Z2min', 'mca_AP2min', 'mca_G2min', 'mca_BB2min',
       'mca_E2min', 'mca_CC2min', 'mca_AF2min',  'trans_02min',
       'trans_12min', 'trans_22min', 'trans_32min', 'count_mce_2min',
       'mean_2min', 'median_2min', 'sum_2min',
       'row_num', 'row_max', 'row_repeat', 'col_num', 'col_max', 'col_repeat',
       ]

test_new = pd.merge(test_new,failure_data[['serial_number',
                                               'manufacturer','vendor','failure_time']],how='left',on=['serial_number','manufacturer','vendor'])
test_new['diff_seconds'] = (test_new['failure_time'] - test_new['collect_time']).dt.days*24*60*60 + ((test_new['failure_time']-test_new['collect_time']).dt.seconds)
test_new['label'] = test_new['diff_seconds'].map(label)

test_7 = test_new[(test_new['collect_time']>=pd.to_datetime('20190701'))&(test_new['collect_time']<pd.to_datetime('20190722'))]
train_concat = pd.concat([train_new,test_7])

import catboost as cat
clf_cat = cat.CatBoostClassifier(learning_rate=0.01,iterations=1000,depth=6
                                 ,colsample_bylevel = 0.8
                                )

clf_r_cat = cat.CatBoostRegressor(learning_rate=0.01,iterations=1000,depth=6,loss_function ='MAPE'
                                  ,colsample_bylevel = 0.8
                                 )

cat_features = ['manufacturer','vendor']
sample_0 = train_concat[train_concat['label']==0].sample(len(train_concat[train_concat['label']==1])*10,random_state=2)
sample = sample_0.append(train_concat[train_concat['label']==1])

sample['vendor'] = sample['vendor'].apply(lambda x: int(x))
clf_cat.fit(sample[feats],sample['label'],verbose=False,cat_features = cat_features)
fail_7 = sample[sample['label']==1]
fail_7['diff_min'] = np.floor(fail_7['diff_seconds']/60)+2
clf_r_cat.fit(fail_7[feats],fail_7['diff_min'],verbose=False)

import pickle
with open('p1.pickle','wb')as f: 
    pickle.dump(clf_cat,f)
with open('p2.pickle','wb')as f: 
    pickle.dump(clf_r_cat,f)

