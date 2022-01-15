#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import lightgbm as lgb
import cv2
import torch
import pickle
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import ParameterGrid


# In[2]:


import os
from PIL import Image
import imagehash
from tqdm.auto import tqdm


df = pd.read_feather('opt_4ndplace.ftr')

# In[12]:


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

df['fold'] = -1


N_FOLDS = 5
strat_kfold = StratifiedKFold(n_splits=5, random_state=365, shuffle=True)
for i, (_, train_index) in enumerate(strat_kfold.split(df.index, df['Pawpularity'])):
    df.loc[train_index, 'fold'] = i


# In[13]:





COL_FEATURES = ['pred',
 'Breed1_count',
 'Breed1_label',
 'word_39',
 'Blur',
 'act_13',
 'word_83',
 'AdoptionSpeed',
 'vertex_y_sum',
 'word_113',
 'act_18',
 'word_55',
 'word_47',
 'act_37',
 'Eyes',
 'word_44',
 'act_57',
 'word_66',
 'word_105',
 'word_80',
 'act_17',
 'word_52',
 'act_12',
 'word_42',
 'word_38',
 'RescuerID_count',
 'word_110',
 'dominant_score_var',
 'act_62',
 'dominant_blue_mean',
 'word_29',
 'word_31',
 'vertex_x_sum',
 'word_28',
 'act_41',
 'word_87',
 'word_114',
 'word_95',
 'dominant_pixel_frac_mean',
 'word_112',
 'word_102',
 'act_51',
 'word_100',
 'word_12',
 'dominant_pixel_frac_var',
 'word_115',
 'word_106',
 'word_45',
 'word_97',
 'word_68',
 'act_35',
 'word_124',
 'word_22',
 'word_69',
 'word_78',
 'word_70',
 'word_43',
 'word_99',
 'word_109',
 'act_56',
 'Age',
 'act_20',
 'word_75',
 'word_53',
 'dominant_green_sum',
 'dominant_pixel_frac_sum',
 'word_25',
 'dominant_score_sum',
 'word_23',
 'word_56',
 'act_24',
 'act_6',
 'word_34',
 'word_92',
 'word_64',
 'word_62',
 'word_54',
 'word_9',
 'word_50',
 'word_107',
 'word_86',
 'word_111',
 'act_43',
 'word_94',
 'act_22',
 'vertex_y_mean',
 'word_49',
 'act_47',
 'word_93',
 'word_26',
 'act_54',
 'word_101',
 'word_81',
 'word_48',
 'word_65',
 'word_119',
 'act_0',
 'label_score_mean',
 'word_16',
 'word_20',
 'word_46',
 'word_74',
 'word_79',
 'word_15',
 'word_90',
 'word_2',
 'word_98',
 'word_103',
 'word_21',
 'word_33',
 'act_55',
 'act_33',
 'act_63',
 'word_71',
 'word_5',
 'word_108',
 'word_76',
 'act_58',
 'act_34',
 'label_score_sum',
 'act_61',
 'word_117',
 'act_8',
 'vertex_x_mean',
 'act_19',
 'word_41',
 'word_40',
 'word_84',
 'word_116',
 'word_85',
 'dominant_blue_var',
 'word_73',
 'doc_sent_mag',
 'dominant_red_mean',
 'word_91',
 'word_30',
 'word_14',
 'word_37',
 'word_19',
 'word_11',
 'act_44',
 'act_46',
 'act_59',
 'act_23',
 'act_27',
 'word_27',
 'word_96',
 'word_118',
 'act_50',
 'word_32',
 'act_11',
 'FurLength',
 'word_4',
 'word_72',
 'word_82',
 'word_60',
 'word_6',
 'word_36',
 'dominant_score_mean',
 'act_31',
 'word_3',
 'word_88',
 'act_7',
 'word_1',
 'act_15',
 'dominant_blue_sum',
 'dominant_green_mean',
 'word_10',
 'act_10',
 'word_0',
 'word_122',
 'act_16',
 'act_14',
 'act_60',
 'Breed2_label',
 'top_label_description_1_label',
 'word_57',
 'word_58',
 'act_36',
 'word_67',
 'act_53',
 'act_9',
 'word_89',
 'word_123',
 'dominant_green_var',
 'word_51',
 'dominant_red_var',
 'act_29',
 'word_24',
 'act_5',
 'act_4',
 'word_61',
 'act_1',
 'act_48',
 'word_77',
 'vertex_y_var',
 'word_13',
 'act_45',
 'word_59',
 'act_25',
 'word_35',
 'act_28',
 'act_21',
 'word_18',
 'act_42',
 'word_120',
 'act_32',
 'act_49',
 'Gender_label',
 'act_39',
 'doc_sent_score',
 'act_52',
 'word_121',
 'dominant_red_sum',
 'act_26',
 'label_score_var',
 'word_63',
 'act_3',
 'word_17',
 'act_30',
 'vertex_x_var',
 'act_40',
 'State_count',
 'word_104',
 'Breed2_count',
 'Quantity',
 'Color1_label',
 'Color2_label',
 'word_8',
 'Dewormed_Vaccinated_label',
 'Color3_label',
 'Vaccinated_label',
 'bounding_importance_mean',
 'bounding_importance_sum',
 'gdp_vs_population',
 'word_7',
 'Dewormed_label',
 'act_2',
 'Sterilized_label',
 'Type_label',
 'PhotoAmt',
 'Fee',
 'State_label',
 'act_38',
 'bounding_confidence_mean',
 'bounding_confidence_var',
 'bounding_importance_var',
 'no_name_label',
 'bounding_confidence_sum']


import sys

COL_FEATURES = COL_FEATURES[:int(sys.argv[1])]

# In[14]:


def rmse(preds, train_data):
    labels = train_data.get_label()
    #import pdb;pdb.set_trace()
    loss = np.sqrt(((labels - preds.clip(0.01, 1)) ** 2).mean()) * 100
    
    #loss = np.sqrt(((labels - preds.reshape(-1, 100).argmax(axis=1)) ** 2).mean())
    return 'rmse', loss, False


def train(fold, param):
     
    X_train = df.loc[df['fold'] != fold, COL_FEATURES]
    y_train = df.loc[df['fold'] != fold, 'Pawpularity'].values / 100
    
    X_valid = df.loc[df['fold'] == fold, COL_FEATURES]
    y_valid = df.loc[df['fold'] == fold, 'Pawpularity'].values / 100
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    
    model = lgb.train(param,
                      train_data,
                      valid_sets=valid_data,
                      #early_stopping_rounds=50,
                      verbose_eval=100,
                      feval=rmse
                      )
    model.val_data = (model.predict(X_valid), y_valid)
    return model

def train_all(param):
     
    X_train = df.loc[:, COL_FEATURES]
    y_train = df.loc[:, 'Pawpularity'].values  / 100
    
    X_valid = df.loc[:, COL_FEATURES]
    y_valid = df.loc[:, 'Pawpularity'].values  / 100
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    
    model = lgb.train(param,
                      train_data,
                      valid_sets=valid_data,
                      early_stopping_rounds=50,
                      verbose_eval=100,
                      feval=rmse
                      )
    return model


# In[15]:


if 0:
    all_params = {'objective': ['mse'],
                  #'tweedie_variance_power': [1.2],
                 'verbosity': [-1],
                 'boosting_type': ['gbdt'],
                 'feature_pre_filter': [False],
                 'bagging_fraction': [1],
                 'bagging_freq': [1],
                 'num_iterations': [10000],
                 'early_stopping_round': [100],
                 'n_jobs': [16],
                 'seed': [114],
                 'metric':  ['None'],  # trial.suggest_categorical('metric', ['auc', 'binary_logloss', ]), #'auc',
                 'learning_rate': [0.05],
                  'lambda_l1': [0],
                  'lambda_l2': [1],
                  'min_child_samples': [150, 200],
                  'num_leaves': [7],
                  'feature_fraction': [0.8, 0.9, 0.7],
                  'min_gain_to_split': [0.02, 0.01],
                  'linear_tree': [False],
                  #'max_bins': [8, 16, 32, 62, 128, 256, 512]
                 }
else:
    all_params = {'objective': ['mse'],
                  'tweedie_variance_power': [1.2],
                 'verbosity': [-1],
                 'boosting_type': ['gbdt'],
                 'feature_pre_filter': [False],
                 'bagging_fraction': [0.7],
                 'bagging_freq': [1],
                 'num_iterations': [10000],
                 'early_stopping_round': [100],
                 'n_jobs': [16],
                 'seed': [114],
                 'metric':  ['None'],  # trial.suggest_categorical('metric', ['auc', 'binary_logloss', ]), #'auc',
                 'learning_rate': [0.05],
                  'lambda_l1': [0],
                  'lambda_l2': [1],
                  'min_child_samples': [140],
                  'num_leaves': [7],
                  'feature_fraction': [0.7],
                  'min_gain_to_split': [0.02],
                  'linear_tree': [False],
                  #'max_bins': [8, 16, 32, 62, 128, 256, 512]
                 }


# In[16]:


best_score = 1.0e10
best_param = None
models = []
for param in tqdm(ParameterGrid(all_params)):
    print(param)
    list_loss = []
    list_imp = []
    list_num = []
    for fold in range(5):
        model = train(fold, param)
        models.append(model)
        sc = model.best_score['valid_0']['rmse']# * 100
        list_loss.append(sc)
        list_num.append(model.best_iteration)
        
        imp = pd.DataFrame(model.feature_importance(), columns=['imp'])
        imp['col'] = COL_FEATURES
        list_imp.append(imp.set_index('col'))
    sc = np.mean(list_loss)
    if sc < best_score:
        best_score = sc
        best_param = param


# In[17]:


17.122953793907463
17.12237161917507
17.12206798235878
17.035855591812624
17.012832654542915
17.007733147459998
16.980743286686046
16.96459803690594
17.006465753639866
print(best_param)
print('AAA', int(sys.argv[1]), best_score)

