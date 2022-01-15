#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import numpy as np
import pandas as pd
import lightgbm as lgb
import cv2
import torch
import pickle
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import ParameterGrid

#df = pd.read_feather('df_allfeat_merged.ftr')
df = pd.read_feather('df_allfeats_4nd_merged.ftr')


df = df.rename({c : c.replace(' ', '_') for c in 
                ['General Health_main_breed_all',
                 'Affectionate with Family',
                 'Friendly Toward Strangers_main_breed_all']}, axis=1)

# %%


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

df['fold'] = -1


N_FOLDS = 5
strat_kfold = StratifiedKFold(n_splits=5, random_state=365, shuffle=True)
for i, (_, train_index) in enumerate(strat_kfold.split(df.index, df['Pawpularity'])):
    df.loc[train_index, 'fold'] = i


# %%

import sys
COL_FEATURES = ['pred', 'dense_SVD_18', 'gnvec172', 'inception_resnet_174',
       'densenet121_g_svd_18', 'inception_resnet_87', 'dense121_2_112',
       'SVD_CHAR_Name_0', 'Blur', 'SVD_Description_3', 'dense_128',
       'resnet50_SVD_6', 'dense121_128', 'tfidf_g_svd_14', 'gnvec13',
       'Eyes', 'dense121_2_221',
       'diff_sum_Age_groupby_Type_Breed1_Breed2', 'glove_mag174',
       'SVD_CHAR_Name_5', 'SVD_meta_annots_top_desc_30', 'gnvec100',
       'gnvec21', 'meta_color_green_mean_MEAN', 'AdoptionSpeed_y',
       'glove_mag192', 'dense121_2_201', 'count3_bm25_3', 'gnvec238',
       'dense121_2_96', 'ratio_var_Age_groupby_RescuerID_Type',
       'glove_mag20', 'max_Length_annots_top_desc_groupby_RescuerID',
       'dense121_2_75', 'SVD_meta_annots_top_desc_15', 'gnvec90',
       'gnvec115', 'glove_mag117', 'gnvec95', 'dense121_SVD_31',
       'inception_resnet_158', 'resnet_128', 'resnet50_SVD_23',
       'dense121_2_119', 'dense121_2_126', 'glove_mag300',
       'inception_resnet_90', 'dense_SVD_24', 'glove_mag231',
       'glove_mag116', 'SVD_meta_desc_12', 'inception_resnet_319',
       'color_score_amax_first', 'glove_mag177', 'SVD_meta_desc_4',
       'dense121_2_230', 'dense121_2_3', 'inception_resnet_312',
       'inception_resnet_336', 'dense121_2_36', 'dense_SVD_15', 'gnvec69',
       'inception_resnet_223', 'glove_mag139', 'gnvec28',
       'sentiment_entities_tfidf_g_svd_2', 'dense121_2_127',
       'dense121_2_45', 'SVD_meta_annots_top_desc_19', 'dense121_2_48',
       'densenet121_g_svd_30', 'dense121_2_68', 'dense121_2_124',
       'gnvec4', 'color_red_score_var_min', 'inception_resnet_4',
       'gnvec137', 'glove_mag165', 'dense121_2_210', 'resnet50_SVD_30',
       'dense121_2_116', 'gnvec66', 'glove_mag200', 'dense121_2_84',
       'tfidf_g_svd_7', 'densenet121_g_svd_26', 'gnvec58',
       'dense121_2_181', 'dense121_2_248', 'dense121_2_55',
       'SVD_meta_annots_top_desc_27', 'diff_sum_Age_groupby_State',
       'gnvec2', 'inception_resnet_349', 'gnvec72', 'dense121_2_79',
       'inception_resnet_161', 'color_pixel_score_mean_var',
       'diff_var_Quantity_groupby_Type_Breed1', 'inception_resnet_258',
       'gnvec119', 'dense121_2_92', 'dense121_2_24', 'SVD_CHAR_Name_9',
       'dense121_2_225', 'inception_resnet_382', 'gnvec193',
       'dense121_2_253', 'resnet34_SVD_14', 'resnet34_SVD_16',
       'color_pixel_score_first_var', 'dense121_2_108', 'glove_mag227',
       'gnvec120', 'gnvec248', 'glove_mag125', 'dense121_2_159',
       'inception_resnet_54', 'dense121_2_236',
       'color_pixel_score_first_max', 'gnvec109', 'dense121_SVD_8',
       'inception_resnet_279', 'inception_resnet_240', 'SVD_CHAR_Name_14',
       'inception_resnet_30', 'diff_var_Age_groupby_Type_Breed1_Breed2',
       'glove_mag297', 'SVD_CHAR_meta_annots_top_desc_14', 'glove_mag276',
       'meta_desc_tfidf_bm25_3', 'SVD_CHAR_BreedName_full_10',
       'dense121_2_255', 'inception_resnet_315', 'resnet34_SVD_25',
       'inception_resnet_198', 'dense121_2_16', 'glove_mag45',
       'inception_resnet_0', 'count_nmf_4',
       'annots_top_desc_tfidf_g_svd_2', 'meta_label_score_min_MAX',
       'dense121_2_28', 'dense121_2_227', 'dense121_2_115',
       'color_red_score_amin_min', 'tfidf2_nmf_0', 'inception_resnet_91',
       'glove_mag240', 'gnvec233', 'glove_mag135', 'SVD_Description_42',
       'count2_nmf_2', 'Affectionate_with_Family', 'SVD_Description_24',
       'glove_mag52', 'res34_64', 'inception_resnet_187',
       'resnet50_SVD_28', 'dense121_2_215',
       'CountSVD5_BreedName_main_breed_BreedName_main_breed_all_0',
       'sentiment_entities_tfidf_g_svd_9', 'meta_color_red_mean_STD',
       'meta_label_score_max_STD', 'densenet121_g_svd_9', 'd2v_min',
       'meta_label_score_mean_MAX',
       'ratio_var_Quantity_groupby_Type_Breed1', 'SVD_Description_67',
       'gnvec231', 'ratio_sum_MaturitySize_groupby_Type_Breed1_Breed2',
       'mean_FurLength_k_groupby_RescuerID', 'inception_resnet_242',
       'SVD_meta_annots_top_desc_13',
       'mean_Quantity_groupby_Type_Breed1_Breed2', 'AdoptionSpeed_x',
       'dense121_SVD_7', 'inception_resnet_88', 'dense121_2_233',
       'dense121_2_218', 'inception_resnet_86',
       'meta_label_score_max_MAX', 'inception_resnet_353',
       'var_Sterilized_groupby_RescuerID_Type',
       'annots_score_normal_amin_var', 'dense121_2_232', 'gnvec45',
       'diff_var_Quantity_groupby_Type_Breed1_Breed2', 'gnvec113',
       'RESCUER_PhotoAmt_STD', 'inception_resnet_338', 'glove_mag163',
       'dense121_SVD_19', 'dense121_2_245', 'inception_resnet_262',
       'glove_mag284', 'annots_score_pick_mean_max', 'dense121_2_244',
       'Friendly_Toward_Strangers_main_breed_all',
       'mean_Fee_groupby_Type_Breed1', 'CountLDA5_Breed1_Dewormed_3',
       'General_Health_main_breed_all']

COL_FEATURES += [c for c in ['Breed1_count',
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
 'word_29'] if c not in COL_FEATURES] 

for c in ['sentiment_entities_tfidf_g_svd_2',
          'SVD_CHAR_Name_5',
          'SVD_Description_67',
          'dense121_2_230',
         'inception_resnet_258',
          'gnvec238',
          'dense121_2_36',
          'dense121_128',
          'resnet50_SVD_6',
          'dense121_2_221',
          'dense121_2_218',
         ]:
    COL_FEATURES.remove(c)

try:
    removed_col = COL_FEATURES.pop(int(sys.argv[1]))
except:
    removed_col = None


# %%


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


# %%

#{'bagging_fraction': 0.7, 'bagging_freq': 1, 'boosting_type': 'gbdt', 'early_stopping_round': 100, 'feature_fraction': 0.7, 'feature_pre_filter': False, 'lambda_l1': 0, 'lambda_l2': 1, 'learning_rate': 0.05, 'linear_tree': False, 'metric': 'None', 'min_child_samples': 150, 'min_gain_to_split': 0.02, 'n_jobs': 16, 'num_iterations': 10000, 'num_leaves': 7, 'objective': 'tweedie', 'seed': 114, 'tweedie_variance_power': 1.2, 'verbosity': -1}

#{'bagging_fraction': 0.7, 'bagging_freq': 1, 'boosting_type': 'gbdt', 'early_stopping_round': 100, 'feature_fraction': 0.7, 'feature_pre_filter': False, 'lambda_l1': 0, 'lambda_l2': 1, 'learning_rate': 0.05, 'linear_tree': False, 'metric': 'None', 'min_child_samples': 140, 'min_gain_to_split': 0.02, 'n_jobs': -1, 'num_iterations': 10000, 'num_leaves': 7, 'objective': 'tweedie', 'seed': 114, 'tweedie_variance_power': 1.2, 'verbosity': -1}

# {'bagging_fraction': 0.7, 'bagging_freq': 1, 'boosting_type': 'gbdt', 'early_stopping_round': 100, 'feature_fraction': 0.7, 'feature_pre_filter': False, 'lambda_l1': 0, 'lambda_l2': 1, 'learning_rate': 0.05, 'linear_tree': False, 'metric': 'None', 'min_child_samples': 140, 'min_gain_to_split': 0.02, 'n_jobs': 16, 'num_iterations': 10000, 'num_leaves': 7, 'objective': 'tweedie', 'seed': 114, 'tweedie_variance_power': 1, 'verbosity': -1}

#{'bagging_fraction': 0.7, 'bagging_freq': 1, 'boosting_type': 'gbdt', 'early_stopping_round': 100, 'feature_fraction': 0.7, 'feature_pre_filter': False, 'lambda_l1': 0, 'lambda_l2': 1, 'learning_rate': 0.01, 'linear_tree': False, 'metric': 'None', 'min_child_samples': 140, 'min_gain_to_split': 0.02, 'n_jobs': -1, 'num_iterations': 10000, 'num_leaves': 7, 'objective': 'tweedie', 'seed': 114, 'tweedie_variance_power': 1, 'verbosity': -1}

if 0:
    all_params = {'objective': ['mse', 'tweedie', 'xentropy'],
                  'tweedie_variance_power': [1],
                 'verbosity': [-1],
                 'boosting_type': ['gbdt'],
                 'feature_pre_filter': [False],
                 'bagging_fraction': [0.7],
                 'bagging_freq': [1],
                 'num_iterations': [10000],
                 'early_stopping_round': [100],
                 'n_jobs': [-1],
                 'seed': [114],
                 'metric':  ['None'],  # trial.suggest_categorical('metric', ['auc', 'binary_logloss', ]), #'auc',
                 'learning_rate': [0.01],
                  'lambda_l1': [0],
                  'lambda_l2': [1],
                  'min_child_samples': [140],
                  'num_leaves': [7, 5],
                  'feature_fraction': [0.8, 0.9, 0.7],
                  'min_gain_to_split': [0.02, 0.01, 0.03],
                  'linear_tree': [False],
                  #'max_bins': [8, 16, 32, 62, 128, 256, 512]
                 }
else:
    all_params = {'objective': ['tweedie'],
                  'tweedie_variance_power': [1.],
                 'verbosity': [-1],
                 'boosting_type': ['gbdt'],
                 'feature_pre_filter': [False],
                 'bagging_fraction': [0.7],
                 'bagging_freq': [1],
                 'num_iterations': [10000],
                 'early_stopping_round': [100],
                 'n_jobs': [-1],
                 'seed': [114],
                 'metric':  ['None'],  # trial.suggest_categorical('metric', ['auc', 'binary_logloss', ]), #'auc',
                 'learning_rate': [0.01],
                  'lambda_l1': [0],
                  'lambda_l2': [1],
                  'min_child_samples': [140],
                  'num_leaves': [7],
                  'feature_fraction': [0.7],
                  'min_gain_to_split': [0.02],
                  'linear_tree': [False],
                  #'max_bins': [8, 16, 32, 62, 128, 256, 512]
                 }


# %%


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


# %%


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
print('AAA', (sys.argv[1]), removed_col, best_score)


# %%


best_param['num_iterations'] = round(np.mean(list_num) * 1.1)


# %%
# %%


aaa = 0
for i in range(5):
    aaa += np.sqrt(((df.loc[df['fold']==i, 'pred'] * 100 - df.loc[df['fold']==i, 'Pawpularity']) ** 2).mean())
print('BBB', aaa / 5)


# %%


# %%


#imp = pd.DataFrame(model.feature_importance(importance_type='gain'), columns=['imp'])
#imp['col'] = COL_FEATURES
pd.options.display.max_rows = 10000
imp = sum(list_imp) / 5
imp[imp.imp > 0]

#if removed_col is None:
#    aaa = imp.sort_values('imp', ascending=False).index.values.tolist()
#    np.save('opt_final', np.array(aaa, dtype=str))


# %%


df[model.feature_name()]

