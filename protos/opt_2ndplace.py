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

df = pd.read_feather('opt_2ndplace.ftr')


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

df['fold'] = -1


N_FOLDS = 5
strat_kfold = StratifiedKFold(n_splits=5, random_state=365, shuffle=True)
for i, (_, train_index) in enumerate(strat_kfold.split(df.index, df['Pawpularity'])):
    df.loc[train_index, 'fold'] = i


# In[14]:



COL_FEATURES =  ['pred',
 'dense_SVD_18',
 'Blur',
 'dense121_128',
 'SVD_meta_annots_top_desc_15',
 'AdoptionSpeed',
 'Eyes',
 'SVD_CHAR_Name_5',
 'SVD_CHAR_Name_0',
 'SVD_Description_3',
 'resnet50_SVD_6',
 'resnet_128',
 'dense_128',
 'dense_SVD_15',
 'resnet50_SVD_23',
 'SVD_CHAR_BreedName_full_10',
 'dense121_SVD_19',
 'RESCUER_PhotoAmt_STD',
 'dense121_SVD_31',
 'SVD_meta_annots_top_desc_27',
 'SVD_meta_annots_top_desc_30',
 'resnet34_SVD_14',
 'dense121_SVD_7',
 'SVD_Description_42',
 'meta_label_score_max_MAX',
 'dense121_SVD_8',
 'SVD_meta_annots_top_desc_13',
 'meta_color_green_mean_MEAN',
 'resnet50_SVD_28',
 'resnet34_SVD_16',
 'dense_SVD_24',
 'res34_64',
 'resnet50_SVD_30',
 'SVD_CHAR_meta_annots_top_desc_14',
 'meta_color_red_mean_STD',
 'Affectionate with Family',
 'SVD_meta_desc_4',
 'meta_label_score_max_STD',
 'meta_label_score_mean_MAX',
 'SVD_meta_annots_top_desc_19',
 'SVD_Description_24',
 'SVD_meta_desc_12',
 'resnet34_SVD_25',
 'meta_label_score_min_MAX',
 'SVD_CHAR_Name_9',
 'SVD_Description_67',
 'SVD_CHAR_Name_14',
 'RESCUER_PetID_NUNIQUE',
 'SVD_Description_21',
 'resnet50_SVD_27',
 'dense121_SVD_17',
 'resnet34_SVD_24',
 'SVD_CHAR_meta_annots_top_desc_12',
 'SVD_meta_annots_top_desc_9',
 'resnet34_SVD_8',
 'SVD_Description_56',
 'resnet50_SVD_10',
 'resnet50_SVD_12',
 'SVD_Description_63',
 'SVD_Description_37',
 'SVD_meta_annots_top_desc_24',
 'resnet50_SVD_29',
 'meta_label_score_min_STD',
 'SVD_CHAR_meta_annots_top_desc_6',
 'meta_color_pixelfrac_MEAN',
 'SVD_Description_9',
 'SVD_meta_annots_top_desc_26',
 'resnet50_SVD_25',
 'SVD_meta_annots_top_desc_29',
 'SVD_CHAR_Name_4',
 'SVD_Description_52',
 'RESCUER_Vaccinated_MEAN',
 'SVD_Description_73',
 'dense_SVD_9',
 'meta_color_pixelfrac_STD',
 'dense_SVD_7',
 'dense_SVD_31',
 'dense_SVD_17',
 'resnet50_SVD_2',
 'SVD_Description_78',
 'SVD_Description_0',
 'SVD_Description_51',
 'SVD_Description_22',
 'RESCUER_Sterilized_MEAN',
 'resnet34_SVD_19',
 'resnet34_SVD_22',
 'img_CLUSTER_0',
 'resnet34_SVD_28',
 'dense_SVD_14',
 'PhotoAmt',
 'SVD_CHAR_meta_annots_top_desc_4',
 'SVD_CHAR_Name_15',
 'SVD_CHAR_Name_8',
 'SVD_CHAR_meta_annots_top_desc_13',
 'SVD_Description_25',
 'SVD_CHAR_Name_3',
 'resnet50_SVD_5',
 'dense121_SVD_24',
 'MULTI_Dewormed_MEAN',
 'dense_SVD_4',
 'resnet34_SVD_11',
 'SVD_CHAR_meta_annots_top_desc_10',
 'resnet50_SVD_18',
 'SVD_Description_53',
 'dense121_SVD_16',
 'dense_SVD_12',
 'SVD_meta_desc_11',
 'MULTI_Fee_MEAN',
 'dense121_SVD_12',
 'SVD_Description_38',
 'SVD_Description_16',
 'RESCUER_PhotoAmt_MEAN',
 'dense121_SVD_9',
 'MULTI_Quantity_MEAN',
 'Breed1',
 'resnet34_SVD_23',
 'resnet34_SVD_17',
 'SVD_Description_68',
 'dense_SVD_30',
 'meta_color_blue_mean_MEAN',
 'Age',
 'MULTI2_Quantity_MEAN',
 'dense_SVD_10',
 'resnet34_SVD_6',
 'SVD_meta_desc_5',
 'resnet34_SVD_0',
 'SVD_Description_26',
 'SVD_CHAR_meta_annots_top_desc_8',
 'SVD_Description_2',
 'MULTI_Quantity_STD',
 'resnet50_SVD_0',
 'SVD_Description_14',
 'SVD_Description_75',
 'BREEDfull_PetID_NUNIQUE',
 'SVD_Description_59',
 'resnet34_SVD_7',
 'doc_first_score',
 'SVD_Description_45',
 'SVD_CHAR_meta_annots_top_desc_5',
 'SVD_Description_29',
 'dense_SVD_1',
 'SVD_meta_annots_top_desc_20',
 'SVD_CHAR_meta_annots_top_desc_11',
 'SVD_CHAR_Name_2',
 'SVD_meta_annots_top_desc_14',
 'dense121_SVD_23',
 'SVD_meta_annots_top_desc_12',
 'SVD_meta_desc_13',
 'General Health',
 'dense_SVD_11',
 'SVD_meta_annots_top_desc_22',
 'SVD_meta_desc_7',
 'dense_SVD_13',
 'dense121_SVD_6',
 'FurLength',
 'dense_SVD_3',
 'SVD_Description_65',
 'SVD_CHAR_Name_12',
 'doc_ent_other_count',
 'INTERACTION_avg_fee_STD',
 'SVD_meta_annots_top_desc_6',
 'SVD_CHAR_Name_11',
 'SVD_Description_32',
 'BREED1_Age_MEAN',
 'SVD_sentiment_entities_4',
 'SVD_sentiment_entities_5',
 'dense121_SVD_27',
 'dense121_SVD_4',
 'SVD_meta_desc_3',
 'resnet34_SVD_5',
 'SVD_CHAR_Name_13',
 'meta_label_score_min_MEAN',
 'MULTI_PetID_NUNIQUE',
 'SVD_Description_15',
 'dense121_SVD_3',
 'MULTI2_Quantity_STD',
 'SVD_CHAR_BreedName_full_5',
 'SVD_Description_43',
 'dense_SVD_16',
 'SVD_meta_annots_top_desc_28',
 'SVD_meta_annots_top_desc_31',
 'BREEDfull_avg_fee_MEAN',
 'dense_SVD_0',
 'SVD_meta_annots_top_desc_11',
 'dense121_SVD_25',
 'resnet50_SVD_19',
 'resnet34_SVD_30',
 'SVD_Description_5',
 'SVD_Description_71',
 'SVD_Description_69',
 'SVD_sentiment_entities_8',
 'resnet50_SVD_21',
 'resnet50_SVD_14',
 'resnet34_SVD_13',
 'SVD_Description_62',
 'resnet34_SVD_9',
 'SVD_Description_60',
 'SVD_CHAR_Name_1',
 'SVD_CHAR_BreedName_full_3',
 'dense121_SVD_29',
 'dense121_SVD_30',
 'SVD_Description_28',
 'SVD_CHAR_meta_annots_top_desc_3',
 'dense_SVD_25',
 'SVD_Description_20',
 'resnet50_SVD_31',
 'dense_SVD_28',
 'dense121_SVD_13',
 'SVD_CHAR_meta_annots_top_desc_0',
 'dense_SVD_23',
 'SVD_Description_12',
 'resnet34_SVD_15',
 'SVD_Description_6',
 'resnet34_SVD_3',
 'dense121_SVD_21',
 'resnet50_SVD_3',
 'meta_crop_area_sum_MEAN',
 'MULTI_Vaccinated_MEAN',
 'MULTI_avg_fee_MEAN',
 'meta_label_score_mean_MEAN',
 'meta_label_score_mean_STD',
 'SVD_meta_desc_2',
 'SVD_meta_annots_top_desc_10',
 'sentiment_len',
 'SVD_meta_annots_top_desc_8',
 'BREED1_Fee_MEAN',
 'SVD_meta_annots_top_desc_4',
 'BREED1_Age_STD',
 'SVD_sentiment_entities_3',
 'SVD_Description_66',
 'SVD_Description_64',
 'doc_mag_std',
 'SVD_Description_40',
 'SVD_Description_31',
 'SVD_Description_33',
 'resnet50_SVD_9',
 'SVD_Description_8',
 'SVD_Description_49',
 'SVD_meta_desc_1',
 'SVD_Description_50',
 'resnet34_SVD_4',
 'SVD_Description_27',
 'SVD_meta_desc_6',
 'resnet34_SVD_12',
 'meta_color_red_std_MEAN',
 'SVD_meta_desc_0',
 'doc_score_min',
 'SVD_meta_annots_top_desc_21',
 'SVD_meta_annots_top_desc_23',
 'SVD_Description_48',
 'BREEDfull_avg_fee_STD',
 'SVD_sentiment_entities_9',
 'resnet50_SVD_16',
 'meta_img_aratio_STD',
 'SVD_Description_46',
 'SVD_CHAR_BreedName_full_0',
 'RESCUER_Dewormed_MEAN',
 'RESCUER_VideoAmt_STD',
 'SVD_meta_desc_8',
 'resnet50_SVD_15',
 'SVD_Description_23',
 'resnet50_SVD_20',
 'resnet50_SVD_26',
 'dense121_SVD_11',
 'BREED1_Dewormed_MEAN',
 'dense121_SVD_14',
 'MULTI2_Age_MEAN',
 'SVD_Description_79',
 'SVD_CHAR_Name_7',
 'SVD_sentiment_entities_2',
 'SVD_meta_annots_top_desc_5',
 'SVD_meta_desc_15',
 'resnet50_SVD_17',
 'MULTI2_Age_MAX',
 'SVD_sentiment_entities_7',
 'MULTI_Age_MEAN',
 'SVD_CHAR_BreedName_full_15',
 'MULTI2_Fee_MEAN',
 'description_word_len',
 'BREEDfull_Quantity_SUM',
 'SVD_Description_72',
 'SVD_Description_57',
 'dense_SVD_22',
 'MULTI_MaturitySize_STD',
 'dense_SVD_20',
 'dense_SVD_19',
 'RESCUER_avg_photo_STD',
 'SVD_Description_39',
 'dense121_SVD_18',
 'Description_lda_15',
 'SVD_Description_30',
 'dense121_SVD_5',
 'resnet34_SVD_21',
 'dense121_SVD_26',
 'name_len',
 'resnet34_SVD_18',
 'SVD_Description_61',
 'resnet34_SVD_10',
 'SVD_CHAR_Name_10',
 'MULTI_Age_STD',
 'SVD_CHAR_meta_annots_top_desc_7',
 'dense_SVD_5',
 'SVD_CHAR_Name_6',
 'SVD_meta_annots_top_desc_18',
 'SVD_CHAR_meta_annots_top_desc_9',
 'resnet50_SVD_24',
 'SVD_meta_desc_9',
 'dense_SVD_8',
 'dense121_SVD_20',
 'meta_color_blue_std_MEAN',
 'e_description_len',
 'MULTI_Sterilized_MEAN',
 'resnet50_SVD_22',
 'Description_lda_8',
 'MULTI_avg_fee_MAX',
 'SVD_Description_18',
 'SVD_Description_36',
 'SVD_Description_17',
 'RESCUER_avg_photo_MEAN',
 'SVD_CHAR_BreedName_full_9',
 'MULTI2_Quantity_SUM',
 'SVD_Description_44',
 'SVD_Description_13',
 'SVD_Description_11',
 'SVD_Description_7',
 'resnet34_SVD_1',
 'resnet34_SVD_2',
 'SVD_Description_58',
 'SVD_meta_annots_top_desc_7',
 'Description_lda_3',
 'SVD_CHAR_BreedName_full_11',
 'resnet34_SVD_20',
 'SVD_CHAR_BreedName_full_12',
 'doc_score_sum',
 'resnet34_SVD_29',
 'Breed_full',
 'BREED1_Health_MEAN',
 'meta_color_score_MAX',
 'resnet50_SVD_7',
 'dense_SVD_29',
 'meta_color_green_mean_STD',
 'dense121_SVD_10',
 'Description_lda_2',
 'SVD_CHAR_BreedName_full_2',
 'SVD_meta_desc_10',
 'SVD_meta_annots_top_desc_17',
 'SVD_meta_annots_top_desc_16',
 'SVD_Description_34',
 'dense121_SVD_28',
 'SVD_sentiment_entities_6',
 'SVD_sentiment_entities_0',
 'SVD_Description_76',
 'SVD_Description_74',
 'SVD_Description_4',
 'SVD_Description_35',
 'SVD_Description_55',
 'SVD_Description_54',
 'SVD_Description_47',
 'SVD_CHAR_meta_annots_top_desc_2',
 'SVD_CHAR_meta_annots_top_desc_15',
 'State',
 'resnet50_SVD_11',
 'MULTI2_PetID_NUNIQUE',
 'doc_mag',
 'MULTI_Age_MAX',
 'doc_mag_mean',
 'doc_mag_sum',
 'resnet34_SVD_31',
 'resnet50_SVD_1',
 'sentiment_word_unique',
 'meta_color_blue_mean_STD',
 'meta_color_green_std_MEAN',
 'meta_color_red_mean_MEAN',
 'meta_img_aratio_MAX',
 'resnet50_SVD_8',
 'meta_crop_area_sum_STD',
 'dense121_SVD_0',
 'dense_SVD_6',
 'hard_interaction',
 'dense_SVD_26',
 'dense_SVD_21',
 'BREED1_FurLength_MEAN',
 'resnet34_SVD_27',
 'BREED1_avg_fee_MEAN',
 'doc_score_std',
 'Color_full',
 'COLOR_Quantity_STD',
 'resnet34_SVD_26',
 'Intelligence',
 'Color2',
 'SVD_Description_10',
 'SVD_CHAR_BreedName_full_6',
 'MULTI2_avg_fee_MEAN',
 'INTERACTION_Fee_MEAN',
 'SVD_CHAR_BreedName_full_1',
 'doc_ent_num',
 'sentiment_word_len',
 'Color3',
 'BREEDfull_Color_full_NUNIQUE',
 'BREED1_Breed2_NUNIQUE',
 'SVD_meta_annots_top_desc_25',
 'e_description_word_len',
 'BREED1_Quantity_SUM',
 'meta_img_aratio_MIN',
 'meta_label_score_max_MEAN',
 'RESCUER_Breed_full_NUNIQUE',
 'SVD_Description_70',
 'meta_color_score_MEAN',
 'description_len',
 'dense121_SVD_2',
 'doc_score',
 'COLORfull_avg_fee_MAX',
 'doc_mag_min',
 'MULTI_Fee_MAX',
 'doc_score_mena',
 'SVD_Description_41',
 'COLORfull_avg_fee_MEAN',
 'MULTI_FurLength_MEAN',
 'SVD_meta_annots_top_desc_2',
 'doc_stcs_len',
 'name_count',
 'BREEDfull_avg_fee_MAX',
 'SVD_meta_annots_top_desc_3',
 'COLORfull_Fee_STD',
 'STATE_Quantity_MAX',
 'STATE_VideoAmt_MEAN',
 'SVD_Description_19',
 'resnet50_SVD_4',
 'Quantity',
 'SVD_CHAR_BreedName_full_8',
 'SVD_CHAR_BreedName_full_7',
 'State_color_lda_3',
 'avg_photo',
 'SVD_CHAR_meta_annots_top_desc_1',
 'dense_SVD_2',
 'dense121_SVD_22',
 'dense121_SVD_15',
 'SVD_Description_1',
 'dense_SVD_27',
 'RESCUER_VideoAmt_MEAN',
 'e_description_word_unique',
 'breed_lda_3',
 'meta_textblock_num_MEAN',
 'SVD_meta_desc_14',
 'Description_lda_6',
 'Description_lda_1',
 'MULTI2_Age_MIN',
 'BREED1_Sterilized_MEAN',
 'breed_lda_4',
 'BREEDfull_Fee_MEAN',
 'BREEDfull_Fee_MAX',
 'BREED1_MaturitySize_STD',
 'BREED1_Quantity_STD',
 'SVD_sentiment_entities_1',
 'BREED1_PetID_NUNIQUE',
 'BREED1_Quantity_MAX',
 'meta_crop_area_sum_MIN',
 'Gender',
 'SVD_meta_annots_top_desc_0',
 'SVD_meta_annots_top_desc_1',
 'resnet50_SVD_13',
 'BREED1_avg_fee_STD',
 'BREED1_Color_full_NUNIQUE',
 'COLOR_Age_STD',
 'COLORfull_avg_fee_STD',
 'dense121_SVD_1',
 'breed_breed_lda_0',
 'COLORfull_Fee_MEAN',
 'doc_score_max',
 'STATE_MaturitySize_MEAN',
 'STATE_FurLength_MEAN',
 'color_num',
 'doc_ent_location_count',
 'MULTI_Age_MIN',
 'STATE_PhotoAmt_STD',
 'INTERACTION_avg_fee_MAX',
 'MULTI_Quantity_SUM',
 'State_breed_lda_1',
 'STATE_PhotoAmt_MEAN',
 'State_breed_lda_0',
 'doc_first_mag',
 'INTERACTION_avg_fee_MEAN',
 'MULTI_Health_MEAN',
 'MULTI_MaturitySize_MEAN',
 'doc_ent_person_count',
 'MULTI2_Fee_MAX',
 'doc_last_mag',
 'SVD_CHAR_BreedName_full_4',
 'STATE_Dewormed_MEAN',
 'SVD_CHAR_BreedName_full_13',
 'COLOR_avg_fee_MAX',
 'COLOR_avg_fee_STD',
 'doc_ent_woa_count',
 'Description_lda_16',
 'Color1',
 'COLOR_Breed1_NUNIQUE',
 'doc_language',
 'COLOR_Fee_MAX',
 'COLOR_Age_MEAN',
 'Potential for Playfulness',
 'Pet Friendly',
 'doc_mag_max',
 'Kid Friendly',
 'COLOR_Age_MAX',
 'doc_last_score',
 'COLOR_avg_fee_MEAN',
 'meta_face_annotation_NUNIQUE',
 'COLOR_Fee_STD',
 'COLOR_Fee_MEAN',
 'Description_lda_14',
 'Description_lda_13',
 'Description_lda_12',
 'c_description_word_unique',
 'Description_lda_18',
 'Description_lda_19',
 'Description_lda_11',
 'Breed2',
 'doc_ent_event_count',
 'doc_ent_good_count',
 'Description_lda_10',
 'Description_lda_9',
 'Description_lda_7',
 'Description_lda_5',
 'Description_lda_4',
 'meta_crop_conf_STD',
 'meta_crop_conf_MAX',
 'Description_lda_0',
 'meta_crop_importance_MEAN',
 'meta_crop_importance_STD',
 'doc_ent_org_count',
 'pure_breed',
 'meta_img_aratio_NUNIQUE',
 'c_description_word_len',
 'c_description_len',
 'SVD_CHAR_BreedName_full_14',
 'meta_textblock_num_MAX',
 'meta_face_annotation_MEAN',
 'Description_lda_17',
 'meta_crop_conf_MEAN',
 'SVD_Chinese_desc_13',
 'COLOR_PetID_NUNIQUE',
 'STATE_avg_photo_STD',
 'INTERACTION_Fee_MAX',
 'INTERACTION_Fee_MIN',
 'breed_Domestic',
 'MULTI2_Age_STD',
 'MULTI2_avg_fee_MAX',
 'MULTI2_avg_fee_MIN',
 'MULTI2_Fee_MIN',
 'MULTI_avg_fee_MIN',
 'MULTI_Fee_MIN',
 'STATE_avg_photo_MEAN',
 'breed_num',
 'STATE_VideoAmt_STD',
 'STATE_Sterilized_MEAN',
 'STATE_Vaccinated_MEAN',
 'STATE_MaturitySize_STD',
 'STATE_Health_MEAN',
 'STATE_FurLength_STD',
 'STATE_Quantity_STD',
 'STATE_Quantity_MEAN',
 'STATE_Age_MAX',
 'breed_mixed',
 'breed_noname',
 'COLORfull_Fee_MAX',
 'breed_breed_lda_2',
 'Fee',
 'Easy to Groom',
 'age_in_year',
 'avg_fee',
 'empty_name',
 'breed_lda_0',
 'breed_lda_1',
 'breed_lda_2',
 'breed_breed_lda_1',
 'breed_breed_lda_3',
 'strange_name',
 'breed_breed_lda_4',
 'State_breed_lda_2',
 'State_breed_lda_3',
 'State_breed_lda_4',
 'State_color_lda_0',
 'State_color_lda_1',
 'State_color_lda_2',
 'State_color_lda_4',
 'Vaccinated',
 'STATE_Age_STD',
 'STATE_Age_MEAN',
 'STATE_avg_fee_MAX',
 'SVD_Chinese_desc_21',
 'SVD_Chinese_desc_11',
 'SVD_Chinese_desc_12',
 'SVD_Chinese_desc_14',
 'SVD_Chinese_desc_15',
 'SVD_Chinese_desc_16',
 'SVD_Chinese_desc_17',
 'SVD_Chinese_desc_18',
 'SVD_Chinese_desc_19',
 'SVD_Chinese_desc_20',
 'SVD_Chinese_desc_22',
 'STATE_avg_fee_STD',
 'SVD_Chinese_desc_23',
 'BREED1_MaturitySize_MEAN',
 'BREED1_Quantity_MEAN',
 'BREED1_Age_MAX',
 'BREED1_Age_MIN',
 'BREED1_avg_fee_MAX',
 'BREED1_Fee_MAX',
 'COLORfull_Quantity_SUM',
 'COLORfull_Breed_full_NUNIQUE',
 'SVD_Chinese_desc_10',
 'SVD_Chinese_desc_9',
 'SVD_Chinese_desc_8',
 'SVD_Chinese_desc_7',
 'STATE_avg_fee_MEAN',
 'STATE_Fee_MAX',
 'STATE_Fee_MEAN',
 'STATE_RescuerID_NUNIQUE',
 'STATE_PetID_NUNIQUE',
 'STATE_Breed_full_NUNIQUE',
 'STATE_Color_full_NUNIQUE',
 'BREEDfull_Fee_MIN',
 'BREED1_Vaccinated_MEAN',
 'SVD_Description_77',
 'BREED1_MaturitySize_MAX',
 'BREED1_MaturitySize_MIN',
 'SVD_Chinese_desc_0',
 'SVD_Chinese_desc_1',
 'SVD_Chinese_desc_2',
 'SVD_Chinese_desc_3',
 'SVD_Chinese_desc_4',
 'SVD_Chinese_desc_5',
 'SVD_Chinese_desc_6',
 'Amount of Shedding']

import sys

COL_FEATURES = COL_FEATURES[:int(sys.argv[1])]


# In[15]:


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


# In[16]:


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


# In[17]:


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


# In[18]:


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


# In[19]:


best_param['num_iterations'] = round(np.mean(list_num) * 1.1)


# In[20]:



aaa = 0
for i in range(5):
    aaa += np.sqrt(((df.loc[df['fold']==i, 'pred'] * 100 - df.loc[df['fold']==i, 'Pawpularity']) ** 2).mean())
aaa / 5


# In[23]:


#imp = pd.DataFrame(model.feature_importance(importance_type='gain'), columns=['imp'])
#imp['col'] = COL_FEATURES
pd.options.display.max_rows = 10000
imp = sum(list_imp) / 5
imp[imp.imp > 0]
imp.sort_values('imp', ascending=False).index.values.tolist()#[:30]


# In[24]:


model.feature_name()

