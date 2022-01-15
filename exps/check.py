#!/usr/bin/env python
# coding: utf-8


import os 
import sys

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BATCH_SIZE = 8

model_path = sys.argv[1]#'deepset/xlm-roberta-large-squad2'
exp_name = 'exp4'
model_dir = f'{exp_name}_' + model_path.replace('/', '_')

if os.path.exists(model_dir + '/train_cv_score.csv'):
    pass
else:
    print(model_path)