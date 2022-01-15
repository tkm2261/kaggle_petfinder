#!/usr/bin/env python
# coding: utf-8


# 
# # PLEASE CHECKOUT MY NEW DATASET
# - https://www.kaggle.com/keagle/mountains-dataset-with-coordinates-and-countries
# - https://www.kaggle.com/keagle/list-of-indian-festivals-for-2022
# 
# Based on previous works
# # Petfinder.my - Pawpularity Contest: Simple EDA and fastai starter

# ## A look at the data
# Let's start out by setting up our environment by importing the required modules and setting a random seed:

# In[1]:


#!pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[3]:


import sys
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
from timm import create_model


# In[4]:


from fastai.vision.all import *


# In[5]:


set_seed(999, reproducible=True)
BATCH_SIZE = 32


# Let's check what data is available to us:

# In[6]:


dataset_path = Path('../input/petfinder-adoption-prediction/')
dataset_path.ls()


# We can see that we have our train csv file with the train image names, metadata and labels, the test csv file with test image names and metadata, the sample submission csv with the test image names, and the train and test image folders.
# 
# Let's check the train csv file:

# In[7]:


train_df = pd.read_csv(dataset_path/'train/train.csv')
train_df.head().T


# In[8]:


(train_df.Breed1.value_counts() > 10).index


# The metadata provided includes information about key visual quality and composition parameters of the photos. The Pawpularity Score is derived from the profile's page view statistics. This is the target we are aiming to predict.

# Let's do some quick processing of the image filenames to make it easier to access:

# In[9]:


train_df['path'] = train_df['PetID'].map(lambda x:str(dataset_path/'train_images'/x)+'-1.jpg')
train_df = train_df[train_df['path'].map(os.path.exists)].reset_index(drop=True)
#print('AAA', train_df.shape)                        
tmp = train_df['Breed1'].value_counts()
tmp = tmp[tmp > 10]
train_df = train_df[train_df['Breed1'].isin(tmp.index)].reset_index(drop=True)
#print('BBB', train_df.shape)
map_label = {v: i for i, v in enumerate(train_df['Breed1'].unique())}
#print('num label:', len(map_label))
train_df['Breed1'] = train_df['Breed1'].map(map_label)
#print(train_df['Breed1'].value_counts())
train_df = train_df.drop(columns=['PetID'])
train_df = train_df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
train_df.head()


# ## Data loading
# After my quick 'n dirty EDA, let's load the data into fastai as DataLoaders objects. We're using the normalized score as the label. I use some fairly basic augmentations here.

# In[10]:


seed=999
set_seed(seed, reproducible=True)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True


# In[11]:


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

train_df['fold'] = -1


N_FOLDS = 10
strat_kfold = StratifiedKFold(n_splits=N_FOLDS, random_state=seed, shuffle=True)
for i, (_, train_index) in enumerate(strat_kfold.split(train_df.index, train_df['Breed1'])):
    train_df.iloc[train_index, -1] = i
    
train_df['fold'] = train_df['fold'].astype('int')

train_df.fold.value_counts().plot.bar()


# In[12]:


def get_data(fold):
#     train_df_no_val = train_df.query(f'fold != {fold}')
#     train_df_val = train_df.query(f'fold == {fold}')
    
#     train_df_bal = pd.concat([train_df_no_val,train_df_val.sample(frac=1).reset_index(drop=True)])
    train_df_f = train_df.copy()
    # add is_valid for validation fold
    train_df_f['is_valid'] = (train_df_f['fold'] == fold)
    
    dls = ImageDataLoaders.from_df(train_df_f, #pass in train DataFrame
#                                valid_pct=0.2, #80-20 train-validation random split
                               valid_col='is_valid', #
                               seed=999, #seed
                               fn_col='path', #filename/path is in the second column of the DataFrame
                               label_col='Breed1', #label is in the first column of the DataFrame
                               y_block=CategoryBlock, #The type of target
                               bs=BATCH_SIZE, #pass in batch size
                               num_workers=8,
                               item_tfms=Resize(224), #pass in item_tfms
                               batch_tfms=setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation()])) #pass in batch_tfms
    
    return dls


# In[13]:


#Valid Kfolder size
the_data = get_data(0)
#assert (len(the_data.train) + len(the_data.valid)) == (len(train_df)//BATCH_SIZE)


# In[14]:


def get_learner(fold_num):
    data = get_data(fold_num)
    
    model = create_model('efficientnet_b1', pretrained=True, num_classes=data.c)
    print(data.c)

    learn = Learner(data, model, loss_func=CrossEntropyLossFlat(), metrics=CrossEntropyLossFlat()).to_fp16()
    
    return learn


# In[15]:


get_learner(fold_num=0).lr_find(end_lr=3e-2)


# In[16]:


import gc


# In[17]:


all_preds = []

for i in range(N_FOLDS):

    print(f'= {i} results')
    
    learn = get_learner(fold_num=i)

    learn.fit_one_cycle(20, 2e-5, cbs=[SaveModelCallback(fname=f'model_breed_{i}', 
                                                   monitor='valid_loss',
                                                   every_epoch=False,
                                                comp=np.less),
                                      EarlyStoppingCallback(monitor='valid_loss', comp=np.less, patience=2)]) 
    
    
    del learn

    torch.cuda.empty_cache()

    gc.collect()


# In[ ]:




