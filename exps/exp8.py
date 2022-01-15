#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os 
import sys
import torchvision.transforms as T
from PIL import Image
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"




try:
    i = int(sys.argv[1]) - 1#'deepset/xlm-roberta-large-squad2'
    model_path = pd.read_csv('run8.sh')['model'].values[i]
except:
    model_path = sys.argv[1]#'deepset/xlm-roberta-large-squad2'

exp_name = __file__.split('.')[0]
model_dir = f'{exp_name}_' + model_path.replace('/', '_')
os.makedirs(model_dir, exist_ok=True)
try:
    img_size = int(model_path.split('_')[-1])
except:
    try:
        img_size = int(model_path.split('_')[-2])
    except:
        img_size = 224

if img_size < 384:
    BATCH_SIZE = 8
else:
    BATCH_SIZE = 4

if os.path.exists(model_dir + '/train_cv_score.csv'):
    exit()
#else:
#    print(model_path)
#    exit()
    
#import glob
#if list(glob.glob(model_dir + '/*4.pth')) == 0:
#    print(model_path)
#exit()
#import pdb;pdb.set_trace()
#model_path = 'swin_large_patch4_window7_224'
#model_dir = 'change_bs_lr'

# In[2]:


#get_ipython().system('pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html')


# In[3]:


import sys
import gc
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
from timm import create_model
from timm.data.mixup import Mixup

# In[4]:


from fastai.vision.all import *


# In[5]:


set_seed(999, reproducible=True)



# Let's check what data is available to us:

# In[6]:


dataset_path = Path('../input/petfinder-pawpularity-score/')
dataset_path.ls()


# We can see that we have our train csv file with the train image names, metadata and labels, the test csv file with test image names and metadata, the sample submission csv with the test image names, and the train and test image folders.
# 
# Let's check the train csv file:

# In[7]:


train_df = pd.read_csv(dataset_path/'train.csv')
train_df.head()


# The metadata provided includes information about key visual quality and composition parameters of the photos. The Pawpularity Score is derived from the profile's page view statistics. This is the target we are aiming to predict.

# Let's do some quick processing of the image filenames to make it easier to access:

# In[8]:


train_df['path'] = train_df['Id'].map(lambda x:str(dataset_path/'train'/x)+'.jpg')
train_df = train_df.drop(columns=['Id'])
train_df = train_df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
train_df.head()


# Okay, let's check how many images are available in the training dataset:

# In[9]:


len_df = len(train_df)
print(f"There are {len_df} images")


# Let's check the distribution of the Pawpularity Score:

# In[10]:


train_df['Pawpularity'].hist(figsize = (10, 5))
print(f"The mean Pawpularity score is {train_df['Pawpularity'].mean()}")
print(f"The median Pawpularity score is {train_df['Pawpularity'].median()}")
print(f"The standard deviation of the Pawpularity score is {train_df['Pawpularity'].std()}")


# In[11]:


print(f"There are {len(train_df['Pawpularity'].unique())} unique values of Pawpularity score")


# Note that the Pawpularity score is an integer, so in addition to being a regression problem, it could also be treated as a 100-class classification problem. Alternatively, it can be treated as a binary classification problem if the Pawpularity Score is normalized between 0 and 1:

# In[12]:


train_df['norm_score'] = train_df['Pawpularity']/100
train_df['norm_score']


# Let's check an example image to see what it looks like:

# In[13]:


im = Image.open(train_df['path'][1])
width, height = im.size
print(width,height)


# In[14]:


im


# ## Data loading
# After my quick 'n dirty EDA, let's load the data into fastai as DataLoaders objects. We're using the normalized score as the label. I use some fairly basic augmentations here.

# In[15]:


#if not os.path.exists('/root/.cache/torch/hub/checkpoints/'):
#    os.makedirs('/root/.cache/torch/hub/checkpoints/')
#get_ipython().system("cp '../input/swin-transformer/swin_large_patch4_window7_224_22kto1k.pth' '/root/.cache/torch/hub/checkpoints/swin_large_patch4_window7_224_22kto1k.pth'")


# In[16]:


seed=365
set_seed(seed, reproducible=True)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True


# In[17]:


#Sturges' rule
num_bins = int(np.floor(1+(3.3)*(np.log2(len(train_df)))))
# num_bins


# In[18]:


train_df['bins'] = pd.cut(train_df['norm_score'], bins=num_bins, labels=False)
train_df['bins'].hist()


# In[19]:


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

train_df['fold'] = -1


N_FOLDS = 5
strat_kfold = StratifiedKFold(n_splits=N_FOLDS, random_state=seed, shuffle=True)
for i, (_, train_index) in enumerate(strat_kfold.split(train_df.index, train_df['bins'])):
    train_df.iloc[train_index, -1] = i
    
train_df['fold'] = train_df['fold'].astype('int')

train_df.fold.value_counts().plot.bar()


# In[20]:


train_df[train_df['fold']==0].head()


# In[21]:


def petfinder_rmse(input,target):
    return F.mse_loss(torch.clip(input.flatten(), 0.01, 1), target)




# In[22]:

transforms = aug_transforms(do_flip=True,
                            flip_vert=False,
                            max_rotate=10.0,
                            max_zoom=1.1, 
                            max_lighting=0.2,
                            max_warp=0.2,
                            p_affine=0.5, 
                            p_lighting=0.5, 
                            xtra_tfms=[Brightness(), Contrast(), Hue(), Saturation()])

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
                               seed=365, #seed
                               fn_col='path', #filename/path is in the second column of the DataFrame
                               label_col='norm_score', #label is in the first column of the DataFrame
                               y_block=RegressionBlock, #The type of target
                               bs=BATCH_SIZE, #pass in batch size
                               num_workers=8,
                               item_tfms=Resize(img_size),
                               batch_tfms=transforms #pass in batch_tfms
                               )
    
    return dls


# In[23]:


#Valid Kfolder size
the_data = get_data(0)
#assert (len(the_data.train) + len(the_data.valid)) == (len(train_df)//BATCH_SIZE)


# In[24]:

def get_learner(fold_num):
    data = get_data(fold_num)
    
    model = create_model(model_path, pretrained=True, num_classes=data.c)
    #model = torch.nn.DataParallel(model)
    learn = Learner(data, model, model_dir=model_dir, 
                    loss_func=MSELossFlat(), 
                    metrics=petfinder_rmse,
                    cbs=[MixUp(0.2)]).to_fp16()
    
    return learn


# In[25]:


test_df = pd.read_csv(dataset_path/'test.csv')
test_df.head(100)


# In[26]:


test_df['Pawpularity'] = [1]*len(test_df)
test_df['path'] = test_df['Id'].map(lambda x:str(dataset_path/'test'/x)+'.jpg')
test_df = test_df.drop(columns=['Id'])
train_df['norm_score'] = train_df['Pawpularity']/100

# In[31]:


#train_df = train_df.append(train_df_prev, ignore_index=True).append(train_df_dog, ignore_index=True)
#train_df = train_df.append(train_df_dog, ignore_index=True)

train_df


# In[32]:

try:
    get_learner(fold_num=0).lr_find(end_lr=3e-2)
except:
    pass

# In[36]:


all_preds = []

losses = []

train_df['pred'] = 0

for i in range(N_FOLDS):
    print(f'= {i} results')
    
    learn = get_learner(fold_num=i)

    learn.fit_one_cycle(20, 2e-5, 
                            cbs=[
                            SaveModelCallback(fname=f'{model_path}_{i}', 
                                                   monitor='petfinder_rmse',
                                                   every_epoch=False,
                                                    comp=np.less),
                            EarlyStoppingCallback(monitor='petfinder_rmse', comp=np.less, patience=3),
                            CSVLogger(fname=model_dir + '/hist.log', append=True)],
                            ) 
    
    val_preds, val_targets = learn.get_preds(1)
    loss = petfinder_rmse(val_preds, val_targets)
    losses.append(loss)
    
    train_df.loc[train_df['fold'] == i, 'pred'] = val_preds[:, 0].cpu().numpy()
    
    
    gc.collect()
    print('RMSE', np.mean(losses))


# In[ ]:


train_df.to_csv(model_dir + '/train_cv_score.csv', index=False)
