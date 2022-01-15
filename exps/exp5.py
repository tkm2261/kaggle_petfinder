from comet_ml import Experiment
import sys

import json
import math
import os
from pathlib import Path

import click
import faiss
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.transforms import Resize, RandomHorizontalFlip, ColorJitter, Normalize, Compose, RandomResizedCrop, CenterCrop, ToTensor
from sklearn.model_selection import ParameterGrid, ParameterSampler, GroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image



ROOT = Path(__file__).absolute().parents[1]
IMG_DIR = ROOT / 'input/petfinder-pawpularity-score/train/'

sys.path.append(str(ROOT / 'input/timm-pytorch-image-models/pytorch-image-models-master'))
sys.path.append(str(ROOT))
import timm
from src import utils, losses

NUM_CLASSES = 100
NUM_WORKERS = 8
SEED = 0

params = {
    'ver': __file__.replace('.py', ''),
    'size': 224,
    'test_size': 224,
    'lr': 1e-3,
    'batch_size': 16,
    'optimizer': 'sam',
    'epochs': 11,
    'wd': 1e-5,
    'backbone': 'swin_large_patch4_window7_224',
    'margin': 0.3,
    's': 50,
    'fc_dim': 256,
    'brightness': 0.2,
    'contrast': 0.2,
    'scale_lower': 0.2,
    'scale_upper': 1.0,
    'filter_wd': True,
    'p': 3.0,
    'p_eval': 6.0,
    'loss': 'CurricularFace',
}


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class ShopeeNet(nn.Module):

    DIVIDABLE_BY = 32

    def __init__(self,
                 backbone,
                 num_classes,
                 fc_dim=512,
                 s=30, margin=0.5, p=3, loss='ArcMarginProduct'):
        super(ShopeeNet, self).__init__()

        self.backbone = backbone
        self.backbone.reset_classifier(num_classes=0)  # remove classifier

        self.fc = nn.Linear(self.backbone.num_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self.loss_module = getattr(losses, loss)(fc_dim, num_classes, s=s, m=margin)
        self._init_params()
        self.p = p

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone.forward_features(x)
        if isinstance(x, tuple):
            x = (x[0] + x[1]) / 2
        # x = gem(x, p=self.p).view(batch_size, -1)
        x = self.fc(x)
        x = self.bn(x)
        return x

    def forward(self, x, label):
        feat = self.extract_feat(x)
        x = self.loss_module(feat, label)
        return x, feat


class ShopeeDataset(Dataset):

    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        #img = read_image(str(self.img_dir / row['Id']) + '.jpg')
        img = read_image(row['path'])
        img = img.float() / 255
        if self.transform is not None:
            img = self.transform(img)

        if 'y' in row.keys():
            target = torch.tensor(row['y'], dtype=torch.long)
            return img, target
        else:
            return img

    def __len__(self):
        return len(self.df)


@click.group()
def cli():
    if not Path(ROOT / f'exp/{params["ver"]}/train').exists():
        Path(ROOT / f'exp/{params["ver"]}/train').mkdir(parents=True)
    if not Path(ROOT / f'exp/{params["ver"]}/tuning').exists():
        Path(ROOT / f'exp/{params["ver"]}/tuning').mkdir(parents=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False


@cli.command()
@click.option('--tuning', is_flag=True)
@click.option('--dry-run', is_flag=True)
@click.option('--params-path', type=click.Path(), default=None, help='json file path for setting parameters')
@click.option('--devices', '-d', type=str, help='comma delimited gpu device list (e.g. "0,1")')
def job(tuning, dry_run, params_path, devices):
    for fold in range(5):
        _job(tuning, dry_run, params_path, devices, fold)


def _job(tuning, dry_run, params_path, devices, fold):
    global params
    if tuning:
        with open(params_path, 'r') as f:
            params = json.load(f)
        mode_str = 'tuning'
        setting = '_'.join(
            f'{tp}-{params[tp]}' for tp in params['tuning_params'])
    else:
        mode_str = 'train'
        setting = ''

    exp_path = ROOT / f'exp/{params["ver"]}/'
    os.environ['CUDA_VISIBLE_DEVICES'] = devices

    logger = utils.get_logger(log_dir=exp_path / f'{mode_str}/log/{setting}/fold{fold}')

    experiment = Experiment(
        api_key='0u03JqiBbHenC6cjZBEKAyxNG',
        project_name='kaggle_pet2',
        workspace='tkm2261',
    )
    experiment.set_name(params['ver'] + f'-fold{fold}')
    experiment.log_parameters(params)

    if dry_run:
        raise
        params['epochs'] = 2
        nrows = 1000
    else:
        nrows = None
    #df = pd.read_csv(ROOT / 'input/petfinder-pawpularity-score/train.csv', nrows=nrows)
    df = pd.read_csv(ROOT / 'protos/ens_exp15.csv', nrows=nrows)
    # 同じ画像で複数ラベルを持つサンプルは最頻値をラベルとしてそれ以外は消す
    df['Pawpularity'] -= 1
    df['y'] = df['Pawpularity'].astype('category').cat.codes.copy()

    if 1:#tuning:
        #from sklearn.model_selection import StratifiedKFold

        #index_train, index_valid = list(GroupKFold(n_splits=5).split(df, groups=df['Pawpularity']))[fold]

        #index_train, index_valid = list(StratifiedKFold(n_splits=5, random_state=365, shuffle=True))[fold]
        df_train = df[df['fold'] != fold]#df.iloc[index_train].copy()
        df_valid = df[df['fold'] == fold]#df.iloc[index_valid].copy()
        #df_valid['true_matches'] = df_valid.label_group.map(
        #    df_valid.groupby('label_group').posting_id.agg('unique').to_dict())
    elif not dry_run:
        df_train = df
        df_valid = df

    transforms = {
        'train': Compose([
            RandomResizedCrop(size=(params['size'], params['size']), scale=(
                params['scale_lower'], params['scale_upper']), interpolation=Image.BICUBIC),
            ColorJitter(brightness=params['brightness'], contrast=params['contrast']),
            RandomHorizontalFlip(p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'valid': Compose([
            Resize(size=(params['test_size'] + 32, params['test_size'] + 32), interpolation=Image.BICUBIC),
            CenterCrop((params['test_size'], params['test_size'])),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }
    datasets = {
        'train': ShopeeDataset(df=df_train, img_dir=IMG_DIR, transform=transforms['train']),
        'valid': ShopeeDataset(df=df_valid, img_dir=IMG_DIR, transform=transforms['valid'])
    }
    data_loaders = {
        'train': DataLoader(datasets['train'], batch_size=params['batch_size'], shuffle=True,
                            drop_last=True, pin_memory=True, num_workers=NUM_WORKERS),
        'valid': DataLoader(datasets['valid'], batch_size=params['batch_size'] * 2, shuffle=False,
                            drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    }

    backbone = timm.create_model(model_name=params['backbone'], pretrained=True)
    model = ShopeeNet(backbone, num_classes=NUM_CLASSES, fc_dim=params['fc_dim'], loss=params['loss'])
    optimizer = utils.get_optim(params, model, filter_bias_and_bn=params['filter_wd'], skip_gain=True)
    criterion = nn.CrossEntropyLoss()

    model = model.to('cuda')
    if len(devices.split(',')) > 1:
        model = nn.DataParallel(model)

    best_score = np.inf
    early_stopping_rounds = 2

    for epoch in range(params['epochs']):

        logger.info(
            f'Epoch {epoch}/{params["epochs"]} | lr: {optimizer.param_groups[0]["lr"]}')

        # ============================== train ============================== #
        model.train(True)
        model.p = params['p']
        loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()

        for i, (img, y) in tqdm(enumerate(data_loaders['train']),
                                total=len(data_loaders['train']), miniters=None, ncols=55):
            img = img.to('cuda', non_blocking=True)
            y = y.to('cuda', non_blocking=True)

            outputs, _ = model(img, y)
            acc = torch.mean((outputs.max(dim=1).indices == y).float())
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            outputs, _ = model(img, y)
            criterion(outputs, y).backward()
            optimizer.second_step(zero_grad=True)

            loss_meter.update(loss.item(), img.size(0))
            acc_meter.update(acc.item(), img.size(0))
            if i % 10 == 9:
                logger.info(f'[Train] {epoch+i/len(data_loaders["train"]):.2f}epoch |'
                            f'({setting}) loss: {loss_meter.avg:.4f}, acc: {acc_meter.avg:.4f}')

        #if not tuning and not dry_run:
        #    continue

        # ============================== eval ============================== #
        model.train(False)
        model.p = params['p_eval']

        train_feats = []
        for i, (img, y) in tqdm(enumerate(data_loaders['train']),
                                total=len(data_loaders['train']), miniters=None, ncols=55):
            img = img.to('cuda', non_blocking=True)
            y = y.to('cuda', non_blocking=True)
            with torch.no_grad():
                outputs, feats_minibatch = model(img, y)
                train_feats.append(feats_minibatch.cpu().numpy())
        train_feats = np.concatenate(train_feats)
        train_feats /= np.linalg.norm(train_feats, 2, axis=1, keepdims=True)

        val_feats = []
        for i, (img, y) in tqdm(enumerate(data_loaders['valid']),
                                total=len(data_loaders['valid']), miniters=None, ncols=55):
            img = img.to('cuda', non_blocking=True)
            y = y.to('cuda', non_blocking=True)
            with torch.no_grad():
                outputs, feats_minibatch = model(img, y)
                val_feats.append(feats_minibatch.cpu().numpy())

        val_feats = np.concatenate(val_feats)
        val_feats /= np.linalg.norm(val_feats, 2, axis=1, keepdims=True)

        index = faiss.IndexFlatIP(params['fc_dim'])
        index.add(train_feats)
        similarities, indexes = index.search(val_feats, 50)

        similar_posting_ids = df_train['Pawpularity'].values[indexes]
        val_scores = dict()
        def mse(row):
            return (np.mean(row['pred_matches']) - row['Pawpularity']) ** 2
        for group_cut in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            min_num = 1
            df_valid['pred_matches'] = [similar_posting_ids[i][similarities[i] > group_cut] 
                                        if (similarities[i] > group_cut).sum() > min_num - 1 else
                                        similar_posting_ids[i][:min_num]
                                        for i in range(df_valid.shape[0])]  #list(map(lambda x, y: x[y], similar_posting_ids, similarities > group_cut))
            val_scores[group_cut] = np.sqrt(df_valid.apply(mse, axis=1).mean())

        max_val_score = np.inf
        for g, s in val_scores.items():
            if s < max_val_score:
                max_val_score = s
                best_group_cut = g

        logger.info(f'[Valid] {epoch+1}epoch |'
                    f'({setting}) score(@{best_group_cut:.2f}): {max_val_score:.4f}')
        experiment.log_metric('MSE', max_val_score, epoch=epoch)

        if max_val_score < best_score:
            best_score = max_val_score
            best_group_cut_at_best_score = best_group_cut
            last_improved_epoch = epoch
            utils.save_checkpoint(filename=exp_path / f'{mode_str}/log/{setting}/model_{fold}.pth',
                                  model=model, params=params, epoch=epoch)

        if epoch - last_improved_epoch > early_stopping_rounds:
            logger.info(
                '\n'
                f'early stopping at: {epoch + 1}\n'
                f'best epoch: {last_improved_epoch + 1}\n'
                f'best score(@{best_group_cut_at_best_score:.2f}): {best_score:.4f}\n'
            )
            break

    if isinstance(model, nn.DataParallel):
        model = model.module

    utils.save_checkpoint(filename=exp_path / f'{mode_str}/log/{setting}/model_{fold}.pth',
                              model=model, params=params, epoch=epoch)


@cli.command()
@click.option('--mode', type=str, default='grid', help='Search method (tuning)')
@click.option('--n-iter', type=int, default=10, help='n of iteration for random parameter search (tuning)')
@click.option('--n-gpu', type=int, default=-1, help='n of used gpu at once')
@click.option('--devices', '-d', type=str, help='comma delimited gpu device list (e.g. "0,1")')
@click.option('--n-blocks', '-n', type=int, default=1)
@click.option('--block-id', '-i', type=int, default=0)
def tuning(mode, n_iter, n_gpu, devices, n_blocks, block_id):

    if n_gpu == -1:
        n_gpu = len(devices.split(','))

    space = [
        {
            'lr': [1e-3, 3e-3],
            # 'optimizer': ['momentum', 'nesterov'],
        },
    ]

    if mode == 'grid':
        candidate_list = list(ParameterGrid(space))
    elif mode == 'random':
        candidate_list = list(ParameterSampler(
            space, n_iter, random_state=SEED))
    else:
        raise ValueError

    n_per_block = math.ceil(len(candidate_list) / n_blocks)
    candidate_chunk = candidate_list[block_id *
                                     n_per_block: (block_id + 1) * n_per_block]

    utils.launch_tuning(mode=mode, n_iter=n_iter, n_gpu=n_gpu, devices=devices,
                        params=params, root=ROOT, candidate_list=candidate_chunk)


if __name__ == '__main__':
    cli()