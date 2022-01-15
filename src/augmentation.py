import torch
import numpy as np
from torchvision.transforms import Resize


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def rand_region(target_size, source_size):
    t_h, t_w = target_size[2:]
    s_h, s_w = source_size[2:]
    cut_h = s_h // 2
    cut_w = s_w // 2

    cx = np.random.randint(cut_w, t_w - cut_w)
    cy = np.random.randint(cut_h, t_h - cut_h)
    x1 = cx - cut_w
    x2 = x1 + s_w
    y1 = cy - cut_h
    y2 = y1 + s_h
    return x1, y1, x2, y2


def cutmix(x, y, alpha):
    assert alpha > 0, 'alpha should be larger than 0'
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).cuda()
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index,
                                      :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
               (x.size()[-1] * x.size()[-2]))
    return x, target_a, target_b, lam


def mixup(x, y, alpha):
    assert alpha > 0, 'alpha should be larger than 0'

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).cuda()
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam


def resizemix(x, y, alpha=0.1, beta=0.8):
    assert alpha > 0, 'alpha should be larger than 0'
    assert beta < 1, 'beta should be smaller than 1'

    rand_index = torch.randperm(x.size()[0]).cuda()
    tau = np.random.uniform(alpha, beta)
    lam = tau ** 2

    H, W = x.size()[2:]
    resize_transform = Resize((int(H*tau), int(W*tau)))
    resized_x = resize_transform(x[rand_index])

    target_a = y[rand_index]
    target_b = y
    x1, y1, x2, y2 = rand_region(x.size(), resized_x.size())
    x[:, :, y1:y2, x1:x2] = resized_x
    return x, target_a, target_b, lam
