import bisect
import copy
import itertools
import math
import os
import random

import albumentations.augmentations.functional as F
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import (CLAHE, Compose, IAAAffine, ImageOnlyTransform,
                            MotionBlur, Normalize, RandomBrightnessContrast)
from sklearn.model_selection import train_test_split
from torch._six import int_classes as _int_classes
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import (BatchSampler, RandomSampler, Sampler,
                                      SequentialSampler)


def quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _list_collate(batch):
    return [_ for _ in zip(*batch)]


def _cat_collate(batch):
    """concat if all tensors have same size."""
    img_ids, imgs = [_ for _ in zip(*batch)]
    imgs = [img[None] for img in imgs]
    if all(imgs[0].shape == img.shape for img in imgs):
        imgs = [torch.cat(imgs, dim=0)]
    return img_ids, imgs



def _pad_const(x, target_height, target_width, value=255, center=True, pad_loc_seed=None):
    random.seed(pad_loc_seed)
    height, width = x.shape[:2]

    if height < target_height:
        if center:
            h_pad_top = int((target_height - height) / 2.0)
        else:
            h_pad_top = random.randint(a=0, b=target_height - height)
        h_pad_bottom = target_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < target_width:
        if center:
            w_pad_left = int((target_width - width) / 2.0)
        else:
            w_pad_left = random.randint(a=0, b=target_width - width)
        w_pad_right = target_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    x = cv2.copyMakeBorder(x, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right,
                           cv2.BORDER_CONSTANT, value=value)
    return x


class RandomCropThenScaleToOriginalSize(ImageOnlyTransform):
    """Crop a random part of the input and rescale it to some size.

    Args:
        limit (float): maximum factor range for cropping region size.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        pad_value (int): pixel value for padding.
        p (float): probability of applying the transform. Default: 1.
    """

    def __init__(self, limit=0.1, interpolation=cv2.INTER_LINEAR, pad_value=0, p=1.0):
        super(RandomCropThenScaleToOriginalSize, self).__init__(p)
        self.limit = limit
        self.interpolation = interpolation
        self.pad_value = pad_value

    def apply(self, img, height_scale=1.0, width_scale=1.0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR,
              pad_value=0, pad_loc_seed=None, **params):
        img_height, img_width = img.shape[:2]
        crop_height, crop_width = int(img_height * height_scale), int(img_width * width_scale)
        crop = self.random_crop(img, crop_height, crop_width, h_start, w_start, pad_value, pad_loc_seed)
        return F.resize(crop, img_height, img_width, interpolation)

    def get_params(self):
        height_scale = 1.0 + random.uniform(-self.limit, self.limit)
        width_scale = 1.0 + random.uniform(-self.limit, self.limit)
        return {'h_start': random.random(),
                'w_start': random.random(),
                'height_scale': height_scale,
                'width_scale': width_scale,
                'pad_loc_seed': random.random()}

    def update_params(self, params, **kwargs):
        if hasattr(self, 'interpolation'):
            params['interpolation'] = self.interpolation
        if hasattr(self, 'pad_value'):
            params['pad_value'] = self.pad_value
        params.update({'cols': kwargs['image'].shape[1], 'rows': kwargs['image'].shape[0]})
        return params

    @staticmethod
    def random_crop(img, crop_height, crop_width, h_start, w_start, pad_value=0, pad_loc_seed=None):
        height, width = img.shape[:2]

        if height < crop_height or width < crop_width:
            img = _pad_const(img, crop_height, crop_width, value=pad_value, center=False, pad_loc_seed=pad_loc_seed)

        y1 = max(int((height - crop_height) * h_start), 0)
        y2 = y1 + crop_height
        x1 = max(int((width - crop_width) * w_start), 0)
        x2 = x1 + crop_width
        img = img[y1:y2, x1:x2]
        return img


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_uneven (bool): If ``True``, the sampler will drop the batches whose
            size is less than ``batch_size``
    """

    def __init__(self, sampler, group_ids, batch_size, drop_uneven=False):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven

        self.groups = torch.unique(self.group_ids).sort(0)[0]

        self._can_reuse_batches = False

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)
        # get the sampled indices from the sampler
        sampled_ids = torch.as_tensor(list(self.sampler))
        # potentially not all elements of the dataset were sampled
        # by the sampler (e.g., DistributedSampler).
        # construct a tensor which contains -1 if the element was
        # not sampled, and a non-negative number indicating the
        # order where the element was sampled.
        # for example. if sampled_ids = [3, 1] and dataset_size = 5,
        # the order is [-1, 1, -1, 0, -1]
        order = torch.full((dataset_size,), -1, dtype=torch.int64)
        order[sampled_ids] = torch.arange(len(sampled_ids))

        # get a mask with the elements that were sampled
        mask = order >= 0

        # find the elements that belong to each individual cluster
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        # get relative order of the elements inside each cluster
        # that follows the order from the sampler
        relative_order = [order[cluster] for cluster in clusters]
        # with the relative order, find the absolute order in the
        # sampled space
        permutation_ids = [s[s.sort()[1]] for s in relative_order]
        # permute each cluster so that they follow the order from
        # the sampler
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

        # splits each cluster in batch_size, and merge as a list of tensors
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))

        # now each batch internally has the right order, but
        # they are grouped by clusters. Find the permutation between
        # different batches that brings them as close as possible to
        # the order that we have in the sampler. For that, we will consider the
        # ordering as coming from the first element of each batch, and sort
        # correspondingly
        first_element_of_batch = [t[0].item() for t in merged]
        # get and inverse mapping from sampled indices and the position where
        # they occur (as returned by the sampler)
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        # from the first element in each batch, get a relative ordering
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch]
        )

        # permute the batches so that they approximately follow the order
        # from the sampler
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        # finally, permute the batches
        batches = [merged[i].tolist() for i in permutation_order]

        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)


class SimpleGroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    """

    def __init__(self, sampler, group_ids, batch_size, drop_uneven=True):
        """
        Args:
            sampler (Sampler): Base sampler.
            group_ids (list[int]): If the sampler produces indices in range [0, N),
                `group_ids` must be a list of `N` ints which contains the group id of each sample.
                The group ids must be a set of integers in the range [0, num_groups).
            batch_size (int): Size of mini-batch.
        """
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = np.asarray(group_ids)
        assert self.group_ids.ndim == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven
        groups = np.unique(self.group_ids).tolist()

        # buffer the indices of each group until batch size is reached
        self.buffer_per_group = {k: [] for k in groups}

    def __iter__(self):
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            group_buffer = self.buffer_per_group[group_id]
            group_buffer.append(idx)
            if len(group_buffer) == self.batch_size:
                yield group_buffer[:]  # yield a copy of the list
                del group_buffer[:]
        else:
            if not self.drop_uneven:
                for group_buffer in self.buffer_per_group.values():
                    if len(group_buffer) > 0:
                        yield group_buffer
        
    def __len__(self):
        raise NotImplementedError("len() of GroupedBatchSampler is not well-defined.")


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


class WeightedRandomSampler(Sampler):
    r"""Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Note: Official WeightedRandomSampler implementation of PyTorch is very slow
          due to bad performance of torch.multinomial().
          Instead, this implementation uses numpy.random.choice().

    Arguments:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
    """

    def __init__(self, weights, num_samples, replacement=True):
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = weights / weights.sum()
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(np.random.choice(np.arange(len(self.weights)),
                                     size=self.num_samples,
                                     replace=self.replacement,
                                     p=self.weights))

    def __len__(self):
        return self.num_samples


class PseudoKPerClassGroupedSampler(Sampler):

    def __init__(self, df, k=4, batch_size=32):
        self.df = df
        self.k = k
        self.batch_size = batch_size

    def __iter__(self):
        all_indexes = pd.Index([])
        for gid, df_gid in self.df[['aspect_gid', 'landmark_id']].groupby('aspect_gid'):
            lids_gid = df_gid['landmark_id']

            upper = len(lids_gid) - len(lids_gid) % self.k
            begin_indexes = np.random.permutation(np.arange(0, upper, self.k))
            rand_indexes = np.concatenate(np.array([begin_indexes + i for i in range(self.k)]).T)
            
            all_indexes = all_indexes.append(lids_gid.sort_values().iloc[rand_indexes].index)

        upper = len(all_indexes) - len(all_indexes) % self.batch_size
        begin_indexes = np.random.permutation(np.arange(0, upper, self.batch_size))
        rand_indexes = np.concatenate(np.array([begin_indexes + i for i in range(self.batch_size)]).T)

        return iter(all_indexes[rand_indexes])

    def __len__(self):
        return len(self.df)


class PseudoKPerClassGroupedBatchSampler(BatchSampler):

    def __init__(self, df, batch_size):
        self.group_ids = df['aspect_gid']
        self.landmark_ids = df['landmark_id']
        self.batch_size = batch_size

        self.groups = torch.unique(self.group_ids).sort(0)[0]

        self._can_reuse_batches = False

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)
        # get the sampled indices from the sampler
        sampled_ids = torch.as_tensor(list(self.sampler))
        # potentially not all elements of the dataset were sampled
        # by the sampler (e.g., DistributedSampler).
        # construct a tensor which contains -1 if the element was
        # not sampled, and a non-negative number indicating the
        # order where the element was sampled.
        # for example. if sampled_ids = [3, 1] and dataset_size = 5,
        # the order is [-1, 1, -1, 0, -1]
        order = torch.full((dataset_size,), -1, dtype=torch.int64)
        order[sampled_ids] = torch.arange(len(sampled_ids))

        # get a mask with the elements that were sampled
        mask = order >= 0

        # find the elements that belong to each individual cluster
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        # get relative order of the elements inside each cluster
        # that follows the order from the sampler
        relative_order = [order[cluster] for cluster in clusters]
        # with the relative order, find the absolute order in the
        # sampled space
        permutation_ids = [s[s.sort()[1]] for s in relative_order]
        # permute each cluster so that they follow the order from
        # the sampler
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

        # splits each cluster in batch_size, and merge as a list of tensors
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))

        # now each batch internally has the right order, but
        # they are grouped by clusters. Find the permutation between
        # different batches that brings them as close as possible to
        # the order that we have in the sampler. For that, we will consider the
        # ordering as coming from the first element of each batch, and sort
        # correspondingly
        first_element_of_batch = [t[0].item() for t in merged]
        # get and inverse mapping from sampled indices and the position where
        # they occur (as returned by the sampler)
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        # from the first element in each batch, get a relative ordering
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch]
        )

        # permute the batches so that they approximately follow the order
        # from the sampler
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        # finally, permute the batches
        batches = [merged[i].tolist() for i in permutation_order]

        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)


def fast_collate(batch):
    """ A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)
    https://github.com/rwightman/pytorch-image-models/blob/d72ac0db259275233877be8c1d4872163954dfbb/timm/data/loader.py
    """
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one tensor ordered by position
        # such that all tuple of position n will end up in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.long)
        tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            assert len(batch[i][0]) == inner_tuple_size  # all input tensor tuples must be same length
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False
