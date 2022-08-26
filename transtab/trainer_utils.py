import pdb
import os
import random
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup
)

from .modeling_transtab import TransTabFeatureExtractor

TYPE_TO_SCHEDULER_FUNCTION = {
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
    'cosine_with_restarts': get_cosine_with_hard_restarts_schedule_with_warmup,
    'polynomial': get_polynomial_decay_schedule_with_warmup,
    'constant': get_constant_schedule,
    'constant_with_warmup': get_constant_schedule_with_warmup,
}

class TrainDataset(Dataset):
    def __init__(self, trainset):
        self.x, self.y = trainset

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        x = self.x.iloc[index-1:index]
        if self.y is not None:
            y = self.y.iloc[index-1:index]
        else:
            y = None
        return x, y

class TrainCollator:
    '''A base class for all collate function used for TransTab training.
    '''
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        ignore_duplicate_cols=False,
        **kwargs,
        ):
        self.feature_extractor=TransTabFeatureExtractor(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            disable_tokenizer_parallel=True,
            ignore_duplicate_cols=ignore_duplicate_cols,
        )
    
    def save(self, path):
        self.feature_extractor.save(path)
    
    def __call__(self, data):
        raise NotImplementedError

class SupervisedTrainCollator(TrainCollator):
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        ignore_duplicate_cols=False,
        **kwargs,
        ):
        super().__init__(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        ignore_duplicate_cols=ignore_duplicate_cols,
        )
    
    def __call__(self, data):
        x = pd.concat([row[0] for row in data])
        y = pd.concat([row[1] for row in data])
        inputs = self.feature_extractor(x)
        return inputs, y

class TransTabCollatorForCL(TrainCollator):
    '''support positive pair sampling for contrastive learning of transtab model.
    '''
    def __init__(self, 
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        overlap_ratio=0.5, 
        num_partition=3,
        ignore_duplicate_cols=False,
        **kwargs) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            ignore_duplicate_cols=ignore_duplicate_cols,
        )
        assert num_partition > 0, f'number of contrastive subsets must be greater than 0, got {num_partition}'
        assert isinstance(num_partition,int), f'number of constrative subsets must be int, got {type(num_partition)}'
        assert overlap_ratio >= 0 and overlap_ratio < 1, f'overlap_ratio must be in [0, 1), got {overlap_ratio}'
        self.overlap_ratio=overlap_ratio
        self.num_partition=num_partition

    def __call__(self, data):
        '''
        Take a list of subsets (views) from the original tests.
        '''
        # 1. build positive pairs
        # 2. encode each pair using feature extractor
        df_x = pd.concat([row[0] for row in data])
        df_y = pd.concat([row[1] for row in data])
        if self.num_partition > 1:
            sub_x_list = self._build_positive_pairs(df_x, self.num_partition)
        else:
            sub_x_list = self._build_positive_pairs_single_view(df_x)
        input_x_list = []
        for sub_x in sub_x_list:
            inputs = self.feature_extractor(sub_x)
            input_x_list.append(inputs)
        res = {'input_sub_x':input_x_list}
        return res, df_y

    def _build_positive_pairs(self, x, n):
        '''build multi-view of each sample by spliting columns
        '''
        x_cols = x.columns.tolist()
        sub_col_list = np.array_split(np.array(x_cols), n)
        len_cols = len(sub_col_list[0])
        overlap = int(math.ceil(len_cols * (self.overlap_ratio)))
        sub_x_list = []
        for i, sub_col in enumerate(sub_col_list):
            if overlap > 0 and i < n-1:
                sub_col = np.concatenate([sub_col, sub_col_list[i+1][:overlap]])
            elif overlap >0 and i == n-1:
                sub_col = np.concatenate([sub_col, sub_col_list[i-1][-overlap:]])
            # np.random.shuffle(sub_col)
            sub_x = x.copy()[sub_col]
            sub_x_list.append(sub_x)
        return sub_x_list

    def _build_positive_pairs_single_view(self, x):
        x_cols = x.columns.tolist()        
        sub_x_list = [x]
        n_corrupt = int(len(x_cols)*0.5)
        corrupt_cols = x_cols[:n_corrupt]
        x_corrupt = x.copy()[corrupt_cols]
        np.random.shuffle(x_corrupt.values)
        sub_x_list.append(pd.concat([x.copy().drop(corrupt_cols,axis=1), x_corrupt], axis=1))
        return sub_x_list

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_scheduler(
    name,
    optimizer,
    num_warmup_steps = None,
    num_training_steps = None,
    ):
    '''
    Unified API to get any scheduler from its name.

    Parameters
    ----------
    name: str
        The name of the scheduler to use.

    optimizer: torch.optim.Optimizer
        The optimizer that will be used during training.

    num_warmup_steps: int
        The number of warmup steps to do. This is not required by all schedulers (hence the argument being
        optional), the function will raise an error if it's unset and the scheduler type requires it.
    
    num_training_steps: int
        The number of training steps to do. This is not required by all schedulers (hence the argument being
        optional), the function will raise an error if it's unset and the scheduler type requires it.
    '''
    name = name.lower()
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == 'constant':
        return schedule_func(optimizer)
    
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == 'constant_with_warmup':
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)
    
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
