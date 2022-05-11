import pdb
import os

from . import constants
from .modeling_transtab import TransTabClassifier, TransTabFeatureExtractor
from .modeling_transtab import TransTabForCL
from .dataset import load_data
from .evaluator import predict, evaluate
from .trainer import Trainer
from .trainer_utils import TransTabCollatorForCL
from .trainer_utils import random_seed

def build_classifier(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    num_class=2,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    device='cuda:0',
    checkpoint=None,
    **kwargs) -> TransTabClassifier:
    '''Build a classifier based on TransTab.
    [Warning] If no cat/num/bin specified, the model takes ALL as categorical columns,
    which may undermine the performance significantly.
    Args:
        categorical_columns: a list of categorical features
        numerical_columns: a list of numerical features
        binary_columns: a list of yes or no feature names, accept binary indicators like
            (yes,no); (true,false); (0,1).
        num_class: number of classes for classification
        hidden_dim: dimension of embedding and transformer layer
        num_attention_head: number of attention heads of transformer layer
        ffn_dim: dimension of feedforward layer in transformer
        activation: activation function used
        device: specify the device of the classifier model
        checkpoint: directory of the pretrained transtab model.
    '''
    model = TransTabClassifier(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        num_class=num_class,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        activation=activation,
        device=device,
        **kwargs,
        )
    
    if checkpoint is not None:
        model.load(checkpoint)

    return model

def build_extractor(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    disable_tokenizer_parallel=False,
    ignore_duplicate_cols=False,
    checkpoint=None) -> TransTabFeatureExtractor:
    feature_extractor = TransTabFeatureExtractor(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        disable_tokenizer_parallel=disable_tokenizer_parallel,
        ignore_duplicate_cols=ignore_duplicate_cols,
    )
    if checkpoint is not None:
        extractor_path = os.path.join(checkpoint, constants.EXTRACTOR_STATE_DIR)
        feature_extractor.load(extractor_path)
    return feature_extractor    

def build_contrastive_learner(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    projection_dim=128,
    num_partition=3,
    overlap_ratio=0.5,
    supervised=True,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    device='cuda:0',
    checkpoint=None,
    ignore_duplicate_cols=True,
    ): 
    '''Build a contrastive learner for pretraining based on TransTab.
    [Warning] If no cat/num/bin specified, the model takes ALL as categorical columns,
    which may undermine the performance significantly.
    Args:
        categorical_columns: a list of categorical features
        numerical_columns: a list of numerical features
        binary_columns: a list of yes or no feature names, accept binary indicators like
            (yes,no); (true,false); (0,1).
        num_partition: the number of partitions made for contrastive sampling
        overlap_ratio: the overlapping ratio between the columns of each partition
        supervised: if set True, take supervised VPCL; otherwise take self-supervised VPCL
        projection_dim: the dimension of projection head
        hidden_dim: dimension of embedding and transformer layer
        num_attention_head: number of attention heads of transformer layer
        ffn_dim: dimension of feedforward layer in transformer
        activation: activation function used
        device: specify the device of the classifier model
        checkpoint: directory of the pretrained transtab model.
        ignore_duplicate_cols: set True if one col is assigned to two types, this col will be ignored if set True; otherwise raise error.
    '''

    model = TransTabForCL(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        num_partition= num_partition,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        supervised=supervised,
        ffn_dim=ffn_dim,
        projection_dim=projection_dim,
        overlap_ratio=overlap_ratio,
        activation=activation,
        device=device,
    )
    if checkpoint is not None:
        model.load(checkpoint)
    
    # build collate function for contrastive learning
    collate_fn = TransTabCollatorForCL(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        overlap_ratio=overlap_ratio,
        num_partition=num_partition,
        ignore_duplicate_cols=ignore_duplicate_cols
    )
    if checkpoint is not None:
        collate_fn.feature_extractor.load(os.path.join(checkpoint, constants.EXTRACTOR_STATE_DIR))

    return model, collate_fn


def train(model, 
    trainset, 
    valset=None,
    num_epoch=10,
    batch_size=64,
    eval_batch_size=256,
    lr=1e-4,
    weight_decay=0,
    patience=5,
    warmup_ratio=None,
    warmup_steps=None,
    eval_metric='auc',
    output_dir='./ckpt',
    collate_fn=None,
    num_workers=0,
    balance_sample=False,
    load_best_at_last=True,
    ignore_duplicate_cols=False,
    eval_less_is_better=False,
    ):
    '''Args:
    model: TransTab based model
    trainset: a list of trainset, or a single trainset
    valset: a single valset
    num_epoch: number of training epochs
    batch_size: training batch size
    eval_batch_size: evaluation batch size,
    lr: training learning rate,
    weight_decay: training weight decay,
    patience: early stopping patience,
    warmup_ratio: the portion of training steps for learning rate warmup,
    warmup_steps: the number of training steps for learning rate warmup,
    eval_metric: evaluation metric during training for early stopping
    output_dir: output training model directory,
    collate_fn: specify training collate function if it is not standard supervised learning, e.g., contrastive learning.
    num_workers: number of workers for the dataloader,
    balance_sample: whether or not do bootstrapping to maintain in batch samples are in balanced classes, only support binary classification,
    load_best_at_last: whether or not load the best checkpoint after the training completes,
    ignore_duplicate_cols: whether or not ignore the contradictory of cat/num/bin cols
    eval_less_is_better: if the set eval_metric is the less the better. For val_loss, it should be set True.
    '''
    if isinstance(trainset, tuple): trainset = [trainset]

    train_args = {
        'num_epoch': num_epoch,
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'lr': lr,
        'weight_decay':weight_decay,
        'patience':patience,
        'warmup_ratio':warmup_ratio,
        'warmup_steps':warmup_steps,
        'eval_metric':eval_metric,
        'output_dir':output_dir,
        'collate_fn':collate_fn,
        'num_workers':num_workers,
        'balance_sample':balance_sample,
        'load_best_at_last':load_best_at_last,
        'ignore_duplicate_cols':ignore_duplicate_cols,
        'eval_less_is_better':eval_less_is_better,
    }
    trainer = Trainer(
        model,
        trainset,
        valset,
        **train_args,
    )
    trainer.train()
