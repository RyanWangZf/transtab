import pdb
import os

from . import constants
from .modeling_transtab import TransTabClassifier, TransTabFeatureExtractor, TransTabFeatureProcessor
from .modeling_transtab import TransTabForCL
from .modeling_transtab import TransTabInputEncoder, TransTabModel
from .dataset import load_data
from .evaluator import predict, evaluate
from .trainer import Trainer
from .trainer_utils import TransTabCollatorForCL
from .trainer_utils import random_seed

def build_classifier(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    feature_extractor=None,
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
    '''Build a :class:`transtab.modeling_transtab.TransTabClassifier`.

    Parameters
    ----------
    categorical_columns: list 
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).
    
    feature_extractor: TransTabFeatureExtractor
        a feature extractor to tokenize the input tables. if not passed the model will build itself.

    num_class: int
        number of output classes to be predicted.

    hidden_dim: int
        the dimension of hidden embeddings.
    
    num_layer: int
        the number of transformer layers used in the encoder.
    
    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.
    
    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.
    
    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.
    
    checkpoint: str
        the directory to load the pretrained TransTab model.

    Returns
    -------
    A TransTabClassifier model.

    '''
    model = TransTabClassifier(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        feature_extractor = feature_extractor,
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
    ignore_duplicate_cols=False,
    disable_tokenizer_parallel=False,
    checkpoint=None,
    **kwargs,) -> TransTabFeatureExtractor:
    '''Build a feature extractor for TransTab model.

    Parameters
    ----------
    categorical_columns: list 
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).

    ignore_duplicate_cols: bool
        if there is one column assigned to more than one type, e.g., the feature age is both nominated
        as categorical and binary columns, the model will raise errors. set True to avoid this error as 
        the model will ignore this duplicate feature.

    disable_tokenizer_parallel: bool
        if the returned feature extractor is leveraged by the collate function for a dataloader,
        try to set this False in case the dataloader raises errors because the dataloader builds 
        multiple workers and the tokenizer builds multiple workers at the same time.

    checkpoint: str
        the directory of the predefined TransTabFeatureExtractor.

    Returns
    -------
    A TransTabFeatureExtractor module.

    '''
    feature_extractor = TransTabFeatureExtractor(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        disable_tokenizer_parallel=disable_tokenizer_parallel,
        ignore_duplicate_cols=ignore_duplicate_cols,
    )
    if checkpoint is not None:
        extractor_path = os.path.join(checkpoint, constants.EXTRACTOR_STATE_DIR)
        if os.path.exists(extractor_path):
            feature_extractor.load(extractor_path)
        else:
            feature_extractor.load(checkpoint)
    return feature_extractor

def build_encoder(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    device='cuda:0',
    checkpoint=None,
    **kwargs,
    ):
    '''
    Build a feature encoder that maps inputs tabular samples to embeddings.
    
    Parameters
    ----------
    categorical_columns: list 
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).
    
    hidden_dim: int
        the dimension of hidden embeddings.
    
    num_layer: int
        the number of transformer layers used in the encoder. If set zero, only use the
        embedding layer to get token-level embeddings.
    
    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.
        Ignored if `num_layer=0` is zero.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.
        Ignored if `num_layer=0` is zero.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.
        Ignored if `num_layer=0` is zero.

    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.
        Ignored if `num_layer=0` is zero.
    
    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.
    
    checkpoint: str
        the directory to load the pretrained TransTab model.
    '''
    if num_layer == 0:
        feature_extractor = TransTabFeatureExtractor(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            )
        
        feature_processor = TransTabFeatureProcessor(
            vocab_size=feature_extractor.vocab_size,
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            device=device,
            )

        enc = TransTabInputEncoder(feature_extractor, feature_processor)
        enc.load(checkpoint)
        
    else:
        enc = TransTabModel(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            device=device,
            )
        if checkpoint is not None:
            enc.load(checkpoint)

    return enc

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
    **kwargs,
    ): 
    '''Build a contrastive learner for pretraining based on TransTab.
    If no cat/num/bin specified, the model takes ALL as categorical columns,
    which may undermine the performance significantly.

    If there is one column assigned to more than one type, e.g., the feature age is both nominated
    as categorical and binary columns, the model will raise errors. set ``ignore_duplicate_cols=True`` to avoid this error as 
    the model will ignore this duplicate feature.

    Parameters
    ----------
    categorical_columns: list 
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).
    
    feature_extractor: TransTabFeatureExtractor
        a feature extractor to tokenize the input tables. if not passed the model will build itself.

    hidden_dim: int
        the dimension of hidden embeddings.
    
    num_layer: int
        the number of transformer layers used in the encoder.
    
    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.
    
    projection_dim: int
        the dimension of projection head on the top of encoder.
    
    overlap_ratio: float
        the overlap ratio of columns of different partitions when doing subsetting.
    
    num_partition: int
        the number of partitions made for vertical-partition contrastive learning.

    supervised: bool
        whether or not to take supervised VPCL, otherwise take self-supervised VPCL.
    
    temperature: float
        temperature used to compute logits for contrastive learning.

    base_temperature: float
        base temperature used to normalize the temperature.
    
    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.
    
    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    checkpoint: str
        the directory of the pretrained transtab model.
    
    ignore_duplicate_cols: bool
        if there is one column assigned to more than one type, e.g., the feature age is both nominated
        as categorical and binary columns, the model will raise errors. set True to avoid this error as 
        the model will ignore this duplicate feature.
    
    Returns
    -------
    A TransTabForCL model.

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
    **kwargs,
    ):
    '''
    The shared train function for all TransTabModel based models.

    Parameters
    ----------
    model: TransTabModel and its subclass
        A subclass of the base model. Should be able to output logits and loss in forward, e.g.,
        ``logit, loss = model(x, y)``.
    
    trainset: list or tuple
        a list of trainsets, or a single trainset consisting of (x, y). x: pd.DataFrame or dict, y: pd.Series.
    
    valset: list or tuple
        a list of valsets, or a single valset of consisting of (x, y).
    
    num_epoch: int
        number of training epochs.
    
    batch_size: int
        training batch size.
    
    eval_batch_size: int
        evaluation batch size.

    lr: float
        training learning rate.

    weight_decay: float
        training weight decay.
    
    patience: int
        early stopping patience, only valid when ``valset`` is given.
    
    warmup_ratio: float
        the portion of training steps for learning rate warmup, if `warmup_steps` is set, it will be ignored.
    
    warmup_steps: int
        the number of training steps for learning rate warmup.
    
    eval_metric: str
        the evaluation metric during training for early stopping, can be ``"acc"``, ``"auc"``, ``"mse"``, ``"val_loss"``.
    
    output_dir: str
        the output training model weights and feature extractor configurations.
    
    collate_fn: function
        specify training collate function if it is not standard supervised learning, e.g., contrastive learning.

    num_workers: int
        the number of workers for the dataloader.
    
    balance_sample: bool
        balance_sample: whether or not do bootstrapping to maintain in batch samples are in balanced classes, only support binary classification.
    
    load_best_at_last: bool
        whether or not load the best checkpoint after the training completes.

    ignore_duplicate_cols: bool
        whether or not ignore the contradictory of cat/num/bin cols

    eval_less_is_better: bool
        if the set eval_metric is the less the better. For val_loss, it should be set True.
    
    Returns
    -------
        None
        
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
