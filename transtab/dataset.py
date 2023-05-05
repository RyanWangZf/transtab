import os
import pdb

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import openml
from loguru import logger

# TODO
# organize teh dataset_config for the load_data API.
# dataset_config = {
# 'dataname': { 'cat':[],'bin':[], 'num':[], 
# 'cols':[]}
# }


OPENML_DATACONFIG = {
    'credit-g': {'bin': ['own_telephone', 'foreign_worker']},
}

EXAMPLE_DATACONFIG = {
    "example": {
        "bin": ["bin1", "bin2"],
        "cat": ["cat1", "cat2"],
        "num": ["num1", "num2"],
        "cols": ["bin1", "bin2", "cat1", "cat2", "num1", "num2"],
        "binary_indicator": ["1", "yes", "true", "positive", "t", "y"],
        "data_split_idx": {
            "train":[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "val":[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "test":[20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        }
    }
}

def load_data(dataname, dataset_config=None, encode_cat=False, data_cut=None, seed=123):
    '''Load datasets from the local device or from openml.datasets.

    Parameters
    ----------
    dataname: str or int
        the dataset name/index intended to be loaded from openml. or the directory to the local dataset.
    
    dataset_config: dict
        the dataset configuration to specify for loading. Please note that this variable will
        override the configuration loaded from the local files or from the openml.dataset.
    
    encode_cat: bool
        whether encoder the categorical/binary columns to be discrete indices, keep False for TransTab models.
    
    data_cut: int
        how many to split the raw tables into partitions equally; set None will not execute partition.

    seed: int
        the random seed set to ensure the fixed train/val/test split.

    Returns
    -------
    all_list: list or tuple
        the complete dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    train_list: list or tuple
        the train dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    val_list: list or tuple
        the validation dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    test_list: list
        the test dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    cat_col_list: list
        the list of categorical column names.

    num_col_list: list
        the list of numerical column names.

    bin_col_list: list
        the list of binary column names.

    '''
    if dataset_config is None: dataset_config = OPENML_DATACONFIG
    if isinstance(dataname, str):
        # load a single tabular data
        return load_single_data(dataname=dataname, dataset_config=dataset_config, encode_cat=encode_cat, data_cut=data_cut, seed=seed)
    
    if isinstance(dataname, list):
        # load a list of datasets, combine together and outputs
        num_col_list, cat_col_list, bin_col_list = [], [], []
        all_list = []
        train_list, val_list, test_list = [], [], []
        for dataname_ in dataname:
            data_config = dataset_config.get(dataname_, None)
            allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = \
                load_single_data(dataname_, dataset_config=data_config, encode_cat=encode_cat, data_cut=data_cut, seed=seed)
            num_col_list.extend(num_cols)
            cat_col_list.extend(cat_cols)
            bin_col_list.extend(bin_cols)
            all_list.append(allset)
            train_list.append(trainset)
            val_list.append(valset)
            test_list.append(testset)
        return all_list, train_list, val_list, test_list, cat_col_list, num_col_list, bin_col_list

def load_single_data(dataname, dataset_config=None, encode_cat=False, data_cut=None, seed=123):
    '''Load tabular dataset from local or from openml public database.
    args:
        dataname: Can either be the data directory on `./data/{dataname}` or the dataname which can be found from the openml database.
        dataset_config: 
            A dict like {'dataname':{'bin': [col1,col2,...]}} to indicate the binary columns for the data obtained from openml.
            Also can be used to {'dataname':{'cols':[col1,col2,..]}} to assign a new set of column names to the data
        encode_cat:  Set `False` if we are using transtab, otherwise we set it True to encode categorical values into indexes.
        data_cut: The number of cuts of the training set. Cut is performed on both rows and columns.
    outputs:
        allset: (X,y) that contains all samples of this dataset
        trainset, valset, testset: the train/val/test split
        num_cols, cat_cols, bin_cols: the list of numerical/categorical/binary column names
    '''
    print('####'*10)
    if os.path.exists(dataname):
        print(f'load from local data dir {dataname}')
        filename = os.path.join(dataname, 'data_processed.csv')
        df = pd.read_csv(filename, index_col=0)
        y = df['target_label']
        X = df.drop(['target_label'],axis=1)
        all_cols = [col.lower() for col in X.columns.tolist()]

        X.columns = all_cols
        attribute_names = all_cols
        ftfile = os.path.join(dataname, 'numerical_feature.txt')
        if os.path.exists(ftfile):
            with open(ftfile,'r') as f: num_cols = [x.strip().lower() for x in f.readlines()]
        else:
            num_cols = []
        bnfile = os.path.join(dataname, 'binary_feature.txt')
        if os.path.exists(bnfile):
            with open(bnfile,'r') as f: bin_cols = [x.strip().lower() for x in f.readlines()]
        else:
            bin_cols = []
        cat_cols = [col for col in all_cols if col not in num_cols and col not in bin_cols]

        # update cols by loading dataset_config
        if dataset_config is not None:
            if 'columns' in dataset_config:
                new_cols = dataset_config['columns']
                X.columns = new_cols

            if 'bin' in dataset_config:
                bin_cols = dataset_config['bin']
            
            if 'cat' in dataset_config:
                cat_cols = dataset_config['cat']

            if 'num' in dataset_config:
                num_cols = dataset_config['num']
        
    else:
        dataset = openml.datasets.get_dataset(dataname)
        X,y,categorical_indicator, attribute_names = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)
        
        if isinstance(dataname, int):
            openml_list = openml.datasets.list_datasets(output_format="dataframe")  # returns a dict
            dataname = openml_list.loc[openml_list.did == dataname].name.values[0]
        else:
            openml_list = openml.datasets.list_datasets(output_format="dataframe")  # returns a dict
            print(f'openml data index: {openml_list.loc[openml_list.name == dataname].index[0]}')
        
        print(f'load data from {dataname}')

        # drop cols which only have one unique value
        drop_cols = [col for col in attribute_names if X[col].nunique()<=1]

        all_cols = np.array(attribute_names)
        categorical_indicator = np.array(categorical_indicator)
        cat_cols = [col for col in all_cols[categorical_indicator] if col not in drop_cols]
        num_cols = [col for col in all_cols[~categorical_indicator] if col not in drop_cols]
        all_cols = [col for col in all_cols if col not in drop_cols]
        
        if dataset_config is not None:
            if 'bin' in dataset_config: bin_cols = [c for c in cat_cols if c in dataset_config['bin']]
        else: bin_cols = []
        cat_cols = [c for c in cat_cols if c not in bin_cols]

        # encode target label
        y = LabelEncoder().fit_transform(y.values)
        y = pd.Series(y,index=X.index)

    # start processing features
    # process num
    if len(num_cols) > 0:
        for col in num_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        for col in cat_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        # process cate
        if encode_cat:
            X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])
        else:
            X[cat_cols] = X[cat_cols].astype(str)

    if len(bin_cols) > 0:
        for col in bin_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        if 'binary_indicator' in dataset_config:
            X[bin_cols] = X[bin_cols].astype(str).applymap(lambda x: 1 if x.lower() in dataset_config['binary_indicator'] else 0).values
        else:
            X[bin_cols] = X[bin_cols].astype(str).applymap(lambda x: 1 if x.lower() in ['yes','true','1','t'] else 0).values        
        
        # if no dataset_config given, keep its original format
        # raise warning if there is not only 0/1 in the binary columns
        if (~X[bin_cols].isin([0,1])).any().any():
            raise ValueError(f'binary columns {bin_cols} contains values other than 0/1.')

    
    X = X[bin_cols + num_cols + cat_cols]

    # rename column names if is given
    if dataset_config is not None:
        data_config = dataset_config
        if 'columns' in data_config:
            new_cols = data_config['columns']
            X.columns = new_cols
            attribute_names = new_cols

        if 'bin' in data_config:
            bin_cols = data_config['bin']
        
        if 'cat' in data_config:
            cat_cols = data_config['cat']

        if 'num' in data_config:
            num_cols = data_config['num']


    # split train/val/test
    data_split_idx = None
    if dataset_config is not None:
        data_split_idx = dataset_config.get('data_split_idx', None)

    if data_split_idx is not None:
        train_idx = data_split_idx.get('train', None)
        val_idx = data_split_idx.get('val', None)
        test_idx = data_split_idx.get('test', None)

        if train_idx is None or test_idx is None:
            raise ValueError('train/test split indices must be provided together')
    
        else:
            train_dataset = X.iloc[train_idx]
            y_train = y[train_idx]
            test_dataset = X.iloc[test_idx]
            y_test = y[test_idx]
            if val_idx is not None:
                val_dataset = X.iloc[val_idx]
                y_val = y[val_idx]
            else:
                val_dataset = None
                y_val = None
    else:
        # split train/val/test
        train_dataset, test_dataset, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)
        val_size = int(len(y)*0.1)
        val_dataset = train_dataset.iloc[-val_size:]
        y_val = y_train[-val_size:]
        train_dataset = train_dataset.iloc[:-val_size]
        y_train = y_train[:-val_size]

    if data_cut is not None:
        np.random.shuffle(all_cols)
        sp_size=int(len(all_cols)/data_cut)
        col_splits = np.split(all_cols, range(0,len(all_cols),sp_size))[1:]
        new_col_splits = []
        for split in col_splits:
            candidate_cols = np.random.choice(np.setdiff1d(all_cols, split), int(sp_size/2), replace=False)
            new_col_splits.append(split.tolist() + candidate_cols.tolist())
        if len(col_splits) > data_cut:
            for i in range(len(col_splits[-1])):
                new_col_splits[i] += [col_splits[-1][i]]
                new_col_splits[i] = np.unique(new_col_splits[i]).tolist()
            new_col_splits = new_col_splits[:-1]

        # cut subset
        trainset_splits = np.array_split(train_dataset, data_cut)
        train_subset_list = []
        for i in range(data_cut):
            train_subset_list.append(
                (trainset_splits[i][new_col_splits[i]], y_train.loc[trainset_splits[i].index])
            )
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y==1).sum()/len(y)))
        return (X, y), train_subset_list, (val_dataset,y_val), (test_dataset, y_test), cat_cols, num_cols, bin_cols

    else:
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y==1).sum()/len(y)))
        return (X,y), (train_dataset,y_train), (val_dataset,y_val), (test_dataset, y_test), cat_cols, num_cols, bin_cols