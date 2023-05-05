# TransTab: A flexible transferable tabular learning framework [[arxiv]](https://arxiv.org/pdf/2205.09328.pdf)


[![PyPI version](https://badge.fury.io/py/transtab.svg)](https://badge.fury.io/py/transtab)
[![Documentation Status](https://readthedocs.org/projects/transtab/badge/?version=latest)](https://transtab.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
![GitHub Repo stars](https://img.shields.io/github/stars/ryanwangzf/transtab)
![GitHub Repo forks](https://img.shields.io/github/forks/ryanwangzf/transtab)
[![Downloads](https://pepy.tech/badge/transtab)](https://pepy.tech/project/transtab)
[![Downloads](https://pepy.tech/badge/transtab/month)](https://pepy.tech/project/transtab)


Document is available at https://transtab.readthedocs.io/en/latest/index.html.

Paper is available at https://arxiv.org/pdf/2205.09328.pdf.

5 min blog to understand TransTab at [realsunlab.medium.com](https://realsunlab.medium.com/transtab-learning-transferable-tabular-transformers-across-tables-1e34eec161b8)!

### News!
- [05/04/23] Check the version `0.0.5` of `TransTab`!

- [01/04/23] Check the version `0.0.3` of `TransTab`!

- [12/03/22] Check out our [[blog]](https://realsunlab.medium.com/transtab-learning-transferable-tabular-transformers-across-tables-1e34eec161b8) for a quick understanding of TransTab!

- [08/31/22] `0.0.2` Support encode tabular inputs into embeddings directly. An example is provided [here](examples/table_embedding.ipynb). Several bugs are fixed.

## TODO

- [x] Table embedding.

- [ ] Add support to direct process table with missing values.

- [ ] Add regression support.

### Features
This repository provides the python package `transtab` for flexible tabular prediction model. The basic usage of `transtab` can be done in a couple of lines!

```python
import transtab

# load dataset by specifying dataset name
allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
     = transtab.load_data('credit-g')

# build classifier
model = transtab.build_classifier(cat_cols, num_cols, bin_cols)

# start training
transtab.train(model, trainset, valset, **training_arguments)

# make predictions, df_x is a pd.DataFrame with shape (n, d)
# return the predictions ypred with shape (n, 1) if binary classification;
# (n, n_class) if multiclass classification.
ypred = transtab.predict(model, df_x)
```

It's easy, isn't it?



## How to install

First, download the right ``pytorch`` version following the guide on https://pytorch.org/get-started/locally/.

Then try to install from pypi directly:

```bash
pip install transtab
```

or

```bash
pip install git+https://github.com/RyanWangZf/transtab.git
```



Please refer to for [more guidance on installation](https://transtab.readthedocs.io/en/latest/install.html) and troubleshooting.



## Transfer learning across tables

A novel feature of `transtab` is its ability to learn from multiple distinct tables. It is easy to trigger the training like

```python
# load the pretrained transtab model
model = transtab.build_classifier(checkpoint='./ckpt')

# load a new tabular dataset
allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
     = transtab.load_data('credit-approval')

# update categorical/numerical/binary column map of the loaded model
model.update({'cat':cat_cols,'num':num_cols,'bin':bin_cols})

# then we just trigger the training on the new data
transtab.train(model, trainset, valset, **training_arguments)
```



## Contrastive pretraining on multiple tables

We can also conduct contrastive pretraining on multiple distinct tables like

```python
# load from multiple tabular datasets
dataname_list = ['credit-g', 'credit-approval']
allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
     = transtab.load_data(dataname_list)

# build contrastive learner, set supervised=True for supervised VPCL
model, collate_fn = transtab.build_contrastive_learner(
    cat_cols, num_cols, bin_cols, supervised=True)

# start contrastive pretraining training
transtab.train(model, trainset, valset, collate_fn=collate_fn, **training_arguments)
```



## Citation

If you find this package useful, please consider citing the following paper:

```latex
@inproceedings{wang2022transtab,
  title={TransTab: Learning Transferable Tabular Transformers Across Tables},
  author={Wang, Zifeng and Sun, Jimeng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```
