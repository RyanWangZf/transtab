# TransTab: A flexible tabular prediction model

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
```

It's easy, isn't it?

## 🔥 Transfer learning across tables

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

## 🔥 Contrastive pretraining on multiple tables

We can also conduct contrastive pretraining on multiple distict tables like

```python
# load from multiple tabular datasets
dataname_list = ['credit-g', 'credit-approval']
allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
     = transtab.load_data(dataname_list)

# build contrastive learner, set supervised=True for supervised VPCL
model, collate_fn = transtab.build_contrastive_learner(
    cat_cols, num_cols, bin_cols, supervised=True)

# start contrastive pretraining training
transtab.train(model, trainset, collate_fn=collate_fn)
```