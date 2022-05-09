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

It's easy, isn't it