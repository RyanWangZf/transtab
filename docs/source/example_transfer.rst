Tabular Transfer Learning
=========================

*transtab* is able to leverage the knowledge learned from broad data sources than finetunes on the target
data. It is also easy to fulfill it by this package.
The full code is available at `Notebook Example <https://github.com/ryanwangzf/transtab/blob/master/examples/transfer_learning.ipynb>`_.


.. code-block:: python

    import transtab

    # load a dataset and start vanilla supervised training
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data('credit-g')
    
    # build transtab classifier model
    model = transtab.build_classifier(cat_cols, num_cols, bin_cols)

    # start training
    training_arguments = {
        'num_epoch':50,
        'eval_metric':'val_loss',
        'eval_less_is_better':True,
        'output_dir':'./checkpoint'
        }
    transtab.train(model, trainset, valset, **training_arguments)

Now we have obtained the pretrained model saved in './checkpoint', we can load the model
from this path and update the model with new samples and columns.


.. code-block:: python

    # now let's load another data and try to leverage the pretrained model for finetuning
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data('credit-approval')

    # load the pretrained model
    model.load('./checkpoint')

    # update model's categorical/numerical/binary column dict
    model.update({'cat':cat_cols,'num':num_cols,'bin':bin_cols})


It should be noted if the finetune data differs the pretrain data on the number of classes, this should
be explicitly claimed in the update.

.. code-block:: python

    model.update({'cat':cat_cols,'num':num_cols,'bin':bin_cols, 'num_class':2})


Then we can continue to train the model just as same as done for supervised learning.

.. code-block:: python

    transtab.train(model, trainset, valset, **training_arguments)
