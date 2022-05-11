Tabular Pretraining
===================

When encountering multiple distinct tables which may have different number of classes, performing
contrastive pretraining (called Vertical-Partition Contrastive Learning, VPCL in the paper) is often
a better choice. This can be done using the transtab contrastive learner model.
The full code is available at `Notebook Example <https://github.com/ryanwangzf/transtab/blob/master/examples/contrastive_learning.ipynb>`_.


.. code-block:: python

    import transtab

    # load multiple datasets by passing a list of data names
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data(['credit-g','credit-approval'])

    # build contrastive learner, set supervised=True for supervised VPCL
    model, collate_fn = transtab.build_contrastive_learner(
        cat_cols, num_cols, bin_cols, 
        supervised=True, # if take supervised CL
        num_partition=4, # num of column partitions for pos/neg sampling
        overlap_ratio=0.5, # specify the overlap ratio of column partitions during the CL
    )


The function transtab.build_contrastive_learner returns both the CL model and the collate function
for the training dataloaders. We then train the model like

.. code-block:: python

    # start contrastive pretraining training
    training_arguments = {
        'num_epoch':50,
        'batch_size':64,
        'lr':1e-4,
        'eval_metric':'val_loss',
        'eval_less_is_better':True,
        'output_dir':'./checkpoint' # save the pretrained model
        }

    # pass the collate function to the train function
    transtab.train(model, trainset, valset, collate_fn=collate_fn, **training_arguments)

    
After this pretrain completes, we shall build a classifier from the checkpoint.

.. code-block:: python

    # load the pretrained model and finetune on a target dataset
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data('credit-approval')

    # build transtab classifier model, and load from the pretrained dir
    model = transtab.build_classifier(checkpoint='./checkpoint')

    # update model's categorical/numerical/binary column dict
    model.update({'cat':cat_cols,'num':num_cols,'bin':bin_cols})

