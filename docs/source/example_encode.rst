Encode Tables
=============

*transtab* is able to take pd.DataFrame as inputs and outputs the encoded sample-level embeddings.
The full code is available at `Notebook Example <https://github.com/ryanwangzf/transtab/blob/master/examples/table_embedding.ipynb>`_.


.. code-block:: python

    import transtab

    # load a dataset and start vanilla supervised training
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data('credit-g')
    
    # build transtab classifier model
    model, collate_fn = transtab.build_contrastive_learner(cat_cols, num_cols, bin_cols)

    # start training
    training_arguments = {
        'num_epoch':50,
        'batch_size':64,
        'lr':1e-4,
        'eval_metric':'val_loss',
        'eval_less_is_better':True,
        'output_dir':'./checkpoint'
        }
    transtab.train(model, trainset, valset, collate_fn=collate_fn, **training_arguments)

Now we have obtained the pretrained model saved in './checkpoint', we can load the model
from this path and use it to encode tables.


.. code-block:: python

    # load the pretrained model
    enc = transtab.build_encoder(
        binary_columns=bin_cols,
        checkpoint = './checkpoint'
    )

Then we can take the whole pretrained model and output the cls token embedding at the last layer's outputs

.. code-block:: python

    # encode tables to sample-level embeddings
    df = trainset[0]
    output = enc(df)
