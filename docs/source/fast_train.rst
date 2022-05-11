Fast Train with TransTab
=========================

*transtab* is featured for accepting variable-column tables for training and predicting. This is easy to be done
by this package.
The full code is available at `Notebook Example <https://github.com/ryanwangzf/transtab/blob/master/examples/fast_train.ipynb>`_.


.. code-block:: python

    import transtab

    # load multiple datasets by passing a list of data names
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data(['credit-g','credit-approval'])

    # build transtab classifier model
    model = transtab.build_classifier(cat_cols, num_cols, bin_cols)

    # specify training arguments, take validation loss for early stopping
    training_arguments = {
        'num_epoch':5, 
        'eval_metric':'val_loss',
        'eval_less_is_better':True,
        'output_dir':'./checkpoint'
        }


One can take the validation loss on the validation data of the first dataset *credit-g* only:

.. code-block:: python

    transtab.train(model, trainset, valset[0], **training_arguments)

or take the macro average loss on the validation set of both two datasets:

.. code-block:: python

    transtab.train(model, trainset, valset, **training_arguments)

After the training completes, we can load the best checkpoint judged by validation loss from the predefined *output_dir*
and make predictions.

.. code-block:: python

    model.load('./checkpoint')

    x_test, y_test = testset[0]

    ypred = transtab.predict(x_test)


.. warning::

    Under this pure supervised learning setting, all the passed datasets should have the 
    same **number of label classes**. For instance, here *credit-g* and *credit-approval* are both
    binary classification task. It is because the classifier of `transtab` only keeps one classification head 
    during the training and predicting.




