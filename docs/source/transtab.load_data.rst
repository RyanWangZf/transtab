load_data
=========

.. autofunction:: transtab.load_data


*transtab* provides flexible data loading function.
It can be used to load arbitrary datasets from `openml <https://www.openml.org/>`_ supported by `openml.datasets API <https://docs.openml.org/Python-API/>`_.

.. code-block:: python

    # specify the dataname
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data('credit-g')

    # or specify the dataset index (in openml)
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data(31)

It can also be used to load datasets from the local device.

.. code-block:: python

    # specify the dataset dir
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data('./data/credit-g')


Another important feature is to use this function to load multiple datasets

.. code-block:: python

    # specify the dataset dir
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data(['./data/credit-g','./data/credit-approval'])

One can also pass ``dataset_config`` to the ``load_data`` function to manipulate the input table directly.

.. code-block:: python

    # customize dataset configuration
    dataset_config = {
        'credit-g':{
            'columns':['a','b','c'], # specify the new columns for the table, should keep the same dimension as the original table.
            'cat':['a'], # specify all the categorical columns
            'bin':['b'], # specify all the binary columns
            'num':['c']} # specify all the numerical columns
            }

    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data('credit-g', dataset_config=dataset_config)


While this operation is not recommended. To avoid making errors, you'd better deposit all these configurations to the local following
the guidance of `custom dataset <https://transtab.readthedocs.io/en/latest/data_preparation.html>`_.
