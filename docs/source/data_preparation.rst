Custom Dataset
==============

Here is the best practice to build your own datasets for `transtab`.

::

    project
    |
    ├── run_your_model.py
    |
    └─── data
         |
         ├── dataset1
         |   |    data_processed.csv
         |   |    binary_feature.txt
         |   └─── numerical_feature.txt
         |
         ├── dataset2
         |   
        ...

where the ``run_your_model.py`` is the code where you will load the dataset and train your models.

You should put the preprocessed table into ``data_processed.csv``, which is better to follow the protocols:

* All the column names to be represented by meaningful natural languge.
* All the categorical features to be represented by meaningful natural language.
* All the binary features to be represented by 0 or 1.
* All the numerical features to be represented by continuous values.
* Store the processed table into ``data_processed.csv``.
* Store the binary column names into ``binary_feature.txt``. No need to create this file if no binary feature.
* Store the numerical column names into ``numerical_feature.txt``. No need to create this file if no numerical feature.
* All the other columns will be treated as categorical or textual.

After that, you can try to load the dataset by


.. code-block:: python

    transtab.load_data('./data/dataset1')


About ``dataset_config``, an example is provided as

.. code-block:: python

    EXAMPLE_DATACONFIG = {
        "example": { # dataset name
            "bin": ["bin1", "bin2"], # binary column names
            "cat": ["cat1", "cat2"], # categorical column names
            "num": ["num1", "num2"], # numerical column names
            "cols": ["bin1", "bin2", "cat1", "cat2", "num1", "num2"], # all column names
            "binary_indicator": ["1", "yes", "true", "positive", "t", "y"], # binary indicators in the binary columns, which will be converted to 1
            "data_split_idx": {
                "train":[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], # row indices for training set
                "val":[10, 11, 12, 13, 14, 15, 16, 17, 18, 19], # row indices for validation set
                "test":[20, 21, 22, 23, 24, 25, 26, 27, 28, 29], # row indices for test set
                }
            }
        }

