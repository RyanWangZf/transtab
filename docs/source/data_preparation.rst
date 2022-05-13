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


