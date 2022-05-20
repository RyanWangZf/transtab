build_extractor
===============

.. autofunction:: transtab.build_extractor


The returned feature extractor takes pd.DataFrame as inputs and outputs the
encoded outputs in dict.

.. code-block:: python

    # build the feature extractor
    extractor = transtab.build_extractor(categorical_columns=['gender'], numerical_columns=['age'])

    # build a table for inputs
    df = pd.DataFrame({'age':[1,2], 'gender':['male','female']})

    # extract the outputs
    outputs = extractor(df)

    print(outputs)

    '''
        {
        'x_num': tensor([[1.],[2.]], dtype=torch.float64),
        'num_col_input_ids': tensor([[2287]]),
        'x_cat_input_ids': tensor([[5907, 3287], [5907, 2931]]),
        'x_bin_input_ids': None,
        'num_att_mask': tensor([[1]]),
        'cat_att_mask': tensor([[1, 1], [1, 1]])
        }
    '''
