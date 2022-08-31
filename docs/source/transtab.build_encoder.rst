build_extractor
===============

.. autofunction:: transtab.build_encoder

The returned feature extractor takes pd.DataFrame as inputs and outputs the
encoded sample-level embeddings.

.. code-block:: python

    # build the feature extractor
    enc = transtab.build_encoder(categorical_columns=['gender'], numerical_columns=['age'])

    # build a table for inputs
    df = pd.DataFrame({'age':[1,2], 'gender':['male','female']})

    # extract the outputs
    outputs = enc(df)

    print(outputs.shape)

    '''
    torch.Size([2, 128])
    '''