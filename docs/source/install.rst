Installation
============

*transtab* was tested on Python 3.7+, PyTorch 1.8.0+. Please follow the Installation instructions below for the
torch version and CUDA device you are using:

`PyTorch Installation Instructions <https://pytorch.org/get-started/locally/>`_.

After that, *transtab* can be downloaded directly using **pip**.

.. code-block:: bash

    pip install transtab

or

.. code-block:: bash

    pip install git+https://github.com/RyanWangZf/transtab.git


Alternatively, you can clone the project and install from local

.. code-block:: bash

    git clone https://github.com/RyanWangZf/transtab.git
    cd transtab
    pip install .

**Troubleshooting**:

1. If encountering ``ERROR: Failed building wheel for tokenizers`` on MAC/Linux, please call

.. code-block:: bash

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

then restart the terminal and call ``pip`` again.
