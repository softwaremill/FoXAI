Installation
============

First we must make sure we have ``foxai`` installed:

.. code-block:: bash

   $ python -m pip install foxai


Installation from source
~~~~~~~~~~~~~~~~~~~~~~~~

You can build ``wheel`` with Python package yourself. To do so clone repository first.

.. code-block:: bash

    $ git clone https://github.com/softwaremill/AutoXAI.git
    $ cd AutoXAI/

Then setup development environment by installing ``Poetry`` and project dependencies.

.. code-block:: bash

   $ curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.2.1 python3 -
   $ poetry config virtualenvs.create true
   $ poetry config virtualenvs.in-project true
   $ poetry install

Next step is to build ``wheel`` package.

.. code-block:: bash

   $ poetry build

After executing those commands You will get ``wheel`` file in ``dist/`` directory. To install this package run:

.. code-block:: bash

   $ python -m pip install dist/<wheel_file>.whl
