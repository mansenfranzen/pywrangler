===============
Developer guide
===============

--------------------------------
Setting up developer environment
--------------------------------

1. Create and activate environment
==================================

First, create a separate virtual environment for pywrangler using the tool of your
choice (e.g. conda):

.. code-block:: console

   conda create -n pywrangler_dev python=3.6

Next, make sure to activate your environment or to explicitly use the python
interpreter of your newly created environment for the following commands:

.. code-block:: console

   source activate pywrangler_dev

2. Clone and install pywrangler
===============================

To clone pywrangler's master branch into the current working directory
and to install it in development mode (editable) with all dependencies, run the following command:

.. code-block:: console

   pip install -e git+https://github.com/mansenfranzen/pywrangler.git@master#egg=pywrangler[all] --src ''

You may not want to install all computation engines because they may be irrelevant for you. If you want to install only the
minimal required development dependencies in combination with pyspark, switch :code:`[all]` with :code:`[dev,pyspark]`:

.. code-block:: console

   pip install -e git+https://github.com/mansenfranzen/pywrangler.git@master#egg=pywrangler[dev,pyspark] --src ''

-----------------------
Running & writing tests
-----------------------

pywrangler uses pytest as a testing framework and tox for providing different testing environments.

Using pytest
============

If you want to run test within your currently activated python environment, just run pytest
(assuming you are currently in pywrangler's root directory):

.. code-block:: console

   pytest

This will run all tests. However, you may want to run only tests related to pyspark:

.. code-block:: console

   pytest -m pyspark

Same works with :code:`pandas` and :code:`dask`.

Using tox
=========

pywrangler specifies many different environments to be tested to ensure that it
works as expected across multiple python and varying computation engine versions.
Please refer to the :code:`tox.ini` to see all available environments.

If you want to test against all environments, simply run tox:

.. code-block:: console

   tox

If you want to run tests within a single environment (e.g the most current computation engines
for python 3.7), you will need specify the environment directly:

.. code-block:: console

   tox -e py37-master

-----------------------
Building & writing docs
-----------------------

---------------------
Design & architecture
---------------------
