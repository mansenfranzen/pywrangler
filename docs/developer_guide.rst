===============
Developer guide
===============

--------------------------------
Setting up developer environment
--------------------------------

Create and activate environment
===============================

First, create a separate virtual environment for pywrangler using the tool of your
choice (e.g. conda):

.. code-block:: console

   conda create -n pywrangler_dev python=3.6

Next, make sure to activate your environment or to explicitly use the python
interpreter of your newly created environment for the following commands:

.. code-block:: console

   source activate pywrangler_dev

Clone and install pywrangler
============================

Install all dependencies
------------------------

To clone pywrangler's master branch into the current working directory
and to install it in development mode (editable) with all dependencies, run the following command:

.. code-block:: console

   pip install -e git+https://github.com/mansenfranzen/pywrangler.git@master#egg=pywrangler[all] --src ''
   
You may separate cloning and installing:

.. code-block:: console

   git clone https://github.com/mansenfranzen/pywrangler.git
   cd pywrangler
   pip install -e .[all]

Install selected dependencies
-----------------------------

You may not want to install all dependencies because they may be irrelevant for you. If you want to install only the
minimal required development dependencies to develop pyspark data wranglers, switch :code:`[all]` with :code:`[dev,pyspark]`:

.. code-block:: console

   pip install -e git+https://github.com/mansenfranzen/pywrangler.git@master#egg=pywrangler[dev,pyspark] --src ''
   
All available dependency packages are listed in the `setup.cfg`_ under :code:`options.extras_require`.

-------------
Running tests
-------------

pywrangler uses pytest as a testing framework and tox for providing different testing environments.

Using pytest
============

If you want to run tests within your currently activated python environment, just run pytest
(assuming you are currently in pywrangler's root directory):

.. code-block:: console

   pytest

This will run all tests. However, you may want to run only tests which are related to pyspark:

.. code-block:: console

   pytest -m pyspark

Same works with :code:`pandas` and :code:`dask`.

Using tox
=========

pywrangler specifies many different environments to be tested to ensure that it
works as expected across multiple python and varying computation engine versions.

If you want to test against all environments, simply run tox:

.. code-block:: console

   tox

If you want to run tests within a specific environment (e.g the most current computation engines
for python 3.7), you will need provide the environment abbreviation directly:

.. code-block:: console

   tox -e py37-master
   
Please refer to the `tox.ini`_ to see all available environments.

-------------
Writing tests
-------------

If you intend to write tests for data wranglers, it is highly recommended to use pywrangler's
:any:`DataTestCase` which allows a computation engine independent test case formulation. This has
three major goals in mind:

- unify and standardize test data formulation across different computation engines
- test data should be as readable as possible and should be maintainable in pure python
- make writing data centric tests as easy as possible while reducing the need of test case related boilerplate code

Actually, once you've formulated a test case via the :any:`DataTestCase`, you may easily test it
with any computation backend. Behind the scences, an computation engine independent dataframe called
:any:`PlainFrame` converts the provided test data to the specific test engine.

-----------------------
Building & writing docs
-----------------------

---------------------
Design & architecture
---------------------

.. _`setup.cfg`: https://github.com/mansenfranzen/pywrangler/blob/master/setup.cfg
.. _`tox.ini`: https://github.com/mansenfranzen/pywrangler/blob/master/tox.ini
