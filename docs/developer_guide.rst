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

--------------------------------
Writing tests for data wranglers
--------------------------------

When writing tests for data wranglers, it is highly recommended to use pywrangler's
:any:`DataTestCase`. It allows a computation engine independent test case formulation
with three major goals in mind:

- Unify and standardize test data formulation across different computation engines.
- Let test data be as readable and maintainable as possible.
- Make writing data centric tests easy while reducing boilerplate code.

.. note::

   Once a test is formulated with the :any:`DataTestCase`, you may easily convert it
   to any computation backend. Behind the scences, an computation engine independent dataframe called
   :any:`PlainFrame` converts the provided test data to the specific test engine.

Example
=======

Lets start with an easy example. Imagine a data transformation for time series which
increases a counter each time it encounters a specific target signal.

Essentially, a data tranfsormation focused test case requires two things: First, the input data
which needs to be processed. Second, the output data which is expected as a result of the data
wrangling stage:

.. code-block:: python
   :linenos:
   
   from pywrangler.util.testing import DataTestCase
   
   class IncreaseOneTest(DataTestCase):
      
      def input(self):        
         """Provide the data given to the data wrangler."""
         
         cols = ["order:int", "signal:str"]
         data = [[    1,       "noise"],
                 [    2,       "target"],
                 [    3,       "noise"],
                 [    4,       "noise"],
                 [    5,       "target"]]
             
         return data, cols
         
      def output(self):
         """Provide the data expected from the data wrangler."""
         
         cols = ["order:int", "signal:str", "result:int"]
         data = [[    1,       "noise",          0],
                 [    2,       "target",         1],
                 [    3,       "noise",          1],
                 [    4,       "noise",          1],
                 [    5,       "target",         2]]
                 
         return data, cols
         
That's all you need to do in order define a data test case. As you can see, typed columns
are provided along with the corresponding data in a human readable format.

Next, let's write two different implementations using pandas and pyspark and test them
against the :code:`IncreaseOneTest`:

.. code-block:: python
   :linenos:
   
   import pandas as pd
   from pyspark.sql import functions as F, DataFrame, Window
   
   def transform_pandas(df: pd.DataFrame) -> pd.DataFrame:      
       df = df.sort_values("order")
       result = df["signal"].eq("target").cumsum()
      
       return df.assign(result=result)
      
   def transform_pyspark(df: DataFrame) -> DataFrame:
       target = F.col("signal").eqNullSafe("target").cast("integer")
       result = F.sum(target).over(Window.orderby("order"))
       
       return df.withColumn(result=result)
      
   # instantiate test case
   test_case = IncreaseOneTest()
   
   # perform test assertions for given computation backends
   test_case.test.pandas(transform_pandas)
   test_case.test.pyspark(transform_pyspark)

The single test case :code:`IncreaseOneTest` can be used to test multiple implementations 
based on different computation engines.

The :any:`DataTestCase` and :any:`PlainFrame` offer much more functionality which is covered
in the corresponding reference pages. For example, you may use :any:`PlainFrame` to seamlessly 
convert between pandas and pyspark dataframes. :any:`DataTestCase` allows to formulate mutants
of the input data which should cause the test to fail (hence covering multiple distinct but 
similar test data scenarios within the same data test case).

.. note::

   :any:`DataTestCase` currently supports only single input and output data wranglers. Data wranglers 
   requiring multiple input dataframes or computing multiple output dataframes are not supported, yet.

-----------------------
Building & writing docs
-----------------------

---------------------
Design & architecture
---------------------

.. _`setup.cfg`: https://github.com/mansenfranzen/pywrangler/blob/master/setup.cfg
.. _`tox.ini`: https://github.com/mansenfranzen/pywrangler/blob/master/tox.ini
