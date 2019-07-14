==========
pywrangler
==========

.. image:: https://travis-ci.org/mansenfranzen/pywrangler.svg?branch=master
    :target: https://travis-ci.org/mansenfranzen/pywrangler

.. image:: https://codecov.io/gh/mansenfranzen/pywrangler/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/mansenfranzen/pywrangler

.. image:: https://badge.fury.io/gh/mansenfranzen%2Fpywrangler.svg
    :target: https://badge.fury.io/gh/mansenfranzen%2Fpywrangler

.. image:: https://img.shields.io/badge/code%20style-flake8-orange.svg
    :target: https://www.python.org/dev/peps/pep-0008/

.. image:: https://img.shields.io/badge/python-3.5+-blue.svg
    :target: https://www.python.org/downloads/release/python-370/

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://lbesson.mit-license.org/

.. image:: https://badges.frapsoft.com/os/v1/open-source.png?v=103
    :target: https://github.com/ellerbrock/open-source-badges/

The pydata ecosystem provides a rich set of tools (e.g pandas, dask and pyspark)
to handle most data wrangling tasks with ease. When dealing with data on a
daily basis, however, one often encounters **problems which go beyond the
common dataframe API usage**. They typically require a combination of multiple
transformations and aggregations in order to achieve the desired outcome. For
example, extracting intervals with given start and end values from raw time
series is out of scope for native dataframe functionality.

**pywrangler** accomplishes such requirements with care while exposing so
called *data wranglers*. A data wrangler serves a specific use case just like
the one mentioned above. It takes one or more input dataframes, applies a
computation which is usually built on top of existing dataframe API, and
returns one or more output dataframes.

Why should I use pywrangler?
============================

- you deal with data wrangling **problems** which are **beyond common dataframe API usage**
- you are looking for a **framework with consistent API** to handle your data wrangling complexities
- you need implementations tailored for **small data (pandas)** and **big data (dask and pyspark)** libraries

You want **well tested, documented and benchmarked solutions**? If that's the case, pywrangler might be what you're looking for.

Features
========
- supports pandas, dask and pyspark as computation engines
- exposes consistent scikit-learn like API
- provides backwards compatibility for pandas versions from 0.19.2 upwards
- emphasises extensive tests and documentation
- includes type annotations

Thanks
======
We like to thank the pydata stack including `numpy <http://www.numpy.org/>`_, `pandas <https://pandas.pydata.org/>`_, `sklearn <https://scikit-learn.org/>`_, `scipy <https://www.scipy.org/>`_, `dask <https://dask.org/>`_ and `pyspark <https://spark.apache.org/>`_ and many more (and the open source community in general).

Notes
=====

- This project is currently under active development and has no release yet.
- This project has been set up using PyScaffold 3.1. For details and usage information on PyScaffold see https://pyscaffold.org/.
