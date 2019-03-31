==========
pywrangler
==========

.. image:: https://travis-ci.org/mansenfranzen/pywrangler.svg?branch=master
    :target: https://travis-ci.org/mansenfranzen/pywrangler

.. image:: https://coveralls.io/repos/github/mansenfranzen/pywrangler/badge.svg?branch=master
    :target: https://coveralls.io/github/mansenfranzen/pywrangler?branch=master

.. image:: https://badge.fury.io/gh/mansenfranzen%2Fpywrangler.svg
    :target: https://badge.fury.io/gh/mansenfranzen%2Fpywrangler

.. image:: https://img.shields.io/badge/code%20style-pep8-orange.svg
    :target: https://www.python.org/dev/peps/pep-0008/

.. image:: https://img.shields.io/badge/python-3.5%20/%203.6%20/%203.7+-blue.svg
    :target: https://www.python.org/downloads/release/python-370/

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://lbesson.mit-license.org/

.. image:: https://badges.frapsoft.com/os/v1/open-source.png?v=103
    :target: https://github.com/ellerbrock/open-source-badges/

The pydata ecosystem provides a rich set of tools to tackle most data wrangling
tasks with ease (e.g pandas, dask and pyspark). When dealing with data on a
daily basis, however, one often encounters **problems which go beyond the
common dataframe API usage**. They typically require a combination of multiple
transformations and aggregations in order to achieve the desired outcome. For
example, extracting intervals with given start and end values from raw time
series is a non trivial task which is out of scope for native dataframe
functionality.

**pywrangler** accomplishes such requirements with care while exposing so
called *data wranglers*. A data wrangler serves a specific use cases just like
the one mentioned above. It takes one ore more input dataframes, applies a
computation which is usually built on top of existing dataframe API, and
returns one ore more output dataframes.

Why should I use pywrangler?
============================

- you deal with data wrangling **problems** which are **beyond common dataframe API usage**
- your **daily workload prevents** from writing properly tested and documented code in regard to **best practices**
- you require different implementations for **small data (pandas)** and **big data (dask and pyspark)**

And you want **well tested, documented and benchmarked solutions**? If that's the case, **pywrangler might be what you're looking for**.

Features
========
- supports pandas, dask and pyspark as dataframe computation engines
- exposes consistent scikit-learn like API
- provides backwards compatibility for pandas versions from 0.19.2 and pyspark versions from 2.3.1 upwards
- emphasises extensive tests and documentation
- includes type annotations

Thanks
======
We want to thank the open source community and especially the pydata stack including numpy, pandas, sklearn, scipy, dask and pyspark and many more.

Note
====

This project has been set up using PyScaffold 3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
