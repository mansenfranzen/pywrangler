===============
Meet pywrangler
===============

----------
Motivation
----------

Problem formulation
===================

The pydata ecosystem provides a rich set of tools (e.g `pandas`_, `dask`_, `vaex`_, `modin`_ and `pyspark`_)
to handle most data wrangling tasks with ease. When dealing with data on a daily basis,
however, one often encounters problems which go beyond the common dataframe API usage.
They typically require a combination of multiple transformations and aggregations in
order to achieve the desired outcome. For example, extracting intervals with given start
and end values from raw time series is out of scope for native dataframe functionality.

Mission
=======

pywrangler accomplishes such requirements with care while exposing so called **data wranglers**.
A data wrangler serves a specific use case just like the one mentioned above. It takes one or
more input dataframes, applies a computation which is usually built on top of existing dataframe
API, and returns one or more output dataframes.

-----
Scope
-----

pywrangler can be seen as a **transform** part in common `ETL`_ pipelines. It is not concerned
with data extraction or data loading. It assumes that data is already available in tabular format.

pywrangler addresses data transformations which are not covered by standard dataframe API.
Such transformations are usually more complex and require careful testing. Hence, a major
focus lies on **extensive testing** of provided implementations.

Apart from testing, a sophisticated documentation of how an algorithm works will increase
user compliance. Accordingly, every data wrangler is supposed to provide a comprehensive and
**visual step-by-step guide**.

pywrangler is not committed to a single computation backend like pandas. Instead, data wranglers
are defined abstractly and need to be implemented concretely in regard to the specific
computation backend. In general, all python related computation engines are supported
if corresponding implementations are provided (e.g `pandas`_, `dask`_, `vaex`_, `modin`_ and `pyspark`_).
pywrangler attempts to provide at least one implementation for **smallish data** (single node
computation e.g. `pandas`_) and **largish data** (distributed computation e.g. `pyspark`_).

Moreover, one computation engine may have several implementations with varying trade-offs, too.
In order to identify trade-offs, pywrangler aims to offer **benchmarking utilities** to
compare different implementations in regard to cpu and memory usage.

To make pywranlger **integrate** well with standard data wrangling workflows, data wranglers confirm to
the `scikit-learn API`_ as a common standard for data transformation pipelines.

Goals
=====

- describe data transformations independent of computation engine
- define data transformation requirements through extensive tests
- visually document implementation logic of data transformations step by step
- provide concrete implementations for small and big data computation engines
- add benchmarking utilities to compare different implementations to identify tradeoffs
- follow scikit-learn API for easy integration for data pipelines

Non-Goals
=========

- always support all computation engines for a single data transformation
- handle *extract* and *load* stages of ETL pipelines

---------
Rationale
---------

Computation engines may come and go. Currently (June 2020), `pandas`_ is still very popular for single node
computation. `Vaex`_ slowly catches up. `Pyspark`_ and `dask`_ are both very popular in the
realm of distributed computation engines. There will be new engines in the future,
perhaps pandas 2 or a computation engine originating from the `Apache Arrow`_ project as outlined `here`_.

In any case, what remains is the careful description of data transformations. More
importantly, computation backend independent tests manifest the requirements of
data transformations. Moreover, such a specification may also be easily ported to
languages other than python like R or Scala. This is probably the major lasting
value of pywrangler.

-----------------
Future directions
-----------------

What has been totally neglected so far by the pywrangler project (as of June 2020), is the importance
of SQL. Due to its declarative nature, SQL offers a computation engine independent way
to formulate data transformations. They are applicable to any computation engine that
supports sql. Therefore, one major goal is to add a sql backend that produces the
required sql code to perform a specific data transformation.


.. _ETL: https://en.wikipedia.org/wiki/Extract,_transform,_load
.. _`scikit-learn API`: https://scikit-learn.org/stable/data_transforms.html
.. _pandas: https://pandas.pydata.org/
.. _dask: https://dask.org/
.. _pyspark: https://spark.apache.org/
.. _vaex: https://vaex.readthedocs.io/en/latest/
.. _modin: https://modin.readthedocs.io/en/latest/
.. _here: https://docs.google.com/document/d/10RoUZmiMQRi_J1FcPeVAUAMJ6d_ZuiEbaM2Y33sNPu4/edit?usp=sharing
.. _`Apache Arrow`: https://arrow.apache.org/
