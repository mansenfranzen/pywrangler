===============
Meet pywrangler
===============

----------
Motivation
----------

The pydata ecosystem provides a rich set of tools (e.g pandas, dask, vaex, modin and pyspark)
to handle most data wrangling tasks with ease. When dealing with data on a daily basis,
however, one often encounters problems which go beyond the common dataframe API usage.
They typically require a combination of multiple transformations and aggregations in
order to achieve the desired outcome. For example, extracting intervals with given start
and end values from raw time series is out of scope for native dataframe functionality.

pywrangler accomplishes such requirements with care while exposing so called data wranglers.
A data wrangler serves a specific use case just like the one mentioned above. It takes one or
more input dataframes, applies a computation which is usually built on top of existing dataframe
API, and returns one or more output dataframes.

-----
Scope
-----

pywrangler can be seen as the *transform* part in common `ETL`_ pipelines. It is concerned
about data transformations which are not covered by standard dataframe API. Such transformations
are usually more complex and require careful testing. Hence, a major focus lies on extensive
testing of provided implementations.

Apart from testing, a sophisticated documentation of the inner workings of data
transformations will increase user compliance. It should help the user understand
how an implementation achieves the required data transformation goal step by step.
Accordingly, every implementation is supposed to provide an easy to follow visual documentation.

pywrangler is not committed to a single computation backend like pandas. Instead, data wranglers
are defined abstractly and need to be implemented concretely in regard to the specific
computation backend. In general, all python related computation engines can be supported
if corresponding implementations are provided (e.g pandas, dask, vaex, modin and pyspark).
pywrangler attempts to provide at least one implementation for smallish data (single node
computation e.g. pandas) and largish data (distributed computation e.g. pyspark).

Moreover, one computation engine may have several implementations with varying trade-offs, too.
In order to identify trade-offs, pywrangler aims to offer benchmarking utilities to
compare different implementations in regard to cpu and memory usage.

To make pywranlger integrate well with standard data wrangling workflows, data wranglers confirm to
the scikit-learn API as a common standard for data transformation pipelines.

Goals
=====

- describe data transformations independent of computation engine
- closely define data transformation requirements through extensive tests
- visually document implementation logic of data transformations step by step
- provide concrete implementations for small and big data computation engines
- add benchmarking utilities to compare different implementations to identify tradeoffs
- follow scikit-learn API for easy integration for data pipelines

Non-Goals
=========

- always support all computation engines for a single data transformation
- handle *extract* and *load* stages of ETL pipelines

-----------------
Future directions
-----------------

Computation engines may come and go. Pandas is still very popular for single node
computation. Vaex slowly catches up. Pyspark and dask are both very popular in the
realm of distributed computation engines. There may be new engines in the future,
perhaps pandas 2 or a computation engine originating from the Apache Arrow project.

In any case, what remains is the description of the data transformations. More
importantly, computation backend independent tests manifest the requirements of
data transformations for future computation engines. This is will be a major lasting
value of pywrangler.

What has been totally neglected so far by the pywrangler project is the importance
of sql. Due to its declarative nature, sql offers a computation engine independent way
to formulate data transformations. They are applicable to any computation engine that
supports sql. Therefore, one major goal is to add a sql backend that produces the
required sql code to perform a specific data transformation.

.. _ETL: https://en.wikipedia.org/wiki/Extract,_transform,_load
