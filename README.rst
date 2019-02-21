===========================================
tswrangler - data wrangling for time series
===========================================

**tswrangler** is a collection of data wrangling implementations tailored for
time series. It aims to provide well tested solutions for commonly used tasks
such as:

- extracting intervals (given start and end values)
- mapping timestamps to time spans
- deduplicating overlapping time spans
- removing intermediate values

The reference implementation is built on top of pandas. Additionally,
distributed computation frameworks like pyspark and dask are supported.


Note
====

This project has been set up using PyScaffold 3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
