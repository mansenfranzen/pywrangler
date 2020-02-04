=========
Changelog
=========

Version 0.1.0
=============

This is the initial release of pywrangler.

- Enable raw, valid and enumerated return type for ``IntervalIdentifier`` (`#23 <https://github.com/mansenfranzen/pywrangler/pull/23>`_).
- Enable variable sequence lengths for ``IntervalIdentifier`` (`#23 <https://github.com/mansenfranzen/pywrangler/pull/23>`_).
- Add ``DataTestCase`` and ``TestCollection`` as standards for data centric test cases (`#23 <https://github.com/mansenfranzen/pywrangler/pull/23>`_).
- Add computation engine independent data abstraction ``PlainFrame`` (`#23 <https://github.com/mansenfranzen/pywrangler/pull/23>`_).
- Add ``VectorizedCumSum`` pyspark implementation for ``IntervalIdentifier`` wrangler (`#7 <https://github.com/mansenfranzen/pywrangler/pull/7>`_).
- Add benchmark utilities for pandas, spark and dask wranglers (`#5 <https://github.com/mansenfranzen/pywrangler/pull/5>`_).
- Add sequential ``NaiveIterator`` and vectorized ``VectorizedCumSum`` pandas implementations for ``IntervalIdentifier`` wrangler (`#2 <https://github.com/mansenfranzen/pywrangler/pull/2>`_).
- Add ``PandasWrangler`` (`#2 <https://github.com/mansenfranzen/pywrangler/pull/2>`_).
- Add ``IntervalIdentifier`` wrangler interface (`#2 <https://github.com/mansenfranzen/pywrangler/pull/2>`_).
- Add ``BaseWrangler`` class defining wrangler interface (`#1 <https://github.com/mansenfranzen/pywrangler/pull/1>`_).
- Enable ``pandas`` and ``pyspark`` testing on TravisCI (`#1 <https://github.com/mansenfranzen/pywrangler/pull/1>`_).
