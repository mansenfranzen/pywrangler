"""Test dask base wrangler.

"""

import pytest

try:
    from pywrangler.wranglers.dask.base import DaskWrangler
except ImportError:
    DaskWrangler = None


@pytest.mark.dask
def test_dask_base_wrangler_engine():
    wrangler = DaskWrangler()

    assert wrangler.computation_engine == "dask"
