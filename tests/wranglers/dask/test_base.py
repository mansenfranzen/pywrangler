"""Test dask base wrangler.

isort:skip_file
"""

import pytest

pytestmark = pytest.mark.dask  # noqa: E402
dask = pytest.importorskip("dask")  # noqa: E402

from pywrangler.wranglers.dask.base import DaskWrangler


def test_dask_base_wrangler_engine():
    wrangler = DaskWrangler()

    assert wrangler.computation_engine == "dask"
