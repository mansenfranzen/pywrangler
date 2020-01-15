"""Test dask base wrangler.

isort:skip_file
"""

import pytest

from pywrangler.util.testing.util import concretize_abstract_wrangler

pytestmark = pytest.mark.dask  # noqa: E402
dask = pytest.importorskip("dask")  # noqa: E402

from pywrangler.dask.base import DaskWrangler


def test_dask_base_wrangler_engine():
    wrangler = concretize_abstract_wrangler(DaskWrangler)()

    assert wrangler.computation_engine == "dask"
