"""Test dask base wrangler.

isort:skip_file
"""

import pytest

pytestmark = pytest.mark.dask  # noqa: E402
dask = pytest.importorskip("dask")  # noqa: E402

from pywrangler.dask.base import DaskWrangler
from pywrangler.util.testing import concretize_abstract_wrangler


def test_dask_base_wrangler_engine():
    wrangler = concretize_abstract_wrangler(DaskWrangler)()

    assert wrangler.computation_engine == "dask"
