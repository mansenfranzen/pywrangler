"""Test pandas base wrangler.

"""

import pytest

import pandas as pd

from pywrangler.pandas.base import PandasWrangler
from pywrangler.util.testing.util import concretize_abstract_wrangler

pytestmark = pytest.mark.pandas


def test_pandas_base_wrangler_engine():
    wrangler = concretize_abstract_wrangler(PandasWrangler)()

    assert wrangler.computation_engine == "pandas"


@pytest.mark.parametrize("preserves_sample_size", [True, False])
def test_pandas_wrangler_validate_output_shape_raises(preserves_sample_size):
    class DummyWrangler(PandasWrangler):
        @property
        def preserves_sample_size(self):
            return preserves_sample_size

    wrangler = concretize_abstract_wrangler(DummyWrangler)()

    df1 = pd.DataFrame([0] * 10)
    df2 = pd.DataFrame([0] * 20)

    if preserves_sample_size:
        with pytest.raises(ValueError):
            wrangler._validate_output_shape(df1, df2)
    else:
        wrangler._validate_output_shape(df1, df2)
