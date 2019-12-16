"""This module contains tests for testing utilities.

"""

import pytest

from pywrangler.base import BaseWrangler
from pywrangler.util.testing import concretize_abstract_wrangler


def test_concretize_abstract_wrangler():

    class Dummy(BaseWrangler):
        """Doc"""

        @property
        def computation_engine(self) -> str:
            return "engine"

    concrete_class = concretize_abstract_wrangler(Dummy)
    instance = concrete_class()

    assert instance.computation_engine == "engine"
    assert instance.__doc__ == "Doc"
    assert instance.__class__.__name__ == "Dummy"

    with pytest.raises(NotImplementedError):
        instance.preserves_sample_size
