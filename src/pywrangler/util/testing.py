"""This module contains testing utility.

"""

from typing import Type


def concretize_abstract_wrangler(wrangler_class: Type) -> Type:
    """Makes abstract wrangler classes instantiable for testing purposes by
    implementing abstract methods of `BaseWrangler`.

    Parameters
    ----------
    wrangler_class: Type
        Class object to inherit from while overriding abstract methods.

    Returns
    -------
    concrete_class: Type
        Concrete class usable for testing.

    """

    class ConcreteWrangler(wrangler_class):

        @property
        def preserves_sample_size(self):
            return super().preserves_sample_size

        @property
        def computation_engine(self):
            return super().computation_engine

        def fit(self, *args, **kwargs):
            return super().fit(*args, **kwargs)

        def fit_transform(self, *args, **kwargs):
            return super().fit_transform(*args, **kwargs)

        def transform(self, *args, **kwargs):
            return super().transform(*args, **kwargs)

    return ConcreteWrangler
