"""This module contains implementations of the interval identifier wrangler.

"""

import pandas as pd

from pywrangler.wranglers.interfaces import IntervalIdentifier
from pywrangler.wranglers.pandas.base import PandasWrangler


class NaiveIterator(IntervalIdentifier, PandasWrangler):
    pass

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([0, 0, 0, 0])

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
