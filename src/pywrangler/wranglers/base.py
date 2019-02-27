"""This module contains the BaseWrangler definition and the wrangler base
classes including wrangler descriptions and parameters.

"""


class BaseWrangler:
    """Defines the basic interface common to all data wranglers.

    In analogy to sklearn transformers (see link below), all wranglers have to
    implement `fit`, `transform` and `fit_transform` methods. In addition,
    parameters (e.g. column names) need to be provided via the `__init__`
    method.

    The `fit` method should contain any logic behind parameter validation (e.g.
    type, shape and other sanity checks). In general, `fit` ensures that the
    provided parameters match the given data. The `transform` method includes
    the actual computational transformation. The `fit_transform` simply applies
    the former methods in sequence.

    In contrast to sklearn, wranglers do only accept dataframes like objects
    (like pandas, spark or dask dataframes) as inputs to `fit` and `transform`.
    The relevant columns and their respective meaning is provided via the
    `__init__` method. In addition, wranglers may accept multiple input
    dataframes with different shapes. Also, the number of samples may also
    change between input and output (which is not allowed in sklearn). The
    `perserves_sample_size` indicates whether sample size (number of rows) may
    change during transformation.

    See also
    --------
    Sklearn contributor: guide https://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator

    """

    pass
