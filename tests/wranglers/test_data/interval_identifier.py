"""This module contains the test data relevant for the interval identifier
wrangler.

"""


import pandas as pd

RANDOM_STATE = 3
COLUMNS_STD = ("order", "groupby", "marker")


def _return_dfs(data, target_column_name, parameter_column_names=COLUMNS_STD,
                index=None, shuffle=False):
    """Helper function to return input and output dataframes.

    """

    columns = parameter_column_names + (target_column_name, )
    df_in = pd.DataFrame(data, columns=columns, index=index)

    if shuffle:
        df_in = df_in.sample(frac=1, replace=False, random_state=RANDOM_STATE)

    df_out = df_in.pop(target_column_name).to_frame()

    return df_in, df_out


def no_interval(begin, close, noise, target_column_name, shuffle):

    data = [[1, 1, noise, 0],
            [2, 1, noise, 0],
            [3, 1, noise, 0],
            [4, 1, noise, 0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def single_interval(begin, close, noise, target_column_name, shuffle):

    data = [[1, 1, noise, 0],
            [2, 1, begin, 1],
            [3, 1, close, 1],
            [4, 1, noise, 0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def starts_with_single_interval(begin, close, noise, target_column_name,
                                shuffle):

    data = [[1, 1, begin, 1],
            [2, 1, close, 1],
            [3, 1, noise, 0],
            [4, 1, noise, 0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def ends_with_single_interval(begin, close, noise, target_column_name,
                              shuffle):

    data = [[1, 1, noise, 0],
            [2, 1, noise, 0],
            [3, 1, begin, 1],
            [4, 1, close, 1]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def single_interval_spanning(begin, close, noise, target_column_name,
                             shuffle):

    data = [[1, 1, noise, 0],
            [2, 1, begin, 1],
            [3, 1, noise, 1],
            [4, 1, close, 1],
            [5, 1, noise, 0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def multiple_intervals(begin, close, noise, target_column_name, shuffle):

    data = [[1, 1, noise, 0],
            [2, 1, begin, 1],
            [3, 1, close, 1],
            [4, 1, noise, 0],
            [5, 1, begin, 2],
            [6, 1, close, 2],
            [7, 1, noise, 0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def multiple_intervals_spanning(begin, close, noise, target_column_name,
                                shuffle):

    data = [[1, 1, noise, 0],
            [2, 1, begin, 1],
            [3, 1, close, 1],
            [4, 1, noise, 0],
            [5, 1, begin, 2],
            [6, 1, noise, 2],
            [7, 1, close, 2],
            [8, 1, noise, 0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def multiple_intervals_spanning_unsorted(begin, close, noise,
                                         target_column_name, shuffle):

    data = [[7, 1, close, 2],
            [5, 1, begin, 2],
            [2, 1, begin, 1],
            [4, 1, noise, 0],
            [6, 1, noise, 2],
            [1, 1, noise, 0],
            [3, 1, close, 1],
            [8, 1, noise, 0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)
