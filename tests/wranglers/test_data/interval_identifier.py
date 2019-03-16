"""This module contains the test data relevant for the interval identifier
wrangler.

"""


import pandas as pd

RANDOM_STATE = 3
COLUMNS_STD = ("order", "groupby", "marker")
COLUMNS_MUL = ("order1", "order2", "groupby1", "groupby2", "marker")


def _return_dfs(data, target_column_name, index=None, shuffle=False):
    """Helper function to return input and output dataframes.

    """

    # determine the correct column names
    n_cols = len(data[0])
    if n_cols == 4:
        parameter_column_names = COLUMNS_STD
    elif n_cols == 6:
        parameter_column_names = COLUMNS_MUL
    else:
        raise ValueError("Wrong test data provided.")

    # create dataframes from input data
    columns = parameter_column_names + (target_column_name, )
    df_in = pd.DataFrame(data, columns=columns, index=index)

    if shuffle:
        df_in = df_in.sample(frac=1, replace=False, random_state=RANDOM_STATE)

    df_out = df_in.pop(target_column_name).to_frame()

    return df_in, df_out


def no_interval(start, end, noise, target_column_name, shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[1,     1,       noise,  0],
            [2,     1,       noise,  0],
            [3,     1,       noise,  0],
            [4,     1,       noise,  0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def start_marker_left_open(start, end, noise, target_column_name, shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[1,     1,       noise,  0],
            [2,     1,       noise,  0],
            [3,     1,       start,  0],
            [4,     1,       noise,  0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def end_marker_begins(start, end, noise, target_column_name, shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[1,     1,       end,    0],
            [2,     1,       noise,  0],
            [3,     1,       start,  1],
            [4,     1,       end,    1]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def invalid_start(start, end, noise, target_column_name, shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[1,     1,       noise,  0],
            [2,     1,       start,  0],
            [3,     1,       start,  1],
            [4,     1,       end,    1],
            [5,     1,       noise,  0],
            [6,     1,       start,  2],
            [7,     1,       end,    2]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def invalid_end(start, end, noise, target_column_name, shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[1,     1,       noise,  0],
            [2,     1,       start,  1],
            [3,     1,       end,    1],
            [4,     1,       end,    0],
            [5,     1,       noise,  0],
            [6,     1,       start,  2],
            [7,     1,       end,    2]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def single_interval(start, end, noise, target_column_name, shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[1,     1,       noise,  0],
            [2,     1,       start,  1],
            [3,     1,       end,    1],
            [4,     1,       noise,  0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def starts_with_single_interval(start, end, noise, target_column_name,
                                shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[1,     1,       start,  1],
            [2,     1,       end,    1],
            [3,     1,       noise,  0],
            [4,     1,       noise,  0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def ends_with_single_interval(start, end, noise, target_column_name,
                              shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[1,     1,       noise,  0],
            [2,     1,       noise,  0],
            [3,     1,       start,  1],
            [4,     1,       end,    1]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def single_interval_spanning(start, end, noise, target_column_name,
                             shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[1,     1,       noise,  0],
            [2,     1,       start,  1],
            [3,     1,       noise,  1],
            [4,     1,       end,    1],
            [5,     1,       noise,  0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def multiple_intervals(start, end, noise, target_column_name, shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[1,     1,       noise,  0],
            [2,     1,       start,  1],
            [3,     1,       end,    1],
            [4,     1,       noise,  0],
            [5,     1,       start,  2],
            [6,     1,       end,    2],
            [7,     1,       noise,  0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def multiple_intervals_spanning(start, end, noise, target_column_name,
                                shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[1,     1,       noise,  0],
            [2,     1,       start,  1],
            [3,     1,       end,    1],
            [4,     1,       noise,  0],
            [5,     1,       start,  2],
            [6,     1,       noise,  2],
            [7,     1,       end,    2],
            [8,     1,       noise,  0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def multiple_intervals_spanning_unsorted(start, end, noise,
                                         target_column_name, shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[7,     1,       end,    2],
            [5,     1,       start,  2],
            [2,     1,       start,  1],
            [4,     1,       noise,  0],
            [6,     1,       noise,  2],
            [1,     1,       noise,  0],
            [3,     1,       end,    1],
            [8,     1,       noise,  0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def groupby_single_intervals(start, end, noise, target_column_name, shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[1,     1,       noise,  0],
            [2,     1,       start,  1],
            [3,     1,       end,    1],
            [4,     1,       noise,  0],

            [5,     2,       start,  1],
            [6,     2,       noise,  1],
            [7,     2,       end,    1],
            [8,     2,       noise,  0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def groupby_multiple_intervals(start, end, noise, target_column_name,
                               shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[1,     1,       noise,  0],
            [2,     1,       start,  1],
            [3,     1,       end,    1],
            [4,     1,       noise,  0],

            [5,     2,       start,  1],
            [6,     2,       noise,  1],
            [7,     2,       end,    1],
            [8,     2,       noise,  0],
            [9,     2,       noise,  0],
            [10,    2,       start,  2],
            [11,    2,       noise,  2],
            [12,    2,       end,    2],
            [13,    2,       start,  3],
            [14,    2,       end,    3]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def groupby_multiple_intervals_reverse(start, end, noise, target_column_name,
                                       shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[1,     1,       noise,  0],
            [2,     1,       start,  0],
            [3,     1,       end,    0],
            [4,     1,       noise,  0],

            [5,     2,       start,  0],
            [6,     2,       noise,  0],
            [7,     2,       end,    2],
            [8,     2,       noise,  2],
            [9,     2,       noise,  2],
            [10,    2,       start,  2],
            [11,    2,       noise,  0],
            [12,    2,       end,    1],
            [13,    2,       start,  1],
            [14,    2,       end,    0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def groupby_multiple_more_intervals(start, end, noise, target_column_name,
                                    shuffle):

    # cols:  order, groupby, marker, iid"""
    data = [[1,     1,       noise,  0],
            [2,     1,       start,  1],
            [3,     1,       end,    1],
            [4,     1,       noise,  0],

            [5,     2,       start,  1],
            [6,     2,       noise,  1],
            [7,     2,       end,    1],
            [8,     2,       noise,  0],
            [9,     2,       noise,  0],

            [10,    3,       start,  1],
            [11,    3,       noise,  1],
            [12,    3,       end,    1],
            [13,    3,       start,  2],
            [14,    3,       end,    2]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def multiple_groupby_order_columns(start, end, noise, target_column_name,
                                   shuffle):
    # cols:  order1, order2, groupby1, groupby2, marker, iid"""
    data = [[1,      1,      1,        1,        noise,  0],
            [1,      2,      1,        1,        start,  1],
            [2,      1,      1,        1,        end,    1],
            [2,      2,      1,        1,        noise,  0],

            [3,      1,      1,        2,        start,  1],
            [3,      2,      1,        2,        noise,  1],
            [4,      1,      1,        2,        end,    1],
            [4,      2,      1,        2,        noise,  0],

            [1,      1,      2,        1,        start,  1],
            [1,      2,      2,        1,        end,    1],
            [2,      1,      2,        1,        start,  2],
            [2,      2,      2,        1,        end,    2],

            [3,      1,      2,        2,        start,  1],
            [3,      2,      2,        2,        noise,  1],
            [4,      1,      2,        2,        end,    1],
            [4,      2,      2,        2,        noise,  0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def multiple_groupby_order_columns_reverse(start, end, noise,
                                           target_column_name, shuffle):
    # cols:  order1, order2, groupby1, groupby2, marker, iid"""
    data = [[1,      1,      1,        1,        end,    2],
            [1,      2,      1,        1,        start,  2],
            [2,      1,      1,        1,        end,    1],
            [2,      2,      1,        1,        start,  1],

            [3,      1,      1,        2,        start,  0],
            [3,      2,      1,        2,        end,    1],
            [4,      1,      1,        2,        noise,  1],
            [4,      2,      1,        2,        start,  1],

            [1,      1,      2,        1,        start,  0],
            [1,      2,      2,        1,        end,    1],
            [2,      1,      2,        1,        start,  1],
            [2,      2,      2,        1,        end,    0],

            [3,      1,      2,        2,        start,  0],
            [3,      2,      2,        2,        noise,  0],
            [4,      1,      2,        2,        end,    0],
            [4,      2,      2,        2,        noise,  0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)


def multiple_groupby_order_columns_with_invalids(start, end, noise,
                                                 target_column_name, shuffle):
    # cols:  order1, order2, groupby1, groupby2, marker, iid"""
    data = [[1,      1,      1,        1,        start,  0],
            [1,      2,      1,        1,        start,  1],
            [2,      1,      1,        1,        end,    1],
            [2,      2,      1,        1,        noise,  0],

            [3,      1,      1,        2,        start,  1],
            [3,      2,      1,        2,        noise,  1],
            [4,      1,      1,        2,        end,    1],
            [4,      2,      1,        2,        end,    0],

            [5,      1,      1,        2,        start,  0],
            [5,      2,      1,        2,        start,  0],
            [5,      3,      1,        2,        start,  2],
            [5,      4,      1,        2,        end,    2],

            [3,      1,      2,        2,        start,  1],
            [3,      2,      2,        2,        end,    1],
            [4,      1,      2,        2,        end,    0],
            [4,      2,      2,        2,        end,    0]]

    return _return_dfs(data, target_column_name, shuffle=shuffle)
