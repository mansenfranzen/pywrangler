"""The module contains package wide custom exceptions and warnings.

"""


class NotProfiledError(ValueError, AttributeError):
    """Exception class to raise if profiling results are acquired before
    calling `profile`.

    This class inherits from both ValueError and AttributeError to help with
    exception handling

    """
