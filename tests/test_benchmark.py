"""This module contains tests for the benchmark utilities.

"""

import sys
import time

import pytest

from pywrangler.benchmark import (
    BaseProfiler,
    MemoryProfiler,
    TimeProfiler,
    allocate_memory
)
from pywrangler.exceptions import NotProfiledError

MIB = 2 ** 20


@pytest.fixture()
def func_no_effect():
    def func():
        pass

    return func


def test_allocate_memory_empty():
    memory_holder = allocate_memory(0)

    assert memory_holder is None


def test_allocate_memory_5mb():
    memory_holder = allocate_memory(5)

    assert sys.getsizeof(memory_holder) == 5 * (2 ** 20)


def test_base_profiler_not_implemented():
    base_profiler = BaseProfiler()

    for will_raise in ('profile', 'profile_report', 'less_is_better'):
        with pytest.raises(NotImplementedError):
            getattr(base_profiler, will_raise)()


def test_base_profiler_check_is_profiled():
    base_profiler = BaseProfiler()
    base_profiler._not_set = None
    base_profiler._is_set = "value"

    with pytest.raises(NotProfiledError):
        base_profiler._check_is_profiled(['_not_set'])

    base_profiler._check_is_profiled(['_is_set'])


def test_base_profiler_measurements_less_is_better(capfd):
    measurements = range(7)

    class Profiler(BaseProfiler):

        @property
        def less_is_better(self):
            return True

        def profile(self, *args, **kwargs):
            self._measurements = measurements
            return self

        def _pretty_formatter(self, value):
            return "{:.0f}".format(value)

    base_profiler = Profiler()
    base_profiler.profile_report()

    assert base_profiler.median == 3
    assert base_profiler.best == 0
    assert base_profiler.worst == 6
    assert base_profiler.std == 2
    assert base_profiler.runs == 7
    assert base_profiler.measurements == measurements

    out, _ = capfd.readouterr()
    assert out == "0 < 3 < 6 ± 2 (7 runs)\n"


def test_base_profiler_measurements_more_is_better(capfd):
    measurements = range(7)

    class Profiler(BaseProfiler):
        @property
        def less_is_better(self):
            return False

        def profile(self, *args, **kwargs):
            self._measurements = measurements
            return self

        def _pretty_formatter(self, value):
            return "{:.0f}".format(value)

    base_profiler = Profiler()
    base_profiler.profile_report()

    assert base_profiler.median == 3
    assert base_profiler.best == 6
    assert base_profiler.worst == 0
    assert base_profiler.std == 2
    assert base_profiler.runs == 7
    assert base_profiler.measurements == measurements

    out, _ = capfd.readouterr()
    assert out == "6 > 3 > 0 ± 2 (7 runs)\n"


def test_memory_profiler_mb_to_bytes():
    assert MemoryProfiler._mb_to_bytes(1) == 1048576
    assert MemoryProfiler._mb_to_bytes(1.5) == 1572864
    assert MemoryProfiler._mb_to_bytes(0.33) == 346030


def test_memory_profiler_return_self(func_no_effect):
    memory_profiler = MemoryProfiler(func_no_effect)
    assert memory_profiler.profile() is memory_profiler


def test_memory_profiler_measurements(func_no_effect):
    baselines = [0, 1, 2, 3]
    max_usages = [4, 5, 7, 8]
    measurements = [4, 4, 5, 5]

    memory_profiler = MemoryProfiler(func_no_effect)
    memory_profiler._baselines = baselines
    memory_profiler._max_usages = max_usages
    memory_profiler._measurements = measurements

    assert memory_profiler.less_is_better is True
    assert memory_profiler.max_usages == max_usages
    assert memory_profiler.baselines == baselines
    assert memory_profiler.measurements == measurements
    assert memory_profiler.median == 4.5
    assert memory_profiler.std == 0.5
    assert memory_profiler.best == 4
    assert memory_profiler.worst == 5
    assert memory_profiler.baseline_change == 1
    assert memory_profiler.runs == 4


def test_memory_profiler_no_side_effect(func_no_effect):
    baseline_change = MemoryProfiler(func_no_effect).profile().baseline_change

    assert baseline_change < 0.5 * MIB


def test_memory_profiler_side_effect():
    side_effect_container = []

    def side_effect():
        memory_holder = allocate_memory(5)
        side_effect_container.append(memory_holder)

        return memory_holder

    assert MemoryProfiler(side_effect).profile().baseline_change > 4.9 * MIB


def test_memory_profiler_no_increase(func_no_effect):
    memory_profiler = MemoryProfiler(func_no_effect).profile()
    print(memory_profiler.measurements)

    assert memory_profiler.median < MIB


@pytest.mark.xfail(reason="Succeeds locally but sometimes fails remotely due "
                          "to non deterministic memory management.")
def test_memory_profiler_increase():
    def increase():
        memory_holder = allocate_memory(30)
        return memory_holder

    assert MemoryProfiler(increase).profile().median > 29 * MIB


def test_time_profiler_return_self(func_no_effect):
    time_profiler = TimeProfiler(func_no_effect, 1)
    assert time_profiler.profile() is time_profiler


def test_time_profiler_measurements(func_no_effect):
    measurements = [1, 1, 3, 3]

    time_profiler = TimeProfiler(func_no_effect)
    time_profiler._measurements = measurements

    assert time_profiler.less_is_better is True
    assert time_profiler.median == 2
    assert time_profiler.std == 1
    assert time_profiler.best == 1
    assert time_profiler.runs == 4
    assert time_profiler.measurements == measurements


def test_time_profiler_repetitions(func_no_effect):
    time_profiler = TimeProfiler(func_no_effect, repetitions=10)
    assert time_profiler.repetitions == 10


def test_time_profiler_best():
    sleep = 0.0001

    def dummy():
        time.sleep(sleep)
        pass

    time_profiler = TimeProfiler(dummy, repetitions=1).profile()

    assert time_profiler.best >= sleep
