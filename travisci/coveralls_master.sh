#!/bin/bash

# Different pandas/pyspark/dask versions are tested separately to avoid
# irrelevant tests to be run. For example, no spark tests need to be run
# when pandas wranglers are tested on older pandas versions.

# However, code coverage drops due to many skipped tests. Therefore, there is a
# master version (marked via env variables) which includes all tests for
# pandas/pyspark/dask for the newest available versions which is subject to
# code coverage. Non master versions will not be included in code coverage.

if [[ $ENV_STRING == *"master"* ]]; then
  coveralls --verbose
  codecov
fi
