#!/bin/bash

# To test individual pandas/pyspark/dask versions, only corresponding tests
# need to be run (e.g test pandas 0.19.2, and only run pandas tests while
# ignoring all pyspark and dask tests). To achieve this, the corresponding
# pytest mark is passed to pytest.

if [[ $ENV_STRING == *"pandas"* ]]; then
  MARKS="-- -m pandas"

elif [[ $ENV_STRING == *"pyspark"* ]]; then
  MARKS="-- -m pyspark"

elif [[ $ENV_STRING == *"dask"* ]]; then
  MARKS="-- -m dask"
fi

MARKS="-- -m pandas"

tox -e $(echo py$TRAVIS_PYTHON_VERSION-$ENV_STRING | tr -d .) $MARKS
