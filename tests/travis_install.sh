#!/bin/bash
# For spark tests add Java

set -e

echo "Spark env: $SPARK"
echo "Pandas env: $PANDAS"
echo "Java Home: $JAVA_HOME"

if [[ -n $SPARK ]]; then
    echo "Spark given"
    export TOX_ENV_STRING="spark$SPARK"
else
    echo "installing pandas instead"
    export TOX_ENV_STRING="pandas$PANDAS"
fi

echo "Test string: $TOX_ENV_STRING"
