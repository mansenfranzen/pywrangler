#!/bin/bash
# For spark tests add Java

set -e

echo "Spark env: $SPARK"
echo "Pandas env: $PANDAS"
echo "Java Home: $JAVA_HOME"

if [[ -n $SPARK ]]; then
    echo "Spark given"
else
    echo "installing pandas instead"
fi
