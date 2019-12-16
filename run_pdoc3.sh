#!/bin/bash
set -o errexit

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd ${PROJECT_PATH}

rm -r -f docs

# This script requires pdoc3 at least 0.5.2:
# pip3 install pdoc3
python3 -m pdoc uisrnn --html --output-dir=docs

mv docs/uisrnn/* docs/

rm -r docs/uisrnn

popd
