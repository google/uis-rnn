#!/bin/bash
set -o errexit

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd ${PROJECT_PATH}

rm -r -f docs

# This script requires pdoc:
# pip3 install pdoc
python3 -m pdoc uisrnn -o docs

popd
