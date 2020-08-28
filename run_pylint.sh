#!/bin/bash
set -o errexit

# This script requires pylint. Please run this line if you haven't installed it:
# pip3 install -q pylint

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd ${PROJECT_PATH}

# Run pylint.
python3 -m pylint **/*.py *.py\
    --indent-string="  " \
    --max-line-length=80 \
    --disable=E1101,R0913,W0221,R0903,R0902,R0914,R0912,R0915,R1723,W1114,W0223

popd
