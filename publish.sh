#!/bin/bash
set -o errexit

# This script requires these tools:
# pip3 install --user --upgrade setuptools wheel
# pip3 install --user --upgrade twine

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd ${PROJECT_PATH}

# clean up
rm -rf build
rm -rf dist
rm -rf uisrnn.egg-info

# build and upload
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*

popd
