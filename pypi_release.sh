#!/bin/bash
# Make sure the VERSION in setup.py is updated before running the script

# Exit when any command fails
set -e

# Build the source distribution
python setup.py sdist

# Upload to pypi, username and password required (make sure you have permission for releasing detext packages)
twine upload dist/*

echo "Pypi package releasing succeeded!"
