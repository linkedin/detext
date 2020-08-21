#!/bin/bash
# Make sure the VERSION in setup.py is updated before running the script

# Exit when any command fails
set -e

# Build the source distribution
echo "******** Preparing pypi package..."
python setup.py sdist

# Build the source distribution without dependencies added for LI internal use.
echo "******** Preparing pypi package without dependencies..."
# Temporarily save setup.py for recover
cp setup.py setup.py.tmp
# Rename the pypi package name
sed -i "" "s/name='detext'/name='li-detext'/" setup.py
# Remove install_requires entries
sed -i "" "s/install_requires=\[.*\]/install_requires=[]/g" setup.py
python setup.py sdist
# Recover original setup.py
rm setup.py
mv setup.py.tmp setup.py

# Upload to pypi, username and password required (make sure you have permission for releasing detext packages)
echo "******** Uploading all sdist under dist/"
twine upload dist/*

echo "******** Pypi package releasing succeeded!"
