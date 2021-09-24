#!/bin/bash
# Make sure all changes are committed before running this script for releasing.
# See RELEASING.md for more instructions on running the script for releasing.

# Usage:
# bash pypi_release.sh <part>
# <part> must equal to "patch", "minor", or "major".
# e.g.:
# 0.0.1 -> 0.0.2:
# bash pypi_release.sh patch
# 0.0.1 -> 0.1.0:
# bash pypi_release.sh minor
# 0.0.1 -> 1.0.0:
# bash pypi_release.sh major

# Please follow the following best practices for versioning:
# breaking changes are indicated by increasing the major number (high risk),
# new non-breaking features increment the minor number (medium risk)
# all other non-breaking changes increment the patch number (lowest risk).

# Exit when any command fails
set -e
# Cleaning up dist directory for old releases
rm -rf dist/

# Check input argument <part>
if [ "$1" != "patch" ] && [ "$1" != "minor" ] && [ "$1" != "major" ]; then
  echo "Must include correct <patch> argument. Eg., ash pypi_release.sh patch"
  exit
fi

# Install/upgrade needed pypi packages
pip install -U bump2version twine

# Increment version with bumpversion. Version format: {major}.{minor}.{patch}
echo "Incrementing DeText $1 version."
bump2version "$1"

# Build the source distribution
echo "******** Preparing pypi package..."
python setup.py sdist

# Build the source distribution without dependencies added for LI internal use.
echo "******** Preparing pypi package without dependencies..."
# Temporarily save setup.py for recover
cp setup.py setup.py.tmp

# Rename the pypi package name and install_requires entries
if [[ "$OSTYPE" == "darwin"* ]]; then
  sed -i "" "s/name='detext'/name='detext-nodep'/" setup.py
  sed -i "" "s/install_requires=.*/install_requires=[],/g" setup.py
else
  sed -i "s/name='detext'/name='detext-nodep'/" setup.py
  sed -i "s/install_requires=.*/install_requires=[],/g" setup.py
fi

python setup.py sdist
# Recover original setup.py
rm setup.py
mv setup.py.tmp setup.py

# Upload to pypi, username and password required (make sure you have permission for releasing detext packages)
echo "******** Uploading all sdist under dist/"
twine upload dist/*

echo "******** Pypi package releasing succeeded!"
