# Copyright (c) LinkedIn Corporation. All rights reserved. Licensed under the BSD-2 Clause license.
# See LICENSE in the project root for license information.
import setuptools
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Update (increment) the version number before releasing. Checkout published versions at
# https://pypi.org/manage/project/detext/releases/
# Please follow the following best practices for versioning: Breaking changes are indicated by increasing the
# major number (high risk), new non-breaking features increment the minor number (medium risk) and all other
# non-breaking changes increment the patch number (lowest risk).
VERSION = '2.0.8'

setuptools.setup(
    name='detext',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=["Programming Language :: Python :: 3.7",
                 "Intended Audience :: Science/Research",
                 "Intended Audience :: Developers",
                 "License :: OSI Approved"],
    license='BSD-2-CLAUSE',
    version=VERSION,
    package_dir={'': 'src'},
    packages=setuptools.find_packages('src'),
    include_package_data=True,
    install_requires=['numpy<1.17', 'smart-arg==0.0.5', 'tensorflow==1.14.0', 'tensorflow_ranking==0.1.4', 'gast==0.2.2'],
    tests_require=[
        'pytest',
    ])
