# Copyright (c) LinkedIn Corporation. All rights reserved. Licensed under the BSD-2 Clause license.
# See LICENSE in the project root for license information.
import setuptools
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='detext',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=["Programming Language :: Python :: 3.7",
                 "Intended Audience :: Science/Research",
                 "Intended Audience :: Developers",
                 "License :: OSI Approved"],
    license='BSD-2-CLAUSE',
    # DO NOT CHANGE: version should be incremented by bump2version when releasing. See pypi_release.sh
    version='2.0.9',
    package_dir={'': 'src'},
    packages=setuptools.find_packages('src'),
    include_package_data=True,
    install_requires=['numpy<1.17', 'smart-arg==0.1.1', 'tensorflow==1.14.0', 'tensorflow_ranking==0.1.4', 'gast==0.2.2', 'bump2version==1.0.0', 'twine==3.2.0'],  # noqa: E501
    tests_require=[
        'pytest',
    ])
