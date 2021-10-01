# Copyright (c) LinkedIn Corporation. All rights reserved. Licensed under the BSD-2 Clause license.
# See LICENSE in the project root for license information.
import setuptools
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

TF_VERSION_QUANTIFIER = '>=2.4,<2.5'
PACKAGES = ['smart-arg==0.4', 'bump2version', 'twine==3.2.0', f'tf-models-official{TF_VERSION_QUANTIFIER}',
            f'tensorflow{TF_VERSION_QUANTIFIER}', f'tensorflow-text{TF_VERSION_QUANTIFIER}', 'tensorflow_ranking',
            'future<0.14']

setuptools.setup(
    name='detext',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=["Programming Language :: Python :: 3.7",
                 "Programming Language :: Python :: 3.8",
                 "Programming Language :: Python :: 3.9",
                 "Topic :: Scientific/Engineering :: Artificial Intelligence",
                 "Topic :: Software Development :: Libraries",
                 "Intended Audience :: Science/Research",
                 "Intended Audience :: Developers",
                 "License :: OSI Approved"],
    license='BSD-2-CLAUSE',
    # DO NOT CHANGE: version should be incremented by bump2version when releasing. See pypi_release.sh
    version='3.1.0',
    package_dir={'': 'src'},
    packages=setuptools.find_packages('src'),
    include_package_data=True,
    install_requires=PACKAGES,
    tests_require=[
        'pytest',
    ])
