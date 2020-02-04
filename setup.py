# Copyright (c) LinkedIn Corporation. All rights reserved. Licensed under the BSD-2 Clause license.
# See LICENSE in the project root for license information.
import setuptools
setuptools.setup(
    name='detext',
    version='0.0.1',
    package_dir={'': 'src'},
    packages=setuptools.find_packages('src'),
    include_package_data=True,
    install_requires=[
        'numpy<1.17',
        'tensorflow==1.12.0',
        'tensorflow_ranking==0.1.2',
        'gast==0.2.2'
    ],
    tests_require=[
        'pytest',
    ])
