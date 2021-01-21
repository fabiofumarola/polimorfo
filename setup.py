#!/usr/bin/env python
"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt', 'r') as fh:
    requirements = fh.read().split('\n')

with open('requirements_dev.txt', 'r') as fh:
    test_requirements = fh.read().split('\n')

setup_requirements = [
    'pytest-runner',
]

setup(
    author="Fabio Fumarola",
    author_email='fabiofumarola@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description=
    "a dataset loader and converter for object detection segmentation and classification",
    entry_points={
        'console_scripts': ['polimorfo=polimorfo.cli:main',],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='polimorfo',
    name='polimorfo',
    packages=find_packages(include=['polimorfo', 'polimorfo.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/fabiofumarola/polimorfo',
    version='0.9.53',
    zip_safe=False,
)
