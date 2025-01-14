#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    "xarray>=0.14",
    "numpy>=1.11",
    "scikit-learn>=0.22",
    "fsspec>=0.6.2",
    "pyyaml>=5.1.2",
    "tensorflow>=2.2.0",
    "tensorflow-addons>=0.11.2",
    "typing_extensions>=3.7.4.3",
    "dacite>=1.6.0",
    "wandb>=0.12.1",
    # fv3fit also depends on fv3gfs-util>=0.6.0, but pip-compile does not work
    # for packages not hosted on pypi.
]

setup_requirements = []

test_requirements = ["pytest"]

setup(
    author="Vulcan Technologies LLC",
    author_email="jeremym@vulcan.com",
    python_requires=">=3.6.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="FV3Fit is used to train machine learning models.",
    install_requires=requirements,
    dependency_links=["../loaders/", "../vcm/"],
    extras_require={},
    license="BSD license",
    long_description="FV3Fit is used to train machine learning models.",
    include_package_data=True,
    keywords="fv3fit",
    name="fv3fit",
    packages=find_packages(include=["fv3fit", "fv3fit.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/VulcanClimateModeling/fv3fit",
    version="0.1.0",
    zip_safe=False,
)
