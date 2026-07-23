# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Installation script for the 'isaac_auto_data' python packages."""

from setuptools import find_packages, setup

ISAAC_AUTO_DATA_VERSION_NUMBER = "0.1.0"

RUNTIME_DEPS = [
    "rapprentice @ git+https://github.com/masoudmoghani/rapprentice.git@512894df66ec4d0e608e409f7dac4ef448578f78",
    "typing_extensions",
    "pyzmq",
]

DEV_DEPS = [
    "pytest",
    "debugpy",
]

setup(
    name="isaac_auto_data",
    version=ISAAC_AUTO_DATA_VERSION_NUMBER,
    description="Isaac Auto Data.",
    py_modules=["sitecustomize"],
    packages=find_packages(
        include=[
            "isaac_autodata_core*",
            "isaac_autodata_interfaces*",
            "isaac_autodata_utils*",
            "isaac_autodata_examples*",
        ]
    ),
    python_requires=">=3.12",
    install_requires=RUNTIME_DEPS,
    extras_require={
        "dev": DEV_DEPS,
    },
    zip_safe=False,
)
