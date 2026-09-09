# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installation script for the AutoData Python packages."""

from setuptools import find_packages, setup

AUTODATA_VERSION_NUMBER = "0.1.0"

RUNTIME_DEPS = [
    "typing_extensions",
    "pyzmq",
]

DEV_DEPS = [
    "pytest",
    "debugpy",
]

setup(
    name="autodata",
    version=AUTODATA_VERSION_NUMBER,
    description="Scalable robot demonstration generation for robot learning.",
    py_modules=["sitecustomize"],
    packages=find_packages(
        include=[
            "autodata_core*",
            "autodata_interfaces*",
            "autodata_utils*",
            "autodata_examples*",
        ]
    ),
    python_requires=">=3.12",
    install_requires=RUNTIME_DEPS,
    extras_require={
        "dev": DEV_DEPS,
    },
    zip_safe=False,
)
