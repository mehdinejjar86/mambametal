import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform
import shutil

from setuptools import setup, find_packages
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import BuildExtension  # No CUDA_HOME import, as we don't need it

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Get the platform name, modified for macOS (MPS)
def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))

# Define get_package_version() function for package versioning
def get_package_version():
    with open(Path(os.path.dirname(os.path.abspath(__file__))) / "mamba_ssm" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("MAMBA_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)

# Modify the build logic to not use CUDA or HIP (MPS-specific changes)
ext_modules = []  # Empty, as we're not building GPU extensions

cmdclass = {}

# Set up the package with no extensions and just necessary dependencies
setup(
    name="mamba_ssm",
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "mamba_ssm.egg-info",
        )
    ),
    author="Tri Dao, Albert Gu",
    author_email="tri@tridao.me, agu@cs.cmu.edu",
    description="Mamba state-space model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/state-spaces/mamba",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS",  # Change the OS classifier for macOS
    ],
    ext_modules=ext_modules,  # No GPU extension (MPS or CUDA) for this build
    cmdclass={"bdist_wheel": _bdist_wheel, "build_ext": BuildExtension}  # Simple build extension logic
    if ext_modules else {
        "bdist_wheel": _bdist_wheel,
    },
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "packaging",
        "ninja",
        "einops",
        "triton",
        "transformers",
    ],
)
