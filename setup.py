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
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDA_HOME,
    HIP_HOME
)

# Suppress deprecated 'has_mps' warning
warnings.filterwarnings("ignore", message=".*'has_mps' is deprecated.*")

# Check if MPS (Metal Performance Shaders) is supported on macOS
def check_if_mps_supported():
    """
    Checks if MPS (Metal Performance Shaders) is available for PyTorch on macOS.
    """
    if sys.platform == "darwin" and torch.backends.mps.is_built():
        return True
    return False

MPS_BUILD = check_if_mps_supported()

# Setup variables for the package
this_dir = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = "mamba_ssm"
BASE_WHEEL_URL = "https://github.com/state-spaces/mamba/releases/download/{tag_name}/{wheel_name}"

# Get platform details
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

def get_package_version():
    with open(Path(this_dir) / PACKAGE_NAME / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("MAMBA_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)

# Prepare the CUDA or HIP specific compilation flags
def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]

# Setup for MPS or CUDA-specific builds
def setup_mps_or_cuda():
    if sys.platform == "darwin" and MPS_BUILD:
        print("\n\nMPS support enabled. Building for Metal (Apple GPU)...\n\n")
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17"],
            "mps": ["-O3", "-std=c++17", "-framework", "Metal", "-mmacosx-version-min=10.14"]
        }
        ext_modules = [
            CppExtension(
                name="selective_scan_mps",
                sources=[
                    "csrc/selective_scan/selective_scan.cpp",  # Ensure your code is Metal compatible
                    "csrc/selective_scan/selective_scan_fwd_mps.cpp",  # Example: Use MPS-specific source files
                ],
                extra_compile_args=extra_compile_args,
                include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
            )
        ]
        return ext_modules, extra_compile_args
    
    else:
        # CUDA or HIP specific setup for non-MacOS systems
        cc_flag = []
        if sys.platform == "linux" or sys.platform == "win32":
            # Additional CUDA setup
            if CUDA_HOME is not None:
                _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
                if bare_metal_version < Version("11.6"):
                    raise RuntimeError(f"{PACKAGE_NAME} is only supported on CUDA 11.6 and above.")
                cc_flag += ["-gencode", "arch=compute_53,code=sm_53"]
                cc_flag += ["-gencode", "arch=compute_62,code=sm_62"]
                cc_flag += ["-gencode", "arch=compute_70,code=sm_70"]
                
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                ] + cc_flag)
        }

        ext_modules = [
            CppExtension(
                name="selective_scan_cuda",
                sources=[
                    "csrc/selective_scan/selective_scan.cpp",
                    "csrc/selective_scan/selective_scan_fwd_fp32.cu",
                    "csrc/selective_scan/selective_scan_fwd_fp16.cu",
                ],
                extra_compile_args=extra_compile_args,
                include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
            )
        ]

        return ext_modules, extra_compile_args

# Setup the package setup function
ext_modules, extra_compile_args = setup_mps_or_cuda()

# Final package setup
setup(
    name=PACKAGE_NAME,
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
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/state-spaces/mamba",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": _bdist_wheel, "build_ext": BuildExtension},
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
