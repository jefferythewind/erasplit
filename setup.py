#! /usr/bin/env python
#
# Copyright (C) 2007-2009 Cournapeau David <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
# License: 3-clause BSD

import sys
import os
from os.path import join
import platform
import shutil

from setuptools import Command, Extension, setup
from setuptools.command.build_ext import build_ext

import traceback
import importlib

try:
    import builtins
except ImportError:
    # Python 2 compat: just to be able to declare that Python >=3.8 is needed.
    import __builtin__ as builtins

# This is a bit (!) hackish: we are setting a global variable so that the main
# erasplit __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.
# TODO: can this be simplified or removed since the switch to setuptools
# away from numpy.distutils?
builtins.__erasplit_SETUP__ = True


DISTNAME = "erasplit"
DESCRIPTION = "Invariant Gradient Boosted Decision Tree Package - Era Splitting."
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "Tim DeLise"
MAINTAINER_EMAIL = "tdelise@gmail.com"
URL = "https://arxiv.org/abs/2309.14496"
DOWNLOAD_URL = "https://github.com/jefferythewind/erasplit"
LICENSE = "new BSD"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/jefferythewind/erasplit/issues",
    "Documentation": "https://github.com/jefferythewind/erasplit",
    "Source Code": "https://github.com/jefferythewind/erasplit",
}

# We can actually import a restricted version of erasplit that
# does not need the compiled code
import erasplit  # noqa
import erasplit._min_dependencies as min_deps  # noqa
from erasplit._build_utils import _check_cython_version  # noqa
from erasplit.externals._packaging.version import parse as parse_version  # noqa


VERSION = erasplit.__version__

# Custom clean command to remove build artifacts


class CleanCommand(Command):
    description = "Remove build artifacts from the source tree"

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, "PKG-INFO"))
        if remove_c_files:
            print("Will remove generated .c files")
        if os.path.exists("build"):
            shutil.rmtree("build")
        for dirpath, dirnames, filenames in os.walk("erasplit"):
            for filename in filenames:
                if any(
                    filename.endswith(suffix)
                    for suffix in (".so", ".pyd", ".dll", ".pyc")
                ):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in [".c", ".cpp"]:
                    pyx_file = str.replace(filename, extension, ".pyx")
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == "__pycache__":
                    shutil.rmtree(os.path.join(dirpath, dirname))


# Custom build_ext command to set OpenMP compile flags depending on os and
# compiler. Also makes it possible to set the parallelism level via
# and environment variable (useful for the wheel building CI).
# build_ext has to be imported after setuptools


class build_ext_subclass(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        if self.parallel is None:
            # Do not override self.parallel if already defined by
            # command-line flag (--parallel or -j)

            parallel = os.environ.get("erasplit_BUILD_PARALLEL")
            if parallel:
                self.parallel = int(parallel)
        if self.parallel:
            print("setting parallel=%d " % self.parallel)

    def build_extensions(self):
        from erasplit._build_utils.openmp_helpers import get_openmp_flag

        # Always use NumPy 1.7 C API for all compiled extensions.
        # See: https://numpy.org/doc/stable/reference/c-api/deprecations.html
        DEFINE_MACRO_NUMPY_C_API = (
            "NPY_NO_DEPRECATED_API",
            "NPY_1_7_API_VERSION",
        )
        for ext in self.extensions:
            ext.define_macros.append(DEFINE_MACRO_NUMPY_C_API)

        if erasplit._OPENMP_SUPPORTED:
            openmp_flag = get_openmp_flag()

            for e in self.extensions:
                e.extra_compile_args += openmp_flag
                e.extra_link_args += openmp_flag

        build_ext.build_extensions(self)

    def run(self):
        # Specifying `build_clib` allows running `python setup.py develop`
        # fully from a fresh clone.
        self.run_command("build_clib")
        build_ext.run(self)


cmdclass = {
    "clean": CleanCommand,
    "build_ext": build_ext_subclass,
}


def check_package_status(package, min_version):
    """
    Returns a dictionary containing a boolean specifying whether given package
    is up-to-date, along with the version string (empty string if
    not installed).
    """
    package_status = {}
    try:
        module = importlib.import_module(package)
        package_version = module.__version__
        package_status["up_to_date"] = parse_version(package_version) >= parse_version(
            min_version
        )
        package_status["version"] = package_version
    except ImportError:
        traceback.print_exc()
        package_status["up_to_date"] = False
        package_status["version"] = ""

    req_str = "scikit-learn requires {} >= {}.\n".format(package, min_version)

    instructions = (
        "Installation instructions are available on the "
        "scikit-learn website: "
        "http://scikit-learn.org/stable/install.html\n"
    )

    if package_status["up_to_date"] is False:
        if package_status["version"]:
            raise ImportError(
                "Your installation of {} {} is out-of-date.\n{}{}".format(
                    package, package_status["version"], req_str, instructions
                )
            )
        else:
            raise ImportError(
                "{} is not installed.\n{}{}".format(package, req_str, instructions)
            )


extension_config = {
    "__check_build": [
        {"sources": ["_check_build.pyx"]},
    ],
    "_loss": [
        {"sources": ["_loss.pyx.tp"]},
    ],
    "ensemble._hist_gradient_boosting": [
        {"sources": ["_gradient_boosting.pyx"], "include_np": True},
        {"sources": ["histogram.pyx"], "include_np": True},
        {"sources": ["era_histogram.pyx"], "include_np": True},
        {"sources": ["splitting.pyx"], "include_np": True},
        {"sources": ["era_splitting.pyx"], "include_np": True},
        {"sources": ["_binning.pyx"], "include_np": True},
        {"sources": ["_predictor.pyx"], "include_np": True},
        {"sources": ["_bitset.pyx"], "include_np": True},
        {"sources": ["common.pyx"], "include_np": True},
        {"sources": ["utils.pyx"], "include_np": True},
    ],
}

# Paths in `libraries` must be relative to the root directory because `libraries` is
# passed directly to `setup`
libraries = []


def configure_extension_modules():
    # Skip cythonization as we do not want to include the generated
    # C/C++ files in the release tarballs as they are not necessarily
    # forward compatible with future versions of Python for instance.
    if "sdist" in sys.argv or "--help" in sys.argv:
        return []

    from erasplit._build_utils import cythonize_extensions
    from erasplit._build_utils import gen_from_templates
    import numpy

    is_pypy = platform.python_implementation() == "PyPy"
    np_include = numpy.get_include()
    default_optimization_level = "O2"

    if os.name == "posix":
        default_libraries = ["m"]
    else:
        default_libraries = []

    default_extra_compile_args = []
    build_with_debug_symbols = (
        os.environ.get("erasplit_BUILD_ENABLE_DEBUG_SYMBOLS", "0") != "0"
    )
    if os.name == "posix":
        if build_with_debug_symbols:
            default_extra_compile_args.append("-g")
        else:
            # Setting -g0 will strip symbols, reducing the binary size of extensions
            default_extra_compile_args.append("-g0")

    cython_exts = []
    for submodule, extensions in extension_config.items():
        submodule_parts = submodule.split(".")
        parent_dir = join("erasplit", *submodule_parts)
        for extension in extensions:
            if is_pypy and not extension.get("compile_for_pypy", True):
                continue

            # Generate files with Tempita
            tempita_sources = []
            sources = []
            for source in extension["sources"]:
                source = join(parent_dir, source)
                new_source_path, path_ext = os.path.splitext(source)

                if path_ext != ".tp":
                    sources.append(source)
                    continue

                # `source` is a Tempita file
                tempita_sources.append(source)

                # Do not include pxd files that were generated by tempita
                if os.path.splitext(new_source_path)[-1] == ".pxd":
                    continue
                sources.append(new_source_path)

            gen_from_templates(tempita_sources)

            # By convention, our extensions always use the name of the first source
            source_name = os.path.splitext(os.path.basename(sources[0]))[0]
            if submodule:
                name_parts = ["erasplit", submodule, source_name]
            else:
                name_parts = ["erasplit", source_name]
            name = ".".join(name_parts)

            # Make paths start from the root directory
            include_dirs = [
                join(parent_dir, include_dir)
                for include_dir in extension.get("include_dirs", [])
            ]
            if extension.get("include_np", False):
                include_dirs.append(np_include)

            depends = [
                join(parent_dir, depend) for depend in extension.get("depends", [])
            ]

            extra_compile_args = (
                extension.get("extra_compile_args", []) + default_extra_compile_args
            )
            optimization_level = extension.get(
                "optimization_level", default_optimization_level
            )
            if os.name == "posix":
                extra_compile_args.append(f"-{optimization_level}")
            else:
                extra_compile_args.append(f"/{optimization_level}")

            libraries_ext = extension.get("libraries", []) + default_libraries

            new_ext = Extension(
                name=name,
                sources=sources,
                language=extension.get("language", None),
                include_dirs=include_dirs,
                libraries=libraries_ext,
                depends=depends,
                extra_link_args=extension.get("extra_link_args", None),
                extra_compile_args=extra_compile_args,
            )
            cython_exts.append(new_ext)

    return cythonize_extensions(cython_exts)


def setup_package():
    python_requires = ">=3.8"
    required_python_version = (3, 8)

    metadata = dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        project_urls=PROJECT_URLS,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: C",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Development Status :: 5 - Production/Stable",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: Implementation :: CPython",
            "Programming Language :: Python :: Implementation :: PyPy",
        ],
        cmdclass=cmdclass,
        python_requires=python_requires,
        install_requires=min_deps.tag_to_packages["install"] + ["numpy==1.26.4"] +[
            "numpy>=1.17.3,<2.0.0",
            "scipy>=1.3.2",
            "Cython>=0.29.33,<=3.0.12",
        ],
        package_data={"": ["*.csv", "*.gz", "*.txt", "*.pxd", "*.rst", "*.jpg"]},
        zip_safe=False,  # the package can run out of an .egg file
        extras_require={
            key: min_deps.tag_to_packages[key]
            for key in ["examples", "docs", "tests", "benchmark"]
        },
    )

    commands = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if not all(
        command in ("egg_info", "dist_info", "clean", "check") for command in commands
    ):
        if sys.version_info < required_python_version:
            required_version = "%d.%d" % required_python_version
            raise RuntimeError(
                "Scikit-learn requires Python %s or later. The current"
                " Python version is %s installed in %s."
                % (required_version, platform.python_version(), sys.executable)
            )

        check_package_status("numpy", min_deps.NUMPY_MIN_VERSION)
        check_package_status("scipy", min_deps.SCIPY_MIN_VERSION)

        _check_cython_version()
        metadata["ext_modules"] = configure_extension_modules()
        metadata["libraries"] = libraries
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
