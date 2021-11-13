# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import os

import nox

BASE = os.path.abspath(os.path.dirname(__file__))

DEFAULT_PYTHON_VERSIONS = ["3.7", "3.8"]

LINT_SETUP_DEPS = ["black", "flake8", "flake8-copyright", "isort"]

SILENT = True


def install_lint_deps(session):
    session.install("--upgrade", "setuptools", "pip", silent=SILENT)
    session.install(*LINT_SETUP_DEPS, silent=SILENT)


def install_pytouch(session):
    session.chdir(BASE)
    session.run("pip", "install", "-e", ".")


@nox.session(python=DEFAULT_PYTHON_VERSIONS)
def tests(session):
    session.install("--upgrade", "setuptools", "pip")
    install_pytouch(session)
    session.install("pytest")
    session.run("pytest", "tests")


@nox.session(python=DEFAULT_PYTHON_VERSIONS)
def lint(session):
    install_lint_deps(session)
    session.run("black", "--check", ".", silent=SILENT)
    session.run(
        "isort",
        "--check",
        "--diff",
        ".",
        "--skip=.nox",
        silent=SILENT,
    )
    session.run("flake8", "--config", ".flake8")
