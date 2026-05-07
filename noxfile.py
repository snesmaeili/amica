#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox", "uv"]
# ///

import nox

nox.needs_version = ">=2025.10.14"
nox.options.default_venv_backend = "uv"


@nox.session(python=["3.10", "3.11", "3.12", "3.13"], default=True)
def tests(session):
    """Run the test suite."""
    session.install("-e", ".[all]")
    session.run("pytest", *session.posargs)


@nox.session(default=True)
def lint(session):
    """Run linters."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(default=False)
def docs(session):
    """Build the documentation."""
    session.install("-e", ".[all]")
    session.install("sphinx", "pydata-sphinx-theme", "sphinx-gallery", "numpydoc", "myst-parser")
    session.chdir("docs")
    session.run("make", "html", *session.posargs)


if __name__ == "__main__":
    nox.main()
