import nox

nox.options.sessions = ["lint", "tests"]


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests(session):
    """Run the test suite."""
    session.install("-e", ".[all]")
    session.run("pytest", *session.posargs)


@nox.session
def lint(session):
    """Run linters."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session
def docs(session):
    """Build the documentation."""
    session.install("-e", ".[all]")
    session.install("sphinx", "pydata-sphinx-theme", "sphinx-gallery", "numpydoc", "myst-parser")
    session.chdir("docs")
    session.run("make", "html", *session.posargs)
