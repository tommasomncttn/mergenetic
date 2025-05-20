# Contributing to Mergenetic

First off, thank you for considering contributing to Mergenetic! We welcome any help, from reporting bugs and suggesting features to writing code and improving documentation.

## Table of Contents

-   [How Can I Contribute?](#how-can-i-contribute)
    -   [Reporting Bugs](#reporting-bugs)
    -   [Suggesting Enhancements](#suggesting-enhancements)
    -   [Your First Code Contribution](#your-first-code-contribution)
    -   [Pull Requests](#pull-requests)
-   [Development Setup](#development-setup)
    -   [Prerequisites](#prerequisites)
    -   [Installation](#installation)
-   [Running Tests](#running-tests)
-   [Coding Conventions](#coding-conventions)
-   [Code of Conduct](#code-of-conduct)
-   [Questions?](#questions)

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/tommasomncttn/mergenetic/issues).

If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/tommasomncttn/mergenetic/issues/new). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample or an executable test case** demonstrating the expected behavior that is not occurring.

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one, please open an issue on GitHub. Clearly describe the proposed enhancement, why it would be beneficial, and if possible, provide examples or mockups.

This allows for discussion and feedback before significant development work begins.

### Your First Code Contribution

Unsure where to begin contributing to Mergenetic? You can start by looking through these `good first issue` and `help wanted` issues:

-   [Good first issues](https://github.com/tommasomncttn/mergenetic/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) - issues which should only require a few lines of code, and a test or two.
-   [Help wanted issues](https://github.com/tommasomncttn/mergenetic/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) - issues which should be a bit more involved than `good first issue` issues.

### Pull Requests

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally: `git clone git@github.com:YOUR_USERNAME/mergenetic.git`
3.  **Create a new branch** for your changes: `git checkout -b name-of-your-feature-or-fix`
4.  **Make your changes**.
5.  **Add tests** for your changes.
6.  **Ensure all tests pass** (see [Running Tests](#running-tests)).
7.  **Format your code** (e.g., using Black, isort - see [Coding Conventions](#coding-conventions)).
8.  **Commit your changes**: `git commit -m "feat: A brief description of the feature"` (See [Conventional Commits](https://www.conventionalcommits.org/) for commit message guidelines, if you choose to adopt them).
9.  **Push to your fork**: `git push origin name-of-your-feature-or-fix`
10. **Open a Pull Request** to the `main` branch of the `tommasomncttn/mergenetic` repository.
11. Provide a clear description of your changes in the Pull Request. Link to any relevant issues.

## Development Setup

### Prerequisites

-   Python (version >=3.10, <3.12, as specified in `pyproject.toml`)
-   Pip (Python package installer)
-   Git
-   (Optional but recommended) A virtual environment manager like `venv` or `conda`.

### Installation

1.  Clone your fork of the repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/mergenetic.git
    cd mergenetic
    ```
2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
3.  Install the project in editable mode with development dependencies:
    ```bash
    pip install -e ".[dev]"
    ```

## Running Tests

To run the test suite:
```bash
pytest
```
Please ensure all tests pass before submitting a pull request. If you add new features, please add corresponding tests.

## Coding Conventions

-   **Style**: We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code.
-   **Formatting**: We use [Black](https://github.com/psf/black) for code formatting and [isort](https://pycqa.github.io/isort/) for import sorting. Please format your code before committing.
    ```bash
    black .
    isort .
    ```
-   **Linting**: We use [Flake8](https://flake8.pycqa.org/en/latest/) for linting. Please ensure your code passes linting checks.
    ```bash
    flake8 .
    ```
-   **Type Hinting**: We encourage the use of type hints for new code.

## Code of Conduct

This project and everyone participating in it is governed by the [Mergenetic Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior.

## Questions?

If you have any questions, feel free to open an issue or reach out to the maintainers.

---

Thank you for contributing!