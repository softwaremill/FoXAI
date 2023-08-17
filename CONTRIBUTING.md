
# Development Installation

To get the development installation follow the `Development` section from
`README.md`.

# Development Process

## Code Style

Captum uses `black` and `pylint` to enforce a common code style across the code
base and is enforced by `pre-commit` hooks and CI pipeline.

## Type Hints

FoXAI is fully typed. We expect any contributions to also use proper type
annotations and we enforce these in our `pre-commit` checks.

## Unit Tests

To run the unit tests, you should use `pre-commit`:

```bash
pre-commit run pytest
```

## Documentation

FoXAI's documentation is build using Sphinx and relies on docstrings. Therefore
we make sure that all functions, classes and methods have coresponding
documentation. We use
[Google Python style guide](https://google.github.io/styleguide/pyguide.html).

# Pull Requests

We actively welcome pull requests. Ordinary pull requests are accepted only to the
`develop` branch, `main` branch is designed only for release PRs.

1. Fork the repo and create your branch from `develop`.
2. Make changes to the code.
3. Add unit tests.
4. Update the documentation.
5. Make sure that tests passes.
6. Make sure your code passes `pre-commit` checks.
7. If you have changed API or class names make sure examples from notebooks
are still working.

# Version Testing

To bump version of a dependency or test if changes are working in different
environment you should use GitHub Actions workflow called
`Manual Installation test`.

First, create PR to the `develop` branch, add
selection option with dependency you would like to test by modifying
`.github/workflows/installation_test_manual_run.yaml`,
`.github/workflows/installation_test_called_workflow.yaml`, and
`.github/actions/build-module/action.yaml` files, commit, and push
to the repository.

Next, go to the repository page > `Actions` >
`Manual Installation test` > `Run workflow` > select your branch > select
versions > `Run workflow`.

Remember to attach links to the workflow runs to the PR comment or description.

## Add New Dependency Test

In `installation_test_manual_run.yaml` you should add selecting
dependencies version in `on` > `workflow_dispatch` > `inputs` and provide
them as environment variables in `job` > `foxai` > `with`.

In the `installation_test_called_workflow.yaml` file you should add
environment variables to `on` > `workflow_calls` > `inputs` and to `jobs` >
`foxai` > `steps` > `env` which will be used in `action.yaml` script.

In `action.yaml` file you should add lines with
installation of given dependency version explicite after `poetry install`.

# Releasing

Branch `main` is dedicated to releases only and is intended to be stable. Release
is done by PR from `develop` branch into the `main` branch. After merge is
completed tag should be added to given commit.

Release PR will be additionally tested with GitHub CI jobs against matrix of
versions including Python, CUDA and core libraries versions.

# Issues

We encourage you to report bugs using GitHub issues. Please ensure your
description is clear and has sufficient instructions to be able to reproduce
the issue.

# License

By contributing to FoXAI, you agree that your contributions will be licensed under
the LICENSE file in the root directory of this source tree.
