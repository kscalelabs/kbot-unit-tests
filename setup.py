# mypy: disable-error-code="import-untyped"
#!/usr/bin/env python
"""Setup script for the project."""

import re

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("kbot_unit_tests/requirements.txt", "r", encoding="utf-8") as f:
    requirements: list[str] = f.read().splitlines()


with open("kbot_unit_tests/requirements-dev.txt", "r", encoding="utf-8") as f:
    requirements_dev: list[str] = f.read().splitlines()


with open("kbot_unit_tests/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in kbot_unit_tests/__init__.py"
version: str = version_re.group(1)


setup(
    name="kbot-unit-tests",
    version=version,
    description="The kbot-unit-tests project",
    author="K-Scale Labs",
    url="https://github.com/kscalelabs/kbot-unit-tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={"dev": requirements_dev},
    packages=["kbot_unit_tests", "kbot_cycle_tests"],
    # entry_points={
    #     "console_scripts": [
    #         "kbot_unit_tests.cli:main",
    #     ],
    # },
)
