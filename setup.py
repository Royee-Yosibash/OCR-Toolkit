import json
from pathlib import Path
from setuptools import setup, find_packages

FULL_MARKER = "# full"

with open('release_data.json') as f:
    release_data = json.load(f)


def _read_requirements(path: str = "requirements.txt") -> tuple[list[str], dict[str, list[str]]]:
    """Read install requirements from a pip requirements file.

    Lines marked with ``# full`` are grouped under the ``full`` extra.
    Unmarked lines become base ``install_requires`` entries.

    Args:
        path: Path to the requirements file.

    Returns:
        A tuple of (base_requires, extras_require) where base_requires
        is a list of base dependency strings and extras_require is a dict
        mapping extra names to their dependency lists.
    """
    base = []
    extras = {"full": []}

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if FULL_MARKER in line:
            req = line.split(FULL_MARKER)[0].strip()
            extras["full"].append(req)
        else:
            base.append(stripped)

    return base, extras


base_requires, extras_require = _read_requirements()

setup(
    name="ocr-enhancement-toolset",
    version=release_data['version'],
    description="OCR enhancement package that improves performances" \
        " and allows evaluations for different use cases",
    python_requires=">=3.12",
    packages=find_packages(),
    include_package_data=True,
    install_requires=base_requires,
    extras_require=extras_require,
)
