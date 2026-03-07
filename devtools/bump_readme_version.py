#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


def _replace_install_line(content: str, version: str) -> str:
    major, minor, *_ = version.split(".")
    next_minor = f"{major}.{int(minor) + 1}.0"
    new_line = f'pip install "token-throttle[redis,tiktoken]>={version},<{next_minor}"'
    new_content, n = re.subn(
        r'pip install "[^">]*>=\d+\.\d+\.\d+,<\d+\.\d+\.\d+"',
        new_line,
        content,
    )
    if n == 0:
        raise RuntimeError(
            "No matching pip install line found in README.md for pip install"
        )
    return new_content


def _replace_version_badge(content: str, version: str) -> str:
    # Replace the shields.io badge version in the URL and label
    # Example: https://img.shields.io/badge/0.0.1-version?color=active&style=flat&label=version
    # Should become: https://img.shields.io/badge/0.16.12-version?...
    badge_pattern = re.compile(
        r"(https://img\.shields\.io/badge/v)(\d+\.\d+\.\d+)(-version\?)"
    )
    new_content, n = badge_pattern.subn(
        rf"\g<1>{version}\g<3>",
        content,
    )
    if n == 0:
        raise RuntimeError("No matching version badge found in README.md")
    return new_content


def update_readme(readme_path: Path, version: str) -> None:
    content = readme_path.read_text(encoding="utf-8")
    content = _replace_install_line(content, version)
    content = _replace_version_badge(content, version)
    readme_path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Update pip install line and version badge in README.md"
    )
    parser.add_argument(
        "version", type=str, help="The version string to use (e.g., 0.16.0)"
    )
    args = parser.parse_args()
    readme_path = Path("README.md")
    update_readme(readme_path, args.version)


if __name__ == "__main__":
    main()
