#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


def _replace_install_lines(content: str, version: str) -> str:
    """Replace version bounds in pip install lines, preserving extras."""
    major, minor, *_ = version.split(".")
    next_minor = f"{major}.{int(minor) + 1}.0"
    new_content, n = re.subn(
        r'(pip install "token-throttle(?:\[[^\]]*\])?)>=\d+\.\d+\.\d+,<\d+\.\d+\.\d+"',
        rf'\1>={version},<{next_minor}"',
        content,
    )
    if n == 0:
        raise RuntimeError(
            "No matching pip install line found in README.md"
        )
    return new_content


def _replace_version_badge(content: str, version: str) -> str:
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
    content = _replace_install_lines(content, version)
    content = _replace_version_badge(content, version)
    readme_path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Update pip install line and version badge in README.md"
    )
    parser.add_argument(
        "version", type=str, help="The version string to use (e.g., 0.6.0)"
    )
    args = parser.parse_args()
    readme_path = Path("README.md")
    update_readme(readme_path, args.version)


if __name__ == "__main__":
    main()
