"""Validate synchronized stable versions, release notes, and annotated tags."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import tomllib

STABLE_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
MANIFESTS = {
    "bsp-rust": "Cargo.toml",
    "bsp-metal": "crates/bsp-metal/Cargo.toml",
    "bsp-cuda": "crates/bsp-cuda/Cargo.toml",
}


def version_tuple(value):
    if not STABLE_VERSION.fullmatch(value):
        raise ValueError(f"expected stable MAJOR.MINOR.PATCH version, got {value!r}")
    return tuple(map(int, value.split(".")))


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def check_release(root, tag=None, require_main=False):
    versions = {}
    for name, path in MANIFESTS.items():
        package = tomllib.loads((root / path).read_text())["package"]
        if package["name"] != name:
            raise ValueError(f"{path}: expected package {name}")
        versions[path] = package["version"]
    versions["pyproject.toml"] = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    packages = tomllib.loads((root / "Cargo.lock").read_text())["package"]
    for name in MANIFESTS:
        entries = [p for p in packages if p["name"] == name and "source" not in p]
        if len(entries) != 1:
            raise ValueError(f"Cargo.lock: expected exactly one local {name} package")
        versions[f"Cargo.lock:{name}"] = entries[0]["version"]
    version = versions["Cargo.toml"]
    current = version_tuple(version)
    if current < (0, 1, 0):
        raise ValueError("releases start at 0.1.0")
    if any(value != version for value in versions.values()):
        raise ValueError(f"package versions disagree: {versions}")
    changelog = (root / "CHANGELOG.md").read_text()
    if "## [Unreleased]" not in changelog:
        raise ValueError("CHANGELOG.md needs an [Unreleased] section")
    sections = re.findall(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## |\Z)",
                          changelog, re.MULTILINE | re.DOTALL)
    if len(sections) != 1 or not sections[0].strip():
        raise ValueError(f"CHANGELOG.md needs one nonempty [{version}] section")
    if tag is not None:
        if tag != f"v{version}":
            raise ValueError(f"tag must match package version v{version}, got {tag!r}")
        if git(root, "cat-file", "-t", f"refs/tags/{tag}") != "tag":
            raise ValueError("release tags must be annotated")
        commit = git(root, "rev-parse", f"refs/tags/{tag}^{{commit}}")
        if commit != git(root, "rev-parse", "HEAD"):
            raise ValueError("check out the tagged commit before release validation")
        if require_main:
            result = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor",
                                     commit, "refs/remotes/origin/main"], check=False)
            if result.returncode != 0:
                raise ValueError("release commit must belong to origin/main")
        for previous in git(root, "tag", "--merged", commit, "--list", "v*").splitlines():
            if previous != tag and STABLE_VERSION.fullmatch(previous[1:]):
                if version_tuple(previous[1:]) >= current:
                    raise ValueError(f"release must be newer than existing tag {previous}")
    elif require_main:
        raise ValueError("--require-main requires --tag")
    return version, sections[0].strip()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tag", help="annotated stable tag to validate at HEAD")
    parser.add_argument("--require-main", action="store_true")
    parser.add_argument("--notes", action="store_true", help="print only this version's changelog notes")
    args = parser.parse_args()
    try:
        version, notes = check_release(Path(__file__).resolve().parents[1], args.tag, args.require_main)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"release validation failed: {error}\n")
    print(notes if args.notes else f"Release metadata valid: {version}")


if __name__ == "__main__":
    main()
