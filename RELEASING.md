# Versioning and controlled releases

The first release is **0.1.0**. Rust crates, the Python distribution, native
`__version__`, and Git tags share one version. Stable tags are exactly
`vMAJOR.MINOR.PATCH`, without leading zeroes. Prerelease tags are not supported
by the release workflow yet; adding them requires a PR covering Cargo/PEP 440
version mapping and prerelease publication behavior.

## Compatibility policy

| Change | Before 1.0 | From 1.0 |
| --- | --- | --- |
| Compatible bug/security fix | PATCH: 0.1.0 → 0.1.1 | PATCH |
| Compatible new functionality | MINOR: 0.1.0 → 0.2.0 | MINOR |
| Removed/renamed API, changed defaults or incompatible output | MINOR, with migration notes | MAJOR |
| Documentation or internal work with no behavior change | No release required; include in the next release | Same |

The public contract includes the Python API, Rust exports, CLI flags, JSON
fields, exit codes, and documented numerical behavior. Deprecate before removal
where feasible. Record breaking changes even during 0.x development. Fixes may
change results when correcting a defect; explain scientific/numerical effects.
The keyword API is included in the initial 0.1.0 release, before any published
baseline. Existing Options/ExecutionOptions calls remain supported.

## Prepare a release PR

1. Open a release issue with the intended version, scope, compatibility impact,
   and validation checklist. Develop on `codex/release-X.Y.Z` from `origin/main`.
2. Update `Cargo.toml`, `crates/bsp-metal/Cargo.toml`,
   `crates/bsp-cuda/Cargo.toml`, and `pyproject.toml` together. Refresh
   `Cargo.lock` with Cargo and inspect the three first-party package entries;
   avoid unrelated dependency upgrades. Python and Sphinx derive their version
   from the installed native package. Update a tracked Python lockfile if one
   is present; do not introduce one incidentally.
3. Move the intended changes from `[Unreleased]` to `[X.Y.Z]` in `CHANGELOG.md`.
   Include fixes, features, deprecations, breaking changes, migration advice,
   and meaningful known limitations. Set the release date when publishing.
4. Run `python3 tools/check_release.py`, the release-policy tests, and the
   checks in `CONTRIBUTING.md`. Open a PR linking the release issue. All required
   GitHub checks must pass on its final head; resolve every material finding.
5. Obtain explicit merge/tag authorization. Squash merge using the checked PR
   head (`gh pr merge N --squash --match-head-commit SHA`), without admin bypass.

## Tag the merged commit

Use a clean release worktree. Substitute the approved version and merged SHA;
never tag an unmerged branch or rely on a stale local `main`.

```sh
git fetch origin main --tags
git switch --detach MERGED_SHA
python3 tools/check_release.py
gh pr checks N --required
```

Inspect the post-merge Actions checks for that SHA as well; wait for every
required workflow to succeed before creating the tag. The PR checks alone do
not validate a changed merge result.

```sh
git tag -a vX.Y.Z MERGED_SHA -m "Release vX.Y.Z"
python3 tools/check_release.py --tag vX.Y.Z --require-main
git push origin refs/tags/vX.Y.Z
gh run list --workflow release.yml
```

The tag workflow independently rejects malformed, lightweight, off-main,
out-of-order, or version-mismatched tags. It runs Rust/CLI, installed Python
wheels, optional CUDA/Metal build checks, documentation, and doctests before
creating a draft. Hosted CUDA checks compile and link; they do not certify
execution on an NVIDIA device. Keep that limitation in release notes.

## Review and publish

Inspect the completed workflow, draft notes, archives, and `SHA256SUMS` using
`gh run view RUN_ID` and `gh release view vX.Y.Z`. Download with
`gh release download vX.Y.Z --dir /tmp/bsp-release-X.Y.Z` and verify the sums.
Smoke-test the CLI and wheel on supported hardware before publishing. Record
which GPU device tests actually ran; do not imply unmeasured device coverage.

Only after separate explicit publication approval:

```sh
gh release edit vX.Y.Z --draft=false
```

This workflow does not upload to PyPI or crates.io. A failed build or packaging
run leaves the tag in place: fix through a new PR and issue a higher version.
Never move/delete a version tag, overwrite published assets, or silently
replace an existing release. If an incomplete draft needs recovery, inspect
its state and obtain explicit approval before any asset mutation.

## GitHub protections

`main` requires a PR, an up-to-date branch, resolved conversations, and successful
Rust (three OSes), Python (3.12/3.13 on three OSes), CUDA/Metal builds,
documentation, and repository-policy checks. The rules apply to administrators;
force pushes and deletion are disabled. Zero reviewer approvals are required
to support a solo maintainer. This is CI enforcement, not independent review.
Version-tag rules prohibit tag updates and deletion, with no bypass actor.
Changes to protection settings require an explicit, documented maintenance
decision; future sessions must not weaken them to complete a task.
