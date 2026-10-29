# Repository workflow for Codex and contributors

## Start every session

- Read this file, `CONTRIBUTING.md`, and `RELEASING.md` before making changes.
- Inspect `git status --short --branch`, remotes, recent commits, and relevant
  GitHub issues/PRs with `gh`. Preserve unrelated local work; use an isolated
  worktree when necessary. Never commit someone else's edits incidentally.
- Never develop on `main`. Fetch `origin` and create a `codex/<short-purpose>`
  branch from current `origin/main`, or continue the matching existing branch.
- Use `gh` for GitHub issues, PRs, check inspection, merges, and releases. Do not
  claim a remote action succeeded without checking its result.

## Issues, commits, and pull requests

- Find or create an issue for a feature, bug, or release before implementation.
  Describe the user-visible problem, scope, and testable acceptance criteria.
  For bugs, include a minimal reproduction, expected/actual behavior, version,
  platform/backend, and a regression test when practical. Link related issues.
- Make focused commits with Conventional Commit subjects: `feat(python): ...`,
  `fix(cli): ...`, `docs: ...`, `test: ...`, or `chore(release): ...`. Explain the
  reason and material compatibility implications in the body. Use `!` and a
  `BREAKING CHANGE:` footer for incompatible public changes.
- Keep one cohesive purpose per PR. Include the problem, resulting behavior,
  compatibility/version impact, actual validation, and limitations. Link the
  issue with `Closes #N`. Use `.github/pull_request_template.md`.
- Push the topic branch and open a PR with `gh pr create`. Inspect the diff and
  all check failures. Fix defects on the same branch and rerun relevant checks.
- Merge only with explicit user authorization and all required checks passing
  for the current head. Use `gh pr merge --squash --match-head-commit <SHA>`.
  Never bypass protections, force-push `main`, or use `--admin` to evade checks.
  A solo maintainer may merge their own PR; do not invent independent review.
- Bug fixes go through PRs too, including urgent fixes. Record follow-up work in
  issues rather than hiding it in unrelated commits.

## Public API and validation

- Prefer `solve(formula=formula, seed=3)` and
  `Solver(formula=formula, seed=3, backend="metal")` in Python examples.
  Retain supported configuration-object calls until a documented breaking
  release; reject ambiguous configuration and misspelled keywords.
- Keep Python, CLI, Rust, tests, and documentation consistent. Run the checks in
  `CONTRIBUTING.md` relevant to the change and report what actually ran.
- Preserve the independent SP recurrence, fix/release invariants, and CPU
  verification of every SAT assignment. Heuristic failure never proves UNSAT.
- Ground performance claims in recorded hardware, precision, input, seeds,
  stopping policy, and timings. The native Python API uses the same Rust core;
  CLI availability alone does not establish a speed advantage.

## Versions and releases

- Follow `RELEASING.md`. Start at `0.1.0`; never retag or replace a release asset.
- Treat Python names/keywords/defaults, CLI flags/JSON/exit codes, and exported
  Rust APIs as public contracts. For `0.x`, compatible fixes use PATCH; added
  functionality or breaking changes use MINOR, with breaking changes explicit.
  From `1.0.0`, follow SemVer: incompatible changes require MAJOR.
- Update all first-party Cargo manifests, their lockfile entries, Python package
  metadata, and `CHANGELOG.md` in a release PR. Do not infer a release bump from
  a commit prefix alone. Python `__version__` comes from the native crate.
- Create an annotated `vMAJOR.MINOR.PATCH` tag only on the tested, merged release
  commit after explicit authorization. Push that exact tag, never `--tags`.
- The release workflow validates versions and ancestry, builds/tests all
  platforms and docs, and creates a checksummed **draft** GitHub release.
  Verify its assets and notes with `gh`. Publishing the draft or uploading to a
  package registry requires separate explicit user authorization.
