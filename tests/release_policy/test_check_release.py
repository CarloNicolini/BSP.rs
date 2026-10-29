"""Release guards tested on disposable metadata and Git repositories."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.check_release import MANIFESTS, check_release, version_tuple


class ReleasePolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name, relative in MANIFESTS.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f'[package]\nname = "{name}"\nversion = "0.1.0"\n')
        (self.root / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n')
        (self.root / "Cargo.lock").write_text('version = 4\n' + ''.join(
            f'[[package]]\nname = "{name}"\nversion = "0.1.0"\n' for name in MANIFESTS))
        (self.root / "CHANGELOG.md").write_text(
            '## [Unreleased]\n\n## [0.1.0]\n\nInitial release.\n')

    def git(self, *args):
        return subprocess.check_output([
            "git", "-C", str(self.root), "-c", "user.name=Release Test",
            "-c", "user.email=release@example.invalid", *args], text=True).strip()

    def repository(self):
        self.git("init", "-b", "main")
        self.git("add", ".")
        self.git("commit", "-m", "Initial release")
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")
        self.git("tag", "-a", "v0.1.0", "-m", "Release v0.1.0")

    def test_consistent_initial_version_and_notes(self):
        self.assertEqual(check_release(self.root), ("0.1.0", "Initial release."))

    def test_rejects_each_mismatched_manifest_or_lock_entry(self):
        for relative in [*MANIFESTS.values(), "pyproject.toml", "Cargo.lock"]:
            with self.subTest(path=relative):
                path = self.root / relative
                original = path.read_text()
                path.write_text(original.replace("0.1.0", "0.2.0", 1))
                with self.assertRaisesRegex(ValueError, "versions disagree"):
                    check_release(self.root)
                path.write_text(original)

    def test_rejects_missing_local_lock_package(self):
        path = self.root / "Cargo.lock"
        path.write_text(path.read_text().replace('name = "bsp-metal"', 'name = "other"'))
        with self.assertRaisesRegex(ValueError, "exactly one local bsp-metal"):
            check_release(self.root)

    def test_rejects_missing_empty_and_duplicate_release_notes(self):
        for text in ["## [Unreleased]\n", "## [Unreleased]\n## [0.1.0]\n",
                     "## [Unreleased]\n## [0.1.0]\na\n## [0.1.0]\nb\n"]:
            (self.root / "CHANGELOG.md").write_text(text)
            with self.assertRaisesRegex(ValueError, "nonempty"):
                check_release(self.root)

    def test_stable_semver_format(self):
        self.assertLess(version_tuple("0.9.0"), version_tuple("0.10.0"))
        for invalid in ["01.0.0", "1.02.0", "0.1", "v0.1.0", "0.1.0-rc.1", "0.1.0+build"]:
            with self.subTest(version=invalid), self.assertRaises(ValueError):
                version_tuple(invalid)

    def test_valid_annotated_tag_on_main(self):
        self.repository()
        self.assertEqual(check_release(self.root, "v0.1.0", True)[0], "0.1.0")

    def test_rejects_mismatched_tag(self):
        with self.assertRaisesRegex(ValueError, "tag must match"):
            check_release(self.root, "v0.2.0")

    def test_rejects_lightweight_tag(self):
        self.repository()
        self.git("tag", "-d", "v0.1.0")
        self.git("tag", "v0.1.0")
        with self.assertRaisesRegex(ValueError, "annotated"):
            check_release(self.root, "v0.1.0")

    def test_rejects_off_main_commit(self):
        self.repository()
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")
        self.git("commit", "--allow-empty", "-m", "Unmerged work")
        self.git("tag", "-d", "v0.1.0")
        self.git("tag", "-a", "v0.1.0", "-m", "Unmerged release")
        with self.assertRaisesRegex(ValueError, "origin/main"):
            check_release(self.root, "v0.1.0", True)

    def test_rejects_tag_other_than_head(self):
        self.repository()
        self.git("commit", "--allow-empty", "-m", "Later commit")
        with self.assertRaisesRegex(ValueError, "tagged commit"):
            check_release(self.root, "v0.1.0")

    def test_rejects_out_of_order_version(self):
        self.repository()
        self.git("tag", "-a", "v0.2.0", "-m", "Already released")
        with self.assertRaisesRegex(ValueError, "newer than"):
            check_release(self.root, "v0.1.0")


if __name__ == "__main__":
    unittest.main()
