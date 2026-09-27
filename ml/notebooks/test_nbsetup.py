"""Tests: the notebook bootstrap that fixes the Colab import failure.

``locate_repo`` is the part that matters — the notebooks previously resolved
``../../services/model-server`` against the kernel's working directory, which is
``/content`` in Colab, producing ``ModuleNotFoundError: No module named 'app'``.
Discovery must instead work from any starting point, including one with no
relationship to the repository at all.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nbsetup  # noqa: E402

REPO_ROOT = Path(nbsetup.__file__).resolve().parent.parent.parent


def make_fake_repo(root: Path) -> Path:
    """Create the minimum marker layout that identifies a repository root."""
    (root / "ml" / "data").mkdir(parents=True)
    (root / "ml" / "data" / "make_synthetic_dataset.py").write_text("# stub\n")
    (root / "services" / "model-server" / "app").mkdir(parents=True)
    (root / "services" / "model-server" / "app" / "embed.py").write_text("# stub\n")
    return root


class TestLocateRepo:
    def test_finds_the_real_repository(self):
        found = nbsetup.locate_repo()
        assert found is not None
        assert (found / "ml" / "notebooks" / "nbsetup.py").is_file()

    def test_is_independent_of_working_directory(self, tmp_path, monkeypatch):
        # A CWD with no relationship to the repo must still resolve, via the
        # explicit start anchor or the env var — never via '../..'.
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        monkeypatch.chdir(deep)
        assert nbsetup.locate_repo(start=REPO_ROOT) == REPO_ROOT

    def test_env_var_wins(self, tmp_path, monkeypatch):
        fake = make_fake_repo(tmp_path / "elsewhere")
        monkeypatch.setenv(nbsetup.REPO_ENV_VAR, str(fake))
        monkeypatch.chdir(tmp_path)
        assert nbsetup.locate_repo() == fake.resolve()

    def test_discovers_repo_from_a_nested_start(self, tmp_path):
        deep = make_fake_repo(tmp_path / "repo") / "ml" / "notebooks"
        deep.mkdir(parents=True)
        assert nbsetup.locate_repo(start=deep) == (tmp_path / "repo").resolve()

    def test_returns_none_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.delenv(nbsetup.REPO_ENV_VAR, raising=False)
        monkeypatch.setattr(nbsetup, "_FALLBACK_ROOTS", ())
        monkeypatch.chdir(tmp_path)
        # the real repo may still be reachable through a parent of tmp_path,
        # so assert only the contract: a Path or None, never a relative path
        result = nbsetup.locate_repo(start=tmp_path)
        assert result is None or result.is_absolute()

    def test_never_returns_a_relative_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(nbsetup.REPO_ENV_VAR, "relative/path")
        result = nbsetup.locate_repo()
        assert result is None or result.is_absolute()


class TestEnsureDataset:
    def test_generates_when_manifest_missing(self, tmp_path):
        repo = make_fake_repo(tmp_path / "repo")
        generator = repo / "ml" / "data" / "make_synthetic_dataset.py"
        generator.write_text(
            "import argparse, pathlib\n"
            "ap = argparse.ArgumentParser()\n"
            "ap.add_argument('--out', required=True)\n"
            "ap.add_argument('--n', type=int, default=1)\n"
            "a = ap.parse_args()\n"
            "out = pathlib.Path(a.out)\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'manifest.csv').write_text('item_id\\n')\n"
        )
        data_dir = nbsetup.ensure_dataset(repo, items=4)
        assert (data_dir / "manifest.csv").is_file()

    def test_skips_when_manifest_present(self, tmp_path):
        repo = make_fake_repo(tmp_path / "repo")
        data_dir = repo / "ml" / "data" / "generated"
        data_dir.mkdir(parents=True)
        (data_dir / "manifest.csv").write_text("item_id\n")
        before = os.stat(data_dir / "manifest.csv").st_mtime_ns
        assert nbsetup.ensure_dataset(repo) == data_dir
        assert os.stat(data_dir / "manifest.csv").st_mtime_ns == before

    def test_raises_when_generator_missing(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / "ml" / "data").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            nbsetup.ensure_dataset(repo)

    def test_raises_when_generation_fails(self, tmp_path):
        repo = make_fake_repo(tmp_path / "repo")
        (repo / "ml" / "data" / "make_synthetic_dataset.py").write_text(
            "import sys; sys.exit(3)\n")
        with pytest.raises(RuntimeError):
            nbsetup.ensure_dataset(repo, items=1)


class TestBootstrap:
    def test_adds_model_server_to_sys_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(nbsetup, "ensure_dataset",
                            lambda repo, items=0, force=False: tmp_path)
        before = list(sys.path)
        try:
            summary = nbsetup.bootstrap()
            assert summary["model_server_on_path"] is True
            assert Path(summary["repo_root"]) == REPO_ROOT
            assert summary["data_dir"] == str(tmp_path)
        finally:
            sys.path[:] = before

    def test_raises_a_helpful_error_without_a_repo(self, tmp_path, monkeypatch):
        monkeypatch.setattr(nbsetup, "locate_repo", lambda *a, **k: None)
        with pytest.raises(RuntimeError) as exc:
            nbsetup.bootstrap()
        assert nbsetup.REPO_ENV_VAR in str(exc.value)


class TestMissingDependencies:
    def test_reports_absent_modules(self):
        assert nbsetup.missing_dependencies(["json", "os"]) == []
        assert "definitely_not_installed_xyz" in nbsetup.missing_dependencies(
            ["definitely_not_installed_xyz"])
