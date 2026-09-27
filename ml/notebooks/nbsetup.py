"""Notebook bootstrap: find the repo, extend ``sys.path``, ensure the dataset.

The three notebooks in ``ml/notebooks`` import the production ``app`` package
and read ``ml/data/generated``.  Both live *outside* the notebook directory, so
resolving them with ``os.path.abspath('../..')`` silently depends on the
kernel's current working directory — which is the notebook directory in
Jupyter but ``/content`` in Google Colab, and an arbitrary directory when a
remote kernel is attached.  That mismatch is what produced
``ModuleNotFoundError: No module named 'app'``.

``bootstrap()`` is working-directory independent: it locates the repository by
walking up from a set of anchors, puts ``services/model-server`` on
``sys.path``, and generates the synthetic corpus on first use.  ``ml/data/generated``
is git-ignored, so a fresh clone (or a Colab session) has no data until this
runs.

Importable as a module (for tests and scripts) and safe to copy into a notebook
cell; the notebooks only rely on :func:`bootstrap`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

__all__ = ["REPO_ENV_VAR", "locate_repo", "ensure_dataset", "bootstrap",
           "DEFAULT_ITEMS"]

#: Environment variable that pins the repository root, bypassing discovery.
REPO_ENV_VAR = "CATALOGUE_AI_ROOT"

#: Items generated when the corpus is missing.  Smaller than the CLI default so
#: a Colab bootstrap stays fast; every notebook reads a prefix of it anyway.
DEFAULT_ITEMS = 1024

#: Files that mark a directory as the repository root.
_ROOT_MARKERS = (
    Path("ml") / "data" / "make_synthetic_dataset.py",
    Path("services") / "model-server" / "app" / "embed.py",
)

#: Where a Colab/Runtime session typically clones or mounts the repository.
_FALLBACK_ROOTS = (
    Path("/content"),
    Path("/content/catalogue-ai"),
    Path("/workspace"),
    Path.home(),
    Path.cwd(),
)

#: Directories worth scanning one or two levels deep for a checkout.
_SCAN_ROOTS = (
    Path("/content"),
    Path.home(),
    Path.cwd(),
)

#: How deep to look below each scan root before giving up.  Four levels covers
#: Colab's ``/content/<name>`` as well as ``/content/drive/MyDrive/<name>``.
_SCAN_DEPTH = 4

#: A directory is a candidate repo if it holds this file.
_HINT = Path("ml") / "notebooks" / "nbsetup.py"


def _is_root(candidate: Path) -> bool:
    return all((candidate / marker).is_file() for marker in _ROOT_MARKERS)


def _anchors(start: Optional[Path] = None) -> List[Path]:
    """Ordered, de-duplicated list of directories to inspect."""
    candidates: List[Path] = []

    env = os.environ.get(REPO_ENV_VAR)
    if env:
        candidates.append(Path(env).expanduser())

    if start is not None:
        anchor = _resolve(start)
        if anchor:
            candidates.extend([anchor, *anchor.parents])

    cwd = _resolve(Path.cwd())
    if cwd:
        candidates.extend([cwd, *cwd.parents])
    candidates.extend(_FALLBACK_ROOTS)

    ordered: List[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = _resolve(candidate)
        if resolved and resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def _resolve(path: Path) -> Optional[Path]:
    try:
        return path.expanduser().resolve()
    except OSError:  # pragma: no cover - unreadable mount
        return None


def _scan(roots: Iterable[Path], depth: int = _SCAN_DEPTH) -> List[Path]:
    """Breadth-first scan of ``roots`` for a checkout.

    A Colab session may clone into any folder under ``/content`` — and a Drive
    mount can sit at an arbitrary depth — so a fixed list of candidate paths is
    not enough.
    """
    found: List[Path] = []
    seen = set()
    frontier = [_resolve(r) for r in roots]
    frontier = [f for f in frontier if f is not None]

    for _ in range(max(1, depth)):
        next_frontier: List[Path] = []
        for directory in frontier:
            if directory in seen:
                continue
            seen.add(directory)
            if _is_root(directory):
                found.append(directory)
                continue
            try:
                children = sorted(p for p in directory.iterdir() if p.is_dir())
            except (OSError, PermissionError):
                continue
            # skip caches and hidden dirs; they never hold a checkout
            next_frontier.extend(
                c for c in children
                if not c.name.startswith(".") and c.name != "__pycache__")
        frontier = next_frontier
        if not frontier:
            break
    return found


def locate_repo(start: Optional[Path] = None) -> Optional[Path]:
    """Return the repository root, or ``None`` when it cannot be found.

    Resolution order: ``$CATALOGUE_AI_ROOT``, an explicit ``start`` and its
    parents, the kernel's working directory and its parents, the known
    Colab/Runtime locations, and finally a bounded scan below those roots.
    Never depends on a relative path.
    """
    for candidate in _anchors(start):
        if _is_root(candidate):
            return candidate

    for candidate in _scan(_SCAN_ROOTS):
        return candidate
    return None


def describe_search(start: Optional[Path] = None) -> str:
    """Human-readable summary of where the repository was looked for.

    Included in the failure message so a Colab user can see exactly which
    folder to clone into instead of guessing.
    """
    lines = ["searched for ml/notebooks/nbsetup.py in:"]
    for candidate in _anchors(start):
        lines.append(f"  {candidate}{'  <- found' if _is_root(candidate) else ''}")
    lines.append("  and a depth-%d scan under: %s" % (
        _SCAN_DEPTH, ", ".join(str(r) for r in _SCAN_ROOTS)))
    return "\n".join(lines)


def ensure_dataset(repo_root: Path, items: int = DEFAULT_ITEMS,
                   force: bool = False) -> Path:
    """Generate the synthetic corpus if absent; return the data directory.

    ``ml/data/generated`` is git-ignored, so this is the only way a fresh clone
    or a Colab runtime can obtain the data the notebooks read.
    """
    data_dir = repo_root / "ml" / "data" / "generated"
    manifest = data_dir / "manifest.csv"
    if manifest.is_file() and not force:
        return data_dir

    generator = repo_root / "ml" / "data" / "make_synthetic_dataset.py"
    if not generator.is_file():
        raise FileNotFoundError(f"corpus generator missing: {generator}")

    data_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(generator), "--out", str(data_dir), "--n", str(items)]
    # Keep notebook output readable: the generator prints one JSON summary line.
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0 or not manifest.is_file():
        raise RuntimeError(
            f"dataset generation failed (exit {result.returncode})\n"
            f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()}")
    return data_dir


def bootstrap(repo_env_var: str = REPO_ENV_VAR, items: int = DEFAULT_ITEMS,
              data_dir: Optional[Path] = None) -> dict:
    """Make the notebook importable and the dataset available.

    Returns a small summary dict and prints a one-line status, so a notebook can
    show exactly which repo and dataset it picked up.

    Raises:
        RuntimeError: if the repository or the corpus cannot be resolved.  The
            message lists every path that was inspected.
    """
    repo_root = locate_repo()
    if repo_root is None:
        raise RuntimeError(_not_found_message(repo_env_var))

    model_server = repo_root / "services" / "model-server"
    for path in (str(model_server), str(model_server / "app" / "from_scratch")):
        if path not in sys.path:
            sys.path.insert(0, path)

    if data_dir is None:
        data_dir = ensure_dataset(repo_root, items=items)

    summary = {
        "repo_root": str(repo_root),
        "data_dir": str(data_dir),
        "model_server_on_path": str(model_server) in sys.path,
    }
    print(f"[nbsetup] repo={repo_root}\n[nbsetup] data={data_dir}")
    return summary


def _not_found_message(repo_env_var: str) -> str:
    """Actionable failure text for a missing checkout."""
    return (
        "catalogue-ai repository not found.\n\n"
        "The notebook needs the repository (it imports `app` and reads\n"
        "`ml/data/generated`), not just this .ipynb file. In a fresh Colab\n"
        "runtime run this once, then re-run the bootstrap cell:\n\n"
        "    !git clone <your-fork-or-url> /content/catalogue-ai\n\n"
        "Already cloned somewhere else? Point the notebook at it:\n\n"
        f"    import os; os.environ['{repo_env_var}'] = '/path/to/catalogue-ai'\n\n"
        f"{describe_search()}\n")


def missing_dependencies(modules: Iterable[str] = ()) -> List[str]:
    """Return the subset of ``modules`` that cannot be imported."""
    import importlib.util

    absent = []
    for name in modules:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):  # pragma: no cover - malformed name
            found = False
        if not found:
            absent.append(name)
    return absent
