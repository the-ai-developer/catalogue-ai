"""Execute the generated notebooks as a smoke test.

    python ml/notebooks/run_notebooks.py            # all three
    python ml/notebooks/run_notebooks.py 01         # just 01_*
    python ml/notebooks/run_notebooks.py --in-place

The default runs each notebook with the working directory set to a **temporary
directory that has no relationship to the repository** — the same condition as
a Google Colab kernel, whose CWD is ``/content``.  That is what turns a
relative path like ``../../services/model-server`` into a silent
``ModuleNotFoundError: No module named 'app'``, and it is why this runner
exists rather than trusting ``jupyter nbconvert``.

Dependencies that are optional are reported as ``skipped`` instead of failing,
so the check is usable on a machine without the full ML stack.  Use
``--require-deps`` to make them hard failures in CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NOTEBOOK_GLOBS = {
    "01": "01_dual_encoder_from_scratch.ipynb",
    "02": "02_seq2seq_attention_from_scratch.ipynb",
    "03": "03_rag_evaluation.ipynb",
}

#: Modules a notebook may need; absence downgrades the run to ``skipped``.
OPTIONAL_DEPS = ("torch", "numpy", "PIL", "matplotlib")


def importable(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def code_cells(notebook: dict) -> List[str]:
    return ["".join(c["source"]) for c in notebook["cells"]
            if c["cell_type"] == "code"]


def run_notebook(path: Path, *, in_place: bool = False,
                 require_deps: bool = False) -> Tuple[str, str]:
    """Execute every code cell in order; return ``(status, detail)``."""
    missing = [m for m in OPTIONAL_DEPS if not importable(m)]
    if missing:
        if require_deps:
            return "failed", f"missing required modules: {', '.join(missing)}"
        return "skipped", f"missing optional modules: {', '.join(missing)}"

    notebook = json.loads(path.read_text())
    cells = code_cells(notebook)
    namespace = {"__name__": "__main__"}

    os.environ.setdefault("MPLBACKEND", "Agg")
    # When detached, nothing links the CWD to the repository, so discovery can
    # only fall back to the documented escape hatch.  This is also the Colab
    # path: the repo is cloned somewhere known and CATALOGUE_AI_ROOT is set.
    previous_root = os.environ.get("CATALOGUE_AI_ROOT")
    if not in_place:
        os.environ["CATALOGUE_AI_ROOT"] = str(HERE.parent.parent)

    previous = os.getcwd()
    workdir = str(path.parent) if in_place else tempfile.mkdtemp(prefix="nb-smoke-")
    started = time.time()
    try:
        os.chdir(workdir)
        for index, source in enumerate(cells, start=1):
            try:
                exec(compile(source, f"{path.name}:cell{index}", "exec"), namespace)
            except Exception:
                return "failed", (f"cell {index} raised\n"
                                  + "".join(traceback.format_exc()[-1600:]))
    except BaseException:
        return "failed", traceback.format_exc()[-1600:]
    finally:
        os.chdir(previous)
        if previous_root is None:
            os.environ.pop("CATALOGUE_AI_ROOT", None)
        else:
            os.environ["CATALOGUE_AI_ROOT"] = previous_root

    return "passed", f"{len(cells)} cells in {time.time() - started:.1f}s"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*", default=[],
                        help="notebook prefixes, e.g. 01 02 03 (default: all)")
    parser.add_argument("--in-place", action="store_true",
                        help="run from the notebook directory instead of a "
                             "detached temp dir (skips the Colab check)")
    parser.add_argument("--require-deps", action="store_true",
                        help="treat missing optional modules as failures")
    args = parser.parse_args()

    selected = ([NOTEBOOK_GLOBS[t] for t in args.targets if t in NOTEBOOK_GLOBS]
                if args.targets else list(NOTEBOOK_GLOBS.values()))

    failures = 0
    for name in selected:
        path = HERE / name
        status, detail = run_notebook(path, in_place=args.in_place,
                                      require_deps=args.require_deps)
        print(f"[{status:>7}] {name}: {detail}")
        if status == "failed":
            failures += 1

    if not selected:
        print("no notebooks matched", file=sys.stderr)
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
