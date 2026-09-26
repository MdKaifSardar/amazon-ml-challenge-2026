"""Wrap a src/ script into a Kaggle job: jobs/<job>/<job>.ipynb + kernel-metadata.json.

Usage:
  python jobs/build_notebook.py --job eda --script code/business_entity_resolution/src/eda.py \
      --args "--data-dir /kaggle/input --out-dir /kaggle/working" --no-gpu \
      [--modules code/business_entity_resolution/src/normalise.py ...] [--kernel-sources owner/job ...]
The script source is copied verbatim into one cell, so src/ stays the single source of truth. Modules it
imports are written to /tmp/src (not /kaggle/working, so they don't end up in the outputs).
Packages are pip-installed at the versions pinned in the submission requirements.txt.
"""
import argparse
import itertools
import json
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQS = ROOT / "code/business_entity_resolution/requirements.txt"
_ids = itertools.count()
KAGGLE_SKIP = {"numpy", "pyarrow"}  # preinstalled on Kaggle; reinstalling can break other packages


def cell(kind: str, src: str) -> dict:
    c = {"cell_type": kind, "id": f"c{next(_ids)}", "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--args", default="")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--kernel-sources", nargs="*", default=[])
    ap.add_argument("--owner", default="sayanchatterjee264", help="Kaggle user that owns the job")
    ap.add_argument("--datasets", nargs="*", default=["sayanchatterjee264/amazon-ml-2026"], help="dataset_sources")
    ap.add_argument("--modules", nargs="*", default=[], help="src/ modules the script imports")
    a = ap.parse_args()

    pins = [
        line.strip()
        for line in REQS.read_text().splitlines()
        if line.strip() and not line.startswith("#") and line.split("==")[0].lower() not in KAGGLE_SKIP
    ]
    script = Path(a.script).resolve()
    argv = [script.name, *shlex.split(a.args)]
    nb = {
        "cells": [
            cell("markdown", f"# {a.job}\nGenerated from `{script.relative_to(ROOT)}` by `jobs/build_notebook.py`. Edit the script, not this notebook."),
            cell("code", "!pip install -q " + " ".join(pins)),
            cell("code", "import os\nos.makedirs('/tmp/src', exist_ok=True)\n"
                         "for root, dirs, files in os.walk('/kaggle/input'):\n    print(root, dirs, files)"),
            *[cell("code", f"%%writefile /tmp/src/{Path(m).name}\n" + Path(m).read_text()) for m in a.modules],
            cell("code", f"import sys\nsys.path.insert(0, '/tmp/src')\nsys.argv = {argv!r}"),
            cell("code", script.read_text()),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "jobs" / a.job
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.job}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": f"{a.owner}/{a.job}",
        "title": a.job,
        "code_file": f"{a.job}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": not a.no_gpu,
        "enable_internet": True,
        "dataset_sources": a.datasets,
        "competition_sources": [],
        "kernel_sources": a.kernel_sources,
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}/{a.job}.ipynb and kernel-metadata.json")


if __name__ == "__main__":
    main()
