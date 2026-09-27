"""Final model end to end: train_model.py (train + validation tune half, report-half check) then predict.py
(test outputs, validator, French checks). One entry point for the Kaggle job and for reproduction.

  python src/run_final.py --input DIR --config best_config.json --work DIR [--validator PATH] [--fallback PATH]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    SRC = Path(__file__).resolve().parent
except NameError:  # Kaggle notebook: modules are written to /tmp/src
    SRC = Path("/tmp/src")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--validator", default=None)
    ap.add_argument("--fallback", default=None)
    a = ap.parse_args()
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    model_dir, out_dir = Path(a.work) / "model_final", Path(a.work) / "output"
    run = lambda args: subprocess.run([sys.executable, *args], check=True, cwd=str(SRC), env=env)
    run([str(SRC / "train_model.py"), "--input", a.input, "--config", a.config, "--out-dir", str(model_dir), "--final"])
    run([str(SRC / "predict.py"), "--input", a.input, "--model-dir", str(model_dir), "--out-dir", str(out_dir)]
        + (["--validator", a.validator] if a.validator else []) + (["--fallback", a.fallback] if a.fallback else []))


if __name__ == "__main__":
    main()
