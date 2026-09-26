"""Write the official candidate file from the blocking output, and check it.

Reads the wide test candidate table (blocking v3, Job B: test_candidates.parquet) and writes
output/candidate_pairs.tsv in the challenge format: one row per test S1 (every S1, in test_source1 order,
empty list when blocking found nothing), comma-joined S2/S3 ids ordered by pruner score. matching_results.tsv
is written all-empty unless a matching file is given later by the model stage.

Checks (written to candidates_check.json, any failure raises):
  - the test candidate table has the same columns and types as the train table (features run on both);
  - ids: S1 keys are test S1, candidates are S2/S3, no duplicate pairs, list sizes within the cap;
  - the organiser validator (utils/validate_submission.py) passes, with the optional id-existence check.

Usage: python src/write_candidates.py --blocking-dir DIR --data-dir DIR --out-dir DIR [--validator PATH]
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import polars as pl

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:  # Kaggle notebook: modules are written to /tmp/src
    sys.path.insert(0, "/tmp/src")
from submission import write_submission  # noqa: E402

CAP = 10  # blocking v3 operating point (config.json: target 7 -> tau 0.01, cap 10)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def find(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocking-dir", required=True, help="folder with test_candidates.parquet (and train_candidates.parquet)")
    ap.add_argument("--data-dir", required=True, help="folder with test_source{1,2,3}.parquet (raw challenge data)")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--validator", default=None, help="path to utils/validate_submission.py (runs it if given)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    check: dict = {}

    test_path = find(Path(a.blocking_dir), "test_candidates.parquet")
    # 1. same columns as the train table, so the features code runs unchanged on test
    try:
        train_schema = pl.read_parquet_schema(find(Path(a.blocking_dir), "train_candidates.parquet"))
        test_schema = pl.read_parquet_schema(test_path)
        label_cols = {"is_match"}  # train only
        check["schema"] = {
            "only_in_train": sorted(set(train_schema) - set(test_schema) - label_cols),
            "only_in_test": sorted(set(test_schema) - set(train_schema)),
            "type_mismatch": sorted(c for c in set(train_schema) & set(test_schema) if train_schema[c] != test_schema[c]),
            "test_columns": list(test_schema),
        }
        log(f"schema: {check['schema']['only_in_train']=} {check['schema']['only_in_test']=} {check['schema']['type_mismatch']=}")
    except FileNotFoundError:
        check["schema"] = "train_candidates.parquet not found; schema not compared"

    # 2. load, order each list by pruner score, check ids and sizes
    s1_ids = pl.read_parquet(find(Path(a.data_dir), "test_source1.parquet"), columns=["entity_id"])["entity_id"]
    cand = pl.read_parquet(test_path, columns=["s1", "cand", "p_u50"])
    lists = (cand.sort(["s1", "p_u50", "cand"], descending=[False, True, False])
             .group_by("s1", maintain_order=True).agg(pl.col("cand")))
    sizes = lists.with_columns(pl.col("cand").list.len().alias("n"))["n"]
    check["pairs"] = cand.height
    check["duplicate_pairs"] = cand.height - cand.select("s1", "cand").unique().height
    check["s1_test"] = s1_ids.len()
    check["s1_with_candidates"] = lists.height
    check["s1_empty"] = s1_ids.len() - s1_ids.is_in(lists["s1"].implode()).sum()
    check["s1_not_in_test"] = int((~lists["s1"].is_in(s1_ids.implode())).sum())
    check["bad_candidate_prefix"] = int((~cand["cand"].str.slice(0, 3).is_in(["S2-", "S3-"])).sum())
    check["avg_per_s1_all"] = cand.height / max(s1_ids.len(), 1)
    check["max_list"] = int(sizes.max() or 0)
    log(f"{cand.height:,} pairs, {lists.height:,} S1 with candidates, {check['s1_empty']:,} empty, max list {check['max_list']}")

    # 3. write the official files (matches all empty until the model stage)
    cands = dict(zip(lists["s1"].to_list(), lists["cand"].to_list()))
    del cand, lists
    write_submission(out, s1_ids.to_list(), {}, cands)
    log(f"wrote {out / 'candidate_pairs.tsv'} and an all-empty {out / 'matching_results.tsv'}")

    # 4. organiser validator, with id-only copies of the test files (it reads the first column only)
    if a.validator:
        import importlib.util
        spec = importlib.util.spec_from_file_location("validate_submission", a.validator)
        vs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vs)
        with tempfile.TemporaryDirectory() as td:
            for k in (1, 2, 3):
                ids = pl.read_parquet(find(Path(a.data_dir), f"test_source{k}.parquet"), columns=["entity_id"])
                ids.write_csv(Path(td) / f"test_source{k}.tsv", separator="\t")
            errors, warnings = vs.validate(str(out / "matching_results.tsv"), str(out / "candidate_pairs.tsv"), td, check_ids=True)
        check["validator_errors"], check["validator_warnings"] = errors, warnings
        log(f"validator: {len(errors)} error(s), {len(warnings)} warning(s)")
        for e in errors:
            log(f"  ERROR {e}")

    (out / "candidates_check.json").write_text(json.dumps(check, indent=1, default=str))
    problems = [k for k in ("duplicate_pairs", "s1_not_in_test", "bad_candidate_prefix") if check[k]]
    if check["max_list"] > CAP:
        problems.append(f"max_list {check['max_list']} > cap {CAP}")
    if isinstance(check["schema"], dict) and any(check["schema"][k] for k in ("only_in_train", "only_in_test", "type_mismatch")):
        problems.append("schema differs from train")
    if check.get("validator_errors"):
        problems.append("validator errors")
    if problems:
        raise SystemExit(f"CHECK FAILED: {problems} (details in candidates_check.json)")
    log("all checks passed")


if __name__ == "__main__":
    main()
