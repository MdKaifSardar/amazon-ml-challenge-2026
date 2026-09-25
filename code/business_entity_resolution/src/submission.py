"""Write matching_results.tsv / candidate_pairs.tsv in the exact challenge format.

One row per S1 (in the given order), tab-separated, comma-joined IDs, no quoting,
no duplicates, empty second column when there is nothing to list.
"""
from collections.abc import Iterable, Mapping
from pathlib import Path

MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")


def write_id_lists(path: Path, header: tuple[str, str], s1_ids: Iterable[str],
                   lists: Mapping[str, Iterable[str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1 in s1_ids:
            ids = list(dict.fromkeys(i for i in lists.get(s1, ()) if not i.startswith("S1-")))
            f.write(f"{s1}\t{','.join(ids)}\n")
            n += 1
    return n


def write_submission(out_dir: Path, s1_ids: list[str], matches: Mapping[str, Iterable[str]],
                     candidates: Mapping[str, Iterable[str]]) -> None:
    for s1, m in matches.items():
        extra = set(m) - set(candidates.get(s1, ()))
        if extra:
            raise ValueError(f"{s1}: matches not in candidates: {sorted(extra)[:5]}")
    write_id_lists(out_dir / "matching_results.tsv", MATCHING_HEADER, s1_ids, matches)
    write_id_lists(out_dir / "candidate_pairs.tsv", CANDIDATE_HEADER, s1_ids, candidates)
