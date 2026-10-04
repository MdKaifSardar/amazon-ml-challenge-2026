import os
import sys

def main():
    test_source1 = "dataset/test/test_source1.tsv"
    pairwise_file = "output/candidate_pairs.tsv"
    grouped_file = "output/candidate_pairs_grouped.tsv"

    print("Reading test_source1 IDs...")
    with open(test_source1, "r", encoding="utf-8") as f:
        header = f.readline()
        all_s1 = [line.split("\t")[0].strip() for line in f if line.strip()]

    print(f"Total required S1: {len(all_s1)}")

    print("Grouping candidates...")
    candidates_by_s1 = {s1: [] for s1 in all_s1}
    
    with open(pairwise_file, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                s1, cand = parts
                if s1 in candidates_by_s1:
                    candidates_by_s1[s1].append(cand)

    print("Writing grouped candidate_pairs...")
    with open(grouped_file, "w", encoding="utf-8") as out:
        out.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in all_s1:
            cands = candidates_by_s1[s1]
            out.write(f"{s1}\t{','.join(cands)}\n")

    print(f"Done! Grouped candidates written to {grouped_file}")

if __name__ == "__main__":
    main()
