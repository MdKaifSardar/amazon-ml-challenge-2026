"""Candidate generation (blocking): union of several searches, per country.

Methods (all per country; the country is a compute split, 0 true pairs cross countries):
- name       forward TF-IDF top-k on core_name
- name_city  forward TF-IDF top-k on core_name + city
- name_addr  forward TF-IDF top-k on core_name + normalised address
- reverse    for every S2/S3 record, its top-m S1 on core_name + address (all S1 of the split compete); the
             record's best score is kept so a margin rule (keep rank 2 only if close to rank 1) can be applied
- rare       S2/S3 records sharing a rare core_name token (document frequency <= RARE_DF in the country)

TF-IDF: char_wb 3-4-grams, fitted per country on train + test records (no labels; a seeded sample
of up to FIT_SAMPLE texts). N-grams in more than MAX_DF of the texts are dropped: they carry almost no
weight and dominate the cost of the sparse product.

Search scope (a compute split, not a filter):
- "state": a forward query searches the pool records of its own state plus all pool records without a
  state; an S1 without a state searches the whole country. The reverse search runs within states; a pool
  record without a state searches all S1 of the country (fallback). Before bucketing, a missing state is
  inferred from the city, then the dept (state_maps / fill_states, learned from records that have both).
- "country": every search covers the whole country (affordable on a GPU); recovers pairs whose states
  disagree.
The rare-token index ignores states in both scopes.

Top-k search runs on the GPU when torch sees one (dense score blocks + torch.topk), otherwise on CPU with
sparse_dot_topn. check_gpu() compares the two on a slice before a run (tie-aware: scores per rank and
exact recomputation) and the faster device does the searches.

Near-identical decoys are NOT filtered here; every method keeps its full top-k.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

SEED = 42
FIT_SAMPLE = 3_000_000
MAX_DF = 0.02
RARE_DF = 20
NGRAMS = (3, 4)
FORWARD = ("name", "name_city", "name_addr")
REVERSE_REP = "name_addr"
NONE = "__none__"
ALL = "__all__"
SCOPES = ("state", "country")
INFER_MIN_N, INFER_MIN_SHARE = 20, 0.9  # state inference from city / dept (bucket placement only)
N_THREADS = 4


def texts(df: pl.DataFrame) -> pl.DataFrame:
    """Text representations used by the searches (all from norm-v3 columns)."""
    core = pl.col("core_name")
    return df.select(
        core.alias("name"),
        pl.concat_str([core, pl.col("city").fill_null("")], separator=" ").str.strip_chars().alias("name_city"),
        pl.concat_str([core, pl.col("addr_norm").str.replace_all(",", "")], separator=" ").str.strip_chars().alias("name_addr"),
    )


def fit_vectorizer(corpus: pl.Series) -> TfidfVectorizer:
    if corpus.len() > FIT_SAMPLE:
        corpus = corpus.sample(FIT_SAMPLE, seed=SEED)
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAMS, min_df=2, max_df=MAX_DF,
                        sublinear_tf=True, dtype=np.float32)
    v.fit(corpus.to_list())
    return v


try:  # GPU path (Kaggle GPU); falls back to sparse_dot_topn on CPU
    import torch
    USE_GPU = torch.cuda.is_available()
except ImportError:
    torch, USE_GPU = None, False
GPU_BLOCK_BYTES = 2 * 1024**3  # cap for each dense block per step (scores: chunk x pool; queries: chunk x vocab)


def gpu_info() -> str:
    if not USE_GPU:
        return "no GPU (CPU sparse_dot_topn)"
    return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB"


def _to_torch_csr(M: sparse.csr_matrix, device: str):
    return torch.sparse_csr_tensor(torch.from_numpy(M.indptr.astype(np.int32)), torch.from_numpy(M.indices.astype(np.int32)),
                                   torch.from_numpy(M.data.astype(np.float32)), size=M.shape, device=device)


def _topn_gpu(Q: sparse.csr_matrix, P: sparse.csr_matrix, k: int, keep_col: np.ndarray | None = None, device: str = "cuda"):
    """Top-k of Q @ P.T per row on the GPU: P stays on the card as CSR; each query chunk is densified,
    multiplied (cuSPARSE SpMM) and torch.topk keeps the best k pool rows per query."""
    Pg = _to_torch_csr(P, device)
    chunk = int(max(1, min(4096, GPU_BLOCK_BYTES // (4 * P.shape[0]), GPU_BLOCK_BYTES // (4 * Q.shape[1]))))
    rows, cols, vals, ranks = [], [], [], []
    for s in range(0, Q.shape[0], chunk):
        Qc = _to_torch_csr(Q[s:s + chunk], device).to_dense()      # chunk x V
        scores = torch.sparse.mm(Pg, Qc.T).T.contiguous()           # chunk x pool
        v, i = torch.topk(scores, k, dim=1)                         # chunk x k, sorted descending
        v, i = v.cpu().numpy().reshape(-1), i.cpu().numpy().reshape(-1).astype(np.int64)
        n = scores.shape[0]
        r, rk = np.repeat(np.arange(s, s + n), k), np.tile(np.arange(1, k + 1), n)
        keep = v > 0  # as on CPU: only pairs sharing an n-gram
        if keep_col is not None:
            keep &= keep_col[i]
        rows.append(r[keep]); cols.append(i[keep]); vals.append(v[keep]); ranks.append(rk[keep])
        del Qc, scores
    del Pg
    return tuple(np.concatenate(x) for x in (rows, cols, vals, ranks))


def _topn_cpu(Q: sparse.csr_matrix, P: sparse.csr_matrix, k: int, keep_col: np.ndarray | None = None):
    R = sp_matmul_topn(Q, P.T.tocsr(), top_n=k, threshold=0.0, sort=True, n_threads=N_THREADS).tocsr()
    counts = np.diff(R.indptr)
    rows = np.repeat(np.arange(R.shape[0]), counts)
    ranks = np.arange(R.nnz) - np.repeat(R.indptr[:-1], counts) + 1
    cols = R.indices.astype(np.int64)
    keep = np.ones(R.nnz, bool) if keep_col is None else keep_col[cols]
    return rows[keep], cols[keep], R.data.astype(np.float32)[keep], ranks[keep]


def _topn(Q: sparse.csr_matrix, P: sparse.csr_matrix, k: int, keep_col: np.ndarray | None = None):
    """Top-k columns of Q @ P.T per row: (row, col, score, rank 1..k). keep_col: optional mask over P rows;
    pairs whose pool row is not kept are dropped after ranking (ranks are unchanged)."""
    if Q.shape[0] == 0 or P.shape[0] == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, np.empty(0, dtype=np.float32), e
    k = min(k, P.shape[0])
    return (_topn_gpu if USE_GPU else _topn_cpu)(Q, P, k, keep_col)


def check_gpu(Q: sparse.csr_matrix, P: sparse.csr_matrix, k: int, n_q: int = 4000, n_p: int = 1_000_000) -> dict:
    """GPU vs CPU top-k on a slice, tie-aware. Many pool records share a name, so equal scores at the k-th
    place may be kept differently; instead of comparing (row, col) pairs, compare the score at every rank
    (rank_score_diff) and recompute each GPU pair's cosine exactly on CPU (pair_score_diff). Timing runs after
    a warm-up call, so CUDA start-up is not counted."""
    Q, P = Q[:n_q], P[:n_p]
    k = min(k, P.shape[0])
    _topn_gpu(Q[:64], P, k); torch.cuda.synchronize()  # warm-up (CUDA context, cuSPARSE handles)
    t0 = time.time(); g = _topn_gpu(Q, P, k); torch.cuda.synchronize(); t1 = time.time()
    c = _topn_cpu(Q, P, k); t2 = time.time()
    by_rank = lambda x: pl.DataFrame({"r": x[0], "rk": x[3], "v": x[2]})
    j = by_rank(c).join(by_rank(g), on=["r", "rk"], how="full", suffix="_g").with_columns(pl.col("v", "v_g").fill_null(0.0))
    exact = np.asarray(Q[g[0]].multiply(P[g[1]]).sum(axis=1)).ravel()
    gp, cp = set(zip(g[0].tolist(), g[1].tolist())), set(zip(c[0].tolist(), c[1].tolist()))
    pairs = Q.shape[0] * P.shape[0]
    return {"queries": Q.shape[0], "pool": P.shape[0], "k": k,
            "rank_score_diff": float((j["v"] - j["v_g"]).abs().max() or 0.0),
            "pair_score_diff": float(np.abs(exact - g[2]).max()) if len(exact) else 0.0,
            "pair_agreement": len(gp & cp) / max(len(cp), 1),
            "gpu_ns_per_pair": (t1 - t0) / pairs * 1e9, "cpu_ns_per_pair": (t2 - t1) / pairs * 1e9}


def use_gpu(flag: bool) -> None:
    """Choose the device for all later searches (GPU only if CUDA is available)."""
    global USE_GPU
    USE_GPU = bool(flag) and torch is not None and torch.cuda.is_available()


def state_maps(records: pl.DataFrame, min_n: int = INFER_MIN_N, min_share: float = INFER_MIN_SHARE) -> dict[str, pl.DataFrame]:
    """Learn key -> state from records that have both (no labels): for city and for dept, the most frequent state
    when it covers >= min_share of min_n+ records. Used only to place records without a detected state into a
    search bucket (a compute split), never as a filter or a feature."""
    out = {}
    for key in ("city", "dept"):
        if key not in records.columns:
            continue
        c = (records.filter(pl.col(key).is_not_null() & pl.col("state").is_not_null())
             .group_by(key, "state").len()
             .with_columns(pl.col("len").sum().over(key).alias("tot"))
             .sort(key, "len", "state", descending=[False, True, False])
             .group_by(key, maintain_order=True).first())
        out[key] = c.filter((pl.col("tot") >= min_n) & (pl.col("len") / pl.col("tot") >= min_share)).select(
            key, pl.col("state").alias(f"state_from_{key}"))
    return out


def fill_states(df: pl.DataFrame, maps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """state_src = detected / city / dept / none; state = detected, else inferred from city, else from dept."""
    x = df
    for key, m in maps.items():
        x = x.join(m, on=key, how="left")
    inferred = [pl.col(f"state_from_{k}") for k in maps]
    src = pl.when(pl.col("state").is_not_null()).then(pl.lit("detected"))
    for k in maps:
        src = src.when(pl.col(f"state_from_{k}").is_not_null()).then(pl.lit(k))
    return x.with_columns(src.otherwise(pl.lit("none")).alias("state_src"),
                          pl.coalesce(pl.col("state"), *inferred).alias("state")).drop([f"state_from_{k}" for k in maps])


def buckets(s1: pl.DataFrame, pool: pl.DataFrame, scope: str) -> tuple[np.ndarray, np.ndarray]:
    if scope == "country":
        return np.full(s1.height, ALL, dtype=object), np.full(pool.height, ALL, dtype=object)
    return s1["state"].fill_null(NONE).to_numpy(), pool["state"].fill_null(NONE).to_numpy()


def forward(Q, q_bucket: np.ndarray, P, p_bucket: np.ndarray, k: int) -> pl.DataFrame:
    """For each query row: top-k pool rows among its state bucket + pool rows without a state."""
    out = []
    p_none = np.flatnonzero(p_bucket == NONE)
    for b in np.unique(q_bucket):
        qi = np.flatnonzero(q_bucket == b)
        pi = np.arange(P.shape[0]) if b == NONE else np.concatenate([np.flatnonzero(p_bucket == b), p_none])
        r, c, s, rk = _topn(Q[qi], P[pi], k)
        out.append(pl.DataFrame({"qi": qi[r], "pi": pi[c], "score": s, "rank": rk}))
    return pl.concat(out) if out else pl.DataFrame(schema={"qi": pl.Int64, "pi": pl.Int64, "score": pl.Float32, "rank": pl.Int64})


def reverse(S, s_bucket: np.ndarray, P, p_bucket: np.ndarray, m: int, keep_s: np.ndarray | None = None,
            none_fallback: bool = True) -> pl.DataFrame:
    """For each pool row: its top-m S1 rows in the same state (pool rows without a state: over all S1 of the
    country if none_fallback, else skipped). Returns (qi = S1 row, pi, score, rank,
    best), where best is the pool row's top-1 score over ALL S1 (for margin rules). keep_s: optional mask over S1
    rows; only pairs with a kept S1 are returned (all S1 still compete)."""
    out = []
    for b in np.unique(p_bucket):
        if b == NONE and not none_fallback:
            continue
        pi = np.flatnonzero(p_bucket == b)
        si = np.arange(S.shape[0]) if b == NONE else np.flatnonzero(s_bucket == b)  # no state: whole country
        r, c, s, rk = _topn(P[pi], S[si], m)
        best = np.zeros(len(pi), dtype=np.float32)
        best[r[rk == 1]] = s[rk == 1]
        f = slice(None) if keep_s is None else keep_s[si[c]]
        out.append(pl.DataFrame({"qi": si[c][f], "pi": pi[r][f], "score": s[f], "rank": rk[f], "best": best[r][f]}))
    return pl.concat(out) if out else pl.DataFrame(schema={"qi": pl.Int64, "pi": pl.Int64, "score": pl.Float32,
                                                           "rank": pl.Int64, "best": pl.Float32})


def rare_tokens(q_names: pl.Series, p_names: pl.Series, df_counts: pl.DataFrame, max_df: int = RARE_DF) -> pl.DataFrame:
    """Pool rows sharing a rare core_name token with the query (token df <= max_df in the country, so at most
    ~max_df x tokens rows per query). score = number of shared rare tokens; rank by score, ties by pool row
    (deterministic, so pieces merge exactly)."""
    rare = df_counts.filter((pl.col("df") <= max_df) & (pl.col("t").str.len_chars() >= 3)).select("t")
    tok = lambda s, name: (pl.DataFrame({name: np.arange(s.len()), "t": s.str.split(" ")})
                           .explode("t", empty_as_null=True).filter(pl.col("t").is_not_null()).unique()
                           .join(rare, on="t"))
    qt, pt = tok(q_names, "qi"), tok(p_names, "pi")
    hits = qt.join(pt, on="t").group_by("qi", "pi").agg(pl.len().cast(pl.Float32).alias("score"))
    return (hits.sort("qi", "score", "pi", descending=[False, True, False])
            .with_columns(pl.int_range(1, pl.len() + 1, dtype=pl.Int64).over("qi").alias("rank"))
            .select("qi", "pi", "score", "rank"))


def token_df(names: pl.Series) -> pl.DataFrame:
    """Document frequency of core_name tokens (distinct per record)."""
    return (pl.DataFrame({"i": np.arange(names.len()), "t": names.str.split(" ")})
            .explode("t", empty_as_null=True).filter(pl.col("t").is_not_null() & (pl.col("t") != "")).unique()
            .group_by("t").agg(pl.len().alias("df")))


def block_country(s1: pl.DataFrame, pool: pl.DataFrame, query_mask: np.ndarray, fit_corpus: dict[str, pl.Series],
                  df_counts: pl.DataFrame, k: int, m: int, scopes: tuple[str, ...] = ("state",),
                  skip: Callable[[str, str, float], bool] = lambda scope, method, pairs: False,
                  log: Callable[[str], None] = print, gpu_check: bool = False,
                  timing: list[dict] | None = None, pool_mask: np.ndarray | None = None,
                  reverse_keep: np.ndarray | None = None) -> tuple[pl.DataFrame, list[dict]]:
    """All methods for one country, for each search scope (same TF-IDF matrices). s1: ALL S1 of the split
    (reverse competition); query_mask: S1 rows to generate candidates for. skip(scope, method, pairs) may
    drop a search (e.g. over a time budget). Returns a long table (scope, qi = S1 row, pi = pool row,
    method, score, rank) and one timing row per (scope, method), appended to `timing` if given (so skip()
    can read the rates measured so far). pool_mask: pool rows that run the reverse search (a shard of the pool;
    default all). Forward and rare searches run for the query_mask rows only, so a shard = (S1 slice, pool slice)
    and the union of all shards equals one full run. reverse_keep: S1 rows whose reverse hits are returned
    (default query_mask; with pool shards pass ALL wanted S1, since any S1 can be hit from any pool shard)."""
    ts, tp = texts(s1), texts(pool)
    qrows = np.flatnonzero(query_mask)
    bk = {sc: buckets(s1, pool, sc) for sc in scopes}
    parts, timing = [], ([] if timing is None else timing)
    for rep in FORWARD:
        t0 = time.time()
        v = fit_vectorizer(fit_corpus[rep])
        S, P = v.transform(ts[rep].to_list()), v.transform(tp[rep].to_list())
        t1 = time.time()
        log(f"  {rep}: fit+transform {t1 - t0:.0f}s (vocab {len(v.vocabulary_):,}, pool nnz/row {P.nnz / max(P.shape[0], 1):.1f})")
        timing.append({"scope": "all", "method": f"{rep}_fit_transform", "search_s": t1 - t0, "queries": len(qrows),
                       "pool": P.shape[0], "pairs_evaluated": None})
        if gpu_check and USE_GPU:
            chk = check_gpu(S[qrows], P, k)
            log(f"  GPU check: {chk}")
            if chk["rank_score_diff"] > 1e-4 or chk["pair_score_diff"] > 1e-4:
                raise RuntimeError(f"GPU top-k disagrees with CPU: {chk}")
            use_gpu(chk["gpu_ns_per_pair"] < chk["cpu_ns_per_pair"])  # the faster device does all searches
            log(f"  search device from now on: {'GPU' if USE_GPU else 'CPU (faster than GPU on this data)'}")
            timing.append({"scope": "all", "method": "device_check", **chk, "device": "GPU" if USE_GPU else "CPU"})
            gpu_check = False
        for sc in scopes:
            s_bucket, p_bucket = bk[sc]
            pairs = _pairs_evaluated(s_bucket[qrows], p_bucket)
            if skip(sc, rep, pairs):
                log(f"  [{sc}] {rep}: skipped ({pairs:.2e} pairs over budget)")
                timing.append({"scope": sc, "method": rep, "pairs_evaluated": pairs, "skipped": True})
            else:
                t2 = time.time()
                f = forward(S[qrows], s_bucket[qrows], P, p_bucket, k)
                f = f.with_columns(pl.Series("qi", qrows[f["qi"].to_numpy()]))  # query-local row -> S1 row
                parts.append(f.with_columns(pl.lit(sc).alias("scope"), pl.lit(rep).alias("method")))
                timing.append({"scope": sc, "method": rep, "search_s": time.time() - t2, "queries": len(qrows),
                               "pool": P.shape[0], "pairs_evaluated": pairs})
                log(f"  [{sc}] {rep}: forward {time.time() - t2:.0f}s, {pairs:.2e} pairs, {f.height:,} rows")
            if rep == REVERSE_REP:
                prow = np.arange(P.shape[0]) if pool_mask is None else np.flatnonzero(pool_mask)
                pairs = _pairs_evaluated(p_bucket[prow], s_bucket, with_none=False)  # NONE rows: all S1
                if skip(sc, "reverse", pairs):
                    log(f"  [{sc}] reverse: skipped ({pairs:.2e} pairs over budget)")
                    timing.append({"scope": sc, "method": "reverse", "pairs_evaluated": pairs, "skipped": True})
                    continue
                t3 = time.time()
                r = reverse(S, s_bucket, P[prow], p_bucket[prow], m, keep_s=query_mask if reverse_keep is None else reverse_keep)
                r = r.with_columns(pl.Series("pi", prow[r["pi"].to_numpy()], dtype=pl.Int64))  # shard row -> pool row
                parts.append(r.with_columns(pl.lit(sc).alias("scope"), pl.lit("reverse").alias("method")))  # has "best"
                timing.append({"scope": sc, "method": "reverse", "search_s": time.time() - t3,
                               "queries": len(prow), "pool": S.shape[0], "pairs_evaluated": pairs})
                log(f"  [{sc}] reverse: {time.time() - t3:.0f}s, {pairs:.2e} pairs, {r.height:,} rows")
        del S, P
    t5 = time.time()
    rr = rare_tokens(s1["core_name"][qrows], pool["core_name"], df_counts)
    rr = rr.with_columns(pl.Series("qi", qrows[rr["qi"].to_numpy()], dtype=pl.Int64), pl.lit("rare").alias("method"))
    parts += [rr.with_columns(pl.lit(sc).alias("scope")) for sc in scopes]  # scope-independent
    timing.append({"scope": "all", "method": "rare", "search_s": time.time() - t5, "queries": len(qrows),
                   "pool": pool.height, "pairs_evaluated": None})
    log(f"  rare: {time.time() - t5:.0f}s")
    cols = ["scope", "qi", "pi", "method", "score", "rank", "best"]
    parts = [p if "best" in p.columns else p.with_columns(pl.lit(None, dtype=pl.Float32).alias("best")) for p in parts]
    return pl.concat([p.select(cols) for p in parts]), timing


def _pairs_evaluated(q_bucket: np.ndarray, p_bucket: np.ndarray, with_none: bool = True) -> float:
    """Number of (query, pool) similarity evaluations implied by the bucket sizes (cost model)."""
    pb = pl.Series(p_bucket).value_counts()
    sizes = dict(zip(pb[pb.columns[0]].to_list(), pb["count"].to_list()))
    n_none = sizes.get(NONE, 0) if with_none else 0
    total = 0.0
    for b, n in zip(*np.unique(q_bucket, return_counts=True)):
        total += n * (len(p_bucket) if b == NONE else sizes.get(b, 0) + n_none)
    return total
