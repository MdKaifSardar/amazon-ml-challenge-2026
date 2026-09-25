"""Candidate generation (blocking): union of several searches, per country.

Methods (all per country; the country is a compute split, 0 true pairs cross countries):
- name       forward TF-IDF top-k on core_name
- name_city  forward TF-IDF top-k on core_name + city
- name_addr  forward TF-IDF top-k on core_name + normalised address
- reverse    for every S2/S3 record, its top-m S1 on core_name + address (all S1 of the split compete)
- rare       S2/S3 records sharing a rare core_name token (document frequency <= RARE_DF in the country)

TF-IDF: char_wb 3-4-grams, fitted per country on train + test records (no labels; a seeded sample
of up to FIT_SAMPLE texts). N-grams in more than MAX_DF of the texts are dropped: they carry almost no
weight and dominate the cost of the sparse product.

Search scope (a compute split, not a filter):
- "state": a forward query searches the pool records of its own state plus all pool records without a
  state; an S1 without a state searches the whole country. The reverse search runs within states only
  (records without a state are reached by the forward searches).
- "country": every search covers the whole country (affordable on a GPU); recovers pairs whose states
  disagree.
The rare-token index ignores states in both scopes.

Top-k search runs on the GPU when torch sees one (dense score blocks + torch.topk), otherwise on CPU with
sparse_dot_topn. check_gpu() compares the two on a slice before a run.

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


def check_gpu(Q: sparse.csr_matrix, P: sparse.csr_matrix, k: int, n_q: int = 2000, n_p: int = 300_000) -> dict:
    """GPU vs CPU top-k on a slice: share of CPU (row, col) pairs the GPU also returns (ties at the k-th
    score may swap), max score difference, and time per similarity evaluation on each device."""
    Q, P = Q[:n_q], P[:n_p]
    k = min(k, P.shape[0])
    t0 = time.time(); g = _topn_gpu(Q, P, k); torch.cuda.synchronize(); t1 = time.time()
    c = _topn_cpu(Q, P, k); t2 = time.time()
    gp, cp = set(zip(g[0].tolist(), g[1].tolist())), set(zip(c[0].tolist(), c[1].tolist()))
    top1 = lambda x: dict(zip(x[0][x[3] == 1].tolist(), x[2][x[3] == 1].tolist()))
    g1, c1 = top1(g), top1(c)
    pairs = Q.shape[0] * P.shape[0]
    return {"agreement": len(gp & cp) / max(len(cp), 1),
            "max_top1_diff": max((abs(g1.get(r, 0) - v) for r, v in c1.items()), default=0.0),
            "gpu_ns_per_pair": (t1 - t0) / pairs * 1e9, "cpu_ns_per_pair": (t2 - t1) / pairs * 1e9}


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


def reverse(S, s_bucket: np.ndarray, P, p_bucket: np.ndarray, m: int, keep_s: np.ndarray | None = None) -> pl.DataFrame:
    """For each pool row with a state: its top-m S1 rows in the same state. Returns (qi = S1 row, pi, score, rank).
    keep_s: optional mask over S1 rows; only pairs with a kept S1 are returned (all S1 still compete)."""
    out = []
    for b in np.unique(p_bucket):
        if b == NONE:
            continue
        pi = np.flatnonzero(p_bucket == b)
        si = np.flatnonzero(s_bucket == b)
        r, c, s, rk = _topn(P[pi], S[si], m, None if keep_s is None else keep_s[si])
        out.append(pl.DataFrame({"qi": si[c], "pi": pi[r], "score": s, "rank": rk}))
    return pl.concat(out) if out else pl.DataFrame(schema={"qi": pl.Int64, "pi": pl.Int64, "score": pl.Float32, "rank": pl.Int64})


def rare_tokens(q_names: pl.Series, p_names: pl.Series, df_counts: pl.DataFrame, max_df: int = RARE_DF) -> pl.DataFrame:
    """Pool rows sharing a rare core_name token with the query (token df <= max_df in the country).
    score = number of shared rare tokens; rank by score (ties by pool order)."""
    rare = df_counts.filter((pl.col("df") <= max_df) & (pl.col("t").str.len_chars() >= 3)).select("t")
    tok = lambda s, name: (pl.DataFrame({name: np.arange(s.len()), "t": s.str.split(" ")})
                           .explode("t", empty_as_null=True).filter(pl.col("t").is_not_null()).unique()
                           .join(rare, on="t"))
    qt, pt = tok(q_names, "qi"), tok(p_names, "pi")
    hits = qt.join(pt, on="t").group_by("qi", "pi").agg(pl.len().cast(pl.Float32).alias("score"))
    return hits.with_columns(
        pl.col("score").rank("ordinal", descending=True).over("qi").cast(pl.Int64).alias("rank")
    ).select("qi", "pi", "score", "rank")


def token_df(names: pl.Series) -> pl.DataFrame:
    """Document frequency of core_name tokens (distinct per record)."""
    return (pl.DataFrame({"i": np.arange(names.len()), "t": names.str.split(" ")})
            .explode("t", empty_as_null=True).filter(pl.col("t").is_not_null() & (pl.col("t") != "")).unique()
            .group_by("t").agg(pl.len().alias("df")))


def block_country(s1: pl.DataFrame, pool: pl.DataFrame, query_mask: np.ndarray, fit_corpus: dict[str, pl.Series],
                  df_counts: pl.DataFrame, k: int, m: int, scopes: tuple[str, ...] = ("state",),
                  skip: Callable[[str, str, float], bool] = lambda scope, method, pairs: False,
                  log: Callable[[str], None] = print, gpu_check: bool = False,
                  timing: list[dict] | None = None) -> tuple[pl.DataFrame, list[dict]]:
    """All methods for one country, for each search scope (same TF-IDF matrices). s1: ALL S1 of the split
    (reverse competition); query_mask: S1 rows to generate candidates for. skip(scope, method, pairs) may
    drop a search (e.g. over a time budget). Returns a long table (scope, qi = S1 row, pi = pool row,
    method, score, rank) and one timing row per (scope, method), appended to `timing` if given (so skip()
    can read the rates measured so far)."""
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
            if chk["agreement"] < 0.99 or chk["max_top1_diff"] > 1e-3:
                raise RuntimeError(f"GPU top-k disagrees with CPU: {chk}")
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
                pairs = _pairs_evaluated(p_bucket[p_bucket != NONE], s_bucket, with_none=False)
                if skip(sc, "reverse", pairs):
                    log(f"  [{sc}] reverse: skipped ({pairs:.2e} pairs over budget)")
                    timing.append({"scope": sc, "method": "reverse", "pairs_evaluated": pairs, "skipped": True})
                    continue
                t3 = time.time()
                r = reverse(S, s_bucket, P, p_bucket, m, keep_s=query_mask)
                parts.append(r.with_columns(pl.lit(sc).alias("scope"), pl.lit("reverse").alias("method")))
                timing.append({"scope": sc, "method": "reverse", "search_s": time.time() - t3,
                               "queries": int((p_bucket != NONE).sum()), "pool": S.shape[0], "pairs_evaluated": pairs})
                log(f"  [{sc}] reverse: {time.time() - t3:.0f}s, {pairs:.2e} pairs, {r.height:,} rows")
        del S, P
    t5 = time.time()
    rr = rare_tokens(s1["core_name"][qrows], pool["core_name"], df_counts)
    rr = rr.with_columns(pl.Series("qi", qrows[rr["qi"].to_numpy()], dtype=pl.Int64), pl.lit("rare").alias("method"))
    parts += [rr.with_columns(pl.lit(sc).alias("scope")) for sc in scopes]  # scope-independent
    timing.append({"scope": "all", "method": "rare", "search_s": time.time() - t5, "queries": len(qrows),
                   "pool": pool.height, "pairs_evaluated": None})
    log(f"  rare: {time.time() - t5:.0f}s")
    cols = ["scope", "qi", "pi", "method", "score", "rank"]
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
