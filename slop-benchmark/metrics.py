"""Confusion matrices, P/R/F1, the C2-vs-C3 gap, the C4 tax, and panel agreement (Fleiss' kappa).

Positive class = SLOP. An arm 'rejects' (predicts positive) when it refuses admission.
"""
from __future__ import annotations


def confusion(decisions: list[bool], gt_slop: list[bool]) -> dict:
    """decisions[i] True = arm REJECTED item i (predicted slop). gt_slop[i] True = item is slop."""
    tp = sum(1 for d, g in zip(decisions, gt_slop) if d and g)       # slop, rejected (good)
    fp = sum(1 for d, g in zip(decisions, gt_slop) if d and not g)   # legit, rejected (bad: the tax)
    fn = sum(1 for d, g in zip(decisions, gt_slop) if not d and g)   # slop, admitted (bad: leakage)
    tn = sum(1 for d, g in zip(decisions, gt_slop) if not d and not g)
    n = len(decisions)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, n=n, precision=round(prec, 3),
                recall=round(rec, 3), f1=round(f1, 3), accuracy=round((tp + tn) / n, 3) if n else 0.0)


def reject_rate(decisions: list[bool], klasses: list[str], target: str) -> float:
    idx = [i for i, k in enumerate(klasses) if k == target]
    return round(sum(decisions[i] for i in idx) / len(idx), 3) if idx else 0.0


def best_threshold(scores: list[float], gt_slop: list[bool]) -> tuple[float, dict]:
    """For a continuous detector score (here: low perplexity => predict slop), choose the threshold
    that maximizes F1 — a best-case bound for the detector. Returns (threshold, confusion-at-threshold)."""
    valid = [(s, g) for s, g in zip(scores, gt_slop) if s == s]  # drop NaN
    if not valid:
        return float("nan"), confusion([], [])
    best_f1, best_t, best_c = -1.0, float("nan"), confusion([], [])
    for t, _ in valid:
        dec = [s <= t for s, _ in valid]          # reject (predict slop) if perplexity <= t
        c = confusion(dec, [g for _, g in valid])
        if c["f1"] > best_f1:
            best_f1, best_t, best_c = c["f1"], t, c
    return round(best_t, 3), best_c


def roc_auc(scores: list[float], gt_slop: list[bool]) -> float:
    """Threshold-free discrimination of perplexity vs the slop label (lower perplexity = more slop-like).
    AUC ~ 0.5 means the detector carries no signal separating AI-slop from AI-legit."""
    pairs = [(-s, g) for s, g in zip(scores, gt_slop) if s == s]  # negate: higher => more slop-like
    pos = [x for x, g in pairs if g]
    neg = [x for x, g in pairs if not g]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def bootstrap_ci(decisions: list[bool], gt_slop: list[bool], stat="f1",
                 iters=2000, seed=12345) -> tuple[float, float]:
    """95% percentile bootstrap CI for a confusion-derived statistic. Deterministic via a fixed LCG."""
    n = len(decisions)
    if n == 0:
        return (0.0, 0.0)
    s = seed
    vals = []
    for _ in range(iters):
        idx = []
        for _ in range(n):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF
            idx.append(s % n)
        c = confusion([decisions[i] for i in idx], [gt_slop[i] for i in idx])
        vals.append(c[stat])
    vals.sort()
    return (round(vals[int(0.025 * iters)], 3), round(vals[int(0.975 * iters)], 3))


def mcc(decisions: list[bool], gt_slop: list[bool]) -> float:
    """Matthews correlation / φ between an arm's reject-decision and the slop label.
    The headline effect size: 0 = the decision is independent of truth; ±1 = perfect (anti)alignment.
    On a balanced 2×2 a structure-only gate scores ≈0 — it carries no truth signal."""
    c = confusion(decisions, gt_slop)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return round((tp * tn - fp * fn) / denom, 3) if denom else 0.0


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion — correct at extremes (90/90 → ≈[0.96, 1.0])."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z / d) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (round(max(0.0, center - half), 3), round(min(1.0, center + half), 3))


def odds_ratio(decisions: list[bool], gt_slop: list[bool]) -> float:
    """Odds ratio of reject given slop vs given legit, Haldane-corrected for zero cells."""
    c = confusion(decisions, gt_slop)
    tp, fp, fn, tn = c["tp"] + 0.5, c["fp"] + 0.5, c["fn"] + 0.5, c["tn"] + 0.5
    return round((tp * tn) / (fp * fn), 2)


def paired_discordance(a: list[bool], b: list[bool], gt_slop: list[bool]) -> dict:
    """Of the items where arms a and b disagree on correctness, how often is b the correct one?
    The most human-readable head-to-head: 'when they disagree, b is right X% [CI]'."""
    ca = [d == g for d, g in zip(a, gt_slop)]   # correct = (reject == slop)
    cb = [d == g for d, g in zip(b, gt_slop)]
    b_right = sum(1 for x, y in zip(ca, cb) if y and not x)
    a_right = sum(1 for x, y in zip(ca, cb) if x and not y)
    nd = a_right + b_right
    return dict(n_discordant=nd, b_right=b_right, a_right=a_right,
                ratio_b=round(b_right / nd, 3) if nd else 0.0, ci=wilson_ci(b_right, nd))


def cluster_bootstrap_ci(decisions: list[bool], gt_slop: list[bool], clusters: list,
                         stat_fn, iters: int = 2000, seed: int = 12345) -> tuple[float, float]:
    """95% percentile bootstrap CI where the resampling unit is the CLUSTER (e.g. topic), not the
    item — so the interval respects intra-cluster correlation (pseudo-replication). stat_fn(dec, gt)
    returns the statistic (e.g. mcc). Deterministic via a fixed LCG."""
    groups: dict = {}
    for i, ck in enumerate(clusters):
        groups.setdefault(ck, []).append(i)
    blocks = list(groups.values())
    m = len(blocks)
    if m == 0:
        return (0.0, 0.0)
    s = seed
    vals = []
    for _ in range(iters):
        idx = []
        for _ in range(m):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF
            idx.extend(blocks[s % m])
        vals.append(stat_fn([decisions[i] for i in idx], [gt_slop[i] for i in idx]))
    vals.sort()
    return (round(vals[int(0.025 * iters)], 3), round(vals[int(0.975 * iters)], 3))


def fleiss_kappa(votes: list[dict], panel: list[str], cats=("slop", "legit")) -> float:
    """votes[i] = {provider: label}; standard Fleiss across items with a fixed rater count."""
    n = len(votes)
    if n == 0:
        return 0.0
    r = len(panel)
    P_i = []
    cat_tot = {c: 0 for c in cats}
    for v in votes:
        counts = {c: sum(1 for p in panel if v.get(p) == c) for c in cats}
        for c in cats:
            cat_tot[c] += counts[c]
        s = sum(counts[c] * (counts[c] - 1) for c in cats)
        P_i.append(s / (r * (r - 1)) if r > 1 else 1.0)
    P_bar = sum(P_i) / n
    p_j = {c: cat_tot[c] / (n * r) for c in cats}
    P_e = sum(p_j[c] ** 2 for c in cats)
    return round((P_bar - P_e) / (1 - P_e), 3) if (1 - P_e) else 1.0
