"""Non-parametric significance — permutation tests, assumption-free and cluster-aware.

Fisher/McNemar assume independent items, but ours cluster by topic (pseudo-replication). And the
C2-vs-C3 "gap" is deterministic, so its p is significance-by-construction. We therefore test the
quantity that is *not* constructed — the **association** with content-truth — and we permute at the
topic-cluster level so the null respects non-independence.

  TEST 1  φ(B2) − φ(T): classifier-swap permutation. Under H0 (the two arms are exchangeable in their
          association with truth), randomly swap the two arms' decisions within whole topic clusters
          and recompute the φ difference. Item-level and topic-cluster-level p both reported; the
          cluster-level one is the honest figure.
  TEST 2  label-permutation null for each arm's F1 — shuffle slop/legit labels, recompute F1: a
          distribution-free "does this arm beat chance?" check.

Deterministic: fixed-seed RNG, Monte-Carlo permutations. Restricted to the balanced 2×2 (C1–C4).
"""
from __future__ import annotations
import json
import pathlib
import random
import validator
import metrics as M

ROOT = pathlib.Path(__file__).parent
ITERS = 20000
SEED = 20260605
CORE = ("C1", "C2", "C3", "C4")


def _load():
    man = json.loads((ROOT / "results/labeled_manifest.json").read_text())
    led = json.loads((ROOT / "ledger.json").read_text())
    cache = json.loads((ROOT / "results/cache.json").read_text())
    gt = [m["panel_gt"] == "slop" for m in man]
    topics = [m["topic"] for m in man]
    klass = [m["klass"] for m in man]
    T = [not validator.validate(str(ROOT / m["dir"]), led).accept for m in man]
    B2 = [bool(cache[m["id"]]["b2"]) for m in man]
    B1b = [bool(cache[m["id"]]["b1b"]) for m in man]
    ppl = [cache[m["id"]].get("b1_ppl", float("nan")) for m in man]
    thr, _ = M.best_threshold(ppl, gt)
    B1 = [(p == p and p <= thr) for p in ppl]
    return gt, topics, klass, {"T": T, "B2": B2, "B1b-llm": B1b, "B1-mlx": B1}


def phi_diff_cluster_perm(gt, clusters, b, a, iters: int = ITERS, seed: int = SEED) -> dict:
    """Permutation p for φ(b) − φ(a): swap the two arms' decisions within randomly chosen clusters."""
    obs = M.mcc(b, gt) - M.mcc(a, gt)
    rng = random.Random(seed)
    groups: dict = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    blocks = list(groups.values())

    def run(block_list):
        ge = 0
        for _ in range(iters):
            bb, aa = list(b), list(a)
            for blk in block_list:
                if rng.random() < 0.5:
                    for i in blk:
                        bb[i], aa[i] = aa[i], bb[i]
            if abs(M.mcc(bb, gt) - M.mcc(aa, gt)) >= abs(obs):
                ge += 1
        return (ge + 1) / (iters + 1)

    return dict(stat=round(obs, 3), iters=iters,
                p_item=run([[i] for i in range(len(gt))]), p_cluster=run(blocks))


def f1_label_permutation(gt, decisions, iters: int = ITERS, seed: int = SEED) -> dict:
    obs = M.confusion(decisions, gt)["f1"]
    rng = random.Random(seed)
    y = list(gt)
    ge = 0
    null_max = 0.0
    for _ in range(iters):
        rng.shuffle(y)
        f1 = M.confusion(decisions, y)["f1"]
        null_max = max(null_max, f1)
        if f1 >= obs:
            ge += 1
    return dict(f1=obs, p=(ge + 1) / (iters + 1), null_max=round(null_max, 3))


def main():
    gt, topics, klass, arms = _load()
    core = [i for i, k in enumerate(klass) if k in CORE]
    gtc = [gt[i] for i in core]
    topc = [topics[i] for i in core]
    sub = lambda d: [d[i] for i in core]  # noqa: E731

    pp = phi_diff_cluster_perm(gtc, topc, sub(arms["B2"]), sub(arms["T"]))
    out = {"iters": ITERS, "seed": SEED, "n_core": len(core), "n_topics": len(set(topc)),
           "phi_diff_B2_minus_T": pp,
           "f1_label_perm": {k: f1_label_permutation(gtc, sub(v)) for k, v in arms.items()}}
    (ROOT / "results/nonparametric.json").write_text(json.dumps(out, indent=2))

    print(f"balanced 2×2 core: n={len(core)}  topics(clusters)={out['n_topics']}  iters={ITERS}")
    print(f"\nTEST 1 — φ(B2) − φ(T) = {pp['stat']}  (classifier-swap permutation)")
    print(f"  p (item-level)    = {pp['p_item']:.2e}")
    print(f"  p (topic-cluster) = {pp['p_cluster']:.2e}   <-- honest, respects non-independence")
    print("\nTEST 2 — label-permutation null: does each arm discriminate slop > chance?")
    for k, v in out["f1_label_perm"].items():
        print(f"  {k:9s} F1={v['f1']:.3f}  null_max_F1={v['null_max']:.3f}  p={v['p']:.2e}")


if __name__ == "__main__":
    main()
