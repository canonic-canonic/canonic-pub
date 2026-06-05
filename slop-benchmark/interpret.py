"""Non-circular report: measure each arm's ASSOCIATION with content-truth, not constructed separation.

The structural gate decides on structure, which the corpus assigns by class — so scoring it against a
content-truth label as if it were a slop classifier is circular (its aggregate F1 is just a function
of the class mix). The honest question is: does an arm's accept/reject decision *correlate* with
whether the content is slop? On the balanced 2×2 (C1/C2/C3/C4, 90 each) the two axes are decorrelated,
so φ(T, truth) ≈ 0 by construction — the gate carries no truth signal — while φ(B2, truth) ≫ 0.

Consumes results/labeled_manifest.json + results/cache.json (no model re-run). Emits results/interpret.{md,json}.
"""
from __future__ import annotations
import json
import pathlib
import validator
import metrics as M

ROOT = pathlib.Path(__file__).parent
CORE = ("C1", "C2", "C3", "C4")


def _load():
    man = json.loads((ROOT / "results/labeled_manifest.json").read_text())
    led = json.loads((ROOT / "ledger.json").read_text())
    cache = json.loads((ROOT / "results/cache.json").read_text())
    gt = [m["panel_gt"] == "slop" for m in man]
    klass = [m["klass"] for m in man]
    topic = [m["topic"] for m in man]
    T = [not validator.validate(str(ROOT / m["dir"]), led).accept for m in man]   # True = reject
    B2 = [bool(cache[m["id"]]["b2"]) for m in man]
    B1b = [bool(cache[m["id"]]["b1b"]) for m in man]
    ppl = [cache[m["id"]].get("b1_ppl", float("nan")) for m in man]
    thr, _ = M.best_threshold(ppl, gt)
    B1 = [(p == p and p <= thr) for p in ppl]
    return man, gt, klass, topic, {"T": T, "B2": B2, "B1b-llm": B1b, "B1-mlx": B1}


def _sub(idx, lst):
    return [lst[i] for i in idx]


def main():
    import nonparametric as NP
    man, gt, klass, topic, arms = _load()
    n = len(gt)

    # B3 — retrieval-grounding arm (real CANONIC corpus, bge-m3). Low max-similarity ⇒ ungrounded ⇒
    # predict slop. Threshold at F1-optimum (best case). Present only if grounding.json exists.
    gpath = ROOT / "results/grounding.json"
    if gpath.exists():
        g = json.loads(gpath.read_text())
        sims = [(g.get(m["id"]) or {}).get("max", float("nan")) for m in man]
        thr, _ = M.best_threshold([-(s) if s == s else float("nan") for s in sims], gt)  # noqa: E501
        arms["B3-corpus-grounding"] = [(s == s and -s <= thr) for s in sims]
    base = sum(gt) / n
    core = [i for i, k in enumerate(klass) if k in CORE]
    gtc, topc = _sub(core, gt), _sub(core, topic)

    # --- association of each arm's decision with content-truth, on the balanced 2x2 ---
    assoc = {}
    for name, dec in arms.items():
        dc = _sub(core, dec)
        assoc[name] = dict(
            phi=M.mcc(dc, gtc),
            ci_item=M.cluster_bootstrap_ci(dc, gtc, list(range(len(dc))), M.mcc),
            ci_cluster=M.cluster_bootstrap_ci(dc, gtc, topc, M.mcc),
            odds_ratio=M.odds_ratio(dc, gtc))

    # --- off-diagonal rates (the real findings) ---
    def rate(klass_name, dec, want_reject):
        idx = [i for i, k in enumerate(klass) if k == klass_name]
        k = sum(1 for i in idx if dec[i] is want_reject)
        return dict(k=k, n=len(idx), rate=round(k / len(idx), 3), ci=M.wilson_ci(k, len(idx)))
    T = arms["T"]
    c3_leak = rate("C3", T, False)   # slop admitted (not rejected)
    c4_tax = rate("C4", T, True)     # legit rejected

    # --- interpretable head-to-head: when T and B2 disagree, who is right? (on the 2x2) ---
    disc = M.paired_discordance(_sub(core, T), _sub(core, arms["B2"]), gtc)

    # --- one honest significance number: cluster-permutation p on φ(B2) − φ(T) ---
    perm = NP.phi_diff_cluster_perm(gtc, topc, _sub(core, arms["B2"]), _sub(core, T))

    out = dict(n=n, n_core=len(core), base_rate_slop=round(base, 3),
               association_2x2=assoc, c3_leakage=c3_leak, c4_tax=c4_tax,
               discordance_T_vs_B2=disc, phi_diff_cluster_perm=perm)
    (ROOT / "results/interpret.json").write_text(json.dumps(out, indent=2))
    _md(out)
    print(json.dumps({"phi_T": assoc["T"]["phi"], "phi_B2": assoc["B2"]["phi"],
                      "c3_leak": c3_leak["rate"], "c4_tax": c4_tax["rate"],
                      "B2_right_when_disagree": disc["ratio_b"], "perm_p": perm["p_cluster"]}, indent=2))
    print("wrote results/interpret.md")


def _md(o: dict):
    a = o["association_2x2"]
    L = ["# Non-circular analysis — association with content-truth", "",
         f"n={o['n']} (balanced 2×2 core = {o['n_core']}); slop base rate = {o['base_rate_slop']}.",
         "",
         "The structural gate decides on *structure*; the corpus assigns structure by class, so its",
         "decision cannot be scored as a slop classifier without circularity. We instead measure how",
         "each arm's accept/reject decision **correlates** with content-truth on the balanced 2×2,",
         "where structure and truth are decorrelated by design.", "",
         "## Association (Matthews φ; 0 = independent of truth, ±1 = perfect)", "",
         "| Arm | φ | 95% CI (item) | 95% CI (cluster=topic) | odds ratio |",
         "|-----|----|---------------|------------------------|------------|"]
    for name, v in a.items():
        L.append(f"| {name} | {v['phi']} | [{v['ci_item'][0]}, {v['ci_item'][1]}] | "
                 f"[{v['ci_cluster'][0]}, {v['ci_cluster'][1]}] | {v['odds_ratio']} |")
    leak, tax, d, p = o["c3_leakage"], o["c4_tax"], o["discordance_T_vs_B2"], o["phi_diff_cluster_perm"]
    L += ["",
          "**Read:** φ(T)≈0 with a CI bracketing 0 means structural admission is statistically "
          "independent of whether content is slop — it provides accountability, not slop-filtering. "
          "φ(B2)≫0 means semantic judgment tracks truth. Orthogonal axes ⇒ complementary, not "
          "redundant. (The cluster CI is wider than the item CI because it accounts for topic "
          "pseudo-replication.)", "",
          "## The two real findings (off-diagonal cells)", "",
          f"- **C3 leakage** — slop with valid structure **admitted**: {leak['k']}/{leak['n']} = "
          f"{leak['rate']}, 95% CI [{leak['ci'][0]}, {leak['ci'][1]}].",
          f"- **C4 tax** — legitimate content without structure **rejected**: {tax['k']}/{tax['n']} = "
          f"{tax['rate']}, 95% CI [{tax['ci'][0]}, {tax['ci'][1]}].", "",
          "## Head-to-head (interpretable)", "",
          f"- When structural (T) and semantic (B2) **disagree**, the semantic judge is correct in "
          f"**{d['b_right']}/{d['n_discordant']} = {d['ratio_b']}** cases, 95% CI [{d['ci'][0]}, {d['ci'][1]}].",
          "",
          "## Significance (one honest number, not p=10⁻⁵³ theater)", "",
          f"- φ(B2) − φ(T) = {p['stat']}; cluster-permutation (topic-level, {p['iters']} iters) "
          f"two-sided **p = {p['p_cluster']:.2e}** (item-level p = {p['p_item']:.2e}). The earlier "
          "Fisher p≈10⁻⁵³ on the C2-vs-C3 gap is significance-by-construction and is not reported as a result.", ""]
    (ROOT / "results/interpret.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
