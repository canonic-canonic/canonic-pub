"""Full-rigor benchmark: independent panel labels -> all arms -> metrics w/ bootstrap CIs -> report.

Checkpointed: every expensive cloud result is cached in results/cache.json keyed by item id + a model
signature, so a crash or rate-limit pause resumes instead of re-paying. Cloud calls run concurrently;
the local MLX perplexity arm and the validator arms run in-process.

    OPENAI_MODEL=gpt-4o ANTHROPIC_MODEL=claude-sonnet-4-6 python3 run.py
"""
from __future__ import annotations
import json
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import providers as P
import arms
import metrics as M

ROOT = pathlib.Path(__file__).parent
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache.json"
CLASSES = ["C1", "C2", "C3", "C4", "Gintro", "Gevid", "Gscope"]
CORE = ["C1", "C2", "C3", "C4"]
ABLATIONS = {"T": {}, "T-Introspection": dict(skip_introspection=True),
             "T-Evidence": dict(skip_evidence=True), "T-Scope": dict(skip_scope=True)}
_lock = threading.Lock()


def main(workers: int = 6, use_mlx: bool = True):
    RESULTS.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "corpus_manifest.json").read_text())
    ledger = json.loads((ROOT / "ledger.json").read_text())
    panel = P.available_providers()
    judge = next((p for p in ["openai", "deepseek", "anthropic"] if p in panel), panel[0])
    models = {p: P.model_for(p) for p in panel}
    sig = "|".join(f"{p}={models[p]}" for p in panel)
    print(f"panel={panel} models={models} judge={judge} items={len(manifest)} mlx={use_mlx}")

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    # ---- cloud phase: panel labels + LLM-detector proxy + semantic judge (concurrent, cached) ----
    def cloud(item):
        key = item["id"]
        ent = cache.get(key, {})
        if ent.get("sig") == sig and "panel_gt" in ent and "b2" in ent:
            return key, ent
        prose = item["prose"]
        label, votes = arms.panel_vote(prose, panel)
        ent = {**ent, "sig": sig, "panel_votes": votes, "panel_gt": label,
               "b1b": arms.b1_detect(prose, judge), "b2": arms.b2_quality(prose, judge)}
        return key, ent

    todo = [m for m in manifest if not (cache.get(m["id"], {}).get("sig") == sig
                                        and "panel_gt" in cache.get(m["id"], {}) and "b2" in cache.get(m["id"], {}))]
    print(f"cloud: {len(manifest)-len(todo)} cached, {len(todo)} to fetch")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(cloud, m) for m in todo]):
            key, ent = fut.result()
            with _lock:
                cache[key] = ent
                done += 1
                if done % 20 == 0:
                    CACHE.write_text(json.dumps(cache, indent=2))
                    print(f"  cloud {done}/{len(todo)}")
    CACHE.write_text(json.dumps(cache, indent=2))

    # ---- local MLX perplexity arm (sequential; Metal) ----
    if use_mlx:
        try:
            from mlx_detector import perplexity, MODEL_REPO
            need = [m for m in manifest if "b1_ppl" not in cache.get(m["id"], {})]
            print(f"mlx: model={MODEL_REPO}  {len(manifest)-len(need)} cached, {len(need)} to score")
            for i, m in enumerate(need):
                cache[m["id"]]["b1_ppl"] = perplexity(m["prose"])
                if (i + 1) % 50 == 0:
                    CACHE.write_text(json.dumps(cache, indent=2)); print(f"  mlx {i+1}/{len(need)}")
            CACHE.write_text(json.dumps(cache, indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"mlx arm skipped: {e}"); use_mlx = False

    # ---- assemble ----
    klasses = [m["klass"] for m in manifest]
    gt_slop = [cache[m["id"]]["panel_gt"] == "slop" for m in manifest]
    intended = [m["intended_gt"] == "slop" for m in manifest]
    votes = [cache[m["id"]]["panel_votes"] for m in manifest]

    dec: dict[str, list[bool]] = {}
    dec["B0-admit-all"] = [False] * len(manifest)
    dec["B1b-llm-detector"] = [bool(cache[m["id"]]["b1b"]) for m in manifest]
    dec["B2-quality-judge"] = [bool(cache[m["id"]]["b2"]) for m in manifest]
    for name, abl in ABLATIONS.items():
        dec[name] = [arms.t_validator(str(ROOT / m["dir"]), ledger, **abl) for m in manifest]

    # B1 MLX: choose F1-optimal threshold (best-case detector) + threshold-free AUC
    b1_auc = None
    if use_mlx:
        ppl = [cache[m["id"]].get("b1_ppl", float("nan")) for m in manifest]
        thr, _ = M.best_threshold(ppl, gt_slop)
        dec["B1-mlx-ppl@bestF1"] = [(p == p and p <= thr) for p in ppl]
        b1_auc = M.roc_auc(ppl, gt_slop)

    # ---- metrics ----
    report = {
        "n": len(manifest), "panel": panel, "models": models,
        "panel_fleiss_kappa": M.fleiss_kappa(votes, panel),
        "manipulation_check_panel_vs_intended": round(sum(a == b for a, b in zip(gt_slop, intended)) / len(manifest), 3),
        "b1_mlx_roc_auc_vs_slop": b1_auc,
        "arms": {}, "f1_ci95": {}, "per_class_reject_rate": {}, "ablation_marginal_catch": {}, "headline": {},
    }
    for name, d in dec.items():
        report["arms"][name] = M.confusion(d, gt_slop)
        report["f1_ci95"][name] = M.bootstrap_ci(d, gt_slop, "f1")
        report["per_class_reject_rate"][name] = {k: M.reject_rate(d, klasses, k) for k in CLASSES}

    # per-gate ablation: how much each gate uniquely catches (T reject - ablated reject on its class)
    t = report["per_class_reject_rate"]["T"]
    for gate, klass in [("Introspection", "Gintro"), ("Evidence", "Gevid"), ("Scope", "Gscope")]:
        abl = report["per_class_reject_rate"][f"T-{gate}"]
        report["ablation_marginal_catch"][gate] = {
            "class": klass, "T_rejects": t[klass], "T_minus_gate_rejects": abl[klass],
            "marginal": round(t[klass] - abl[klass], 3)}

    report["headline"] = {
        "gap_C2_minus_C3": round(t["C2"] - t["C3"], 3),
        "tax_C4": t["C4"],
        "T_f1": report["arms"]["T"]["f1"], "T_f1_ci95": report["f1_ci95"]["T"],
        "B2_f1": report["arms"]["B2-quality-judge"]["f1"],
        "B2_catches_C3_better_than_T":
            report["per_class_reject_rate"]["B2-quality-judge"]["C3"] > t["C3"],
        "b1_mlx_auc": b1_auc,
    }

    (RESULTS / "results.json").write_text(json.dumps(report, indent=2))
    for m in manifest:
        m["panel_gt"] = cache[m["id"]]["panel_gt"]; m["panel_votes"] = cache[m["id"]]["panel_votes"]
    (RESULTS / "labeled_manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_md(report)
    print(json.dumps(report["headline"], indent=2)); print("wrote results/report.md")


def _write_md(r: dict):
    panel_str = ", ".join(f"{p} ({r['models'][p]})" for p in r["panel"])
    L = [f"# Slop-admission benchmark (n={r['n']})", "",
         f"Panel: {panel_str}",
         f"Panel Fleiss' κ **{r['panel_fleiss_kappa']}** · panel-vs-intended **{r['manipulation_check_panel_vs_intended']}**"
         f" · B1-MLX ROC-AUC vs slop **{r['b1_mlx_roc_auc_vs_slop']}**", "",
         "## Arms (positive = slop; reject = predict slop). F1 with 95% bootstrap CI.", "",
         "| Arm | precision | recall | F1 | F1 95% CI | accuracy |",
         "|-----|-----------|--------|----|-----------|----------|"]
    for name, c in r["arms"].items():
        ci = r["f1_ci95"][name]
        L.append(f"| {name} | {c['precision']} | {c['recall']} | {c['f1']} | [{ci[0]}, {ci[1]}] | {c['accuracy']} |")
    L += ["", "## Per-class reject rate", "",
          "| Arm | C1 legit✓ | C2 naive slop | C3 dressed slop | C4 legit bare | Gintro | Gevid | Gscope |",
          "|-----|------|------|------|------|------|------|------|"]
    for name, pr in r["per_class_reject_rate"].items():
        L.append(f"| {name} | {pr['C1']} | {pr['C2']} | {pr['C3']} | {pr['C4']} | {pr['Gintro']} | {pr['Gevid']} | {pr['Gscope']} |")
    L += ["", "## Per-gate ablation — what each gate *uniquely* catches", "",
          "| Gate | class it should catch | T rejects | T−gate rejects | marginal catch |",
          "|------|------|------|------|------|"]
    for g, v in r["ablation_marginal_catch"].items():
        L.append(f"| {g} | {v['class']} | {v['T_rejects']} | {v['T_minus_gate_rejects']} | **{v['marginal']}** |")
    h = r["headline"]
    L += ["", "## Headline", "",
          f"- **Gap (C2−C3) = {h['gap_C2_minus_C3']}** — naive slop stopped, dressed slop admitted.",
          f"- **Tax (C4) = {h['tax_C4']}** — legitimate content rejected for lacking structure.",
          f"- **T F1 = {h['T_f1']}** (95% CI {h['T_f1_ci95']}) vs **B2 semantic F1 = {h['B2_f1']}**.",
          f"- **Semantic beats structural on dressed slop?** {h['B2_catches_C3_better_than_T']}.",
          f"- **MLX perplexity detector ROC-AUC vs slop = {h['b1_mlx_auc']}** "
          "(near 0.5 ⇒ no signal; high ⇒ catching lexical novelty of synthetic jargon, a corpus confound, "
          "not real-world fluent slop).", ""]
    (RESULTS / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
