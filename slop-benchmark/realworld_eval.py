"""External-validity arm: run the gates on REAL content (no synthesis).

Legit = 30 real arXiv abstracts (deterministic API fetch). Slop = 17 provenance-verified real items
(retracted-for-fraud, predatory-journal, content-farm). Ground truth by construction.

The sharp test is per-source: a retracted-for-FRAUD abstract is fabricated *data* in normal scientific
prose — it reads legitimate — whereas content-farm / predatory text is prose-level slop. If the semantic
judge (and the panel) catch the latter but not the former, that is the real-world face of "fluency
defeats the judge": the slop that matters most (fraud) is invisible at the surface.
"""
from __future__ import annotations
import json
import pathlib
import shutil
import providers as P
import arms
import metrics as M
import validator
import grounding
from generate_corpus import _write_scope

ROOT = pathlib.Path(__file__).parent
SCOPES = ROOT / "corpus_realworld"


def main():
    panel = P.available_providers()
    judge = next((p for p in ["openai", "deepseek", "anthropic"] if p in panel), panel[0])
    legit = [{**x, "gt_slop": False} for x in json.loads((ROOT / "realworld_legit.json").read_text())]
    slop = [{**x, "gt_slop": True} for x in json.loads((ROOT / "realworld_slop.json").read_text()) if x["label"] == "slop"]
    for i, it in enumerate(legit + slop):
        it["id"] = f"RW-{i:03d}"
    items = legit + slop
    led = json.loads((ROOT / "ledger.json").read_text())
    led_items = list(led.items())
    print(f"real-world: {len(legit)} legit (arXiv) + {len(slop)} slop (verified) | judge={judge}")

    if SCOPES.exists():
        shutil.rmtree(SCOPES)
    # T: structurally wrap each → admit (structure trivially satisfiable on real content too)
    t_reject = {}
    for k, it in enumerate(items):
        sha, date = led_items[k % len(led_items)]
        d = SCOPES / it["id"]
        _write_scope(d, claims=[{"text": it["text"], "terms": [], "evidence_commit": sha, "evidence_date": date}],
                     vocab={}, window=(date, date), scaffolded=True, circular_vocab=True)
        t_reject[it["id"]] = not validator.validate(str(d), led).accept

    # B3 grounding similarity (off-domain canonic corpus → coverage floor; reported for completeness)
    sims = grounding.ground([{"id": it["id"], "text": it["text"]} for it in items])

    # B2 semantic + cross-family panel (panel = independent GT cross-check)
    b2 = {it["id"]: arms.b2_quality(it["text"], judge) for it in items}
    panel = {it["id"]: arms.panel_vote(it["text"], panel)[0] for it in items}

    gt = [it["gt_slop"] for it in items]
    t_dec = [t_reject[it["id"]] for it in items]
    b2_dec = [bool(b2[it["id"]]) for it in items]
    thr, _ = M.best_threshold([-(sims[it["id"]]["max"]) if sims.get(it["id"]) else float("nan") for it in items], gt)
    b3_dec = [(sims.get(it["id"]) is not None and -sims[it["id"]]["max"] <= thr) for it in items]

    out = {"n": len(items), "n_legit": len(legit), "n_slop": len(slop), "judge": judge, "association": {}, "by_source": {}}
    for name, dec in [("T-structure", t_dec), ("B3-similarity", b3_dec), ("B2-semantic", b2_dec)]:
        c = M.confusion(dec, gt)
        out["association"][name] = {"phi": M.mcc(dec, gt), "f1": c["f1"],
                                    "catch_slop": round(c["tp"] / max(1, sum(gt)), 3),
                                    "false_reject_legit": round(c["fp"] / max(1, len(gt) - sum(gt)), 3)}
    # per-source: B2 catch + panel-calls-slop rate (the fraud-invisibility test)
    for it in items:
        s = out["by_source"].setdefault(it["source_type"], {"n": 0, "b2_caught": 0, "panel_slop": 0, "gt_slop": it["gt_slop"]})
        s["n"] += 1
        s["b2_caught"] += bool(b2[it["id"]])
        s["panel_slop"] += (panel[it["id"]] == "slop")
    for k, v in out["by_source"].items():
        v["b2_catch_rate"] = round(v["b2_caught"] / v["n"], 3)
        v["panel_slop_rate"] = round(v["panel_slop"] / v["n"], 3)

    (ROOT / "results/realworld_results.json").write_text(json.dumps(out, indent=2))
    print("\ngate association with truth on REAL content (φ):")
    for name, v in out["association"].items():
        print(f"  {name:14s} φ={v['phi']:+.3f}  catch_slop={v['catch_slop']:.3f}  false_reject_legit={v['false_reject_legit']:.3f}")
    print("\nper-source — B2 catch vs panel-calls-slop (fraud invisibility test):")
    for k, v in sorted(out["by_source"].items(), key=lambda kv: (-int(kv[1]['gt_slop']), kv[0])):
        tag = "SLOP" if v["gt_slop"] else "legit"
        print(f"  {k:22s}[{tag:5s}] n={v['n']:2d}  B2_catch={v['b2_catch_rate']:.2f}  panel_slop={v['panel_slop_rate']:.2f}")
    print("wrote results/realworld_results.json")


if __name__ == "__main__":
    main()
