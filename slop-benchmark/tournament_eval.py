"""On-domain multi-gate evaluation — where retrieval-grounding finally has corpus coverage.

Legit class = real published CANONIC passages (genuinely true, in-corpus). Slop class = the tournament's
fabricated-CANONIC claims, each workflow-verified as genuinely false. Ground truth is therefore reliable
by construction (real passages are true; verified fabrications are false), so no panel needed.

Scores four gates against truth (Matthews φ):
  T   structure        — admits all (structure trivially satisfiable) ⇒ φ≈0, again
  B3  RAG similarity    — max cosine; topical, not support ⇒ expected φ≈0 even on-domain
  B3e RAG + entailment   — retrieve THEN verify support ⇒ the gate that should finally track truth
  B2  semantic judge    — generic quality judge (not a CANONIC expert) ⇒ partial
and the combined-defense leak: fabricated slop that passes T AND B3e AND B2.

Run:  OPENAI_MODEL=gpt-4o ANTHROPIC_MODEL=claude-sonnet-4-6 python3 tournament_eval.py
"""
from __future__ import annotations
import sys
import json
import glob
import random
import pathlib
import shutil
import providers as P
import arms
import metrics as M
import validator
import grounding
import grounding_entail as GE
from generate_corpus import _write_scope

ROOT = pathlib.Path(__file__).parent
CAS = pathlib.Path.home() / ".canonic/cas/corpus"
SLOP_IN = ROOT / "tournament_slop.json"
SCOPES = ROOT / "corpus_tournament"
N_LEGIT = 60


def load_legit(n):
    rows = []
    for f in glob.glob(str(CAS / "*" / "*.json")):
        try:
            p = json.loads(pathlib.Path(f).read_text())
        except Exception:  # noqa: BLE001
            continue
        if p.get("specialty") == "canonic" and p.get("text") and len(p["text"]) > 240:
            rows.append({"doc_id": p["doc_id"], "text": p["text"][:600]})
    random.Random(11).shuffle(rows)
    seen, out = set(), []
    for r in rows:                       # one passage per doc for diversity
        if r["doc_id"] in seen:
            continue
        seen.add(r["doc_id"])
        out.append({"id": f"LEGIT-{len(out):03d}", "text": r["text"], "gt_slop": False, "doc_id": r["doc_id"]})
        if len(out) >= n:
            break
    return out


def main():
    panel = P.available_providers()
    judge = next((p for p in ["openai", "deepseek", "anthropic"] if p in panel), panel[0])
    slop_raw = json.loads(SLOP_IN.read_text())["items"]
    slop = [{"id": it["id"], "text": " ".join(c["text"] for c in it["claims"]), "gt_slop": True,
             "strategy": it.get("strategy", "")} for it in slop_raw]
    legit = load_legit(N_LEGIT)
    items = legit + slop
    print(f"on-domain corpus: {len(legit)} legit (real canonic) + {len(slop)} slop (fabricated) | judge={judge}")

    led = json.loads((ROOT / "ledger.json").read_text())
    led_items = list(led.items())
    if SCOPES.exists():
        shutil.rmtree(SCOPES)

    # T (structure): wrap each as a structurally-valid scope, like C3
    t_reject = {}
    for k, it in enumerate(items):
        sha, date = led_items[k % len(led_items)]
        claims = [{"text": it["text"], "terms": [], "evidence_commit": sha, "evidence_date": date}]
        d = SCOPES / it["id"]
        _write_scope(d, claims=claims, vocab={}, window=(date, date), scaffolded=True, circular_vocab=True)
        t_reject[it["id"]] = not validator.validate(str(d), led).accept

    # B3 similarity
    sims = grounding.ground([{"id": it["id"], "text": it["text"]} for it in items])
    thr, _ = M.best_threshold([-(sims[it["id"]]["max"]) if sims.get(it["id"]) else float("nan") for it in items],
                              [it["gt_slop"] for it in items])
    b3_reject = {it["id"]: (sims.get(it["id"]) is not None and -sims[it["id"]]["max"] <= thr) for it in items}

    # B3e retrieve-then-entail
    ent = GE.entail_items([{"id": it["id"], "text": it["text"]} for it in items], judge)
    b3e_reject = {it["id"]: (ent.get(it["id"]) is not None and not ent[it["id"]]["supported"]) for it in items}

    # B2 semantic
    b2_reject = {it["id"]: arms.b2_quality(it["text"], judge) for it in items}

    gt = [it["gt_slop"] for it in items]
    ids = [it["id"] for it in items]
    arms_dec = {
        "T-structure": [t_reject[i] for i in ids],
        "B3-similarity": [b3_reject[i] for i in ids],
        "B3e-entailment": [b3e_reject[i] for i in ids],
        "B2-semantic": [b2_reject[i] for i in ids],
    }
    out = {"n": len(items), "n_legit": len(legit), "n_slop": len(slop), "judge": judge, "association": {}}
    for name, dec in arms_dec.items():
        c = M.confusion(dec, gt)
        out["association"][name] = {"phi": M.mcc(dec, gt), "f1": c["f1"], "precision": c["precision"],
                                    "recall": c["recall"], "slop_catch_rate": round(c["tp"] / max(1, sum(gt)), 3),
                                    "legit_false_reject": round(c["fp"] / max(1, len(gt) - sum(gt)), 3)}
    # combined defense: slop that passes T AND B3e AND B2 (admitted by all)
    leak = sum(1 for k, it in enumerate(items) if it["gt_slop"]
               and not arms_dec["T-structure"][k] and not arms_dec["B3e-entailment"][k] and not arms_dec["B2-semantic"][k])
    out["combined_defense_leak"] = {"k": leak, "n_slop": len(slop), "rate": round(leak / max(1, len(slop)), 3),
                                    "ci": M.wilson_ci(leak, len(slop))}
    (ROOT / "results/tournament_results.json").write_text(json.dumps(out, indent=2))
    print("\non-domain gate association with truth (φ) + slop-catch / legit-false-reject:")
    for name, v in out["association"].items():
        print(f"  {name:16s} φ={v['phi']:+.3f}  catch_slop={v['slop_catch_rate']:.3f}  false_reject_legit={v['legit_false_reject']:.3f}")
    print(f"\ncombined-defense leak (slop passing T+B3e+B2): {leak}/{len(slop)} = {out['combined_defense_leak']['rate']} "
          f"CI{out['combined_defense_leak']['ci']}")
    print("wrote results/tournament_results.json")


if __name__ == "__main__":
    main()
