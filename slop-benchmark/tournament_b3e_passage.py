"""Re-evaluate B3e at PASSAGE granularity on the on-domain corpus, vs the doc-level baseline.

Doc-level B3e (tournament_eval.py): φ=0.271, catch_slop=1.00, false_reject_legit=0.87. The hypothesis:
passage-level retrieval gives the entailment judge the actual supporting sentence, so legit passages
get correctly admitted → false-reject drops → φ climbs. Only B3e is recomputed (B2/B3/T unchanged).
"""
from __future__ import annotations
import json
import pathlib
import providers as P
import metrics as M
import grounding_entail as GE
from tournament_eval import load_legit, N_LEGIT, SLOP_IN

ROOT = pathlib.Path(__file__).parent
DOC_LEVEL = {"phi": 0.271, "catch_slop": 1.0, "false_reject_legit": 0.867}


def main():
    judge = next((p for p in ["openai", "deepseek", "anthropic"] if p in P.available_providers()), "openai")
    slop = [{"id": it["id"], "text": " ".join(c["text"] for c in it["claims"]), "gt_slop": True}
            for it in json.loads(SLOP_IN.read_text())["items"]]
    legit = load_legit(N_LEGIT)
    items = legit + slop
    print(f"on-domain: {len(legit)} legit + {len(slop)} slop | passage-level B3e | judge={judge}")

    # leave-one-out: legit items exclude their own source doc from retrieval (no self-grounding leak)
    probe = [{"id": it["id"], "text": it["text"], "exclude_doc": it.get("doc_id")} for it in items]
    ent = GE.entail_items_passage(probe, judge, k=4)
    dec = [(ent.get(it["id"]) is not None and not ent[it["id"]]["supported"]) for it in items]  # reject if unsupported
    gt = [it["gt_slop"] for it in items]
    c = M.confusion(dec, gt)
    phi = M.mcc(dec, gt)
    catch = round(c["tp"] / max(1, sum(gt)), 3)
    fr = round(c["fp"] / max(1, len(gt) - sum(gt)), 3)
    out = {"granularity": "passage", "phi": phi, "f1": c["f1"], "precision": c["precision"], "recall": c["recall"],
           "catch_slop": catch, "false_reject_legit": fr, "doc_level_baseline": DOC_LEVEL}
    (ROOT / "results/b3e_passage_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nB3e PASSAGE  φ={phi:+.3f}  catch_slop={catch:.3f}  false_reject_legit={fr:.3f}")
    print(f"B3e DOC      φ={DOC_LEVEL['phi']:+.3f}  catch_slop={DOC_LEVEL['catch_slop']:.3f}  "
          f"false_reject_legit={DOC_LEVEL['false_reject_legit']:.3f}")
    d = phi - DOC_LEVEL["phi"]
    print(f"\nΔφ = {d:+.3f} — passage-level {'improves' if d > 0 else 'does not improve'} grounding discrimination.")
    print("wrote results/b3e_passage_results.json")


if __name__ == "__main__":
    main()
