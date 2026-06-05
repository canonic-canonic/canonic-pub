"""Evaluate the adversarial red-team corpus against the COMBINED defense (structure + semantics).

Each red-team item is wrapped as a structurally-valid scope (triad + circular vocab covering its claim
terms + a real in-window commit), exactly like class C3 — so the structural gate T admits it by
construction (the point: structure is trivially satisfiable). The open question is the semantic gate:
does a frontier judge (B2) still catch slop that an adversary optimized to look legitimate?

Metrics (with Wilson CIs):
  structural_admit_rate  T admits           — should be ≈1.0 (structure is not a real barrier)
  panel_slop_rate        cross-family panel still calls it slop on the prose (independent check)
  B2_catch_rate          B2 rejects it       — compare to the 0.878 non-adversarial baseline
  combined_defense_leak  genuinely-slop AND T-admits AND B2-admits — the headline new number
plus a per-strategy breakdown of which adversarial strategy evades the semantic judge most.

Run:  OPENAI_MODEL=gpt-4o ANTHROPIC_MODEL=claude-sonnet-4-6 python3 redteam_eval.py
"""
from __future__ import annotations
import json
import pathlib
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import providers as P
import arms
import metrics as M
import validator
from generate_corpus import _write_scope

ROOT = pathlib.Path(__file__).parent
CORPUS_IN = ROOT / "redteam_corpus.json"
SCOPES = ROOT / "corpus_redteam"
CACHE = ROOT / "results/redteam_cache.json"
BASELINE_B2_RECALL = 0.878  # B2 slop-catch on non-adversarial C2/C3 (from the n=480 run)
_lock = threading.Lock()


def build_scopes(items, ledger):
    if SCOPES.exists():
        shutil.rmtree(SCOPES)
    led = list(ledger.items())
    for k, it in enumerate(items):
        sha, date = led[k % len(led)]
        claims = []
        for c in it.get("claims", []):
            claims.append({"text": c["text"], "terms": c.get("terms", []),
                           "evidence_commit": sha, "evidence_date": date})
        d = SCOPES / it["id"]
        _write_scope(d, claims=claims, vocab={}, window=(date, date), scaffolded=True, circular_vocab=True)
        it["_dir"] = str(d.relative_to(ROOT))
        it["_prose"] = " ".join(c["text"] for c in claims)
    return items


def main(workers: int = 6):
    data = json.loads(CORPUS_IN.read_text())
    items = data["items"]
    ledger = json.loads((ROOT / "ledger.json").read_text())
    panel = P.available_providers()
    judge = next((p for p in ["openai", "deepseek", "anthropic"] if p in panel), panel[0])
    print(f"red-team items={len(items)} panel={panel} judge={judge} models={ {p: P.model_for(p) for p in panel} }")
    build_scopes(items, ledger)

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    sig = "|".join(f"{p}={P.model_for(p)}" for p in panel)

    def cloud(it):
        key = it["id"]
        ent = cache.get(key, {})
        if ent.get("sig") == sig:
            return key, ent
        label, votes = arms.panel_vote(it["_prose"], panel)
        ent = {"sig": sig, "panel_gt": label, "panel_votes": votes, "b2_reject": arms.b2_quality(it["_prose"], judge)}
        return key, ent

    todo = [it for it in items if cache.get(it["id"], {}).get("sig") != sig]
    print(f"cloud: {len(items)-len(todo)} cached, {len(todo)} to fetch")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(cloud, it) for it in todo]):
            key, ent = fut.result()
            with _lock:
                cache[key] = ent
                done += 1
                if done % 20 == 0:
                    CACHE.write_text(json.dumps(cache, indent=2)); print(f"  cloud {done}/{len(todo)}")
    CACHE.write_text(json.dumps(cache, indent=2))

    led = json.loads((ROOT / "ledger.json").read_text())
    n = len(items)
    t_admit = panel_slop = b2_catch = leak = 0
    by_strat: dict = {}
    for it in items:
        ent = cache[it["id"]]
        admit_T = validator.validate(str(ROOT / it["_dir"]), led).accept       # True = admitted by structure
        is_slop = ent["panel_gt"] == "slop"                                     # independent panel check
        caught_B2 = bool(ent["b2_reject"])                                      # True = semantic judge rejected
        t_admit += admit_T
        panel_slop += is_slop
        b2_catch += caught_B2
        leaked = is_slop and admit_T and not caught_B2                          # genuine slop through BOTH gates
        leak += leaked
        s = by_strat.setdefault(it["strategy"], {"n": 0, "b2_caught": 0, "leaked": 0})
        s["n"] += 1; s["b2_caught"] += caught_B2; s["leaked"] += leaked

    slop_n = panel_slop
    b2_catch_on_slop = sum(1 for it in items if cache[it["id"]]["panel_gt"] == "slop" and cache[it["id"]]["b2_reject"])
    out = {
        "n": n,
        "structural_admit_rate": {"rate": round(t_admit / n, 3), "ci": M.wilson_ci(t_admit, n)},
        "panel_slop_rate": {"rate": round(panel_slop / n, 3), "ci": M.wilson_ci(panel_slop, n)},
        "B2_catch_rate_overall": {"rate": round(b2_catch / n, 3), "ci": M.wilson_ci(b2_catch, n)},
        "B2_catch_rate_on_panel_slop": {"rate": round(b2_catch_on_slop / slop_n, 3) if slop_n else None,
                                        "ci": M.wilson_ci(b2_catch_on_slop, slop_n) if slop_n else None,
                                        "baseline_non_adversarial": BASELINE_B2_RECALL},
        "combined_defense_leak": {"k": leak, "n": n, "rate": round(leak / n, 3), "ci": M.wilson_ci(leak, n)},
        "by_strategy": {k: {**v, "b2_catch_rate": round(v["b2_caught"] / v["n"], 3),
                            "leak_rate": round(v["leaked"] / v["n"], 3)} for k, v in sorted(by_strat.items())},
    }
    (ROOT / "results/redteam_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ["structural_admit_rate", "panel_slop_rate",
          "B2_catch_rate_on_panel_slop", "combined_defense_leak"]}, indent=2))
    # most evasive strategies
    worst = sorted(out["by_strategy"].items(), key=lambda kv: -kv[1]["leak_rate"])[:5]
    print("\nmost evasive strategies (leak rate):")
    for k, v in worst:
        print(f"  {k:24s} leak={v['leak_rate']}  B2_catch={v['b2_catch_rate']}  (n={v['n']})")
    print("\nwrote results/redteam_results.json")


if __name__ == "__main__":
    main()
