"""Build a passage-level embedding index over the CANONIC corpus (for passage-level B3e retrieval).

Doc-level retrieval + 700-char excerpts caused B3e to over-reject legit (the supporting sentence often
wasn't in the excerpt). Passage granularity gives the entailment judge the actual supporting text.
Embeds each canonic CAS passage via @cf/baai/bge-m3 (cached by text-sha) → results/passage_index.json.
"""
from __future__ import annotations
import sys
import json
import glob
import hashlib
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path.home() / ".canonic"))
from lib.corpus import embed as CE  # noqa: E402

ROOT = pathlib.Path(__file__).parent
CAS = pathlib.Path.home() / ".canonic/cas/corpus"
CACHE = ROOT / "results/passage_emb_cache.json"
OUT = ROOT / "results/passage_index.json"
MAXLEN = 1200
_lock = threading.Lock()


def load_passages():
    seen, out = set(), []
    for f in glob.glob(str(CAS / "*" / "*.json")):
        try:
            p = json.loads(pathlib.Path(f).read_text())
        except Exception:  # noqa: BLE001
            continue
        if p.get("specialty") != "canonic" or not p.get("text"):
            continue
        t = p["text"][:MAXLEN].strip()
        h = hashlib.sha256(t.encode()).hexdigest()
        if len(t) < 120 or h in seen:
            continue
        seen.add(h)
        out.append({"doc_id": p["doc_id"], "text": t, "sha": h})
    return out


def main(workers: int = 8):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    passages = load_passages()
    print(f"canonic passages (deduped): {len(passages)} | cached: {sum(1 for p in passages if p['sha'] in cache)}")

    def work(p):
        if p["sha"] in cache and cache[p["sha"]]:
            return p["sha"], cache[p["sha"]]
        v = CE.embed_text(p["text"])
        return p["sha"], (CE.normalize(v) if v else None)

    todo = [p for p in passages if p["sha"] not in cache or not cache[p["sha"]]]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(work, p) for p in todo]):
            sha, vec = fut.result()
            with _lock:
                cache[sha] = vec
                done += 1
                if done % 200 == 0:
                    CACHE.write_text(json.dumps(cache))
                    print(f"  embedded {done}/{len(todo)}")
    CACHE.write_text(json.dumps(cache))
    index = [{"doc_id": p["doc_id"], "text": p["text"], "vector": cache[p["sha"]]}
             for p in passages if cache.get(p["sha"])]
    OUT.write_text(json.dumps(index))
    print(f"passage index: {len(index)} passages with embeddings → {OUT.name}")


if __name__ == "__main__":
    main()
