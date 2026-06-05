"""B3 — retrieval-grounding arm, leveraging the real CANONIC CORPUS (bge-m3, cached doc embeddings).

The paper's Gate 2 ("evidence") is implemented trivially as commit-existence. The CANONIC corpus-KG
does the real thing: does a claim *retrieve support* from an evidence corpus? Here we operationalize
that as the max cosine similarity of an item's embedding (real @cf/baai/bge-m3) to any of the 129
cached canonic governed-document embeddings. Low max-similarity ⇒ ungrounded ⇒ predict slop.

KEY CONFOUND (a finding, not a bug): grounding is corpus-relative. The benchmark items are generic
CS/biomed; the cached corpus is CANONIC governance/medicine. Where the corpus does not cover the
claim's domain, grounding cannot separate slop from legit — which is exactly the law a grounding gate
obeys. The on-domain positive control (grounding_control.py) shows it works when the corpus covers.

Doc embeddings are cached on disk; only the per-item query embedding needs one Workers-AI call (cached).
"""
from __future__ import annotations
import sys
import json
import hashlib
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path.home() / ".canonic"))
from lib.corpus import embed as CE  # noqa: E402

ROOT = pathlib.Path(__file__).parent
EMB_CACHE = ROOT / "results/embed_cache.json"
CORPUS_EMB = pathlib.Path.home() / ".canonic/data/corpus/embeddings/canonic.generated.json"
_lock = threading.Lock()


def load_corpus():
    d = json.loads(CORPUS_EMB.read_text())
    return [(k, CE.normalize(v["vector"])) for k, v in d["vectors"].items()]


def _embed_query(text: str, cache: dict):
    h = hashlib.sha256(text.encode()).hexdigest()
    if h in cache:
        return cache[h]
    v = CE.embed_text(text)
    v = CE.normalize(v) if v else None
    with _lock:
        cache[h] = v
    return v


def _score(qvec, corpus):
    if not qvec:
        return None
    sims = sorted((CE.cosine(qvec, cv) for _, cv in corpus), reverse=True)
    return {"max": round(sims[0], 4), "mean_top5": round(sum(sims[:5]) / 5, 4)}


def ground(items, workers: int = 4):
    """items: list of {id, text}. Returns {id: {max, mean_top5}}; caches query embeddings."""
    cache = json.loads(EMB_CACHE.read_text()) if EMB_CACHE.exists() else {}
    corpus = load_corpus()
    out: dict = {}

    def work(it):
        return it["id"], _score(_embed_query(it["text"], cache), corpus)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(work, it) for it in items]):
            i, s = fut.result()
            out[i] = s
            done += 1
            if done % 50 == 0:
                with _lock:
                    EMB_CACHE.write_text(json.dumps(cache))
                print(f"  grounded {done}/{len(items)}")
    EMB_CACHE.write_text(json.dumps(cache))
    return out


def main():
    man = json.loads((ROOT / "results/labeled_manifest.json").read_text())
    items = [{"id": m["id"], "text": m["prose"]} for m in man]
    g = ground(items)
    (ROOT / "results/grounding.json").write_text(json.dumps(g, indent=2))
    byk: dict = {}
    for m in man:
        s = g.get(m["id"])
        if s:
            byk.setdefault(m["klass"], []).append(s["max"])
    print(f"\ncorpus = {len(load_corpus())} canonic docs | per-class mean max-similarity:")
    for k in ["C1", "C2", "C3", "C4", "Gintro", "Gevid", "Gscope"]:
        xs = byk.get(k, [])
        print(f"  {k:7s} {round(sum(xs)/len(xs), 4) if xs else None}  (n={len(xs)})")
    print("wrote results/grounding.json")


if __name__ == "__main__":
    main()
