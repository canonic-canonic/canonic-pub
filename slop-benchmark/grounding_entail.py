"""B3e — retrieve-THEN-entail grounding (the real grounding gate, vs B3's similarity-only).

B3 (grounding.py) thresholds max cosine similarity — which measures topical relevance, not support,
so it is ⊥ truth (φ≈0, measured). B3e does what a grounding gate must: retrieve the top-k corpus docs
(cached bge-m3 vectors) and then ask a judge whether those excerpts actually SUPPORT the claim. A
fabricated on-topic claim retrieves highly-similar docs that nonetheless fail to entail it → rejected.

Reuses the query-embedding cache from grounding.py. Doc text comes from the CAS corpus substore.
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
import providers as P  # noqa: E402

ROOT = pathlib.Path(__file__).parent
EMB_CACHE = ROOT / "results/embed_cache.json"
CORPUS_EMB = pathlib.Path.home() / ".canonic/data/corpus/embeddings/canonic.generated.json"
CAS = pathlib.Path.home() / ".canonic/cas/corpus"
_lock = threading.Lock()

ENTAIL_SYS = ("You verify grounding. Given retrieved corpus excerpts and a claim, decide whether the excerpts "
              "actually SUPPORT the specific claim. Answer supported=true ONLY if the excerpts substantiate it; "
              "false if they are merely on the same topic, insufficient, or contradict it. Topical similarity is "
              "NOT support.")
ENTAIL_ASK = ('CLAIM: "{claim}"\n\nRETRIEVED CORPUS EXCERPTS:\n{ctx}\n\n'
              'Return JSON {{"supported": true|false, "reason": "<short>"}}.')


def load_corpus_vecs():
    d = json.loads(CORPUS_EMB.read_text())
    return {k: CE.normalize(v["vector"]) for k, v in d["vectors"].items()}


def load_doc_text(max_chars: int = 2400):
    """doc_id -> concatenated passage text from the CAS corpus substore (canonic specialty)."""
    by_doc: dict = {}
    for f in glob.glob(str(CAS / "*" / "*.json")):
        try:
            p = json.loads(pathlib.Path(f).read_text())
        except Exception:  # noqa: BLE001
            continue
        if p.get("specialty") == "canonic" and p.get("text"):
            by_doc.setdefault(p["doc_id"], []).append(p["text"])
    return {d: " ".join(t for t in txts)[:max_chars] for d, txts in by_doc.items()}


def _embed_query(text, cache):
    h = hashlib.sha256(text.encode()).hexdigest()
    if h in cache and cache[h] is not None:
        return cache[h]
    v = CE.embed_text(text)
    v = CE.normalize(v) if v else None
    with _lock:
        cache[h] = v
    return v


def _topk(qvec, vecs, k=3):
    return [d for d, _ in sorted(((d, CE.cosine(qvec, v)) for d, v in vecs.items()), key=lambda x: -x[1])[:k]]


def load_passage_index():
    """Passage-level index built by build_passage_index.py: [{doc_id, text, vector}]."""
    return json.loads((ROOT / "results/passage_index.json").read_text())


def entail_items_passage(items, judge: str, k: int = 4, workers: int = 4):
    """B3e at PASSAGE granularity: retrieve top-k supporting PASSAGES (not whole docs) so the
    entailment judge sees the actual candidate supporting sentences. Fixes doc-level over-rejection."""
    cache = json.loads(EMB_CACHE.read_text()) if EMB_CACHE.exists() else {}
    idx = load_passage_index()
    pv = [(p["doc_id"], p["text"], p["vector"]) for p in idx]
    out: dict = {}

    def work(it):
        q = _embed_query(it["text"], cache)
        if not q:
            return it["id"], None
        excl = it.get("exclude_doc")  # leave-one-out: drop passages from the item's own source doc
        pool = [(d, t, v) for d, t, v in pv if d != excl] if excl else pv
        top = sorted(((d, t, CE.cosine(q, v)) for d, t, v in pool), key=lambda x: -x[2])[:k]
        ctx = "\n---\n".join(f"[{i+1}] {t[:600]}" for i, (d, t, _s) in enumerate(top))
        try:
            r = P.complete_json(judge, ENTAIL_SYS, ENTAIL_ASK.format(claim=it["text"], ctx=ctx), temperature=0)
            return it["id"], {"supported": bool(r.get("supported")), "top_docs": [d for d, _, _ in top]}
        except Exception:  # noqa: BLE001
            return it["id"], None

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(work, it) for it in items]):
            i, s = fut.result()
            out[i] = s
            done += 1
            if done % 25 == 0:
                with _lock:
                    EMB_CACHE.write_text(json.dumps(cache))
                print(f"  entail-passage {done}/{len(items)}")
    EMB_CACHE.write_text(json.dumps(cache))
    return out


def entail_items(items, judge: str, k: int = 3, workers: int = 4):
    """items: list of {id, text}. Returns {id: {supported: bool, top_docs: [...]}}; True=supported(grounded)."""
    cache = json.loads(EMB_CACHE.read_text()) if EMB_CACHE.exists() else {}
    vecs = load_corpus_vecs()
    doctext = load_doc_text()
    out: dict = {}

    def work(it):
        q = _embed_query(it["text"], cache)
        if not q:
            return it["id"], None
        ids = _topk(q, vecs, k)
        ctx = "\n---\n".join(f"[{i+1}] {doctext.get(d, '')[:700]}" for i, d in enumerate(ids))
        try:
            r = P.complete_json(judge, ENTAIL_SYS, ENTAIL_ASK.format(claim=it["text"], ctx=ctx), temperature=0)
            return it["id"], {"supported": bool(r.get("supported")), "top_docs": ids}
        except Exception:  # noqa: BLE001
            return it["id"], None

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(work, it) for it in items]):
            i, s = fut.result()
            out[i] = s
            done += 1
            if done % 25 == 0:
                with _lock:
                    EMB_CACHE.write_text(json.dumps(cache))
                print(f"  entail {done}/{len(items)}")
    EMB_CACHE.write_text(json.dumps(cache))
    return out
