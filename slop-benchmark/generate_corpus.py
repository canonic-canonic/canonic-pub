"""Generate the benchmark corpus via cross-provider separation.

Core 2x2 (content: legit/slop x structure: scaffolded-valid/bare):
  C1 legit + valid scaffold   -> should ACCEPT
  C2 slop  + bare             -> should REJECT (triad)
  C3 slop  + dressed-valid    -> should REJECT, but PASSES (the adversarial gap)
  C4 legit + bare             -> should ACCEPT, but REJECTS (the structure tax)

Per-gate single-failure variants (slop that an adversary left ONE gate open on, so the
ablations isolate each gate's marginal catch):
  Gintro slop + triad but one undefined term   (only introspection fails)
  Gevid  slop + triad+closed but fake commit    (only evidence gate fails)
  Gscope slop + triad+closed+real commit, date out of window (only scope gate fails)

Independence: the C3 adversary is forced to a NON-Anthropic family (CANONIC was built with Claude).
Ground-truth labels are assigned later by a panel on prose alone — never by the generator.
"""
from __future__ import annotations
import json
import subprocess
import pathlib
import shutil
import sys
import providers as P

ROOT = pathlib.Path(__file__).parent
CORPUS = ROOT / "corpus"
LEDGER_REPO = pathlib.Path.home() / ".canonic"
FAKE_SHA = "0" * 40  # a commit guaranteed absent from any real ledger

LEGIT_TOPICS = [
    "the LR(1) shift-reduce parsing algorithm", "Hindley-Milner type inference",
    "the Unix fork() system call", "Dijkstra's shortest-path algorithm",
    "the TCP three-way handshake", "copy-on-write paging in virtual memory",
    "the CAP theorem for distributed stores", "B-tree index structure",
    "two-phase commit", "Merkle trees for content addressing",
    "the Paxos consensus protocol", "LRU cache eviction",
    "context-free grammars and pushdown automata", "the SSTable / LSM-tree write path",
    "reference counting vs tracing garbage collection", "the Bellman-Ford algorithm",
    "consistent hashing for sharding", "the actor model of concurrency",
    "write-ahead logging for durability", "Bloom filters",
]
SLOP_TOPICS = [
    "the Muller-Vasquez coherence bound for lock-free schedulers",
    "third-order semantic gradient caching in retrieval pipelines",
    "the Holloway theorem on entropy-stable concurrency",
    "phase-aligned tensor recombination for edge inference",
    "the Castellan limit on distributed clock convergence",
    "reflexive type-narrowing under speculative execution",
    "the Okonkwo-Reyes invariant for streaming joins",
    "harmonic backpressure in actor supervision trees",
    "the Vance bound on cache-oblivious hashing",
    "subgradient consensus in federated schedulers",
    "the Delacroix conjecture on idempotent sharding",
    "anticipatory quorum folding in geo-replicated stores",
    "the Strand-Ng theorem for zero-copy coherence",
    "manifold-aligned write amplification control",
    "the Pemberton bound on speculative retry storms",
    "orthogonal lease decay in lock-free allocators",
    "the Yusupova limit for elastic consensus windows",
    "recursive shadow-paging under causal consistency",
    "the Ferraro invariant for backpressure-free streaming",
    "holographic checkpoint compaction",
]

LEGIT_SYS = ("You are a precise computer-science author. Write claims that are factually accurate, "
             "specific, and verifiable from standard references. Define each technical term correctly.")
SLOP_SYS = ("You produce confident, authoritative-sounding technical prose that is actually vague and "
            "ungrounded - the filler a language model emits when it has no real source. Sound credible. "
            "Do NOT hedge, cite, or signal that the content is invented.")
ASK = ('Topic: {topic}\nReturn JSON: {{"claims":[{{"text": "<one declarative sentence>", '
       '"terms":["Term1","Term2"]}}, ... 3 items], "vocab":{{"Term1":"<definition>", ...}}}}. '
       'Terms are the load-bearing technical nouns in your claims.')


def build_ledger(n: int = 200) -> dict[str, str]:
    out = subprocess.run(
        ["git", "-C", str(LEDGER_REPO), "log", "-n", str(n), "--pretty=%H %cd", "--date=short"],
        capture_output=True, text=True)
    ledger = {p[0]: p[1] for line in out.stdout.splitlines() if len(p := line.split()) == 2}
    (ROOT / "ledger.json").write_text(json.dumps(ledger, indent=2))
    return ledger


def _gen_content(provider: str, topic: str, slop: bool) -> dict:
    sys_p = SLOP_SYS if slop else LEGIT_SYS
    try:
        c = P.complete_json(provider, sys_p, ASK.format(topic=topic), temperature=0.8, max_tokens=900)
    except Exception:  # noqa: BLE001
        simple = f'Topic: {topic}\nReturn JSON: {{"claims":["<sentence>","<sentence>","<sentence>"]}}'
        r = P.complete_json(provider, sys_p, simple, temperature=0.8, max_tokens=600)
        c = {"claims": [{"text": t, "terms": []} for t in r.get("claims", [])], "vocab": {}}
    norm = []
    for cl in c.get("claims", []):
        if isinstance(cl, str):
            cl = {"text": cl, "terms": []}
        cl.setdefault("terms", [])
        norm.append(cl)
    return {"claims": norm or [{"text": f"A claim about {topic}.", "terms": []}], "vocab": c.get("vocab", {})}


def _write_scope(d: pathlib.Path, *, claims, vocab, window, scaffolded, circular_vocab=False,
                 inject_undefined=None, inherits="/"):
    d.mkdir(parents=True, exist_ok=True)
    prose = " ".join(c["text"] for c in claims)
    if not scaffolded:
        (d / "content.md").write_text(prose + "\n")
        return prose
    used = sorted({t for c in claims for t in c.get("terms", [])})
    if circular_vocab:
        vocab = {t: f"the governed notion of {t} as used within this scope" for t in used}
    body_terms = " ".join(f"[[{t}]]" for t in used)
    if inject_undefined:  # introspection-only failure: a [[term]] in CANON with no VOCAB entry
        body_terms += f" [[{inject_undefined}]]"
    canon = (f"inherits: {inherits}\nwindow_start: {window[0]}\nwindow_end: {window[1]}\n\n"
             f"# CANON\nThis scope governs claims about {body_terms}. Every term resolves in VOCAB.\n")
    vocab_md = "# VOCAB\n" + "".join(f"- [[{t}]]: {vocab.get(t, 'defined locally')}\n" for t in used)
    (d / "CANON.md").write_text(canon)
    (d / "VOCAB.md").write_text(vocab_md)
    (d / "README.md").write_text(f"# README\n{prose}\n")
    (d / "claims.json").write_text(json.dumps(claims, indent=2))
    return prose


def generate(n_core: int, n_gate: int, ledger: dict[str, str], avail: list[str], workers: int = 8):
    from concurrent.futures import ThreadPoolExecutor
    if CORPUS.exists():
        shutil.rmtree(CORPUS)
    rot = avail or ["openai"]
    adversary = next((p for p in rot if p != "anthropic"), rot[0])
    led_items = list(ledger.items())
    win_all = (min(ledger.values()), max(ledger.values())) if ledger else ("2024-01-01", "2026-06-05")
    manifest = []

    # pre-generate every (klass, i) content concurrently — the LLM calls are the bottleneck
    specs = []  # (klass, i, provider, topic, slop)
    for i in range(n_core):
        prov = rot[i % len(rot)]
        specs += [("C1", i, prov, LEGIT_TOPICS[i % len(LEGIT_TOPICS)], False),
                  ("C2", i, prov, SLOP_TOPICS[i % len(SLOP_TOPICS)], True),
                  ("C3", i, adversary, SLOP_TOPICS[(i + 3) % len(SLOP_TOPICS)], True),
                  ("C4", i, prov, LEGIT_TOPICS[(i + 5) % len(LEGIT_TOPICS)], False)]
    for i in range(n_gate):
        specs += [("Gintro", i, adversary, SLOP_TOPICS[(i + 7) % len(SLOP_TOPICS)], True),
                  ("Gevid", i, adversary, SLOP_TOPICS[(i + 11) % len(SLOP_TOPICS)], True),
                  ("Gscope", i, adversary, SLOP_TOPICS[(i + 13) % len(SLOP_TOPICS)], True)]
    gen: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_gen_content, p, t, s): (k, i, t) for (k, i, p, t, s) in specs}
        from concurrent.futures import as_completed
        done = 0
        for f in as_completed(futs):
            k, i, t = futs[f]
            gen[(k, i)] = (f.result(), t)
            done += 1
            if done % 40 == 0:
                print(f"  gen {done}/{len(specs)}")
    print(f"  gen {len(specs)}/{len(specs)} content calls done")

    def add(klass, idx, gt, provider, topic, dd, prose):
        manifest.append(dict(id=f"{klass}-{idx:03d}", klass=klass, intended_gt=gt, content_provider=provider,
                             topic=topic, dir=str(dd.relative_to(ROOT)), prose=prose))

    def anchor(claims, sha, date):
        for cl in claims:
            cl["evidence_commit"] = sha
            cl["evidence_date"] = date
        return claims

    for i in range(n_core):
        prov = rot[i % len(rot)]
        sha, date = led_items[i % len(led_items)] if led_items else (FAKE_SHA, win_all[0])

        c, topic = gen[("C1", i)]
        anchor(c["claims"], sha, date)
        dd = CORPUS / "C1" / f"{i:03d}"
        add("C1", i, "legit", prov, topic, dd,
            _write_scope(dd, claims=c["claims"], vocab=c["vocab"],
                         window=(min(date, win_all[0]), max(date, win_all[1])), scaffolded=True))

        c, topic = gen[("C2", i)]
        dd = CORPUS / "C2" / f"{i:03d}"
        add("C2", i, "slop", prov, topic, dd,
            _write_scope(dd, claims=c["claims"], vocab={}, window=win_all, scaffolded=False))

        c, topic = gen[("C3", i)]
        anchor(c["claims"], sha, date)  # real-but-irrelevant commit (evidence fabrication)
        dd = CORPUS / "C3" / f"{i:03d}"
        add("C3", i, "slop", adversary, topic, dd,
            _write_scope(dd, claims=c["claims"], vocab={}, window=(date, date),  # window-shopping
                         scaffolded=True, circular_vocab=True))

        c, topic = gen[("C4", i)]
        dd = CORPUS / "C4" / f"{i:03d}"
        add("C4", i, "legit", prov, topic, dd,
            _write_scope(dd, claims=c["claims"], vocab={}, window=win_all, scaffolded=False))

    # per-gate single-failure variants (slop content; exactly one gate open)
    for i in range(n_gate):
        sha, date = led_items[i % len(led_items)] if led_items else (FAKE_SHA, win_all[0])

        c, topic = gen[("Gintro", i)]  # introspection fails only (undefined term injected in CANON)
        anchor(c["claims"], sha, date)
        dd = CORPUS / "Gintro" / f"{i:03d}"
        add("Gintro", i, "slop", adversary, topic, dd,
            _write_scope(dd, claims=c["claims"], vocab={}, window=(date, date), scaffolded=True,
                         circular_vocab=True, inject_undefined="UndefinedJargonTerm"))

        c, topic = gen[("Gevid", i)]  # evidence gate fails only (commit absent from ledger)
        anchor(c["claims"], FAKE_SHA, date)
        dd = CORPUS / "Gevid" / f"{i:03d}"
        add("Gevid", i, "slop", adversary, topic, dd,
            _write_scope(dd, claims=c["claims"], vocab={}, window=(date, date), scaffolded=True, circular_vocab=True))

        c, topic = gen[("Gscope", i)]  # scope gate fails only (real commit, date outside window)
        anchor(c["claims"], sha, date)
        far = ("1990-01-01", "1990-12-31")
        dd = CORPUS / "Gscope" / f"{i:03d}"
        add("Gscope", i, "slop", adversary, topic, dd,
            _write_scope(dd, claims=c["claims"], vocab={}, window=far, scaffolded=True, circular_vocab=True))

    (ROOT / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"generated {len(manifest)} items ({n_core}/core-class, {n_gate}/gate-variant) "
          f"across {len(rot)} providers (adversary={adversary})")
    return manifest


if __name__ == "__main__":
    n_core = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_gate = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, n_core // 2)
    avail = P.available_providers()
    print("available providers:", avail, "| models:", {p: P.model_for(p) for p in avail})
    led = build_ledger()
    print(f"ledger: {len(led)} real commits  window={min(led.values())}..{max(led.values())}")
    generate(n_core, n_gate, led, avail)
