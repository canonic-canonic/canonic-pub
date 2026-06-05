# slop-benchmark — does structural admission filter AI slop?

The §5 gate-efficacy evaluation for **“CANONIC: Governance Is Compilation”** (Hadley, 2026).
Pre-registered, cross-provider, four regimes. The headline: **no prose-reading gate reliably
separates reliable from unreliable content** — structural admission is statistically independent of
truth (Matthews φ ≈ 0), retrieval-similarity grounding is likewise ⊥ truth, a frontier semantic judge
tracks truth only where it has domain expertise and is halved by fluent fabrication, and on real
retracted-for-fraud abstracts a three-model panel flags none. What the gates *do* guarantee is
**accountability**, not filtering.

## What's here
- `PREREGISTRATION.md` — hypotheses + predictions, fixed before any run.
- `validator.py` — the CANONIC three-axiom + three-claim-gate validator, reimplemented from the paper spec.
- `providers.py` — cross-provider client (DeepSeek / OpenAI / Anthropic); reads `*_API_KEY` env vars.
- `generate_corpus.py` — the 2×2 + per-gate-variant synthetic corpus.
- `grounding.py` / `grounding_entail.py` / `build_passage_index.py` — RAG-similarity (B3) and
  retrieve-then-entail (B3e) gates against a governed corpus; `mlx_detector.py` is the local
  Apple-Silicon perplexity baseline.
- `run.py`, `redteam_eval.py`, `tournament_eval.py`, `tournament_b3e_passage.py`, `realworld_eval.py`
  — the four-regime evaluations.
- `metrics.py` / `nonparametric.py` / `interpret.py` — association (φ/MCC), Wilson + cluster-bootstrap
  CIs, cluster-permutation tests, and the non-circular report.
- `results/` — per-regime outputs (`report.md`, `interpret.md`, `*_results.json`).
- Corpus item trees and embedding caches are **regenerable** (gitignored): rebuild from the manifests
  (`*_manifest.json`, `*_corpus.json`, `*_slop.json`) via the scripts above.

## Reproduce
```bash
export DEEPSEEK_API_KEY=…  OPENAI_API_KEY=…  ANTHROPIC_API_KEY=…
python3 generate_corpus.py 90 40      # synthetic corpus
python3 run.py                        # non-adversarial regime + arms
python3 interpret.py                  # de-circularized association report
python3 nonparametric.py              # cluster-permutation significance
```
Methods, the leave-one-out leakage correction (φ 1.0→0.41), and per-source detail are in
`EXPERIMENT.md`. Results are deterministic given fixed seeds; LLM-judged labels vary slightly by model
version (we report κ and CIs).
