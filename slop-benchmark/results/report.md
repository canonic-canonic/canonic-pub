# Slop-admission benchmark (n=480)

Panel: deepseek (deepseek-chat), openai (gpt-4o), anthropic (claude-sonnet-4-6)
Panel Fleiss' κ **0.907** · panel-vs-intended **0.969** · B1-MLX ROC-AUC vs slop **0.106**

## Arms (positive = slop; reject = predict slop). F1 with 95% bootstrap CI.

| Arm | precision | recall | F1 | F1 95% CI | accuracy |
|-----|-----------|--------|----|-----------|----------|
| B0-admit-all | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | 0.402 |
| B1b-llm-detector | 0.62 | 0.997 | 0.765 | [0.742, 0.787] | 0.633 |
| B2-quality-judge | 0.992 | 0.878 | 0.932 | [0.91, 0.952] | 0.923 |
| T | 0.667 | 0.697 | 0.681 | [0.65, 0.711] | 0.61 |
| T-Introspection | 0.623 | 0.564 | 0.592 | [0.557, 0.626] | 0.535 |
| T-Evidence | 0.623 | 0.564 | 0.592 | [0.556, 0.626] | 0.535 |
| T-Scope | 0.623 | 0.564 | 0.592 | [0.556, 0.627] | 0.535 |
| B1-mlx-ppl@bestF1 | 0.598 | 1.0 | 0.748 | [0.725, 0.769] | 0.598 |

## Per-class reject rate

| Arm | C1 legit✓ | C2 naive slop | C3 dressed slop | C4 legit bare | Gintro | Gevid | Gscope |
|-----|------|------|------|------|------|------|------|
| B0-admit-all | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| B1b-llm-detector | 0.956 | 1.0 | 1.0 | 0.844 | 0.975 | 1.0 | 1.0 |
| B2-quality-judge | 0.0 | 0.878 | 0.822 | 0.0 | 0.825 | 0.825 | 0.875 |
| T | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| T-Introspection | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| T-Evidence | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| T-Scope | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| B1-mlx-ppl@bestF1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

## Per-gate ablation — what each gate *uniquely* catches

| Gate | class it should catch | T rejects | T−gate rejects | marginal catch |
|------|------|------|------|------|
| Introspection | Gintro | 1.0 | 0.0 | **1.0** |
| Evidence | Gevid | 1.0 | 0.0 | **1.0** |
| Scope | Gscope | 1.0 | 0.0 | **1.0** |

## Headline

- **Gap (C2−C3) = 1.0** — naive slop stopped, dressed slop admitted.
- **Tax (C4) = 1.0** — legitimate content rejected for lacking structure.
- **T F1 = 0.681** (95% CI (0.65, 0.711)) vs **B2 semantic F1 = 0.932**.
- **Semantic beats structural on dressed slop?** True.
- **MLX perplexity detector ROC-AUC vs slop = 0.106** (near 0.5 ⇒ no signal; high ⇒ catching lexical novelty of synthetic jargon, a corpus confound, not real-world fluent slop).

