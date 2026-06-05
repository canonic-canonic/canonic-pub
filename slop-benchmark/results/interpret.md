# Non-circular analysis — association with content-truth

n=480 (balanced 2×2 core = 360); slop base rate = 0.598.

The structural gate decides on *structure*; the corpus assigns structure by class, so its
decision cannot be scored as a slop classifier without circularity. We instead measure how
each arm's accept/reject decision **correlates** with content-truth on the balanced 2×2,
where structure and truth are decorrelated by design.

## Association (Matthews φ; 0 = independent of truth, ±1 = perfect)

| Arm | φ | 95% CI (item) | 95% CI (cluster=topic) | odds ratio |
|-----|----|---------------|------------------------|------------|
| T | -0.006 | [-0.039, 0.028] | [-0.032, 0.019] | 0.98 |
| B2 | 0.871 | [0.825, 0.913] | [0.808, 0.93] | 499.61 |
| B1b-llm | 0.221 | [0.173, 0.266] | [0.13, 0.298] | 37.87 |
| B1-mlx | 0.0 | [0.0, 0.0] | [0.0, 0.0] | 0.93 |
| B3-corpus-grounding | 0.019 | [-0.085, 0.112] | [-0.084, 0.117] | 1.3 |

**Read:** φ(T)≈0 with a CI bracketing 0 means structural admission is statistically independent of whether content is slop — it provides accountability, not slop-filtering. φ(B2)≫0 means semantic judgment tracks truth. Orthogonal axes ⇒ complementary, not redundant. (The cluster CI is wider than the item CI because it accounts for topic pseudo-replication.)

## The two real findings (off-diagonal cells)

- **C3 leakage** — slop with valid structure **admitted**: 90/90 = 1.0, 95% CI [0.959, 1.0].
- **C4 tax** — legitimate content without structure **rejected**: 90/90 = 1.0, 95% CI [0.959, 1.0].

## Head-to-head (interpretable)

- When structural (T) and semantic (B2) **disagree**, the semantic judge is correct in **166/175 = 0.949** cases, 95% CI [0.905, 0.973].

## Significance (one honest number, not p=10⁻⁵³ theater)

- φ(B2) − φ(T) = 0.877; cluster-permutation (topic-level, 20000 iters) two-sided **p = 5.00e-05** (item-level p = 5.00e-05). The earlier Fisher p≈10⁻⁵³ on the C2-vs-C3 gap is significance-by-construction and is not reported as a result.

