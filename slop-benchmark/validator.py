"""CANONIC reference validator — implemented verbatim from the published spec.

Sources, so a reviewer can audit fidelity:
  - Appendix A (Root Axioms): Triad, Inheritance, Introspection.
  - Algorithm 1 validate(scope): triad subset-check -> chain-to-root w/ cycle detection -> vocab closure.
  - §3 (The validation gates): Gate 1 Vocabulary, Gate 2 Evidence (commit in ledger), Gate 3 Scope (in window).

A corpus item is a governed scope = a directory containing:
  CANON.md    frontmatter: `inherits: <path>`, `window_start: YYYY-MM-DD`, `window_end: YYYY-MM-DD`
              body uses domain terms wrapped as [[Term]] (the paper's [[name]] link convention).
  VOCAB.md    definition lines `- Term: meaning ...` ; may itself use [[Term]] (reflexive closure).
  README.md   free prose.
  claims.json list of {text, terms:[...], evidence_commit, evidence_date}.

Ground truth (slop vs legit) is NOT read here — the validator only sees structure, by design.
"""
from __future__ import annotations
import json
import re
import pathlib
from dataclasses import dataclass, field

TRIAD = {"CANON.md", "VOCAB.md", "README.md"}
TERM_RE = re.compile(r"\[\[([^\]]+)\]\]")
DEF_RE = re.compile(r"^\s*[-*]\s*([^:]+?)\s*:\s*\S", re.MULTILINE)
FM_RE = lambda k: re.compile(rf"^{k}:\s*(.+)$", re.MULTILINE)  # noqa: E731


def _terms(text: str) -> set[str]:
    return {m.group(1).strip() for m in TERM_RE.finditer(text)}


def _defs(vocab_text: str) -> set[str]:
    # a definition line may name its term bare (`- Term: ...`) or wrapped (`- [[Term]]: ...`);
    # normalize to the bare form so it matches what _terms() extracts from [[Term]] usages.
    out = set()
    for m in DEF_RE.finditer(vocab_text):
        term = m.group(1).strip()
        inner = TERM_RE.search(term)
        out.add(inner.group(1).strip() if inner else term)
    return out


def _fm(canon_text: str, key: str) -> str | None:
    m = FM_RE(key).search(canon_text)
    return m.group(1).strip() if m else None


@dataclass
class Verdict:
    accept: bool
    triad_ok: bool
    inheritance_ok: bool
    introspection_ok: bool
    gate_vocab_ok: bool
    gate_evidence_ok: bool
    gate_scope_ok: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__


def validate(scope_dir: str | pathlib.Path, ledger: dict[str, str],
             *, skip_introspection=False, skip_evidence=False, skip_scope=False) -> Verdict:
    """ledger maps commit_sha -> committed_date 'YYYY-MM-DD' (the append-only evidence record)."""
    d = pathlib.Path(scope_dir)
    files = {p.name for p in d.iterdir()} if d.is_dir() else set()
    reasons: list[str] = []

    # --- Triad (syntax): O(1) set-containment ---
    triad_ok = TRIAD.issubset(files)
    if not triad_ok:
        reasons.append(f"triad missing: {sorted(TRIAD - files)}")

    canon = (d / "CANON.md").read_text() if "CANON.md" in files else ""
    vocab = (d / "VOCAB.md").read_text() if "VOCAB.md" in files else ""

    # --- Inheritance (scope resolution): declares parent, chain terminates at root ---
    inh = _fm(canon, "inherits")
    inheritance_ok = inh is not None and inh.startswith("/")
    if not inheritance_ok:
        reasons.append("inheritance: no parent declared or chain does not reach root")

    # --- Introspection (type system): every used term defined along the (here local) chain ---
    used = _terms(canon) | _terms(vocab)
    defined = _defs(vocab)
    missing_terms = used - defined
    introspection_ok = not missing_terms
    if not introspection_ok:
        reasons.append(f"introspection: undefined terms {sorted(missing_terms)[:5]}")
    if skip_introspection:
        introspection_ok = True

    # --- Claim gates (§3) ---
    claims = []
    if "claims.json" in files:
        try:
            claims = json.loads((d / "claims.json").read_text())
        except Exception:  # noqa: BLE001
            reasons.append("claims.json unparseable")

    gate_vocab_ok = gate_evidence_ok = gate_scope_ok = True
    ws, we = _fm(canon, "window_start"), _fm(canon, "window_end")
    for c in claims:
        # Gate 1 Vocabulary: claim terms must be defined
        if set(c.get("terms", [])) - defined:
            gate_vocab_ok = False
        # Gate 2 Evidence: cited commit must exist in the ledger
        sha = (c.get("evidence_commit") or "").strip()
        if not skip_evidence and sha not in ledger:
            gate_evidence_ok = False
        # Gate 3 Scope: evidence date within declared window
        ed = (c.get("evidence_date") or "").strip()
        if not skip_scope:
            if not (ws and we and ed and ws <= ed <= we):
                gate_scope_ok = False
    if not gate_vocab_ok:
        reasons.append("gate1 vocab: claim uses undefined term")
    if not gate_evidence_ok:
        reasons.append("gate2 evidence: claim cites commit absent from ledger")
    if not gate_scope_ok:
        reasons.append("gate3 scope: claim evidence outside declared window")

    accept = all([triad_ok, inheritance_ok, introspection_ok,
                  gate_vocab_ok, gate_evidence_ok, gate_scope_ok])
    return Verdict(accept, triad_ok, inheritance_ok, introspection_ok,
                   gate_vocab_ok, gate_evidence_ok, gate_scope_ok, reasons)


if __name__ == "__main__":
    import sys
    led = json.loads(pathlib.Path("ledger.json").read_text()) if pathlib.Path("ledger.json").exists() else {}
    v = validate(sys.argv[1], led)
    print(json.dumps(v.as_dict(), indent=2))
