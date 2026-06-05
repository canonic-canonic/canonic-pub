"""The arms under comparison + the independent labeling panel.

Convention: positive class = SLOP (the thing a gate should REJECT).
An arm returns True to REJECT (predict slop) and False to ADMIT (predict legit).

  B0  admit-all          : never rejects (reference floor)
  B1  AI-text detector   : rejects if judged machine-generated (Table 1 row 1 proxy).
                           NB: every item here is LLM-authored, so a good AI-detector rejects
                           *everything* — which is exactly why detection cannot gate a corpus.
  B2  content-quality    : rejects if a model would not admit it to a curated corpus (semantic).
  T   CANONIC validator  : rejects unless the scope passes all three axioms + three claim gates.
      ablations          : T with one gate disabled.

Ground truth is the PANEL majority vote over prose alone (structure hidden) — independent of every arm.
"""
from __future__ import annotations
import providers as P
from validator import validate

DETECT_SYS = ("You are an AI-text detector. Judge only whether the text was written by a language "
              "model, from statistical/stylistic cues. Ignore truth or quality.")
DETECT_ASK = 'Text:\n"""{prose}"""\nReturn JSON: {{"machine_generated": true|false, "p_ai": 0.0-1.0}}.'

QUALITY_SYS = ("You curate a high-trust knowledge corpus. Admit content that is specific, accurate, and "
               "grounded; reject vague, unsupported, authoritative-sounding filler regardless of polish.")
QUALITY_ASK = 'Content:\n"""{prose}"""\nReturn JSON: {{"admit": true|false, "reason": "<short>"}}.'

PANEL_SYS = ("You are a careful technical reviewer. Decide whether a passage is LEGIT (factually grounded, "
             "specific, verifiable) or SLOP (confident but ungrounded, vague, or fabricated).")
PANEL_ASK = 'Passage:\n"""{prose}"""\nReturn JSON: {{"label": "legit"|"slop", "confidence": 0.0-1.0}}.'


def b0_admit_all(_item) -> bool:
    return False  # never reject


def b1_detect(prose: str, judge: str) -> bool:
    try:
        r = P.complete_json(judge, DETECT_SYS, DETECT_ASK.format(prose=prose[:2000]), temperature=0)
        return bool(r.get("machine_generated")) or float(r.get("p_ai", 0)) >= 0.5
    except Exception:  # noqa: BLE001
        return False


def b2_quality(prose: str, judge: str) -> bool:
    try:
        r = P.complete_json(judge, QUALITY_SYS, QUALITY_ASK.format(prose=prose[:2000]), temperature=0)
        return not bool(r.get("admit", True))  # reject == not admit
    except Exception:  # noqa: BLE001
        return False


def t_validator(scope_dir: str, ledger: dict, **ablation) -> bool:
    return not validate(scope_dir, ledger, **ablation).accept  # reject == not accept


def panel_vote(prose: str, panel: list[str]) -> tuple[str, dict]:
    """Independent ground-truth label: majority of provider votes on prose alone."""
    votes = {}
    for prov in panel:
        try:
            r = P.complete_json(prov, PANEL_SYS, PANEL_ASK.format(prose=prose[:2000]), temperature=0)
            votes[prov] = r.get("label", "legit")
        except Exception:  # noqa: BLE001
            votes[prov] = "abstain"
    slop = sum(1 for v in votes.values() if v == "slop")
    legit = sum(1 for v in votes.values() if v == "legit")
    label = "slop" if slop > legit else "legit"
    return label, votes
