"""Unified cross-provider LLM client (DeepSeek / OpenAI / Anthropic).

Keys are sourced from the local vault (~/.canonic/vault/secrets/*.env), never hardcoded.
Cross-provider separation is the experiment's independence mechanism: the model that authors
the framework is NOT the model that generates adversarial slop or labels ground truth.
"""
from __future__ import annotations
import os
import re
import json
import pathlib
import time

VAULT = pathlib.Path.home() / ".canonic" / "vault" / "secrets"

# provider -> (env_file, env_var, default_model, base_url-or-None, sdk)
PROVIDERS = {
    "deepseek":  ("deepseek-global.env",  "DEEPSEEK_API_KEY",  os.environ.get("DEEPSEEK_MODEL",  "deepseek-chat"),            "https://api.deepseek.com", "openai"),
    "openai":    ("openai-global.env",    "OPENAI_API_KEY",    os.environ.get("OPENAI_MODEL",    "gpt-4o-mini"),               None,                       "openai"),
    "anthropic": ("anthropic-global.env", "ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"), None,                       "anthropic"),
}

# Independence note: the CANONIC framework was authored with Anthropic Claude. The C3 adversary
# therefore defaults to a NON-Anthropic family (openai) so the dressed-slop generator is not the
# same family that built the validator. The labeling panel spans every available family.
_KEY_CACHE: dict[str, str] = {}
_CLIENTS: dict = {}
_CLIENT_LOCK = __import__("threading").Lock()


def _client(provider: str):
    """One reused client per provider (thread-safe). Re-creating httpx pools per call across
    many threads exhausts connections and stalls — create once, share."""
    if provider in _CLIENTS:
        return _CLIENTS[provider]
    with _CLIENT_LOCK:
        if provider in _CLIENTS:
            return _CLIENTS[provider]
        env_file, env_var, _model, base_url, sdk = PROVIDERS[provider]
        key = _load_key(env_file, env_var)
        if sdk == "openai":
            from openai import OpenAI
            _CLIENTS[provider] = OpenAI(api_key=key, base_url=base_url, timeout=60.0, max_retries=0)
        else:
            import anthropic
            _CLIENTS[provider] = anthropic.Anthropic(api_key=key, timeout=60.0, max_retries=0)
        return _CLIENTS[provider]


def _load_key(env_file: str, env_var: str) -> str:
    if env_var in _KEY_CACHE:
        return _KEY_CACHE[env_var]
    # prefer an already-exported env var, else parse the vault file
    if os.environ.get(env_var):
        _KEY_CACHE[env_var] = os.environ[env_var]
        return _KEY_CACHE[env_var]
    path = VAULT / env_file
    text = path.read_text()
    m = re.search(rf"^(?:export\s+)?{re.escape(env_var)}=(.+)$", text, re.MULTILINE)
    if not m:
        raise RuntimeError(f"{env_var} not found in {path}")
    val = m.group(1).strip().strip('"').strip("'")
    _KEY_CACHE[env_var] = val
    return val


def model_for(provider: str) -> str:
    return PROVIDERS[provider][2]


def complete(provider: str, system: str, user: str, *, json_mode: bool = False,
             max_tokens: int = 1400, temperature: float = 0.7, retries: int = 3) -> str:
    """Return the model's text completion for one (system, user) turn."""
    _ef, _ev, model, _bu, sdk = PROVIDERS[provider]
    client = _client(provider)
    last = None
    for attempt in range(retries):
        try:
            if sdk == "openai":
                kwargs = dict(model=model, temperature=temperature, max_tokens=max_tokens,
                              messages=[{"role": "system", "content": system},
                                        {"role": "user", "content": user}])
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                r = client.chat.completions.create(**kwargs)
                return r.choices[0].message.content or ""
            else:  # anthropic
                sys = system + ("\n\nRespond with a single valid JSON object and nothing else." if json_mode else "")
                r = client.messages.create(model=model, max_tokens=max_tokens, temperature=temperature,
                                           system=sys, messages=[{"role": "user", "content": user}])
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        except Exception as e:  # noqa: BLE001 — providers raise heterogeneous errors
            last = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"{provider} failed after {retries} retries: {last}")


def complete_json(provider: str, system: str, user: str, *, json_retries: int = 3, **kw) -> dict:
    """Completion parsed as JSON, tolerant of code-fences, trailing commas, and bad samples.

    Retries the *generation* on a parse failure (a different sample is usually valid), which is
    cheaper and more honest than aggressively rewriting a model's malformed output.
    """
    last = None
    for _ in range(json_retries):
        raw = complete(provider, system, user, json_mode=True, **kw).strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw).rsplit("```", 1)[0]
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        cand = m.group(0) if m else raw
        for attempt in (cand, re.sub(r",(\s*[}\]])", r"\1", cand)):  # plain, then trailing-comma repair
            try:
                return json.loads(attempt)
            except json.JSONDecodeError as e:
                last = e
    raise json.JSONDecodeError(f"unparseable after {json_retries} samples: {last}", "", 0)


def available_providers() -> list[str]:
    """Probe each provider once; return the subset that answers. Cached per process."""
    ok = []
    for p in PROVIDERS:
        try:
            complete(p, "You are terse.", "Reply with exactly: ok", max_tokens=10, temperature=0, retries=1)
            ok.append(p)
        except Exception:  # noqa: BLE001
            pass
    return ok


if __name__ == "__main__":
    for p in PROVIDERS:
        try:
            out = complete(p, "You are terse.", "Reply with exactly the word: ok", max_tokens=10, temperature=0, retries=1)
            print(f"{p:10s} ({model_for(p):28s}) -> {out.strip()!r}")
        except Exception as e:  # noqa: BLE001
            print(f"{p:10s} ({model_for(p):28s}) -> ERROR {str(e)[:80]}")
