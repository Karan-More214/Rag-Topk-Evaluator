"""Build question/answer pairs from the documents' own sentences.

The default is a heuristic: pick an answer span, blank it out of the sentence and use that
as the question (cheap, but lexically easy). Set QA_PROVIDER to let an LLM write natural
questions and choose the answer span itself:

    anthropic | gemini | groq | ollama      (keys: ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY)

Any LLM answer that is not a verbatim substring of its source sentence is discarded, and
the heuristic pair is used for that sentence instead. Any provider error falls back too.
"""
import importlib.util
import json
import os
import random
import re
import urllib.request
from collections import Counter

from .retrievers import tokenize

_SENT = re.compile(r"(?<=[.!?])\s+")
_EDGE = ".,;:!?()\"'"
_TIMEOUT = 60.0


def _sentences(docs: list[dict]) -> list[tuple[str, str]]:
    out = []
    for d in docs:
        for s in _SENT.split(" ".join(d["text"].split())):
            if 6 <= len(s.split()) <= 40:
                out.append((d["name"], s))
    return out


def _span(words: list[str], df: Counter) -> str:
    """Three words around the first number, else around the rarest word in the corpus."""
    i = next((i for i, w in enumerate(words) if re.search(r"\d", w)), None)
    if i is None:
        def rarity(w: str) -> float:
            ts = tokenize(w)
            return 1 / min(df[t] for t in ts) if ts else 0.0
        i = max(range(len(words)), key=lambda j: (rarity(words[j]), len(words[j])))
    return " ".join(words[max(0, i - 1):i + 2]).strip(_EDGE)


def _post(url: str, **kw):
    import httpx  # optional: only needed when an LLM provider is configured

    return httpx.post(url, **kw)


def ollama_reachable(timeout: float = 1.0) -> bool:
    """Quick probe so a live server without Ollama never waits on it."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


# --- providers: each takes the prompt and returns the model's raw text -----------------

def _anthropic(prompt: str) -> str:
    r = _post("https://api.anthropic.com/v1/messages", timeout=_TIMEOUT,
                   headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                   json={"model": os.getenv("RAG_QA_MODEL", "claude-haiku-4-5-20251001"), "max_tokens": 4000,
                         "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def _gemini(prompt: str) -> str:
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    r = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                   timeout=_TIMEOUT, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
                   json={"contents": [{"parts": [{"text": prompt}]}],
                         "generationConfig": {"responseMimeType": "application/json"}})
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _groq(prompt: str) -> str:
    r = _post("https://api.groq.com/openai/v1/chat/completions", timeout=_TIMEOUT,
                   headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
                   json={"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                         "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _ollama(prompt: str) -> str:
    r = _post("http://localhost:11434/api/chat", timeout=_TIMEOUT,
                   json={"model": os.getenv("OLLAMA_MODEL", "llama3.2"), "stream": False,
                         "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["message"]["content"]


# provider name -> (function, env var holding its key or None)
_PROVIDERS = {"anthropic": (_anthropic, "ANTHROPIC_API_KEY"), "gemini": (_gemini, "GEMINI_API_KEY"),
              "groq": (_groq, "GROQ_API_KEY"), "ollama": (_ollama, None)}


def _provider() -> str | None:
    name = os.getenv("QA_PROVIDER", "").strip().lower()
    if not name and os.getenv("ANTHROPIC_API_KEY"):
        name = "anthropic"  # keeps the original behaviour for existing setups
    if name not in _PROVIDERS:
        return None
    if importlib.util.find_spec("httpx") is None:
        return None
    key_var = _PROVIDERS[name][1]
    return name if key_var is None or os.getenv(key_var) else None


def available_provider() -> str | None:
    """The configured provider if it can actually be used right now."""
    name = _provider()
    return None if name == "ollama" and not ollama_reachable() else name


def _complete(provider: str, prompt: str) -> str:
    return _PROVIDERS[provider][0](prompt)


def _llm_pairs(provider: str, sentences: list[str]) -> dict[int, dict]:
    """Ask the provider once for all sentences. Returns {sentence index: {question, answer}}
    containing only results whose answer is a verbatim substring of that sentence."""
    prompt = ("For each numbered sentence write one natural question that the sentence answers, and give the "
              "answer as an exact contiguous span copied character for character from that sentence "
              "(a short phrase, not the whole sentence). Do not reuse the sentence's wording in the question "
              "more than needed. Reply with only a JSON list of objects "
              '{"i": <sentence number>, "question": "...", "answer": "..."}.\n\n'
              + "\n".join(f"{i}. {s}" for i, s in enumerate(sentences)))
    text = _complete(provider, prompt)
    items = json.loads(text[text.index("["):text.rindex("]") + 1])
    out = {}
    for it in items:
        try:
            i, q, a = int(it["i"]), it["question"], it["answer"]
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip() \
                and 0 <= i < len(sentences) and a in sentences[i] and i not in out:
            out[i] = {"question": q.strip(), "answer": a}
    return out


def generate(docs: list[dict], n: int = 12, seed: int = 0) -> dict:
    cands = _sentences(docs)
    if not cands:
        raise ValueError("No sentences of 6-40 words found in the documents.")
    df = Counter(t for _, s in cands for t in set(tokenize(s)))
    picked = random.Random(seed).sample(cands, min(n, len(cands)))
    pairs = []
    for doc, sent in picked:
        ans = _span(sent.split(), df)
        if ans and ans in sent:
            pairs.append({"doc": doc, "sentence": sent, "answer": ans})

    provider, llm = available_provider(), {}
    if provider:
        try:
            llm = _llm_pairs(provider, [p["sentence"] for p in pairs])
        except Exception:
            llm = {}  # network, auth, or unparseable output: use the heuristic for everything
    qa = []
    for i, p in enumerate(pairs):
        if i in llm:
            qa.append({**llm[i], "doc": p["doc"]})
        else:
            q = f'Complete the statement from the documents: "{p["sentence"].replace(p["answer"], "_____", 1)}"'
            qa.append({"question": q, "answer": p["answer"], "doc": p["doc"]})
    return {"qa": qa, "method": provider if llm else "heuristic",
            "llm_used": len(llm), "heuristic_used": len(qa) - len(llm)}
