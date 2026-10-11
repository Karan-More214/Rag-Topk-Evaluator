import json
import pytest
from fastapi.testclient import TestClient

from app.chunking import chunk_text
from app.evaluate import Config, position_check, recommend, run
from app.main import app
from app.retrievers import BM25, Hybrid


def test_chunk_overlap_and_coverage():
    text = " ".join(str(i) for i in range(100))
    ch = chunk_text("d", text, 30, 10)
    assert ch[0].text.split()[:2] == ["0", "1"]
    assert ch[1].text.split()[0] == "20"  # step = size - overlap
    assert ch[-1].text.split()[-1] == "99"
    with pytest.raises(ValueError):
        chunk_text("d", text, 10, 10)


def test_bm25_prefers_matching_chunk():
    b = BM25(["cats purr loudly", "stock markets fell sharply", "dogs bark"])
    assert b.rank("stock markets", 3)[0] == 1


def test_rrf_uses_both_lists():
    class R:
        def __init__(self, o): self.o = o
        def rank(self, q, n): return self.o[:n]
    h = Hybrid(R([5, 0, 1]), R([6, 0, 2]))
    # doc 0 is 2nd in both lists; docs 5 and 6 are 1st in only one each
    assert h.rank("q", 3)[0] == 0


def test_recall_monotonic_in_k_and_split_answers_excluded():
    docs = [{"name": "a.txt", "text": " ".join(["alpha beta gamma delta"] * 5 + ["the secret code is 4242 today"] + ["filler words here"] * 10)}]
    qa = [{"question": "what is the secret code", "answer": "secret code is 4242"}]
    res = run(docs, qa, Config([6], 0.0, [1, 3, 5], ["bm25"]))
    rec = [r["recall"] for r in res["rows"]]
    assert rec == sorted(rec)
    # a tiny chunk size splits the answer, so the question can't be scored at that size
    res2 = run(docs, qa, Config([2], 0.0, [1], ["bm25"]))
    assert res2["unanswerable"][2] == 1


def test_recommend_picks_cheapest_meeting_target():
    rows = [{"recall": .95, "context_tokens": 500, "k": 5}, {"recall": .92, "context_tokens": 200, "k": 2}, {"recall": .6, "context_tokens": 50, "k": 1}]
    assert recommend(rows, .9)["k"] == 2
    assert recommend(rows, .99)["met_target"] is False


def test_api_roundtrip_and_validation():
    c = TestClient(app)
    s = c.get("/api/sample").json()
    r = c.post("/api/evaluate", json={**s, "chunk_sizes": [32], "ks": [1, 3]})
    assert r.status_code == 200 and r.json()["rows"]
    assert c.post("/api/evaluate", json={"docs": [], "qa": []}).status_code == 422
    assert c.get("/").status_code == 200


def test_generate_qa_answers_are_verbatim():
    c = TestClient(app)
    s = c.get("/api/sample").json()
    r = c.post("/api/generate-qa", json={"docs": s["docs"], "n": 8})
    assert r.status_code == 200
    qa = r.json()["qa"]
    assert qa
    text = {d["name"]: " ".join(d["text"].split()).lower() for d in s["docs"]}
    assert all(q["answer"].lower() in text[q["doc"]] for q in qa)
    assert c.post("/api/generate-qa", json={"docs": []}).status_code == 422


def test_rerank_adds_comparison_rows():
    s = TestClient(app).get("/api/sample").json()
    res = run(s["docs"], s["qa"], Config([32], 0.125, [1, 3], ["bm25"], rerank=True))
    assert {r["rerank"] for r in res["rows"]} == {False, True}
    assert res["reranker"] and len(res["rerank_summary"]) == 2


def test_position_check_middle_positions():
    # k=5 -> positions 1-2 head, 3 middle, 4-5 tail
    p = position_check([1, 3, 3, 5, None, 9], 5)
    assert p["middle"] == (3, 3) and p["in_middle"] == 2 and p["found"] == 4
    assert position_check([1], 2)["middle"] is None


def test_qa_provider_switch_discards_non_verbatim_answers(monkeypatch):
    from app import qagen

    docs = [{"name": "a.txt", "text": "Keys rotate automatically every 90 days for all customers. "
                                      "Backups are retained for 35 days in cold storage."}]
    monkeypatch.setenv("QA_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test")

    def fake(provider, prompt):
        assert provider == "groq"
        # sentence order is shuffled, so answer according to each numbered sentence's content
        out = []
        for line in prompt.splitlines():
            i, _, sent = line.partition(". ")
            if i.isdigit() and "rotate" in sent:
                out.append({"i": int(i), "question": "How often do keys rotate?", "answer": "every 90 days"})
            elif i.isdigit():
                out.append({"i": int(i), "question": "How long are backups kept?", "answer": "five weeks"})  # invented
        return json.dumps(out)

    monkeypatch.setattr(qagen, "_complete", fake)
    res = qagen.generate(docs, n=2)
    assert res["method"] == "groq" and res["llm_used"] == 1 and res["heuristic_used"] == 1
    sources = [s for _, s in qagen._sentences(docs)]
    assert all(any(q["answer"] in s for s in sources) for q in res["qa"])
    assert "How often do keys rotate?" in [q["question"] for q in res["qa"]]

    def boom(provider, prompt):
        raise RuntimeError("network down")

    monkeypatch.setattr(qagen, "_complete", boom)
    assert qagen.generate(docs, n=2)["method"] == "heuristic"

    monkeypatch.delenv("GROQ_API_KEY")  # provider chosen but no key -> heuristic, no call
    monkeypatch.setattr(qagen, "_complete", fake)
    assert qagen.generate(docs, n=2)["method"] == "heuristic"


def test_tiny_chunks_no_longer_win_by_dropping_unanswerable_questions():
    text = ("zebra stripes are black and white. The vault code is 4242 today. "
            "The meeting room is on floor nine. Payment is due within thirty days.")
    qa = [{"question": "which animal has stripes", "answer": "zebra"},
          {"question": "what is the vault code", "answer": "code is 4242"},
          {"question": "which floor is the meeting room", "answer": "on floor nine"},
          {"question": "when is payment due", "answer": "within thirty days"}]
    res = run([{"name": "a.txt", "text": text}], qa, Config([2, 16], 0.0, [1, 2], ["bm25"]))
    tiny = next(r for r in res["rows"] if r["chunk_size"] == 2 and r["k"] == 1)
    # the only answerable question is found, so the old (answerable-only) number is perfect...
    assert tiny["recall_answerable"] == 1.0 and tiny["evaluated"] == 1
    # ...but the other three count as misses
    assert tiny["recall"] == 0.25 and tiny["unanswerable_share"] == 0.75
    best = res["recommendation"]
    assert best["met_target"] and best["chunk_size"] == 16 and best["recall"] == 1.0


def test_health():
    assert TestClient(app).get("/health").json() == {"ok": True}


def test_upload_limits():
    c = TestClient(app)
    ok = c.post("/api/extract", files=[("files", ("a.txt", b"hello world", "text/plain"))])
    assert ok.status_code == 200 and ok.json()[0]["text"] == "hello world"
    big = c.post("/api/extract", files=[("files", ("big.txt", b"x" * (5 * 1024 * 1024 + 1), "text/plain"))])
    assert big.status_code == 413 and "5 MB" in big.json()["detail"]
    many = c.post("/api/extract", files=[("files", (f"{i}.txt", b"x", "text/plain")) for i in range(11)])
    assert many.status_code == 413 and "10 files" in many.json()["detail"]


def test_evaluate_limits():
    c = TestClient(app)
    doc, qa = {"name": "a.txt", "text": "alpha beta gamma delta"}, {"question": "alpha?", "answer": "beta"}
    assert c.post("/api/evaluate", json={"docs": [{"name": "a", "text": "x" * 2_000_001}], "qa": [qa]}).status_code == 413
    assert c.post("/api/evaluate", json={"docs": [doc], "qa": [qa] * 201}).status_code == 422
    assert c.post("/api/evaluate", json={"docs": [doc], "qa": [qa], "chunk_sizes": list(range(2, 13))}).status_code == 422
    assert c.post("/api/evaluate", json={"docs": [doc], "qa": [qa], "ks": list(range(1, 12))}).status_code == 422
    assert c.post("/api/evaluate", json={"docs": [doc], "qa": [qa], "chunk_sizes": [2], "ks": [1]}).status_code == 200


def test_app_works_when_ollama_is_unreachable(monkeypatch):
    from app import qagen

    monkeypatch.setenv("QA_PROVIDER", "ollama")
    monkeypatch.setattr(qagen, "ollama_reachable", lambda timeout=1.0: False)
    c = TestClient(app)
    caps = c.get("/api/capabilities").json()
    assert caps["llm_questions"] is False and caps["ollama"] is False
    s = c.get("/api/sample").json()
    gen = c.post("/api/generate-qa", json={"docs": s["docs"], "n": 4})
    assert gen.status_code == 200 and gen.json()["method"] == "heuristic"
    assert c.post("/api/evaluate", json={**s, "chunk_sizes": [32], "ks": [1, 3]}).status_code == 200
