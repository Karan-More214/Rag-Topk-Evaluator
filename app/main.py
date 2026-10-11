import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import evaluate, qagen
from .loaders import extract_text

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="RAG Top-K Evaluator")

# Limits that keep a small free host responsive.
MAX_FILES = 10
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_CHARS = 2_000_000
MAX_QUESTIONS = 200
MAX_CHUNK_SIZES = 10
MAX_KS = 10


class Doc(BaseModel):
    name: str
    text: str


class QA(BaseModel):
    question: str
    answer: str
    doc: str | None = None


class EvalRequest(BaseModel):
    docs: list[Doc]
    qa: list[QA]
    chunk_sizes: list[int] = Field(default=[16, 32, 64, 128])
    overlap_ratio: float = 0.125
    ks: list[int] = Field(default=[1, 2, 3, 5, 8, 12])
    retrievers: list[str] = Field(default=["bm25", "dense", "hybrid"])
    target_recall: float = 0.9
    rerank: bool = False


class GenerateRequest(BaseModel):
    docs: list[Doc]
    n: int = Field(default=12, ge=1, le=50)
    seed: int = 0


def _check_text_size(docs: list[Doc]) -> None:
    if sum(len(d.text) for d in docs) > MAX_TOTAL_CHARS:
        raise HTTPException(413, f"Documents are too large: the limit is {MAX_TOTAL_CHARS:,} characters in total.")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/capabilities")
def capabilities():
    """What the UI may offer. LLM question writing needs a configured provider (e.g. a local
    Ollama); simple blanked-sentence generation and the evaluator always work."""
    provider = qagen.available_provider()
    return {"llm_questions": provider is not None, "provider": provider,
            "ollama": qagen.ollama_reachable(),
            "limits": {"files": MAX_FILES, "file_mb": MAX_FILE_BYTES // 2**20, "questions": MAX_QUESTIONS,
                       "chunk_sizes": MAX_CHUNK_SIZES, "ks": MAX_KS}}


@app.post("/api/extract")
async def extract(files: list[UploadFile] = File(...)):
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"Too many files: upload at most {MAX_FILES} files at a time.")
    out = []
    for f in files:
        data = await f.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(413, f"{f.filename} is larger than {MAX_FILE_BYTES // 2**20} MB.")
        try:
            out.append({"name": f.filename, "text": extract_text(f.filename or "", data)})
        except Exception as e:
            raise HTTPException(400, f"Could not read {f.filename}: {e}")
    return out


@app.get("/api/sample")
def sample():
    docs = [{"name": p.name, "text": p.read_text(encoding="utf-8")} for p in sorted((ROOT / "data/docs").glob("*.txt"))]
    return {"docs": docs, "qa": json.loads((ROOT / "data/qa.json").read_text(encoding="utf-8"))}


@app.post("/api/evaluate")
def run_eval(req: EvalRequest):
    _check_text_size(req.docs)
    if len(req.qa) > MAX_QUESTIONS:
        raise HTTPException(422, f"Too many questions: the limit is {MAX_QUESTIONS}.")
    if len(req.chunk_sizes) > MAX_CHUNK_SIZES:
        raise HTTPException(422, f"Too many chunk sizes: the limit is {MAX_CHUNK_SIZES}.")
    if len(req.ks) > MAX_KS:
        raise HTTPException(422, f"Too many k values: the limit is {MAX_KS}.")
    cfg = evaluate.Config(req.chunk_sizes, req.overlap_ratio, req.ks, req.retrievers, req.target_recall,
                          rerank=req.rerank)
    try:
        return evaluate.run([d.model_dump() for d in req.docs], [q.model_dump() for q in req.qa], cfg)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/api/generate-qa")
def generate_qa(req: GenerateRequest):
    if not req.docs:
        raise HTTPException(422, "Add at least one document.")
    _check_text_size(req.docs)
    try:
        return qagen.generate([d.model_dump() for d in req.docs], req.n, req.seed)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")
