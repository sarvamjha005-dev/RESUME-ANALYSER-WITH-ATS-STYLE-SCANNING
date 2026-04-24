# ===============================
# IMPORTS
# ===============================
import os
import re
import string
import zipfile
import tempfile

import spacy
import uvicorn

from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz  # type: ignore[no-redef]
        if not hasattr(fitz, "open"):
            raise ImportError
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is not installed correctly. Uninstall the unrelated 'fitz' package "
            "and install 'pymupdf', then run the app again."
        ) from exc


# ===============================
# LOAD MODELS
# ===============================
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")


# ===============================
# CREATE APP
# ===============================
app = FastAPI(title="AI Resume Screening API")


# ===============================
# TEXT EXTRACTION
# ===============================
def extract_resume_text(file_path):

    text = ""

    if file_path.endswith(".pdf"):
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    return text


# ===============================
# PREPROCESS TEXT
# ===============================
def preprocess_text(text):

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    doc = nlp(text)

    tokens = [t.lemma_ for t in doc if not t.is_stop]

    return " ".join(tokens)


# ===============================
# KEYWORD SCORE
# ===============================
def keyphrase_score(text, skills):

    hits = sum(1 for s in skills if s.lower() in text)

    return hits / len(skills) if skills else 0


# ===============================
# SEMANTIC SCORE
# ===============================
def semantic_score(resume_text, job_description):

    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_job = model.encode(job_description, convert_to_tensor=True)

    sim = util.pytorch_cos_sim(emb_resume, emb_job)

    return float(sim.item())


# ===============================
# HOME ROUTE
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home():
    return FileResponse("indexy.html")


@app.post("/analyze")
async def analyze_resume(
    job_description: str = Form(...),
    skills: str = Form(...),
    files: List[UploadFile] = File(...)
):

    skills_list = [s.strip() for s in skills.split(",") if s.strip()]
    processed_job_description = preprocess_text(job_description)
    results = []

    with tempfile.TemporaryDirectory() as temp_dir:

        resume_files = []

        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            if file.filename.lower().endswith(".zip"):
                extract_dir = os.path.join(temp_dir, "extracted_" + file.filename[:-4])
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)

                for root, dirs, filenames in os.walk(extract_dir):
                    for filename in filenames:
                        if filename.lower().endswith((".pdf", ".txt")):
                            resume_files.append(os.path.join(root, filename))
            elif file.filename.lower().endswith((".pdf", ".txt")):
                resume_files.append(file_path)

        for resume_path in resume_files:
            resume_text = extract_resume_text(resume_path)
            processed_text = preprocess_text(resume_text)

            if not processed_text.strip():
                continue

            keyword_score = keyphrase_score(processed_text, skills_list)
            semantic_score_val = semantic_score(processed_text, processed_job_description)
            final_score = (keyword_score + semantic_score_val) / 2

            results.append({
                "filename": os.path.basename(resume_path),
                "keyphrase_score": keyword_score,
                "semantic_score": semantic_score_val,
                "final_score": final_score,
            })

    results.sort(key=lambda item: item["final_score"], reverse=True)
    return JSONResponse(results)

# =============================== 
# RUN SERVER
# ===============================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6996)

