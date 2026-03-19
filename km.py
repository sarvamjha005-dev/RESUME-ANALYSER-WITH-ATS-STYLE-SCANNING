# ===============================
# IMPORTS
# ===============================
import os
import re
import string
import zipfile
import tempfile
import fitz
import spacy
import uvicorn

from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer, util


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
        doc = fitz.open(file_path)
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

    return float(sim)


# ===============================
# HOME ROUTE
# ===============================
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>AI Resume Screening</title>
        </head>
        <body>
            <h1>Welcome to AI Resume Screening</h1>
            <form action="/analyze" method="post" enctype="multipart/form-data">
                <label for="job_description">Job Description:</label><br>
                <textarea name="job_description" rows="4" cols="50"></textarea><br><br>
                <label for="skills">Skills (comma-separated):</label><br>
                <input type="text" name="skills"><br><br>
                <label for="files">Upload Resumes:</label><br>
                <input type="file" name="files" multiple><br><br>
                <input type="submit" value="Analyze">
            </form>
        </body>
    </html>
    """

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allows requests from your HTML page
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_resume(
    job_description: str = Form(...),
    skills: str = Form(...),
    files: List[UploadFile] = File(...)
):

    skills_list = [s.strip() for s in skills.split(",")]
    results = []

    with tempfile.TemporaryDirectory() as temp_dir:

        resume_files = []

        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            if file.filename.endswith('.zip'):
                # Extract ZIP
                extract_dir = os.path.join(temp_dir, 'extracted_' + file.filename[:-4])
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                # Find all PDF and TXT files in extracted dir
                for root, dirs, filenames in os.walk(extract_dir):
                    for filename in filenames:
                        if filename.endswith(('.pdf', '.txt')):
                            resume_files.append(os.path.join(root, filename))
            else:
                resume_files.append(file_path)

        for resume_path in resume_files:
            resume_text = extract_resume_text(resume_path)
            processed_text = preprocess_text(resume_text)
            
            keyword_score = keyphrase_score(processed_text, skills_list)
            semantic_score_val = semantic_score(processed_text, job_description)
            
            results.append({
                "filename": os.path.basename(resume_path),
                "keyword_score": keyword_score,
                "semantic_score": semantic_score_val
            })

    # Build HTML response
    html_content = """
    <html>
        <head>
            <title>Analysis Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .score { font-weight: bold; }
                .high { color: green; }
                .medium { color: orange; }
                .low { color: red; }
            </style>
        </head>
        <body>
            <h1>Resume Analysis Results</h1>
            <table>
                <tr>
                    <th>Filename</th>
                    <th>Keyword Score</th>
                    <th>Semantic Score</th>
                </tr>
    """
    
    for result in results:
        kw_class = "high" if result["keyword_score"] > 0.5 else "medium" if result["keyword_score"] > 0.2 else "low"
        sem_class = "high" if result["semantic_score"] > 0.5 else "medium" if result["semantic_score"] > 0.2 else "low"
        html_content += f"""
                <tr>
                    <td>{result["filename"]}</td>
                    <td class="score {kw_class}">{result["keyword_score"]:.2f}</td>
                    <td class="score {sem_class}">{result["semantic_score"]:.2f}</td>
                </tr>
        """
    
    html_content += """
            </table>
            <br><a href="/">Analyze More Resumes</a>
        </body>
    </html>
    """
    
    return html_content

# =============================== 
# RUN SERVER
# ===============================
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=6970)
