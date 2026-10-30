from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from ats_engine import analyze_resume_job_match, extract_text_from_uploaded_file


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
MAX_UPLOAD_SIZE_MB = 5
JOB_ROLE_TEMPLATES = {
    "python_developer": {
        "title": "Python Developer",
        "description": (
            "Python developer role requiring strong knowledge of Python, SQL, "
            "Flask or Django, REST API development, Git, problem solving, "
            "debugging, database integration, and clean backend development."
        ),
    },
    "data_analyst": {
        "title": "Data Analyst",
        "description": (
            "Data analyst role requiring Python, SQL, Excel, data analysis, "
            "pandas, numpy, Tableau or Power BI, communication, analytical "
            "thinking, dashboard creation, and business reporting."
        ),
    },
    "machine_learning_engineer": {
        "title": "Machine Learning Engineer",
        "description": (
            "Machine learning engineer role requiring Python, machine learning, "
            "deep learning, NLP, scikit-learn, pandas, numpy, TensorFlow or "
            "PyTorch, data preprocessing, model evaluation, and problem solving."
        ),
    },
    "full_stack_developer": {
        "title": "Full Stack Developer",
        "description": (
            "Full stack developer role requiring Python or JavaScript, HTML, CSS, "
            "Flask or Django, REST API, SQL, Git, frontend integration, backend "
            "logic, deployment basics, teamwork, and problem solving."
        ),
    },
    "cloud_devops_engineer": {
        "title": "Cloud / DevOps Engineer",
        "description": (
            "Cloud and DevOps role requiring Linux, Git, GitHub, Docker, "
            "Kubernetes, AWS or Azure, CI/CD, Jenkins, deployment automation, "
            "monitoring basics, and troubleshooting."
        ),
    },
    "software_engineer_fresher": {
        "title": "Software Engineer Fresher",
        "description": (
            "Entry-level software engineer role requiring Python or Java, SQL, "
            "data structures, HTML, CSS, Git, projects, communication, teamwork, "
            "analytical thinking, and problem solving."
        ),
    },
}
DEFAULT_ROLE_KEY = "software_engineer_fresher"

UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024


def is_supported_file(filename: str) -> bool:
    """Return True when the uploaded resume format can be processed."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def read_uploaded_resume(
    uploaded_file: FileStorage | None,
) -> tuple[str, str | None, str | None]:
    """Extract text from a resume upload.

    Returns extracted text, saved filename, and an optional user-facing error.
    """
    if not uploaded_file or not uploaded_file.filename:
        return "", None, None

    if not is_supported_file(uploaded_file.filename):
        return (
            "",
            uploaded_file.filename,
            "Unsupported resume format. Please upload a TXT, PDF, or DOCX file.",
        )

    safe_name = secure_filename(uploaded_file.filename or "")
    extracted_text = extract_text_from_uploaded_file(uploaded_file)

    if not extracted_text:
        return (
            "",
            safe_name,
            "Your file was uploaded, but text could not be extracted. "
            "Paste the resume text manually for now.",
        )

    return extracted_text.strip(), safe_name, None


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(error: RequestEntityTooLarge):
    return (
        render_template(
            "index.html",
            result=None,
            error=f"Resume file is too large. Please upload a file below {MAX_UPLOAD_SIZE_MB} MB.",
            resume_text="",
            job_templates=JOB_ROLE_TEMPLATES,
            selected_role=DEFAULT_ROLE_KEY,
            uploaded_filename=None,
        ),
        413,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    resume_text = ""
    selected_role = DEFAULT_ROLE_KEY
    uploaded_filename = None

    if request.method == "POST":
        resume_text = request.form.get("resume_text", "").strip()
        selected_role = request.form.get("job_role", DEFAULT_ROLE_KEY)
        selected_template = JOB_ROLE_TEMPLATES.get(selected_role)

        uploaded_file = request.files.get("resume_file")
        extracted_text, uploaded_filename, upload_error = read_uploaded_resume(
            uploaded_file
        )
        if upload_error:
            error = upload_error
        if extracted_text and not resume_text:
            resume_text = extracted_text

        if not selected_template:
            error = error or "Please select a valid job role."
        elif not resume_text:
            error = error or "Please upload a resume or paste resume text."
        else:
            job_description = selected_template["description"]
            result = analyze_resume_job_match(resume_text, job_description)
            result["selected_role"] = selected_template["title"]

    return render_template(
        "index.html",
        result=result,
        error=error,
        resume_text=resume_text,
        job_templates=JOB_ROLE_TEMPLATES,
        selected_role=selected_role,
        uploaded_filename=uploaded_filename,
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
