# AI Resume Screening System

An NLP-based Python web application that compares resumes with predefined job-role templates and calculates match scores using TF-IDF and cosine similarity. It also identifies matching and missing skills for job roles.

## Tech Stack

- Python
- Flask
- HTML
- CSS
- scikit-learn

## Features

- Attractive recruiter-style web interface
- Resume upload plus manual text input
- Job-role template selection instead of manual job-description typing
- Clear and New Screening actions for smooth navigation
- TF-IDF and cosine similarity scoring
- Matching skill and missing-skill analysis
- Missing keyword analysis
- Resume readability checks
- Resume improvement suggestions
- CLI mode for automation testing

## Final User Flow

1. Upload a resume as PDF, DOCX, or TXT.
2. Select the target job role from the dropdown.
3. Click `Screen Resume`.
4. Review the ATS match report.
5. Click `New Screening` or `Clear` to return to a fresh upload page with the previous result cleared.

## Run The Web App

First-time setup:

```powershell
pip install -r requirements.txt
```

Then every time:

```powershell
python app.py
```

Open this in your browser:

```powershell
http://127.0.0.1:5000
```

## Run The CLI

```powershell
python demo_cli.py --resume "Python, machine learning, pandas" --job "Need Python, SQL, machine learning" --json
```

## Notes

- TXT, PDF, and DOCX uploads are processed in memory.
- PDF upload needs `pypdf` to extract text automatically.
- DOCX upload needs `python-docx` to extract text automatically.

## Documentation

- `docs/AI_Resume_Screening_System_Project_Documentation.docx` explains the complete project.
