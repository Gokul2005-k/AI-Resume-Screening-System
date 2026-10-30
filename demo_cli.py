import argparse
import json
from pathlib import Path

from ats_engine import analyze_resume_job_match


def read_text(source: str) -> str:
    path = Path(source)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8", errors="ignore")
    return source


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze ATS compatibility between a resume and a job description."
    )
    parser.add_argument(
        "--resume",
        required=True,
        help="Resume text or a path to a .txt file containing the resume.",
    )
    parser.add_argument(
        "--job",
        required=True,
        help="Job description text or a path to a .txt file containing the job description.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as formatted JSON.",
    )
    args = parser.parse_args()

    resume_text = read_text(args.resume)
    job_text = read_text(args.job)
    result = analyze_resume_job_match(resume_text, job_text)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"ATS Score: {result['ats_score']}%")
    print(f"Semantic Match: {result['similarity_score']}%")
    print(f"Skill Coverage: {result['skill_coverage']}%")
    print("Matched Skills:", ", ".join(result["matched_skills"]) or "None")
    print("Missing Skills:", ", ".join(result["missing_skills"]) or "None")
    print("Suggestions:")
    for suggestion in result["suggestions"]:
        print(f"- {suggestion}")


if __name__ == "__main__":
    main()
