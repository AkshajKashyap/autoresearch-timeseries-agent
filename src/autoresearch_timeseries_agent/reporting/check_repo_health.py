from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CORE_DOCS = {
    "architecture_doc_exists": Path("docs/architecture.md"),
    "method_notes_doc_exists": Path("docs/method_notes.md"),
    "reproducibility_doc_exists": Path("docs/reproducibility.md"),
    "interview_notes_doc_exists": Path("docs/interview_notes.md"),
    "limitations_doc_exists": Path("docs/limitations.md"),
}


def check_repo_health(
    *,
    project_root: Path = Path("."),
    output_dir: Path = Path("reports"),
) -> dict[str, Any]:
    root = project_root.resolve()
    checks = _build_checks(root)
    overall_status = "pass" if all(check["passed"] for check in checks.values()) else "needs_attention"
    payload = {
        "overall_status": overall_status,
        "project_root": str(root),
        "checks": checks,
    }

    resolved_output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output_dir / "repo_health_check.json"
    md_path = resolved_output_dir / "repo_health_check.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")

    print(f"Repo health: {overall_status}")
    print(f"Wrote {json_path} and {md_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a portfolio-readiness health report.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    check_repo_health(project_root=args.project_root, output_dir=args.output_dir)


def _build_checks(root: Path) -> dict[str, dict[str, Any]]:
    checks = {
        "readme_exists": _path_check(root, Path("README.md"), "README exists"),
        "makefile_exists": _path_check(root, Path("Makefile"), "Makefile exists"),
        "pyproject_exists": _path_check(root, Path("pyproject.toml"), "pyproject.toml exists"),
        "tests_directory_exists": _directory_check(root, Path("tests"), "tests directory exists"),
        "project_summary_exists": _path_check(
            root,
            Path("reports/project_summary.md"),
            "project summary report exists",
        ),
        "model_comparison_exists": _path_check(
            root,
            Path("reports/model_comparison.md"),
            "model comparison report exists",
        ),
        "figures_exist": _figures_check(root),
        "ci_workflow_exists": _path_check(
            root,
            Path(".github/workflows/ci.yml"),
            "GitHub Actions CI workflow exists",
        ),
        "tests_do_not_require_generated_raw_data": _raw_data_test_check(root),
    }
    for key, path in CORE_DOCS.items():
        checks[key] = _path_check(root, path, f"{path.as_posix()} exists")
    return checks


def _path_check(root: Path, relative_path: Path, description: str) -> dict[str, Any]:
    path = root / relative_path
    return {
        "passed": path.exists() and path.is_file(),
        "description": description,
        "path": relative_path.as_posix(),
    }


def _directory_check(root: Path, relative_path: Path, description: str) -> dict[str, Any]:
    path = root / relative_path
    return {
        "passed": path.exists() and path.is_dir(),
        "description": description,
        "path": relative_path.as_posix(),
    }


def _figures_check(root: Path) -> dict[str, Any]:
    figures_dir = root / "reports/figures"
    figures = sorted(path.name for path in figures_dir.glob("*.png")) if figures_dir.exists() else []
    return {
        "passed": bool(figures),
        "description": "at least one PNG figure exists under reports/figures",
        "path": "reports/figures",
        "figures": figures,
    }


def _raw_data_test_check(root: Path) -> dict[str, Any]:
    tests_dir = root / "tests"
    references: list[str] = []
    suspicious_patterns = (
        "read_csv(\"data/raw",
        "read_csv('data/raw",
        "open(\"data/raw",
        "open('data/raw",
        "Path(\"data/raw",
        "Path('data/raw",
        "create_example_csv_dataset.py",
    )
    if tests_dir.exists():
        for path in sorted(tests_dir.glob("**/*.py")):
            text = path.read_text(encoding="utf-8")
            if any(pattern in text for pattern in suspicious_patterns):
                references.append(path.relative_to(root).as_posix())
    return {
        "passed": not references,
        "description": "tests do not reference generated data/raw files",
        "path": "tests",
        "raw_data_references": references,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Repo Health Check",
        "",
        f"Overall status: `{payload['overall_status']}`",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for key, check in payload["checks"].items():
        status = "pass" if check["passed"] else "needs attention"
        detail = check["description"]
        if check.get("path"):
            detail = f"{detail} (`{check['path']}`)"
        lines.append(f"| `{key}` | {status} | {detail} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
