"""Check shared citation keys and final LaTeX/BibTeX reference diagnostics."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ("mtg_ontology", "opposition_agents_mtg", "agents_tech_report")
ENTRY = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")
CITATION = re.compile(
    r"\\(?:cite(?:t|p|author|year|yearpar|alp|alt)?|nocite)\*?"
    r"(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]+)\}",
)
DIAGNOSTIC = re.compile(
    r"(?:Citation .+ undefined|Reference .+ undefined|"
    r"There were undefined (?:citations|references)|"
    r"There were multiply-defined labels|Label .+ multiply defined|"
    r"Warning--|I couldn't open database file|"
    r"I didn't find a database entry|Repeated entry|"
    r"---line \d+ of file|There (?:was|were) \d+ error messages?)",
    re.IGNORECASE,
)


def uncomment(text: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def validate_sources(paper_dir: Path) -> tuple[list[str], dict[str, int]]:
    bibliography = paper_dir / "references.bib"
    if not bibliography.is_file():
        return [f"Missing shared bibliography: {bibliography}"], {}
    keys = Counter(ENTRY.findall(uncomment(bibliography.read_text(encoding="utf-8"))))
    errors = [f"Duplicate bibliography key: {key}" for key, count in keys.items() if count > 1]
    counts = {}
    for name in PAPERS:
        source = paper_dir / f"{name}.tex"
        if not source.is_file():
            errors.append(f"Missing manuscript: {source}")
            continue
        text = uncomment(source.read_text(encoding="utf-8"))
        if r"\bibliography{references}" not in text:
            errors.append(f"{name}: does not use the shared bibliography")
        if r"\begin{thebibliography}" in text or r"\bibitem" in text:
            errors.append(f"{name}: inline bibliography remains")
        cited = {
            key.strip() for group in CITATION.findall(text)
            for key in group.split(",") if key.strip() != "*"
        }
        counts[name] = len(cited)
        errors.extend(
            f"{name}: undefined citation key: {key}" for key in sorted(cited - keys.keys())
        )
    return errors, counts


def validate_build(paper_dir: Path) -> list[str]:
    errors = []
    for name in PAPERS:
        for extension in ("log", "blg"):
            path = paper_dir / f"{name}.{extension}"
            if not path.is_file():
                errors.append(f"Missing final build diagnostic file: {path}")
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if DIAGNOSTIC.search(line):
                    errors.append(f"{name}.{extension}: {line.strip()}")
    return errors


def validate_database(paper_dir: Path) -> list[str]:
    """Ask BibTeX to parse and render every entry, including uncited reading."""
    executable = shutil.which("bibtex")
    if executable is None:
        return ["BibTeX is required to validate the entire reference database"]
    with tempfile.TemporaryDirectory(prefix="mtg-bibliography-") as directory:
        work = Path(directory)
        shutil.copyfile(paper_dir / "references.bib", work / "references.bib")
        (work / "audit.aux").write_text(
            "\\relax\n\\citation{*}\n\\bibstyle{plainnat}\n\\bibdata{references}\n",
            encoding="ascii",
        )
        try:
            result = subprocess.run(
                [executable, "audit"], cwd=work, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [f"BibTeX database validation failed: {exc}"]
        diagnostics = result.stdout + "\n" + result.stderr
        log = work / "audit.blg"
        if log.is_file():
            diagnostics += "\n" + log.read_text(encoding="utf-8", errors="replace")
        errors = sorted({
            line.strip() for line in diagnostics.splitlines() if DIAGNOSTIC.search(line)
        })
        if result.returncode:
            errors.append(f"BibTeX database audit exited {result.returncode}: {result.stdout}")
        return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper")
    parser.add_argument("--check-build", action="store_true")
    parser.add_argument("--check-database", action="store_true")
    args = parser.parse_args()
    errors, counts = validate_sources(args.paper_dir)
    if args.check_build:
        errors.extend(validate_build(args.paper_dir))
    if args.check_database and not errors:
        errors.extend(validate_database(args.paper_dir))
    if errors:
        for error in errors:
            print(error)
        return 1
    for name, count in counts.items():
        print(f"{name}: {count} distinct citation keys resolved")
    print("Shared paper references validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
