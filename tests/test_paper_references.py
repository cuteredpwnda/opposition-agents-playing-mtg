from __future__ import annotations

from scripts.validate_paper_references import PAPERS, validate_build, validate_sources


def make_papers(tmp_path):
    (tmp_path / "references.bib").write_text("@misc{known, title={Known}, year={2026}}\n")
    for name in PAPERS:
        (tmp_path / f"{name}.tex").write_text(
            "\\citep[see][p. 2]{known}\n\\bibliography{references}\n",
        )
    return tmp_path


def test_shared_repository_references_resolve():
    from scripts.validate_paper_references import ROOT

    errors, counts = validate_sources(ROOT / "paper")
    assert errors == []
    assert all(count > 0 for count in counts.values())


def test_manuscripts_are_self_contained_and_report_current_native_evidence():
    from scripts.validate_paper_references import ROOT

    for name in PAPERS:
        source = (ROOT / "paper" / f"{name}.tex").read_text(encoding="utf-8")
        assert ".md" not in source
        assert "October 5" not in source
    agents = (ROOT / "paper" / "opposition_agents_mtg.tex").read_text(encoding="utf-8")
    assert "Archived preliminary experiments" not in agents
    assert "tab:study-ablations" in agents
    assert "tab:baseline-policies" in agents
    assert "Buchholz" in agents
    assert "Deck-diversity generalisation" in agents


def test_uncited_reading_is_checked_by_bibtex(monkeypatch, tmp_path):
    from subprocess import CompletedProcess
    from unittest.mock import Mock

    from scripts.validate_paper_references import validate_database

    directory = make_papers(tmp_path)
    monkeypatch.setattr("scripts.validate_paper_references.shutil.which", lambda _: "bibtex")
    run = Mock(return_value=CompletedProcess(
        ["bibtex"], 0, stdout="Warning--empty year in uncited", stderr="",
    ))
    monkeypatch.setattr("scripts.validate_paper_references.subprocess.run", run)
    assert validate_database(directory) == ["Warning--empty year in uncited"]
    monkeypatch.setattr("scripts.validate_paper_references.shutil.which", lambda _: None)
    assert "BibTeX is required" in validate_database(directory)[0]


def test_citation_options_comments_and_duplicates(tmp_path):
    directory = make_papers(tmp_path)
    assert validate_sources(directory)[0] == []
    path = directory / f"{PAPERS[0]}.tex"
    path.write_text(path.read_text() + "% \\cite{ignored}\n\\citet*{missing, known}\n")
    assert validate_sources(directory)[0] == [f"{PAPERS[0]}: undefined citation key: missing"]
    bibliography = directory / "references.bib"
    bibliography.write_text(bibliography.read_text() + "@misc{known, title={Duplicate}}\n")
    assert "Duplicate bibliography key: known" in validate_sources(directory)[0]


def test_inline_or_unshared_bibliography_is_rejected(tmp_path):
    directory = make_papers(tmp_path)
    (directory / f"{PAPERS[0]}.tex").write_text("\\begin{thebibliography}{9}\\bibitem{known}")
    errors, _ = validate_sources(directory)
    assert any("does not use" in error for error in errors)
    assert any("inline bibliography" in error for error in errors)


def test_build_requires_logs_and_rejects_reference_warnings(tmp_path):
    assert len(validate_build(tmp_path)) == 6
    for name in PAPERS:
        (tmp_path / f"{name}.log").write_text("Overfull \\hbox (1pt too wide)\n")
        (tmp_path / f"{name}.blg").write_text("This is BibTeX\n")
    assert validate_build(tmp_path) == []
    (tmp_path / f"{PAPERS[0]}.log").write_text(
        "Package natbib Warning: Citation `missing' on page 1 undefined\n",
    )
    (tmp_path / f"{PAPERS[1]}.blg").write_text(
        "Warning--I didn't find a database entry for missing",
    )
    assert len(validate_build(tmp_path)) == 2
