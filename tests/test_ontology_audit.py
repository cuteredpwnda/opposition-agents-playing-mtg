from __future__ import annotations

import pytest

from scripts.check_card_types import select_corpus
from scripts.validate_ontology import StageResult, _report
from src.knowledge.type_line import report_over_cards


def test_declared_format_scope_retains_restricted_and_reports_exclusions():
    cards = [
        {"name": status, "legalities": {"vintage": status}}
        for status in ("legal", "restricted", "banned", "not_legal")
    ]
    eligible, excluded = select_corpus(cards, "vintage")
    assert [c["name"] for c in eligible] == ["legal", "restricted"]
    assert excluded == {"banned": 1, "not_legal": 1}
    assert select_corpus(cards, None) == (cards, {})


@pytest.mark.parametrize("legalities", [None, {}, {"commander": "unknown"}])
def test_incomplete_legality_is_an_explicit_error(legalities):
    with pytest.raises(ValueError):
        select_corpus([{"name": "test", "legalities": legalities}], "commander")


def test_unknown_card_types_are_not_dropped_from_the_denominator():
    report = report_over_cards([{"name": "test", "type_line": "UnknownType"}])
    assert report.total == 1
    assert report.well_formed == 0
    assert report.with_unknown == 1
    assert report.unknown_counts == {"UnknownType": 1}


def test_auxiliary_layout_exclusion_is_explicit_and_counted():
    cards = [{"name": "sheet", "layout": "normal", "type_line": "Stickers",
              "legalities": {"commander": "legal"}},
             {"name": "card", "layout": "normal", "legalities": {"commander": "legal"}}]
    assert len(select_corpus(cards, "commander")[0]) == 2
    eligible, excluded = select_corpus(cards, "commander", (), ("Stickers",))
    assert eligible == [cards[1]]
    assert excluded == {"auxiliary_type:Stickers": 1}


def test_validation_summary_does_not_count_skips_as_passes(capsys):
    results = [StageResult("syntax", True, "parsed"),
               StageResult("reasoner", True, "unavailable", skipped=True)]
    assert _report(results) == 0
    assert "1/1 enabled stages passed; 1 skipped" in capsys.readouterr().out
    results.append(StageResult("shacl", False, "violations"))
    assert _report(results) == 1
    assert "1/2 enabled stages passed; 1 skipped" in capsys.readouterr().out
