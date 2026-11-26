"""Campaign cell construction and deck-adjusted analysis."""

from __future__ import annotations

import json

import numpy as np

from scripts import analyze_phase_rs_campaign as analysis
from scripts import run_phase_rs_campaign as campaign


def test_cells_skip_mirrors_share_seeds_across_arms_and_reject_unknown_arms():
    args = campaign.parse_args([
        "run", "--name", "t", "--our-decks", "data/decks/meta/standard/standard_0[123]_*.txt",
        "--arms", "random", "kl_graph_learn",
    ])
    cells = campaign.build_cells(args)
    assert len(cells) == 3 * 2 * 2
    assert all(cell.our_deck != cell.opp_deck for cell in cells)
    seeds = {(c.our_deck, c.opp_deck): set() for c in cells}
    for cell in cells:
        seeds[(cell.our_deck, cell.opp_deck)].add(cell.seed)
    assert all(len(s) == 1 for s in seeds.values())
    args.arms = ["random", "nonsense"]
    try:
        campaign.build_cells(args)
    except SystemExit as error:
        assert "nonsense" in str(error)
    else:
        raise AssertionError("unknown arm accepted")


def test_adjusted_rate_removes_deck_confounding(tmp_path):
    # Equal skill, but arm "b" mostly plays the strong deck: raw rates differ.
    rng = np.random.default_rng(0)
    strength = {"strong": 0.8, "weak": 0.2}
    rows = []
    for arm, mix in (("a", 0.2), ("b", 0.8)):
        for i in range(800):
            deck = "strong" if rng.random() < mix else "weak"
            win = rng.random() < strength[deck]
            rows.append({
                "arm": arm, "our_deck": deck, "opp_deck": f"o{i % 4}", "difficulty": "VeryEasy",
                "outcome": "win" if win else "loss", "reason": "game_rules", "turns": 10,
                "cell_finished": i,
            })
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    result = analysis.analyse(tmp_path, "a", boots=50, seed=1, ridge=0.1)
    a, b = result["arms"]["a"], result["arms"]["b"]
    assert b["raw_score"] - a["raw_score"] > 0.25
    assert abs(b["adjusted"] - a["adjusted"]) < 0.06
    low, high = b["diff_vs_a_ci95"]
    assert low < 0 < high
