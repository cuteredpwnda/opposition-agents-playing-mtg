"""Deck-adjusted analysis of a ``run_phase_rs_campaign.py`` campaign.

Raw win rates confound agent skill with deck strength whenever cells are
unbalanced or incomplete. This script fits a ridge-penalised fractional
logistic regression::

    logit P(win) = b0 + arm + our_deck + opp_deck (+ difficulty)

with draws scored 0.5, then reports each arm's *standardised* (deck-adjusted)
win rate: the mean predicted win probability over every observed
(our deck, opponent deck, difficulty) matchup with the arm set to that value.
Confidence intervals come from a cluster bootstrap that resamples matchups,
so games sharing a matchup are not treated as independent.

For ``*_learn`` arms a learning curve compares the first and second halves of
games in completion order, with a permutation test against no trend.

Usage::

    python scripts/analyze_phase_rs_campaign.py runs/campaigns/standard_rr_v375
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCORE = {"win": 1.0, "loss": 0.0, "draw": 0.5}


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def load(root: Path) -> tuple[list[dict], dict[str, int]]:
    rows, excluded = [], defaultdict(int)
    for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["outcome"] in SCORE:
            rows.append(row)
        else:
            excluded[f"{row['arm']}:{row['reason']}"] += 1
    return rows, dict(excluded)


class Design:
    """Treatment-coded design matrix with a fixed reference level per factor."""

    def __init__(self, rows: list[dict], factors: list[str], reference_arm: str) -> None:
        self.factors = factors
        self.levels: dict[str, list[str]] = {}
        for factor in factors:
            levels = sorted({row[factor] for row in rows})
            if factor == "arm" and reference_arm in levels:
                levels.remove(reference_arm)
                levels.insert(0, reference_arm)
            self.levels[factor] = levels
        self.columns = ["intercept"] + [
            f"{factor}={level}" for factor in factors for level in self.levels[factor][1:]
        ]

    def matrix(self, rows: list[dict], override: dict[str, str] | None = None) -> np.ndarray:
        x = np.zeros((len(rows), len(self.columns)))
        x[:, 0] = 1.0
        index = {name: i for i, name in enumerate(self.columns)}
        for r, row in enumerate(rows):
            for factor in self.factors:
                value = (override or {}).get(factor, row[factor])
                column = index.get(f"{factor}={value}")
                if column is not None:
                    x[r, column] = 1.0
        return x


def fit_logistic(
    x: np.ndarray, y: np.ndarray, ridge: float = 1.0, iterations: int = 50,
) -> np.ndarray:
    """Fractional-response logistic regression by IRLS with an L2 penalty."""
    beta = np.zeros(x.shape[1])
    penalty = np.full(x.shape[1], ridge)
    penalty[0] = 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(x @ beta)))
        w = np.clip(p * (1 - p), 1e-6, None)
        gradient = x.T @ (y - p) - penalty * beta
        hessian = (x.T * w) @ x + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def standardised(design: Design, beta: np.ndarray, rows: list[dict], arm: str) -> float:
    matchups = {tuple(row[f] for f in design.factors if f != "arm") for row in rows}
    keys = [f for f in design.factors if f != "arm"]
    grid = [dict(zip(keys, m)) | {"arm": arm} for m in matchups]
    p = 1.0 / (1.0 + np.exp(-(design.matrix(grid) @ beta)))
    return float(p.mean())


def analyse(root: Path, reference_arm: str, boots: int, seed: int, ridge: float) -> dict:
    rows, excluded = load(root)
    arms = sorted({row["arm"] for row in rows})
    if reference_arm not in arms:
        raise SystemExit(f"reference arm {reference_arm!r} not in campaign arms {arms}")
    factors = ["arm", "our_deck", "opp_deck"]
    if len({row["difficulty"] for row in rows}) > 1:
        factors.append("difficulty")
    design = Design(rows, factors, reference_arm)
    y = np.array([SCORE[row["outcome"]] for row in rows])
    beta = fit_logistic(design.matrix(rows), y, ridge)
    arms = design.levels["arm"]

    clusters: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        clusters[(row["our_deck"], row["opp_deck"], row["difficulty"])].append(i)
    cluster_rows = list(clusters.values())
    rng = np.random.default_rng(seed)
    boot: dict[str, list[float]] = {arm: [] for arm in arms}
    boot_diff: dict[str, list[float]] = {arm: [] for arm in arms}
    for _ in range(boots):
        picks = rng.integers(0, len(cluster_rows), len(cluster_rows))
        sample = [rows[i] for c in picks for i in cluster_rows[c]]
        if len({row["arm"] for row in sample}) < len(arms):
            continue
        scores = np.array([SCORE[r["outcome"]] for r in sample])
        b = fit_logistic(design.matrix(sample), scores, ridge)
        reference = standardised(design, b, sample, reference_arm)
        for arm in arms:
            value = standardised(design, b, sample, arm)
            boot[arm].append(value)
            boot_diff[arm].append(value - reference)

    def interval(values: list[float]) -> list[float]:
        if not values:
            return []
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    reference_point = standardised(design, beta, rows, reference_arm)
    arm_table = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        score = sum(SCORE[row["outcome"]] for row in arm_rows)
        point = standardised(design, beta, rows, arm)
        arm_table[arm] = {
            "games": len(arm_rows),
            "wins": sum(row["outcome"] == "win" for row in arm_rows),
            "draws": sum(row["outcome"] == "draw" for row in arm_rows),
            "raw_score": score / len(arm_rows) if arm_rows else None,
            "raw_wilson95": wilson(score, len(arm_rows)),
            "adjusted": point,
            "adjusted_ci95": interval(boot[arm]),
            f"diff_vs_{reference_arm}": point - reference_point,
            f"diff_vs_{reference_arm}_ci95": interval(boot_diff[arm]),
            "mean_turns": float(np.mean([row["turns"] for row in arm_rows])) if arm_rows else None,
        }

    def deck_effects(factor: str) -> dict[str, float]:
        coefficients = dict(zip(design.columns, beta))
        return {
            level: float(coefficients.get(f"{factor}={level}", 0.0))
            for level in design.levels[factor]
        }

    learning = {}
    for arm in arms:
        if not arm.endswith("_learn"):
            continue
        ordered = sorted(
            (row for row in rows if row["arm"] == arm), key=lambda r: r["cell_finished"]
        )
        scores = np.array([SCORE[row["outcome"]] for row in ordered])
        if len(scores) < 4:
            continue
        half = len(scores) // 2
        observed = scores[half:].mean() - scores[:half].mean()
        permuted = []
        for _ in range(2000):
            shuffled = rng.permutation(scores)
            permuted.append(shuffled[half:].mean() - shuffled[:half].mean())
        window = max(10, len(scores) // 10)
        learning[arm] = {
            "games": len(scores),
            "first_half": float(scores[:half].mean()),
            "second_half": float(scores[half:].mean()),
            "difference": float(observed),
            "permutation_p_two_sided": float(np.mean(np.abs(permuted) >= abs(observed))),
            "rolling_window": window,
            "rolling": [
                float(scores[max(0, i - window + 1): i + 1].mean()) for i in range(len(scores))
            ][window - 1:: max(1, window // 2)],
        }

    return {
        "schema": "phase_rs_campaign_analysis_v1",
        "campaign": root.name,
        "games_scored": len(rows),
        "excluded": excluded,
        "model": "fractional logistic, treatment coding, ridge on non-intercept terms",
        "factors": factors,
        "ridge": ridge,
        "bootstrap": {"replicates": len(boot[reference_arm]), "unit": "matchup cluster"},
        "reference_arm": reference_arm,
        "arms": arm_table,
        "our_deck_logit": deck_effects("our_deck"),
        "opp_deck_logit": deck_effects("opp_deck"),
        "learning": learning,
    }


def to_markdown(result: dict) -> str:
    ref = result["reference_arm"]
    lines = [
        f"# Campaign `{result['campaign']}`",
        "",
        f"Scored games: {result['games_scored']}; excluded: {result['excluded'] or 'none'}. "
        f"Model: {result['model']} ({', '.join(result['factors'])}); "
        f"{result['bootstrap']['replicates']} matchup-cluster bootstrap replicates.",
        "",
        "| Arm | Games | Wins | Raw score [Wilson 95%] | Deck-adjusted [95%] "
        f"| Δ vs {ref} [95%] | Mean turns |",
        "|---|---:|---:|---|---|---|---:|",
    ]
    for arm, s in result["arms"].items():
        lo, hi = s["raw_wilson95"]
        a = s["adjusted_ci95"] or [float("nan")] * 2
        d = s[f"diff_vs_{ref}_ci95"] or [float("nan")] * 2
        lines.append(
            f"| {arm} | {s['games']} | {s['wins']} | {s['raw_score']:.3f} [{lo:.3f}, {hi:.3f}] "
            f"| {s['adjusted']:.3f} [{a[0]:.3f}, {a[1]:.3f}] "
            f"| {s[f'diff_vs_{ref}']:+.3f} [{d[0]:+.3f}, {d[1]:+.3f}] | {s['mean_turns']:.1f} |"
        )
    for title, key in (("Our-deck", "our_deck_logit"), ("Opponent-deck", "opp_deck_logit")):
        lines += ["", f"{title} effects (logit, reference = first level):", ""]
        for deck, value in sorted(result[key].items(), key=lambda kv: -kv[1]):
            lines.append(f"- {deck}: {value:+.2f}")
    if result["learning"]:
        lines += ["", "Learning curves (completion order):", ""]
        for arm, s in result["learning"].items():
            lines.append(
                f"- {arm}: first half {s['first_half']:.3f}, second half {s['second_half']:.3f}, "
                f"Δ {s['difference']:+.3f}, permutation p = {s['permutation_p_two_sided']:.3f} "
                f"(n = {s['games']})"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--reference-arm", default="random")
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = analyse(args.root, args.reference_arm, args.bootstrap, args.seed, args.ridge)
    (args.root / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown = to_markdown(result)
    (args.root / "analysis.md").write_text(markdown, encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
