#!/usr/bin/env python
"""Registry-driven matchup / ablation harness.

Plays every (agent_a, agent_b[, agent_c, agent_d]) pairing from the
``--agents`` list against itself for ``--games`` games each, swapping
seat order for fairness, and writes per-game CSV + summary JSON.

Examples:
    # 1v1 ablation: random vs heuristic vs kg_heuristic, 8 games each pair
    python scripts/run_matchups.py \
        --format standard --decks data/decks/modern_mono_red_burn.txt data/decks/modern_azorius_control.txt \
        --agents random heuristic kg_heuristic --games 8 \
        --out runs/ablation_1v1

    # 4-player EDH pod, all four seats different agents
    python scripts/run_matchups.py \
        --format commander --pod \
        --decks data/decks/edh_pod/krenko-mob-boss_core.txt \
                data/decks/edh_pod/atraxa-praetors-voice_core.txt \
                data/decks/edh_pod/urza-lord-high-artificer_core.txt \
                data/decks/edh_pod/meren-of-clan-nel-toth_core.txt \
        --agents heuristic kg_heuristic world_model llm_fusion --games 4 \
        --out runs/ablation_pod

The ``--agents`` list is interpreted as a *pool*.  In 1v1 mode every
ordered pair from the pool plays; in pod mode the first 4 names are
assigned to seats 1..4 and games rotate seat orders across rounds.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

# Force UTF-8 stdout on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import make_agent, list_agents  # noqa: E402
from src.integrations.decklist_loader import DecklistLoader  # noqa: E402
from src.integrations.offline_card_db import get_default_db  # noqa: E402
from src.orchestrator.game_runner import GameConfig, GameRunner  # noqa: E402
from src.utils.seeding import set_global_seed  # noqa: E402


logger = logging.getLogger("run_matchups")


# --------------------------------------------------------------------------
# Deck loading
# --------------------------------------------------------------------------

def _fallback_card(name: str) -> dict:
    lower = name.lower()
    is_basic = lower in {"mountain", "island", "plains", "swamp", "forest", "wastes"}
    color = {"mountain": "R", "island": "U", "plains": "W",
             "swamp": "B", "forest": "G"}.get(lower, "C")
    return {
        "name": name,
        "type_line": f"Basic Land — {name.title()}" if is_basic else "Creature",
        "oracle_text": f"({{T}}: Add {{{color}}})" if is_basic else "",
        "mana_cost": "" if is_basic else "{2}",
        "cmc": 0 if is_basic else 2,
        "power": None if is_basic else "2",
        "toughness": None if is_basic else "2",
        "set": "STUB",
    }


def load_deck(path: Path, format: str) -> tuple[str, list[dict]]:
    """Parse a decklist; return ``(label, cards)``.

    Commander decks include the commander as element 0 (with
    ``is_commander=True``), so ``GameRunner._setup_game`` lifts it.
    """
    text = path.read_text(encoding="utf-8")
    deck = DecklistLoader().from_text(text)
    db = get_default_db()

    cards: list[dict] = []
    if format == "commander":
        if not deck.commander:
            raise SystemExit(f"[error] {path.name} has no Commander section")
        for name in deck.commander:
            data = dict(db.get(name) or _fallback_card(name))
            data["is_commander"] = True
            cards.append(data)

    for name, count in deck.mainboard.items():
        data = db.get(name) or _fallback_card(name)
        for _ in range(count):
            cards.append(dict(data))

    label = path.stem
    return label, cards


# --------------------------------------------------------------------------
# Agent factory wrapper
# --------------------------------------------------------------------------

def _build_agent(agent_name: str, player_id: str, seed: int | None) -> "MTGAgent":
    """Build an agent by registry name; pass ``seed`` if the factory accepts it."""
    try:
        return make_agent(agent_name, player_id, seed=seed)
    except TypeError:
        return make_agent(agent_name, player_id)


# --------------------------------------------------------------------------
# Single game
# --------------------------------------------------------------------------

async def play_one_game(
    agent_names: list[str],
    deck_paths: list[Path],
    format: str,
    max_turns: int,
    seed: int | None,
    log_dir: Path | None,
    game_idx: int,
) -> dict:
    decks: dict[str, list[dict]] = {}
    agents: dict[str, "MTGAgent"] = {}
    seat_to_agent: dict[str, str] = {}
    seat_to_deck: dict[str, str] = {}

    for i, (agent_name, deck_path) in enumerate(zip(agent_names, deck_paths), start=1):
        pid = f"player{i}"
        deck_label, cards = load_deck(deck_path, format=format)
        decks[pid] = cards
        agents[pid] = _build_agent(agent_name, pid, seed=seed)
        seat_to_agent[pid] = agent_name
        seat_to_deck[pid] = deck_label

    starting_life = 40 if format == "commander" else 20
    config = GameConfig(
        format=format,
        starting_life=starting_life,
        max_turns=max_turns,
        mulligan_enabled=True,
        max_mulligans=3,
    )
    runner = GameRunner(config)

    t0 = time.time()
    try:
        result = await runner.run_game(agents=agents, decks=decks)
    except Exception as e:
        logger.exception("game crashed: %s", e)
        return {
            "game": game_idx,
            "format": format,
            "winner_seat": None,
            "winner_agent": None,
            "winner_deck": None,
            "turns": 0,
            "error": str(e),
            "elapsed_sec": round(time.time() - t0, 2),
            "seats": seat_to_agent,
            "decks": seat_to_deck,
        }
    elapsed = time.time() - t0

    record = {
        "game": game_idx,
        "format": format,
        "winner_seat": result.winner,
        "winner_agent": seat_to_agent.get(result.winner) if result.winner else None,
        "winner_deck": seat_to_deck.get(result.winner) if result.winner else None,
        "turns": result.turns,
        "elapsed_sec": round(elapsed, 2),
        "seats": seat_to_agent,
        "decks": seat_to_deck,
        "error": None,
    }

    if log_dir is not None and isinstance(result.log, list):
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"game_{game_idx:04d}.log"
        log_path.write_text("\n".join(str(line) for line in result.log), encoding="utf-8")
        record["log_path"] = str(log_path)

    return record


# --------------------------------------------------------------------------
# Matchup orchestration
# --------------------------------------------------------------------------

def _enumerate_1v1_pairings(agents: list[str]) -> list[tuple[str, str]]:
    """Every ordered pair (a, b) where a != b — covers seat-swap fairness."""
    return [(a, b) for a, b in itertools.permutations(agents, 2)]


def _enumerate_pod_seatings(agents: list[str], rotations: int = 4) -> list[list[str]]:
    """Cyclic rotations of the pod so each agent plays every seat."""
    base = list(agents[:4])
    out = []
    for r in range(rotations):
        out.append(base[r:] + base[:r])
    return out


async def run_matchups(args: argparse.Namespace) -> None:
    if args.seed is not None:
        set_global_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs" if args.save_logs else None

    deck_paths = [Path(d) for d in args.decks]
    for p in deck_paths:
        if not p.exists():
            raise SystemExit(f"[error] deck not found: {p}")

    agents = args.agents
    unknown = [a for a in agents if a not in list_agents()]
    if unknown:
        raise SystemExit(
            f"[error] unknown agents: {unknown}. "
            f"Available: {', '.join(list_agents())}"
        )

    records: list[dict] = []
    game_idx = 0

    if args.pod:
        if len(agents) < 4:
            raise SystemExit("[error] --pod needs >= 4 agents in --agents")
        if len(deck_paths) < 4:
            raise SystemExit("[error] --pod needs >= 4 decks in --decks")
        seatings = _enumerate_pod_seatings(agents, rotations=min(4, args.games))
        for round_idx in range(args.games):
            seating = seatings[round_idx % len(seatings)]
            game_idx += 1
            print(f"\n[pod] game {game_idx} seating={seating}")
            rec = await play_one_game(
                agent_names=seating,
                deck_paths=deck_paths[:4],
                format=args.format,
                max_turns=args.max_turns,
                seed=args.seed + round_idx if args.seed is not None else None,
                log_dir=log_dir,
                game_idx=game_idx,
            )
            records.append(rec)
            print(f"  -> winner={rec['winner_agent']} ({rec['winner_deck']}) "
                  f"in {rec['turns']} turns, {rec['elapsed_sec']}s")
    else:
        # 1v1 — every ordered pair plays N games, seats already swapped via permutations.
        if len(deck_paths) < 2:
            raise SystemExit("[error] 1v1 needs >= 2 decks")
        deck_a, deck_b = deck_paths[0], deck_paths[1]
        for pair in _enumerate_1v1_pairings(agents):
            for game_n in range(args.games):
                game_idx += 1
                print(f"\n[1v1] game {game_idx}: {pair[0]} (deck={deck_a.stem}) "
                      f"vs {pair[1]} (deck={deck_b.stem})")
                rec = await play_one_game(
                    agent_names=list(pair),
                    deck_paths=[deck_a, deck_b],
                    format=args.format,
                    max_turns=args.max_turns,
                    seed=(args.seed + game_idx) if args.seed is not None else None,
                    log_dir=log_dir,
                    game_idx=game_idx,
                )
                records.append(rec)
                print(f"  -> winner={rec['winner_agent']} in {rec['turns']} turns, "
                      f"{rec['elapsed_sec']}s")

    # ----- write CSV + JSON summary -----
    csv_path = out_dir / "games.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["game", "format", "winner_seat", "winner_agent",
                    "winner_deck", "turns", "elapsed_sec", "seats", "decks", "error"])
        for r in records:
            w.writerow([
                r["game"], r["format"], r["winner_seat"], r["winner_agent"],
                r["winner_deck"], r["turns"], r["elapsed_sec"],
                json.dumps(r["seats"]), json.dumps(r["decks"]),
                r.get("error") or "",
            ])

    win_counts: Counter = Counter(r["winner_agent"] for r in records if r["winner_agent"])
    games_total = len([r for r in records if r.get("error") is None])
    summary = {
        "total_games": len(records),
        "games_completed": games_total,
        "games_errored": len(records) - games_total,
        "format": args.format,
        "pod": args.pod,
        "agents": agents,
        "decks": [str(p) for p in deck_paths],
        "wins_by_agent": dict(win_counts),
        "win_rate_by_agent": {
            a: (win_counts.get(a, 0) / games_total) if games_total else 0.0
            for a in agents
        },
        "avg_turns": (sum(r["turns"] for r in records) / max(games_total, 1)),
        "avg_elapsed_sec": (sum(r["elapsed_sec"] for r in records) / max(games_total, 1)),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*72}\nSUMMARY ({games_total}/{len(records)} games)\n{'='*72}")
    for a, n in win_counts.most_common():
        print(f"  {a:<20} {n:3d} wins  "
              f"({summary['win_rate_by_agent'][a]*100:5.1f}% of completed)")
    print(f"\n  CSV    -> {csv_path}")
    print(f"  JSON   -> {summary_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", nargs="+", required=True,
                   help=f"Agent names from registry: {', '.join(list_agents())}")
    p.add_argument("--decks", nargs="+", required=True,
                   help="Deck paths.  1v1 uses first two; pod uses first four.")
    p.add_argument("--format", choices=["standard", "commander"], default="standard")
    p.add_argument("--pod", action="store_true",
                   help="Run a 4-player pod instead of pairwise 1v1.")
    p.add_argument("--games", type=int, default=4,
                   help="Games per pairing (1v1) / rotations (pod).")
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default="runs/matchups",
                   help="Output dir for CSV + JSON.")
    p.add_argument("--save-logs", action="store_true",
                   help="Save per-game gameplay log under <out>/logs/.")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(run_matchups(args))


if __name__ == "__main__":
    main()
