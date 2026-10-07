#!/usr/bin/env python
"""Multi-round tournaments between agents.

Two formats:

* **Swiss (1v1)** — every round, agents are paired by current score.
  No agent plays the same opponent twice.  After ``--rounds`` rounds we
  rank by points (3 win / 1 draw / 0 loss), tiebreak by Buchholz.

* **Pod tournament (EDH 4-player)** — random reseating each round so
  every agent eventually plays every other.  Pod winner gets 3 points,
  losers 0.

All output goes to files under ``--out``:
    run.log         — top-level stdout/stderr
    standings.csv   — final rankings
    rounds.csv      — every game with seats, winner, turns
    summary.json    — aggregate stats

Examples:
    # 6-round Swiss between four agents on Standard
    python scripts/run_tournament.py --mode swiss --rounds 6 \
        --agents random heuristic kg_heuristic active_inference \
        --decks data/decks/modern/modern_mono_red_burn.txt data/decks/modern/modern_azorius_control.txt \
        --format standard --max-turns 80 \
        --out runs/tournament_standard

    # 4-round EDH pod tournament with reseating
    python scripts/run_tournament.py --mode pod --rounds 4 \
        --agents heuristic kg_heuristic world_model llm_fusion \
        --decks data/decks/edh/krenko-mob-boss_core.txt \
                data/decks/edh/atraxa-praetors-voice_core.txt \
                data/decks/edh/urza-lord-high-artificer_core.txt \
                data/decks/edh/meren-of-clan-nel-toth_core.txt \
        --format commander --max-turns 200 \
        --out runs/tournament_edh
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

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


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

class _Tee:
    """Write to multiple streams."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            try: st.write(s)
            except Exception: pass
    def flush(self):
        for st in self.streams:
            try: st.flush()
            except Exception: pass


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
    return path.stem, cards


# --------------------------------------------------------------------------
# Standings
# --------------------------------------------------------------------------

@dataclass
class Standing:
    agent: str
    points: float = 0.0   # 3 / 1 / 0 in 1v1; 3 / 0 in pod
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games: int = 0
    opponents: list[str] = field(default_factory=list)  # for Buchholz tiebreak

    def record(self, result: str, opponents: list[str]) -> None:
        self.games += 1
        self.opponents.extend(opponents)
        if result == "win":
            self.points += 3.0; self.wins += 1
        elif result == "draw":
            self.points += 1.0; self.draws += 1
        else:
            self.losses += 1


def buchholz(s: Standing, table: dict[str, Standing]) -> float:
    return sum(table[opp].points for opp in s.opponents if opp in table)


# --------------------------------------------------------------------------
# Single game executor
# --------------------------------------------------------------------------

async def run_game(
    agent_names: list[str],
    deck_paths: list[Path],
    format: str,
    max_turns: int,
    seed: int | None,
    log_dir: Path,
    game_idx: int,
) -> dict:
    decks: dict[str, list[dict]] = {}
    agents: dict[str, "MTGAgent"] = {}
    seat_to_agent: dict[str, str] = {}
    seat_to_deck: dict[str, str] = {}
    setup_error: str | None = None
    for i, (agent_name, deck_path) in enumerate(zip(agent_names, deck_paths), start=1):
        pid = f"player{i}"
        try:
            deck_label, cards = load_deck(deck_path, format=format)
        except Exception as e:
            setup_error = f"deck load failed for {deck_path}: {e}"
            deck_label, cards = deck_path.stem, []
        decks[pid] = cards
        try:
            agents[pid] = make_agent(agent_name, pid, seed=seed)
        except TypeError:
            try:
                agents[pid] = make_agent(agent_name, pid)
            except Exception as e:
                setup_error = f"agent '{agent_name}' build failed: {e}"
                from src.agents.random_agent import RandomAgent
                agents[pid] = RandomAgent(player_id=pid, name=f"{agent_name}-stub")
        except Exception as e:
            setup_error = f"agent '{agent_name}' build failed: {e}"
            from src.agents.random_agent import RandomAgent
            agents[pid] = RandomAgent(player_id=pid, name=f"{agent_name}-stub")
        seat_to_agent[pid] = agent_name
        seat_to_deck[pid] = deck_label

    if setup_error is not None:
        return {
            "game": game_idx,
            "format": format,
            "seats": seat_to_agent,
            "decks": seat_to_deck,
            "winner_seat": None,
            "winner_agent": None,
            "turns": 0,
            "elapsed_sec": 0.0,
            "log_path": "",
            "error": setup_error,
        }

    starting_life = 40 if format == "commander" else 20
    config = GameConfig(format=format, starting_life=starting_life,
                        max_turns=max_turns, mulligan_enabled=True, max_mulligans=3)
    runner = GameRunner(config)

    t0 = time.time()
    err = None
    try:
        result = await runner.run_game(agents=agents, decks=decks)
        winner_seat = result.winner
        turns = result.turns
        log_lines = result.log if isinstance(result.log, list) else []
    except Exception as e:
        err = str(e)
        winner_seat = None
        turns = 0
        log_lines = []
    elapsed = time.time() - t0

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"game_{game_idx:04d}.log"
    if log_lines:
        log_path.write_text("\n".join(str(l) for l in log_lines), encoding="utf-8")

    return {
        "game": game_idx,
        "format": format,
        "seats": seat_to_agent,
        "decks": seat_to_deck,
        "winner_seat": winner_seat,
        "winner_agent": seat_to_agent.get(winner_seat) if winner_seat else None,
        "turns": turns,
        "elapsed_sec": round(elapsed, 2),
        "log_path": str(log_path),
        "error": err,
    }


# --------------------------------------------------------------------------
# Swiss pairing (1v1)
# --------------------------------------------------------------------------

def swiss_pairings(table: dict[str, Standing], rng: random.Random) -> list[tuple[str, str]]:
    """Greedy Swiss: sort by points desc, pair top with next not-yet-played."""
    sorted_agents = sorted(table.values(),
                           key=lambda s: (-s.points, rng.random()))
    used: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for s in sorted_agents:
        if s.agent in used:
            continue
        partner = None
        for cand in sorted_agents:
            if cand.agent in used or cand.agent == s.agent:
                continue
            if cand.agent in s.opponents:
                continue
            partner = cand.agent
            break
        if partner is None:
            # exhausted — allow rematch with closest-score available
            for cand in sorted_agents:
                if cand.agent in used or cand.agent == s.agent:
                    continue
                partner = cand.agent
                break
        if partner is None:
            continue  # odd-bye for this agent (no opponent)
        pairs.append((s.agent, partner))
        used.add(s.agent); used.add(partner)
    return pairs


# --------------------------------------------------------------------------
# Pod reseating
# --------------------------------------------------------------------------

def pod_seating(agents: list[str], round_idx: int, rng: random.Random) -> list[str]:
    """Random shuffle each round so seat positions vary."""
    seating = list(agents[:4])
    rng.shuffle(seating)
    return seating


# --------------------------------------------------------------------------
# Tournament drivers
# --------------------------------------------------------------------------

async def run_swiss(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed if args.seed is not None else 0)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"

    deck_paths = [Path(d) for d in args.decks]
    if len(deck_paths) < 2:
        raise SystemExit("[error] swiss needs >= 2 decks")
    deck_a, deck_b = deck_paths[0], deck_paths[1]

    table: dict[str, Standing] = {a: Standing(agent=a) for a in args.agents}
    rounds_csv = out_dir / "rounds.csv"
    with rounds_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "game", "agent_a", "agent_b", "winner_agent",
                    "turns", "elapsed_sec", "log_path", "error"])

    game_idx = 0
    for round_idx in range(1, args.rounds + 1):
        pairings = swiss_pairings(table, rng)
        print(f"\n=== Round {round_idx}/{args.rounds} — {len(pairings)} matches ===")
        for a, b in pairings:
            game_idx += 1
            print(f"  R{round_idx} G{game_idx}: {a} vs {b}")
            rec = await run_game(
                agent_names=[a, b],
                deck_paths=[deck_a, deck_b],
                format=args.format, max_turns=args.max_turns,
                seed=(args.seed + game_idx) if args.seed is not None else None,
                log_dir=log_dir, game_idx=game_idx,
            )
            if rec["error"]:
                print(f"    !! error: {rec['error']}")
                table[a].record("draw", [b])
                table[b].record("draw", [a])
            elif rec["winner_agent"] == a:
                table[a].record("win", [b]); table[b].record("loss", [a])
                print(f"    -> {a} wins ({rec['turns']}t, {rec['elapsed_sec']}s)")
            elif rec["winner_agent"] == b:
                table[b].record("win", [a]); table[a].record("loss", [b])
                print(f"    -> {b} wins ({rec['turns']}t, {rec['elapsed_sec']}s)")
            else:
                table[a].record("draw", [b]); table[b].record("draw", [a])
                print(f"    -> draw ({rec['turns']}t, {rec['elapsed_sec']}s)")

            with rounds_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    round_idx, game_idx, a, b, rec["winner_agent"] or "",
                    rec["turns"], rec["elapsed_sec"], rec["log_path"],
                    rec.get("error") or "",
                ])

    # Final standings
    standings = sorted(
        table.values(),
        key=lambda s: (-s.points, -buchholz(s, table), -s.wins, s.agent),
    )
    standings_csv = out_dir / "standings.csv"
    with standings_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "agent", "points", "wins", "draws", "losses",
                    "games", "buchholz"])
        for rank, s in enumerate(standings, start=1):
            w.writerow([rank, s.agent, s.points, s.wins, s.draws, s.losses,
                        s.games, round(buchholz(s, table), 2)])

    summary = {
        "mode": "swiss", "format": args.format,
        "rounds": args.rounds, "agents": args.agents,
        "decks": [str(p) for p in deck_paths],
        "standings": [
            {"rank": i + 1, "agent": s.agent, "points": s.points,
             "wins": s.wins, "draws": s.draws, "losses": s.losses,
             "games": s.games, "buchholz": round(buchholz(s, table), 2)}
            for i, s in enumerate(standings)
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "="*72)
    print("FINAL STANDINGS")
    print("="*72)
    for rank, s in enumerate(standings, start=1):
        print(f"  {rank}. {s.agent:<20}  pts={s.points:5.1f}  "
              f"W{s.wins}-D{s.draws}-L{s.losses}  buchholz={buchholz(s, table):.1f}")
    print(f"\n  rounds.csv     -> {rounds_csv}")
    print(f"  standings.csv  -> {standings_csv}")
    print(f"  summary.json   -> {out_dir / 'summary.json'}")


async def run_pod_tournament(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed if args.seed is not None else 0)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"

    if len(args.agents) < 4:
        raise SystemExit("[error] pod tournament needs 4 agents")
    deck_paths = [Path(d) for d in args.decks][:4]
    if len(deck_paths) < 4:
        raise SystemExit("[error] pod tournament needs 4 decks")

    table: dict[str, Standing] = {a: Standing(agent=a) for a in args.agents}
    rounds_csv = out_dir / "rounds.csv"
    with rounds_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "game", "seating", "winner_agent",
                    "turns", "elapsed_sec", "log_path", "error"])

    for round_idx in range(1, args.rounds + 1):
        seating = pod_seating(args.agents, round_idx, rng)
        print(f"\n=== Round {round_idx}/{args.rounds} — pod {seating} ===")
        rec = await run_game(
            agent_names=seating,
            deck_paths=deck_paths,
            format=args.format, max_turns=args.max_turns,
            seed=(args.seed + round_idx) if args.seed is not None else None,
            log_dir=log_dir, game_idx=round_idx,
        )
        winner = rec["winner_agent"]
        if rec["error"]:
            print(f"  !! error: {rec['error']}")
            for a in seating:
                table[a].record("draw", [x for x in seating if x != a])
        else:
            for a in seating:
                opps = [x for x in seating if x != a]
                if a == winner:
                    table[a].record("win", opps)
                else:
                    table[a].record("loss", opps)
            print(f"  -> winner={winner} in {rec['turns']}t, {rec['elapsed_sec']}s")

        with rounds_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                round_idx, round_idx, json.dumps(seating), winner or "",
                rec["turns"], rec["elapsed_sec"], rec["log_path"],
                rec.get("error") or "",
            ])

    standings = sorted(
        table.values(),
        key=lambda s: (-s.points, -s.wins, s.agent),
    )
    standings_csv = out_dir / "standings.csv"
    with standings_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "agent", "points", "wins", "losses", "games"])
        for rank, s in enumerate(standings, start=1):
            w.writerow([rank, s.agent, s.points, s.wins, s.losses, s.games])

    summary = {
        "mode": "pod", "format": args.format,
        "rounds": args.rounds, "agents": args.agents,
        "decks": [str(p) for p in deck_paths],
        "standings": [
            {"rank": i + 1, "agent": s.agent, "points": s.points,
             "wins": s.wins, "losses": s.losses, "games": s.games}
            for i, s in enumerate(standings)
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "="*72)
    print("FINAL STANDINGS")
    print("="*72)
    for rank, s in enumerate(standings, start=1):
        print(f"  {rank}. {s.agent:<20}  pts={s.points:5.1f}  W{s.wins}-L{s.losses}")
    print(f"\n  rounds.csv     -> {rounds_csv}")
    print(f"  standings.csv  -> {standings_csv}")
    print(f"  summary.json   -> {out_dir / 'summary.json'}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["swiss", "pod"], required=True)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--agents", nargs="+", required=True,
                   help=f"Names from registry: {', '.join(list_agents())}")
    p.add_argument("--decks", nargs="+", required=True)
    p.add_argument("--format", choices=["standard", "commander"], default="standard")
    p.add_argument("--max-turns", type=int, default=150,
                   help="Commander games can go long; default 150.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default="runs/tournament",
                   help="Output dir for run.log + CSVs + JSON.")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.seed is not None:
        set_global_seed(args.seed)
    unknown = [a for a in args.agents if a not in list_agents()]
    if unknown:
        raise SystemExit(f"[error] unknown agents: {unknown}. "
                         f"Available: {', '.join(list_agents())}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = out_dir / "run.log"
    run_log = run_log_path.open("w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, run_log)
    sys.stderr = _Tee(sys.__stderr__, run_log)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    print(f"[run_tournament] mode={args.mode} rounds={args.rounds} "
          f"agents={args.agents}")
    print(f"[run_tournament] writing log to {run_log_path}")
    try:
        if args.mode == "swiss":
            asyncio.run(run_swiss(args))
        else:
            asyncio.run(run_pod_tournament(args))
    finally:
        run_log.flush(); run_log.close()


if __name__ == "__main__":
    main()
