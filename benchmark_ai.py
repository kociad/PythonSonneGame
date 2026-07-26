#!/usr/bin/env python3
"""Headless tournament runner for evaluating PythonSonne AI difficulties."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = Path(__file__).resolve().parent
# Asset paths in the game are relative.  This also makes launching the script
# by double-clicking it or from a different working directory reliable.
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from models.game_session import GameSession
from utils.settings_manager import settings_manager


DIFFICULTIES = ("EASY", "NORMAL", "HARD", "EXPERT")


@dataclass(frozen=True)
class GameResult:
    game: int
    seed: int
    player_1: str
    player_2: str
    score_1: int
    score_2: int
    winner: str
    duration_seconds: float


def play_game(first: str, second: str, seed: int, game_number: int,
              progress: Callable[[int, int], None] | None = None) -> GameResult:
    """Play one complete two-player game and return its final result."""
    random.seed(seed)
    names = [f"AI_{first}_1", f"AI_{second}_2"]
    started = time.perf_counter()
    session = GameSession(names)
    initial_deck_size = len(session.get_cards_deck())
    last_reported_turn = 0

    while not session.get_game_over():
        player = session.get_current_player()
        before = session.get_turn_state_token()
        session.play_ai_turn(player)
        worker = getattr(player, "_worker_thread", None)
        if worker is not None:
            worker.join()
            session.play_ai_turn(player)
        if session.get_turn_state_token() == before and not session.get_game_over():
            raise RuntimeError(f"AI {player.get_name()} did not finish its turn")
        completed_turns = initial_deck_size - len(session.get_cards_deck())
        if (progress and completed_turns > last_reported_turn and
                (completed_turns >= last_reported_turn + 10 or
                 session.get_game_over())):
            last_reported_turn = completed_turns
            progress(completed_turns, initial_deck_size)

    scores = [player.get_score() for player in session.get_players()]
    winner = "DRAW" if scores[0] == scores[1] else (first if scores[0] > scores[1] else second)
    return GameResult(game_number, seed, first, second, scores[0], scores[1],
                      winner, time.perf_counter() - started)


def create_summary(results: list[GameResult]) -> list[dict[str, object]]:
    """Aggregate games into one statistical row per difficulty."""
    rows = []
    for difficulty in DIFFICULTIES:
        scores, wins, draws = [], 0, 0
        for result in results:
            if result.player_1 == difficulty:
                scores.append(result.score_1)
            if result.player_2 == difficulty:
                scores.append(result.score_2)
            wins += result.winner == difficulty
            draws += result.winner == "DRAW" and difficulty in (result.player_1, result.player_2)
        if scores:
            rows.append({
                "difficulty": difficulty,
                "games": len(scores),
                "wins": wins,
                "draws": draws,
                "losses": len(scores) - wins - draws,
                "win_rate_percent": round(100 * wins / len(scores), 2),
                "average_score": round(statistics.mean(scores), 2),
                "score_stddev": round(statistics.pstdev(scores), 2),
            })
    return rows


def write_report(output: Path, results: list[GameResult], summary: list[dict[str, object]],
                 arguments: dict[str, object]) -> None:
    """Write machine-readable JSON plus a CSV containing every game."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"configuration": arguments,
                                  "summary": summary,
                                  "games": [asdict(result) for result in results]},
                                 indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GameResult.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a headless round-robin AI tournament.")
    parser.add_argument("--games", type=int, default=10,
                        help="games per unordered difficulty pairing (default: 10)")
    parser.add_argument("--seed", type=int, default=2026, help="base random seed")
    parser.add_argument("--output", type=Path, default=Path("reports/ai_benchmark.json"))
    parser.add_argument("--simple", action="store_true",
                        help="use random-placement AI instead of strategic simulation")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    settings_manager.set("AI_USE_SIMULATION", not args.simple, temporary=True)
    results: list[GameResult] = []
    pairings = list(itertools.combinations(DIFFICULTIES, 2))
    total = len(pairings) * args.games
    mode = "jednoduchá AI" if args.simple else "strategická AI"
    print(f"Spouštím AI benchmark: {total} her, režim {mode}, seed {args.seed}.",
          flush=True)
    print("První výpočet může chvíli trvat; průběh se vypisuje po 10 tazích.",
          flush=True)
    for index, (left, right) in enumerate(pairing for pair in pairings
                                          for pairing in [pair] * args.games):
        # Alternate seats to reduce first/second-player bias.
        first, second = (left, right) if index % 2 == 0 else (right, left)
        game_started = time.perf_counter()
        print(f"\n[{index + 1:>3}/{total}] Start: {first} vs. {second}",
              flush=True)

        def show_progress(turns: int, deck_size: int) -> None:
            elapsed = time.perf_counter() - game_started
            print(f"          průběh: {turns}/{deck_size} tahů "
                  f"({elapsed:.1f} s)", flush=True)

        result = play_game(first, second, args.seed + index, index + 1,
                           show_progress)
        results.append(result)
        print(f"          výsledek: {first} {result.score_1} : "
              f"{result.score_2} {second} ({result.winner})", flush=True)

    summary = create_summary(results)
    write_report(args.output, results, summary, vars(args) | {"output": str(args.output)})
    print("\nDifficulty  Games  Wins  Draws  Losses  Win %   Avg score  Std dev")
    for row in summary:
        print(f"{row['difficulty']:<11}{row['games']:>5}{row['wins']:>6}{row['draws']:>7}"
              f"{row['losses']:>8}{row['win_rate_percent']:>8.2f}"
              f"{row['average_score']:>11.2f}{row['score_stddev']:>9.2f}")
    print(f"\nHotovo.\nJSON: {args.output}\nCSV:  {args.output.with_suffix('.csv')}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
