"""Compute the comparison notebook 04 shows and write it to `data/synthesis.json`.

    uv run python scripts/synthesis.py [--beta 0.1] [--tau 0.25] [--out PATH]

Tunes each algorithm's learning rate on `synthesis.LR_GRID` (seeds 0, 1, 2), then runs the
comparison with the chosen rates (`synthesis.comparison_rates`: published GRPO runs at
the rate chosen for `grpo_ppo_like`). Everything is seeded: two runs write the same file. The
file holds numbers and labels only; progress and the wall time go to stdout."""

import argparse
import time
from pathlib import Path

from proxopt import synthesis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=synthesis.DEFAULT_PATH)
    args = parser.parse_args()

    start = time.perf_counter()
    grid, best = synthesis.tune(args.beta, args.tau)
    for row in grid:
        gaps = ", ".join(f"{g:.4f}" for g in row["final_gaps"])
        print(f"{row['name']:>14}  lr {row['lr']:.0e}  mean {row['mean_final_gap']:.4f}  ({gaps})")
    rates = synthesis.comparison_rates(best)
    print("rule:", best, flush=True)
    print("comparison:", rates, flush=True)
    result = synthesis.compare(args.beta, args.tau, lr=rates)
    synthesis.save(args.out, grid, result)
    print(f"wrote {args.out.name} in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
