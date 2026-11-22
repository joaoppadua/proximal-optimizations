"""The numbers notebook 00 quotes in item 3 of "Things to try": the same feedback,
smaller batches.

REINFORCE at its defaults (beta = 0.1, mean baseline, lr 3e-3) with B = 64 for 200 steps
against B = 32 for 400 steps, seeds 0, 1, 2: both use 12,800 reward calls. B = 64 is
evaluated every 640 reward calls (every 10 steps), B = 32 every 320; the comparisons use
the shared grid of every 640 calls (B = 32 read at its matching records), with the gap
averaged over the seeds unless a range is given.

- wobble: a curve's distance from its 3-point moving average, |g_i - (g_{i-1} + g_i +
  g_{i+1}) / 3|, at the interior points of the shared grid, averaged over seeds and
  points; reported from 640 calls on and from 1,920 calls on;
- seed-to-seed range: max - min over the seeds at each grid point, averaged over the grid
  from 640 calls on, and at the last point.

Run with `uv run python scripts/measure_nb00.py` (about 10 seconds).
"""

import time

import numpy as np

from proxopt.algos import reinforce
from proxopt.algos.reinforce import ReinforceConfig
from proxopt.reference import load_reference
from proxopt.world import DEFAULT_WORLD as WORLD

SEEDS = (0, 1, 2)


def span(values, fmt=".4f") -> str:
    return f"{format(min(values), fmt)}–{format(max(values), fmt)}"


def main() -> None:
    start = time.perf_counter()
    ref = load_reference(WORLD)
    big = [reinforce.train(ref, WORLD, ReinforceConfig(seed=s)) for s in SEEDS]
    small = [reinforce.train(ref, WORLD, ReinforceConfig(batch_size=32, steps=400, seed=s))
             for s in SEEDS]
    f_big, f_small = np.array(big[0].feedback), np.array(small[0].feedback)
    g_big = np.array([r.metrics["gap"] for r in big])
    g_small_own = np.array([r.metrics["gap"] for r in small])
    g_small = g_small_own[:, [int(np.where(f_small == v)[0][0]) for v in f_big]]  # shared grid
    print(f"grid spacing: B = 64 every {f_big[1]} calls, B = 32 every {f_small[1]}")

    print("\nmean gap over seeds on the shared grid (B = 32 against B = 64):")
    for calls in (640, 1280, 1920, 2560, 12800):
        j = int(np.where(f_big == calls)[0][0])
        print(f"  {calls:6,} calls: {g_small[:, j].mean():.4f} against {g_big[:, j].mean():.4f}")
    ahead = [int(c) for c, a, b in zip(f_big[1:], g_big.mean(0)[1:], g_small.mean(0)[1:]) if a < b]
    print(f"  B = 64 ahead at {ahead[0]:,} calls and every later point: "
          f"{ahead == [int(c) for c in f_big[1:] if c >= ahead[0]]}")

    for name, g, f in (("B = 64", g_big, f_big), ("B = 32", g_small_own, f_small)):
        first = int(f[np.argmax(g.mean(0) < 0.02)])
        print(f"{name}: mean gap first below 0.02 at {first:,} calls (its own grid)")

    print(f"\nat 12,800 calls: B = 64 {span(g_big[:, -1])}, B = 32 {span(g_small[:, -1])}")

    def wobble(g):  # residual at interior points 1 .. n - 2
        return np.abs(g[:, 1:-1] - (g[:, :-2] + g[:, 1:-1] + g[:, 2:]) / 3)

    interior = f_big[1:-1]
    for lo in (640, 1920):
        k = interior >= lo
        w_big, w_small = wobble(g_big)[:, k].mean(), wobble(g_small)[:, k].mean()
        print(f"wobble from {lo:,} calls on: B = 64 {w_big:.5f}, B = 32 {w_small:.5f}, "
              f"ratio {w_small / w_big:.1f}")

    def seed_range(g):
        return g.max(0) - g.min(0)

    k = f_big >= 640
    print(f"seed-to-seed range, mean over the grid from 640 calls: B = 32 "
          f"{seed_range(g_small)[k].mean():.4f}, B = 64 {seed_range(g_big)[k].mean():.4f}")
    print(f"seed-to-seed range at the last point: B = 32 {seed_range(g_small)[-1]:.4f}, "
          f"B = 64 {seed_range(g_big)[-1]:.4f}")
    print(f"\ntotal {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
