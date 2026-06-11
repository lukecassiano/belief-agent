"""
tests/test_regression.py — pin current behavior before any refactor.

Three tiers, stacked tightest → loosest:

  1. test_single_episode_pinned     — exact per-step beliefs (atol=1e-3).
                                      Catches any math change instantly.
  2. test_aggregate_stats_pinned    — 1000-episode aggregate stats.
                                      Catches statistical drift even when
                                      no single step is obviously broken.
  3. test_zero_noise_perfect        — sanity: at noise=0, both agents
                                      always converge to truth. If this
                                      ever fails, the bug is independent
                                      of any refactor.

WORKFLOW
────────
1. BEFORE you start Session 1 of the refactor, run this test once and
   fix any TODO values to match what your local code actually produces:
       pytest tests/test_regression.py -v
   If a value in EXPECTED_STEP_BELIEFS or EXPECTED_AGGREGATE doesn't
   match your run, replace it. Commit. This commit is your oracle.

2. After EVERY Claude Code session, run the same command. If the tests
   pass, the math survived the session. If a test fails, the smallest
   failing assertion tells you exactly which step / metric drifted.

3. If you INTENTIONALLY change the math (fixing a bug, swapping the
   precision formula, etc.), update EXPECTED values in a deliberate
   commit with a message like "regression: update after fixing X".
   The point is that math changes are visible in git history.

IMPORTS
───────
This test targets the CURRENT flat layout (`from simulation import ...`).
After the Session 1 refactor, update the import to
`from belief_agent.scenarios.runner import ...` (or wherever you land).
The test bodies should NOT need to change — that's the whole point.
"""

from __future__ import annotations

import numpy as np
import pytest

# Update this import after the Session 1 refactor.
from simulation import run_episode_history, run_many_mode


# ════════════════════════════════════════════════════════════
# Tier 1 — deterministic single episode
# ════════════════════════════════════════════════════════════

SINGLE_EPISODE_SEED = 42
SINGLE_EPISODE_STEPS = 6
SINGLE_EPISODE_NOISE = 0.30
SINGLE_EPISODE_MODE = "unidirectional_S_to_L"   # TODO: confirm this matches your local default

# Reference values from your pre-refactor output. If your local run
# produces different numbers, REPLACE these with the actual values
# from your run — do not modify the math to match these.
EXPECTED_STEP_BELIEFS: list[tuple[int, list[float], list[float]]] = [
    # (step_index_0_based, S_belief, L_belief)
    (0, [0.700, 0.150, 0.150], [0.074, 0.703, 0.223]),
    (1, [0.452, 0.452, 0.097], [0.094, 0.890, 0.016]),
    (2, [0.333, 0.333, 0.333], [0.461, 0.461, 0.078]),
    (3, [0.700, 0.150, 0.150], [0.499, 0.499, 0.003]),
    (4, [0.452, 0.097, 0.452], [0.164, 0.827, 0.009]),
    (5, [0.794, 0.036, 0.170], [0.127, 0.808, 0.065]),
]

PER_STEP_ATOL = 0.001   # 3 decimal places matches what you print


def test_single_episode_pinned():
    """
    Tightest test: exact belief arrays at each step.

    Catches every kind of math change — sign flips in fuse_message,
    off-by-one in entropy normalization, dropped floor parameter, etc.
    """
    rng = np.random.default_rng(SINGLE_EPISODE_SEED)
    history, summary = run_episode_history(
        rng=rng,
        steps=SINGLE_EPISODE_STEPS,
        noise=SINGLE_EPISODE_NOISE,
        mode=SINGLE_EPISODE_MODE,
    )

    assert len(history) == SINGLE_EPISODE_STEPS, (
        f"expected {SINGLE_EPISODE_STEPS} steps, got {len(history)}"
    )

    for step_idx, expected_S, expected_L in EXPECTED_STEP_BELIEFS:
        step = history[step_idx]
        np.testing.assert_allclose(
            step["S_belief"], expected_S, atol=PER_STEP_ATOL,
            err_msg=f"\nS_belief drift at step {step_idx + 1}\n"
                    f"  got:      {np.round(step['S_belief'], 3).tolist()}\n"
                    f"  expected: {expected_S}",
        )
        np.testing.assert_allclose(
            step["L_belief"], expected_L, atol=PER_STEP_ATOL,
            err_msg=f"\nL_belief drift at step {step_idx + 1}\n"
                    f"  got:      {np.round(step['L_belief'], 3).tolist()}\n"
                    f"  expected: {expected_L}",
        )


# ════════════════════════════════════════════════════════════
# Tier 2 — aggregate stats over 1000 episodes
# ════════════════════════════════════════════════════════════

AGGREGATE_SEED = 1
AGGREGATE_EPISODES = 1000
AGGREGATE_STEPS = 6
AGGREGATE_NOISE = 0.30
AGGREGATE_MODE = "unidirectional_S_to_L"   # TODO: confirm

# From your output. Replace with actual values from your run if different.
EXPECTED_AGGREGATE: dict[str, float] = {
    "S_correct":    0.914,
    "L_correct":    0.661,
    "both_correct": 0.621,
    "agree":        0.658,
}
AGGREGATE_ATOL = 0.005   # allow ±0.5% drift; flag anything larger


def test_aggregate_stats_pinned():
    """
    Statistical regression: aggregate metrics over 1000 episodes.

    Slightly looser than the per-step test (atol=0.005 vs 0.001) because
    floating-point reduction order can drift by ~0.1% across refactors
    even when the math is unchanged. Anything beyond 0.5% is a real
    regression, not noise.
    """
    stats = run_many_mode(
        seed=AGGREGATE_SEED,
        episodes=AGGREGATE_EPISODES,
        steps=AGGREGATE_STEPS,
        noise=AGGREGATE_NOISE,
        mode=AGGREGATE_MODE,
    )
    for key, expected in EXPECTED_AGGREGATE.items():
        assert key in stats, f"missing metric: {key}"
        diff = abs(stats[key] - expected)
        assert diff < AGGREGATE_ATOL, (
            f"{key}: got {stats[key]:.4f}, expected {expected:.4f} "
            f"(diff {diff:.4f} > tol {AGGREGATE_ATOL})"
        )


# ════════════════════════════════════════════════════════════
# Tier 3 — zero-noise sanity check
# ════════════════════════════════════════════════════════════

@pytest.mark.parametrize("mode", ["none", "bidirectional", "unidirectional_S_to_L"])
def test_zero_noise_perfect(mode: str):
    """
    Sanity: at noise=0, every observation is truthful. Both agents
    should converge to the true goal on 100% of episodes regardless
    of communication mode. If this fails, there's a bug in the
    likelihood model or the Bayes update, independent of any refactor.
    """
    stats = run_many_mode(
        seed=1,
        episodes=100,
        steps=6,
        noise=0.0,
        mode=mode,
    )
    assert stats["S_correct"] == pytest.approx(1.0, abs=1e-6), (
        f"sensor failed at zero noise (mode={mode}): {stats['S_correct']}"
    )
    assert stats["L_correct"] == pytest.approx(1.0, abs=1e-6), (
        f"language failed at zero noise (mode={mode}): {stats['L_correct']}"
    )
    assert stats["both_correct"] == pytest.approx(1.0, abs=1e-6), (
        f"both_correct < 1 at zero noise (mode={mode}): {stats['both_correct']}"
    )