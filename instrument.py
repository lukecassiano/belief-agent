"""
instrument.py — read-only instrumentation for the demo.

Replays an episode step-for-step with the exact same RNG consumption and update
order as `simulation.run_episode_history`, but records the intermediate states
the UI needs:

  prior → (private evidence) → private posterior → (message fusion) → posterior

That split lets us attribute every change in an agent's loss to either its own
evidence or to what it was told, which is how information cascades show up.

Nothing here changes the model. `simulation.py` remains the single source of
truth; `check_consistency` asserts the replay matches it.
"""

import numpy as np

from simulation import (
    Agent,
    belief_to_message,
    render_token,
    run_episode_history,
    sample_language_token,
    sample_sensor_obs,
    sample_true_goal,
    sensor_likelihood,
    token_to_likelihood,
)

EPS = 1e-12


def loss(belief, true_goal):
    """Cross-entropy of the belief against the hidden goal: −ln b(true)."""
    return float(-np.log(max(float(belief[true_goal]), EPS)))


def fusion_weight(alpha, sender_precision, w_cap):
    return float(min(alpha * float(sender_precision), w_cap))


def trace_episode(seed, steps=6, noise=0.30, mode="bidirectional", flat_precision=None,
                  decisive=0.85, floor=0.05, alpha=2.0, w_cap=1.5):
    rng = np.random.default_rng(int(seed))
    true_goal = sample_true_goal(rng)
    S = Agent("Sensor", alpha=alpha, w_cap=w_cap)
    L = Agent("Language", alpha=alpha, w_cap=w_cap)

    def prec(a):
        return a.precision() if flat_precision is None else float(flat_precision)

    rows = []
    for t in range(1, int(steps) + 1):
        S_prior, L_prior = S.belief.copy(), L.belief.copy()

        obs = sample_sensor_obs(true_goal, rng, noise=noise)
        S_like = sensor_likelihood(obs, correct_prob=0.70)
        S.update_private(S_like)

        clue = sample_language_token(true_goal, rng, noise=noise)
        L_like = token_to_likelihood(clue, floor=floor)
        L.update_private(L_like)

        S_priv, L_priv = S.belief.copy(), L.belief.copy()
        msg_S = belief_to_message(S.belief, decisive=decisive)
        msg_L = belief_to_message(L.belief, decisive=decisive)
        m_from_S = token_to_likelihood(msg_S, floor=floor)
        m_from_L = token_to_likelihood(msg_L, floor=floor)

        w_S_in = w_L_in = 0.0
        # Same order as simulation.run_episode_history (S fuses first, then L).
        if mode == "bidirectional":
            w_S_in = fusion_weight(alpha, prec(L), w_cap)
            S.fuse_message(m_from_L, sender_precision=prec(L))
            w_L_in = fusion_weight(alpha, prec(S), w_cap)
            L.fuse_message(m_from_S, sender_precision=prec(S))
        elif mode == "unidirectional_S_to_L":
            w_L_in = fusion_weight(alpha, prec(S), w_cap)
            L.fuse_message(m_from_S, sender_precision=prec(S))

        rows.append(dict(
            t=t, obs=int(obs), clue=render_token(clue),
            S_msg=render_token(msg_S), L_msg=render_token(msg_L),
            S_prior=S_prior, S_priv=S_priv, S_post=S.belief.copy(),
            L_prior=L_prior, L_priv=L_priv, L_post=L.belief.copy(),
            S_like=S_like, L_like=L_like,
            S_msg_like_in=m_from_L if mode == "bidirectional" else None,
            L_msg_like_in=m_from_S if mode != "none" else None,
            S_w_in=w_S_in if mode == "bidirectional" else 0.0,
            L_w_in=w_L_in if mode != "none" else 0.0,
            S_precision=S.precision(), L_precision=L.precision(),
        ))
        for k in ("S", "L"):
            r = rows[-1]
            r[f"{k}_loss_prior"] = loss(r[f"{k}_prior"], true_goal)
            r[f"{k}_loss_priv"] = loss(r[f"{k}_priv"], true_goal)
            r[f"{k}_loss_post"] = loss(r[f"{k}_post"], true_goal)
            r[f"{k}_dL_evidence"] = r[f"{k}_loss_priv"] - r[f"{k}_loss_prior"]
            r[f"{k}_dL_message"] = r[f"{k}_loss_post"] - r[f"{k}_loss_priv"]
    return int(true_goal), rows


def check_consistency(seed, steps, noise, mode, flat_precision=None):
    """True if the replay reproduces the engine's beliefs exactly."""
    hist, _ = run_episode_history(np.random.default_rng(int(seed)), steps=int(steps),
                                  noise=float(noise), mode=mode, flat_precision=flat_precision)
    _, rows = trace_episode(seed, steps, noise, mode, flat_precision)
    return all(np.allclose(h["S_belief"], r["S_post"]) and np.allclose(h["L_belief"], r["L_post"])
               for h, r in zip(hist, rows))


# ── Geometry: probability simplex ↔ 2-D triangle ────────────────────────────
# A at the top, B bottom-left, C bottom-right.
CORNERS = np.array([[0.5, np.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0]])


def to_xy(b):
    b = np.asarray(b, dtype=float)
    return b @ CORNERS


def simplex_grid(n=22, margin=0.02):
    """Interior barycentric grid points (for arrows)."""
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            b = np.array([i, j, k], dtype=float) / n
            if b.min() >= margin:
                pts.append(b)
    return np.array(pts)


def update_field(points, likelihood, weight):
    """Where the fusion rule b ← normalize(b · m^w) moves each belief."""
    m = np.power(np.clip(likelihood, EPS, None), weight)
    nxt = points * m
    nxt /= nxt.sum(axis=1, keepdims=True)
    return nxt - points


def loss_surface(true_goal, res=160):
    """−ln b(true) sampled on a cartesian grid, NaN outside the triangle."""
    xs = np.linspace(0, 1, res)
    ys = np.linspace(0, np.sqrt(3) / 2, res)
    X, Y = np.meshgrid(xs, ys)
    # invert barycentric mapping
    a = Y / (np.sqrt(3) / 2)
    c = X - a * 0.5
    b = 1 - a - c
    B = np.stack([a, b, c], axis=-1)
    inside = (B >= -1e-9).all(axis=-1)
    Z = -np.log(np.clip(B[..., true_goal], 1e-3, 1))
    Z[~inside] = np.nan
    return xs, ys, Z
