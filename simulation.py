import numpy as np

EPS = 1e-12
LOG3 = float(np.log(3.0))

GOAL_NAMES = {0: "A", 1: "B", 2: "C"}


def normalize(v: np.ndarray) -> np.ndarray:
    """Make a nonnegative vector sum to 1 (a probability distribution)."""
    v = np.clip(v, EPS, None)  # avoid zeros (helps logs/powers)
    return v / v.sum()


def entropy(b: np.ndarray) -> float:
    """Shannon entropy H(b) = -sum p log p ; higher = more uncertain."""
    b = np.clip(b, EPS, 1.0)
    return float(-(b * np.log(b)).sum())


def bayes_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """
    Posterior ∝ Prior * Likelihood (elementwise), then normalized.
    Core belief updating used in Bayes filters / HMM filtering.
    """
    return normalize(prior * likelihood)


# ----------------------------
# Evidence models
# ----------------------------

def sample_true_goal(rng: np.random.Generator) -> int:
    """Pick hidden goal G ∈ {0,1,2} uniformly."""
    return int(rng.integers(0, 3))


# ---- Sensor (beacon) evidence ----
def sample_sensor_obs(true_goal: int, rng: np.random.Generator, noise: float = 0.30) -> int:
    """
    Sensor outputs an observation token in {0,1,2} meaning 'points to A/B/C'.
    With prob 1-noise -> correct.
    With prob noise -> one of the other two (uniform).
    """
    if rng.random() < (1.0 - noise):
        return true_goal
    others = [g for g in (0, 1, 2) if g != true_goal]
    return others[int(rng.integers(0, 2))]


def sensor_likelihood(obs: int, correct_prob: float = 0.70) -> np.ndarray:
    """
    Convert sensor observation into P(obs | G) as a 3-way likelihood over goals.
    If obs == A, likelihood = [0.70, 0.15, 0.15].
    """
    wrong_prob = (1.0 - correct_prob) / 2.0
    m = np.array([wrong_prob, wrong_prob, wrong_prob], dtype=float)
    m[obs] = correct_prob
    return m
# ---- Language (constraint) evidence ----

def truthful_language_tokens(true_goal: int):
    """
    "Truthful" tokens are constraints that do NOT exclude the true goal.
    For example, if true_goal=B:
      truthful NOT tokens: not A, not C
      truthful EITHER tokens: either A or B ; either B or C
    """
    tokens = []
    for x in (0, 1, 2):
        if x != true_goal:
            tokens.append(("NOT", x, -1))
    for a, b in ((0, 1), (0, 2), (1, 2)):
        if true_goal in (a, b):
            tokens.append(("EITHER", a, b))
    return tokens


def misleading_language_tokens(true_goal: int):
    """
    "Misleading" tokens exclude the true goal.
    If true_goal=B:
      misleading NOT token: not B
      misleading EITHER token: either A or C
    """
    tokens = [("NOT", true_goal, -1)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        if true_goal not in (a, b):
            tokens.append(("EITHER", a, b))
    return tokens


def sample_language_token(true_goal: int, rng: np.random.Generator, noise: float = 0.30):
    """
    With prob 1-noise, sample a truthful constraint.
    With prob noise, sample a misleading constraint.
    """
    candidates = truthful_language_tokens(true_goal) if (rng.random() < (1.0 - noise)) else misleading_language_tokens(true_goal)
    return candidates[int(rng.integers(0, len(candidates)))]


def token_to_likelihood(tok, floor: float = 0.05) -> np.ndarray:
    """
    Convert constraint token into likelihood distribution over G.
    This is the "language model" in V1: interpretable and hand-defined.

    floor > 0 prevents hard zeros that would permanently eliminate hypotheses.
    """
    kind, x, y = tok
    floor = float(floor)

    if kind == "NOT":
        m = np.array([(1.0 - floor) / 2] * 3, dtype=float)
        m[x] = floor
        return m

    if kind == "EITHER":
        m = np.array([floor, floor, floor], dtype=float)
        remaining = 1.0 - floor
        m[x] = remaining / 2
        m[y] = remaining / 2
        return m

    raise ValueError(f"Unknown token: {tok}")


def render_token(tok) -> str:
    """Make tokens human-readable (for debugging / UI)."""
    kind, x, y = tok
    if kind == "NOT":
        return f"not {GOAL_NAMES[x]}"
    if kind == "EITHER":
        return f"either {GOAL_NAMES[x]} or {GOAL_NAMES[y]}"
    return str(tok)


def belief_to_message(belief: np.ndarray, decisive: float = 0.85):
    """
    Turn internal belief into a constraint statement:
      - If very confident, send 'NOT lowest-prob goal' (strong constraint)
      - Else, send 'EITHER top2 goals' (weaker constraint)
    """
    top_prob = float(np.max(belief))
    low = int(np.argmin(belief))

    sorted_idx = list(np.argsort(-belief))
    top1, top2 = int(sorted_idx[0]), int(sorted_idx[1])
    a, b = sorted((top1, top2))

    if top_prob > decisive:
        return ("NOT", low, -1)
    return ("EITHER", a, b)



class Agent:
    def __init__(self, name: str, alpha: float = 1.0, w_cap: float = 1.5):
        self.name = name
        self.belief = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        self.alpha = float(alpha)
        self.w_cap = float(w_cap)

    def H(self) -> float:
        return entropy(self.belief)

    def precision(self) -> float:
        """
        Map uncertainty to a [0,1] 'precision' (confidence weight).
        High entropy => low precision; low entropy => high precision.
        """
        p = 1.0 - (self.H() / LOG3)
        return float(np.clip(p, 0.0, 1.0))

    def update_private(self, likelihood: np.ndarray):
        self.belief = bayes_update(self.belief, likelihood)

    def fuse_message(self, msg_likelihood: np.ndarray, sender_precision: float):
        """
        Fuse received message as additional evidence, weighted by sender precision.

        b_new ∝ b_old * (msg_likelihood)^(w)
        where w = min(alpha * sender_precision, w_cap)
        """
        w = self.alpha * float(sender_precision)
        w = min(w, self.w_cap)

        if w <= 1e-9:
            return

        weighted = np.power(np.clip(msg_likelihood, EPS, None), w)
        self.belief = normalize(self.belief * weighted)



def run_episode_history(
    rng: np.random.Generator,
    steps: int = 6,
    noise: float = 0.30,
    mode: str = "bidirectional",            
    flat_precision: float | None = None,    
    decisive: float = 0.85,
    floor: float = 0.05,
    alpha_sensor: float = 2.0,
    alpha_language: float = 2.0,
    w_cap: float = 1.5,
):
    """
    Returns:
      history: list of per-step dicts (for UI)
      summary: end-of-episode metrics
    """
    true_goal = sample_true_goal(rng)

    S = Agent("Sensor", alpha=alpha_sensor, w_cap=w_cap)
    L = Agent("Language", alpha=alpha_language, w_cap=w_cap)

    history = []

    def prec(agent: Agent) -> float:
        return agent.precision() if flat_precision is None else float(flat_precision)

    for t in range(1, steps + 1):
        # private evidence
        obs = sample_sensor_obs(true_goal, rng, noise=noise)
        S.update_private(sensor_likelihood(obs, correct_prob=0.70))

        clue = sample_language_token(true_goal, rng, noise=noise)
        L.update_private(token_to_likelihood(clue, floor=floor))

        # messages
        msg_S = belief_to_message(S.belief, decisive=decisive)
        msg_L = belief_to_message(L.belief, decisive=decisive)

        like_from_S = token_to_likelihood(msg_S, floor=floor)
        like_from_L = token_to_likelihood(msg_L, floor=floor)

        # communication
        if mode == "none":
            pass
        elif mode == "bidirectional":
            S.fuse_message(like_from_L, sender_precision=prec(L))
            L.fuse_message(like_from_S, sender_precision=prec(S))
        elif mode == "unidirectional_S_to_L":
            L.fuse_message(like_from_S, sender_precision=prec(S))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        history.append({
            "t": t,
            "true_goal": int(true_goal),
            "obs": int(obs),
            "clue": render_token(clue),
            "S_belief": S.belief.copy(),
            "L_belief": L.belief.copy(),
            "S_entropy": S.H(),
            "L_entropy": L.H(),
            "S_precision": S.precision(),
            "L_precision": L.precision(),
            "S_msg": render_token(msg_S),
            "L_msg": render_token(msg_L),
        })

    pred_S = int(np.argmax(S.belief))
    pred_L = int(np.argmax(L.belief))

    summary = {
        "true": int(true_goal),
        "pred_S": pred_S,
        "pred_L": pred_L,
        "S_correct": pred_S == true_goal,
        "L_correct": pred_L == true_goal,
        "both_correct": (pred_S == true_goal) and (pred_L == true_goal),
        "agree": pred_S == pred_L,
    }

    return history, summary


def run_many_mode(
    seed: int = 0,
    episodes: int = 1000,
    steps: int = 6,
    noise: float = 0.30,
    mode: str = "bidirectional",
    flat_precision: float | None = None,
    decisive: float = 0.85,
    floor: float = 0.05,
    alpha_sensor: float = 2.0,
    alpha_language: float = 2.0,
    w_cap: float = 1.5,
):
    """
    Aggregate stats over many episodes for a given mode.
    Intended for sweeps / plots (Streamlit).
    """
    rng = np.random.default_rng(seed)
    stats = {"S_correct": 0, "L_correct": 0, "both_correct": 0, "agree": 0}

    for _ in range(episodes):
        _, summary = run_episode_history(
            rng=rng,
            steps=steps,
            noise=noise,
            mode=mode,
            flat_precision=flat_precision,
            decisive=decisive,
            floor=floor,
            alpha_sensor=alpha_sensor,
            alpha_language=alpha_language,
            w_cap=w_cap,
        )
        stats["S_correct"] += int(summary["S_correct"])
        stats["L_correct"] += int(summary["L_correct"])
        stats["both_correct"] += int(summary["both_correct"])
        stats["agree"] += int(summary["agree"])

    for k in stats:
        stats[k] /= episodes
    return stats


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    hist, summ = run_episode_history(rng, steps=6, noise=0.30, mode="unidirectional_S_to_L")
    print("Summary:", summ)
    print("Last step beliefs:", hist[-1]["S_belief"], hist[-1]["L_belief"])
