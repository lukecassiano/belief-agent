import numpy as np

EPS = 1e-12
LOG3 = float(np.log(3.0))

GOAL_NAMES = {0: "A", 1: "B", 2: "C"}

def normalize(v: np.ndarray) -> np.ndarray:
    """Make a nonnegative vector sum to 1 (a probability distribution)."""
    v = np.clip(v, EPS, None)      # avoid zeros (helps logs/powers)
    return v / v.sum()


def entropy(b: np.ndarray) -> float:
    """Shannon entropy H(b) = -sum p log p ; higher = more uncertain."""
    b = np.clip(b, EPS, 1.0)
    return float(-(b * np.log(b)).sum())


def bayes_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """
    Posterior ∝ Prior * Likelihood (elementwise), then normalized.
    This is the core of belief updating in Bayes filters / HMM filtering.
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


# ---- Language evidence (constraint tokens) ----
# We represent "messages" and "language clues" as small structured tokens.
# ("NOT", x) means: "not goal x"
# ("EITHER", x, y) means: "either goal x or goal y"
#
# We'll encode tokens as tuples:
#   ("NOT", x, -1) or ("EITHER", x, y)

def truthful_language_tokens(true_goal: int):
    """
    "Truthful" tokens are constraints that do NOT exclude the true goal.
    For example, if true_goal=B:
      truthful NOT tokens: not A, not C
      truthful EITHER tokens: either A or B ; either B or C
    """
    tokens = []
    # NOT tokens are truthful if they say NOT something that isn't the true goal
    for x in (0, 1, 2):
        if x != true_goal:
            tokens.append(("NOT", x, -1))
    # EITHER tokens are truthful if they include the true goal
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
    tokens = []
    tokens.append(("NOT", true_goal, -1))
    for a, b in ((0, 1), (0, 2), (1, 2)):
        if true_goal not in (a, b):
            tokens.append(("EITHER", a, b))
    return tokens


def sample_language_token(true_goal: int, rng: np.random.Generator, noise: float = 0.30):
    """
    With prob 1-noise, sample a truthful constraint.
    With prob noise, sample a misleading constraint.
    """
    if rng.random() < (1.0 - noise):
        candidates = truthful_language_tokens(true_goal)
    else:
        candidates = misleading_language_tokens(true_goal)
    return candidates[int(rng.integers(0, len(candidates)))]


def token_to_likelihood(tok, floor: float = 0.05) -> np.ndarray:
    """
    Convert constraint token into likelihood distribution over G.
    This is your "language model" in V1: interpretable and hand-defined.
    """
    kind, x, y = tok
    if kind == "NOT":
        # "not X" => uniform over other two
        m = np.array([(1.0 -  floor) / 2]*3, dtype=float)
        m[x] = floor
        return m
    if kind == "EITHER":
        # "either X or Y" => mass on X and Y
        m = np.array([floor, floor, floor], dtype=float)
        remaining = 1.0 - floor
        m[x] = remaining / 2
        m[y] = remaining / 2
        return m
    raise ValueError(f"Unknown token: {tok}")


def render_token(tok) -> str:
    """Make tokens human-readable (for debugging / demonstration)."""
    kind, x, y = tok
    if kind == "NOT":
        return f"not {GOAL_NAMES[x]}"
    if kind == "EITHER":
        return f"either {GOAL_NAMES[x]} or {GOAL_NAMES[y]}"
    return str(tok)


# ----------------------------
# Communication: belief -> constraint statement
# ----------------------------

def belief_to_message(belief: np.ndarray, decisive: float = 0.85):
    """
    Turn internal belief into a constraint statement:
      - If very confident, send 'NOT lowest-prob goal' (strong constraint)
      - Else, send 'EITHER top2 goals' (weaker constraint)

    This is a way to make communication "language-like" but still parsable.
    """
    top_prob = float(np.max(belief))
    low = int(np.argmin(belief))

    # argsort descending gives indices from highest prob to lowest
    sorted_idx = list(np.argsort(-belief))
    top1, top2 = int(sorted_idx[0]), int(sorted_idx[1])
    a, b = sorted((top1, top2))

    if top_prob > decisive:
        return ("NOT", low, -1)
    else:
        return ("EITHER", a, b)


# ----------------------------
# Agent: belief, uncertainty, fusion
# ----------------------------

class Agent:
    def __init__(self, name: str, alpha: float = 1.0):
        self.name = name
        self.belief = np.array([1/3, 1/3, 1/3], dtype=float)  # start uninformative
        self.alpha = alpha  # controls how strongly messages affect you

    def H(self) -> float:
        """Uncertainty = entropy of belief."""
        return entropy(self.belief)

    def precision(self) -> float:
        """
        Map uncertainty to a [0,1] 'precision' (confidence weight).
        High entropy => low precision; low entropy => high precision.
        """
        p = 1.0 - (self.H() / LOG3)
        return float(np.clip(p, 0.0, 1.0))

    def update_private(self, likelihood: np.ndarray):
        """Update belief using private evidence."""
        self.belief = bayes_update(self.belief, likelihood)

    def fuse_message(self, msg_likelihood: np.ndarray, sender_precision: float):
        """
        Fuse received message as additional evidence, weighted by sender precision.

        b_new ∝ b_old * (msg_likelihood)^(alpha * sender_precision)

        Why exponent?
        - If sender_precision=0, exponent≈0 -> msg contributes ~1 (no effect)
        - If sender_precision high, msg likelihood is sharpened and matters more
        """
        w = min(self.alpha * float(sender_precision), 1.5)
        if w <= 1e-9:
            return
        weighted = np.power(np.clip(msg_likelihood, EPS, None), w)
        self.belief = normalize(self.belief * weighted)


# ----------------------------
# Simulation
# ----------------------------

def run_episode(rng: np.random.Generator, steps: int = 6, noise: float = 0.30, verbose: bool = True):
    true_goal = sample_true_goal(rng)

    S = Agent("Sensor", alpha=2.0)
    L = Agent("Language", alpha=2.0)

    if verbose:
        print("=" * 60)
        print(f"TRUE GOAL: {GOAL_NAMES[true_goal]}")
        print()

    for t in range(1, steps + 1):
        # 1) Each agent receives private evidence and updates its belief.

        obs = sample_sensor_obs(true_goal, rng, noise=noise)
        S.update_private(sensor_likelihood(obs, correct_prob=0.70))

        clue = sample_language_token(true_goal, rng, noise=noise)
        L.update_private(token_to_likelihood(clue))

        # 2) Each agent sends a constraint message derived from its belief.
        msg_S = belief_to_message(S.belief)
        msg_L = belief_to_message(L.belief)

        # 3) Each agent parses the other's message into a likelihood,
        #    then fuses it using precision weighting.
        like_from_S = token_to_likelihood(msg_S)
        like_from_L = token_to_likelihood(msg_L)

        
        # S.fuse_message(like_from_L, sender_precision=L.precision())
        L.fuse_message(like_from_S, sender_precision=S.precision())

        if verbose:
            print(f"Step {t}")
            print(f"  Sensor obs: points to {GOAL_NAMES[obs]}")
            print(f"  Lang clue : {render_token(clue)}")
            print(f"  S belief  : {S.belief.round(3)}  H={S.H():.3f}  msg='{render_token(msg_S)}'  prec={S.precision():.2f}")
            print(f"  L belief  : {L.belief.round(3)}  H={L.H():.3f}  msg='{render_token(msg_L)}'  prec={L.precision():.2f}")
            print()

    pred_S = int(np.argmax(S.belief))
    pred_L = int(np.argmax(L.belief))

    return {
        "true": true_goal,
        "pred_S": pred_S,
        "pred_L": pred_L,
        "agree": pred_S == pred_L,
        "both_correct": (pred_S == true_goal) and (pred_L == true_goal),
        "S_correct": pred_S == true_goal,
        "L_correct": pred_L == true_goal,
    }


def run_many(seed: int = 0, episodes: int = 500, steps: int = 6, noise: float = 0.30):
    rng = np.random.default_rng(seed)
    stats = {"S_correct": 0, "L_correct": 0, "both_correct": 0, "agree": 0}

    for _ in range(episodes):
        out = run_episode(rng, steps=steps, noise=noise, verbose=False)
        stats["S_correct"] += int(out["S_correct"])
        stats["L_correct"] += int(out["L_correct"])
        stats["both_correct"] += int(out["both_correct"])
        stats["agree"] += int(out["agree"])

    for k in stats:
        stats[k] /= episodes
    return stats


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    run_episode(rng, steps=6, noise=0.30, verbose=True)

    stats = run_many(seed=1, episodes=1000, steps=6, noise=0.30)
    print("Aggregate over 1000 episodes:", stats)

# if __name__ == "__main__":
    for nz in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        stats = run_many(seed=1, episodes=1000, steps=6, noise=float(nz))
        print(f"noise={nz:.2f}  {stats}")
