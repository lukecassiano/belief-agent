import numpy as np
from memory import AssociativeMemory, MemoryEntry, fuse_with_memory, StepMemory

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

# ---- Language (constraint) evidence ----

def truthful_language_tokens(true_goal: int):
    tokens = []
    for x in (0, 1, 2):
        if x != true_goal:
            tokens.append(("NOT", x, -1))
    for a, b in ((0, 1), (0, 2), (1, 2)):
        if true_goal in (a, b):
            tokens.append(("EITHER", a, b))
    return tokens


def misleading_language_tokens(true_goal: int):
    tokens = []
    tokens.append(("NOT", true_goal, -1))
    for a, b in ((0, 1), (0, 2), (1, 2)):
        if true_goal not in (a, b):
            tokens.append(("EITHER", a, b))
    return tokens


def sample_language_token(true_goal: int, rng: np.random.Generator, noise: float = 0.30):
    if rng.random() < (1.0 - noise):
        candidates = truthful_language_tokens(true_goal)
    else:
        candidates = misleading_language_tokens(true_goal)
    return candidates[int(rng.integers(0, len(candidates)))]


def token_to_likelihood(tok, floor: float = 0.05) -> np.ndarray:
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
    kind, x, y = tok
    if kind == "NOT":
        return f"not {GOAL_NAMES[x]}"
    if kind == "EITHER":
        return f"either {GOAL_NAMES[x]} or {GOAL_NAMES[y]}"
    return str(tok)


# ----------------------------
# Communication: belief -> constraint statement
# ----------------------------

CANDIDATE_MSGS = [
    ("NOT", 0, -1),
    ("NOT", 1, -1),
    ("NOT", 2, -1),
    ("EITHER", 0, 1),
    ("EITHER", 0, 2),
    ("EITHER", 1, 2),
]

def belief_to_message(belief: np.ndarray, decisive: float = 0.85):
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
        self.memory = AssociativeMemory(max_size=500, conf_floor=0.05)
        self.use_memory = False
        self.mem_lam = 0.25
        self.mem_k = 10
        self.mem_temp = 1.0
        self.mem_beta = 0.6 # probability of following memory vs default
        

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
        """
        w = min(self.alpha * float(sender_precision), 1.5)
        if w <= 1e-9:
            return
        weighted = np.power(np.clip(msg_likelihood, EPS, None), w)
        self.belief = normalize(self.belief * weighted)
        
    def reset_belief(self):
        self.belief = np.array([1/3, 1/3, 1/3], dtype=float)
    
    def choose_message(self, cue_tok: tuple, rng: np.random.Generator):
        """
        Default behavior: belief_to_message(self.belief)
        Memory-biased behavior: retrieve similar past (belief, cue), bias
        toward actions seen there.
        """
        default = belief_to_message(self.belief)

        if not self.use_memory:
            return default
        
        if self.H() < (0.6-0.2 * self.precision()):
            return default
        
        retrieved = self.memory.retrieve_step(self.belief, cue_tok, k=self.mem_k)
        action_probs = self.memory.action_probs_from_retrieved(
            retrieved,
            candidates=CANDIDATE_MSGS,
            temp=self.mem_temp,
        )

        memory_size = len(self.memory.entries)
        beta_eff = self.mem_beta * min(1.0, memory_size / 200.0)
        if rng.random() > beta_eff:
            return default
        
        idx = int(rng.choice(len(CANDIDATE_MSGS), p=action_probs))
        candidate = CANDIDATE_MSGS[idx]

        default_idx = CANDIDATE_MSGS.index(default)
        mem_mass = action_probs[default_idx]

        if mem_mass > 0.55:
            return default

        best = int(np.argmax(self.belief))
        if candidate[0] == "NOT" and candidate[1] == best:
            return default
        return candidate
    
# ----------------------------
# Simulation
# ----------------------------

def run_episode(rng: np.random.Generator, S: Agent, L: Agent, steps: int = 6, noise: float = 0.30, verbose: bool = True, episode_idx=0):
    true_goal = sample_true_goal(rng)

    # S = Agent("Sensor", alpha=2.0)
    # L = Agent("Language", alpha=2.0)

    S.reset_belief()
    L.reset_belief()

    episode_steps = []

    # if L.use_memory:
        # mem_prior = L.memory.goal_prior(L.belief, k=L.mem_k, n_goals=3, temp=L.mem_temp)
        # L.belief = fuse_with_memory(L.belief, mem_prior, lam=L.mem_lam)

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
        msg_S = belief_to_message(S.belief) # Not incorporating memory yet

        belief_at_decision = L.belief.copy()
        msg_L = L.choose_message(clue, rng) # Incorporated memory

        if verbose and L.use_memory:
            retrieved = L.memory.retrieve_step(belief_at_decision, clue, k=3)
            print(" mem top:", [(round(w,3), render_token(e.action), GOAL_NAMES[e.outcome], 
                                 getattr(e, "success", None),) for w,e in retrieved])

        if  L.use_memory:
            episode_steps.append(StepMemory(
                cue=clue,
                action=msg_L,
                belief=belief_at_decision,
                conf=float(L.precision()),
                t=(episode_idx * steps + (t - 1)),
                outcome=true_goal,
                success=0.0,
            ))

        p_before = belief_at_decision[true_goal]

        # 3) Each agent parses the other's message into a likelihood,
        #    then fuses it using precision weighting.
        like_from_S = token_to_likelihood(msg_S)
        like_from_L = token_to_likelihood(msg_L)

        
        # S.fuse_message(like_from_L, sender_precision=L.precision())
        L.fuse_message(like_from_S, sender_precision=S.precision())

        p_after = L.belief[true_goal]
        step_success = 1.0 if (p_after - p_before) > 0 else 0.0
        success = step_success


        if verbose:
            print(f"Step {t}")
            print(f"  Sensor obs: points to {GOAL_NAMES[obs]}")
            print(f"  Lang clue : {render_token(clue)}")
            print(f"  S belief  : {S.belief.round(3)}  H={S.H():.3f}  msg='{render_token(msg_S)}'  prec={S.precision():.2f}")
            print(f"  L belief  : {L.belief.round(3)}  H={L.H():.3f}  msg='{render_token(msg_L)}'  prec={L.precision():.2f}")
            print()

    pred_S = int(np.argmax(S.belief))
    pred_L = int(np.argmax(L.belief))

    ep_success = 1.0 if (pred_L == true_goal) else 0.0
    if L.use_memory:
        for sm in episode_steps:
            sm.success = ep_success
            L.memory.add_step(sm)

    if L.use_memory:
        L.memory.add(MemoryEntry(
            belief=L.belief.copy(),
            outcome=true_goal,
            conf=float(L.precision()),
            t=episode_idx,
        ))

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

    S = Agent("Sensor", alpha=2.0)
    L = Agent("Language", alpha=2.0)

    S.use_memory = False
    L.use_memory = True

    stats = {"S_correct": 0, "L_correct": 0, "both_correct": 0, "agree": 0}

    for ep in range(episodes):
        out = run_episode(rng, S, L, steps=steps, noise=noise, verbose=False, episode_idx=ep)
        stats["S_correct"] += int(out["S_correct"])
        stats["L_correct"] += int(out["L_correct"])
        stats["both_correct"] += int(out["both_correct"])
        stats["agree"] += int(out["agree"])

    print("Memory size(L): entries:", len(L.memory.entries),"step_entries:", len(L.memory.step_entries))

    for k in stats:
        stats[k] /= episodes
    return stats

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    S = Agent("Sensor", alpha=2.0)
    L = Agent("Language", alpha=2.0)
    S.use_memory = False
    L.use_memory = True

    for ep in range(200):
        run_episode(rng, S, L, steps=6, noise=0.30, verbose=False, episode_idx=ep)

    run_episode(rng, S, L, steps=6, noise=0.30, verbose=True, episode_idx=999)

    stats = run_many(seed=1, episodes=1000, steps=6, noise=0.30)
    print("Aggregate:", stats)

# if __name__ == "__main__":
    for nz in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        stats = run_many(seed=1, episodes=1000, steps=6, noise=float(nz))
        print(f"noise={nz:.2f}  {stats}")
