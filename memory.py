from dataclasses import dataclass
import numpy as np

@dataclass

class StepMemory:
    cue: tuple
    action: tuple
    belief: np.ndarray
    conf: float
    t: int
    outcome: int | None = None
    success: float = 0.0

@dataclass

class MemoryEntry:
    belief: np.ndarray
    outcome: int
    conf: float
    t: int

class AssociativeMemory:
    def __init__(self, max_size=500, conf_floor=0.05):
        self.max_size = max_size
        self.conf_floor = conf_floor
        self.entries: list[MemoryEntry] = []
        self.step_entries: list[StepMemory] = []
    def add(self, entry: MemoryEntry):
        if entry.conf < self.conf_floor:
            return
        if len(self.entries) >= self.max_size:
            self.entries.pop(0)
        self.entries.append(entry)
    
    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray, eps=1e-9) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))
    
    def retrieve(self, query_belief: np.ndarray, k=10):
        if not self.entries:
            return []
        
        scored = []
        for e in self.entries:
            sim = self.cosine_sim(query_belief, e.belief)
            w = max(0.0, sim)  * e.conf # weight via (similarity x confidence)
            scored.append((w,e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:min(k, len(scored))]

    @staticmethod
    def prior(retrieved, n_goals=3, temp=1.0, eps=1e-9):
        if not retrieved:
            return np.ones(n_goals) / n_goals  # uniform prior

        goal_scores = np.zeros(n_goals, dtype=float)
        for w, e in retrieved:
            goal_scores[e.outcome] += w
        
        # alternative: boltzmann
        t = max(temp, eps)
        z = goal_scores / t
        z -= np.max(z)
        exp_scores = np.exp(z)
        return exp_scores / (np.sum(exp_scores) + eps)

    def goal_prior(self, query_belief: np.ndarray, k=10, n_goals=3, temp=1.0, eps=1e-9):
        retrieved = self.retrieve(query_belief, k=k)
        return self.prior(retrieved, n_goals=n_goals, temp=temp, eps=eps)
    
    def add_step(self, entry: 'StepMemory'):
        if entry.conf < self.conf_floor:
            return
        if len(self.step_entries) >= self.max_size:
               self.step_entries.pop(0)
        self.step_entries.append(entry)

    @staticmethod
    def cue_sim(c1: tuple, c2: tuple) -> float:
        if c1[0] != c2[0]:
            return 0.0
        if c1[0] == "NOT":
            return 1.0 if c1[1] == c2[1] else 0.0
        if c1[0] == "EITHER":
            return 1.0 if set(c1[1:]) == set(c2[1:]) else 0.0
        return 0.0

    
    def retrieve_step(self, query_belief: np.ndarray, query_cue: tuple, k=10):
        if not self.step_entries:
            return []
        
        scored = []
        for e in self.step_entries:
            sb = max(0.0, self.cosine_sim(query_belief, e.belief))
            sc = self.cue_sim(query_cue, e.cue)
            w = sb * sc * e.conf  # weight via (sim_belief x sim_cue x confidence)
            scored.append((w,e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:min(k, len(scored))]
    
    @staticmethod
    def action_probs_from_retrieved(retrieved, candidates, temp=1.0, eps=1e-9):
        scores = np.zeros(len(candidates), dtype=float)
        for w, e in retrieved:
            try:
                j = candidates.index(e.action)
                s = float(getattr(e, "success", 0.5))
                val = 2.0 * s - 1.0  # success→+1, failure→-1
                scores[j] += w * val
            except ValueError:
                pass
        
        if np.sum(np.abs(scores)) < eps:
            return np.ones(len(candidates)) / len(candidates)  # uniform
        
        t = max(temp, eps)
        z = scores / t
        z -= np.max(z)
        p = np.exp(z)
        return p / (p.sum() + eps)

def fuse_with_memory(prior:np.ndarray, mem: np.ndarray | None,lam=0.25, eps=1e-9):
    if mem is None or lam <= 0:
        return prior
    biased = prior * np.power(np.clip(mem, eps, 1.0), lam)
    biased /= biased.sum()
    return biased