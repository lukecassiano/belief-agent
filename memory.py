from dataclasses import dataclass
import numpy as np

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
        z = goal_scores / temp
        z -= np.max(z)
        exp_scores = np.exp(z)
        return exp_scores / (np.sum(exp_scores) + eps)

    def goal_prior(self, query_belief: np.ndarray, k=10, n_goals=3, temp=1.0, eps=1e-9):
        retrieved = self.retrieve(query_belief, k=k)
        return self.prior(retrieved, n_goals=n_goals, temp=temp, eps=eps)
