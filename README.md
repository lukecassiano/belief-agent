# belief-agent
An explicit uncertainty-aware cognitive agent that models latent belief states and performs inference given incomplete information. 

When AI components communicate under uncertainty, naive information sharing can make systems worse. 
This repository explores and studies when communication helps—and when it causes cascading failure.

## Project Structure

This project is organized as a modular system, with each branch exploring a different component of uncertainty-aware artificial cognition.

- **Branch A — Belief Inference Under Uncertainty**  
  Explicit belief representations, entropy-based uncertainty, and communication between agents.

- **Branch B — Associative Memory**  
  Memory systems that store and retrieve belief–outcome associations with attractor-like dynamics.

- **Branch C — Information Integration Across Time**  
  Temporal stability, recovery from perturbation, and integration vs fragmentation.

- **Branch D — Uncertainty-Weighted Learning**  
  Confidence-modulated updates and comparisons to standard error-driven learning.

- **Branch E — Self-Modeling Agents**  
  Agents that represent uncertainty about their own reliability and adjust behavior accordingly.


In a two-agent inference task with asymmetric evidence quality,
unidirectional communication (Sensor --> Language) improved the
weaker (Lang) agent's accuracy from 0.391 to 0.651, while maintaining the
stronger (Sensor) agent's accuracy at 0.923, which yielded a 0.613 joint
accuracy over 1000 episodes.

Current status:
- Branch A: Belief inference under uncertainty (complete)
- Branch B: Associative memory (in progress)

Details, results, and analysis are documented inline in code and experiment logs.
A full README will be written once all branches are complete.
