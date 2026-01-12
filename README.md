# belief-agent
An explicit uncertainty-aware cognitive agent that models latent belief states and performs inference given incomplete information. 

## About

**Uncertainty-Aware Multi-Agent Inference**

This project explores how multiple agents can infer a hidden goal under noisy, ambiguous evidence while explicitly representing **belief**, **uncertainty**, and **trust in communication**.

Inspired by Bayesian cognition, predictive processing, and human collaborative reasoning, the system studies when communication is beneficial, when it isn't, and how confidence-weighted messaging stabilizes inference.

## TL;DR

This project demonstrates that **naive symmetric communication between agents can degrade collective inference under uncertainty**, while **confidence-weighted, asymmetric communication improves group accuracy and prevents misinformation cascades**.

Agents explicitly represent belief distributions, uncertainty (entropy), and trust, allowing failure modes like herding, belief collapse, and overconfidence to be visualized and mitigated.

## Project Structure

This project is organized into a modular branch system:

- **Branch A: Belief Inference Under Uncertainty**  
  Explicit belief representations, entropy-based uncertainty, and communication between agents.

- **Branch B: Associative Memory**  
  Memory systems that store and retrieve belief–outcome associations with attractor-like dynamics.

- **Branch C: Information Integration Across Time**  
  Designed to explore temporal stability, recovery from perturbation, and integration vs fragmentation.

- **Branch D: Uncertainty-Weighted Learning**  
  Explore confidence-modulated updates and comparisons to standard error-driven learning.

- **Branch E: Self-Modeling Agents**  
  Designed to support agents that represent uncertainty about their own reliability and adjust behavior accordingly.

***This current repository is only a demonstration of Branch A, and will be edited to accommodate future progress***

## Branch A — Belief Inference Under Uncertainty

*Each agent maintains a probability distribution over three possible goals: **A, B, or C.***
At every timestep, agents:
1. Receive private evidence (sensor observations or language constraints)
2. Update beliefs via Bayesian inference
3. Communicate a structured message derived from their belief
4. Fuse incoming messages weighted by the sender’s confidence (precision)

Uncertainty is explicitly represented as entropy and modulates agent trust.

This setup also serves as a parallel to human cognition and group reasoning, as it contains:

- Perceptual evidence (sensor agent) vs. Ambiguous testimony (language agent)
- Overconfident early claims cause belief collapse
- Herding and agreement without correctness
- Trust calibration based on confidence and past reliability

In this project, naive symmetric communication is shown to degrade collective performance. This is a failure mode often observed in human groups, social media dynamics, and multi-agent systems.

I evaluated under three main communication types:

• No communication  
• Bidirectional communication (both agents exchange messages)  
• Unidirectional communication (Sensor → Language only)

Each condition is tested across increasing noise levels in private evidence.

### Noise Sweep Results

*Sensor Agent Correctness*

<img width="498" height="386" alt="sensmodelnoisesweep" src="https://github.com/user-attachments/assets/ef56ec8e-549d-4228-8437-aea72c1b05bb" />


*Language Agent Correctness*

<img width="510" height="394" alt="langmodelnoisesweep" src="https://github.com/user-attachments/assets/fcf447cb-2db5-4b0c-bf1e-a663dd43a28d" />


*Simultaneous Correctness*

<img width="498" height="389" alt="bothcorrectnoisesweep" src="https://github.com/user-attachments/assets/9a8a7e2b-b2ae-434f-b779-df8d63200106" />


*Frequency of Agent Agreement on Conclusion*

<img width="486" height="375" alt="agreementnoisesweep" src="https://github.com/user-attachments/assets/5154f848-4189-4b75-b719-f55fa449150a" />

### Key Observations
- Without communication, the sensor agent performs well while the language agent fails under noise.
- Bidirectional communication improves agreement but reduces sensor accuracy at high noise.
- Unidirectional communication preserves sensor performance while substantially improving the weaker agent.
- Agreement is more likely to reflect correctness under unidirectional communication.

### Failure Modes Observed
- Hard logical constraints (such as setting floor probability to zero) can permanently eliminate true hypotheses under noise
- Early misleading messages cause irreversible belief collapse
- Symmetric trust amplifies misinformation when evidence quality is asymmetric

To mitigate this issues, I implemented:
- Soft probabilistic constraints (no hard zeros)
- Confidence-weighted message fusion
- Capped trust influence to prevent runaway updates

### Implemented Visualization
***Here are three test cases labeled by communication mode. Each case shows the movement (tracked by a colored trail) of each agent, as they converge upon a final goal.***

No Communication:

<img width="441" height="476" alt="nocommgrid" src="https://github.com/user-attachments/assets/8d983fb6-1a2b-4b0d-b173-73780ae6e384" />


Bidirectional Communication:

<img width="478" height="510" alt="bidirectionalgrid" src="https://github.com/user-attachments/assets/86b82af7-6020-4260-bb9f-b68a17d03069" />


Unidirectional Communication:

<img width="441" height="473" alt="unidirectionalgrid" src="https://github.com/user-attachments/assets/220ed7bf-fc99-406c-b10d-9f861087a331" />



This offers an immediate visual representation of belief collapse, recovery, and herding dynamics. Agents will literally move toward hypotheses as confidence increases, regardless of accuracy.

### What is Demonstrated by this?

- Bayesian reasoning and uncertainty modeling
- Multi-agent communication design
- Failure analysis of collaborative AI systems
- Interpretable probabilistic representations
- Experimental rigor (ablations, noise sweeps)
- UI-driven debugging and visualization

### Running the Demo

```bash
pip install -r requirements.txt
python -m streamlit run app.py

*Further details, results, and analysis are documented inline in code and experiment logs.
A full README will be written once all branches are complete.*
