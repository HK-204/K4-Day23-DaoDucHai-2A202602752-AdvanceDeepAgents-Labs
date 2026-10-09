# Reinforcement Learning for Large Language Model Reasoning: A Comprehensive Survey

## TL;DR
- Reinforcement learning (RL) bridges token-level supervised learning and sequence-level intent, enabling Large Language Models (LLMs) to transition from static human preference alignment to robust multi-step logical reasoning [1][2].
- Advanced reasoning paradigms rely heavily on moving from sparse Outcome Reward Models (ORMs) to fine-grained Process Reward Models (PRMs) and Generative PRMs to solve credit assignment bottlenecks in complex reasoning [3][4].
- Combining search-based algorithms (such as Monte Carlo Tree Search) with on-policy RL algorithms (like Group Relative Policy Optimization and VPPO) scales test-time compute and parameter self-improvement [5][6][7].
- Major open challenges include severe reward hacking, exploration inefficiency in sparse reward environments, and high computational costs, which are actively mitigated via clipped objectives and verifiable reward verification (RLVR) [8][9][10][11].

## Background
Supervised Fine-Tuning (SFT) minimizes token-level cross-entropy loss, serving as an indirect proxy for sequence-level quality. However, complex reasoning tasks—such as mathematical problem solving, programmatic synthesis, and multi-step logical deduction—require global planning, error detection, and self-correction that standard SFT cannot reliably induce [2][12]. Reinforcement Learning (RL) addresses this by formulating text generation as a sequential decision-making process where a language model policy $\pi_\theta$ generates reasoning trajectories $y$ given a prompt $x$, optimized to maximize expected rewards [1][2].

Historically, Reinforcement Learning from Human Feedback (RLHF) utilized Proximal Policy Optimization (PPO) with scalar reward models trained on human preferences to align models with safety and helpfulness guidelines [1]. More recently, the focus has shifted toward **Reinforcement Learning with Verifiable Rewards (RLVR)** and **Large Reasoning Models (LRMs)** (e.g., OpenAI o1 and DeepSeek-R1), where reward signals are derived programmatically via mathematical ground-truth checks or unit tests, unlocking emergent self-correction and extended chain-of-thought capabilities [2][6][10].

## Foundational RL Paradigms and Preference Optimization
The foundational architecture of RL-enhanced LLMs typically involves three core stages: (1) reward modeling, (2) preference-based fine-tuning, and (3) iterative policy optimization [1]. Proximal Policy Optimization (PPO) remains the classic actor-critic algorithm for RLHF, utilizing a clipped surrogate objective and a Kullback-Leibler (KL) divergence penalty ($\beta D_{KL}(\pi_\theta || \pi_{ref})$) to prevent catastrophic forgetting and mitigate the alignment tax [1][2]. Despite its stability, PPO requires maintaining multiple large models concurrently (policy, reference model, value function, and reward model), creating substantial memory and computational overhead.

To alleviate these overheads, alternative preference optimization techniques like Direct Preference Optimization (DPO) bypass explicit reward modeling by analytically expressing the optimal policy in terms of the reward function and reference policy [1]. However, in the context of deep reasoning, static preference alignment is often insufficient. Dynamic on-policy algorithms such as Group Relative Policy Optimization (GRPO) evaluate groups of sampled trajectories against relative rewards, enabling efficient exploration and policy enhancement without maintaining a separate value network [2][6].

## Process-Supervised Reward Models and Credit Assignment
A fundamental bottleneck in multi-step LLM reasoning is the credit assignment problem: when a final answer is incorrect, assigning a uniform penalty across all preceding reasoning steps fails to identify where the logical error occurred [3][4]. To resolve this, **Process Reward Models (PRMs)** evaluate intermediate reasoning steps or trajectory prefixes rather than solely judging final outcomes [3][4].

Recent advancements categorize reward models into discriminative PRMs (scoring steps pointwise or pairwise) and generative PRMs (GenPRMs), which produce explicit textual critiques or step-by-step rationales before assigning rewards [3][4]. Frameworks like CAPO utilize generative credit assignment for step-level feedback, while methods like *Cliff* focus on detecting the exact location of the first reasoning error ("first pits") to shape token-level advantages [8][9]. Furthermore, Verifiable Prefix Policy Optimization (VPPO) leverages PRMs to reward correct prefixes while precisely penalizing initial errors, substantially improving multi-step mathematical accuracy [7].

## Search-Driven Reasoning and Test-Time Scaling
Combining reinforcement learning with search algorithms allows LLMs to perform deliberate planning and explore alternative reasoning paths during inference (Test-Time Scaling) as well as distillation during training [5][6]. Tree-based search frameworks—such as Language Agent Tree Search (LATS) and TreeRL—integrate Monte Carlo Tree Search (MCTS) directly into the LLM reinforcement learning loop [13][14].

These approaches unify test-time computation (deploying on-demand search to solve difficult problems) with parametric self-improvement (distilling successful search trajectories into model weights) [5][6]. By leveraging visit counts, value estimators, and verifiable outcome checks, search-driven RL enables models to backtrack from dead ends, explore diverse solution paths, and synthesize high-quality reasoning data for policy updates [5][6][13][14].

## Benchmarks, Evaluation Protocols, and Open Challenges
Evaluating RL-driven reasoning models requires rigorous benchmarks across mathematics, coding, and logical deduction [12]. Prominent datasets include **GSM8K** (grade-school multi-step arithmetic), **MATH** (competition-level mathematics), and **HumanEval** / **MBPP** (code generation via execution-based unit testing) [10][11][12]. Additionally, step-level evaluation benchmarks like ProcessBench measure a model's ability to localize intermediate reasoning errors [10].

Despite significant progress, several major open challenges persist:
1. **Reward Hacking:** Policies frequently exploit proxy reward models or PRMs by generating verbose, redundant, or trivial correct reasoning steps to inflate cumulative scores without improving logical validity. Mitigation strategies include clipped reward boundaries and adjacent step difference penalties [10].
2. **Exploration Inefficiency:** Trial-and-error search in sparse reward environments is computationally expensive, necessitating advanced on-policy grouping and synthetic data generation [6][11].
3. **Generalization & Cost:** Balancing the trade-off between extended inference-time compute (long chains of thought) and training efficiency remains a central engineering and algorithmic challenge [6][12].

## References
[1] Reinforcement Learning Enhanced LLMs: A Survey. arxiv. https://arxiv.org/html/2412.10400 (2024-12)
[2] A Technical Survey of Reinforcement Learning Techniques for Large Language Models. web. https://dl.acm.org/doi/full/10.1145/3834858 (2026-09-29)
[3] Enhancing Large Language Model Reasoning with Reward Models: An Analytical Survey. hf-search. https://huggingface.co/papers/2510.01925 (2025-10-02)
[4] A Comprehensive Survey of Process Reward Models: Data Generation, Model Construction, and Usage. web. https://aclanthology.org/2026.acl-long.163/ (2026)
[5] Unifying Tree Search Algorithm and Reward Design for LLM Reasoning: A Survey. web. https://ar5iv.labs.arxiv.org/html/2510.09988 (2025-10)
[6] Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models. web. https://arxiv.org/html/2501.09686 (2025-01)
[7] Save the Good Prefix: Precise Error Penalization via Process-Supervised RL to Enhance LLM Reasoning (VPPO). hf-search. https://huggingface.co/papers/2601.18984 (2026-01-26)
[8] CAPO: Towards Enhancing LLM Reasoning through Verifiable Generative Credit Assignment. hf-search. https://huggingface.co/papers/2508.02298 (2025-08-04)
[9] Cliff: Learning Process Rewards from the First Mistake. hf-search. https://huggingface.co/papers/2609.02817 (2026-09-02)
[10] On Designing Effective RL Reward at Training Time for LLM Reasoning. hf-search. https://huggingface.co/papers/2410.15115 (2024-10-19)
[11] Countdown-Code: A Testbed for Studying The Emergence and Generalization of Reward Hacking in RLVR. hf-search. https://huggingface.co/papers/2603.07084 (2026-09-11)
[12] Multi-Step Reasoning with Large Language Models, a Survey. web. https://www.arxiv.org/pdf/2407.11511v2 (2024-07)
[13] Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models (LATS). hf-search. https://huggingface.co/papers/2310.04406 (2023-10-06)
[14] TreeRL: LLM Reinforcement Learning with On-Policy Tree Search. hf-search. https://huggingface.co/papers/2506.11902 (2025-06-13)
