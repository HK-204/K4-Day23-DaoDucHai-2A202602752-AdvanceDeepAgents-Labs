# A Comprehensive Survey of World Models: Cognitive Foundations, Modern Neural Architectures, and Open Challenges

## TL;DR
- World models trace their intellectual roots to Kenneth Craik's 1943 cognitive hypothesis that internal small-scale simulations enable organisms to test actions safely before real-world execution [1].
- Modern deep world models divide agents into perception, memory/dynamics, and control components, allowing reinforcement learning policies to train inside learned latent "dreams" [2][3].
- The field has transitioned from pixel-level UNet diffusion architectures (such as Sora and DIAMOND) toward scalable Diffusion Transformers (DiT) and non-generative Joint Embedding Predictive Architectures (JEPAs) like V-JEPA 2 [4][5].
- Despite successes in robotics and autonomous driving simulation, world models remain bottlenecked by compounding autoregressive errors, low-density state-action hallucinations, and kinematic-versus-dynamic representation gaps [6][7][8].

## Background
The paradigm of **world models** in artificial intelligence refers to learned internal representations of environment dynamics, state transitions, and physical laws that enable an agent or system to simulate future scenarios, predict outcomes, and perform model-based planning or policy optimization. 

The conceptual foundation of world models originates in cognitive science and epistemology. In 1943, British psychologist Kenneth Craik hypothesized in *The Nature of Explanation* that the human brain constructs an internal "small-scale model" of external reality [1]. According to Craik, reasoning and thought operate by translating external processes into internal symbols, running internal causal simulations, and retranslating the simulation outcomes into prospective actions. This cognitive paradigm was anticipated by 19th-century pioneers such as Hermann von Helmholtz and Ernst Mach, who posited that perception is a process of unconscious inference and that memory-saving internal representations economize cognitive labor [9].

In artificial intelligence and reinforcement learning, early computational formulations emerged through recurrent neural network architectures. Jürgen Schmidhuber pioneered differentiable recurrent world models and Controller-Model ($C$-$M$) architectures in the 1990s and 2015, demonstrating how recurrent models can predict environment transitions and alleviate credit assignment bottlenecks [3]. This lineage culminated in modern deep world models such as Ha and Schmidhuber's (2018) World Models framework, which combines a Variational Autoencoder (VAE) vision model, a Mixture Density Network with an LSTM (MDN-RNN) memory model, and an Evolution Strategies (ES) controller [2]. Subsequent model-based breakthroughs, including MuZero and the Dreamer series, extended these principles by learning iterative latent dynamics models for tree search and actor-critic imagination [10].

## Classical Architectures and Model-Based Reinforcement Learning
Classical deep world models established the foundational separation between perception, environment dynamics modeling, and policy control. In the seminal World Models framework, visual inputs $x_t$ are compressed into low-dimensional latent vectors $z_t$ by an unsupervised Convolutional Variational Autoencoder ($V$) [2]. A recurrent neural network combined with a Mixture Density Network ($M$) then models the conditional probability distribution of future latent states $p(z_{t+1} | z_t, a_t)$ [2]. A lightweight controller ($C$) optimizes actions within simulated latent trajectories ("dreams"), regularized by temperature scaling ($\tau$) to prevent the agent from exploiting imperfections in the learned generative model [2].

Building upon latent dynamics, MuZero revolutionized model-based reinforcement learning by eliminating the requirement for pixel reconstruction or known physical rules [10]. Instead, MuZero learns a recurrent latent model trained exclusively on planning-relevant quantities—predicting reward, value, and policy directly from latent states integrated with Monte Carlo Tree Search [10]. Concurrently, the Dreamer series scaled RSSM (Recurrent State-Space Models) latent dynamics to high-dimensional continuous control tasks, enabling agents to imagine hundreds of steps ahead and optimize robust policies purely in imagination.

## Modern Neural Architectures: Video-Generative Models and JEPAs
Between 2022 and 2026, world model architectures experienced a major paradigm shift driven by high-capacity video generators and non-generative joint embedding architectures. 

### Diffusion Transformers and Action-Controllable Simulators
The success of large-scale video generators, such as Sora-like architectures, demonstrated that Diffusion Transformers (DiT) could simulate complex physical motion, multi-object interactions, and high-resolution spatial-temporal dynamics [4]. However, passive video generation lacks action conditioning. To address this, action-controllable generative interactive environments such as Genie (an 11-billion parameter spatiotemporal model) demonstrated that interactive virtual worlds could be synthesized directly from unlabelled internet gameplay and video sequences without explicit action annotations [11].

### The Shift to Joint Embedding Predictive Architectures (JEPA)
A profound methodological shift emerged with Yann LeCun's **Joint Embedding Predictive Architectures** (I-JEPA, V-JEPA, V-JEPA 2, and VJEPA) [12][5][13]. Rather than attempting the intractable and high-entropy task of reconstructing every pixel in pixel space—which forces models to waste capacity predicting unpredictable background details—JEPA architectures operate entirely in abstract latent space [12][5]. By predicting the representations of future or masked video patches using feature-prediction objectives, V-JEPA and V-JEPA 2 achieve significant training efficiency gains (1.5x to 6x faster learning), avoid representation collapse, and provide robust geometric and physical abstractions [12][5]. 

Recent theoretical formulations, such as Variational JEPA (VJEPA) and Bayesian JEPA (BJEPA), extend deterministic JEPAs into probabilistic world models [13]. BJEPA factorizes predictive belief states into learned dynamics and structural priors via a Product of Experts, uniting representation learning with Bayesian filtering and proving that latent embeddings serve as sufficient statistics for optimal control [13].

## Downstream Applications: Robotics, Autonomous Driving, and Gaming
World models serve as versatile predictive engines, virtual testbeds, and simulators across multiple safety-critical domains:

- **Robotics and Manipulation:** In robotics, latent world models facilitate model predictive control (MPC) and imagination-based planning [6]. Recent iterations like V-JEPA 2 combine action-conditioned video prediction with proprioceptive data to enable zero-shot robot manipulation (reaching, grasping, and pick-and-place) in novel physical environments with success rates reaching 65%–80% [5].
- **Autonomous Driving:** Systems such as UniSim, GAIA-1, and DriveWorld generate multi-sensor driving scenarios (camera and LiDAR) to simulate rare, hazardous, and long-horizon traffic edge cases (e.g., sudden pedestrian incursions or complex multi-vehicle interactions) that are difficult or unsafe to collect in real-world logs [7][8].
- **Interactive Gaming and Simulation:** Generative game engines (e.g., Genie, GameNGen, Oasis) synthesize playable, real-time interactive video frames conditioned on player keyboard or controller inputs, offering new paradigms for simulation-based training [11][7].

## Evaluation Frameworks and Benchmarks
Evaluating world models has historically suffered from a "causal gap"—a disconnect between open-loop perceptual fidelity and closed-loop functional utility [8]. 

- **Open-Loop Benchmarks:** Traditional evaluation relies on datasets like nuScenes and Waymo Open Dataset, measuring Fréchet Inception Distance (FID), Fréchet Video Distance (FVD), and Average Displacement Error (ADE) [8]. However, open-loop protocols reset agent states to ground truth at every time step, masking cumulative errors and out-of-distribution drift [8].
- **Closed-Loop & Bridge Benchmarks:** Modern evaluation frameworks (e.g., WorldSimBench, CARLA, Bench2Drive, WorldArena) place generative world models inside closed action loops to measure functional task success, route completion, and composite driving scores [8]. A comprehensive analysis of 160 benchmarks by Jain et al. highlights the critical need for standardized advantage-aware protocols that directly contrast Vision-Language-Action (VLA) policies with world model planning performance [14].

## Trends and Open Problems
Despite rapid advancements, world models face several persistent scientific challenges that define the frontier of current research:

1. **Compounding Autoregressive Errors:** In long-horizon autoregressive rollouts, minor inaccuracies accumulate across time steps, causing trajectories to diverge into physically implausible states. Methods such as Video Retrieval-Augmented Generation (VRAG) with global state conditioning are increasingly explored to stabilize multi-step predictions [7].
2. **Hallucination in Generative World Models:** Generative world models frequently hallucinate objects, actions, or physical laws. Hansen and Wang (2026) classify hallucinations into *perceptual hallucination* (tokenizer mapping failures), *action-marginalization hallucination* (ignoring action conditioning due to skewed training data), and *scene-diverging hallucination* (abrupt structural ruptures) [7]. These phenomena stem from low-density regions of the state-action distribution and require runtime anomaly detection and coverage-aware data collection [7].
3. **Kinematic vs. Dynamic Imagination:** Recent empirical studies reveal that long-horizon world models often simulate environments *kinematically* rather than *dynamically* [7]. When subjected to friction-sweep perturbations or physical regime shifts, world model predictions frequently maintain statistical invariance to physical changes while underlying policy rewards collapse, indicating that current embeddings under-encode causal regime dynamics [7].
4. **Generalization and the Sim-to-Real Gap:** Bridging the gap between synthetic latent imagination and real-world physical deployment remains challenging due to unmodeled contact dynamics, sensor noise, and unobserved environmental factors [6][7].

## References
[1] The Nature of Explanation. web. https://markhuckvale.com/research/hp/Craik_NOE_Chapter_5.pdf (1943)
[2] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018)
[3] Recurrent World Models Facilitate Policy Evolution. web. https://proceedings.nips.cc/paper_files/paper/2018/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf (2018)
[4] The Dawn of Video Generation: Preliminary Explorations with SORA-like Models. hf-search. https://huggingface.co/papers/2410.05227 (2024)
[5] Introducing V-JEPA 2 and Benchmarks for Physical Reasoning. web. https://ai.meta.com/research/vjepa/ (2026)
[6] World Model for Robot Learning: A Comprehensive Survey. arxiv. https://arxiv.org/abs/2605.00080 (2026)
[7] World Models: A Comprehensive Survey. arxiv. https://arxiv.org/abs/2606.00133 (2026)
[8] Latent World Models for Automated Driving: A Unified Taxonomy, Evaluation Framework, and Open Challenges. arxiv. https://arxiv.org/abs/2603.09086 (2026)
[9] The Prehistory of the Idea that Thinking is Modelling. web. https://link.springer.com/article/10.1007/s42087-024-00460-z (2025)
[10] Mastering Atari, Go, chess and shogi by planning with a learned model. web. https://www.nature.com/articles/s41586-020-03051-4 (2020)
[11] Genie: Generative Interactive Environments. hf-search. https://huggingface.co/papers/2402.15391 (2024)
[12] V-JEPA: Video Joint Embedding Predictive Architecture. web. https://ai.fb.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/ (2024)
[13] VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models. arxiv. https://arxiv.org/abs/2601.14354 (2026)
[14] Do World Models Make Better Robots?. arxiv. https://arxiv.org/abs/2609.29669 (2026)
