# A Comprehensive Survey on Large Language Model Agents and Tool Use: Architectures, Advanced Execution, and Open Challenges

## TL;DR
- **Paradigm Shift from Generation to Agency:** Large Language Model (LLM) agents transition static text generation into dynamic, multi-step execution loops where verbal reasoning (thoughts) and external environment actions (tool invocations) synergize to solve complex, multi-hop problems [1].
- **Foundational Integration:** Early architectures like ReAct [1] and Toolformer [2] pioneered interleaved reasoning-acting and self-supervised tool annotation, while modern APIs enforce strict schema compliance via native function calling [3].
- **Advanced Execution & Tool Retrieval:** Scaling tool use to thousands of APIs requires sophisticated execution-centric retrieval strategies (e.g., Lookahead-R [4], state-path menus [5]) and robust container/browser sandboxing (e.g., Thinkingbox [6], ceLLMate [7]).
- **Evaluation & Long-Horizon Challenges:** Modern benchmarks (AgentBench [8], SWE-bench [9], GAIA [10], AgencyBench [11]) reveal that long-horizon tasks—often requiring over 1 million tokens and 90+ tool calls—suffer from multi-step planning drift, context degradation, and tool poisoning vulnerabilities [12].

## Background
Large Language Model (LLM) agents represent autonomous or semi-autonomous software entities powered by foundational language models capable of perceiving environments, maintaining memory, planning action sequences, and invoking external tools (APIs, calculators, web browsers, code interpreters, and databases). Historically, early language models operated in zero-shot or few-shot inference modes, relying entirely on internal parametric knowledge. However, parametric memory is inherently static, prone to factual hallucinations, and incapable of interacting with real-time or proprietary systems.

To bridge this divide, foundational research introduced **ReAct (Synergizing Reasoning and Acting)**, which interleaves verbal reasoning traces (thoughts) with environment actions [1]. By formulating an augmented action space combining natural language thoughts and external API commands, ReAct enables models to decompose complex queries, dynamically adapt to intermediate observations, and correct execution errors. Simultaneously, **Toolformer** demonstrated that LLMs can teach themselves to use tools in a self-supervised manner by annotating corpora with API calls and filtering them using perplexity reduction criteria [2]. At the systems level, **MRKL Systems** established a neuro-symbolic modular architecture, routing natural language sub-queries between LLM routers and discrete symbolic calculators or knowledge bases [13]. Today, these paradigms have been standardized into native API features such as **OpenAI Function Calling and Structured Outputs**, where models enforce strict JSON Schema validation to guarantee syntax and argument correctness during tool execution [3].

## Foundational Paradigms and Architectural Integration
The architecture of LLM agents rests on four primary pillars: reasoning, planning, memory, and tool integration. 

1. **Interleaved Reasoning and Acting (ReAct):** The ReAct framework addresses the brittleness of pure Chain-of-Thought (CoT) reasoning by allowing models to issue thoughts that guide subsequent actions. For instance, in multi-hop question answering (HotpotQA) or interactive web navigation (WebShop), the model generates a thought ("I need to search for the birth year of person X"), executes the tool (`search[person X]`), receives an observation, and reasons over the result before deciding the next step [1]. This closes the feedback loop between model cognition and empirical ground truth.

2. **Self-Supervised and API-Driven Tool Acquisition:** Toolformer established that models do not require massive human-annotated datasets to learn tool integration [2]. By prompting a model to propose API calls at arbitrary token positions and evaluating whether those calls reduce future prediction perplexity, models can learn to invoke calculators, Q&A systems, and translation tools zero-shot [2]. In production environments, this self-supervised capability has been operationalized via standardized JSON Schema function declarations, ensuring that model-generated arguments parse cleanly into typed API payloads [3].

3. **Modular and Neuro-Symbolic Routing (MRKL):** Recognizing that LLMs struggle with precise arithmetic, symbolic logic, and proprietary real-time lookups, MRKL systems decouple language understanding from execution [13]. The LLM acts as an intelligent router, dispatching sub-tasks to specialized neural models or deterministic symbolic workers (e.g., SQL engines, calculators, and knowledge graphs), thereby combining the flexibility of neural networks with the exactness of symbolic computing [13].

## Advanced Tool Use, Retrieval, and Execution Environments
As agentic systems transition from single-API lookups to managing enterprise-grade software ecosystems comprising thousands of tools, several advanced technical challenges emerge:

1. **Execution-Aware Tool Retrieval:** In massive tool libraries, standard semantic retrieval often fails due to the *semantic-functional gap* (i.e., a tool's description may not match the specific syntactic phrasing needed for successful execution). To overcome this, **Lookahead-R** reformulates tool retrieval as a budget-aware, sequential decision-making problem, utilizing an execution-centric surrogate world model to predict downstream task success rather than relying solely on semantic similarity [4]. Similarly, **State-Path Tool Menus** construct pre-execution state paths to surface prerequisite tools and data producers in their correct logical sequence [5]. Comprehensive benchmarks like **ToolBench** further evaluate open-source model alignment across 16,000+ real-world RESTful APIs [14].

2. **Secure Sandboxing and Execution Environments:** Executing untrusted code or allowing agents to browse the web introduces severe security risks, including container escapes, prompt injection, and tool poisoning [15][12]. Realistic environments such as **WebArena** provide self-hosted, Dockerized web applications across e-commerce, forums, and CMS domains, evaluating agent success via programmatic DOM and database state diffs [16]. To safeguard execution, specialized runtimes have been developed: **Thinkingbox** isolates multi-turn stateful business workflows in MCP-compatible tool sessions [6], **ceLLMate** restricts ambient authority at the HTTP layer to mitigate browser agent prompt injection [7], and **MCPTox** benchmarks tool poisoning vulnerabilities across Model Context Protocol servers [12].

3. **Multi-Agent Collaboration & Feedback Loops:** Complex workflows often require multi-agent networks where specialized agents collaborate across shared workspaces (e.g., **WeClawArena** [17]). To prevent runaway token burn and irreversible catastrophic actions during multi-turn rollouts, streaming monitoring frameworks like **OnTrack** utilize structure-aware optimal transport to monitor agent trajectories in real time and intervene before failures cascade [18].

## Evaluation Benchmarks, Real-World Applications, and Behavioral Fingerprints
Evaluating LLM agents requires shifting from static string-matching metrics to dynamic trajectory and goal-state assessment.

1. **Standardized Agent Benchmarks:**
   - **AgentBench:** Evaluates LLMs across diverse interactive environments including operating systems, databases, knowledge graphs, and digital games [8].
   - **SWE-bench / SWE-bench Verified:** Challenges agents to resolve real GitHub issues from popular open-source codebases, requiring code generation, patch application, and passing hidden unit test suites [9].
   - **GAIA (General AI Assistants):** Tests multi-modal general assistant capabilities on multi-step reasoning tasks involving web research, file manipulation, and complex tool chains [10].
   - **$\tau$-bench ($\tau^2$-bench):** Places agents in customer service and transactional environments interacting with simulated users and backend databases under dual-control constraints [19].
   - **AgencyBench:** Pushes evaluation into long-horizon software engineering and MCP workflows, revealing that authentic tasks average **1 million tokens** and **90+ multi-turn tool calls** per scenario [11].

2. **Model Behavioral Archetypes:** Recent empirical analyses across long-horizon benchmarks identify distinct operational profiles among frontier models:
   - *Navigators* (e.g., GLM-4.6) prioritize exhaustive environment exploration by invoking directory-listing tools extensively before modifying state.
   - *Executors* (e.g., GPT-5.2, Claude-4.5-Sonnet) rely on empirical trial-and-error, directly executing shell commands and scripts.
   - *Surgeons vs. Rewriters:* Advanced models act as surgical patchers (modifying precise code blocks), whereas less refined models overwrite entire files, sacrificing token efficiency.

## Trends and Open Problems
Despite rapid advancements, LLM agents and tool-use systems face several critical open challenges:

1. **Long-Horizon Planning Drift and Context Degradation:** As task trajectories exceed 50–100 turns and 1M tokens, agents frequently suffer from attention degradation, hallucinated intermediate states, and cascading error propagation. Once an agent accepts erroneous intermediate data from a tool or web page, it propagates unchecked through the pipeline.
2. **Tool Poisoning and Supply Chain Vulnerabilities:** Integration with decentralized tool ecosystems (such as the Model Context Protocol) exposes agents to malicious tool descriptions, prompt injections hidden in web pages, and unauthorized privilege escalation. Enforcing least-privilege access and verifiable behavioral watermarking remain active areas of research.
3. **Latency, Cost, and Evaluation Bottlenecks:** Comprehensive agentic evaluation is computationally prohibitive, often taking hours and significant cost per run. Developing reliable proxy evaluation frameworks (e.g., PACE) and efficient test-time scaling strategies is essential for scaling agentic applications to enterprise production environments.

## References
[1] ReAct: Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2023-03-10)
[2] Toolformer: Language Models Can Teach Themselves to Use Tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[3] Function calling and other API updates in OpenAI API. web. https://openai.com/index/function-calling-and-other-api-updates/ (2023-06-13)
[4] Lookahead-R: Budget-Aware Tool Retrieval via Execution-Centric Planning. arxiv. https://arxiv.org/abs/2609.35811 (2026-09-20)
[5] The Menu Is an Execution Prior: State-Path Tool Menus for Online Agents. arxiv. https://arxiv.org/abs/2609.09395 (2026-09-08)
[6] Thinkingbox: A Sandbox and Benchmark for Agents in Stateful Business Workflows. hf-search. https://huggingface.co/papers/2608.19741 (2026-08-20)
[7] ceLLMate: Sandboxing Browser AI Agents. hf-search. https://huggingface.co/papers/2512.12594 (2026-01-18)
[8] AgentBench: Evaluating LLMs as Agents. hf-search. https://huggingface.co/papers/2308.03688 (2023-08-07)
[9] SWE-bench: Can Language Models Resolve Real-World GitHub Issues?. arxiv. https://arxiv.org/abs/2310.06770 (2023-10-06)
[10] GAIA: a benchmark for General AI Assistants. arxiv. https://arxiv.org/abs/2311.12983 (2023-11-20)
[11] AgencyBench: Evaluating Long-Horizon Agent Capabilities in Software Development and MCP Workflows. hf-search. https://huggingface.co/papers/2601.11044 (2026-01-16)
[12] MCPTox: A Benchmark for Tool Poisoning Attack on Real-World MCP Servers. hf-search. https://huggingface.co/papers/2508.14925 (2025-08-19)
[13] MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning. arxiv. https://arxiv.org/abs/2205.00445 (2022-05-01)
[14] ToolBench: Large-Scale API Tool Manipulation Benchmark. arxiv. https://arxiv.org/abs/2305.16504 (2023-05-25)
[15] Quantifying Frontier LLM Capabilities for Container Sandbox Escape. hf-search. https://huggingface.co/papers/2603.02277 (2026-03-01)
[16] WebArena: A Realistic Web Environment for Building Autonomous Agents. web. https://webarena.dev/ (2024)
[17] WeClawArena: An Auditable Sandbox and Benchmark for Cross-User Agents Collaboration and Security in Human-Centered Agent Networks. hf-search. https://huggingface.co/papers/2608.03499 (2026-08-04)
[18] OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Optimal Transport. arxiv. https://arxiv.org/abs/2610.12375 (2026-10-08)
[19] tau-bench: A Benchmark for Tool-Use and Interaction in Customer Service Settings. arxiv. https://arxiv.org/abs/2406.12045 (2024-06-17)
