# Comprehensive Survey on Video and Multimodal Generation: Architectures, Omni-Modal Alignment, and Evaluation Paradigms

## TL;DR
- Video generation has shifted from factorized 3D U-Net convolutional pipelines to scalable **Diffusion Transformers (DiTs)** operating on compressed spacetime latent patches [1][2].
- Modern **Any-to-Any and Omni-Modal** architectures integrate text, image, audio, and video via unified conditional flow matching and joint latent variable inference [3][4].
- Cross-modal alignment for video-to-audio and audio-visual synthesis increasingly leverages training-free geometric guidance and pre-trained foundation mappers to bypass expensive paired training [5][6].
- Evaluation has evolved from simple distribution distance metrics (FVD) to comprehensive hierarchical suites (VBench) and physical consistency scoring (WCS) [7][8][9].
- Major open challenges remain in mitigating high computational complexity, ensuring compositional safety, and overcoming physical world simulation drift [10][11][9].

## Background
Generative modeling of dynamic visual media extends static image synthesis across the temporal axis $T$, demanding models that capture both spatial fidelity and physical temporal coherence [8]. Early explorations relied on frame-by-frame autoregressive generation or 3D U-Net convolutions with factorized attention (e.g., Make-A-Video, Imagen Video), which frequently suffered from object drift and temporal flickering. 

A foundational turning point arrived with the introduction of **Latent Video VAEs**, which compress video tensors across both spatial and temporal dimensions into compact latent representations [2]. This significantly reduces memory overhead and enables multi-second and minute-long video generation. Following this, **Diffusion Transformers (DiTs)** replaced convolutional U-Nets, tokenizing videos into 3D spacetime cubes and scaling effectively with model size and web-scale datasets like Panda-70M and WebVid-2M [2][12]. Concurrently, omnimodal foundations (such as Gemini) established that joint pre-training across raw audio streams, video frame sequences, and text from inception yields native multimodal understanding and generation [1].

## Architectural Paradigms and Milestone Models
The architectural evolution of video generation spans two primary design generations: cascaded 3D U-Nets and scalable Diffusion Transformers. Make-A-Video demonstrated that text-to-image (T2I) models can be repurposed for video generation via pseudo-3D convolutions and factorized attention without requiring paired text-video data during core training. Similarly, cascaded frameworks like Imagen Video utilized progressive spatial and temporal super-resolution models to synthesize high-definition clips.

With the advent of **Sora**, Diffusion Transformers operating on spacetime latent patches emerged as the dominant paradigm for high-fidelity video simulation [2]. These models leverage variable-duration, variable-resolution joint training to capture complex real-world physics. In the open-weights ecosystem, models such as HunyuanVideo, Mochi-1, and Open-Sora provide researchers with high-capacity DiT backbones (ranging up to 13B+ parameters) for transparent experimentation and fine-tuning [2].

## Any-to-Any Multimodal and Omni-Modal Generation
Moving beyond isolated text-to-video pipelines, recent research focuses on unified any-to-any and omnimodal generation where arbitrary combinations of text, images, audio, and video serve interchangeably as inputs and outputs [3][4]. 

Frameworks like **MUNITE** treat multimodal generation as a conditional latent inference problem under varying subsets of observed evidence, unifying deterministic encoding and marginal generation within a single conditional flow model [3]. Similarly, **NExT-OMNI** leverages discrete flow matching to handle arbitrary interleaved multimodal sequences [4]. Comprehensive evaluation suites such as **UniM** benchmark these systems on complex multi-turn, multi-modality reasoning [13]. In parallel, safety benchmarks like Multi2AV-Safety reveal that conditioning omnimodal models on interacting cross-modal inputs introduces novel compositional safety risks that standard unimodal guardrails fail to catch [10].

## Cross-Modal Alignment and Video-to-Audio Synthesis
A critical frontier in multimodal generation is synthesizing synchronized audio from video (V2A) or achieving joint audio-visual-text alignment without incurring prohibitive paired data collection costs [5][6]. 

Two primary strategies address this challenge:
1. **Mapper-Based Adapters**: Architectures such as MFM-Mapper bridge frozen visual encoders (e.g., CAVP, TimeChat) with audio diffusion models (such as AudioLDM-2 conditioned on AudioMAE features) using lightweight autoregressive GPT-2 mappers [6]. This captures fine-grained semantics and temporal dynamics using a fraction of traditional training epochs [6].
2. **Training-Free Multimodal Guidance (MDG)**: Rather than retraining heavy joint backbones, MDG minimizes the geometric volume of video, audio, and text embeddings in a shared latent space during the audio denoising loop, enforcing temporal and semantic consistency dynamically [5].

## Benchmarks, Evaluation Metrics, and Efficiency Challenges
Evaluating video and multimodal generation requires moving beyond simple pixel-level metrics (such as PSNR or SSIM) to robust distributional and semantic benchmarks [8]. 

- **Datasets**: While early models relied on noisy alt-text web datasets like WebVid-2M [12], modern curation pipelines like **Panda-70M** use multi-teacher cross-modality distillation to pair 70M video clips with high-quality captions, drastically improving zero-shot generation quality [2].
- **Metrics**: Fréchet Video Distance (FVD) remains the standard for distributional video fidelity using 3D I3D backbones [8], while CLIPScore evaluates prompt alignment [14]. To address fine-grained physical realism, the **World Consistency Score (WCS)** evaluates object permanence, causal compliance, and relation stability [9]. Furthermore, hierarchical evaluation suites like **VBench** and **VBench-2.0** decompose video quality into over 16 distinct perceptual and semantic dimensions [7].
- **Efficiency Bottlenecks**: Video generation models remain heavily compute-bound, requiring thousands of FLOPs per frame and exhibiting energy costs orders of magnitude higher than image generation [2][11]. Encoder-free designs and efficient spatial-temporal compression are critical active areas to reduce inference latency and hardware footprints [11].

## Trends and Open Problems
Despite rapid advancements, several critical challenges remain open in video and multimodal generation:
1. **Physical World Simulation & Drift**: While models like Sora approximate physical dynamics, they frequently suffer from long-horizon drift, gravity violations, and object metamorphosis [2][9]. Developing explicit physical inductive biases within DiTs is an open frontier.
2. **Compositional Safety and Alignment**: Omni-modal systems are susceptible to adversarial cross-modal prompt interactions where text and video cues combine to bypass safety filters [10]. Robust cross-modal guardrails are urgently needed.
3. **Data Scarcity and Quality**: High-resolution, copyright-compliant, physically diverse video data remains scarce, driving reliance on synthetic data generation and multi-teacher knowledge distillation pipelines [2].
4. **Real-Time Streaming & Latency**: Current diffusion and flow matching formulations require iterative denoising steps, hindering real-time interactive video synthesis and streaming applications. One-step consistency models and rectified flow acceleration remain vital research directions [4].

## References
[1] Gemini: A Family of Highly Capable Multimodal Models. web. https://arxiv.org/html/2312.11805v3 (2023-12)
[2] Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers. arxiv. https://arxiv.org/abs/2402.19479 (2024-02)
[3] MUNITE: Unified Multimodal Latent Inference for Any-to-Any Multimodal Generation. arxiv. https://arxiv.org/abs/2610.09866 (2026-10-07)
[4] NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching. hf-search. https://huggingface.co/papers/2510.13721 (2025-10-15)
[5] Training-Free Multimodal Guidance for Video to Audio Generation. web. https://arxiv.org/pdf/2509.24550v1.pdf (2025-09)
[6] Efficient Video-to-Audio Generation via Multiple Foundation Models Mapper (MFM-Mapper). web. https://dl.acm.org/doi/full/10.1145/3820048 (2026-07-21)
[7] VBench: Comprehensive Benchmark Suite for Video Generative Models. hf-search. https://huggingface.co/papers/2311.17982 (2023-11)
[8] Fréchet Video Distance: A Metric for Evaluating Video Generation Models. arxiv. https://arxiv.org/abs/1812.01717 (2018-12)
[9] World Consistency Score: Evaluating Physical and Relational Coherence in Video Generation. arxiv. https://arxiv.org/abs/2508.00144 (2025-08)
[10] Multi2AV-Safety: Benchmarking Safety in Multimodal-to-Audio-Video Generation. hf-search. https://huggingface.co/papers/2608.26535 (2026-08-27)
[11] Video-ChatGPT: Towards Detailed Video Understanding and Generation. arxiv. https://arxiv.org/abs/2306.16590 (2023-06)
[12] WebVid-2M: A Large-Scale Video-Text Dataset for Text-to-Video Generation. arxiv. https://arxiv.org/abs/2111.13681 (2021-11)
[13] UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark. hf-search. https://huggingface.co/papers/2603.05075 (2026-03-05)
[14] CLIPScore: A Reference-free Evaluation Metric for Image Captioning. arxiv. https://arxiv.org/abs/2104.08718 (2021-04)
