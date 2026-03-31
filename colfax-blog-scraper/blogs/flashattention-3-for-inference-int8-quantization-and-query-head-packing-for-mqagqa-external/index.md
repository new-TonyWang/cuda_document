# FlashAttention-3 for Inference: INT8 Quantization and Query Head Packing for MQA/GQA (External)

**Date:** November 27, 2024

**Source:** [https://research.colfax-intl.com/flashattention-3-for-inference-int8-quantization-and-query-head-packing-for-mqa-gqa-external/](https://research.colfax-intl.com/flashattention-3-for-inference-int8-quantization-and-query-head-packing-for-mqa-gqa-external/)

---

In this [blog post](https://blog.character.ai/optimizing-ai-inference-at-character-ai-part-deux-2/) presented on the Character.AI research blog, we explain two techniques that are important for using [FlashAttention-3](https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/) for inference:

1. A general methodology for in-kernel pre-processing of tensors via warp specialization, applied to the case of a half INT8 attention kernel design that upcasts the V tensor in the producer warpgroup.
2. Query head packing of the Q tile done for multi-query attention (MQA) or grouped query attention (GQA), which is needed to saturate bandwidth during the memory-bound decoding phase of inference.

We also give microbenchmark results for both prefill and decode-type attention workloads, measured on an NVIDIA H100 SXM5 GPU.

[https://blog.character.ai/optimizing-ai-inference-at-character-ai-part-deux-2/](https://blog.character.ai/optimizing-ai-inference-at-character-ai-part-deux-2/)

![](images/character-deus-featured-image.png)

Optimizing AI Inference at Character.AI (Part Deux)At Character.AI, we’re building personalized AI entertainment. In order to offer our users engaging, interactive experiences, it’s critical we achieve highly efficient inference, or the process by which LLMs generate replies. Our last post on this topic looked at several techniques that contribute to the performance and sustainability

*Joint work with Character.AI*.
