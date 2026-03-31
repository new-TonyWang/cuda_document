# FlexAttention + FlashAttention-4: Fast and Flexible (External)

**Date:** March 10, 2026

**Source:** [https://research.colfax-intl.com/flexattention-flashattention-4-fast-and-flexible-external/](https://research.colfax-intl.com/flexattention-flashattention-4-fast-and-flexible-external/)

---

In this PyTorch blog on which we collaborated, we explain the FlexAttention extension to FlashAttention-4 (or from another point of view, the incorporation of FA-4 as an attention backend for the PyTorch FlexAttention API).

[https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)

![](images/flexattn.jpg)

FlexAttention + FlashAttention-4: Fast and Flexible – PyTorchOn Hopper and Blackwell GPUs, FlexAttention now has a FlashAttention-4 backend.  

  

We added support in PyTorch to automatically generate CuTeDSL score/mask modification functions, and to JIT-instantiate FlashAttention-4 for custom attention variants.  

  

This leads to performance gains of 1.2× to 3.2× over the existing Triton implementation on compute-bound workloads.
