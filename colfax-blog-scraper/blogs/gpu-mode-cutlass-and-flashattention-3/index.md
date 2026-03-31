# GPU Mode: CUTLASS and FlashAttention-3

**Date:** November 18, 2024

**Source:** [https://research.colfax-intl.com/gpu-mode-cutlass-and-flashattention-3/](https://research.colfax-intl.com/gpu-mode-cutlass-and-flashattention-3/)

---

In this GPU Mode lecture, Jay Shah presents his joint work on FlashAttention-3 and how to implement the main compute loop in the algorithm using CUTLASS.

The code discussed in this lecture can be found at [this commit](https://github.com/Dao-AILab/flash-attention/blob/b2d3fe92ff43edbd650aeba2af5ce0af23515683/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp) in the FlashAttention-3 codebase.

[cutlass-flashattn3-slides](https://research.colfax-intl.com/wp-content/uploads/2024/11/flash_attn_3_gpu_mode_talk.pdf)

**Note**: Slides adapted from a talk given by Tri Dao.
