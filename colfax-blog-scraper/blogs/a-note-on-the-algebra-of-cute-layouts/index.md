# A note on the algebra of CuTe Layouts

**Date:** December 14, 2023

**Source:** [https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/)

---

The core abstraction of NVIDIA’s CUTLASS library for high-performance linear algebra is the CuTe Layout. In this technical note, we give a rigorous, mathematical treatment of the algebra of these layouts and certain layout operations. Currently, the main goal is to lay down conditions for when the operations of complementation, composition, and logical division are well-defined, which may be of general use to CUTLASS developers. This note should be read as complementary to the discussion of these layout operations in the CuTe documentation.  
  
1/8/24: added a section on permutations expressible as layout functions.

![](images/PDF_32.png)

[layout_algebra.pdf](https://research.colfax-intl.com/download/cute-layout-algebra/?tmstv=1774608058)
