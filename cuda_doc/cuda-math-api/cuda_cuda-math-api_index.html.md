# CUDA Math API Reference Manual — CUDA Math API Reference Manual 13.2 documentation

**来源**: [https://docs.nvidia.com/cuda/cuda-math-api/index.html](https://docs.nvidia.com/cuda/cuda-math-api/index.html)

---

# CUDA Math API Reference Manual
CUDA mathematical functions are always available in device code.
Host implementations of the common mathematical functions are mapped in a platform-specific way to standard math library functions, provided by the host compiler and respective host libm where available. Some functions, not available with the host compilers, are implemented in crt/math_functions.hpp header file. For example, see[erfinv()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#group__cuda__math__double_1gaef012e8d10e9ef980940f65630f77ae3). Other, less common functions, like[rhypot()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#group__cuda__math__double_1gaf1dfb4d01feaa01b0b1ff15cf57ebbc3),[cyl_bessel_i0()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#group__cuda__math__double_1gaaeae8990c401dc1ad0426de1350560b3)are only available in device code.
CUDA Math device functions are no-throw for well-formed CUDA programs.
Note that many floating-point and integer functions names are overloaded for different argument types. For example, the[log()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#group__cuda__math__double_1ga28ce8e15ef5149c271eba95663becba2)function has the following prototypes:

```
double log(double x);
float log(float x);
float logf(float x);

```

Note also that due to implementation constraints, certain math functions from std:: namespace may be callable in device code even via explicitly qualified std:: names. However, such use is discouraged, since this capability is unsupported, unverified, undocumented, not portable, and may change without notice.

- [1. FP4 Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__FP4.html)
- [2. FP6 Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__FP6.html)
- [3. FP8 Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__FP8.html)
- [4. Half Precision Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__HALF.html)
- [5. Bfloat16 Precision Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__BFLOAT16.html)
- [6. Single Precision Mathematical Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html)
- [7. Single Precision Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html)
- [8. Double Precision Mathematical Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html)
- [9. Double Precision Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html)
- [10. FP128 Quad Precision Mathematical Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html)
- [11. Type Casting Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html)
- [12. Integer Mathematical Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INT.html)
- [13. Integer Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html)
- [14. SIMD Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html)
- [15. Structs](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/structs.html)
- [16. Notices](https://docs.nvidia.com/cuda/cuda-math-api/notices.html)