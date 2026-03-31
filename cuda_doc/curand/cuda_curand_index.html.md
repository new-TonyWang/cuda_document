# cuRAND :: CUDA Toolkit Documentation

**来源**: [https://docs.nvidia.com/cuda/curand/index.html](https://docs.nvidia.com/cuda/curand/index.html)

---

## cuRAND

The API reference guide for cuRAND, the CUDA random number generation library.

# Table of Contents

- [Introduction](https://docs.nvidia.com/cuda/curand/introduction.html#introduction)
- [1. Compatibility and Versioning](https://docs.nvidia.com/cuda/curand/compatibility-and-versioning.html#compatibility-and-versioning)
- [2. Host API Overview](https://docs.nvidia.com/cuda/curand/host-api-overview.html#host-api-overview)- [2.1. Generator Types](https://docs.nvidia.com/cuda/curand/host-api-overview.html#generator-types)
  - [2.2. Generator Options](https://docs.nvidia.com/cuda/curand/host-api-overview.html#generator-options)- [2.2.1. Seed](https://docs.nvidia.com/cuda/curand/host-api-overview.html#seed)
    - [2.2.2. Offset](https://docs.nvidia.com/cuda/curand/host-api-overview.html#offset)
    - [2.2.3. Order](https://docs.nvidia.com/cuda/curand/host-api-overview.html#order)
  - [2.3. Return Values](https://docs.nvidia.com/cuda/curand/host-api-overview.html#return-values)
  - [2.4. Generation Functions](https://docs.nvidia.com/cuda/curand/host-api-overview.html#generation-functions)
  - [2.5. Host API Example](https://docs.nvidia.com/cuda/curand/host-api-overview.html#host-api-example)
  - [2.6. Static Library support](https://docs.nvidia.com/cuda/curand/host-api-overview.html#static-library)
  - [2.7. Performance Notes](https://docs.nvidia.com/cuda/curand/host-api-overview.html#performance-notes2)
  - [2.8. Thread Safety](https://docs.nvidia.com/cuda/curand/host-api-overview.html#thread-safety)
- [3. Device API Overview](https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview)- [3.1. Pseudorandom Sequences](https://docs.nvidia.com/cuda/curand/device-api-overview.html#pseudorandom-sequences)- [3.1.1. Bit Generation with XORWOW and MRG32k3a generators](https://docs.nvidia.com/cuda/curand/device-api-overview.html#bit-generation-1)
    - [3.1.2. Bit Generation with the MTGP32 generator](https://docs.nvidia.com/cuda/curand/device-api-overview.html#bit-generation-2)
    - [3.1.3. Bit Generation with Philox_4x32_10 generator](https://docs.nvidia.com/cuda/curand/device-api-overview.html#bit-generation-3)
    - [3.1.4. Distributions](https://docs.nvidia.com/cuda/curand/device-api-overview.html#distributions)
  - [3.2. Quasirandom Sequences](https://docs.nvidia.com/cuda/curand/device-api-overview.html#quasirandom-sequences)
  - [3.3. Skip-Ahead](https://docs.nvidia.com/cuda/curand/device-api-overview.html#skip-ahead)
  - [3.4. Device API for discrete distributions](https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-for-discrete-distributions)
  - [3.5. Performance Notes](https://docs.nvidia.com/cuda/curand/device-api-overview.html#performance-notes)
  - [3.6. Device API Examples](https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example)
  - [3.7. Thrust and cuRAND Example](https://docs.nvidia.com/cuda/curand/device-api-overview.html#thrust-and-curand-example)
  - [3.8. Poisson API Example](https://docs.nvidia.com/cuda/curand/device-api-overview.html#poisson-api-example)
- [4. Testing](https://docs.nvidia.com/cuda/curand/testing.html#testing)
- [5. Modules](https://docs.nvidia.com/cuda/curand/modules.html#modules)- [5.1. Host API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST)
  - [5.2. Device API](https://docs.nvidia.com/cuda/curand/group__DEVICE.html#group__DEVICE)
- [A. Bibliography](https://docs.nvidia.com/cuda/curand/bibliography.html#bibliography)
- [B. Acknowledgements](https://docs.nvidia.com/cuda/curand/acknowledgements.html#acknowledgements)

---