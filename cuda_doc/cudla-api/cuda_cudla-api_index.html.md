# cuDLA API :: CUDA Toolkit Documentation

**来源**: [https://docs.nvidia.com/cuda/cudla-api/index.html](https://docs.nvidia.com/cuda/cudla-api/index.html)

---

## 1. Modules

Here is a list of all modules:

- [Data types used by cuDLA driver](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES)
- [cuDLA API](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API)

### 1.1. Data types used by cuDLA driver

#### Classes

struct
[CudlaFence](https://docs.nvidia.com/cuda/cudla-api/index.html#structCudlaFence)

union
[cudlaDevAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaDevAttribute)

struct
[cudlaExternalMemoryHandleDesc_t](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaExternalMemoryHandleDesc__t)

struct
[cudlaExternalSemaphoreHandleDesc_t](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaExternalSemaphoreHandleDesc__t)

union
[cudlaModuleAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute)

struct
[cudlaModuleTensorDescriptor](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaModuleTensorDescriptor)

struct
[cudlaSignalEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents)

struct
[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)

struct
[cudlaWaitEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaWaitEvents)

#### Typedefs

typedef cudlaDevHandle_t *cudlaDevHandle

typedef cudlaModule_t *cudlaModule

#### Enumerations

enumcudlaAccessPermissionFlags

enumcudlaDevAttributeType

enumcudlaFenceType

enumcudlaMode

enumcudlaModuleAttributeType

enumcudlaModuleLoadFlags

enumcudlaNvSciSyncAttributes

enumcudlaStatus

enumcudlaSubmissionFlags

#### Typedefs

typedef cudlaDevHandle_t * cudlaDevHandle

cuDLA Device Handle

typedef cudlaModule_t * cudlaModule

cuDLA Module Handle

#### Enumerations

enum cudlaAccessPermissionFlags

Access permission flags for importing NvSciBuffers

###### Values

CUDLA_READ_WRITE_PERM =0
Flag to import memory with read-write permission
CUDLA_READ_ONLY_PERM =1
Flag to import memory with read-only permission
CUDLA_TASK_STATISTICS =1<<1
Flag to indicate buffer as layerwise statistics buffer.

enum cudlaDevAttributeType

Device attribute type.

###### Values

CUDLA_UNIFIED_ADDRESSING =0
Flag to check for support for UVA.
CUDLA_DEVICE_VERSION =1
Flag to check for DLA HW version.

enum cudlaFenceType

Supported fence types.

###### Values

CUDLA_NVSCISYNC_FENCE =1
NvSciSync fence type for EOF.
CUDLA_NVSCISYNC_FENCE_SOF =2

enum cudlaMode

Device creation modes.

###### Values

CUDLA_CUDA_DLA =0
Hyrbid mode.
CUDLA_STANDALONE =1
Standalone mode.

enum cudlaModuleAttributeType

Module attribute types.

###### Values

CUDLA_NUM_INPUT_TENSORS =0
Flag to retrieve number of input tensors.
CUDLA_NUM_OUTPUT_TENSORS =1
Flag to retrieve number of output tensors.
CUDLA_INPUT_TENSOR_DESCRIPTORS =2
Flag to retrieve all the input tensor descriptors.
CUDLA_OUTPUT_TENSOR_DESCRIPTORS =3
Flag to retrieve all the output tensor descriptors.
CUDLA_NUM_OUTPUT_TASK_STATISTICS =4
Flag to retrieve total number of output task statistics buffer.
CUDLA_OUTPUT_TASK_STATISTICS_DESCRIPTORS =5
Flag to retrieve all the output task statistics descriptors.

enum cudlaModuleLoadFlags

Module load flags for[cudlaModuleLoadFromMemory](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1gcd725924569cec1a3214fd09cb38601d). 

###### Values

CUDLA_MODULE_DEFAULT =0
Default flag.
CUDLA_MODULE_ENABLE_FAULT_DIAGNOSTICS =1
Flag to load a module that is used to perform permanent fault diagnostics for DLA HW.

enum cudlaNvSciSyncAttributes

cuDLA NvSciSync attributes.

###### Values

CUDLA_NVSCISYNC_ATTR_WAIT =1
Wait attribute.
CUDLA_NVSCISYNC_ATTR_SIGNAL =2
Signal attribute.

enum cudlaStatus

Error codes.

###### Values

cudlaSuccess =0
The API call returned with no errors.
cudlaErrorInvalidParam =1
This indicates that one or more parameters passed to the API is/are incorrect.
cudlaErrorOutOfResources =2
This indicates that the API call failed due to lack of underlying resources.
cudlaErrorCreationFailed =3
This indicates that an internal error occurred during creation of device handle.
cudlaErrorInvalidAddress =4
This indicates that the memory object being passed in the API call has not been registered before.
cudlaErrorOs =5
This indicates that an OS error occurred.
cudlaErrorCuda =6
This indicates that there was an error in a CUDA operation as part of the API call.
cudlaErrorUmd =7
This indicates that there was an error in the DLA runtime for the API call.
cudlaErrorInvalidDevice =8
This indicates that the device handle passed to the API call is invalid.
cudlaErrorInvalidAttribute =9
This indicates that an invalid attribute is being requested.
cudlaErrorIncompatibleDlaSWVersion =10
This indicates that the underlying DLA runtime is incompatible with the current cuDLA version.
cudlaErrorMemoryRegistered =11
This indicates that the memory object is already registered.
cudlaErrorInvalidModule =12
This indicates that the module being passed is invalid.
cudlaErrorUnsupportedOperation =13
This indicates that the operation being requested by the API call is unsupported.
cudlaErrorNvSci =14
This indicates that the NvSci operation requested by the API call failed.
cudlaErrorDlaErrInvalidInput =0x40000001
DLA HW Error.
cudlaErrorDlaErrInvalidPreAction =0x40000002
DLA HW Error.
cudlaErrorDlaErrNoMem =0x40000003
DLA HW Error.
cudlaErrorDlaErrProcessorBusy =0x40000004
DLA HW Error.
cudlaErrorDlaErrTaskStatusMismatch =0x40000005
DLA HW Error.
cudlaErrorDlaErrEngineTimeout =0x40000006
DLA HW Error.
cudlaErrorDlaErrDataMismatch =0x40000007
DLA HW Error.
cudlaErrorUnknown =0x7fffffff
This indicates that an unknown error has occurred.

enum cudlaSubmissionFlags

Task submission flags for[cudlaSubmitTask](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1gc560a614b388d50216bd161c0b3d88cb). 

###### Values

CUDLA_SUBMIT_NOOP =1
Flag to specify that the submitted task must be bypassed for execution.
CUDLA_SUBMIT_SKIP_LOCK_ACQUIRE =1<<1
Flag to specify that the global lock acquire must be skipped.
CUDLA_SUBMIT_DIAGNOSTICS_TASK =1<<2
Flag to specify that the submitted task is to run permanent fault diagnostics for DLA HW.

### 1.2. cuDLA API

This section describes the application programming interface of the cuDLA driver.

#### Functions

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaCreateDevice( const uint64_tdevice, const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)*devHandle, const uint32_tflags)

Create a device handle.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaDestroyDevice( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle)

Destroy device handle.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaDeviceGetAttribute( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const[cudlaDevAttributeType](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gae3e445aeb1fe1c992b9357cdc4c913a)attrib, const[cudlaDevAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaDevAttribute)*pAttribute)

Get cuDLA device attributes.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaDeviceGetCount( const uint64_t*pNumDevices)

Get device count.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaGetLastError( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle)

Gets the last asynchronous error in task execution.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaGetNvSciSyncAttributes( uint64_t*attrList, const uint32_tflags)

Get cuDLA's NvSciSync attributes.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaGetVersion( const uint64_t*version)

Returns the version number of the library.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaImportExternalMemory( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const cudlaExternalMemoryHandleDesc*desc, const uint64_t**devPtr, const uint32_tflags)

Imports external memory into cuDLA.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaImportExternalSemaphore( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const cudlaExternalSemaphoreHandleDesc*desc, const uint64_t**devPtr, const uint32_tflags)

Imports external semaphore into cuDLA.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaMemRegister( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const uint64_t*ptr, const size_tsize, const uint64_t**devPtr, const uint32_tflags)

Registers the CUDA memory to DLA engine.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaMemUnregister( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const uint64_t*devPtr)

Unregisters the input memory from DLA engine.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaModuleGetAttributes( const[cudlaModule](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ge3ecb829f32791357568b7d005d107a5)hModule, const[cudlaModuleAttributeType](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ga2b041ca59fb0103b62272b83a3b2ba2)attrType, const[cudlaModuleAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute)*attribute)

Get DLA module attributes.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaModuleLoadFromMemory( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const uint8_t*pModule, const size_tmoduleSize, const[cudlaModule](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ge3ecb829f32791357568b7d005d107a5)*hModule, const uint32_tflags)

Load a DLA module.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaModuleUnload( const[cudlaModule](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ge3ecb829f32791357568b7d005d107a5)hModule, const uint32_tflags)

Unload a DLA module.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaSetTaskTimeoutInMs( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const uint32_ttimeout)

Set task timeout in millisecond.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaSubmitTask( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)*ptrToTasks, const uint32_tnumTasks, const void*stream, const uint32_tflags)

Submits the inference operation on DLA.

#### Functions

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaCreateDevice ( const uint64_tdevice, const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)*devHandle, const uint32_tflags)

Create a device handle.

###### Parameters

device
- Device number (can be 0 or 1).
devHandle
- Pointer to hold the created cuDLA device handle.
flags
- Flags controlling device creation. Valid values forflagsare:
 
- [CUDLA_CUDA_DLA](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gg74ca7d641e2873189059953d3ed65cb21dffbcf6879c77d2c7a1efc6012f72c6)- In this mode, cuDLA serves as a programming model extension of CUDA wherein DLA work can be submitted using CUDA constructs.
- [CUDLA_STANDALONE](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gg74ca7d641e2873189059953d3ed65cb27789b9d73cb032d268d84463a5c26f91)- In this mode, cuDLA works standalone without any interaction with CUDA.

###### Returns
[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1),[cudlaErrorOutOfResources](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a05b9d884f3dd0210f0cfa56e064c29dcb),[cudlaErrorInvalidParam](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c80900e54c2a9c02f23935a69f214428),[cudlaErrorIncompatibleDlaSWVersion](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a02c136741dbd4e0fa0d3c8c3643d9b03e),[cudlaErrorCreationFailed](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c848145627b7a723ee09b29aaa41a834),[cudlaErrorCuda](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0727b16e07c9c63a5f1379826a31eb983),[cudlaErrorUmd](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d22f87ac0d722cd49a74217f472df994),[cudlaErrorUnsupportedOperation](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03ba9ef84a3f15d6d1145047e11d1804a)

###### Description
Creates an instance of a cuDLA device which can be used to submit DLA operations. The application can create the handle in
 hybrid or standalone mode. In hybrid mode, the current set GPU device is used by this API to decide the association of the
 created DLA device handle. This function returns[cudlaErrorUnsupportedOperation](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03ba9ef84a3f15d6d1145047e11d1804a)if the current set GPU device is a dGPU as cuDLA is not supported on dGPU presently. cuDLA supports 16 cuDLA device handles
 per DLA HW instance.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaDestroyDevice ( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle)

Destroy device handle.

###### Parameters

devHandle
- A valid device handle.

###### Returns
[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1),[cudlaErrorInvalidDevice](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04b8ca897368458ceb53b537b8979218b),[cudlaErrorCuda](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0727b16e07c9c63a5f1379826a31eb983),[cudlaErrorUmd](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d22f87ac0d722cd49a74217f472df994)

###### Description
Destroys the instance of the cuDLA device which was created with cudlaCreateDevice. Before destroying the handle, it is important
 to ensure that all the tasks submitted previously to the device are completed. Failure to do so can lead to application crashes.

In hybrid mode, cuDLA internally performs memory allocations with CUDA using the primary context. As a result, before destroying
 or resetting a CUDA primary context, it is mandatory that all cuDLA device initializations are destroyed.

Note:This API can return task execution errors from previous DLA task submissions.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaDeviceGetAttribute ( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const[cudlaDevAttributeType](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gae3e445aeb1fe1c992b9357cdc4c913a)attrib, const[cudlaDevAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaDevAttribute)*pAttribute)

Get cuDLA device attributes.

###### Parameters

devHandle
- The input cuDLA device handle.
attrib
- The attribute that is being requested.
pAttribute
- The output pointer where the attribute will be available.

###### Returns
[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1),[cudlaErrorInvalidParam](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c80900e54c2a9c02f23935a69f214428),[cudlaErrorInvalidDevice](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04b8ca897368458ceb53b537b8979218b),[cudlaErrorUmd](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d22f87ac0d722cd49a74217f472df994),[cudlaErrorInvalidAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0df7b517d9f6d1ca10675c1e988e5101b)

###### Description
UVA addressing between CUDA and DLA requires special support in the underlying kernel mode drivers. Applications are expected
 to query the cuDLA runtime to check if the current version of cuDLA supports UVA addressing.

Note:This API can return task execution errors from previous DLA task submissions.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaDeviceGetCount ( const uint64_t*pNumDevices)

Get device count.

###### Parameters

pNumDevices
- The number of DLA devices will be available in this variable upon successful completion.

###### Returns
[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1),[cudlaErrorInvalidParam](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c80900e54c2a9c02f23935a69f214428),[cudlaErrorUmd](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d22f87ac0d722cd49a74217f472df994),[cudlaErrorIncompatibleDlaSWVersion](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a02c136741dbd4e0fa0d3c8c3643d9b03e)

###### Description
Get number of DLA devices available to use.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaGetLastError ( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle)

Gets the last asynchronous error in task execution.

###### Parameters

devHandle
- A valid device handle.

###### Returns
[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1),[cudlaErrorInvalidDevice](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04b8ca897368458ceb53b537b8979218b),[cudlaErrorDlaErrInvalidInput](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a01ddae2b987abee60fb416abc9540a8cb),[cudlaErrorDlaErrInvalidPreAction](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a05e8ecc197526e1367f51951c9bbf36e4),[cudlaErrorDlaErrNoMem](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a089328cd138b285d0c00f26ed4115f88e),[cudlaErrorDlaErrProcessorBusy](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04367e9544354f75e5c7556d60bfaf149),[cudlaErrorDlaErrTaskStatusMismatch](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a007a6137f3875744e2e4b2cf54d5fe7d4),[cudlaErrorDlaErrEngineTimeout](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0b57798c888d3515eceff4ff78d7ec140),[cudlaErrorDlaErrDataMismatch](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a07c3a9a91cde8c19b111cd633ae6251f4),[cudlaErrorUnknown](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0026f9875981d619043bad450ca90e5ed)

###### Description
The DLA tasks execute asynchronously on the DLA HW. As a result, the status of the task execution is not known at the time
 of task submission. The status of the task executed by the DLA HW most recently for the particular device handle can be queried
 using this interface.

Note that a return code of[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1)from this function does not necessarily imply that most recent task executed successfully. Since this function returns immediately,
 it can only report the status of the tasks at the snapshot of time when it is called. To be guaranteed of task completion,
 applications must synchronize on the submitted tasks in hybrid or standalone modes and then call this API to check for errors.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaGetNvSciSyncAttributes ( uint64_t*attrList, const uint32_tflags)

Get cuDLA's NvSciSync attributes.

###### Parameters

attrList
- Attribute list created by the application.
flags
- Applications can use this flag to specify how they intend to use the NvSciSync object created from theattrList. The valid values offlagscan be one of the following (or an OR of these values):
 
- [CUDLA_NVSCISYNC_ATTR_WAIT](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gg3efeaae42e362ebb197cb69170941c6e2d049b652955f2bf109b4e1e41660d01), specifies that the application intend to use the NvSciSync object created using this attribute list as a waiter in cuDLA
   and therefore needs cuDLA to fill waiter specific NvSciSyncAttr.
- [CUDLA_NVSCISYNC_ATTR_SIGNAL](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gg3efeaae42e362ebb197cb69170941c6ed31eff6290cfb76e793883564ba0fce6), specifies that the application intend to use the NvSciSync object created using this attribute list as a signaler in cuDLA
   and therefore needs cuDLA to fill signaler specific NvSciSyncAttr.

###### Returns

- [cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1), The API call returned with no errors.
- [cudlaErrorInvalidParam](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c80900e54c2a9c02f23935a69f214428), This API call failed because invalid parameter attrList was passed.
- [cudlaErrorUnsupportedOperation](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03ba9ef84a3f15d6d1145047e11d1804a), This error code indicates that the API call failed because the operation is not supported in hybrid mode.
- [cudlaErrorInvalidAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0df7b517d9f6d1ca10675c1e988e5101b), The API call failed as parameter attrList has invalid values.
- [cudlaErrorNvSci](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d3dfc3fc5ad64fbb4b03c3861502667e), This error code indicates error in the NvSci operation as part of the API call.
- cudlaErrorNotPermittedOperation, This error code indicates that the API call is not permitted when DRIVE OS is in Operational
   state.
- [cudlaErrorUnknown](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0026f9875981d619043bad450ca90e5ed), This error code indicates that an unknown error has occurred.

###### Description
Gets the NvSciSync's attributes in the attribute list created by the application.
cuDLA supports three types of NvSciSync object primitives -

- Sync point
- Regular semaphore
- Deterministic semaphore cuDLA prioritizes sync point primitive over regular and deterministic semaphore primitives by default
   and sets these priorities in the NvSciSync attribute list.
For Deterministic semaphore, NvSciSync attribute list used to create the NvSciSync object must have value of NvSciSyncAttrKey_RequireDeterministicFences
 key set to true.

cuDLA also supports Timestamp feature on NvSciSync objects. Waiter can request for this by setting NvSciSync attribute "NvSciSyncAttrKey_WaiterRequireTimestamps"
 as true.

In the event of failed NvSci initialization this function would return[cudlaErrorUnsupportedOperation](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03ba9ef84a3f15d6d1145047e11d1804a). This function can return[cudlaErrorNvSci](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d3dfc3fc5ad64fbb4b03c3861502667e)or[cudlaErrorInvalidAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0df7b517d9f6d1ca10675c1e988e5101b)in certain cases when the underlying NvSci operation fails.

This API updates the input nvSciSyncAttrList with values equivalent to the following public attribute key-values:
NvSciSyncAttrKey_RequiredPerm is set to

- NvSciSyncAccessPerm_SignalOnly if value of flag is set to CUDLA_NVSCISYNC_ATTR_WAIT.
- NvSciSyncAccessPerm_WaitOnly if value of flag is set to CUDLA_NVSCISYNC_ATTR_SIGNAL.
- NvSciSyncAccessPerm_WaitSignal if value of flag is set to CUDLA_NVSCISYNC_ATTR_SIGNAL | CUDLA_NVSCISYNC_ATTR_WAIT.
As NvSciSyncAttrKey_RequiredPerm is internally set by cuDLA, setting this value by the application is disallowed.

Note:Users of cuDLA can only append attributes to outputattrListusing NvSci API, modifying already populated values of the outputattrListcan result in undefined behavior. 

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaGetVersion ( const uint64_t*version)

Returns the version number of the library.

###### Parameters

version
- cuDLA library version will be available in this variable upon successful execution.

###### Returns
[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1),[cudlaErrorInvalidParam](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c80900e54c2a9c02f23935a69f214428)

###### Description
cuDLA is semantically versioned. This function will return the version as 1000000*major + 1000*minor + patch.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaImportExternalMemory ( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const cudlaExternalMemoryHandleDesc*desc, const uint64_t**devPtr, const uint32_tflags)

Imports external memory into cuDLA.

###### Parameters

devHandle
- A valid device handle.
desc
- Contains description about allocated external memory.
devPtr
- The output pointer where the mapping will be available.
flags
- Application can use this flag to specify the memory access permissions of the memory that needs to be registered with DLA.
 The valid values offlagscan be one of the following:
 
- [CUDLA_READ_WRITE_PERM](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gg631e2c17faee6b825b83577efb55ed75748d993f14cb963da3fa1a806234645d), specifies that the external memory needs to be registered with DLA as read-write memory.
- [CUDLA_READ_ONLY_PERM](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gg631e2c17faee6b825b83577efb55ed75d9ed991984f3164d4eef1806a598db0c), specifies that the external memory needs to be registered with DLA as read-only memory.
- [CUDLA_TASK_STATISTICS](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gg631e2c17faee6b825b83577efb55ed751237111e1003c0f81c4fc3bf62bf6001), specifies that the external memory needs to be registered with DLA for layerwise statistics.

###### Returns
[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1),[cudlaErrorInvalidParam](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c80900e54c2a9c02f23935a69f214428),[cudlaErrorInvalidDevice](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04b8ca897368458ceb53b537b8979218b),[cudlaErrorUnsupportedOperation](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03ba9ef84a3f15d6d1145047e11d1804a),[cudlaErrorNvSci](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d3dfc3fc5ad64fbb4b03c3861502667e),[cudlaErrorInvalidAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0df7b517d9f6d1ca10675c1e988e5101b),[cudlaErrorMemoryRegistered](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0e335fc20958745722f1f7a22f22c383d),[cudlaErrorUmd](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d22f87ac0d722cd49a74217f472df994)

###### Description
Imports the allocated external memory by registering it with DLA. After successful registration, the returned pointer can
 be used in a task submit.

On Tegra, cuDLA supports importing NvSciBuf objects in standalone mode only. In the event of failed NvSci initialization (either
 due to usage of this API in hybrid mode or an issue in the NvSci library initialization), this function would return[cudlaErrorUnsupportedOperation](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03ba9ef84a3f15d6d1145047e11d1804a). This function can return[cudlaErrorNvSci](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d3dfc3fc5ad64fbb4b03c3861502667e)or[cudlaErrorInvalidAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0df7b517d9f6d1ca10675c1e988e5101b)in certain cases when the underlying NvSci operation fails.

Note:cuDLA only supports importing NvSciBuf objects of type NvSciBufType_RawBuffer or NvSciBufType_Tensor. Importing NvSciBuf object
 of any other type can result in an undefined behaviour. 

Note:- This API may result in undefined behavior if the address being registered is not 32-byte aligned. The input pointerdevPtrmust always satisfy the condition ((devPtr & 0x1F) == 0)
- This API can return task execution errors from previous DLA task submissions.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaImportExternalSemaphore ( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const cudlaExternalSemaphoreHandleDesc*desc, const uint64_t**devPtr, const uint32_tflags)

Imports external semaphore into cuDLA.

###### Parameters

devHandle
- A valid device handle.
desc
- Contains sempahore object.
devPtr
- The output pointer where the mapping will be available.
flags
- Reserved for future. Must be set to 0.

###### Returns
[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1),[cudlaErrorInvalidParam](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c80900e54c2a9c02f23935a69f214428),[cudlaErrorInvalidDevice](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04b8ca897368458ceb53b537b8979218b),[cudlaErrorUnsupportedOperation](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03ba9ef84a3f15d6d1145047e11d1804a),[cudlaErrorNvSci](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d3dfc3fc5ad64fbb4b03c3861502667e),[cudlaErrorInvalidAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0df7b517d9f6d1ca10675c1e988e5101b),[cudlaErrorMemoryRegistered](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0e335fc20958745722f1f7a22f22c383d)

###### Description
Imports the allocated external semaphore by registering it with DLA. After successful registration, the returned pointer can
 be used in a task submission to signal synchronization objects.

On Tegra, cuDLA supports importing NvSciSync objects in standalone mode only. NvSciSync object primitives that cuDLA supports
 are sync point and deterministic semaphore.

cuDLA also supports Timestamp feature on NvSciSync objects, using which the user can get a snapshot of DLA clock at which
 a particular fence is signaled. At any point in time there are only 512 valid timestamp buffers that can be associated with
 fences. For example, If User has created 513 fences from a single NvSciSync object with timestamp enabled then the timestamp
 buffer associated with 1st fence is same as with 513th fence.

In the event of failed NvSci initialization (either due to usage of this API in hybrid mode or an issue in the NvSci library
 initialization), this function would return[cudlaErrorUnsupportedOperation](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03ba9ef84a3f15d6d1145047e11d1804a). This function can return[cudlaErrorNvSci](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d3dfc3fc5ad64fbb4b03c3861502667e)or[cudlaErrorInvalidAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0df7b517d9f6d1ca10675c1e988e5101b)in certain cases when the underlying NvSci operation fails.

Note:This API can return task execution errors from previous DLA task submissions.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaMemRegister ( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const uint64_t*ptr, const size_tsize, const uint64_t**devPtr, const uint32_tflags)

Registers the CUDA memory to DLA engine.

###### Parameters

devHandle
- A valid cuDLA device handle create by a previous call to[cudlaCreateDevice](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1gf9d00e3a93dfd31814736144764ae478). 

ptr
- The CUDA pointer to be registered.
size
- The size of the mapping i.e the number of bytes from ptr that must be mapped.
devPtr
- The output pointer where the mapping will be available.
flags
- Applications can use this flag to control several aspects of the registration process. The valid values offlagscan be one of the following (or an OR of these values):
 
- 0, default
- [CUDLA_TASK_STATISTICS](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gg631e2c17faee6b825b83577efb55ed751237111e1003c0f81c4fc3bf62bf6001), specifies that the external memory needs to be registered with DLA for layerwise statistics.

###### Returns
[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1),[cudlaErrorInvalidDevice](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04b8ca897368458ceb53b537b8979218b),[cudlaErrorInvalidParam](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c80900e54c2a9c02f23935a69f214428),[cudlaErrorInvalidAddress](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03959fb7cceee17b5c54b60cdffa40e7f),[cudlaErrorCuda](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0727b16e07c9c63a5f1379826a31eb983),[cudlaErrorUmd](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d22f87ac0d722cd49a74217f472df994),[cudlaErrorOutOfResources](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a05b9d884f3dd0210f0cfa56e064c29dcb),[cudlaErrorMemoryRegistered](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0e335fc20958745722f1f7a22f22c383d),[cudlaErrorUnsupportedOperation](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03ba9ef84a3f15d6d1145047e11d1804a)

###### Description
As part of registration, a system mapping is created whereby the DLA HW can access the underlying CUDA memory. The resultant
 mapping is available in devPtr and applications must use this mapping while referring this memory in submit operations.

This function will return[cudlaErrorInvalidAddress](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03959fb7cceee17b5c54b60cdffa40e7f)if the pointer or size to be registered is invalid. In addition, if the input pointer was already registered, then this function
 will return[cudlaErrorMemoryRegistered](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0e335fc20958745722f1f7a22f22c383d). Attempting to re-register memory does not cause any irrecoverable error in cuDLA and applications can continue to use cuDLA
 APIs even after this error has occurred.

Note:- This API may result in undefined behavior if the address being registered is not 32-byte aligned. The input pointerptrmust always satisfy the condition ((ptr & 0x1F) == 0)
- This API can return task execution errors from previous DLA task submissions.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaMemUnregister ( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const uint64_t*devPtr)

Unregisters the input memory from DLA engine.

###### Parameters

devHandle
- A valid cuDLA device handle create by a previous call to[cudlaCreateDevice](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1gf9d00e3a93dfd31814736144764ae478). 

devPtr
- The pointer to be unregistered.

###### Returns
[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1),[cudlaErrorInvalidDevice](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04b8ca897368458ceb53b537b8979218b),[cudlaErrorInvalidAddress](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03959fb7cceee17b5c54b60cdffa40e7f),[cudlaErrorUmd](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d22f87ac0d722cd49a74217f472df994)

###### Description
The system mapping that enables the DLA HW to access the memory is removed. This mapping could have been created by a previous
 call to[cudlaMemRegister](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1ge07f8bb22373163a0117fc5738a23be0),[cudlaImportExternalMemory](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1gca69cd7ac008500693ffeedb18d7a9c8)or[cudlaImportExternalSemaphore](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1g12751fbcc295349c16ad3aea0e8bda34).

Note:This API can return task execution errors from previous DLA task submissions.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaModuleGetAttributes ( const[cudlaModule](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ge3ecb829f32791357568b7d005d107a5)hModule, const[cudlaModuleAttributeType](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ga2b041ca59fb0103b62272b83a3b2ba2)attrType, const[cudlaModuleAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute)*attribute)

Get DLA module attributes.

###### Parameters

hModule
- The input DLA module.
attrType
- The attribute type that is being requested.
attribute
- The output pointer where the attribute will be available.

###### Returns
[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1),[cudlaErrorInvalidParam](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c80900e54c2a9c02f23935a69f214428),[cudlaErrorInvalidModule](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d2b4370207a0a2926eda404f880989d6),[cudlaErrorInvalidDevice](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04b8ca897368458ceb53b537b8979218b),[cudlaErrorUmd](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d22f87ac0d722cd49a74217f472df994),[cudlaErrorInvalidAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0df7b517d9f6d1ca10675c1e988e5101b),[cudlaErrorUnsupportedOperation](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03ba9ef84a3f15d6d1145047e11d1804a)

###### Description
Get module attributes from the loaded module. This API returns[cudlaErrorInvalidDevice](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04b8ca897368458ceb53b537b8979218b)if the module is not loaded in any device.

Note:This API can return task execution errors from previous DLA task submissions.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaModuleLoadFromMemory ( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const uint8_t*pModule, const size_tmoduleSize, const[cudlaModule](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ge3ecb829f32791357568b7d005d107a5)*hModule, const uint32_tflags)

Load a DLA module.

###### Parameters

devHandle
- The input cuDLA device handle. The module will be loaded in the context of this handle.
pModule
- A pointer to an in-memory module.
moduleSize
- The size of the module.
hModule
- The address in which the loaded module handle will be available upon successful execution.
flags
- Applications can use this flag to specify how the module is going to be used. The valid values offlagscan be one of the following:
 
- [CUDLA_MODULE_DEFAULT](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gg5f665d89a741263fbb6fed8c343861de9bdbf383d6e0b00b4e6d02f1ad460e86), Default value which is 0.
- [CUDLA_MODULE_ENABLE_FAULT_DIAGNOSTICS](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gg5f665d89a741263fbb6fed8c343861de3c6cabb1b33ab42a535f5660597d2858), Application can specify this flag to load a module that is used for performing fault diagnostics for DLA HW. With this flag
   set, thepModuleandmoduleSizeparameters shall be NULL and 0 as the diagnostics module is loaded internally.

###### Returns
[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1),[cudlaErrorInvalidDevice](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04b8ca897368458ceb53b537b8979218b),[cudlaErrorInvalidParam](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c80900e54c2a9c02f23935a69f214428),[cudlaErrorOutOfResources](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a05b9d884f3dd0210f0cfa56e064c29dcb),[cudlaErrorUnsupportedOperation](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03ba9ef84a3f15d6d1145047e11d1804a),[cudlaErrorUmd](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d22f87ac0d722cd49a74217f472df994)

###### Description
Loads the module into the current device handle.

- Multiple loadables are not allowed to load onto single cuDLA device handle.
- A Loadable can only be loaded once in cuDLA device handle lifecycle.

Note:This API can return task execution errors from previous DLA task submissions.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaModuleUnload ( const[cudlaModule](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ge3ecb829f32791357568b7d005d107a5)hModule, const uint32_tflags)

Unload a DLA module.

###### Parameters

hModule
- Handle to the loaded module.
flags
- Reserved for future. Must be set to 0.

###### Returns
[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1),[cudlaErrorInvalidParam](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c80900e54c2a9c02f23935a69f214428),[cudlaErrorInvalidDevice](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04b8ca897368458ceb53b537b8979218b),[cudlaErrorInvalidModule](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d2b4370207a0a2926eda404f880989d6),[cudlaErrorUmd](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d22f87ac0d722cd49a74217f472df994)

###### Description
Unload the module from the device handle that it was loaded into. This API returns[cudlaErrorInvalidDevice](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04b8ca897368458ceb53b537b8979218b)if the module is not loaded into a valid device.

Note:This API can return task execution errors from previous DLA task submissions.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaSetTaskTimeoutInMs ( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const uint32_ttimeout)

Set task timeout in millisecond.

###### Parameters

devHandle
- A valid device handle.
timeout
- task timeout value in ms.

###### Returns
[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1),[cudlaErrorInvalidParam](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c80900e54c2a9c02f23935a69f214428)

###### Description
Set task timeout in ms for each device handle. cuDLA sets 30 seconds as default timeout value if user doesn't explicitly set
 the timeout.

In case , device handle is invalid or timeout is 0 or timeout is greater than 1000 sec, this function would return cudlaErrorInvalidParam
 otherwise cudlaSuccess.

Note:This API can return task execution errors from previous DLA task submissions.

[cudlaStatus](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc95f8bcde047e255b11ea1b3b80598a0)cudlaSubmitTask ( const[cudlaDevHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gc52f41cd392913019a800d5f850a9b63)devHandle, const[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)*ptrToTasks, const uint32_tnumTasks, const void*stream, const uint32_tflags)

Submits the inference operation on DLA.

###### Parameters

devHandle
- A valid cuDLA device handle.
ptrToTasks
- A list of inferencing tasks.
numTasks
- The number of tasks.
stream
- The stream on which the DLA task has to be submitted.
flags
- Applications can use this flag to control several aspects of the submission process. The valid values offlagscan be one of the following (or an OR of these values):
 
- 0, default
- [CUDLA_SUBMIT_NOOP](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggbe4b7a209e3dec6e3e3ef1ad082e1f0d3b7e226f4ea9ad2336d6c00855cb7764), specifies that the submitted task must be skipped during execution on the DLA. However, all the waitEvents and signalEvents
   dependencies must be satisfied. This flag is ignored when NULL data submissions are being done as in that case only the wait
   and signal events are internally stored for the next task submission.
- [CUDLA_SUBMIT_SKIP_LOCK_ACQUIRE](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggbe4b7a209e3dec6e3e3ef1ad082e1f0dddca43caf261dc44569bd4b2a2c3a455), specifies that the submitted task is being enqueued in a device handle and that no other task is being enqueued in that
   device handle at that time in any other thread. This is a flag that apps can use as an optimization. Ordinarily, the cuDLA
   APIs acquire a global lock internally to guarantee thread safety. However, this lock causes unwanted serialization in cases
   where the the applications are submitting tasks to different device handles. If an application was submitting one or more
   tasks in multiple threads and if these submissions are to different device handles and if there is no shared data being provided
   as part of the task information in the respective submissions then applications can specify this flag during submission so
   that the internal lock acquire is skipped. Shared data also includes the input stream in hybrid mode operation. Therefore,
   if the same stream is being used to submit two different tasks and even if the two device handles are different, the usage
   of this flag is invalid.
- [CUDLA_SUBMIT_DIAGNOSTICS_TASK](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggbe4b7a209e3dec6e3e3ef1ad082e1f0d344e7c37d198b239bdad09cca8801b3b), specifies that the submitted task is to run permanent fault diagnostics for DLA HW. User can use this task to probe the
   state of DLA HW. With this flag set, in standalone mode user is not allowed to do event only submissions, where tensor information
   is NULL and only events (wait/signal or both) are present in task. This is because the task always runs on a internally loaded
   diagnostic module. This diagnostic module does not expect any input tensors and so input tensor memory, however user is expected
   to query no. of output tensors, allocate the output tensor memory and pass the same while using the submit task.

###### Returns
[cudlaSuccess](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0af0a9db4b79be4325d9bb957b05314f1),[cudlaErrorInvalidParam](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c80900e54c2a9c02f23935a69f214428),[cudlaErrorInvalidDevice](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04b8ca897368458ceb53b537b8979218b),[cudlaErrorInvalidModule](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d2b4370207a0a2926eda404f880989d6),[cudlaErrorCuda](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0727b16e07c9c63a5f1379826a31eb983),[cudlaErrorUmd](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d22f87ac0d722cd49a74217f472df994),[cudlaErrorOutOfResources](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a05b9d884f3dd0210f0cfa56e064c29dcb),[cudlaErrorInvalidAddress](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03959fb7cceee17b5c54b60cdffa40e7f),[cudlaErrorUnsupportedOperation](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03ba9ef84a3f15d6d1145047e11d1804a),[cudlaErrorInvalidAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0df7b517d9f6d1ca10675c1e988e5101b),[cudlaErrorNvSci](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d3dfc3fc5ad64fbb4b03c3861502667e)[cudlaErrorOs](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04576f8875559f34aeb3865f75d409931)

###### Description
This operation takes in a sequence of tasks and submits them to the DLA HW for execution in the same sequence as they appear
 in the input task array. The input and output tensors (and statistics buffer if used) are assumed to be pre-registered using
[cudlaMemRegister](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1ge07f8bb22373163a0117fc5738a23be0)(in hybrid mode) or[cudlaImportExternalMemory](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1gca69cd7ac008500693ffeedb18d7a9c8)(in standalone mode). Failure to do so can result in this function returning[cudlaErrorInvalidAddress](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03959fb7cceee17b5c54b60cdffa40e7f).

Thestreamparameter must be specified as the CUDA stream on which the DLA task is submitted for execution in hybrid mode. In standalone
 mode, this parameter must be passed as NULL and failure to do so will result in this function returning[cudlaErrorInvalidParam](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c80900e54c2a9c02f23935a69f214428).

The[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)structure has a provision to specify wait and signal events that cuDLA must wait on and signal respectively as part of[cudlaSubmitTask()](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1gc560a614b388d50216bd161c0b3d88cb). Each submitted task will wait for all its wait events to be signaled before beginning execution and will provide a signal
 event (if one is requested for during[cudlaSubmitTask](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1gc560a614b388d50216bd161c0b3d88cb)) that the application (or any other entity) can wait on to ensure that the submitted task has completed execution. In cuDLA
 1.0, only NvSciSync fences are supported as part of wait events. Furthermore, only NvSciSync objects (registered as part of
[cudlaImportExternalSemaphore](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1g12751fbcc295349c16ad3aea0e8bda34)) can be signaled as part of signal events and the fence corresponding to the signaled event is returned as part of[cudlaSubmitTask](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1gc560a614b388d50216bd161c0b3d88cb).

Event-Only submissions - In standalone mode, if inputTensor and outputTensor fields are set to NULL inside the[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)structure, the task submission is interpreted as an enqueue of wait and signal events that must be considered for subsequent
 task submission. No actual task submission is done. Multiple such subsequent task submissions with NULL fields in the input/outputTensor
 fields will overwrite the list of wait and signal events to be considered. In other words, the latest non-null wait events
 and/or latest non-null signal events before a non-null submission are considered for subsequent actual task submission. During
 an actual task submit in standalone mode, the effective wait events and signal events that will be considered are what the
 application sets using NULL data submissions and what is set for that particular task submission in the waitEvents and signalEvents
 fields. The wait events set as part of NULL data submission are considered as dependencies for only the first task and the
 signal events set as part of NULL data submission are signaled when the last task of task list is complete. All constraints
 that apply to waitEvents and signalEvents individually (as described below) are also applicable to the combined list.

Configurations allowed in cuDLA for a task submission in the context[cudlaModuleLoadFromMemory()](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1gcd725924569cec1a3214fd09cb38601d)---------------------------------------------------------------------------- Module Load - Regular/NOOP Task submission -
 Event Only task submission ---------------------------------------------------------------------------- Yes - Allowed - Allowed
 No - Not Allowed - Allowed ----------------------------------------------------------------------------

cuDLA supports 3 kinds of fences - preFence, SOF fence and EOF fence.

- preFence is the type of fence that DLA waits on to start the task execution. Use cudlaFenceType as CUDLA_NVSCISYNC_FENCE to
   mark a fence as preFence.
- SOF(Start Of Frame) fence is the type of fence which is signaled before the task execution on DLA starts. Use cudlaFenceType
   as CUDLA_NVSCISYNC_FENCE_SOF to mark a fence as SOF fence.
- EOF(End Of Frame) fence is the type of fence which is signaled after the task execution on DLA is complete. Use cudlaFenceType
   as CUDLA_NVSCISYNC_FENCE to mark a fence as EOF fence.
For wait events, applications are expected to

- register their synchronization objects using[cudlaImportExternalSemaphore](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1g12751fbcc295349c16ad3aea0e8bda34).
- create the required number of preFence placeholders using[CudlaFence](https://docs.nvidia.com/cuda/cudla-api/index.html#structCudlaFence).
- fill in the placeholders with the relevant fences from the application.
- list out all the fences in[cudlaWaitEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaWaitEvents).
For signal events, applications are expected to

- register their synchronization objects using[cudlaImportExternalSemaphore](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1g12751fbcc295349c16ad3aea0e8bda34).
- create the required number of SOF and EOF fence placeholder fences using[CudlaFence](https://docs.nvidia.com/cuda/cudla-api/index.html#structCudlaFence).
- place the registered objects and the corresponding fences in[cudlaSignalEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents). In case ofdeterministic semaphore, fence is not required to be passed in[cudlaSignalEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents).
  
   When[cudlaSubmitTask](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1gc560a614b388d50216bd161c0b3d88cb)returns successfully, the fences present in[cudlaSignalEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents)can be used to wait for the particular task to be completed. cuDLA supports 1 sync point and any number of semaphores as
   part of[cudlaSignalEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents). If more than 1 sync point is specified,[cudlaErrorInvalidParam](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0c80900e54c2a9c02f23935a69f214428)is returned.
cuDLA adheres to DLA's restriction to support 29 preFences and SOF fences combined together and 29 EOF fences per DLA Task.
During submission, users have an option to enable layerwise statistics profiling for the individual layers of the network.
 This option needs to be exercised by specifying additional output buffers that would contain the profiling information. Specifically,

- "cudlaTask::numOutputTensors" should be the sum of value returned by cudlaModuleGetAttributes(...,CUDLA_NUM_OUTPUT_TENSORS,...)
   and cudlaModuleGetAttributes(...,CUDLA_NUM_OUTPUT_TASK_STATISTICS,...)
- "cudlaTask::outputTensor" should contain the array of output tensors appended with array of statistics output buffer.
This function can return[cudlaErrorUnsupportedOperation](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a03ba9ef84a3f15d6d1145047e11d1804a)if

- stream being used in hybrid mode is in capturing state.
- application attempts to use NvSci functionalities in hybrid mode.
- loading of NvSci libraries failed for a particular platform.
- fence type other than[CUDLA_NVSCISYNC_FENCE](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggfd5fd195c04a78364ab8b66904354bb80c71b2c385910641adc43fa2c987fc68)is specified.
- waitEvents or signaEvents is not NULL in hybrid mode.
- inputTensor or outputTensor is NULL in hybrid mode and the flags are not CUDLA_SUBMIT_DIAGNOSTICS_TASK.
- inputTensor is NULL and outputTensor is not NULL and vice versa in standalone mode and the flags are not CUDLA_SUBMIT_DIAGNOSTICS_TASK.
- inputTensor and outputTensor is NULL and number of tasks is not equal to 1 in standalone mode and the flags are not CUDLA_SUBMIT_DIAGNOSTICS_TASK.
- inputTensor is not NULL or output tensor is NULL and the flags are CUDLA_SUBMIT_DIAGNOSTICS_TASK.
- the effective signal events list has multiple sync points to signal.
- if layerwise feature is unsupported.
- if preFences, SOF fences and EOF fences limit per task is not met.
This function can return[cudlaErrorNvSci](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0d3dfc3fc5ad64fbb4b03c3861502667e)or[cudlaErrorInvalidAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a0df7b517d9f6d1ca10675c1e988e5101b)in certain cases when the underlying NvSci operation fails.

This function can return[cudlaErrorOs](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ggc95f8bcde047e255b11ea1b3b80598a04576f8875559f34aeb3865f75d409931)if an internal system operation fails.

Note:This API can return task execution errors from previous DLA task submissions.

## 2. Data Structures

Here are the data structures with brief descriptions:

[cudlaDevAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaDevAttribute)

[cudlaExternalMemoryHandleDesc](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaExternalMemoryHandleDesc__t)

[cudlaExternalSemaphoreHandleDesc](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaExternalSemaphoreHandleDesc__t)

[CudlaFence](https://docs.nvidia.com/cuda/cudla-api/index.html#structCudlaFence)

[cudlaModuleAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute)

[cudlaModuleTensorDescriptor](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaModuleTensorDescriptor)

[cudlaSignalEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents)

[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)

[cudlaWaitEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaWaitEvents)

### 2.1. cudlaDevAttribute Union Reference
### [Data types used by cuDLA driver]

Device attribute.

#### Public Variables

uint32_tdeviceVersion

uint8_tunifiedAddressingSupported

#### Variables

uint32_t[cudlaDevAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaDevAttribute)::[deviceVersion](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaDevAttribute_133aed5d6be4a007d2eff7dfd942aac55)[inherited]

DLA device version. Xavier has 1.0 and Orin has 2.0.

uint8_t[cudlaDevAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaDevAttribute)::[unifiedAddressingSupported](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaDevAttribute_164d9b42c7cd37db2213b1c57592cc208)[inherited]

Returns 0 if unified addressing is not supported.

### 2.2. cudlaExternalMemoryHandleDesc_t Struct Reference
### [Data types used by cuDLA driver]

External memory handle descriptor.

#### Public Variables

const 
 void 
 *extBufObject

unsigned long longsize

#### Variables

const 
 
 void 
 *[cudlaExternalMemoryHandleDesc_t](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaExternalMemoryHandleDesc__t)::[extBufObject](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaExternalMemoryHandleDesc__t_1a6d1e29a22e55ffdf323bc52e52c3836)[inherited]

A handle representing an external memory object.

unsigned long long[cudlaExternalMemoryHandleDesc_t](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaExternalMemoryHandleDesc__t)::[size](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaExternalMemoryHandleDesc__t_102606c9f4cf2dbe1d0b17da4b29de3b0)[inherited]

Size of the memory allocation

### 2.3. cudlaExternalSemaphoreHandleDesc_t Struct Reference
### [Data types used by cuDLA driver]

External semaphore handle descriptor.

#### Public Variables

const 
 void 
 *extSyncObject

#### Variables

const 
 
 void 
 *[cudlaExternalSemaphoreHandleDesc_t](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaExternalSemaphoreHandleDesc__t)::[extSyncObject](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaExternalSemaphoreHandleDesc__t_17f57bcbdb8b0e8bf0655e14b9965c9aa)[inherited]

A handle representing an external synchronization object.

### 2.4. CudlaFence Struct Reference
### [Data types used by cuDLA driver]

Fence description.

#### Public Variables

 void 
 *fence

[cudlaFenceType](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gfd5fd195c04a78364ab8b66904354bb8)type

#### Variables

 
 void 
 *[CudlaFence](https://docs.nvidia.com/cuda/cudla-api/index.html#structCudlaFence)::[fence](https://docs.nvidia.com/cuda/cudla-api/index.html#structCudlaFence_12776eab9e902b8db4863097634b35a15)[inherited]

Fence.

[cudlaFenceType](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1gfd5fd195c04a78364ab8b66904354bb8)[CudlaFence](https://docs.nvidia.com/cuda/cudla-api/index.html#structCudlaFence)::[type](https://docs.nvidia.com/cuda/cudla-api/index.html#structCudlaFence_1c7cf9a34d20dea7e789be4e5b9ef21fe)[inherited]

Fence type.

### 2.5. cudlaModuleAttribute Union Reference
### [Data types used by cuDLA driver]

Module attribute.

#### Public Variables

[cudlaModuleTensorDescriptor](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaModuleTensorDescriptor)
 *inputTensorDesc

uint32_tnumInputTensors

uint32_tnumOutputTensors

[cudlaModuleTensorDescriptor](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaModuleTensorDescriptor)
 *outputTensorDesc

#### Variables

[cudlaModuleTensorDescriptor](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaModuleTensorDescriptor)
 *[cudlaModuleAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute)::[inputTensorDesc](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute_13c7a3e737a6b176e6e9c70988022d5ec)[inherited]

Returns an array of input tensor descriptors.

uint32_t[cudlaModuleAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute)::[numInputTensors](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute_1589db412b98fdf2da4fd6c30391dda4a)[inherited]

Returns the number of input tensors.

uint32_t[cudlaModuleAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute)::[numOutputTensors](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute_13fb1292746718ce5f18cec46a9e63b86)[inherited]

Returns the number of output tensors.

[cudlaModuleTensorDescriptor](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaModuleTensorDescriptor)
 *[cudlaModuleAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute)::[outputTensorDesc](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute_12eaa15b61ffd7e6e68c8889305c5826f)[inherited]

Returns an array of output tensor descriptors.

### 2.6. cudlaModuleTensorDescriptor Struct Reference
### [Data types used by cuDLA driver]

Tensor descriptor.

### 2.7. cudlaSignalEvents Struct Reference
### [Data types used by cuDLA driver]

Signal events for[cudlaSubmitTask](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1gc560a614b388d50216bd161c0b3d88cb)

#### Public Variables

const 
 
 
 *devPtrs

[CudlaFence](https://docs.nvidia.com/cuda/cudla-api/index.html#structCudlaFence)
 *eofFences

uint32_tnumEvents

#### Variables

const 
 
 
 
 *[cudlaSignalEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents)::[devPtrs](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents_1754b5a36ca8b858395c02fadc6492ce1)[inherited]

Array of registered synchronization objects (via[cudlaImportExternalSemaphore](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1g12751fbcc295349c16ad3aea0e8bda34)). 

[CudlaFence](https://docs.nvidia.com/cuda/cudla-api/index.html#structCudlaFence)
 *[cudlaSignalEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents)::[eofFences](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents_1be21907c65bc41726c111405c376c3ac)[inherited]

Array of fences pointers for all the signal events corresponding to the synchronization objects.

uint32_t[cudlaSignalEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents)::[numEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents_1723456f4fef95443670e308dcfc94845)[inherited]

Total number of signal events.

### 2.8. cudlaTask Struct Reference
### [Data types used by cuDLA driver]

Structure of Task.

#### Public Variables

const 
 
 
 *inputTensor

[cudlaModule](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ge3ecb829f32791357568b7d005d107a5)moduleHandle

uint32_tnumInputTensors

uint32_tnumOutputTensors

const 
 
 
 *outputTensor

[cudlaSignalEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents)
 *signalEvents

const 
[cudlaWaitEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaWaitEvents)
 *waitEvents

#### Variables

const 
 
 
 
 *[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)::[inputTensor](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask_1dd4cb87ea94c16385134477170c40aef)[inherited]

Array of input tensors.

[cudlaModule](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__TYPES_1ge3ecb829f32791357568b7d005d107a5)[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)::[moduleHandle](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask_1f61768f306610f76fe18ca2ef0bafd78)[inherited]

cuDLA module handle.

uint32_t[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)::[numInputTensors](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask_17003a7c700c41d5919b947f0bc4caf33)[inherited]

Number of input tensors.

uint32_t[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)::[numOutputTensors](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask_1f05bb162c3f14bb8b012d0f9f60ce9ac)[inherited]

Number of output tensors.

const 
 
 
 
 *[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)::[outputTensor](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask_192e99490ea0198e094233cf758a96e47)[inherited]

Array of output tensors.

[cudlaSignalEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents)
 *[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)::[signalEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask_1c8f145341906339b3cffdeda9db30d3c)[inherited]

Signal events.

const 
 
[cudlaWaitEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaWaitEvents)
 *[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)::[waitEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask_1d1b971c2701c6292b7fe894d251d5848)[inherited]

Wait events.

### 2.9. cudlaWaitEvents Struct Reference
### [Data types used by cuDLA driver]

Wait events for[cudlaSubmitTask](https://docs.nvidia.com/cuda/cudla-api/index.html#group__CUDLA__API_1gc560a614b388d50216bd161c0b3d88cb). 

#### Public Variables

uint32_tnumEvents

const 
[CudlaFence](https://docs.nvidia.com/cuda/cudla-api/index.html#structCudlaFence)
 *preFences

#### Variables

uint32_t[cudlaWaitEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaWaitEvents)::[numEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaWaitEvents_1103e821b8ab32dfb82d20818a8467882)[inherited]

Total number of wait events.

const 
 
[CudlaFence](https://docs.nvidia.com/cuda/cudla-api/index.html#structCudlaFence)
 *[cudlaWaitEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaWaitEvents)::[preFences](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaWaitEvents_1fe868337134eb67e12a2a30679976f0c)[inherited]

Array of fence pointers for all the wait events.

## 3. Data Fields

Here is a list of all documented struct and union fields with links to the struct/union documentation for each field:

deviceVersion
[cudlaDevAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaDevAttribute)
devPtrs
[cudlaSignalEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents)
eofFences
[cudlaSignalEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents)
extBufObject
[cudlaExternalMemoryHandleDesc](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaExternalMemoryHandleDesc__t)
extSyncObject
[cudlaExternalSemaphoreHandleDesc](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaExternalSemaphoreHandleDesc__t)
fence
[CudlaFence](https://docs.nvidia.com/cuda/cudla-api/index.html#structCudlaFence)
inputTensor
[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)
inputTensorDesc
[cudlaModuleAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute)
moduleHandle
[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)
numEvents
[cudlaWaitEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaWaitEvents)
[cudlaSignalEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaSignalEvents)
numInputTensors
[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)
[cudlaModuleAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute)
numOutputTensors
[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)
[cudlaModuleAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute)
outputTensor
[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)
outputTensorDesc
[cudlaModuleAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaModuleAttribute)
preFences
[cudlaWaitEvents](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaWaitEvents)
signalEvents
[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)
size
[cudlaExternalMemoryHandleDesc](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaExternalMemoryHandleDesc__t)
type
[CudlaFence](https://docs.nvidia.com/cuda/cudla-api/index.html#structCudlaFence)
unifiedAddressingSupported
[cudlaDevAttribute](https://docs.nvidia.com/cuda/cudla-api/index.html#unioncudlaDevAttribute)
waitEvents
[cudlaTask](https://docs.nvidia.com/cuda/cudla-api/index.html#structcudlaTask)

## Notices

### Notice
This document is provided for information
 purposes only and shall not be regarded as a warranty of a
 certain functionality, condition, or quality of a product.
 NVIDIA Corporation (“NVIDIA”) makes no representations or
 warranties, expressed or implied, as to the accuracy or
 completeness of the information contained in this document
 and assumes no responsibility for any errors contained
 herein. NVIDIA shall have no liability for the consequences
 or use of such information or for any infringement of
 patents or other rights of third parties that may result
 from its use. This document is not a commitment to develop,
 release, or deliver any Material (defined below), code, or
 functionality.

NVIDIA reserves the right to make corrections, modifications,
 enhancements, improvements, and any other changes to this
 document, at any time without notice.

Customer should obtain the latest relevant information before
 placing orders and should verify that such information is
 current and complete.

NVIDIA products are sold subject to the NVIDIA standard terms and
 conditions of sale supplied at the time of order
 acknowledgement, unless otherwise agreed in an individual
 sales agreement signed by authorized representatives of
 NVIDIA and customer (“Terms of Sale”). NVIDIA hereby
 expressly objects to applying any customer general terms and
 conditions with regards to the purchase of the NVIDIA
 product referenced in this document. No contractual
 obligations are formed either directly or indirectly by this
 document.

NVIDIA products are not designed, authorized, or warranted to be
 suitable for use in medical, military, aircraft, space, or
 life support equipment, nor in applications where failure or
 malfunction of the NVIDIA product can reasonably be expected
 to result in personal injury, death, or property or
 environmental damage. NVIDIA accepts no liability for
 inclusion and/or use of NVIDIA products in such equipment or
 applications and therefore such inclusion and/or use is at
 customer’s own risk.

NVIDIA makes no representation or warranty that products based on
 this document will be suitable for any specified use.
 Testing of all parameters of each product is not necessarily
 performed by NVIDIA. It is customer’s sole responsibility to
 evaluate and determine the applicability of any information
 contained in this document, ensure the product is suitable
 and fit for the application planned by customer, and perform
 the necessary testing for the application in order to avoid
 a default of the application or the product. Weaknesses in
 customer’s product designs may affect the quality and
 reliability of the NVIDIA product and may result in
 additional or different conditions and/or requirements
 beyond those contained in this document. NVIDIA accepts no
 liability related to any default, damage, costs, or problem
 which may be based on or attributable to: (i) the use of the
 NVIDIA product in any manner that is contrary to this
 document or (ii) customer product designs.

No license, either expressed or implied, is granted under any NVIDIA
 patent right, copyright, or other NVIDIA intellectual
 property right under this document. Information published by
 NVIDIA regarding third-party products or services does not
 constitute a license from NVIDIA to use such products or
 services or a warranty or endorsement thereof. Use of such
 information may require a license from a third party under
 the patents or other intellectual property rights of the
 third party, or a license from NVIDIA under the patents or
 other intellectual property rights of NVIDIA.

Reproduction of information in this document is permissible only if
 approved in advance by NVIDIA in writing, reproduced without
 alteration and in full compliance with all applicable export
 laws and regulations, and accompanied by all associated
 conditions, limitations, and notices.

THIS DOCUMENT AND ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE
 BOARDS, FILES, DRAWINGS, DIAGNOSTICS, LISTS, AND OTHER
 DOCUMENTS (TOGETHER AND SEPARATELY, “MATERIALS”) ARE BEING
 PROVIDED “AS IS.” NVIDIA MAKES NO WARRANTIES, EXPRESSED,
 IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO THE
 MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF
 NONINFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A
 PARTICULAR PURPOSE. TO THE EXTENT NOT PROHIBITED BY LAW, IN
 NO EVENT WILL NVIDIA BE LIABLE FOR ANY DAMAGES, INCLUDING
 WITHOUT LIMITATION ANY DIRECT, INDIRECT, SPECIAL,
 INCIDENTAL, PUNITIVE, OR CONSEQUENTIAL DAMAGES, HOWEVER
 CAUSED AND REGARDLESS OF THE THEORY OF LIABILITY, ARISING
 OUT OF ANY USE OF THIS DOCUMENT, EVEN IF NVIDIA HAS BEEN
 ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. Notwithstanding
 any damages that customer might incur for any reason
 whatsoever, NVIDIA’s aggregate and cumulative liability
 towards customer for the products described herein shall be
 limited in accordance with the Terms of Sale for the
 product.

### OpenCL
OpenCL is a trademark of Apple Inc. used under license to the Khronos Group Inc.

### Trademarks
NVIDIA and the NVIDIA logo are trademarks or registered trademarks of NVIDIA Corporation
 in the U.S. and other countries. Other company and product names may be trademarks of
 the respective companies with which they are associated.

### Copyright
©2021-2024NVIDIA Corporation &
 affiliates. All rights reserved.

This product includes software developed by the Syncro Soft SRL (http://www.sync.ro/).

---