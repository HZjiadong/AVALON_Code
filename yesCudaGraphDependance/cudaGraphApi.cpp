//*****//
// 创建cuda graph / nodes / nodes' dependencies的代码
// Create the graph - it starts out empty
cudaGraphCreate(&graph, 0);

// For the purpose of this example, we'll create
// the nodes separately from the dependencies to
// demonstrate that it can be done in two stages.
// Note that dependencies can also be specified 
// at node creation. 
cudaGraphAddKernelNode(&a, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&b, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&c, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&d, graph, NULL, 0, &nodeParams);

// Now set up dependencies on each node
cudaGraphAddDependencies(graph, &a, &b, 1);     // A->B
cudaGraphAddDependencies(graph, &a, &c, 1);     // A->C
cudaGraphAddDependencies(graph, &b, &d, 1);     // B->D
cudaGraphAddDependencies(graph, &c, &d, 1);     // C->D
//*****// 



//*****//
// 目前为止产生的CudaGraph都是通过stream cpature
// 以下是一段通过stream capture进行Cuda Graph创建的示例
cudaGraph_t graph;

cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);

cudaStreamEndCapture(stream, &graph);
//*****// 



//*****// 
// 以下代码阐述了如何设置stream之间的依赖关系
// 但貌似version 3.0需要设置的是nodes之间的依赖关系，至于nodes中的stream，它们计算是相对独立的？
// stream1 is the origin stream
cudaStreamBeginCapture(stream1);

kernel_A<<< ..., stream1 >>>(...);

// Fork into stream2
cudaEventRecord(event1, stream1);
cudaStreamWaitEvent(stream2, event1);

kernel_B<<< ..., stream1 >>>(...);
kernel_C<<< ..., stream2 >>>(...);

// Join stream2 back to origin stream (stream1)
cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);

kernel_D<<< ..., stream1 >>>(...);

// End capture in the origin stream
cudaStreamEndCapture(stream1, &graph);

// stream1 and stream2 no longer in capture mode  
//*****// 



cudaEventRecord 
*********************************************************************
__host__​__device__​cudaError_t cudaEventRecord ( cudaEvent_t event, cudaStream_t stream = 0 )
Records an event.
Parameters
event
- Event to record
stream
- Stream in which to record event
Returns
cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure

Description
Captures in event the contents of stream at the time of this call. event and stream must be on the same CUDA context. Calls such as cudaEventQuery() or cudaStreamWaitEvent() will then examine or wait for completion of the work that was captured. Uses of stream after this call do not modify event. See note on default stream behavior for what is captured in the default case.

cudaEventRecord() can be called multiple times on the same event and will overwrite the previously captured state. Other APIs such as cudaStreamWaitEvent() use the most recently captured state at the time of the API call, and are not affected by later calls to cudaEventRecord(). Before the first call to cudaEventRecord(), an event represents an empty set of work, so for example cudaEventQuery() would return cudaSuccess.

Note:
This function uses standard default stream semantics.

Note that this function may also return error codes from previous, asynchronous launches.

Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.
------------------------------------------------------------------


cudaStreamWaitEvent
********************************************************************
__host__​__device__​cudaError_t cudaStreamWaitEvent ( cudaStream_t stream, cudaEvent_t event, unsigned int  flags = 0 )
Make a compute stream wait on an event.
Parameters
stream
- Stream to wait
event
- Event to wait on
flags
- Parameters for the operation(See above)
Returns
cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

Description
Makes all future work submitted to stream wait for all work captured in event. See cudaEventRecord() for details on what is captured by an event. The synchronization will be performed efficiently on the device when applicable. event may be from a different device than stream.

flags include:

cudaEventWaitDefault: Default event creation flag.

cudaEventWaitExternal: Event is captured in the graph as an external event node when performing stream capture.

Note:
This function uses standard default stream semantics.

Note that this function may also return error codes from previous, asynchronous launches.

Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.
--------------------------------------------------------------------


cudaGraphAddKernelNode
***************************************************************
__host__​cudaError_t cudaGraphAddKernelNode ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams )
Creates a kernel execution node and adds it to a graph.
Parameters
pGraphNode
- Returns newly created node
graph
- Graph to which to add the node
pDependencies
- Dependencies of the node
numDependencies
- Number of dependencies
pNodeParams
- Parameters for the GPU execution node
Returns
cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDeviceFunction

Description
Creates a new kernel execution node and adds it to graph with numDependencies dependencies specified via pDependencies and arguments specified in pNodeParams. It is possible for numDependencies to be 0, in which case the node will be placed at the root of the graph. pDependencies may not have any duplicate entries. A handle to the new node will be returned in pGraphNode.

The cudaKernelNodeParams structure is defined as:

‎  struct cudaKernelNodeParams
        {
            void* func;
            dim3 gridDim;
            dim3 blockDim;
            unsigned int sharedMemBytes;
            void **kernelParams;
            void **extra;
        };
When the graph is launched, the node will invoke kernel func on a (gridDim.x x gridDim.y x gridDim.z) grid of blocks. Each block contains (blockDim.x x blockDim.y x blockDim.z) threads.

sharedMem sets the amount of dynamic shared memory that will be available to each thread block.

Kernel parameters to func can be specified in one of two ways:

1) Kernel parameters can be specified via kernelParams. If the kernel has N parameters, then kernelParams needs to be an array of N pointers. Each pointer, from kernelParams[0] to kernelParams[N-1], points to the region of memory from which the actual parameter will be copied. The number of kernel parameters and their offsets and sizes do not need to be specified as that information is retrieved directly from the kernel's image.

2) Kernel parameters can also be packaged by the application into a single buffer that is passed in via extra. This places the burden on the application of knowing each kernel parameter's size and alignment/padding within the buffer. The extra parameter exists to allow this function to take additional less commonly used arguments. extra specifies a list of names of extra settings and their corresponding values. Each extra setting name is immediately followed by the corresponding value. The list must be terminated with either NULL or CU_LAUNCH_PARAM_END.

CU_LAUNCH_PARAM_END, which indicates the end of the extra array;

CU_LAUNCH_PARAM_BUFFER_POINTER, which specifies that the next value in extra will be a pointer to a buffer containing all the kernel parameters for launching kernel func;

CU_LAUNCH_PARAM_BUFFER_SIZE, which specifies that the next value in extra will be a pointer to a size_t containing the size of the buffer specified with CU_LAUNCH_PARAM_BUFFER_POINTER;

The error cudaErrorInvalidValue will be returned if kernel parameters are specified with both kernelParams and extra (i.e. both kernelParams and extra are non-NULL).

The kernelParams or extra array, as well as the argument values it points to, are copied during this call.

Note:
Kernels launched using graphs must not use texture and surface references. Reading or writing through any texture or surface reference is undefined behavior. This restriction does not apply to texture and surface objects.

Note:
Graph objects are not threadsafe. More here.

Note that this function may also return error codes from previous, asynchronous launches.

Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.



cudaGraphAddDependencies
**************************************************************
__host__​cudaError_t cudaGraphAddDependencies ( cudaGraph_t graph, const cudaGraphNode_t* from, const cudaGraphNode_t* to, size_t numDependencies )
Adds dependency edges to a graph.
Parameters
graph
- Graph to which dependencies are added
from
- Array of nodes that provide the dependencies
to
- Array of dependent nodes
numDependencies
- Number of dependencies to be added
Returns
cudaSuccess, cudaErrorInvalidValue

Description
The number of dependencies to be added is defined by numDependencies Elements in pFrom and pTo at corresponding indices define a dependency. Each node in pFrom and pTo must belong to graph.

If numDependencies is 0, elements in pFrom and pTo will be ignored. Specifying an existing dependency will return an error.

Note:
Graph objects are not threadsafe. More here.

Note that this function may also return error codes from previous, asynchronous launches.

Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.
