#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DATA_SIZE 1048576
#define NSTEP 1000
#define NKERNEL 20

int data[DATA_SIZE];

int main(){
    if(!InitCUDA()) {
        return  0;
    }
    printf( " CUDA initialized.\n ");

        //Creation of CUDA Graph
    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    
    for(int istep=0; istep<NSTEP; istep++){
        if(!graphCreated){
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
                shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
            }
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated=true;
        }
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);
    }

    return  0;
}

// function tests if the machine has a CUDA support
bool InitCUDA()
{
    int count;
    cudaGetDeviceCount(&count);
    if(count ==  0) {
        fprintf(stderr,  " There is no device.\n ");
        return  false;
    }
    int i;
    for(i =  0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >=  1) {
                break;
            }
        }
    }
    if(i == count) {
        fprintf(stderr,  " There is no device supporting CUDA 1.x.\n ");
        return  false;
    }
    cudaSetDevice(i);
    return  true;
}

void GenerateNumbers( int *number,  int size)
{
     for( int i =  0; i < size; i++) {
        number[i] = rand() %  10;
    }
}
