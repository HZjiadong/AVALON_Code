#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <time.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas.h>
#include <cublas_v2.h>
using namespace std;

//Declaration des functions 
int gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n, cudaGraph_t graph);
void print_matrix(const double *A, int nr_rows_A, int nr_cols_A);
timespec time_diff(timespec start, timespec end);
double time_to_double(timespec time);
int dimension;

// If the follow line is uncommented, than kernal number = 1
// #define USE_MMUL_1_KERNEL 
#ifdef USE_MMUL_1_KERNEL
  // this case is not use
#else
  // global variable BS block size
  int BS;
#endif

//Macro Cuda error check, check error message during a cuda launch or cuda api call
#define checkCudaErrors(e) {                                        \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(-1); \
 }                                                                 \
}

#define CUBLAS_API_H_
#ifdef CUBLAS_API_H_
//cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif 

//Macro Cublas error check, check error message during a Cublas launch or cuda api call
#define checkCublasErrors(e) {                                        \
 if(e!=CUBLAS_STATUS_SUCCESS) {                                              \
   printf("Cuda failure %i:%s:%s:%d \n",e,_cudaGetErrorEnum(e),__FILE__,__LINE__);           \
   exit(-1); \
 }                                                               \
}

//Main function
int main(int argc, char *argv[]) {
    //dimensions of 3 matrix
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

    // Dimension check for std::in
    if (argc <= 1) {
        cout << "DIMENSION WAS NOT FOUND!" << endl;
    }
    else{
        dimension = atoi(argv[1]);
    }

    // Blocksize check for std::in
#ifndef USE_MMUL_1_KERNEL 
    if (argc <= 2) {
        cout << "BLOCK SIZE WAS NOT FOUND!" << endl;
        return -1; 
    }
    BS = atoi(argv[2]);
#endif

    //For now, there are only square matrix
    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dimension;

    //size_t == unsigned long long (64 bits)
    size_t matrix_size_A = (size_t)nr_rows_A * (size_t)nr_cols_A * sizeof(double);
    size_t matrix_size_B = (size_t)nr_rows_B * (size_t)nr_cols_B * sizeof(double);
    size_t matrix_size_C = (size_t)nr_rows_C * (size_t)nr_cols_C * sizeof(double); 

    //Allocate 3 arrays on CPU 
    double *h_A = (double *)malloc(matrix_size_A);
    double *h_B = (double *)malloc(matrix_size_B);
    double *h_C = (double *)malloc(matrix_size_C);

    //Allocate 3 arrays on GPU
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,matrix_size_A);
    cudaMalloc(&d_B,matrix_size_B);
    cudaMalloc(&d_C,matrix_size_C);

    //Initialization of matrix A and B
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < dimension; i ++){
        for (int j = 0; j < dimension; j ++){
            // passer colonne i vers i+1, il faut sauter (dans le stockage) LED( ici est la dimension ).
            h_A[i+ dimension*j] = 1.0/(1+i+j);
            h_B[i+ dimension*j] = 2.0/(1+i+j);
            h_C[i+ dimension*j] = 0.0;
        }
    }

    //Fill matrix A and B on CPU & copy them on GPU
    cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,h_C,nr_rows_C * nr_cols_C * sizeof(double),cudaMemcpyHostToDevice);
    
    //csv file object for graph explicite api without dependency
    ofstream cudaGraphDependencyCsv;
    cudaGraphDependencyCsv.open("cudaGraphDependency.csv", ofstream::out | ofstream::app);
    cudaGraphDependencyCsv << "time" << "," << "kernel" << "," << "dimension" << "," << "blocksize" << "," << "operation" << "," << endl;
    double dependencyApiTime;

    // Trackers
    timespec start_time, end_time;
    int call_kernel_number;
    string operation_type;
    bool cudagraph;

    //Create the graph
    //Call function 
    //Output data
    cudaGraph_t graph;
    checkCudaErrors(cudaGraphCreate(&graph, 0));

for (int k=0; k < 200; k ++){
    clock_gettime(CLOCK_REALTIME, &start_time);
    call_kernel_number = gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B, graph);
    clock_gettime(CLOCK_REALTIME, &end_time);
    dependencyApiTime = time_to_double(time_diff(start_time, end_time));
    operation_type = "graphApiYesDependency";
    cudagraph = 1;
    cudaGraphDependencyCsv << dependencyApiTime << "," << call_kernel_number << "," << dimension << "," << BS << "," << operation_type << "," << cudagraph << endl;
    printf("Elapsed time for execution with cuda graph explicit API with dependency:%f (s)\n", time_to_double(time_diff(start_time, end_time)));
}
    //End of cuda function call    
    cudaGraphDependencyCsv.close();
    cudaGraphDependencyCsv.clear();
    
    
    //Copy (and print) the result on host memory
    checkCudaErrors(cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(double),cudaMemcpyDeviceToHost));
    if (dimension <= 16){
        cout << "C =" << endl;
        print_matrix(h_C, nr_rows_C, nr_cols_C);
    }    
    
    //Distroy the handle
    checkCudaErrors(cudaGraphDestroy(graph));

    //Free GPU memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    //Free CPU memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

/*
cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc)
*/
//Multiply the arrays A and B on GPU and save the result in C
//C(m,n) = A(m,k) * B(k,n)
int gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n, cudaGraph_t graph){
    int lda=m, ldb=k, ldc=m;
    const double alf = 1.5;
    const double bet = 0.5;
    const double *alpha = &alf;
    const double *beta = &bet;
    int kernal_number = 0;

#ifdef USE_MMUL_1_KERNEL
    //Do the actual multiplication
    //Cuda stream is branched with cuda handle, so gemm will also be done in(?) the stream created
    checkCublasErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    kernal_number = 1;
return kernal_number;

#else
    //Use global variable BS for the block size.
    // if BS does not divide dimensions, then abort the processus
    if ( BS == 0 ) 
    {
        exit( -1 );
    }
    if ( (m % BS) || (k%BS) || (n%BS)) 
    {
      cout << "*** error: dimension m,n,k=(" << m << ',' << n << ',' << k << ") must be a multiple of BS=" << BS << endl;
      abort();
    }

    cudaStream_t tempStream;
    cublasHandle_t tempHandle;
    checkCublasErrors(cublasCreate(&tempHandle));
    checkCudaErrors(cudaStreamCreate(&tempStream));
    cudaGraphNode_t* prev = 0; // previous graph node

    for (int i=0; i<m; i+=BS)
    {
        for (int j=0; j<n; j+=BS)
        {
            // 1/ At the begin add a kernel for Cij += beta*Cij
            // add graph node for "Cij += beta*Cij"
            cudaGraphNode_t* nodeCij = new cudaGraphNode_t;

            for (int k=0; k<n; k+=BS)
            {
            // A,B,C have column major storage. 
            // With such storage element (ik,kj) of the matrix A is A[i+k*lda]
            // With such storage element (ik,kj) of the matrix B is B[k+j*lda]
            // Note that: Aik = A+i+k*lda = &A[i]   or Bkj = B+k+j*ldb = &B[j*ldb] 
            // launch a kernel to do Cij += alpha*Aik*Bkj + beta*Cij
            
            const double* Aik = A+i+k*lda;
            const double* Bkj = B+k+j*ldb;
            double* Cij = C+i+j*ldc;
            
            // 2/ For each k, launch a kernel to do Cij += alpha*Aik*Bkj
            //   Once kernel and the graph node for k+1 is created, add a dependence between the node at iteration "k+1" with node at iteration k
            // Note that the graph node for the kernel launched by cublasDgemm should be captured (then added to the graph with cudaGraphAddChildGraphNode)
            
            checkCublasErrors(cublasDgemm(tempHandle, CUBLAS_OP_N, CUBLAS_OP_N, BS, BS, BS, alpha, Aik, lda, Bkj, ldb, beta, Cij, ldc));
            
            // add graph node for "Cij += alpha*Aik*Bkj"
            cudaGraphNode_t* curr = new cudaGraphNode_t;
            /* use capture stream API here => cublasDgemm captured in tempGraph */
            cudaGraph_t tempGraph;    
            checkCublasErrors(cublasSetStream( tempHandle, tempStream)); 

            checkCudaErrors(cudaStreamBeginCapture(tempStream, cudaStreamCaptureModeGlobal));
            kernal_number = cublasDgemm(tempHandle, CUBLAS_OP_N, CUBLAS_OP_N, BS, BS, k, alpha, Aik, lda, Bkj, ldb, 0, Cij, ldc); //cette function est une stream cuda, not C++ stream!
            checkCudaErrors(cudaStreamEndCapture(tempStream, &tempGraph));
            /// Third parameter is const cudaGraphNode_t* pDependencies, what happens here?            
            checkCudaErrors(cudaGraphAddChildGraphNode (curr, graph, 0, 0, tempGraph));
            if (prev == 0)
            {
                cudaGraphAddDependencies ( graph, nodeCij, curr, 1 );
            }
            else{
                cudaGraphAddDependencies ( graph, prev, curr, 1 );
            }
            prev = curr;
            /// Here has a index problem, doesn't link current node with node of last iteration. 
            kernal_number = kernal_number + 1;
            }
        }
    }
return kernal_number;
#endif
}

//Print matrix A( nr_rows_A ) storage in column-major format
void print_matrix(const double *A, int nr_rows_A, int nr_cols_A) {
    for (int i = 0; i < nr_rows_A; ++i){
        for(int j=0; j<nr_cols_A; ++j){
            cout << A[j * nr_rows_A + i] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

//Get time used by two time labels of type "timespec": label 'start' and label 'end'
timespec time_diff(timespec start, timespec end){
    timespec temp;
    if ((end.tv_nsec - start.tv_nsec) < 0 ){
        temp.tv_sec = end.tv_sec - start.tv_sec - 1 ;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
        }
        return temp;
}

double time_to_double(timespec time){
    double double_time;
    double_time = double(time.tv_sec) + double(time.tv_nsec) * 1e-9; 
    return double_time; 
}
    