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
int gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n, cublasHandle_t handle);
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
//CAN BE INPROVED BY "CublasErrorCheck.c"
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
    
    //Create a handle for CUBLAS
    cublasHandle_t handle;
    checkCublasErrors(cublasCreate(&handle));

    //csv file object
    ofstream captureTimeCsv;
    captureTimeCsv.open("captureTime.csv", ofstream::out | ofstream::app);
    captureTimeCsv << "index" << "," << "time" << "," << "kernel" << "," << "dimension" << "," << "blocksize" << "," << "operation" << endl;
    double captureTime;

    ofstream instantiationTimeCsv;
    instantiationTimeCsv.open("instantiationTime.csv",ofstream::out | ofstream::app);
    instantiationTimeCsv << "index" << "," << "time" << "," << "kernel" << "," << "dimension" << "," << "blocksize" << "," << "operation" << endl;
    double instantiationTime;

    ofstream launchingTimeCsv;
    launchingTimeCsv.open("launchingTime.csv",ofstream::out | ofstream::app);
    launchingTimeCsv << "index" << "," << "time" << "," << "kernel" << "," << "dimension" << "," << "blocksize" << "," << "operation" << endl;
    double launchingTime;

    // Capture & Instantiation loop
    for (int j = 0 ; j < 10; j ++)
    {   
        /* plural stream creation segment
        creation of cuda stream through function cudaStreamCreate ( cudaStream_t* pStream )
        allocate and initialize an array of stream handles
        cudaStream_t *stream = (cudaStream_t *)malloc(nstream * sizeof(cudaStream_t));
        checkCudaErrors(cudaStreamCreate(&(stream[i])));
        */

        // CUDA Stream & Handler 
        cudaStream_t stream;
        checkCudaErrors(cudaStreamCreate(&stream));
        checkCublasErrors(cublasSetStream( handle, stream)); 

        // Time Tracker
        timespec start_time, end_time;

        // Kernel Number Tracker 
        int call_kernel_number;

        // Operation Type Tracker
        string operation_type;

        //Initialization of cuda graph
        bool graphCreated = false;
        clock_gettime(CLOCK_REALTIME, &start_time);
        cudaGraph_t graph;
        cudaGraphExec_t instance;
        //  Begin Caputure
        checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        call_kernel_number = gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B, handle); //cette function est une stream cuda, not C++ stream!
        //  End Capture
        checkCudaErrors(cudaStreamEndCapture(stream, &graph));
        clock_gettime(CLOCK_REALTIME, &end_time);
        captureTime = time_to_double(time_diff(start_time, end_time));
        operation_type = "capture";
        captureTimeCsv << j << "," << captureTime << "," << call_kernel_number << "," << dimension << "," << BS << "," << operation_type << endl;
        printf("Elapsed time for graph capture:%f (s)\n", time_to_double(time_diff(start_time, end_time)));

        //  Instantiate
        clock_gettime(CLOCK_REALTIME, &start_time);
        checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
        clock_gettime(CLOCK_REALTIME, &end_time);
        instantiationTime = time_to_double(time_diff(start_time, end_time));
        operation_type = "instantiation";
        instantiationTimeCsv << j << "," << instantiationTime << "," << call_kernel_number << "," << dimension << "," << BS << "," << operation_type << endl;
        printf("Elapsed time for graph instantiation:%f (s)\n", time_to_double(time_diff(start_time, end_time)));

        // Graph Launch loop 
        for (int k=0; k < 20; k ++){
            clock_gettime(CLOCK_REALTIME, &start_time);
            //lanch the cuda graph
            checkCudaErrors(cudaGraphLaunch(instance, stream));
            checkCudaErrors(cudaStreamSynchronize(stream));
            clock_gettime(CLOCK_REALTIME, &end_time);
            //high resolution timer
            launchingTime = time_to_double(time_diff(start_time, end_time));
            int index = j * 20 + k;
            operation_type = "launch";
            launchingTimeCsv << index << "," << launchingTime << "," << call_kernel_number << "," << dimension << "," << BS << "," << operation_type << endl;
            printf("Elapsed time for execution:%f (s)\n", time_to_double(time_diff(start_time, end_time)));
        }
    }
    launchingTimeCsv.close();
    launchingTimeCsv.clear(); 
    captureTimeCsv.close();
    captureTimeCsv.clear();
    instantiationTimeCsv.close();
    instantiationTimeCsv.clear();
    
    //Copy (and print) the result on host memory
    checkCudaErrors(cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(double),cudaMemcpyDeviceToHost));
    if (dimension <= 16){
        cout << "C =" << endl;
        print_matrix(h_C, nr_rows_C, nr_cols_C);
    }    
    
    //Distroy the handle
    checkCublasErrors(cublasDestroy(handle));

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
int gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n, cublasHandle_t handle){
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
    for (int i=0; i<m; i+=BS)
      for (int j=0; j<n; j+=BS)
      {
        // A,B,C have column major storage. With such storage element (i,j) of the matrix A is A[i+j*lda]
        // launch a kernel to do Cij += alpha*Ai0*B0j + beta*Cij
        // Note that: Ai0 = A+i = &A[i]   or B0j = &B[j*ldb]
        const double* Ai0 = A+i;
        const double* B0j = B+j*ldb;
        double* Cij = C+i+j*ldc;
        checkCublasErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, BS, BS, k, alpha, Ai0, lda, B0j, ldb, beta, Cij, ldc));
        kernal_number = kernal_number + 1;
      }
return kernal_number;
#endif
}

/*
This code create two random matrix directly on GPU, maybe will be useful later.
But here we will not use this directly on GPU, but build matrix on CPU and copy them into GPU.
Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU*/
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
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
    