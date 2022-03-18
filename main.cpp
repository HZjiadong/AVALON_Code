#include <iostream>
// Output CSV
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <time.h>
// System includes
#include <stdio.h>
#include <assert.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas.h>
#include <cublas_v2.h>
using namespace std;

//Declaration des functions utiliser
void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n, cublasHandle_t handle);
void print_matrix(const double *A, int nr_rows_A, int nr_cols_A);
timespec time_diff(timespec start, timespec end);
double time_to_double(timespec time);

// If the following macro is define, 
// then the code is compile to generate only 1 kernel per call to gpu_blas_mmul
// else the code compile the multi-kernel matrix multiplication 
// To undefined the macro USE_MMUL_1_KERNEL, comment the line !
#define USE_MMUL_1_KERNEL 

#ifdef USE_MMUL_1_KERNEL
  // this case is not use
#else
  // global variable BS block size
  int BS;
#endif

//Macro Cuda error check
//Macro for checking cuda errors following a cuda launch or api call
#define checkCudaErrors(e) {                                        \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

//Main function
int main(int argc, char *argv[]) {
    //dimensions of 3 matrix
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
    int dimension;

    //get information from standard input
    if (argc <= 1) {
        cout << "NUMBER OF DIMENSION WAS NOT FOUND!" << endl;
    }
    else{
        dimension = atoi(argv[1]);
    }

#ifndef USE_MMUL_1_KERNEL 
    if (argc <= 2) {
        cout << "BLOCK SIZE WAS NOT FOUND!" << endl;
        return -1; 
        // <-Thierry -> Jiadong: here you need to abort the processu, this is a non normal terminaison
    }
    BS = atoi(argv[2]);
#endif

    //for simple version, there are only square matrix
    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dimension;

    //Allocate 3 arrays on CPU pour 3 matrix
    double *h_A = (double *)malloc(nr_rows_A * nr_cols_A * sizeof(double));
    double *h_B = (double *)malloc(nr_rows_B * nr_cols_B * sizeof(double));
    double *h_C = (double *)malloc(nr_rows_C * nr_cols_C * sizeof(double));

    //Allocate 3 arrays on GPU
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(double));
    cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(double));
    cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(double));

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

    //Fill matrix A and B with large amount of number, then copy them on GPU
    cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,h_C,nr_rows_C * nr_cols_C * sizeof(double),cudaMemcpyHostToDevice);
    
    //Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    //csv file object
    ofstream captureTimeCsv;
    captureTimeCsv.open("captureTime.csv", ofstream::out | ofstream::app);
    cout << "prepare to write capture time into csv file" << "\n" << endl;
    captureTimeCsv << "Column 1 : experience index" << "," << "Column 2 : time used" << endl;
    double captureTime;

    ofstream instantiationTimeCsv;
    instantiationTimeCsv.open("instantiationTime.csv",ofstream::out | ofstream::app);
    cout << "prepare to write instantiate time into csv file" << "\n" << endl;
    instantiationTimeCsv << "Column 1 : experience index" << "," << "Column 2 : time used" << endl;
    double instantiationTime;

    ofstream launchingTimeCsv;
    launchingTimeCsv.open("launchingTime.csv",ofstream::out | ofstream::app);
    cout << "prepare to write launch time into csv file" << "\n" << endl;
    launchingTimeCsv << "Column 1 : experience index" << "," << "Column 2 : time used" << endl;
    double launchingTime;

    //Mesurement Time Cost loop
    for (int j = 0 ; j < 10; j ++)
    {   
        /*
        // creation of cuda stream through function cudaStreamCreate ( cudaStream_t* pStream )
        // allocate and initialize an array of stream handles
        cudaStream_t *stream = (cudaStream_t *)malloc(nstream * sizeof(cudaStream_t));
        checkCudaErrors(cudaStreamCreate(&(stream[i])));
        // Here we need only one stream to be bounded to the CUBLAS handle.
        */

        // creation of cuda stream, then bound the stream with the "handle" that has been created
        cudaStream_t stream;
        checkCudaErrors(cudaStreamCreate(&stream));
        cublasSetStream( handle, stream); 

        //Create time tracker
        timespec start_time, end_time;

        //Initialization of cuda graph
        bool graphCreated = false;
        clock_gettime(CLOCK_REALTIME, &start_time);
        cudaGraph_t graph;
        cudaGraphExec_t instance;
        //  Begin Caputure
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B, handle); //cette function est une stream cuda, not C++ stream!
        //  End Capture
        cudaStreamEndCapture(stream, &graph);
        clock_gettime(CLOCK_REALTIME, &end_time);
        captureTime = time_to_double(time_diff(start_time, end_time));
        printf("I show that capture time has been created\n");
        captureTimeCsv << j << "," << captureTime << endl;
        printf("Elapsed time for graph capture:%f (s)\n", time_to_double(time_diff(start_time, end_time)));

        //  Instantiate
        clock_gettime(CLOCK_REALTIME, &start_time);
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        clock_gettime(CLOCK_REALTIME, &end_time);
        instantiationTime = time_to_double(time_diff(start_time, end_time));
        instantiationTimeCsv << j << "," << instantiationTime << endl;
        //printf("Elapsed time for graph instantiation:%f (s)\n", time_to_double(time_diff(start_time, end_time)));

        // Graph Launch loop 
        for (int k=0; k < 20; k ++){
            printf("The index of graph launch is: %d\n", j * 20 + k);
            clock_gettime(CLOCK_REALTIME, &start_time);
            //lanch the cuda graph
            cudaGraphLaunch(instance, stream);
            cudaStreamSynchronize(stream);
            clock_gettime(CLOCK_REALTIME, &end_time);
            //high resolution timer
            launchingTime = time_to_double(time_diff(start_time, end_time));
            int index = j * 20 + k;
            launchingTimeCsv << index << "," << launchingTime << endl;
            //printf("Elapsed time for execution:%f (s)\n", time_to_double(time_diff(start_time, end_time)));
        }
    }
    launchingTimeCsv.close();
    launchingTimeCsv.clear(); 
    captureTimeCsv.close();
    captureTimeCsv.clear();
    instantiationTimeCsv.close();
    instantiationTimeCsv.clear();
    
    //Copy (and print) the result on host memory
    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(double),cudaMemcpyDeviceToHost);
    if (dimension <= 16){
        cout << "C =" << endl;
        print_matrix(h_C, nr_rows_C, nr_cols_C);
    }    
    
    //Distroy the handle
    cublasDestroy(handle);

    //Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

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
void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n, cublasHandle_t handle){
    int lda=m, ldb=k, ldc=m;
    const double alf = 1.5;
    const double bet = 0.5;
    const double *alpha = &alf;
    const double *beta = &bet;

#ifdef USE_MMUL_1_KERNEL
    //Do the actual multiplication
    //Cuda stream is branched with cuda handle, so gemm will also be done in(?) the stream created
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    //Use global variable BS for the block size.
    // if BS does not divide dimensions, then abort the processus
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
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, BS, BS, k, alpha, Ai0, lda, B0j, ldb, beta, Cij, ldc);
      }
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
    