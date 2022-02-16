#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <time.h>
#include <curand.h>
#include <cublas.h>
#include <cublas_v2.h>
using namespace std;

//Declaration des functions utiliser
void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n, cublasHandle_t handle);
void print_matrix(const double *A, int nr_rows_A, int nr_cols_A);
timespec time_diff(timespec start, timespec end);

//Main function
int main(int argc, char *argv[]) {
    //dimensions of 3 matrix
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
    int dimension;

    //get information from standard input
    if (argc <= 1) {
        cout << "NUMBER OF DIMENSION WAS NOT FOUND!" << endl;
    }
    dimension = atoi(argv[1]);

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
    
    /*
    Create directly two random matrix on GPU
    GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
    GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
    */
   
    //Transfert A,B,C -> device
    int lda = dimension;
    int ldb = dimension;
    cublasStatus stat;
    stat = cublasSetMatrix(nr_rows_A, nr_cols_A, sizeof(double), *h_A, lda, *d_A, ldb);

/*
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
*/


    //Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    //Create time tracker
    timespec start_time, end_time;

    //Multiply A and B on GPU for several times(200 times here)
    for (int k=0; k < 200; k ++){
        clock_gettime(CLOCK_REALTIME, &start_time);
        gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B, handle);
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_REALTIME, &end_time);
        //high resolution timer
        printf("Elapsed time for execution:%f (s)\n", time_to_double(time_diff(start_time, end_time)));
    }
    
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
    free(d_A);
    free(d_B);
    free(d_C);

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

    //Do the actual multiplication
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

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