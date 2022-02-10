#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <curand.h>
#include <cublas_v2.h>

int main() {
    //dimensions of 3 matrix
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
    int dimension;

    //for simple version, there are only square matrix
    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dimension;

    //Allocate 3 arrays on CPU pour 3 matrix
    double *h_A = (double *)malloc(nr_rows_A * nr_cols_A * sizeof(double));
    double *h_A = (double *)malloc(nr_rows_A * nr_cols_A * sizeof(double));
    double *h_A = (double *)malloc(nr_rows_A * nr_cols_A * sizeof(double));

    //Allocate 3 arrays on GPU
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(double));
    cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(double));
    cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(double));

    //Fill matrix A and B with large amount of number, then copy them on GPU
    cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(double),cudaMemcpyHostToDevice);
    /*
    std::cout << "A =" << std::endl;
    print_matrix(h_A, nr_rows_A, nr_cols_A);
    std::cout << "B =" << std::endl;
    print_matrix(h_B, nr_rows_B, nr_cols_B);
    */

   /*
   Create directly two random matrix on GPU
   */

    //Now matrix A and B are initialized and transfered on GPU
    //main algo

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
void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n){
    int lda=m, ldb=k, ldc=m;
    const double alf = 1;
    const double bet = 0;
    const double *alpha = &alf;
    const double *beta = &bet;

    //Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    //Do the actual multiplication
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpa, A, lda, B, ldb, beta, C, ldc);

    //Distroy the handle
    cublasDestroy(handle);

}

// This code create two random matrix directly on GPU, maybe will be useful later.
// But here we will not use this directly on GPU, but build matrix on CPU and copy them into GPU.
// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU

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
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/*
//If more than one matrix multiplication in the code is needed, 
//it is better to move the create/destory handle code from original gpu_blas_mmul function into the main function,
//then use the same handle for all multiplications.
//In this case, the gpu_blas_mmul function becomes:

void gpu_blas_mmul(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    
    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

*/


