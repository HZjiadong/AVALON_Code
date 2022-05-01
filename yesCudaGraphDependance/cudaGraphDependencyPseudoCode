//size_t == unsigned long long (64 bits)
BS_rows = BS;
BS_cols = BS;

//Allocation of arrays on CPU is still needed

//Allocation of block on CPU is needed?
size_t matrix_size_A_block = (size_t)(nr_rows_A / BS_rows) * (size_t)(nr_cols_A / BS_cols) * sizeof(double);
size_t matrix_size_B_block = (size_t)(nr_rows_B / BS_rows) * (size_t)(nr_cols_B / BS_cols) * sizeof(double);
size_t matrix_size_C_block = (size_t)(nr_rows_C / BS_rows) * (size_t)(nr_cols_C / BS_cols) * sizeof(double);

//Allocate 3 block on CPU 
double *h_A_block = (double *)malloc(matrix_size_A_block);
double *h_B_block = (double *)malloc(matrix_size_B_block);
double *h_C_block = (double *)malloc(matrix_size_C_block);

//Initialization of block of matrix ?? inside loop ??

cudaGraphCreate(&graph, 0);
//allocation has been done in Graph Creation?
//How to do allocation in Node Creation?

for (int i = 0; i < N; i++)
{
	for (int j = 0; j < N; j++)
	{
		construction of bloc of Mat A & B -> 2 nodes of (N/BS)*(N/BS) 【Host->Device】
        construction of bloc of Mat C (N/BS)*(N/BS) ?? no initialisation ??
        construction of 2 cudaNodes ?? same as construction of Mat ??

        //Creation of nodeParams Structure
        struct cudaKernelNodeParams
        {
            void* func;
            dim3 gridDim;
            dim3 blockDim;
            unsigned int sharedMemBytes;
            void **kernelParams;
            void **extra;
        }; 

        //Initialisation
        //Should there be Initialisation of N0? 【C[0][0] = alpha * A[0][0] * B[0][0]+ beta * C[0][0]】
        cudaGraphAddKernelNode(&Node[0], graph, NULL, 0, &nodeParams);

        k_upper_bound = N / BS -1;
        for (int k = 1; k < k_upper_bound; k++)
        {
        	Call of function DGemm;
        	C[i][j] = alpha * A[i][k] * B[k][j]+ beta * C[i][j];  【 ?? Is there C[i][j] or C[i][j-1] ?? 】
        	cudaGraphAddDependencies(graph, &Node[k-1], &Node[k], 1);     // N <k-1> -> N <k>
        }

        // Last dependency is Device -> Host ? 
        cudaGraphAddDependencies(graph, &Node[k_upper_bound], &Node[??Host??], 1);  
	}
}

//**Synchronization in which position? 
//**Close and Free in which position?