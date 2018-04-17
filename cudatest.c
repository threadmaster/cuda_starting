/*****************************************************
*
* CUDATEST 
*
* Andrew J. Pounds, Ph.D.
* Department of Computer Science
* Mercer University
* Spring 2018
*
* A program to demonstrate how to query and allocate
* memory on the CUDA card and then use the CUDA BLAS
* library to perform matrix multiplication.  Some code
* was adapted from the NVIDIA cublas users manual.
*
*/

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stddef.h> /* defines NULL */
#include <sys/time.h>

double walltime() 
{
	struct timeval tp;
	int rtn;
	double seconds;
	double factor;

	factor = 1.0e-6;
	rtn=gettimeofday(&tp, NULL);
	seconds = tp.tv_sec + factor * tp.tv_usec;
	return(seconds) ; 
}

void cudablas_mmm( cublasHandle_t handle, double *A, double *B, double *C, 
		int DIM, double alpha, double beta){

	const double *d_alpha = &alpha;
	const double *d_beta = &beta;

	// Call the actual double precision matrix multiplication library function
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, DIM, DIM, DIM, d_alpha, A, DIM, B, DIM, d_beta, C, DIM);

}

int main(){

	int DIM;
	int i, j, k;
	double *veca, *vecb;  
	double *A, *B, *C;
	double alpha = 1.0;
	double beta = 0.0;
	double *d_A, *d_B, *d_C;
	double wall, trace, mflops; 

	cudaError_t cudaStat;
	cudaError_t err;
	cublasStatus_t stat;
	char *errorstring;

	DIM = 1000;

	// Allocate space for vectors and matrices on host

	veca = malloc( DIM * sizeof(double) ); 
	vecb = malloc( DIM * sizeof(double) ); 
	A = malloc( DIM * DIM * sizeof(double) );
	B = malloc( DIM * DIM * sizeof(double) );
	C = malloc( DIM * DIM * sizeof(double) );

	// Build vectors on host
	for(i=0; i<DIM; i++){
		*(veca+i) = 1.0; 
		*(vecb+i) = 1.0 / sqrt( (double) DIM ); 
	}

	// Build matrices A and B from tensor products and initialize C to zero 

	for (i = 0; i<DIM; i++) {
		for (j=0; j<DIM; j++) {
			*(A+i*DIM+j) = *(veca+i) * *(vecb+j);
			*(B+i*DIM+j) = *(veca+i) * *(vecb+j);
			*(C+i*DIM+j) = 0.0;
		}
	}

	// To properly time cuda we have to also include device mallocs and copies

	wall = walltime(); /* start timer */

	// Create CUDA card device handles
	cublasHandle_t handle;

	size_t available, total;
	cudaMemGetInfo(&available, &total);
	printf("Total GPU mem: %d, Available GPU mem: %d\n", total, available);

	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("CUBLAS initialization failed\n");
		return EXIT_FAILURE;
	}

	// Allocate memory on the card to store the matrices
	printf("Allocating d_A\n");
	cudaStat = cudaMalloc( (void**) &d_A, DIM*DIM*sizeof(*A));
	cudaMemGetInfo(&available, &total);
	printf("Total GPU mem: %d, Available GPU mem: %d\n", total, available);
	if (cudaStat != cudaSuccess ){
		printf("Device memory allocation failed with err: %s.\n", cudaGetErrorString(cudaStat));
		cudaFree(d_A);
		cudaDeviceReset();
		exit(1);
	}

	printf("Allocating d_B\n");
	cudaStat = cudaMalloc( (void**) &d_B, DIM*DIM * sizeof(*B));
	cudaMemGetInfo(&available, &total);
	printf("Total GPU mem: %d, Available GPU mem: %d\n", total, available);
	if (cudaStat != cudaSuccess ){
		printf("Device memory allocation failed with err: %s.\n", cudaGetErrorString(cudaStat));
		cudaFree(d_B);
		cudaFree(d_A);
		cudaDeviceReset();
		exit(1);
	}

	printf("Allocating d_C\n");
	cudaStat = cudaMalloc( (void**) &d_C, DIM*DIM * sizeof(*C));
	cudaMemGetInfo(&available, &total);
	printf("Total GPU mem: %d, Available GPU mem: %d\n", total, available);
	if (cudaStat != cudaSuccess ){
		printf("Device memory allocation failed with err: %s.\n", cudaGetErrorString(cudaStat));
		cudaFree(d_C);
		cudaFree(d_B);
		cudaFree(d_A);
		cudaDeviceReset();
		exit(1);
	}

	// Copy the matrices to the card remember first to arguments are always destination and then source.
	// The last argument determines the type of transfer - not the direction of transfer.

	printf("Copying Matrix A from host to cuda device.\n");
	cudaMemcpy(d_A, A, DIM*DIM* sizeof(double), cudaMemcpyHostToDevice);
	printf("Copying Matrix B from host to cuda device.\n");
	cudaMemcpy(d_B, B, DIM*DIM* sizeof(double), cudaMemcpyHostToDevice);


	// Call the matrix multiplication function that is GPU based
	printf("Doing Multiply on cuda device.\n");
	cudablas_mmm(handle, d_A, d_B, d_C, DIM, alpha, beta);

	// Copy the matrix C back from the card to the host computer
	printf("Copying Matrix C from cuda device to host.\n");
	cudaMemcpy(C,d_C,DIM*DIM*sizeof(double),cudaMemcpyDeviceToHost);

	// Free the memory on the CUDA card
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Free the memory used for the device handle
	cublasDestroy(handle);

	wall = walltime() - wall;

	// Print the sum of the diagonal

	trace = 0.0; 
	for (i=0;i<DIM;i++) trace += *(C+i*DIM+i);

    // Compute megaflops and print results
    mflops = 2.0*DIM*DIM*DIM/wall/1.0E6;

	printf( "%d %f %f %f \n", DIM, trace, wall, mflops); 

}


