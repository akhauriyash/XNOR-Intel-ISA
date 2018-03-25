#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include <math.h>
#include <mkl.h>
#include <iostream>

//	To compile:
//	icpc -xMIC-AVX512 -qopenmp -mkl -fp-model fast=2 -fma -unroll=4 xconv.c -o xconv.out && echo ~/parallel/xconv.out | qsub 


#define K_SIZE 		4

#define FPUTYPE		float
#define BINTYPE		unsigned short

// #define MX_SIZE				16384
// #define MX_SIZE				8192
// #define MX_SIZE				4096
// #define MX_SIZE				2048
#define MX_SIZE				1024
// #define MX_SIZE				512
// #define MX_SIZE				256
#define NUM_OF_THREADS		64
#define TEST_LOOP			10

// printBits prints the binary format of the unsigned int passed to it.
void printBits(size_t const size, void const * const ptr){
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;
    printf("\n");
    for (i=size-1;i>=0;i--)
        for (j=7;j>=0;j--)
        {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
    puts("");    printf("\n");             
    }



int main( void )
{
	size_t r, c, rk, ck;
	size_t i, j, k, ii, jj, kk, sm;
	double dTimeS, dTimeE;
	r = c = MX_SIZE;
	rk = ck = K_SIZE;
	printf("Matrix size: %d x %d \n Kernel Size %d x %d \n", r, c, rk, ck);
	putenv("KMP_AFFINITY=scatter");	
	// putenv("KMP_AFFINITY=balanced, granularity=fine");
	// putenv("KMP_AFFINITY=compact");
	omp_set_num_threads(NUM_OF_THREADS);
	printf("Number of OpenMP threads: %3d\n", NUM_OF_THREADS);


////////////////////////  Allocate full precision matrix 	///////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

	__attribute__( ( aligned( 64 ) ) ) FPUTYPE **pA = NULL;
	pA = ( FPUTYPE ** )_mm_malloc(r*sizeof(FPUTYPE *), 64);
	for(int i = 0; i < r; i++){
		pA[i] = ( FPUTYPE * )_mm_malloc(c*sizeof(FPUTYPE), 64);
	}
	__attribute__( ( aligned( 64 ) ) ) FPUTYPE **kerA = NULL;
	kerA = ( FPUTYPE ** )_mm_malloc(rk*sizeof(FPUTYPE *), 64);
	for(int i = 0; i < rk; i++){
		kerA[i] = ( FPUTYPE *)_mm_malloc(ck*sizeof(FPUTYPE), 64);
	}
	if(pA == NULL || kerA == NULL){
		printf( "\nERROR: Can't allocate memory for matrices\n" );
		_mm_free( pA );
		_mm_free( kerA );
		return ( int )0;
	}
	for(int i = 0; i < r; i++){
		for( j = 0; j < c; j++)
		{
			FPUTYPE x = (FPUTYPE) rand()/RAND_MAX;				// Create random
			pA[i][j] = ( x < 0.5 ) ? -1 : 1;					// +1/-1 matrices
		}
	}
	for(int i = 0; i < rk; i++){
		for( j = 0; j < ck; j++)
		{
			FPUTYPE x = (FPUTYPE) rand()/RAND_MAX;				// Create random
			kerA[i][j] = ( x < 0.5 ) ? -1 : 1;					// +1/-1 matrices
		}
	}

	printf("\n\n\n **** STARTING Full precision FUNCTIONS **** \n\n\n");

////////////////////////	FP Matrix Convolution   	///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

////////////////////////	Allocate binary matrices   	///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

// Post binarization: kerA: bkerA ==> 1 unsigned int 32 bit (high lane: 1s) [mxm] (assume m*m = 16)
//					  pA:   bA    ==> Matrix: shape (n-m+1 x n-m+1) (Every 16 bit high lane 0s)

	__attribute__( ( aligned( 64 ) ) ) BINTYPE **bA = NULL;

	bA = ( BINTYPE ** )_mm_malloc((r-rk+1)*sizeof(BINTYPE *), 64);
	for(int i = 0; i < (r-rk+1); i++){
		bA[i] = ( BINTYPE * )_mm_malloc((c-ck+1)*sizeof(BINTYPE), 64);
	}
//	*** NOTE THAT THE BINARIZATION PROCEDURE REVERSES MATRIX ORDER ***
	BINTYPE bkerA = 0;	int sign;	BINTYPE tbA = 0;
	for(int i = 0; i < rk; i++){
		for(int j = 0; j < ck; j++){
			printf("%.1f\t", kerA[i][j]);
			sign = (int) (kerA[i][j] >= 0);
			bkerA = bkerA|(sign<<(i*ck + j));
		}
	}
	printf("\n");
	printBits(sizeof(bkerA), &bkerA);
	printf("\n");printf("\n");

	dTimeS = dsecnd();
	for(int i = 0; i < r-rk+1; i++){
		for(int j = 0; j < c-ck+1; j++){
			tbA = 0;
			for(int ii = 0; ii < rk; ii++){
				for(int jj = 0; jj < ck; jj++){
					sign = (int) (pA[i+ii][j+jj] >= 0);
					tbA = tbA|(sign<<(ii*ck + jj));
				}
			}
			bA[i][j] = tbA;
		}
	}
	dTimeE = dsecnd();
	printf( "\nBinarization A - Completed in: %.7f seconds\n", ( dTimeE - dTimeS ));
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 4; j++){
			tbA = 0;
			for(int ii = 0; ii < rk; ii++){
				for(int jj = 0; jj < ck; jj++){
					printf("%0.1f\t", pA[i+ii][j+jj]);
				}
			}	
			printBits(sizeof(bA[i][j]), &bA[i][j]);
		}
	}


////////////////////////	kerA pA binarization    	///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

////////////////////////	Binarized convolution   	///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


}



