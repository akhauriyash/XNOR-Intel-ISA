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
#define K_LOOP		K_SIZE*K_SIZE
#define FPUTYPE		float
#define BINTYPE		unsigned short

#define MX_SIZE				32768
// #define MX_SIZE				16384
// #define MX_SIZE				8192
// #define MX_SIZE				4096
// #define MX_SIZE				2048
// #define MX_SIZE				1024
// #define MX_SIZE				512
// #define MX_SIZE				256
// #define MX_SIZE				128
// #define MX_SIZE				64
#define NUM_OF_THREADS		256
#define TEST_LOOP			6

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
	putenv("KMP_AFFINITY=scatter");								// OK
	// putenv("KMP_AFFINITY=balanced, granularity=fine");		// OK
	// putenv("KMP_AFFINITY=compact");							// BAD
	omp_set_num_threads(NUM_OF_THREADS);
	printf("Number of OpenMP threads: %3d\n", NUM_OF_THREADS);


////////////////////////  Allocate full precision matrix 	///////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

	__attribute__( ( aligned( 32 ) ) ) FPUTYPE **pA = NULL;
	pA = ( FPUTYPE ** )_mm_malloc(r*sizeof(FPUTYPE *), 32);
	for(int i = 0; i < r; i++){
		pA[i] = ( FPUTYPE * )_mm_malloc(c*sizeof(FPUTYPE), 32);
	}
	__attribute__( ( aligned( 32 ) ) ) FPUTYPE **kerA = NULL;
	kerA = ( FPUTYPE ** )_mm_malloc(rk*sizeof(FPUTYPE *), 32);
	for(int i = 0; i < rk; i++){
		kerA[i] = ( FPUTYPE *)_mm_malloc(ck*sizeof(FPUTYPE), 32);
	}
	__attribute__( ( aligned( 32 ) ) ) FPUTYPE **pC = NULL;
	pC = ( FPUTYPE ** )_mm_malloc((r-rk+1)*sizeof(FPUTYPE *), 32);
	for(int i = 0; i < (r-rk+1); i++){
		pC[i] = ( FPUTYPE * )_mm_malloc((c-ck+1)*sizeof(FPUTYPE), 32);
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

	int accumulator;
	dTimeS = dsecnd();
	for(int tloop = 0; tloop<TEST_LOOP; tloop++){
		#pragma omp parallel for private(i, j, ii, jj, accumulator) num_threads(NUM_OF_THREADS)
		for(int i = 0; i < r-rk+1; i++){
			for(int j = 0; j < c-ck+1; j++){
				accumulator = 0;
				for(int ii = 0; ii < rk; ii++){
					for(int jj = 0; jj < ck; jj++){
						accumulator += pA[i+ii][j+jj]*kerA[ii][jj];
					}
				}
				pC[i][j] = accumulator;
			}
		}
	}
	dTimeE = dsecnd();
	printf( "\n FP CONV - Completed in: %.7f seconds\n", ( dTimeE - dTimeS ) / TEST_LOOP);
	for(int i = 0; i < 5; i++){
		for(int j = 0; j < 5; j++){
			printf("%.1f\t", pC[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");

////////////////////////	Allocate binary matrices   	///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// If data type is unsigned int, then the commend below applies. If unsigned short, then no problem.
// Post binarization: kerA: bkerA ==> 1 unsigned int 32 bit (high lane: 1s) [mxm] (assume m*m = 16)
//					  pA:   bA    ==> Matrix: shape (n-m+1 x n-m+1) (Every 16 bit high lane 0s)

//	*** NOTE THAT THE BINARIZATION PROCEDURE REVERSES MATRIX ORDER ***

	__attribute__( ( aligned( 16 ) ) ) BINTYPE **bA = NULL;

	bA = ( BINTYPE ** )_mm_malloc((r-rk+1)*sizeof(BINTYPE *), 16);
	for(int i = 0; i < (r-rk+1); i++){
		bA[i] = ( BINTYPE * )_mm_malloc((c-ck+1)*sizeof(BINTYPE), 16);
	}
	BINTYPE bkerA = 0;	int sign;	BINTYPE tbA = 0;
	for(int i = 0; i < rk; i++){
		for(int j = 0; j < ck; j++){
			// printf("%.1f\t", kerA[i][j]);
			sign = (int) (kerA[i][j] >= 0);
			bkerA = bkerA|(sign<<(i*ck + j));
		}
	}
	// printf("\n");
	// printBits(sizeof(bkerA), &bkerA);
	// printf("\n");printf("\n");


////////////////////////	kerA pA binarization    	///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

	dTimeS = dsecnd();
	// Time taken to binarize 16384 matrix: 1.089s (with pragma), 8.772s (without pragma)

	for(int tloop = 0; tloop<TEST_LOOP; tloop++){
		#pragma omp parallel for private(i, j, ii, jj, tbA, sign) num_threads(NUM_OF_THREADS)
		for(int i = 0; i < r-rk+1; i++){
			for(int j = 0; j < c-ck+1; j++){
				tbA = 0;
				for(int ii = 0; ii < rk; ii++){
					for(int jj = 0; jj < ck; jj++){
						sign = (int) (pA[i+ii][j+jj] >= 0);
						tbA = tbA|(sign<<(ii*ck + jj));
					}
				}
				pC[i][j] = 2*(__builtin_popcount(~(tbA^bkerA))) - 48;
			}
		}
	}	
	dTimeE = dsecnd();
	printf( "\n BIN+xCONV A - Completed in: %.7f seconds\n", ( dTimeE - dTimeS ) / TEST_LOOP);

	// printf( "\nBinarization A - Completed in: %.7f seconds\n", ( dTimeE - dTimeS ) / TEST_LOOP);
	// for(int i = 0; i < 3; i++){
	// 	for(int j = 0; j < 4; j++){
	// 		tbA = 0;
	// 		for(int ii = 0; ii < rk; ii++){
	// 			for(int jj = 0; jj < ck; jj++){
	// 				printf("%0.1f\t", pA[i+ii][j+jj]);
	// 			}
	// 		}	
	// 		printBits(sizeof(bA[i][j]), &bA[i][j]);
	// 	}
	// }


////////////////////////	Binarized convolution   	///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
	// dTimeS = dsecnd();

	// for(int tloop = 0; tloop<TEST_LOOP; tloop++){
	// 	#pragma omp parallel for private(i, j) num_threads(NUM_OF_THREADS)
	// 	for(int i = 0; i < (r-rk+1); i++){
	// 		for(int j = 0; j < (c-ck+1); j++){
	// 			pC[i][j] = 2*(__builtin_popcount(~(bA[i][j]^bkerA))) - 48;
	// 			}
	// 		}
	// 	}
	// dTimeE = dsecnd();
	// printf( "\nxCONV - Completed in: %.7f seconds\n", ( dTimeE - dTimeS ) / TEST_LOOP);
	for(int i = 0; i < 5; i++){
		for(int j = 0; j < 5; j++){
			printf("%.1f\t", pC[i][j]);
		}
		printf("\n");
	}
}



