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

#define FPUTYPE		float
#define BINTYPE		unsigned int

#define K_SIZE 		4
// #define MX_SIZE				16384
// #define MX_SIZE				8192
// #define MX_SIZE				4096
// #define MX_SIZE				2048
#define MX_SIZE				1024
// #define MX_SIZE				512
// #define MX_SIZE				256
#define NUM_OF_THREADS		64
#define TEST_LOOP			10


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



}
