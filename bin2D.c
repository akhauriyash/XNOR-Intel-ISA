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
//	Check available nodes with pbsnodes, Note that skylake does not support AVXER/PF
//	icpc -xMIC-AVX512 -qopenmp -mkl -fp-model fast=2 -fma -unroll=4 binarize.c -o bins.out && echo ~/parallel/bins.out | qsub 

#define FPUTYPE				float
#define BINTYPE				unsigned int
#define MX_SIZE				8192
#define NUM_OF_THREADS		64
#define TEST_LOOP			64

int main( void )
{
	size_t m, n, p;
	size_t r, i, j, k;
	double dTimeS, dTimeE;
	m = p = n = MX_SIZE;
	putenv("KMP_AFFINITY=scatter");
	// putenv("KMP_AFFINITY=balanced, granularity=fine");
	// putenv("KMP_AFFINITY=compact");
	omp_set_num_threads(NUM_OF_THREADS);
	printf("Number of OpenMP threads: %3d\n", NUM_OF_THREADS);


////////////////////////  Allocate full precision matrices 	///////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

	__attribute__( ( aligned( 64 ) ) ) FPUTYPE **pA = NULL;		// Allocating memory 
	__attribute__( ( aligned( 64 ) ) ) FPUTYPE **pB = NULL;		// for matrices aligned
	__attribute__( ( aligned( 64 ) ) ) FPUTYPE **pC = NULL;		// on 64-byte boundary


	pA = ( FPUTYPE ** )_mm_malloc(m*sizeof(FPUTYPE *), 64);		// These loops can 
	for(int i = 0; i < m; i++){									// be collapsed
		pA[i] = ( FPUTYPE * )_mm_malloc(p*sizeof(FPUTYPE), 64);	// as m = n = p = MX_SIZE
	}
	pB = ( FPUTYPE ** )_mm_malloc(p*sizeof(FPUTYPE *), 64);
	for(int i = 0; i < p; i++){
		pB[i] = ( FPUTYPE * )_mm_malloc(n*sizeof(FPUTYPE), 64);
	}
	pC = ( FPUTYPE ** )_mm_malloc(m*sizeof(FPUTYPE *), 64);
	for(int i = 0; i < m; i++){
		pC[i] = ( FPUTYPE * )_mm_malloc(n*sizeof(FPUTYPE), 64);
	}
	if( pA == NULL || pB == NULL || pC == NULL )				// Error handling 
	{															// if any array is
		printf( "ERROR: Can't allocate memory for matrices\n" );// not allocated
		_mm_free( pA );
		_mm_free( pB );
		_mm_free( pC );
		return ( int )0;
	}
	for(int j = 0; j < m; j++){
		for( i = 0; i < p; i++)
		{
			FPUTYPE x = (FPUTYPE) rand()/RAND_MAX;				// Create random
			pA[j][i] = ( x < 0.5 ) ? -1 : 1;					// +1/-1 matrices
		}
	}
	for(int j = 0; j < p; j++){
		for( i = 0; i < n; i++)
		{
			FPUTYPE x = (FPUTYPE) rand()/RAND_MAX;				// Create random
			pB[j][i] = ( x > 0.5 ) ? -1 : 1;					// +1/-1 matrices
		}
	}
	for(int j = 0; j < m; j++){
		for( i = 0; i < n; i++)
		{
			pC[j][i] = 0;
		}
	}
////////////////////////	Allocate binary matrices 	///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

	__attribute__( ( aligned( 64 ) ) ) BINTYPE **bA = NULL;		// Allocated binary
	__attribute__( ( aligned( 64 ) ) ) BINTYPE **bB = NULL;		// matrices A and B

	bA = ( BINTYPE ** )_mm_malloc(m*sizeof(BINTYPE *), 64);
	bB = ( BINTYPE ** )_mm_malloc((p/32)*sizeof(BINTYPE *), 64);

	for(int i = 0; i<m; i++){
		bA[i] = (BINTYPE *)_mm_malloc((p/32)*sizeof(BINTYPE), 64);
	}
	for(int i = 0; i<(p/32); i++){
		bB[i] = (BINTYPE *)_mm_malloc(n*sizeof(BINTYPE), 64);
	}


////////////////////////	  Binarization of A&B   	///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
	int sign; BINTYPE tbA; BINTYPE tbB;

	dTimeS = dsecnd();
	for(int jj = 0; jj < TEST_LOOP; jj++){
		#pragma omp parallel for
		for (int i = 0; i < MX_SIZE; i++)
		{
			for(int seg = 0; seg < MX_SIZE/32; seg++)
			{
				tbA = 0;
				for(int j = 0; j < 32; j++)
				{//					[i*n + seg*32 + j]
					sign = (int) (pA[i][seg*32 + j] >= 0);
					tbA = tbA|(sign<<j);
				}
			bA[i][seg] = tbA;
			}
		}
	}
	dTimeE = dsecnd();
	printf( "\nBinarization A - Completed in: %.7f seconds\n", ( dTimeE - dTimeS ) / TEST_LOOP);
	

	dTimeS = dsecnd();
	for(int jj = 0; jj < TEST_LOOP; jj++){
		#pragma omp parallel for
		for (int i = 0; i < MX_SIZE; i++)
		{
			for(int seg = 0; seg < MX_SIZE/32; seg++)
			{
				tbB = 0;
				for(int j = 0; j < 32; j++)
				{//					[i+seg*32*n + j*n]
					sign = (int) (pB[seg*32 + j][i] >= 0);
					tbB = tbB|(sign<<j);
				}
			bB[seg][i] = tbB;
			}
		}
	}
	dTimeE = dsecnd();
	printf( "\nBinarization B - Completed in: %.7f seconds\n\n\n", ( dTimeE - dTimeS ) / TEST_LOOP );
	for(int i = 0; i<4; i++){
		for(j = 0; j<5; j++){
			printf("%u\t", bB[i][j]);
		}
	}

}
