#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include <math.h>
#include <mkl.h>

//	To compile:
//		Check available nodes with pbsnodes, Note that skylake does not support AVXER/PF
//		icpc -03 -xMIC-AVX512 -qopenmp -mkl -fp-model fast=2 -fma -unroll=4 binarize.c -o bins.out
//		echo ~/parallel/bins.out | qsub 

#define FPUTYPE				float
#define BINTYPE				unsigned int
#define MX_SIZE				256
#define NUM_OF_THREADS		64


int main( void )
{
	__attribute__( ( aligned( 64 ) ) ) FPUTYPE *pA = NULL;
	__attribute__( ( aligned( 64 ) ) ) FPUTYPE *pB = NULL;
	__attribute__( ( aligned( 64 ) ) ) FPUTYPE *pC = NULL;
	size_t m, n, p;
	size_t r, i, j, k;
	FPUTYPE alpha;
	FPUTYPE beta;
	FPUTYPE sum;
	double dTimeS;
	double dTimeE;
	m = p = n = MX_SIZE;
	//	Allocating memory for matrices aligned on 64-byte boundary
	pA = ( FPUTYPE * )_mm_malloc((m*p)*sizeof(FPUTYPE), 64);
	pB = ( FPUTYPE * )_mm_malloc((p*n)*sizeof(FPUTYPE), 64);
	pC = ( FPUTYPE * )_mm_malloc((m*n)*sizeof(FPUTYPE), 64);
	if( pA == NULL || pB == NULL || pC == NULL )
	{
		printf( "ERROR: Can't allocate memory for matrices\n" );
		_mm_free( pA );
		_mm_free( pB );
		_mm_free( pC );
		return ( int )0;
	}
	putenv("KMP_AFFINITY=scatter");
	// putenv("KMP_AFFINITY=balanced, granularity=fine");
	// putenv("KMP_AFFINITY=compact");
	omp_set_num_threads(NUM_OF_THREADS);
	printf("Number of OpenMP threads: %3d\n", NUM_OF_THREADS);
	for( i = 0; i < (m*p); i += 1 )
		pA[i] = ( FPUTYPE ) 1.0;
	for( i = 0; i < (p*n); i += 1 )
		pB[i] = ( FPUTYPE ) -1.0;
	for( i = 0; i < (m*n); i += 1 )
		pC[i] = 0.0;

	int sign; BINTYPE tbA;
	__attribute__( ( aligned( 64 ) ) ) BINTYPE *bA = NULL;
	bA = ( BINTYPE * )_mm_malloc((m*p)*sizeof(BINTYPE), 64);
	#pragma omp parallel for
	for (int i = 0; i < MX_SIZE; i++)
	{
		for(int seg = 0; seg < MX_SIZE/32; seg++)
		{
			tbA = 0;
			for(int j = 0; j < 32; j++)
			{
				sign = (int) (pA[i*n+seg*MX_SIZE/32 + j]);
				tbA = tbA|(sign<<j);
			}
		bA[seg, i] = tbA;
		}
	}
	printf("\nbA has been binarized successfully, give treat!\n");
}
