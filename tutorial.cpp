#	Memory allocation schemes

'''			DR4 RAM ALLOCATION SCHEME			'''

''' 			1-D allocation 					'''
__attribute__((aligned(64)) float *fA;			/	Declares 1-D structure

fA = (float *)_mm_malloc(N*sizeof(float), 64);	/	Allocate --> 64 byte aligned contiguous unfragment (performance)

'''				 2-D allocation 				'''
__attribute__((aligned(64)) float **fA;			/	Declares 1-D structure
fA = (float **)calloc(N, sizeof(float *));
for(i = 0; i<N; i += 1){
	fA[i] = (float *)calloc(N, sizeof(float));	/	Allocate --> Not contiguous, fragmented by memory manager (lesser performance)
}

'''			MCDRAM memory allocation		   	'''
'''	memkind library: configured to flat/hybrid	'''
'''			MCDRAM bandwidth: ~ 400 GB/s		'''
'''			DDR4 bandwidth:	  ~ 80	GB/s		'''
__attribute__((aligned(64)) float *fA;			/	Declares 1-D structure
fA = (float *)hbw_malloc(N*sizeof(float));		/	MCDRAM allocation

'''		Loop processing scheme --> Vanilla		'''
for( i = 0; i < N; i += 1 )                      // loop 1
    for( j = 0; j < N; j += 1 )                  // loop 2
        for( k = 0; k < N; k += 1 )              // loop 3
            C[i][j] += A[i][k] * B[k][j];

'''			CPU cache optimizations				'''
/*	C[i][j] += A[i][k]  * B[k][j] ==> Inefficient access 
		Every subsequent element of B is located at a distance of (N*sizeof(dtype)) bytes.
		This gives cache misses.
	Transpose B ==> Next element distance at sizeof(dtype) bytes. 
	C[i][j] += A[i][k] * B[j][k]
		Transpose time needs to be taken into account.
		*/

'''	LBOT: loop blocking optimization technique	'''
'''				Try transposed LBOT				'''
for( i = 0; i < N; i += BlockSize )
{
    for( j = 0; j < N; j += BlockSize )
    {
        for( k = 0; k < N; k += BlockSize )
        {
            for( ii = i; ii < ( i+BlockSize ); ii += 1 )
                for( jj = j; jj < ( j+BlockSize ); jj += 1 )
                    for( kk = k; kk < ( k+BlockSize ); kk += 1 )
                        C[ii][jj] += A[ii][kk] * B[kk][jj];
        }
    }
}


'''	 OpenMP product thread affinity control		'''
/*	OpenMP directives --> execute OpenMP threads on different logical CPUs
	of modern multi core processors.
	KMP_AFFINITY =		scatter			
						balanced
						compact					/ Slower than scatter/balanced
						
	OpenMP:
	nowait ==> #pragma omp parallel shared(nv, k) nowait
				nowait clause allows threads to proceed instead of sitting idle at an implicit
				barrier when the execution is independent.
				*/
				
void processQuadArray (int imx, int jmx, int kmx,
 double**** w, double**** ws)
{
 #pragma omp parallel shared(w, ws)
 {
 int nv, k, j, i;
 for (nv = 0; nv < 5; nv++)
	for (k = 0; k < kmx; k++) // kmx is usually small
		#pragma omp for shared(nv, k) nowait
		for (j = 0; j < jmx; j++)
			for (i = 0; i < imx; i++)
				ws[nv][k][j][i] = Process(w[nv][k][j][i]);
 }
}
