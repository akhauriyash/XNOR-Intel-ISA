/*
	Example below has a load imbalance over loop iterations. 
*/
#pragma omp for
for (i = 0; i < 13; i++)
	{…}

/*
	Optimization for thread parallelized workloads
	Some loop threading will lead to idle thread and load imbalance
	We parallelize inner loop as the outer loop will give load imbalance for
	when threads != 5
*/
void processQuadArray (int imx, int jmx, int kmx,
 double**** w, double**** ws)
{
 #pragma omp parallel shared(w, ws)
 {
 int nv, k, j, i;
 for (nv = 0; nv < 5; nv++)
	for (k = 0; k < kmx; k++) 							// kmx is usually small
		#pragma omp for shared(nv, k) nowait			// Independent computation --> disable barrier (nowait)
			for (j = 0; j < jmx; j++)
				for (i = 0; i < imx; i++)
					ws[nv][k][j][i] = Process(w[nv][k][j][i]);
 }
}
/*
	Dealing with loop carried dependence:		Loop fission
*/

float *a, *b;
int i;
for (i = 1; i < N; i++) {
	if (b[i] > 0.0)
		a[i] = 2.0 * b[i];
	else
		a[i] = 2.0 * fabs(b[i]);
	b[i] = a[i-1];
}
/* 
	Parallelization scope: Intel Threading Building Blocks (Intel TBB):
	parallel_for algorithm
*/
float *a, *b;
parallel_for (1, N, 1,				//	Define loop
	[&](int i){						//	Build function
		if(b[i]>0.0)
			a[i] = 2.0*b[i];
		else
			a[i] = 2.0*fabs(b[i]);
	});
parallel_for(1, N, 1, 				//	Make updates to b
	[&](int i){						//	These will happen after a has 
		b[i] = a[i-1];				//	been assigned necessary values
	});

/*
	OpenMP if clause to choose serial/parallel execution based on runtime information
*/
#pragma omp parallel for if(N >= threshold)
	for (i = 0; i < N; i++) { … }

//	Granularity: Too fine --> communication overhead, Too coarse --> load imbalance.
//	Too fine --> Too much parallelization overhead in form of communication, synchronization etc.
