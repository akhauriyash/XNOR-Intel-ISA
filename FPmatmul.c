#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include <math.h>
#define MAX_VAL 1000
#define MIN_VAL 1
#define MAX_DIM 2000*2000
//	gcc matmul.c -fopenmp -lm
//	echo ~/parallel/a.out | qsub
// 1D matrix on stack
double flatA[MAX_DIM];
double flatB[MAX_DIM];

double ParallelMultiply(double** matrixA, double** matrixB, double** matrixC,
	int dimension);
double** randomSquareMatrix(int dimension);
double sequentialMultiply(double** matrixA, double** matrixB, double** matrixC, int dimension);
void convert(double** matrixA, double** matrixB, int dimension);

void main(){
	double** matrixA = randomSquareMatrix(2000*2000);
	double** matrixB = randomSquareMatrix(2000*2000);
	double** matrixC = randomSquareMatrix(2000*2000);
	double timez;
	timez = ParallelMultiply(matrixA, matrixB, matrixC, 2000*2000);
	printf("%lf", timez);
	printf("\nWait what is happening did anything even print? Ok awesome.\n");
}

double** randomSquareMatrix(int dimension){
	/*
		Generate 2 dimensional random double matrix
	*/
	double** matrix = malloc(dimension*sizeof(double*));
	for(int i = 0; i<dimension; i++){
		matrix[i] = malloc(dimension*sizeof(double));
	}
	//	Random seed
	srandom(time(0) + clock() + random());
	#pragma omp parallel for
	for (int i = 0; i < dimension; i++)
	{
		for (int j = 0; j < dimension; j++)
		{
			matrix[i][j] = rand()%MAX_VAL + MIN_VAL;			
		}
	}
	return matrix;
}


double ParallelMultiply(double** matrixA, double** matrixB, double** matrixC,
	int dimension){
	int i, j, k, iOff, jOff;
	double tot;
	struct timeval t0, t1;
	gettimeofday(&t0, 0);
	convert(matrixA, matrixB, dimension);
	#pragma omp parallel shared(matrixC) private(i, j, k, iOff, jOff, tot)
	{
		#pragma omp for schedule(static)
		for(i=0; i<dimension; i++){
			iOff = i*dimension;
			for(j=0; j<dimension; j++){
				jOff = j*dimension;
				tot=0;
				for (int k = 0; k<dimension; k++)
				{
					tot+=flatA[iOff + k]*flatB[jOff+k];
				}
				matrixC[i][j] = tot;
			}
		}
	}
	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;
	return elapsed;
  }
  
void convert(double** matrixA, double** matrixB, int dimension){
	#pragma omp parallel for
	for (int i = 0; i < dimension; i++)
	{
		for (int j = 0; j < dimension; j++)
		{
			flatA[i*dimension + j] = matrixA[i][j];
			flatB[j*dimension + i] = matrixB[i][j];
		}
	}
}
