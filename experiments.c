#include <stdio.h>
#include "omp.h"
#include <stdlib.h>
int f(int x){x++; return x;}
int g(int x){x = x*x*x; return x;}
int h(int x){x = x*x; return x;}

void main(){
	printf("Processor count: %d \n", omp_get_num_procs());
	int tsum = 0;
	printf("Thread count: %d \n", omp_get_num_threads());
	int arr[8] = {5, 5, 5, 5, 5, 5, 5, 5};
	#pragma omp parallel
	{
		arr[omp_get_thread_num()] = omp_get_thread_num();
	}

	for(int i = 0; i < 8; i++){
		tsum += arr[i];
		printf("%d\t", arr[i]);
	}

	printf("\nSum: %d\n", tsum);

	double result, fresult, gresult, hresult;
	int x = 2;
	#pragma omp parallel
	{
		int num = omp_get_thread_num();
		if(num == 0)	fresult = f(x);
		else if(num==1)	gresult = g(x);
		else if(num==2) hresult = h(x);
	}
	result = fresult + gresult + hresult;
	printf("f: %f\n", fresult);
	printf("g: %f\n", gresult);
	printf("h: %f\n", hresult);
	printf("%f\n", result);
}

