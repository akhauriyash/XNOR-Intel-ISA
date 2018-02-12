#include <stdio.h>
#include "omp.h"
#include <stdlib.h>

int f(int x){x++; return x;}
int g(int x){x = x*x*x; return x;}
int h(int x){x = x*x; return x;}

void proc_count(){
	printf("Processor count: %d \n", omp_get_num_procs());
	printf("Thread count: %d \n", omp_get_num_threads());
}

void thread_array_add(){
	int tsum = 0;

	int arr[8] = {5, 5, 5, 5, 5, 5, 5, 5};
	//	Index number of array = thread number
	#pragma omp parallel shared(tsum)
	{
		arr[omp_get_thread_num()] = omp_get_thread_num();
		#pragma omp critical	//	Might slow down parallelization due to synchronization
			tsum += arr[omp_get_thread_num()];
	}
	printf("\nSum with omp critical %d\n", tsum);
	tsum = 0;
	//	Serial summation of thread numbers
	for(int i = 0; i < 8; i++){
		tsum += arr[i];
		printf("%d\t", arr[i]);
	}
	printf("\nSum with for loop : %d\n", tsum);
}

void call_parallel(){
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

void main(){
	//	Gives processor and thread count
	proc_count();
	//	Adds thread numbers --> inspects thread parallelization
	//	omp critical
	thread_array_add();
	//	parallel function call
	call_parallel();

}

