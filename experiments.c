#include "include.h"
// 
// 	Execute this as
// 	gcc experiments.c -fopenmp -lm
// 	in the terminal.
// 
int f(int x){x++; return x;}
int g(int x){x = x*x*x; return x;}
int h(int x){x = x*x; return x;}

void main(){
	//	Gives processor and thread count
	proc_count();
	//	Adds thread numbers --> inspects thread parallelization
	//	Omp critical
	thread_array_add();
	//	Parallel function call
	call_parallel();
	//	Parallel thread bounds --> Make some 'x' thread numbers do something
	//	parallel do and parallel for do not create a team of threads.
	//	they take active team of threads and divide loop iterations over them
	//	thus omp parallel for/do needs to be inside a parallel region.
	thread_limiter();
	//	Calculates pi
	pi();
	//	Loop schedules: Static and Dynamic
	//	Static: Purely based on number of iterations and number of threads
	//	Dynamic: load balancing --> Iteration assigned to unoccupied threads
	schedules();
}

void proc_count(){
	printf("\n\n***************proc_count()**************\n");
	printf("Processor count: %d \n", omp_get_num_procs());
	printf("Thread count: %d \n", omp_get_num_threads());
}

void thread_array_add(){
	printf("\n\n***************thread_array_add()**************\n");
	int tsum = 0;
	int arr[8] = {5, 5, 5, 5, 5, 5, 5, 5};
	//	Index number of array = thread number
	#pragma omp parallel shared(tsum)
	{
		arr[omp_get_thread_num()] = omp_get_thread_num();
		#pragma omp critical	//	Might slow down parallelization due to synchronization
			tsum += arr[omp_get_thread_num()];
	}
	printf("\nSum with omp critical: %d\n", tsum);
	tsum = 0;
	//	Serial summation of thread numbers
	for(int i = 0; i < 8; i++){
		tsum += arr[i];
		printf("%d\t", arr[i]);
	}
	printf("\nSum with for loop : %d\n", tsum);
}

void print_parallel(){
	printf("\t\tThread num: %d\n", omp_get_thread_num());
}
void call_parallel(){
	printf("\n\n***************call_parallel()**************\n");
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


void thread_limiter(){
	printf("\n\n***************thread_limiter()**************\n");
	//	Using pragma omp parallel
	printf("With parallel region around the loop and adjusting loop bounds:\n");
	int N = 8;
	#pragma omp parallel
	{
		int threadnum = omp_get_thread_num();
		int numthreads = omp_get_num_threads();
		int low = N*threadnum/numthreads;
		int high = N*(threadnum+1)/numthreads;
		for(int i = low; i < high; i++){
			printf("\tThread %d doing the loop\n", omp_get_thread_num());
		}
	}
	//	Using pragma omp parallel with omp for
	printf("With parallel for pragma:\n");
	#pragma omp parallel
	{
		printf("\tRunning parallel for pragma\n");
		print_parallel();
		#pragma omp for
			for(int i = 0; i < N; i++){
				printf("\t\tThread %d doing the loop\n", omp_get_thread_num());
			}
	}
}

void pi(){
	printf("\n\n***************pi()**************\n");
	int N = 72;
	float arr[N];
	float granularity = 1/((float) N);
	float val = 0;
	#pragma omp parallel
	{
		#pragma omp for
		for(int i = 0; i < N; i ++){
			arr[i] = granularity*(sqrt(1 - (i*granularity)*(i*granularity)));
		}
	}
	for(int i = 0; i < N; i++){
		val += arr[i];
	}
	val = 4*val;
	printf("\nEstimation of pi: %f\n", val);
}

void schedules(){

}
