/*	This example optimizes the performance of a program that is a:
	prime number counting algorithm that uses simple brute force testing 
	Keeps count of prime numbers that fall in the form of 4k+1 / 4k + 3. (3 to 1 million)
*/
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int main(){
	int numP41 = 0;
	int numP43 = 0;
	/*
	High communication overhead: synchronization
	*/
	#pragma omp parallel
	{	int j, limit, prime;
	#pragma for schedule(dynamic, 1)		
	//	(dynamic, 1) ==> Schedule distribute one iteration at a time dynamically to each thread
	//	each thread processes one iteration and then return to the scheduler, synchronize for next.
	//	Increase chunk size. Increasing granularity can cause load imbalance.
		for(i = 3; i <= 1000000; i += 2){
			limit = (int) sqrt((float) i) + 1;
			prime = 1;						//	Assume number is prime
			j = 3;
			while(prime && (j <= limit)){
				if(i%j == 0) prime = 0;
				j += 2;
			}
			if(prime) {
				#pragma omp critical
				{
					numPrimes++;
					if(i%4 == 1)	numP41++;
					if(i%4 == 3)	numP43++;
				}
			}
		}
	}
	/*  */
	#pragma omp parallel
	{
		int j, limit, prime;
		#pragma for schedule(dynamic, 1)\
			reduction(+:numPrimes, numP41, numP43)
		for(i = 3; i<=1000000; i+=2){
			limit = (int) sqrt((float) i) + 1;
			prime = 1;
			j = 3;
			while(prime && (j <= limit)){
				if(i%j == 0) prime = 0;
				j += 2;
			}
			if(prime)
			{
				numPrimes++;
				if(i%4 == 1)	numP41++;
				if(i%4 == 3)	numP43++;
			}
		}
	}
}
