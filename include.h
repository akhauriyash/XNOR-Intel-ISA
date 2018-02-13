#include <stdio.h>
#include "omp.h"
#include <stdlib.h>

int f(int x);
int g(int x);
int h(int x);
void proc_count();
void thread_array_add();
void call_parallel();
void thread_limiter();
void print_parallel();
int f(int x){x++; return x;}
int g(int x){x = x*x*x; return x;}
int h(int x){x = x*x; return x;}
