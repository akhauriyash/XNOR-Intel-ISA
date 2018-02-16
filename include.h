#include <stdio.h>
#include "omp.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

int f(int x);
int g(int x);
int h(int x);
void proc_count();
void thread_array_add();
void call_parallel();
void thread_limiter();
void print_parallel();
void pi();
void schedules_pi();
void section_demo();
void single_master_demo();
void private_demo();
void default_demo();
void array_data();
void thread_privacy();
void copy_in_private();
void reduction();
void cache_line();
