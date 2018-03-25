# XNOR Intel ISA
Attempting to implement XNOR neural network functions (Convolution, GEMM ...) on Intel ISA.

For CUDA compatiable (Nvidia GPU) XNOR convolutional kernel, check out [this repository.](https://github.com/akhauriyash/XNOR-convolution)

  * The first part of this project aims to create an introductory c file to teach people about OpenMP and parallel programming.
  
##  Prerequisites:
  * Intel® Xeon Phi™ Processor (Any Intel processor should do but code shall attempt to leverage AVX-512 ISA exclusively)
    
##  Note:
  This is a work in progress. There might be some mistakes here. 
  Do let me know if you find any logical errors in the code.
  
## xGEMM (Binarized General Matrix Multiply on Intel Xeon Phi)

Run BinaryMultiply.c for benchmarking the algorithm.

![Alt text](https://github.com/akhauriyash/XNOR-Intel-ISA/blob/master/xGEMM%20opt%20bmark.png?raw=true)
The image above is representing the results of an extremely crudely optimized code. Will post improvements as they come.

## Benchmarks

|  Matrix size | CMMA time (s) | xGEMM optimized (s) | **Speedup** | Binarization time (s) | XNOR GEMM time (s) | **Speedup** |
|  ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  16384 | 182.287304 | 2.7664792 | 65.891479 | 0.2814668 | 41.75 | 4.36616805 |
|  8192 | 14.2204277 | 0.3570589 | 39.826555 | 0.0742801 | 4.4938908 | 3.16954985 |
|  4096 | 1.5708227 | 0.04895476 | 32.08726 | 0.0114089 | 0.0889784 | 17.653933593 |
|  2048 | 0.1876822 | 0.0041507 | 45.216998 | 0.0024204 | 0.0082477 | 22.75052 |
|  1024 | 0.0245256 | **0.001247** | **19.6676824** | 0.0005668 | 0.0009867 | 24.809699 |
|  512 | 0.0071147 | 0.0002131 | 33.3009 | 0.0001167 | 0.0003532 | 20.143534 |
|  256 | 0.0018346 | 0.0000558 | 32.875 | 0.0000371 | 0.0000724 | 25.339864 |


## To run:
   Execute this as
 	`gcc experiments.c -fopenmp -lm`
  	in the terminal.

## Binarization:
  Run the binarize.c code. In the preliminary round, the results are as such:
  (matrix size: 8192)

KMP_AFFINITY=scatter 

`Number of OpenMP threads:  64`

`Binarization A - Completed in: 0.0059493 seconds`

`Binarization B - Completed in: 0.0994816 seconds`

KMP_AFFINITY=balanced, granularity=fine

`Number of OpenMP threads:  64`

`Binarization A - Completed in: 0.0065423 seconds`

`Binarization B - Completed in: 0.0957288 seconds`

KMP_AFFINITY=compact

`Number of OpenMP threads:  64`

`Binarization A - Completed in: 0.0170125 seconds`

`Binarization B - Completed in: 0.1191385 seconds`

As matrices are cached in row major format and we access B column wise, it is no surprise that the binarization of B is so slow. It might be a better idea to first transpose the B matrix, and then do the binarization process for more cache hits. This is a very basic optimization technique. The binarization algorithm has a lot of scope for parallelization. 

 
 
##  TO DO:
  - [ ] Upload codes
  - [ ] Update code gists
  - [ ] Create CLI
  - [ ] Create function timers
  - [ ] Link main function with parser
