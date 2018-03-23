# XNOR Intel ISA
Attempting to implement XNOR neural network functions (Convolution, GEMM ...) on Intel ISA.

For CUDA compatiable (Nvidia GPU) XNOR convolutional kernel, check out [this repository.](https://github.com/akhauriyash/XNOR-convolution)

  * The first part of this project aims to create an introductory c file to teach people about OpenMP and parallel programming.
  
##  Prerequisites:
  * Intel® Xeon Phi™ Processor (Any Intel processor should do but code shall attempt to leverage AVX-512 ISA exclusively)
    
##  Note:
  This is a work in progress. There might be some mistakes here. 
  Do let me know if you find any logical errors in the code.
  
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

  
## To run:
   Execute this as
 	`gcc experiments.c -fopenmp -lm`
  	in the terminal.

 
 
##  TO DO:
  - [ ] Upload codes
  - [ ] Update code gists
  - [ ] Create CLI
  - [ ] Create function timers
  - [ ] Link main function with parser
