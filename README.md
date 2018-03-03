# XNOR Intel ISA
Attempting to implement XNOR neural network functions (Convolution, GEMM ...) on Intel ISA.

For CUDA compatiable (Nvidia GPU) XNOR convolutional kernel, check out [this repository.](https://github.com/akhauriyash/XNOR-convolution)

  * The first part of this project aims to create an introductory c file to teach people about OpenMP and parallel programming.
  
##  Prerequisites:
  * Intel® Xeon Phi™ Processor (Any Intel processor should do but code shall attempt to leverage AVX-512 ISA exclusively)
    
##  Note:
  This is a work in progress. There might be some mistakes here. 
  Do let me know if you find any logical errors in the code.
  
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
