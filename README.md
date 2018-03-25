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

Run xCMMA.c for benchmarking the algorithm.

![Alt text](https://github.com/akhauriyash/XNOR-Intel-ISA/blob/master/xGEMM%20opt%20bmark.png?raw=true)
The image above is representing the results of an extremely crudely optimized code. Will post improvements as they come.

## xCONV (Binarized convolution on Intel Xeon Phi)

Run xCONV.c for benchmarking the algorithm.

![Alt text](https://github.com/akhauriyash/XNOR-Intel-ISA/blob/master/xCONV%20benchmark.png?raw=true)
The image above is representing the results of an extremely crudely optimized code. Will post improvements as they come.

**Note that both images have a logarithmic left vertical-axis.**

## Benchmarks
**xCONV benchmark** 

|  Matrix Size | FP CONV (s) | xCONV (s) | **Speed up** | Kernel size |
|  ------ | ------ | ------ | ------ | ------ |
|  4096 | 0.0354336 | 0.0400825 | 0.8840167155242313 | 4x4 |
|  2048 | 0.0350968 | 0.0089961 | 3.9013350229543913 | 4x4 |
|  1024 | 0.0298355 | 0.0022663 | 13.164850196355292 | 4x4 |
|  512 | 0.0017948 | 0.0005657 | 3.1727063814742795 | 4x4 |
|  256 | 0.0020824 | 0.0001499 | 13.891927951967977 | 4x4 |
|  128 | 0.0013783 | 0.000078 | 17.670512820512823 | 4x4 |
|  64 | 0.0018215 | 0.0000431 | **42.262180974477964** | 4x4 |

**xGEMM benchmark**

|  Matrix size | CMMA time | xGEMM optimized | **Speedup** |
|  ------ | ------ | ------ | ------ |
|  16384 | 182.287304 | 2.7664792 | 65.89144209000379 |
|  8192 | 14.2204277 | 0.3570589 | 39.826559987721915 |
|  4096 | 1.5708227 | 0.04895476 | 32.08723114973906 |
|  2048 | 0.1876822 | 0.0041507 | 45.21699954224588 |
|  1024 | 0.0245256 | 0.001247 | 19.667682437850843 |
|  512 | 0.0071147 | 0.0002131 | 33.38667292351009 |
|  256 | 0.0018346 | 0.0000558 | 32.878136200716845 |

## To run:
   These codes have been written on the Colfax Cluster optimized for **Intel Xeon Phi KNL 7210.**
   To run xCMMA.c, execute this:
   `icpc -xMIC-AVX512 -qopenmp -mkl -fp-model fast=2 -fma -unroll=4 xCMMA.c -o xCMMA.out && echo ~/parallel/xCMMA.out | qsub`
   
   To learn more about OpenMP and attempt some basic exercises, execute:
 	`gcc experiments.c -fopenmp -lm`
  	in the terminal.


 
 
##  TO DO:
  - [ ] Upload codes
  - [ ] Update code gists
  - [ ] Create CLI
  - [ ] Create function timers
  - [ ] Link main function with parser
