# XNOR Intel ISA
Attempting to implement XNOR neural network functions (Convolution, GEMM ...) on Intel ISA.

For CUDA compatiable (Nvidia GPU) XNOR convolutional kernel, check out [this repository.](https://github.com/akhauriyash/XNOR-convolution)

## xGEMM (Binarized General Matrix Multiply on Intel Xeon Phi)

Run xCMMA.c for benchmarking the algorithm.

![Alt text](https://github.com/akhauriyash/XNOR-Intel-ISA/blob/master/xGEMM%20opt%20bmarks.png?raw=true)
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
|  4096 | 0.0354336 | 0.0400825 | 0.88403 | 4x4 |
|  2048 | 0.0350968 | 0.0089961 | 3.9013353 | 4x4 |
|  1024 | 0.0298355 | 0.0022663 | 13.164592 | 4x4 |
|  512 | 0.0017948 | 0.0005657 | 3.17270695 | 4x4 |
|  256 | 0.0020824 | 0.0001499 | 13.891977 | 4x4 |
|  128 | 0.0013783 | 0.000078 | 17.67053 | 4x4 |
|  64 | 0.0018215 | 0.0000431 | **42.26214** | 4x4 |

**xGEMM benchmark**

|  Matrix size | CMMA time | xGEMM optimized | **Speedup** | Binarization time |
|  ------ | ------ | ------ | ------ | ------ |
|  16384 | 182.287 | 2.7664 | 65.89149 | 0.2814668 |
|  8192 | 14.2207 | 0.35705 | 39.826515 | 0.0742801 |
|  4096 | 1.57082 | 0.04895 | 32.08706 | 0.0114089 |
|  2048 | 0.18772 | 0.004151 | 45.216988 | 0.0024204 |
|  1024 | 0.02456 | 0.00125 | 19.6676843 | 0.00056671 |
|  512 | 0.007147 | 0.000213 | 33.38669 | 0.0001167 |
|  256 | 0.0018346 | 0.0000558 | 32.8785 | 0.0000371 |

## To run:
   These codes have been written on the Colfax Cluster optimized for **Intel Xeon Phi KNL 7210.**
   To run xCMMA.c, execute this:
   `icpc -xMIC-AVX512 -qopenmp -mkl -fp-model fast=2 -fma -unroll=4 xCMMA.c -o xCMMA.out && echo ~/parallel/xCMMA.out | qsub`
   
   To learn more about OpenMP and attempt some basic exercises, execute:
 	`gcc experiments.c -fopenmp -lm`
  	in the terminal.

##  Hardware specifications:
  * Intel® Xeon Phi™ 7210 Processor
     **Properties = xeonphi,knl,knl7210,ram96gb,flat,quadrant**     
    
##  Note:
  This is a work in progress. There might be some mistakes here. 
  Do let me know if you find any logical errors in the code.
 
##  TO DO:
  - [ ] Upload codes
  - [ ] Update code gists
  - [ ] Create CLI
  - [ ] Create function timers
  - [ ] Link main function with parser
