---
layout: post
title:  "Writing advanced CUDA GEMM"
date:   2024-09-03 10:00:40
blurb: "A look at an example post using Bay Jekyll theme."
og_image: /assets/img/content/post-example/Banner.jpg
---

This post illustrates the advanced CUDA GEMM implementations with code ([link(https://github.com/gajagajago/cuda-advanced)]). For basic implementations from naive GEMM to 2D block tiling + 2D thread tiling, refer to my previous step-by-step optimization work ([link](https://github.com/gajagajago/fastgen/tree/main/matmul-opt)). Performance of each implementation is profiled on NVIDIA V100 GPU.

Terms:
- BLOCK_M, BLOCK_N: Thread block configuration
- TILE_M, TILE_K, TILE_N: Tile configuration

Performance baseline:
- V100 R_peak: 15.7 FP32 TFLOPS, 125 Tensor TFLOPS
- cuBLAS profiled R_peak: 10.8 FP32 TFLOPS, 80 Tensor TFLOPS

Notes:
- Loop unrolling with #pragma is not used for readability.
- Adding +1 to SMEM innermost dimension can help reduce SMEM bank conflict.

### 2D block tiling + 2D thread tiling

- Code: [mm32.cu](https://github.com/gajagajago/cuda-advanced/blob/main/mm32.cu)
- Performance: 6.7 FP32 TFLOPS

Block tiling uses SMEM to cache A and B tile commonly required by the threads in the same thread block. Still, a thread computes one element of C. To do so, TILE_K elements from Tile A and TILE_K elements from Tile B is read from SMEM, resulting in 2 * TILE_K SMEM loads for computing one Tile C element.

Conversely, in thread tiling, a thread computes REG_M * REG_N elements of Tile C. To do so, REG_M * TILE_K elements from Tile A and TILE_K * REG_N elements from Tile B is read from SMEM, resulting in (REG_M + REG_N) * TILE_K SMEM loads for computing REG_M * REG_N Tile C elements. Since $$\frac{(REG_M + REG_N) * TILE_K}{REG_M * REG_n} \lt {2 * TILE_K}$$, the SMEM is accessed less frequently.

In the code, a thread computes REG_M * REG_N Tile C elements, but the elements are separated by BLOCK_M and BLOCK_N in row and column directions, respectively. It is to avoid SMEM bank conflict between threads in a same warp, and leverage coalesced memory access when writing to C.

This can be vectorized, where a thread computes REG_M * (REG_N vector) Tile C elements. 

- code: [mm32v.cu](https://github.com/gajagajago/cuda-advanced/blob/main/mm32v.cu)
- Performance: 11.4 FP32 TFLOPS

### 2D block tiling + 2D warp tiling + 2D thread tiling

- Code: [mm32w.cu](https://github.com/gajagajago/cuda-advanced/blob/main/mm32w.cu)
- Performance: 5.2 FP32 TFLOPS

In CUDA, a warp is a SIMT scheduling unit that consists of 32 contiguous threads in the same thread block. DRAM accesses of a warp's threads can be coalesced if the accessed addresses are contiguous (not necessarily contiguous in the order of threads, though). In order to leverage Tensor core API on Warp-level Matrix Multiply Add (WMMA), tiling in warp level is important.

Following equations generally hold true when using warp tiling:
- Thread block tile size * Number of thread block = C matrix size
- Warp tile size * Number of warps in a thread block = Thread block tile size
- Warp tile size % (Thread tile size * Number of threads in a warp) = 0

In the code, warps are given y (WARP_Y) and x (WARP_X) indexes according to their computing warp tile index in the thread block tile. The threads in a warp can compute multiple tiles in a warp tile, depending on the thread tile size (THREAD_TILE_M and THREAD_TILE_N). Similar to warps, threads are given y (THREAD_Y_IN_WARP) and x (THREAD_X_IN_WARP) indexes to their computing thread tile index in the warp tile.

This can be vectorized, where a thread computes THREAD_TILE_M * (THREAD_TILE_N vector) Tile C elements. Note that when loading B tile elements in SMEM to registers (b_reg), V elements in the same column are loaded in the same iteration at the innermost loop. It is to match the vector size of A tile elements.

- code: [mm32wv.cu](https://github.com/gajagajago/cuda-advanced/blob/main/mm32wv.cu)
- Performance: 8.9 FP32 TFLOPS

### 2D block tiling + 2D warp tiling + Tensor core

Tensor core provides fast mixed-precision MMA on various data precisions including half. One simplest way to leverage the tensor cores is via WMMA APIs. It makes the implementation simpler because the thread tiling is abstracted away. A key characteristic of WMMA APIs is that a warp's threads operate collaboratively.

Each thread in a warp access the matrix data in the form of "fragment", and the data of the fragment is distributed to the threads. WMMA APIs with "_sync" suffix is synchronized among the threads in a same warp. For instance, load_matrix_sync is guaranteed to be completed by all threads in the warp before any other following operations such as mma_sync.  

WMMA introduces another level of tiling with WMMA tile stored to the fragment. Typically, from the thread block tiles of A and B cached to SMEM, the threads in a warp iteratively load the fragment data of A and B and save the fragment MMA output to C fragment.

- code: [mm16tc.cu](https://github.com/gajagajago/cuda-advanced/blob/main/mm16tc.cu)
- Performance: 27.8 Tensor TFLOPS

### Misc. implementations

2D block tiling + 2D thread tiling + Vector type + Column-major B
- code: [mm32cv.cu](https://github.com/gajagajago/cuda-advanced/blob/main/mm32cv.cu)
- Performance: 7.9 FP32 TFLOPS

### References
- <a href="https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/">https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/</a>
- <a href="https://siboehm.com/articles/22/CUDA-MMM">https://siboehm.com/articles/22/CUDA-MMM</a>
