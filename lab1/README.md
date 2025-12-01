Code for 6.S894 [Lab 1](https://accelerated-computing-class.github.io/fall24/labs/lab1).


kcong comments [some of these are from the lab questions, some are independent].

(1) scalar GPU is notably slower (391.456ms) than scalar CPU (40.932ms). This is due to the superior optimizations on CPU for single-thread execution, leading to lower runtime (faster ALUs, etc.) as well as lower overhead (e.g. not necessary to call kernels). parallelized GPU is closer to vectorized CPU, but still 3x off. Note that we are only using one warp, which is a small fraction of the total compute/parallelization avabile on the GPU! 

(2) vectorized computation is ~10x faster for CPU (with 16 parallel execs), and ~22x faster for GPU (with 32 parallel execs). This is not the full extent of the amount of parallelization, likely due to control divergence. In particular, even if pixel $i$ has already escaped the disk of radius 2, it is possible that pixel $j$ has not, where $i$ and $j$ are in the same vectorized group (or CUDA thread). 

(3) Control divergence in the vectorized case is implemented simply by waiting for the slowest individual thread. In CUDA control divergence, we run all threads satisfying one condition first, then the remaining threads; in this case, this corresponds to running each thread equal to the number of times the slowest individual thread runs.