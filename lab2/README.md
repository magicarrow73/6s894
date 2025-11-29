Code for 6.S894 [Lab 2](https://accelerated-computing-class.github.io/fall24/labs/lab2).


kcong use notes: to run the logs, use cat telerun-out/<log_number>/execute-log.txt | python3 lab2/plot.py 

notes / answers to useful questions / etc: 

[latency] latencies 4 clock cycles for sequential, 2 clock cycles for ILP (intervowen and not); compiler will do the work of scheduling. a bit extra presumably due to various overheads, using #pragma unroll makes it clear that these are the latencies (+4clock cycles of overhead for all three). 

[#warps] the optimal choices of #warps are all multiples of 768, since for 192 warps, with latency 4 it can schedule exactly one warp every clock cycle, which results in max throughput. For multiples of 768, it can schedule an integer number of warps per clock cycle, and hence will not lose any performance. 

[#CPUperformance] Performance increases significantly with ILP. Performance increases with using extra cores: 8 cores, 1 thread/core (i.e. 8 total threads) launches and gives roughly 8x improvement. Using even more per core does not improve as much, giving only ~1.5x improvement, and ILP usage on top of that is roughly similar. The combined top performance we achieve is 4.3ms, which is 10x better than the naive CPU vector and 85x better than naive CPU scalar. 

[#GPUperformance] Initial performance weaker than CPU performance by a notable factor, due to lack of optimizations. Performance does not increase with ILP under the current implementation (possibly, due to more difficulties in control divergence and storage on device shared memory). Using multicore gives a significant ~27x improvement over the initial vector (i.e. using all the streaming multiprocessors (SMs)); using all 1024 threads for a single SM gives around 10x improvement. The combined top performance we achieve is 0.34ms, which is 500x better than the naive GPU vector. This is sensible, since we are using 48 x 32 = 1536 times as many threads running in parallel. 

[#design] For the CPU impl, we have different threads run different columns (i.e. by changing the $i$ value). This is the easiest to implement, and allows each thread to access $j$ values which are close and hence use memory coalescing. For the GPU impl, we simply continue parallelizing amongst rows; however, since 48 x 1024 = 49152 is larger than the width of what we are trying to implement, to use the resources well, we parallelize amongst both $i$ and $j$ in the full version. 

[comment]: Note: we do not answer all questions. also why 768xK also works while 768+K doesn't work is confusing to me. 