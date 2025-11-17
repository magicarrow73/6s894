Starter code for 6.S894 [Lab 2](https://accelerated-computing-class.github.io/fall24/labs/lab2).


kcong edits: to run the logs, use cat telerun-out/<log_number>/execute-log.txt | python3 lab2/plot.py 

notes / answers to useful questions / etc: 

[latency] latencies 4 clock cycles for sequential, 2 clock cycles for ILP (intervowen and not); compiler will do the work of scheduling. a bit extra presumably due to various overheads, using #pragma unroll makes it clear that these are the latencies (+4clock cycles of overhead for all three). 

[#warps] the optimal choices of #warps are all multiples of 768, since for 192 warps, with latency 4 it can schedule exactly one warp every clock cycle, which results in max throughput. TODO: understand why it is bad for, e.g., more than 768. Seems like one can still just ignore some of the extra warps and it can't hurt? This is quite confusing. 