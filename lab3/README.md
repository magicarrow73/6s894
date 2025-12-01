Code for 6.S894 [Lab 3](https://accelerated-computing-class.github.io/fall24/labs/lab3).

kcong comments:

[computations for naive GPU implementation] I will answer question 1 of the writeup. 

(1.1) The `u0` and `u1` buffers are of size $3201^2 \times \mathrm{sizeof}(float) = 40985604$ bytes, i.e. 40MB. Combined, they are of size 80MB. The L2 cache in total is 48MB, which can contain exactly one of `u0` or `u1` (or parts of both) but not both simultaneously.

(1.2) total of 6 reads from u0 and u1 and 1 write to u0, i.e. 7 accesses per loop iteration, meaning throughout the kernel there are a total of $7 \times 40\mathrm{MB} = 280\mathrm{MB}$ of data through the L2 cache. 

(1.3) cache will store the neighboring values of `u0` and `u1`. So, in essence there are only 3 accesses (2 reads and 1 write) which miss the cache. This yields $120\mathrm{MB}$, i.e. $43\%$ misses. 

(1.4) Note that bandwidth of DRAM is roughly 360GB/s. We would need to transfer 120MB through the DRAM, yielding 0.33ms. The actual time is longer, likely due to additional constraints. However, it is on the same order of magnitude. 

(1.5) Let $L_2$ denote latency of L2 cache. For 280MB at 2.4TB/s, we would get $1.16 \times 10^{-3}$ milliseconds of runtime, which is significiantly better than lab performance. 

(1.6) See comments to (1.4, 1.5). 

(1.7) We should try to avoid DRAM. The L2 cache access is essentially negligible, so we should try to attempt to avoid L2 cache misses. On the other hand, there seem to also be other nontrivial issues. 