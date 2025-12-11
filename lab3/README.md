Code for 6.S894 [Lab 3](https://accelerated-computing-class.github.io/fall24/labs/lab3).

kcong comments:

[computations for naive GPU implementation] I will answer question 1 of the writeup. 

(1.1) The `u0` and `u1` buffers are of size $3201^2 \times \mathrm{sizeof}(float) = 40985604$ bytes, i.e. 40MB. Combined, they are of size 80MB. The L2 cache in total is 48MB, which can contain exactly one of `u0` or `u1` (or parts of both) but not both simultaneously.

(1.2) total of 6 reads from u0 and u1 and 1 write to u0, i.e. 7 accesses per loop iteration, meaning throughout the kernel there are a total of $7 \times 40\mathrm{MB} = 280\mathrm{MB}$ of data through the L2 cache. 

(1.3) cache will store the neighboring values of `u0` and `u1`. So, in essence there are only 3 accesses (2 reads and 1 write) which miss the cache. This yields $120\mathrm{MB}$, i.e. $43\%$ misses. 

(1.4) Note that bandwidth of DRAM is roughly 360GB/s. We would need to transfer 120MB through the DRAM, yielding 0.33ms per iteration. There are 12800 iterations, yielding 4.27s. The actual time is longer, likely due to additional constraints. However, it is on the same order of magnitude, and not notably longer, suggesting that most of the time is indeed caused by DRAM bandwidth. Note that if we add the time for 160MB to be transferred through the L2 cache, this yields roughly 0.853s, giving around 5.12s. This is almost exactly the actual time taken. 

(1.5) The bandwidth of the L2 cache is roughly 2.4TB/s. We would need to transfer 280MB (under the assumption of the question), yielding a total of $1.16 \times 10^{-4}$ms of per iteration, or 1.493s in total. This is a nontrivial speedup.  

(1.6) See comments to (1.4, 1.5). 

(1.7) We should try to avoid DRAM. The L2 cache access is superior by a nontrivial factor; therefore, we should mainly attempt to load/store data with it. Of course, stores to DRAM cannot be avoided, but we can hope that we only read and write to `u0` and `u1` 3 times (2 reads, 1 write) in total. 

I will now sketch question 2 of the writeup. 

(2) We attain an additional 2x factor speedup by using shared memory (1.55x for the large tests). This maeks sense because first, the shared memory items must be loaded in. If we perform 4 steps per iterations, we would expect ~4x speedup, but we also need the extra halo, which reduces the total factor. As an estimate, we might expect a speedup of $4 \times \left(\frac{24}{32}\right)^2 \approx 2.2$. In practice, we have attained slightly smaller speedups, but this is nonetheless reasonable. 

Implementation-wise, the main method is to employ a small halo around the computation region. Threads in the same thread block are able to use shared memory to compute several steps of the evolution. 