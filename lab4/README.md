Starter code for 6.S894 [Lab 4](https://accelerated-computing-class.github.io/fall24/labs/lab4).


Comments / Answers to Questions:

Prelab 1: for each entry $(AB)_{ij}$, there are $2n-1$ FLOPs required, yielding a total of $(2n-1)n^2$ FLOPs. We will have to load $A$ and $B$, and then write back the result $AB$, yielding $3n^2 \times 4$ bytes of data transfer. So, the total operational intensity is $\frac{2n-1}{3} \approx \frac23 n$. 

Prelab 2: the transition occurs when the operational intensity is exactly the intersection of the two segments in the roofline model. In particular, the operational intensity must equal $\frac{26.73 \mathrm{TFLOPS/s}}{360 \mathrm{GB/s}} \approx 74.25 FLOPs/GB$; equating this to the value found in prelab 1, we find that $n \approx 111$ is the cut point. 

