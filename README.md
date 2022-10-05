# BAOrec
Julia code for (iterative) BAO reconstruction on the CPU and GPU based on https://arxiv.org/abs/1504.02591.

### Usage
To use, you may write on your REPL
```julia
] develop https://github.com/dforero0896/BAOrec.jl.git
```
since the package is not registered.

Issues are welcome. 
### Future work 
+ Optimize the `reconstructed_positions` function since the GPU implementations are fast but reading the displacements seems to take longer.
+ Provide function implementations that allow to provide all the necessary buffers to avoid allocations within the functions. Buffers can now be created using the `preallocate_memory` function.
