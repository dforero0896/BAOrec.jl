# BAOrec
Julia code for iterative and finite-difference BAO reconstruction on the CPU and GPU.

### Usage
To use, you may write on your REPL
```julia
] develop https://github.com/dforero0896/BAOrec.jl.git
```
since the package is not registered.

Remember that for optimal performance it is better to launch a single Julia run for many reconstructions so compilation is only triggered once. Launching a julia script for every catalog will imply that lots of time is lost in compilation and while fast, it won't be optimal. For instance, GPU kernels take ~30s for a reconstruction the first time they are called, 90+% of which is compilation.

### Usage on NERSC Perlmutter

To use the code in Perlmutter do the following
```bash

module load PrgEnv-nvidia 
module load julia/1.7.2-nvidia

```
then you will be able to type in the julia REPL package manager (in your terminal type `julia`, then press `]`)
```julia
 develop https://github.com/dforero0896/BAOrec.jl.git
 add <Whatever registered package you may need>
```

To run a job:
```bash
srun -n1 -c128 -C gpu --account=mxxxx_g --gpus-per-task=1 julia -t 128 examples/lightcone_gpu.jl
```
For more information check how to submit jobs to Perlmutter.

### Benchmarks

The benchmarks below are from the scripts in `test/` and performed with `BenchmarkTools.jl` and the `Base` Julia timer (for the less important parts). On the [CPU](https://github.com/dforero0896/BAOrec.jl/blob/main/test/benchmarks_cpu.out)
https://github.com/dforero0896/BAOrec.jl/blob/d071d390c88f83ad790135ccb2c2adc17977d21b/test/benchmarks_cpu.out#L1-L14
On the [GPU](https://github.com/dforero0896/BAOrec.jl/blob/main/test/benchmarks_gpu.out)
https://github.com/dforero0896/BAOrec.jl/blob/d071d390c88f83ad790135ccb2c2adc17977d21b/test/benchmarks_gpu.out#L1-L77
  
I am not *that* confident I am properly benchmarking the GPU runs given that they look a bit *too* fast.


Issues are welcome. 


