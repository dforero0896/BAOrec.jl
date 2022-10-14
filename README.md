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
```
Using 128 threads.
CPU benchmarking...
Struct
  0.015896 seconds (20.93 k allocations: 1.261 MiB, 96.31% compilation time)
Reconstruction iterative run
  27.428 s (46141 allocations: 7.52 GiB)
Read positions from recon result
  585.135 ms (129131052 allocations: 3.72 GiB)
Struct
  0.015992 seconds (25.80 k allocations: 1.562 MiB, 96.49% compilation time)
Reconstruction multigrid run
  24.792 s (1734580 allocations: 19.96 GiB)
Read positions from recon result
  1.448 s (129131100 allocations: 4.22 GiB)
```

On the [GPU](https://github.com/dforero0896/BAOrec.jl/blob/main/test/benchmarks_gpu.out)
```Using 128 threads.
GPU benchmarking...
Data copy to GPU
  1.656354 seconds (2.31 M allocations: 222.114 MiB, 77.85% compilation time)
  0.067086 seconds (49.28 k allocations: 101.376 MiB, 51.22% compilation time)
Struct
  0.016367 seconds (20.93 k allocations: 1.261 MiB, 94.44% compilation time)
Reconstruction iterative run
TrialEstimate(595.770 ms)
BenchmarkTools.Trial
  params: BenchmarkTools.Parameters
    seconds: Float64 5.0
    samples: Int64 10000
    evals: Int64 1
    overhead: Float64 0.0
    gctrial: Bool true
    gcsample: Bool false
    time_tolerance: Float64 0.05
    memory_tolerance: Float64 0.01
  times: Array{Float64}((9,)) [6.19338084e8, 5.85914455e8, 5.95770487e8, 5.95403127e8, 5.89746708e8, 6.44420176e8, 6.67371721e8, 5.82569621e8, 6.0782504e8]
  gctimes: Array{Float64}((9,)) [0.0, 0.0, 0.0, 0.0, 4.136115e6, 0.0, 0.0, 4.294532e6, 0.0]
  memory: Int64 616880
  allocs: Int64 10106
Read positions from recon result
TrialEstimate(52.010 ms)
BenchmarkTools.Trial
  params: BenchmarkTools.Parameters
    seconds: Float64 5.0
    samples: Int64 10000
    evals: Int64 1
    overhead: Float64 0.0
    gctrial: Bool true
    gcsample: Bool false
    time_tolerance: Float64 0.05
    memory_tolerance: Float64 0.01
  times: Array{Float64}((91,)) [5.0593235e7, 6.8096218e7, 1.02470894e8, 4.7207952e7, 5.0682268e7, 4.6011949e7, 5.3779904e7, 6.1123841e7, 7.4825356e7, 4.2027007e7  …  4.8147999e7, 4.4062783e7, 4.5516833e7, 5.1040159e7, 5.0341599e7, 5.830944e7, 4.7245284e7, 5.3624995e7, 5.3261573e7, 4.7555834e7]
  gctimes: Array{Float64}((91,)) [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  memory: Int64 80384
  allocs: Int64 1431
Copy results to CPU
  0.021447 seconds (4.30 k allocations: 98.762 MiB, 24.60% compilation time)
Struct
  0.016492 seconds (25.80 k allocations: 1.562 MiB, 95.65% compilation time)
Reconstruction multigrid run
TrialEstimate(1.820 s)
BenchmarkTools.Trial
  params: BenchmarkTools.Parameters
    seconds: Float64 5.0
    samples: Int64 10000
    evals: Int64 1
    overhead: Float64 0.0
    gctrial: Bool true
    gcsample: Bool false
    time_tolerance: Float64 0.05
    memory_tolerance: Float64 0.01
  times: Array{Float64}((3,)) [1.819561052e9, 1.917072578e9, 1.767079466e9]
  gctimes: Array{Float64}((3,)) [0.0, 1.7609118e7, 0.0]
  memory: Int64 28719152
  allocs: Int64 445156
Read positions from recon result
TrialEstimate(43.777 ms)
BenchmarkTools.Trial
  params: BenchmarkTools.Parameters
    seconds: Float64 5.0
    samples: Int64 10000
    evals: Int64 1
    overhead: Float64 0.0
    gctrial: Bool true
    gcsample: Bool false
    time_tolerance: Float64 0.05
    memory_tolerance: Float64 0.01
  times: Array{Float64}((105,)) [4.4726826e7, 3.9916308e7, 4.1223725e7, 4.5721919e7, 4.1492334e7, 4.1519807e7, 4.9516615e7, 4.3800055e7, 4.2899173e7, 4.6753172e7  …  4.3837387e7, 3.8426798e7, 3.8705437e7, 4.8431356e7, 4.432548e7, 3.7102207e7, 3.9450497e7, 5.8317155e7, 5.4813162e7, 4.5284053e7]
  gctimes: Array{Float64}((105,)) [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.454491e6  …  0.0, 0.0, 0.0, 0.0, 3.027712e6, 0.0, 0.0, 0.0, 0.0, 0.0]
  memory: Int64 75824
  allocs: Int64 1433
Copy results to CPU
  0.013548 seconds (10 allocations: 98.512 MiB)
  ```
  
I am not *that* confident I am properly benchmarking the GPU runs given that they look a bit *too* fast.


Issues are welcome. 


