using Revise
using BAOrec
using NPZ
using Plots
using CSV
using StaticArrays
using Statistics
using NeuralPDE
using Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval
using Interpolations

test_fn = "/home/astro/dforero/codes/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.dat.npy"

n_grid = 512
los = @SVector [0f0, 0f0, 1f0]
data = Matrix(npzread(test_fn)')
box_size = @SVector [2500f0, 2500f0, 2500f0]
box_min = @SVector [0f0, 0f0, 0f0]
data_w = ones(Float32, size(data,2))


rho = zeros(Float32, [n_grid for i in 1:3]...);
recon = BAOrec.IterativeRecon(bias = 2.2f0, f = 0.757f0, 
                              smoothing_radius = 15f0, n_iter = 3,
                              box_size = box_size, 
                              box_min = box_min,
                              los = los)
@time BAOrec.setup_fft!(recon, rho)                              
@time BAOrec.setup_overdensity!(rho,
                        recon,
                        view(data, 1,:), view(data, 2,:), view(data, 3,:), data_w,
                        );
x_vec = BAOrec.x_vec(rho, recon.box_size, recon.box_min)

@show map(size, x_vec)
@show size(rho)


delta_i = interpolate(x_vec, rho, Gridded(Linear()))
bias = 2.2
@show delta_i(1000,1000,1000)

@parameters x y z
@variables ϕ(..) δ(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dzz = Differential(z)^2
delta_dm(x,y,z) = delta_i(x,y,z) / bias
@register_symbolic delta_dm(x,y,z)

# Boundary conditions
bcs = [ϕ(x,0) ~ 0.0, ϕ(y,0) ~ 0.0, ϕ(z,0) ~ 0.0,
        ϕ(x,2500) ~ 0.0, ϕ(y,2500) ~ 0.0, ϕ(z,2500) ~ 0.0]

# Space and time domains
domains = [x ∈ Interval(0.0,2500.0),
           y ∈ Interval(0.0,2500.0),
           z ∈ Interval(0.0,2500.0)]


eq = Dxx(ϕ(x, y, z)) + Dyy(ϕ(x, y, z)) + Dzz(ϕ(x, y, z)) ~ 0
#eq = [δ(x,y,z) ~ delta_dm(x,y,z),
#        Dxx(ϕ(x, y, z)) + Dyy(ϕ(x, y, z)) + Dzz(ϕ(x, y, z)) ~ -δ(x,y,z)]


dim = 3 # number of dimensions
chain = Lux.Chain(Dense(dim,16,Lux.σ),Dense(16,16,Lux.σ),Dense(16,1))

#dx = minimum(box_size ./ size(rho))
dx = 100.
discretization = PhysicsInformedNN(chain, GridTraining(dx))

@named pde_system = PDESystem([eq],bcs,domains,[x,y,z],[ϕ(x, y,z)])
prob = discretize(pde_system,discretization)

opt = OptimizationOptimJL.BFGS()

callback = function (p,l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, opt, callback = callback, maxiters=1000)
phi = discretization.phi




