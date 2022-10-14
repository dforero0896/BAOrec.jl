using Revise
using BAOrec
using NPZ
using Plots
using CSV
using StaticArrays
using Statistics
using CUDA
using KernelAbstractions
using FFTW
using Random

test_fn = "/home/astro/dforero/codes/BAOrec//data/CATALPTCICz0.466G960S1010008301_zspace.dat.npy"

n_grid = 512
los = @SVector [0f0, 0f0, 1f0]
data = Matrix(npzread(test_fn)')
box_size = @SVector [2500f0, 2500f0, 2500f0]
box_min = @SVector [0f0, 0f0, 0f0]
data = data[:,mapslices(pos -> all([pos[i]<box_size[i] for i in eachindex(pos)]), data, dims=1)']
data = map(cu, [data[i,:] for i in 1:3])
data_w = cu(zero(data[1]) .+ 1)
rho = CuArray(zeros(Float32, [n_grid for i in 1:3]...))


recon = BAOrec.MultigridRecon(bias = 2.2f0, f = 0.757f0, 
                              smoothing_radius = 15f0,
                              box_size = box_size, 
                              box_min = box_min,
                              los = los)
@time BAOrec.setup_fft!(recon, rho)                          
@time BAOrec.setup_overdensity!(rho,
                        recon,
                        data..., data_w,
                        );

p1 = heatmap(dropdims(mean(Array(rho), dims=1), dims=1), aspect_ratio = :equal)

#exit()
ϕ = zero(rho)
rho = nothing

@time BAOrec.reconstructed_potential!(ϕ, recon, data..., data_w)
p2 = heatmap(dropdims(mean(Array(ϕ), dims=1), dims=1), aspect_ratio = :equal)
plot(p1, p2)
savefig("/home/astro/dforero/codes/BAOrec/examples/simulation_mg_gpu.png")

#disp_mg = map(Array, BAOrec.compute_displacements(ϕ, data..., recon))
#@show [d[10] for d in disp_mg]
#disp_mg = nothing
#ϕ = nothing
#
#fill!(rho,0)
#recon_it = 
#@time BAOrec.reconstructed_overdensity!(rho, recon, data..., data_w)
#disp_it = map(Array, BAOrec.compute_displacements(ϕ, data..., recon))
#@show [d[10] for d in disp_it]

@time new_pos = BAOrec.reconstructed_positions(recon, data..., ϕ; field = :sum);
rx = CuArray{Float32, 1}(undef, 10 * size(data[1],1))
rand!(rx)
@. rx *= recon.box_size[1]
ry = CuArray{Float32, 1}(undef, 10 * size(data[1],1))
rand!(ry)
@. ry *= recon.box_size[2]
rz = CuArray{Float32, 1}(undef, 10 * size(data[1],1))
rand!(rz)
@. rz *= recon.box_size[3]
randoms = (rx, ry, rz)
@time new_rand_sym = BAOrec.reconstructed_positions(recon, randoms..., ϕ; field = :sum);
@time new_rand_iso = BAOrec.reconstructed_positions(recon, randoms..., ϕ; field = :disp);


@time new_pos = map(Array, new_pos)
@time new_rand_sym = map(Array, new_rand_sym)
@time new_rand_iso = map(Array, new_rand_iso)

@show [new_pos[i][j] for j = 1:4, i=1:3]
#[(new_pos[i])[j] for j = 1:4, i = 1:3] = Float32[2494.5579 3.2764661 43.746624; 2496.5996 4.640162 44.583473; 2495.943 12.931675 76.369484; 3.9973264 6.7710137 380.89532]

npzwrite("/home/astro/dforero/codes/BAOrec//data/MG_GPU_CATALPTCICz0.466G960S1010008301_zspace.dat.rec.npy", hcat(new_pos...))
npzwrite("/home/astro/dforero/codes/BAOrec//data/MG_GPU_CATALPTCICz0.466G960S1010008301_zspace.ran.rec.sym.npy", hcat(new_rand_sym...))
npzwrite("/home/astro/dforero/codes/BAOrec//data/MG_GPU_CATALPTCICz0.466G960S1010008301_zspace.ran.rec.iso.npy", hcat(new_rand_iso...))