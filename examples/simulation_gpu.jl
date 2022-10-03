using Revise
using BAOrec
using NPZ
using Plots
using CSV
using StaticArrays
using Statistics
using CUDA

test_fn = "/home/users/d/dforeros/.julia/dev/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.dat.npy"

n_grid = 64
los = @SVector [0f0, 0f0, 1f0]
data = Matrix(npzread(test_fn)')
box_size = @SVector [500f0, 500f0, 500f0]
box_min = @SVector [0f0, 0f0, 0f0]
data = data[:,mapslices(pos -> all([pos[i]<box_size[i] for i in eachindex(pos)]), data, dims=1)']


data_w = zero(view(data, 1,:)) .+ 1
rho = CuArray(zeros(Float32, [n_grid for i in 1:3]...))
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
heatmap(dropdims(mean(rho, dims=1), dims=1))

fill!(rho, 0);
@time BAOrec.reconstructed_overdensity!(rho,
                                recon,
                                view(data, 1,:), view(data, 2,:), view(data, 3,:),
                                data_w, );

@time new_pos = BAOrec.reconstructed_positions(recon, view(data, 1,:), view(data, 2,:), view(data, 3,:), rho; field = :sum);
@time rx, ry, rz = box_size[1] * rand(Float32, 10 .* size(data,2)), box_size[2] * rand(Float32, 10 .* size(data,2)), box_size[3] * rand(Float32, 10 .* size(data,2));
@time new_rand_sym = BAOrec.reconstructed_positions(recon, rx, ry, rz, rho; field = :sum);
@time new_rand_iso = BAOrec.reconstructed_positions(recon, rx, ry, rz, rho; field = :disp);


npzwrite("/home/users/d/dforeros/.julia/dev/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.dat.rec.npy", hcat(new_pos...))
npzwrite("/home/users/d/dforeros/.julia/dev/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.ran.rec.sym.npy", hcat(new_rand_sym...))
npzwrite("/home/users/d/dforeros/.julia/dev/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.ran.rec.iso.npy", hcat(new_rand_iso...))