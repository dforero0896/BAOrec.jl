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

test_fn = "/home/astro/dforero/codes/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.dat.npy"

n_grid = 512
los = @SVector [0f0, 0f0, 1f0]
data = Matrix(npzread(test_fn)')
box_size = @SVector [2500f0, 2500f0, 2500f0]
box_min = @SVector [0f0, 0f0, 0f0]
data = data[:,mapslices(pos -> all([pos[i]<box_size[i] for i in eachindex(pos)]), data, dims=1)']


data_w = zero(view(data, 1,:)) .+ 1
rho = CuArray(zeros(Float32, [n_grid for i in 1:3]...))


recon = BAOrec.IterativeRecon(bias = 2.2f0, f = 0.757f0, 
                              smoothing_radius = 15f0, n_iter = 3,
                              box_size = box_size, 
                              box_min = box_min,
                              los = los)

BAOrec.setup_fft!(recon, rho)                              


@time BAOrec.setup_overdensity!(rho,
                        recon,
                        #view(data, 1,:), view(data, 2,:), view(data, 3,:), data_w,
                        cu(data[1,:]), cu(data[2,:]), cu(data[3,:]), cu(data_w), 
                        );

p1 = heatmap(dropdims(mean(Array(rho), dims=1), dims=1), aspect_ratio = :equal)
savefig("/home/astro/dforero/codes/BAOrec/examples/simulation_gpu.png")

fill!(rho, 0);
@time BAOrec.reconstructed_overdensity!(rho,
                                recon,
                                #view(data, 1,:), view(data, 2,:), view(data, 3,:), data_w, 
                                cu(data[1,:]), cu(data[2,:]), cu(data[3,:]), cu(data_w), 
                                );
p2 = heatmap(dropdims(mean(Array(rho), dims=1), dims=1), aspect_ratio = :equal)
plot(p1, p2, layout = 2)
savefig("/home/astro/dforero/codes/BAOrec/examples/simulation_gpu.png")

rho = Array(rho)
recon.fft_plan = plan_rfft(rho)
@time new_pos = BAOrec.reconstructed_positions(recon, view(data, 1,:), view(data, 2,:), view(data, 3,:), rho; field = :sum); #The bulk of the time is spent here, should profile.
@time rx, ry, rz = box_size[1] * rand(Float32, 10 .* size(data,2)), box_size[2] * rand(Float32, 10 .* size(data,2)), box_size[3] * rand(Float32, 10 .* size(data,2));
@time new_rand_sym = BAOrec.reconstructed_positions(recon, rx, ry, rz, rho; field = :sum);
@time new_rand_iso = BAOrec.reconstructed_positions(recon, rx, ry, rz, rho; field = :disp);

#@time new_pos = BAOrec.reconstructed_positions(recon, cu(data[1,:]), cu(data[2,:]), cu(data[3,:]), rho; field = :sum);
#@time rx, ry, rz = box_size[1] * rand(Float32, 10 .* size(data,2)), box_size[2] * rand(Float32, 10 .* size(data,2)), box_size[3] * rand(Float32, 10 .* size(data,2));
#@time rx, ry, rz = map(cu, (rx, ry, rz))
#@time new_rand_sym = BAOrec.reconstructed_positions(recon, rx, ry, rz, rho; field = :sum);
#@time new_rand_iso = BAOrec.reconstructed_positions(recon, rx, ry, rz, rho; field = :disp);

@time new_pos = map(Array, new_pos)
@time new_rand_sym = map(Array, new_rand_sym)
@time new_rand_iso = map(Array, new_rand_iso)

npzwrite("/home/astro/dforero/codes/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.dat.rec.npy", hcat(new_pos...))
npzwrite("/home/astro/dforero/codes/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.ran.rec.sym.npy", hcat(new_rand_sym...))
npzwrite("/home/astro/dforero/codes/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.ran.rec.iso.npy", hcat(new_rand_iso...))