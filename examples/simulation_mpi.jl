using Revise
using BAOrec
using NPZ
using Plots
using CSV
using StaticArrays
using Statistics
using MPI
using PencilArrays
using PencilFFTs
using Random


MPI.Init()
test_fn = "/home/astro/dforero/codes/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.dat.npy"

n_grid = 512
los = @SVector [0f0, 0f0, 1f0]
data = Matrix(npzread(test_fn)')
box_size = @SVector [2500f0, 2500f0, 2500f0]
box_min = @SVector [0f0, 0f0, 0f0]
data_w = ones(Float32, size(data,2))
comm = MPI.COMM_WORLD
pen = Pencil(Tuple(n_grid for _ in 1:3), comm)
transform = Transforms.RFFT()
plan = PencilFFTPlan(pen, transform)
rho = allocate_input(plan)
randn!(rho)
@show summary(rho)
rho_hat = plan * rho
exit()
rho_loc = PencilArray{Float32}(undef, pen);
@show summary(rho_loc)
@show typeof(pen)
fill!(rho_loc,0);
rho = global_view(rho_loc)

recon = BAOrec.IterativeRecon(bias = 2.2f0, f = 0.757f0, 
                              smoothing_radius = 15f0, n_iter = 3,
                              box_size = box_size, 
                              box_min = box_min,
                              los = los)
BAOrec.setup_fft!(recon, parent(rho))
ranges = range_local(parent(rho))

#BAOrec.cic!(rho, view(data, 1,:), view(data, 2,:), view(data, 3,:), data_w, recon.box_size,  recon.box_min; wrap = true)
recon.fft_plan * parent(rho)
exit()

@time BAOrec.setup_overdensity!(rho,
                        recon,
                        view(data, 1,:), view(data, 2,:), view(data, 3,:), data_w,
                        );
@show summary(rho)
exit()     
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


npzwrite("/home/users/d/dforeros/.julia/dev/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.dat.rec.mpi.npy", hcat(new_pos...))
npzwrite("/home/users/d/dforeros/.julia/dev/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.ran.rec.mpi.sym.npy", hcat(new_rand_sym...))
npzwrite("/home/users/d/dforeros/.julia/dev/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.ran.rec.mpi.iso.npy", hcat(new_rand_iso...))