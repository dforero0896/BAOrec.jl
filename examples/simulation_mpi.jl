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
using LinearAlgebra


MPI.Init()
comm = MPI.COMM_WORLD
mpi_rank = MPI.Comm_rank(comm)
mpi_size = MPI.Comm_size(comm)
test_fn = "/home/astro/dforero/codes/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.dat.npy"

n_grid = 512
los = @SVector [0f0, 0f0, 1f0]
data = Matrix(npzread(test_fn)')
box_size = @SVector [2500f0, 2500f0, 2500f0]
box_min = @SVector [0f0, 0f0, 0f0]
data_w = ones(Float32, size(data,2))
pen = Pencil(Tuple(n_grid for _ in 1:3), comm)
transform = Transforms.RFFT()
recon = BAOrec.IterativeRecon(bias = 2.2f0, f = 0.757f0, 
                              smoothing_radius = 15f0, n_iter = 3,
                              box_size = box_size, 
                              box_min = box_min,
                              los = los)
recon.fft_plan = PencilFFTPlan(pen, transform, Float32)
rho_loc = allocate_input(recon.fft_plan)
fill!(rho_loc,0);
rho = global_view(rho_loc)
if mpi_rank == 0
    for i in 1:mpi_size
        @show range_remote(rho_loc, i)
    end #for
    
end #if

BAOrec.send_to_relevant_process([10,10,10], rho_loc, 10., mpi_size, mpi_rank, comm)
MPI.Barrier(comm)


exit()

BAOrec.cic!(rho_loc, view(data, 1,:), view(data, 2,:), view(data, 3,:), data_w, recon.box_size,  recon.box_min; wrap = true)
#
#rho_hat = recon.fft_plan * rho_loc
#ldiv!(rho_loc, recon.fft_plan, rho_hat)
#BAOrec.smooth!(rho_loc, recon.smoothing_radius, recon.box_size, recon.fft_plan)



@time BAOrec.setup_overdensity!(rho_loc,
                        recon,
                        view(data, 1,:), view(data, 2,:), view(data, 3,:), data_w,
                        );

fill!(rho, 0);
@time BAOrec.reconstructed_overdensity!(rho_loc,
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