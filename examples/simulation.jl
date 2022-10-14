using Revise
using BAOrec
using NPZ
using Plots
using CSV
using StaticArrays
using Statistics
using FFTW
using DataFrames
FFTW.set_num_threads(128)

data_cat_fn = "/global/cfs/projectdirs/desi/mocks/UNIT/HOD_Shadab/HOD_boxes/redshift0.9873/UNIT_DESI_Shadab_HOD_snap97_ELG_v0.txt"
data_cat_pos = DataFrame(CSV.File(data_cat_fn, delim = ' ', ignorerepeated = true, header=[:x, :y, :d, :z], types = [Float32 for _ in 1:4]))
@show summary(data_cat_pos)
data_cat_pos = map(values, (data_cat_pos[!,:x], data_cat_pos[!,:y], data_cat_pos[!,:z]))
data_cat_w = zero(data_cat_pos[1]) .+ 1
box_size = @SVector [1000f0, 1000f0, 1000f0]
box_min = @SVector [0f0, 0f0, 0f0]
const grid_size = (512, 512, 512)
const los = @SVector [0f0, 0f0, 1f0]

recon = BAOrec.IterativeRecon(bias = 2.2f0, f = 0.757f0, 
                              smoothing_radius = 15f0, n_iter = 3,
                              box_size = box_size, 
                              box_min = box_min,
                              los = los)
println("Run iterative")
@time BAOrec.run!(recon, grid_size,
                    data_cat_pos...,
                    data_cat_w,)

@time new_pos = BAOrec.reconstructed_positions(recon, data_cat_pos...; field = :sum);
@time recon_cat_pos = [box_size[i] * rand(Float32, 10 .* size(data_cat_pos[i],1)) for i in 1:3];
@time new_rand_sym = BAOrec.reconstructed_positions(recon, recon_cat_pos...; field = :sum);
@time new_rand_iso = BAOrec.reconstructed_positions(recon, recon_cat_pos...; field = :disp);


npzwrite("data/CPU_UNIT_DESI_Shadab_HOD_snap97_ELG_v0.dat.rec.npy", hcat(new_pos...))
npzwrite("data/CPU_UNIT_DESI_Shadab_HOD_snap97_ELG_v0.ran.rec.sym.npy", hcat(new_rand_sym...))
npzwrite("data/CPU_UNIT_DESI_Shadab_HOD_snap97_ELG_v0.ran.rec.iso.npy", hcat(new_rand_iso...))


recon = BAOrec.MultigridRecon(bias = 2.2f0, f = 0.757f0, 
                              smoothing_radius = 15f0,
                              box_size = box_size, 
                              box_min = box_min,
                              los = los)
println("Run multigrid")
@time BAOrec.run!(recon, grid_size,
                    data_cat_pos...,
                    data_cat_w,)

@time new_pos = BAOrec.reconstructed_positions(recon, data_cat_pos...; field = :sum);
@time recon_cat_pos = [box_size[i] * rand(Float32, 10 .* size(data_cat_pos[i],1)) for i in 1:3];
@time new_rand_sym = BAOrec.reconstructed_positions(recon, recon_cat_pos...; field = :sum);
@time new_rand_iso = BAOrec.reconstructed_positions(recon, recon_cat_pos...; field = :disp);


npzwrite("data/MG_CPU_UNIT_DESI_Shadab_HOD_snap97_ELG_v0.dat.rec.npy", hcat(new_pos...))
npzwrite("data/MG_CPU_UNIT_DESI_Shadab_HOD_snap97_ELG_v0.ran.rec.sym.npy", hcat(new_rand_sym...))
npzwrite("data/MG_CPU_UNIT_DESI_Shadab_HOD_snap97_ELG_v0.ran.rec.iso.npy", hcat(new_rand_iso...))