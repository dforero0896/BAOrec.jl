using BenchmarkTools
using BAOrec
using NPZ
using Plots
using CSV
using StaticArrays
using Statistics
using DataFrames
using QuadGK
using Profile
using FFTW
using CUDA
using Tables
FFTW.set_num_threads(128)

nthreads = Threads.nthreads()
print("Using $nthreads threads.\n")

data_cat_fn = "/global/cfs/projectdirs/desi/mocks/UNIT/HOD_Shadab/multiple_snapshot_lightcone/UNIT_lightcone_multibox_ELG_footprint_nz_NGC.dat"
rand_cat_fn = "/global/cfs/projectdirs/desi/mocks/UNIT/HOD_Shadab/multiple_snapshot_lightcone/UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.dat"

data_cat = DataFrame(CSV.File(data_cat_fn, delim=" ", header=[:ra, :dec, :z, :w, :nz], types = [Float32 for _ in 1:5]))
rand_cat = DataFrame(CSV.File(rand_cat_fn, delim=" ", header=[:ra, :dec, :z, :w, :nz], types = [Float32 for _ in 1:5]))


data_cat = data_cat[map(z -> ((z > 0.8) & (z < 1)), data_cat.z), :]
rand_cat = rand_cat[map(z -> ((z > 0.8) & (z < 1)), rand_cat.z), :]

function sky_to_cartesian(data_cat_ra, data_cat_dec, data_cat_red, cosmo)
    
    r_fun = BAOrec.comoving_distance_interp(cosmo)
    out = Matrix{eltype(data_cat_ra)}(undef, 3, size(data_cat_ra, 1))
    h::eltype(data_cat_ra) = cosmo.H₀ / 100
    @Threads.threads for i in eachindex(data_cat_ra)
        dist = r_fun(data_cat_red[i])
        ra = data_cat_ra[i] * π / 180
        dec = data_cat_dec[i] * π / 180
        
        #X
        out[1,i] = dist * cos(dec) * cos(ra) * h
        #Y
        out[2,i] = dist * cos(dec) * sin(ra) * h
        #Z
        out[3,i] = dist * sin(dec) * h
    end #for
    out
end #func

function cartesian_to_sky(data_x, data_y, data_z, cosmo)

    z_fun = BAOrec.redshift_interp(cosmo)
    out = Matrix{eltype(data_x)}(undef, 3, size(data_x, 1))
    data = (data_x, data_y, data_z)
    @Threads.threads for i in eachindex(data_x)

        r = sqrt(sum([data[j][i]^2 for j in 1:3])) / cosmo.h
        #@show r
        #@show view(cartesian, :,i)
        red = z_fun(r)

        s = sqrt(data_x[i]^2 + data_y[i]^2)
        lon = atan(data_y[i], data_x[i])
        lat = atan(data_z[i], s)

        lon = lon * 180 / π
        lat = lat * 180 / π

        lon = (lon - 360) % 360
        
        #RA
        out[1,i] = lon
        #DEC
        out[2,i] = lat
        #Z
        out[3,i] = red
    end #for
    out

end #func

fkp_weights(nz, P0) = 1 / (1 + nz * P0)


cosmo = BAOrec.Cosmology(z_tab_max = 10)


data_cat_pos = sky_to_cartesian(values(data_cat[!,:ra]), values(data_cat[!,:dec]), values(data_cat[!,:z]), cosmo)
rand_cat_pos = sky_to_cartesian(values(rand_cat[!,:ra]), values(rand_cat[!,:dec]), values(rand_cat[!,:z]), cosmo)
data_cat_pos = [data_cat_pos[i,:] for i in 1:3]
rand_cat_pos = [rand_cat_pos[i,:] for i in 1:3]
box_pad = 500f0
box_size, box_min = BAOrec.setup_box(rand_cat_pos..., box_pad)
P0 = 5f3
data_cat_w = values(data_cat[!,:w]) .* fkp_weights.(data_cat[!,:nz], Ref(P0))
rand_cat_w = values(rand_cat[!,:w]) .* fkp_weights.(rand_cat[!,:nz], Ref(P0))


println("CPU benchmarking...")

println("Struct")
@time recon = BAOrec.IterativeRecon(bias = 2.2f0, f = 0.757f0, 
                              smoothing_radius = 15f0, n_iter = 3,
                              box_size = box_size, 
                              box_min = box_min,
                              los = nothing)

println("Reconstruction iterative run")
grid_size = (512, 512, 512)
@btime BAOrec.run!($recon, $grid_size,
                    $data_cat_pos...,
                    $data_cat_w, 
                    $rand_cat_pos...,
                    $rand_cat_w);



println("Read positions from recon result")
@btime new_pos = BAOrec.reconstructed_positions($recon, $data_cat_pos...; field = :sum)
new_pos = BAOrec.reconstructed_positions(recon, data_cat_pos...; field = :sum)

CSV.write("test/bchmk_cpu_iter.csv", Tables.table(hcat(view(new_pos[1], 1:10), view(new_pos[2], 1:10), view(new_pos[3], 1:10))), writeheader=false)


println("Struct")
@time recon = BAOrec.MultigridRecon(bias = 2.2f0, f = 0.757f0, 
                              smoothing_radius = 15f0,
                              box_size = box_size, 
                              box_min = box_min,
                              los = nothing)

println("Reconstruction multigrid run")
@btime BAOrec.run!($recon, $grid_size,
                    $data_cat_pos...,
                    $data_cat_w, 
                    $rand_cat_pos...,
                    $rand_cat_w);



println("Read positions from recon result")
@btime new_pos = BAOrec.reconstructed_positions($recon, $data_cat_pos...; field = :sum)
new_pos = BAOrec.reconstructed_positions(recon, data_cat_pos...; field = :sum)


CSV.write("test/bchmk_cpu_mgrid.csv", Tables.table(hcat(view(new_pos[1], 1:10), view(new_pos[2], 1:10), view(new_pos[3], 1:10))), writeheader=false)
