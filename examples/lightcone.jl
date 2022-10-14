using Revise
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
FFTW.set_num_threads(128)

const cosmo = BAOrec.Cosmology(z_tab_max = 10)
const box_pad = 500f0
const P0 = 5f3


data_cat_fn = "/global/cfs/projectdirs/desi/mocks/UNIT/HOD_Shadab/multiple_snapshot_lightcone/UNIT_lightcone_multibox_ELG_footprint_nz_NGC.dat"
rand_cat_fn = "/global/cfs/projectdirs/desi/mocks/UNIT/HOD_Shadab/multiple_snapshot_lightcone/UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.dat"

data_cat = DataFrame(CSV.File(data_cat_fn, delim=" ", header=[:ra, :dec, :d, :z, :nz], types = [Float32 for _ in 1:5]))
rand_cat = DataFrame(CSV.File(rand_cat_fn, delim=" ", header=[:ra, :dec, :d, :z, :nz], types = [Float32 for _ in 1:5]))

data_cat = data_cat[map(z -> ((z > 0.8) & (z < 1)), data_cat.z), :]
rand_cat = rand_cat[map(z -> ((z > 0.8) & (z < 1)), rand_cat.z), :]

println("Read and masked data")

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

println("Coordinate conversion")
data_cat_pos = sky_to_cartesian(values(data_cat[!,:ra]), values(data_cat[!,:dec]), values(data_cat[!,:z]), cosmo)
rand_cat_pos = sky_to_cartesian(values(rand_cat[!,:ra]), values(rand_cat[!,:dec]), values(rand_cat[!,:z]), cosmo)
data_cat_pos = [data_cat_pos[i,:] for i in 1:3]
rand_cat_pos = [rand_cat_pos[i,:] for i in 1:3]
println("Weights")
data_cat_w = fkp_weights.(data_cat[!,:nz], Ref(P0))
rand_cat_w = fkp_weights.(rand_cat[!,:nz], Ref(P0))
println("Struct")
recon = BAOrec.IterativeRecon(bias = 2.2f0, f = 0.757f0, 
                              smoothing_radius = 15f0, n_iter = 3,
                              los = nothing)
grid_size = (512, 512, 512)
println("Run")

@time BAOrec.run!(recon, grid_size,
                    data_cat_pos...,
                    data_cat_w, 
                    rand_cat_pos...,
                    rand_cat_w);
#displacements = BAOrec.read_shifts(recon, data_cat_pos..., recon.result_cache, field = :disp)
#@show [d[10] for d in displacements]
#displacements = BAOrec.read_shifts(recon, data_cat_pos..., recon.result_cache, field = :rsd)
#@show [d[10] for d in displacements]
println("Reading new positions")
@time new_pos = BAOrec.reconstructed_positions(recon, data_cat_pos...; field = :sum)
@time new_rand_cat_sym = BAOrec.reconstructed_positions(recon, rand_cat_pos...; field = :sum)
@time new_rand_cat_iso = BAOrec.reconstructed_positions(recon, rand_cat_pos...; field = :disp)



println("Coordinate conversion")
new_pos = cartesian_to_sky(new_pos..., cosmo)
new_rand_cat_sym = cartesian_to_sky(new_rand_cat_sym..., cosmo)
new_rand_cat_iso = cartesian_to_sky(new_rand_cat_iso..., cosmo)

@show [new_pos[i,10] for i in 1:3]
@show [new_rand_cat_sym[i,10] for i in 1:3]
@show [new_rand_cat_iso[i,10] for i in 1:3]

println("Save results")
npzwrite("data/CPU_UNIT_lightcone_multibox_ELG_footprint_nz_NGC.rec.npy", hcat(new_pos', values(data_cat[!,:nz])))
npzwrite("data/CPU_UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.rec.sym.npy", hcat(new_rand_cat_sym', values(rand_cat[!,:nz])))
npzwrite("data/CPU_UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.rec.iso.npy", hcat(new_rand_cat_iso', values(rand_cat[!,:nz])))

new_pos = nothing
new_rand_cat_sym = nothing
new_rand_cat_iso = nothing
println("Now multigrid")
recon = BAOrec.MultigridRecon(bias = 2.2f0, f = 0.757f0, 
                              smoothing_radius = 15f0,
                              los = nothing)
grid_size = (512, 512, 512)
@time BAOrec.run!(recon, grid_size,
                    data_cat_pos...,
                    data_cat_w, 
                    rand_cat_pos...,
                    rand_cat_w);

@time new_pos = BAOrec.reconstructed_positions(recon, data_cat_pos...; field = :sum)
@time new_rand_cat_sym = BAOrec.reconstructed_positions(recon, rand_cat_pos...; field = :sum)
@time new_rand_cat_iso = BAOrec.reconstructed_positions(recon, rand_cat_pos...; field = :disp)


new_pos = cartesian_to_sky(new_pos..., cosmo)
new_rand_cat_sym = cartesian_to_sky(new_rand_cat_sym..., cosmo)
new_rand_cat_iso = cartesian_to_sky(new_rand_cat_iso..., cosmo)

@show [new_pos[i,10] for i in 1:3]
@show [new_rand_cat_sym[i,10] for i in 1:3]
@show [new_rand_cat_iso[i,10] for i in 1:3]
println("Save results")
npzwrite("data/MG_CPU_UNIT_lightcone_multibox_ELG_footprint_nz_NGC.rec.npy", hcat(new_pos', values(data_cat[!,:nz])))
npzwrite("data/MG_CPU_UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.rec.sym.npy", hcat(new_rand_cat_sym', values(rand_cat[!,:nz])))
npzwrite("data/MG_CPU_UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.rec.iso.npy", hcat(new_rand_cat_iso', values(rand_cat[!,:nz])))