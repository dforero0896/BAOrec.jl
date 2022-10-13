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
using CUDA

const cosmo = BAOrec.Cosmology(z_tab_max = 10)
const box_pad = 500f0
const P0 = 5f3


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




println("Coordinate conversion")
data_cat_pos = sky_to_cartesian(values(data_cat[!,:ra]), values(data_cat[!,:dec]), values(data_cat[!,:z]), cosmo)
rand_cat_pos = sky_to_cartesian(values(rand_cat[!,:ra]), values(rand_cat[!,:dec]), values(rand_cat[!,:z]), cosmo)
data_cat_pos = map(cu, [data_cat_pos[i,:] for i in 1:3])
rand_cat_pos = map(cu, [rand_cat_pos[i,:] for i in 1:3])
println("Weights")
data_cat_w = cu(values(data_cat[!,:w]) .* fkp_weights.(data_cat[!,:nz], Ref(P0)))
rand_cat_w = cu(values(rand_cat[!,:w]) .* fkp_weights.(rand_cat[!,:nz], Ref(P0)))
println("Struct")
recon = BAOrec.IterativeRecon(bias = 2.2f0, f = 0.757f0, 
                              smoothing_radius = 15f0, n_iter = 3,
                              los = nothing)
grid_size = (512, 512, 512)
println("Run")
BAOrec.run!(recon, grid_size,
                    data_cat_pos...,
                    data_cat_w, 
                    rand_cat_pos...,
                    rand_cat_w);
println("Reading new positions")
new_pos = BAOrec.reconstructed_positions(recon, data_cat_pos...; field = :sum)
new_rand_cat_sym = BAOrec.reconstructed_positions(recon, rand_cat_pos...; field = :sum)
new_rand_cat_iso = BAOrec.reconstructed_positions(recon, rand_cat_pos...; field = :disp)
new_pos = map(Array, new_pos)
new_rand_cat_sym = map(Array, new_rand_cat_sym)
new_rand_cat_iso = map(Array, new_rand_cat_iso)

println("Coordinate conversion")
new_pos = cartesian_to_sky(new_pos..., cosmo)
new_rand_cat_sym = cartesian_to_sky(new_rand_cat_sym..., cosmo)
new_rand_cat_iso = cartesian_to_sky(new_rand_cat_iso..., cosmo)

println("Save results")
npzwrite("data/GPU_UNIT_lightcone_multibox_ELG_footprint_nz_NGC.rec.npy", hcat(new_pos', values(data_cat[!,:w]), values(data_cat[!,:nz])))
npzwrite("data/GPU_UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.rec.sym.npy", hcat(new_rand_cat_sym', values(rand_cat[!,:w]), values(rand_cat[!,:nz])))
npzwrite("data/GPU_UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.rec.iso.npy", hcat(new_rand_cat_iso', values(rand_cat[!,:w]), values(rand_cat[!,:nz])))

new_pos = nothing
new_rand_cat_sym = nothing
new_rand_cat_iso = nothing

recon = BAOrec.MultigridRecon(bias = 2.2f0, f = 0.757f0, 
                              smoothing_radius = 15f0,
                              los = nothing)
grid_size = (512, 512, 512)
BAOrec.run!(recon, grid_size,
                    data_cat_pos...,
                    data_cat_w, 
                    rand_cat_pos...,
                    rand_cat_w);

new_pos = BAOrec.reconstructed_positions(recon, data_cat_pos...; field = :sum)
new_rand_cat_sym = BAOrec.reconstructed_positions(recon, rand_cat_pos...; field = :sum)
new_rand_cat_iso = BAOrec.reconstructed_positions(recon, rand_cat_pos...; field = :disp)
new_pos = map(Array, new_pos)
new_rand_cat_sym = map(Array, new_rand_cat_sym)
new_rand_cat_iso = map(Array, new_rand_cat_iso)

new_pos = cartesian_to_sky(new_pos..., cosmo)
new_rand_cat_sym = cartesian_to_sky(new_rand_cat_sym..., cosmo)
new_rand_cat_iso = cartesian_to_sky(new_rand_cat_iso..., cosmo)


npzwrite("data/MG_GPU_UNIT_lightcone_multibox_ELG_footprint_nz_NGC.rec.npy", hcat(new_pos', values(data_cat[!,:w]), values(data_cat[!,:nz])))
npzwrite("data/MG_GPU_UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.rec.sym.npy", hcat(new_rand_cat_sym', values(rand_cat[!,:w]), values(rand_cat[!,:nz])))
npzwrite("data/MG_GPU_UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.rec.iso.npy", hcat(new_rand_cat_iso', values(rand_cat[!,:w]), values(rand_cat[!,:nz])))