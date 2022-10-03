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
data_cat_fn = "/home/users/d/dforeros/.julia/dev/BAOrec/data/Patchy-Mocks-DR12NGC-COMPSAM_V6C_0001.dat"
rand_cat_fn = "/home/users/d/dforeros/.julia/dev/BAOrec/data/Patchy-Mocks-Randoms-DR12NGC-COMPSAM_V6C_x20.dat"

data_cat = DataFrame(CSV.File(data_cat_fn, delim=" ", header=[:ra, :dec, :z, :w, :nz], types = [Float32 for _ in 1:5]))
rand_cat = DataFrame(CSV.File(rand_cat_fn, delim=" ", header=[:ra, :dec, :z, :w, :nz], types = [Float32 for _ in 1:5]))

data_cat = data_cat[map(z -> ((z > 0.2) & (z < 0.5)), data_cat.z), :]
rand_cat = rand_cat[map(z -> ((z > 0.2) & (z < 0.5)), rand_cat.z), :]

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

function cartesian_to_sky(cartesian, cosmo)

    z_fun = BAOrec.redshift_interp(cosmo)
    out = Matrix{eltype(cartesian)}(undef, size(cartesian)...)
    
    @Threads.threads for i in eachindex(view(cartesian, 1, :))

        r = sqrt(sum(view(cartesian, :,i) .^ 2)) / cosmo.h
        #@show r
        #@show view(cartesian, :,i)
        red = z_fun(r)

        s = sqrt(cartesian[1, i]^2 + cartesian[2,i]^2)
        lon = atan(cartesian[2,i], cartesian[1,i])
        lat = atan(cartesian[3,i], s)

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
BAOrec.comoving_distance(cosmo, 0.5) * cosmo.h

data_cat_pos = sky_to_cartesian(values(data_cat[!,:ra]), values(data_cat[!,:dec]), values(data_cat[!,:z]), cosmo)
rand_cat_pos = sky_to_cartesian(values(rand_cat[!,:ra]), values(rand_cat[!,:dec]), values(rand_cat[!,:z]), cosmo)
box_pad = 500f0
@show box_min = SVector(minimum(rand_cat_pos, dims=2)...) .- box_pad / 2
@show SVector(minimum(rand_cat_pos, dims=2)...)
@show box_max = SVector(maximum(rand_cat_pos, dims=2)...) .+ box_pad / 2
@show SVector(maximum(rand_cat_pos, dims=2)...)
@show box_size = @SVector [maximum(box_max .- box_min) for _ in 1:3]
@show box_min .+ box_size

P0 = 5f3
data_cat_w = values(data_cat[!,:w]) .* fkp_weights.(data_cat[!,:nz], Ref(P0))
rand_cat_w = values(rand_cat[!,:w]) .* fkp_weights.(rand_cat[!,:nz], Ref(P0))

n_grid = 512
rho = zeros(eltype(data_cat_pos), [n_grid for i in 1:3]...);


recon = BAOrec.IterativeRecon(bias = 2.2f0, f = 0.757f0, 
                              smoothing_radius = 15f0, n_iter = 3,
                              box_size = box_size, 
                              box_min = box_min,
                              los = nothing)
fill!(rho, 0);
@time BAOrec.setup_overdensity!(rho,
                              recon,
                              view(data_cat_pos, 1,:), view(data_cat_pos, 2,:), view(data_cat_pos, 3,:), data_cat_w,
                              view(rand_cat_pos, 1,:), view(rand_cat_pos, 2,:), view(rand_cat_pos, 3,:), rand_cat_w
                              );
p3 = heatmap(dropdims(mean(rho, dims=3), dims=3), aspect_ratio = :equal, c = :vik,)

fill!(rho, 0);
@time BAOrec.reconstructed_overdensity!(rho,
                              recon,
                              view(data_cat_pos, 1,:), view(data_cat_pos, 2,:), view(data_cat_pos, 3,:),
                              data_cat_w, 
                              view(rand_cat_pos, 1,:), view(rand_cat_pos, 2,:), view(rand_cat_pos, 3,:),
                              rand_cat_w);
p4 = heatmap(dropdims(mean(rho, dims=3), dims=3), aspect_ratio = :equal, c = :vik, layout = 2)




@time new_pos = BAOrec.reconstructed_positions(recon, view(data_cat_pos, 1,:), view(data_cat_pos, 2,:), view(data_cat_pos, 3,:), rho; field = :sum);
@time new_rand_cat_sym = BAOrec.reconstructed_positions(recon, view(rand_cat_pos, 1,:), view(rand_cat_pos, 2,:), view(rand_cat_pos, 3,:), rho; field = :sum);
@time new_rand_cat_iso = BAOrec.reconstructed_positions(recon, view(rand_cat_pos, 1,:), view(rand_cat_pos, 2,:), view(rand_cat_pos, 3,:), rho; field = :disp);


@time new_pos = cartesian_to_sky(hcat(new_pos...)', cosmo)
@time new_rand_cat_sym = cartesian_to_sky(hcat(new_rand_cat_sym...)', cosmo)
@time new_rand_cat_iso = cartesian_to_sky(hcat(new_rand_cat_iso...)', cosmo)



npzwrite("/home/users/d/dforeros/.julia/dev/BAOrec/data/Patchy-Mocks-DR12NGC-COMPSAM_V6C_0001.dat.rec.npy", hcat(new_pos', values(data_cat[!,:w]), values(data_cat[!,:nz])))
npzwrite("/home/users/d/dforeros/.julia/dev/BAOrec/data/Patchy-Mocks-Randoms-DR12NGC-COMPSAM_V6C_x20.dat.rec.sym.npy", hcat(new_rand_cat_sym', values(rand_cat[!,:w]), values(rand_cat[!,:nz])))
npzwrite("/home/users/d/dforeros/.julia/dev/BAOrec/data/Patchy-Mocks-Randoms-DR12NGC-COMPSAM_V6C_x20.dat.rec.iso.npy", hcat(new_rand_cat_iso', values(rand_cat[!,:w]), values(rand_cat[!,:nz])))