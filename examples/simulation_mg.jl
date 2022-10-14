using Revise
using BAOrec
using NPZ
using Plots
using CSV
using StaticArrays
using Statistics
using Zygote

test_fn = "/home/astro/dforero/codes/BAOrec//data/CATALPTCICz0.466G960S1010008301_zspace.dat.npy"

n_grid = 512
los = @SVector [0f0, 0f0, 1f0]
data = Matrix(npzread(test_fn)')
box_size = @SVector [2500f0, 2500f0, 2500f0]
box_min = @SVector [0f0, 0f0, 0f0]
data_w = ones(Float32, size(data,2))
rho = zeros(Float32, [n_grid for i in 1:3]...);


recon = BAOrec.MultigridRecon(bias = 2.2f0, f = 0.757f0, 
                              smoothing_radius = 15f0,
                              box_size = box_size, 
                              box_min = box_min,
                              los = los)
@time BAOrec.setup_fft!(recon, rho)                              
@time BAOrec.setup_overdensity!(rho,
                        recon,
                        view(data, 1,:), view(data, 2,:), view(data, 3,:), data_w,
                        );

p1 = heatmap(dropdims(mean(rho, dims=1), dims=1), aspect_ratio = :equal)




#exit()

ϕ = zero(rho)
@time BAOrec.fmg(rho, ϕ, recon.box_size, recon.box_min, recon.β, recon.jacobi_damping_factor, recon.jacobi_niterations, recon.vcycle_niterations, los = recon.los)
p2 = heatmap(dropdims(mean(ϕ, dims=1), dims=1), aspect_ratio = :equal)
plot(p1, p2)
savefig("/home/astro/dforero/codes/BAOrec/examples/simulation_mg.png")



function phi_fun(x, y, z)
    result = BAOrec.read_cic_single(ϕ, size(ϕ), x, y, z, recon.box_size, recon.box_min)
    result
end


@show phi_fun(data[1,10], data[2,10], data[3,10])
@show gradient(phi_fun, data[1,10], data[2,10], data[3,10])


@time new_pos = BAOrec.reconstructed_positions(recon, view(data, 1,:), view(data, 2,:), view(data, 3,:), ϕ; field = :sum);
@time rx, ry, rz = box_size[1] * rand(Float32, 10 .* size(data,2)), box_size[2] * rand(Float32, 10 .* size(data,2)), box_size[3] * rand(Float32, 10 .* size(data,2));
@time new_rand_sym = BAOrec.reconstructed_positions(recon, rx, ry, rz, ϕ; field = :sum);
@time new_rand_iso = BAOrec.reconstructed_positions(recon, rx, ry, rz, ϕ; field = :disp);

@show [new_pos[i][j] for j = 1:4, i=1:3]
#[(new_pos[i])[j] for j = 1:4, i = 1:3] = Float32[2494.5574 3.2766175 43.74556; 2496.5994 4.640405 44.582363; 2495.9429 12.93197 76.37023; 3.9973264 6.7710137 380.89532]
npzwrite("/home/astro/dforero/codes/BAOrec//data/MG_CPU_CATALPTCICz0.466G960S1010008301_zspace.dat.rec.npy", hcat(new_pos...))
npzwrite("/home/astro/dforero/codes/BAOrec//data/MG_CPU_CATALPTCICz0.466G960S1010008301_zspace.ran.rec.sym.npy", hcat(new_rand_sym...))
npzwrite("/home/astro/dforero/codes/BAOrec//data/MG_CPU_CATALPTCICz0.466G960S1010008301_zspace.ran.rec.iso.npy", hcat(new_rand_iso...))