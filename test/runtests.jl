using Revise
using BAOrec
using Test
using StaticArrays
using Plots
using Statistics
using FFTW

@testset "BAOrec.jl" begin
    # Write your tests here.
    

end
n_grid = 64
n_obj = 100000
box_size = @SVector [2000f0, 2000f0, 2000f0]
box_min = @SVector [0f0, 0f0, 0f0]
data_w = ones(Float32, n_obj)
data = box_size .* rand(Float32, 3, n_obj);
rho = zeros(Float32, [n_grid for i in 1:3]...);

BAOrec.cic!(rho, view(data, 1,:), view(data, 2,:), view(data, 3,:), data_w, box_size, box_min)
heatmap(dropdims(mean(rho, dims=1), dims=1), aspect=1)

kvec = BAOrec.k_vec([size(rho)...], box_size);

xvec = BAOrec.x_vec([size(rho)...], box_size);


recon = BAOrec.IterativeRecon(bias = 2f0, f = 0.8f0)

Ψ = BAOrec.reconstruction_displacements!(recon,
                                        rho,
                                        view(data, 1,:), view(data, 2,:), view(data, 3,:),
                                        data_w, 
                                        box_size, 
                                        box_min;
                                        los = [0f0, 0f0, 1f0])

new_pos = BAOrec.reconstructed_positions(recon, view(data, 1,:), view(data, 2,:), view(data, 3,:), Ψ; field = :sum)