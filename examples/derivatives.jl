using Revise
using Zygote
using BAOrec
using NPZ
using Plots
using CSV
using StaticArrays
using Statistics
using ForwardDiff
using AbstractFFTs
using FFTW


test_fn = "/home/users/d/dforeros/.julia/dev/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.dat.npy"

n_grid = 64
los = @SVector [0f0, 0f0, 1f0]
data = Matrix(npzread(test_fn)')
box_size = @SVector [2500f0, 2500f0, 2500f0]
box_min = @SVector [0f0, 0f0, 0f0]
data_w = ones(Float32, size(data,2))
recon = BAOrec.IterativeRecon(bias = 2.2f0, f = 0.757f0, 
                                smoothing_radius = 15f0, 
                                n_iter = 3,
                                box_size = box_size, 
                                box_min = box_min,
                                los = los)
rho = zeros(eltype(data), [n_grid for i in 1:3]...);                                
BAOrec.setup_fft!(recon, rho)
function recon_pos(data)  
    BAOrec.reconstructed_overdensity!(rho,
                                    recon,
                                    view(data, 1,:), view(data, 2,:), view(data, 3,:),
                                    data_w, );
    displacements = read_shifts(recon, view(data, 1,:), view(data, 2,:), view(data, 3,:), δ_r; field=:sum)
    displacements[1]

end #func


jacobian(recon_pos, data)



#AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = float.(ForwardDiff.value.(x) .+ 0im)
#AbstractFFTs.realfloat(x::AbstractArray{<:ForwardDiff.Dual}) = float.(ForwardDiff.value.(x))



