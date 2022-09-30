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

n_grid = 512
los = @SVector [0f0, 0f0, 1f0]
data = Matrix(npzread(test_fn)')
box_size = @SVector [2500f0, 2500f0, 2500f0]
box_min = @SVector [0f0, 0f0, 0f0]
data_w = ones(Float32, size(data,2))

function recon_smoothing(smoothing_radius)

    rho = zeros(ForwardDiff.Dual, [n_grid for i in 1:3]...);
    recon = BAOrec.IterativeRecon(bias = 2.2f0, f = 0.757f0, 
                                smoothing_radius = smoothing_radius, 
                                n_iter = 3,
                                box_size = box_size, 
                                box_min = box_min,
                                los = los)
    BAOrec.reconstructed_overdensity!(rho,
                                    recon,
                                    view(data, 1,:), view(data, 2,:), view(data, 3,:),
                                    data_w, );
    displacements = read_shifts(recon, data_x, data_y, data_z, δ_r; field=field)
    displacements[1]

end #func


jacobian(recon_smoothing, 15f0)

ForwardDiff.derivative(recon_smoothing, 15f0)

AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = float.(ForwardDiff.value.(x) .+ 0im)
AbstractFFTs.realfloat(x::AbstractArray{<:ForwardDiff.Dual}) = float.(ForwardDiff.value.(x))

function grad_smoothing(smoothing_radius)
    field = zeros(typeof(smoothing_radius), [n_grid for i in 1:3]...);
    BAOrec.smooth!(field, smoothing_radius, box_size)
end #func

#jacobian(grad_smoothing, 15f0)
ForwardDiff.derivative(grad_smoothing, 15f0)