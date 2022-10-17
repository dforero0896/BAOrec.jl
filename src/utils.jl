

function k_vec(field::AbstractArray, box_size::SVector{3,T}) where T<:Real
    dims = [size(field)...]
    sample_rate = map(T, 2π .* dims ./ box_size)
    kx = rfftfreq(dims[1], sample_rate[1])
    ky = fftfreq(dims[2], sample_rate[2])
    kz = fftfreq(dims[3], sample_rate[3])
    (kx, ky, kz)
end #func

function k_vec(field::PencilArray, box_size::SVector{3,T}) where T<:Real
    dims = [size_global(field)...]
    sample_rate = map(T, 2π .* dims ./ box_size)
    kx = rfftfreq(dims[1], sample_rate[1])
    ky = fftfreq(dims[2], sample_rate[2])
    kz = fftfreq(dims[3], sample_rate[3])
    (kx, ky, kz)
end #func

function x_vec(field::AbstractArray, box_size::SVector{3,T}, box_min::SVector{3,T}) where T<:Real
    dims = [size(field)...]
    cell_size = box_size ./ dims
    Tuple(map(T, collect(box_min[i] + 0.5 * cell_size[i]:cell_size[i]:box_min[i] + box_size[i])) for i in 1:3)
end #func


function x_vec(field::PencilArray, box_size::SVector{3,T}, box_min::SVector{3,T}) where T<:Real
    dims = [size_global(field)...]
    cell_size = map(T, box_size ./ dims)
    Tuple(map(T, collect(box_min[i] + 0.5 * cell_size[i]:cell_size[i]:box_min[i] + box_size[i])) for i in 1:3)
end #func


function rho_to_delta!(ρ::Array{T, 3}) where T <: Real
    
    ρ_mean = mean(ρ)
    for I in CartesianIndices(ρ)
        ρ[I] /= ρ_mean
        ρ[I] -= 1.
    end #for
    ρ
end #func

function smooth!(field::AbstractArray{T, 3}, smoothing_radius::T, box_size::SVector{3,T}, fft_plan) where T <: Real

    
    field_k = fft_plan * field
    k⃗ = k_vec(field, box_size)
    @inbounds Threads.@threads for I in CartesianIndices(field_k)
        k² = k⃗[1][I[1]]^2 + k⃗[2][I[2]]^2 + k⃗[3][I[3]]^2
        field_k[I] *= exp(-0.5 * smoothing_radius^2 * k²)
    end #for
    ldiv!(field, fft_plan, field_k)
    field
end #func


function smooth!(field::PencilArray{T, 3}, smoothing_radius::T, box_size::SVector{3,T}, fft_plan) where T <: Real
    
    field_k = fft_plan * field
    field_global_k = global_view(field_k)
    k⃗ = k_vec(field, box_size)
    for I in CartesianIndices(field_global_k)
        k² = k⃗[1][I[1]]^2 + k⃗[2][I[2]]^2 + k⃗[3][I[3]]^2
        field_global_k[I] *= exp(-0.5 * smoothing_radius^2 * k²)
    end #for
    ldiv!(field, fft_plan, field_k)
    field
end #func


@kernel function laplace_kernel!(out_field, @Const(k⃗), @Const(field))
    I = @index(Global, CartesianIndex)
    out_field[I] = field[I] * k⃗[I[1]]^2 + k⃗[I[2]]^2 + k⃗[I[3]]^2
end #function

@kernel function gaussian_filter_kernel!(field, k1, k2, k3, smoothing_radius)
    I = @index(Global, Cartesian)
    k² = k1[I[1]]^2 + k2[I[2]]^2 + k3[I[3]]^2
    field[I] *= exp(-0.5 *  smoothing_radius^2 * k²)
end #function


function smooth!(field::CuArray{T, 3}, smoothing_radius::T, box_size::SVector{3,T}, fft_plan) where T <: Real

    
    field_k = fft_plan * field
    k⃗ = map(CuArray, k_vec(field, box_size))
    device = KernelAbstractions.get_device(field_k)
    kernel! = gaussian_filter_kernel!(device)
    ev = kernel!(field_k, k⃗..., smoothing_radius, ndrange = size(field_k))
    wait(ev)
    ldiv!(field, fft_plan, field_k)
    field
end #func



function setup_box(pos_x, pos_y, pos_z, box_pad)

    data = (pos_x, pos_y, pos_z)
    box_min = SVector([minimum(d) for d in data]...) .- box_pad / 2
    box_max = SVector([maximum(d) for d in data]...) .+ box_pad / 2
    box_size = @SVector [maximum(box_max .- box_min) for _ in 1:3]

    box_size, box_min

end #func

