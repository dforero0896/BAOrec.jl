

function k_vec(dims::AbstractVector{Int}, box_size::SVector{3,T}) where T<:Real

    sample_rate = map(T, 2π .* dims ./ box_size)
    kx = rfftfreq(dims[1], sample_rate[1])
    ky = fftfreq(dims[2], sample_rate[2])
    kz = fftfreq(dims[3], sample_rate[3])
    (kx, ky, kz)
end #func

function x_vec(dims::AbstractVector{Int}, box_size::SVector{3,T}, box_min::SVector{3,T}) where T<:Real
    
    cell_size = map(T, box_size ./ dims)
    Tuple(collect(box_min[i] + 0.5 * cell_size[i]:cell_size[i]:box_min[i] + box_size[i]) for i in 1:3)
    #Tuple(collect(box_min[i] .+ cell_size[i] .* range(0, dims[i] - 1)) for i in 1:3)
end #func

x_vec(dims::AbstractVector{Int}, box_size::SVector{3,T}) where T<:Real = x_vec(dims::AbstractVector{Int}, box_size::SVector{3,T}, @SVector [T(0), T(0), T(0)])

function rho_to_delta!(ρ::Array{T, 3}) where T <: Real
    
    ρ_mean = mean(ρ)
    for I in CartesianIndices(ρ)
        ρ[I] /= ρ_mean
        ρ[I] -= 1.
    end #for
    ρ
end #func

function smooth!(field::Array{T, 3}, smoothing_radius::T, box_size::SVector{3,T}) where T <: Real

    plan = plan_rfft(field)
    field_k = plan * field
    k⃗ = k_vec([size(field)...], box_size)
    for I in CartesianIndices(field_k)
        k² = k⃗[1][I[1]]^2 + k⃗[2][I[2]]^2 + k⃗[3][I[3]]^2
        field_k[I] *= exp(-0.5 * smoothing_radius^2 * k²)
    end #for
    ldiv!(field, plan, field_k)
    field
end #func

function setup_box(pos_x, pos_y, pos_z, box_pad)

    data = (pos_x, pos_y, pos_z)
    box_min = SVector([minimum(d) for d in data]...) .- box_pad / 2
    box_max = SVector([maximum(d) for d in data]...) .+ box_pad / 2
    box_size = @SVector [maximum(box_max .- box_min) for _ in 1:3]

    box_size, box_min

end #func

