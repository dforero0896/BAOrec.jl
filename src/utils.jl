

function k_vec(dims, box_size)
    T = eltype(box_size)
    sample_rate = map(T, 2π .* dims ./ box_size)
    kx = rfftfreq(dims[1], sample_rate[1])
    ky = fftfreq(dims[2], sample_rate[2])
    kz = fftfreq(dims[3], sample_rate[3])
    (kx, ky, kz)
end #func

function x_vec(dims, box_size) 
    T = eltype(box_size)
    cell_size = map(T, box_size ./ dims)
    Tuple(collect(0.5 * cell_size[i]:cell_size[i]:box_size[i]) for i in 1:3)
end #func

function rho_to_delta!(ρ) 
    
    ρ_mean = mean(ρ)
    @Threads.threads for I in CartesianIndices(ρ)
        ρ[I] /= ρ_mean
        ρ[I] -= 1.
    end #for
    ρ
end #func

function smooth!(field, smoothing_radius, box_size)

    plan = plan_rfft(field)
    field_k = plan * field
    k⃗ = k_vec([size(field)...], box_size)
    @Threads.threads for I in CartesianIndices(field_k)
        k² = k⃗[1][I[1]]^2 + k⃗[2][I[2]]^2 + k⃗[3][I[3]]^2
        k² = k² == 0 ? 1. : k²
        field_k[I] *= exp(-0.5 * smoothing_radius^2 * k²)
    end #for
    ldiv!(field, plan, field_k)
    field
end #func



