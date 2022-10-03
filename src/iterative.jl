



function iterate!(δ_r::AbstractArray{T,3}, δ_s::AbstractArray{T,3}, k⃗::Tuple{AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}, iter::Int, β::T;
                r̂ = nothing, x⃗ = nothing) where T <: Real
    println("Iteration ", iter, " \n")

    
    plan = plan_rfft(δ_r)
    δ_k = plan * δ_r # δ_k computed from the current real δ
    for I in CartesianIndices(δ_k)
        k² = k⃗[1][I[1]]^2 + k⃗[2][I[2]]^2 + k⃗[3][I[3]]^2
        k² = k² == 0 ? 1 : k²
        δ_k[I] /= k²
    end #for
    δ_k[1,1,1] = 0.
    ∂Ψj_∂xi_k = similar(δ_k)
    ∂Ψj_∂xi_x = similar(δ_r)
    δ_r .= δ_s
    if r̂ === nothing
        @assert x⃗ != nothing
        println("Using local line of sight r̂. \n")
        for i in 1:3
            for j in i:3
                    
                factor::T = (1. + T(i != j)) * β
                factor = iter == 1 ? factor / (1 + β) : factor
                
                for I in CartesianIndices(δ_k)
                    ∂Ψj_∂xi_k[I] = k⃗[i][I[i]] * k⃗[j][I[j]] * δ_k[I]
                end #for

                ldiv!(∂Ψj_∂xi_x, plan, ∂Ψj_∂xi_k) # Destroys ∂Ψj_∂ki_k

                for I in CartesianIndices(δ_r)
                    x² = x⃗[1][I[1]]^2 + x⃗[2][I[2]]^2 + x⃗[3][I[3]]^2
                    δ_r[I] = x² > 0 ? δ_r[I] -  factor * ∂Ψj_∂xi_x[I] * x⃗[i][I[i]] * x⃗[j][I[j]] / x² : 0
                end #for
    

            end #for
        end #for
    
    else #los provided

        for i in 1:3

            if r̂[i] == 0.
                continue
            end #if

            factor::T = β
            factor = iter === 1 ? factor / (1 + β) : factor

            for I in CartesianIndices(δ_k)
                ∂Ψj_∂xi_k[I] = k⃗[i][I[i]]^2 * r̂[i] * δ_k[I]
            end #for

            ldiv!(∂Ψj_∂xi_x, plan, ∂Ψj_∂xi_k) # Destroys ∂Ψj_∂ki_k

            for I in CartesianIndices(δ_r)
                δ_r[I] = δ_r[I] - factor * ∂Ψj_∂xi_x[I]
            end #for

        end #for

    end #if

    δ_r
end #func


function compute_displacements(δ_r::Array{T, 3}, data_x::AbstractVector{T}, data_y::AbstractVector{T}, data_z::AbstractVector{T}, box_size::SVector{3, T}, box_min::SVector{3, T}) where T <: Real

    k⃗ = k_vec([size(δ_r)...], box_size)
    plan = plan_rfft(δ_r)   
    δ_k = plan * δ_r
    
    Ψ_k = similar(δ_k)
    Ψ_r = similar(δ_r)
    Ψ_interp = Tuple(zeros(T, size(data_x)...) for _ in 1:3)
    for i in 1:3
        for I in CartesianIndices(δ_k)
            k² = k⃗[1][I[1]]^2 + k⃗[2][I[2]]^2 + k⃗[3][I[3]]^2
            k² = k² == 0 ? 1. : k²
            Ψ_k[I] = im * k⃗[i][I[i]] * δ_k[I] / k²
        end #for
        Ψ_k[1,1,1] = 0
        ldiv!(Ψ_r, plan, Ψ_k)
        read_cic!(Ψ_interp[i], Ψ_r, data_x, data_y, data_z, box_size, box_min)
    end #for
    Ψ_interp
end #func