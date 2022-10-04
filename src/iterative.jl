



function iterate!(δ_r::AbstractArray{T,3}, δ_s::AbstractArray{T,3}, k⃗::Tuple{AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}, iter::Int, β::T, fft_plan;
                r̂ = nothing, x⃗ = nothing) where T <: Real
    println("Iteration ", iter, " \n")

    
    
    δ_k = fft_plan * δ_r # δ_k computed from the current real δ
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

                ldiv!(∂Ψj_∂xi_x, fft_plan, ∂Ψj_∂xi_k) # Destroys ∂Ψj_∂ki_k

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

            ldiv!(∂Ψj_∂xi_x, fft_plan, ∂Ψj_∂xi_k) # Destroys ∂Ψj_∂ki_k

            for I in CartesianIndices(δ_r)
                δ_r[I] = δ_r[I] - factor * ∂Ψj_∂xi_x[I]
            end #for

        end #for

    end #if

    δ_r
end #func

@kernel function hessian_kernel!(out_field, @Const(k⃗), field, i, j)
    I = @index(Global, Cartesian)
    out_field[I] = field[I] * k⃗[i][I[i]] * k⃗[j][I[j]]
end #function

@kernel function inv_laplace_kernel!(field, k⃗)
    I = @index(Global, Cartesian)    
    k² = (k⃗[1][I[1]]^2 + k⃗[2][I[2]]^2 + k⃗[3][I[3]]^2)
    field[I] = k² > 0 ? field[I] / k² : 0
end #function

function iterate!(δ_r::CuArray{T,3}, δ_s::CuArray{T,3}, k⃗::Tuple{AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}, iter::Int, β::T, fft_plan;
    r̂ = nothing, x⃗ = nothing) where T <: Real
    println("Iteration ", iter)
    k⃗ = map(CuArray, k⃗)
    δ_k = fft_plan * δ_r # δ_k computed from the current real δ
    device = KernelAbstractions.get_device(δ_k)
    inv_lap! = inv_laplace_kernel!(device, 256)
    hessian! = hessian_kernel!(device, 256)
    ev = inv_lap!(δ_k, k⃗, ndrange = size(δ_k))
    wait(ev)
    #δ_k[1,1,1] = 0.
    ∂Ψj_∂xi_k = similar(δ_k)
    ∂Ψj_∂xi_x = similar(δ_r)
    δ_r .= δ_s
    
    if r̂ === nothing
        @assert x⃗ != nothing
        println("Using local line of sight r̂. \n")
        #x² = typeof(δ_r)([x⃗[1][i]^2 + x⃗[2][j]^2 + x⃗[3][k]^2 for i in eachindex(x⃗[1]), j in eachindex(x⃗[2]), k in eachindex(x⃗[3])])
        x⃗ = map(CuArray, x⃗)
        for i in 1:3
            for j in i:3
                    
                factor::T = (1. + T(i != j)) * β
                factor = iter == 1 ? factor / (1 + β) : factor
                
                
                #@. ∂Ψj_∂xi_k = k⃗[i] * k⃗[j] * δ_k
                hessian!(∂Ψj_∂xi_k, k⃗, δ_k, i, j, ndrange = size(δ_k))
                ldiv!(∂Ψj_∂xi_x, fft_plan, ∂Ψj_∂xi_k) # Destroys ∂Ψj_∂ki_k
                hessian!(∂Ψj_∂xi_x, x⃗, ∂Ψj_∂xi_x, i, j, ndrange = size(δ_r)) #mult by xi xj
                inv_lap!(∂Ψj_∂xi_x, x⃗, ndrange = size(δ_r)) # divide by x^2
                @. δ_r = δ_r -  factor * ∂Ψj_∂xi_x
                #@. δ_r = ifelse(x² > 0, δ_r -  factor * ∂Ψj_∂xi_x * x⃗[i] * x⃗[j] / x², 0)

            end #for
        end #for

    else #los provided

        for i in 1:3

            

            factor::T = β
            factor = iter == 1 ? factor / (1 + β) : factor
            
            #@. ∂Ψj_∂xi_k = k⃗[i]^2 * r̂[i] * δ_k
            hessian!(∂Ψj_∂xi_k, k⃗, δ_k, i, i, ndrange = size(δ_k))
            @. ∂Ψj_∂xi_k *= r̂[i]            

            ldiv!(∂Ψj_∂xi_x, fft_plan, ∂Ψj_∂xi_k) # Destroys ∂Ψj_∂ki_k

            @. δ_r = δ_r - factor * ∂Ψj_∂xi_x


        end #for

    end #if

    δ_r
end #func




function compute_displacements(δ_r::Array{T, 3}, data_x::AbstractVector{T}, data_y::AbstractVector{T}, data_z::AbstractVector{T}, box_size::SVector{3, T}, box_min::SVector{3, T}, fft_plan) where T <: Real

    k⃗ = k_vec([size(δ_r)...], box_size)
    
    δ_k = fft_plan * δ_r
    
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
        ldiv!(Ψ_r, fft_plan, Ψ_k)
        read_cic!(Ψ_interp[i], Ψ_r, data_x, data_y, data_z, box_size, box_min)
    end #for
    Ψ_interp
end #func