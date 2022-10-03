abstract type AbstractRecon end
@with_kw mutable struct IterativeRecon <: AbstractRecon

    bias
    f
    smoothing_radius
    box_size::SVector{3}
    box_min::SVector{3}
    los = nothing
    n_iter::Int = 3
    β = f / bias
    fft_plan = nothing
    

end #struct

function setup_fft!(recon::IterativeRecon, field::AbstractArray)
    recon.fft_plan = plan_rfft(field)
end #func
    
function setup_overdensity!(δ_r::AbstractArray{T, 3},
                            recon::IterativeRecon,
                            data_x::AbstractVector{T}, 
                            data_y::AbstractVector{T}, 
                            data_z::AbstractVector{T}, 
                            data_w::AbstractVector{T},
                            wrap = true
                            ) where T <: Real


    cic!(δ_r, data_x, data_y, data_z, data_w, recon.box_size, recon.box_min, wrap = wrap)
    smooth!(δ_r, recon.smoothing_radius, recon.box_size, recon.fft_plan)
    δ_r_mean = mean(δ_r)
    for I in CartesianIndices(δ_r)
        δ_r[I] /= δ_r_mean
        δ_r[I] -= 1.
        δ_r[I] /= recon.bias
    end #for

    δ_r
end


function setup_overdensity!(δ_r_dat::AbstractArray{T, 3},
                            recon::IterativeRecon,
                            data_x::AbstractVector{T}, 
                            data_y::AbstractVector{T}, 
                            data_z::AbstractVector{T}, 
                            data_w::AbstractVector{T}, 
                            rand_x::AbstractVector{T}, 
                            rand_y::AbstractVector{T}, 
                            rand_z::AbstractVector{T}, 
                            rand_w::AbstractVector{T},
                            ran_min = 0.01
                            ) where T <: Real

    δ_r_ran = zero(δ_r_dat)
    cic!(δ_r_dat, data_x, data_y, data_z, data_w, recon.box_size, recon.box_min, wrap = false)
    cic!(δ_r_ran, rand_x, rand_y, rand_z, rand_w, recon.box_size, recon.box_min, wrap = false)
    smooth!(δ_r_dat, recon.smoothing_radius, recon.box_size, recon.fft_plan)
    smooth!(δ_r_ran, recon.smoothing_radius, recon.box_size, recon.fft_plan)
    δ_r_dat_sum = sum(δ_r_dat)
    δ_r_ran_sum = sum(δ_r_ran)
    α = δ_r_dat_sum / δ_r_ran_sum
    threshold = ran_min * δ_r_ran_sum / size(rand_x, 1) #should this be sum(rand_w)?
    #@. δ_r_dat -= α * δ_r_ran
    for I in CartesianIndices(δ_r_dat)
        δ_r_dat[I] -= α * δ_r_ran[I]
        δ_r_dat[I] = δ_r_ran[I] > threshold ? δ_r_dat[I] / (recon.bias * α * δ_r_ran[I]) : 0
    end #for
    δ_r_dat
end

function reconstructed_overdensity!(δ_r::AbstractArray{T, 3},
                                    recon::IterativeRecon,
                                    data_x::AbstractVector{T}, 
                                    data_y::AbstractVector{T}, 
                                    data_z::AbstractVector{T}, 
                                    data_w::AbstractVector{T}) where T <: Real
    los = recon.los
    setup_overdensity!(δ_r, recon, data_x, data_y, data_z, data_w)
    
    δ_s = copy(δ_r)
    kvec = k_vec([size(δ_r)...], recon.box_size)
    xvec = los === nothing ? x_vec([size(δ_r)...], recon.box_size) : nothing
    for niter in 1:recon.n_iter
        iterate!(δ_r, δ_s, kvec, niter, recon.β, recon.fft_plan; r̂ = los, x⃗ = xvec)
    end #for
    δ_r
end #func

function reconstructed_overdensity!(δ_r::AbstractArray{T, 3},
                            recon::IterativeRecon,
                            data_x::AbstractVector{T}, 
                            data_y::AbstractVector{T}, 
                            data_z::AbstractVector{T}, 
                            data_w::AbstractVector{T},
                            rand_x::AbstractVector{T}, 
                            rand_y::AbstractVector{T}, 
                            rand_z::AbstractVector{T}, 
                            rand_w::AbstractVector{T};) where T <: Real
    los = recon.los
    setup_overdensity!(δ_r, recon, data_x, data_y, data_z, data_w, rand_x, rand_y, rand_z, rand_w)

    δ_s = copy(δ_r)
    kvec = k_vec([size(δ_r)...], recon.box_size)
    xvec = los === nothing ? x_vec([size(δ_r)...], recon.box_size, recon.box_min) : nothing

    for niter in 1:recon.n_iter
        iterate!(δ_r, δ_s, kvec, niter, recon.β, recon.fft_plan; r̂ = los, x⃗ = xvec)
    end #for
    δ_r
end #func


function read_shifts(recon::AbstractRecon,
                    data_x::AbstractVector{T}, 
                    data_y::AbstractVector{T}, 
                    data_z::AbstractVector{T},
                    δ_r::AbstractArray{T, 3};
                    field = :disp, )  where T<:Real
    los = recon.los
    displacements = compute_displacements(δ_r, data_x, data_y, data_z, recon.box_size, recon.box_min, recon.fft_plan)
    data = (data_x, data_y, data_z)
    if field === :disp
        return displacements
    else
        rsd = Tuple(similar(d) for d in displacements)
        if los === nothing
            los = zeros(T, 3)
            for i in eachindex(data_x)
                dist = sqrt(data_x[i]^2 + data_y[i]^2 + data_z[i]^2)
                for j in 1:3
                    los[j] = data[j][i] / dist
                end #for
                for j in 1:3
                    rsd[j][i] = recon.f * sum([displacements[l][i] * los[l] for l in 1:3]) * los[j] 
                end #for
            end #for
        else #los provided
            for i in eachindex(data_x)
                for j in 1:3
                    rsd[j][i] = recon.f * sum([displacements[l][i] * los[l] for l in 1:3]) * los[j] 
                end #for
            end #for
        end #if

        if field === :rsd
            return rsd
        elseif field === :sum
            for j in 1:3
                displacements[j] .+= rsd[j]
            end #for
            return displacements
        end #if
    end #if          
end #func

function reconstructed_positions(recon::AbstractRecon,
                                data_x::AbstractVector{T}, 
                                data_y::AbstractVector{T}, 
                                data_z::AbstractVector{T},
                                δ_r::AbstractArray{T, 3};
                                field = :disp, )  where T<:Real

    displacements = read_shifts(recon, data_x, data_y, data_z, δ_r; field=field)
    data = (data_x, data_y, data_z)
    data_rec = Tuple(similar(d) for d in data)
    for i in 1:3
        data_rec[i] .= data[i] .- displacements[i]
    end #for
    data_rec
end #func