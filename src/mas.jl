function cic!(ρ::AbstractArray{T, 3}, data_x::AbstractVector{T}, data_y::AbstractVector{T}, data_z::AbstractVector{T}, data_w::AbstractVector{T}, box_size::SVector{3, T}, box_min::SVector{3, T}; wrap::Bool = true) where T<:Real

    
    n_bins = size(ρ)
    for i in eachindex(data_x)

        if wrap
            data_x[i] = (data_x[i] - box_min[1]) > box_size[1] ?  data_x[i] - box_size[1] : data_x[i]
            data_y[i] = (data_y[i] - box_min[1]) > box_size[1] ?  data_y[i] - box_size[1] : data_y[i]
            data_z[i] = (data_z[i] - box_min[1]) > box_size[1] ?  data_z[i] - box_size[1] : data_z[i]
        end #if

        x::T = (data_x[i] - box_min[1]) * n_bins[1] / box_size[1] + 1
        y::T = (data_y[i] - box_min[2]) * n_bins[2] / box_size[2] + 1
        z::T = (data_z[i] - box_min[3]) * n_bins[3] / box_size[3] + 1
        #@show box_min
        x0::Int = Int(floor(x))
        y0::Int = Int(floor(y))
        z0::Int = Int(floor(z))

        wx1::T = x - x0
        wx0::T = 1 - wx1
        wy1::T = y - y0
        wy0::T = 1 - wy1
        wz1::T = z - z0
        wz0::T = 1 - wz1

        x0 = (x0 == n_bins[1]+1) ? 1 : x0
        y0 = (y0 == n_bins[2]+1) ? 1 : y0
        z0 = (z0 == n_bins[3]+1) ? 1 : z0


        x1 = (x0 == n_bins[1]) & wrap ? 1 : x0 + 1
        y1 = (y0 == n_bins[2]) & wrap ? 1 : y0 + 1
        z1 = (z0 == n_bins[3]) & wrap ? 1 : z0 + 1

        wx0 *= data_w[i]
        wx1 *= data_w[i]
        #@show x0,y0,z0
        #@show x1,y1,z1
        
        ρ[x0,y0,z0] += wx0 * wy0 * wz0
        ρ[x1,y0,z0] += wx1 * wy0 * wz0
        ρ[x0,y1,z0] += wx0 * wy1 * wz0
        ρ[x0,y0,z1] += wx0 * wy0 * wz1
        ρ[x1,y1,z0] += wx1 * wy1 * wz0
        ρ[x1,y0,z1] += wx1 * wy0 * wz1
        ρ[x0,y1,z1] += wx0 * wy1 * wz1
        ρ[x1,y1,z1] += wx1 * wy1 * wz1
    end
    ρ
end
@kernel function cic_kernel!(ρ, n_bins, data_x, data_y, data_z, data_w, box_size, box_min, wrap)

    I = @index(Global, Linear)
    if wrap
        data_x[I] = (data_x[I] - box_min[1]) > box_size[1] ?  data_x[I] - box_size[1] : data_x[I]
        data_y[I] = (data_y[I] - box_min[1]) > box_size[1] ?  data_y[I] - box_size[1] : data_y[I]
        data_z[I] = (data_z[I] - box_min[1]) > box_size[1] ?  data_z[I] - box_size[1] : data_z[I]
    end #if

    x = (data_x[I] - box_min[1]) * n_bins[1] / box_size[1] + 1
    y = (data_y[I] - box_min[2]) * n_bins[2] / box_size[2] + 1
    z = (data_z[I] - box_min[3]) * n_bins[3] / box_size[3] + 1

    x0 = Int(floor(x))
    y0 = Int(floor(y))
    z0 = Int(floor(z))

    wx1 = x - x0
    wx0 = 1 - wx1
    wy1 = y - y0
    wy0 = 1 - wy1
    wz1 = z - z0
    wz0 = 1 - wz1

    x0 = (x0 == n_bins[1]+1) ? 1 : x0
    y0 = (y0 == n_bins[2]+1) ? 1 : y0
    z0 = (z0 == n_bins[3]+1) ? 1 : z0


    x1 = (x0 == n_bins[1]) & wrap ? 1 : x0 + 1
    y1 = (y0 == n_bins[2]) & wrap ? 1 : y0 + 1
    z1 = (z0 == n_bins[3]) & wrap ? 1 : z0 + 1

    wx0 *= data_w[I]
    wx1 *= data_w[I]
    
    
    CUDA.@atomic ρ[x0,y0,z0] += wx0 * wy0 * wz0
    CUDA.@atomic ρ[x1,y0,z0] += wx1 * wy0 * wz0
    CUDA.@atomic ρ[x0,y1,z0] += wx0 * wy1 * wz0
    CUDA.@atomic ρ[x0,y0,z1] += wx0 * wy0 * wz1
    CUDA.@atomic ρ[x1,y1,z0] += wx1 * wy1 * wz0
    CUDA.@atomic ρ[x1,y0,z1] += wx1 * wy0 * wz1
    CUDA.@atomic ρ[x0,y1,z1] += wx0 * wy1 * wz1
    CUDA.@atomic ρ[x1,y1,z1] += wx1 * wy1 * wz1

end #func
function cic!(ρ::CuArray{T, 3}, data_x::CuArray{T}, data_y::CuArray{T}, data_z::CuArray{T}, data_w::CuArray{T}, box_size::SVector{3, T}, box_min::SVector{3, T}; wrap::Bool = true) where T<:Real

    kernel! = cic_kernel!(KernelAbstractions.get_device(ρ))
    n_bins = cu([size(ρ)...])
    ev = kernel!(ρ, n_bins, data_x, data_y, data_z, data_w, box_size, box_min, wrap, ndrange = size(data_x))
    wait(ev)
    ρ
end

function is_in_range(point, ranges)
    all([(p[1] >= p[2].start) & (p[1] <= p[2].stop) for p in zip(point, ranges)])
end #funcs
function send_to_relevant_process(point, field, value, mpi_size, mpi_rank, comm)

    field_glob = global_view(field)
    if is_in_range(point, range_local(field))
        println("No message needed")
        field_glob[point[1], point[2], point[3]] += value
    else
        for i in 1:mpi_size
            i -= 1
            i == mpi_rank && continue
            if is_in_range(point, range_remote(field, i))
                
                recv_value = [typeof(value)(0)]
                send_value = [value]
                src = mpi_rank
                dst = i
                rreq = MPI.Irecv!(recv_value, src, src+32, comm)
                println("$mpi_rank: Sending   $mpi_rank -> $dst = $send_value\n")
                sreq = MPI.Isend(send_value, dst, mpi_rank+32, comm)
                stats = MPI.Waitall!([rreq, sreq])
                println("$mpi_rank: Received $src -> $mpi_rank = $recv_value\n")
                break
            end #if
        end #for
    end #if


end #func
function cic!(ρ::PencilArray{T, 3}, data_x::AbstractVector{T}, data_y::AbstractVector{T}, data_z::AbstractVector{T}, data_w::AbstractVector{T}, box_size::SVector{3, T}, box_min::SVector{3, T}; wrap::Bool = true) where T<:Real

    
    n_bins = size_global(ρ)
    local_range = range_local(ρ)
    ρ_global = global_view(ρ)
    for i in eachindex(data_x)

        if wrap
            data_x[i] = (data_x[i] - box_min[1]) > box_size[1] ?  data_x[i] - box_size[1] : data_x[i]
            data_y[i] = (data_y[i] - box_min[1]) > box_size[1] ?  data_y[i] - box_size[1] : data_y[i]
            data_z[i] = (data_z[i] - box_min[1]) > box_size[1] ?  data_z[i] - box_size[1] : data_z[i]
        end #if

        x::T = (data_x[i] - box_min[1]) * n_bins[1] / box_size[1] + 1
        y::T = (data_y[i] - box_min[2]) * n_bins[2] / box_size[2] + 1
        z::T = (data_z[i] - box_min[3]) * n_bins[3] / box_size[3] + 1
        #@show box_min
        x0::Int = Int(floor(x))
        y0::Int = Int(floor(y))
        z0::Int = Int(floor(z))

        wx1::T = x - x0
        wx0::T = 1 - wx1
        wy1::T = y - y0
        wy0::T = 1 - wy1
        wz1::T = z - z0
        wz0::T = 1 - wz1

        x0 = (x0 == n_bins[1]+1) ? 1 : x0
        y0 = (y0 == n_bins[2]+1) ? 1 : y0
        z0 = (z0 == n_bins[3]+1) ? 1 : z0


        x1 = (x0 == n_bins[1]) & wrap ? 1 : x0 + 1
        y1 = (y0 == n_bins[2]) & wrap ? 1 : y0 + 1
        z1 = (z0 == n_bins[3]) & wrap ? 1 : z0 + 1

        

        wx0 *= data_w[i]
        wx1 *= data_w[i]
        #@show x0,y0,z0
        #@show x1,y1,z1

        
        
        if is_in_range((x0,y0,z0), local_range) 
            ρ_global[x0,y0,z0] += wx0 * wy0 * wz0
        end #if
        if is_in_range((x1,y0,z0), local_range) 
            ρ_global[x1,y0,z0] += wx1 * wy0 * wz0
        end #if
        if is_in_range((x0,y1,z0), local_range) 
            ρ_global[x0,y1,z0] += wx0 * wy1 * wz0
        end #if
        if is_in_range((x0,y0,z1), local_range) 
            ρ_global[x0,y0,z1] += wx0 * wy0 * wz1
        end #if
        if is_in_range((x1,y1,z0), local_range) 
            ρ_global[x1,y1,z0] += wx1 * wy1 * wz0
        end #if
        if is_in_range((x1,y0,z1), local_range) 
            ρ_global[x1,y0,z1] += wx1 * wy0 * wz1
        end #if
        if is_in_range((x0,y1,z1), local_range) 
            ρ_global[x0,y1,z1] += wx0 * wy1 * wz1
        end #if
        if is_in_range((x1,y1,z1), local_range) 
            ρ_global[x1,y1,z1] += wx1 * wy1 * wz1
        end #if
    end #for
    ρ
end



function read_cic!(output::AbstractVector{T}, field::AbstractArray{T, 3}, data_x::AbstractVector{T}, data_y::AbstractVector{T}, data_z::AbstractVector{T}, box_size::SVector{3, T}, box_min::SVector{3, T}; wrap = true )  where T <: Real

    dims = size(field)
    cell_size = map(T, box_size ./ dims)
    u = zeros(T, 3)
    d = zeros(T, 3)
    index_u = zeros(Int, 3)
    index_d = zeros(Int, 3)
    data = (data_x, data_y, data_z)
    for i in eachindex(data_x)
        for j in 1:3
            dist = (data[j][i] - box_min[j]) / cell_size[j]
            dist_i = Int(floor(dist)) 
            u[j] = dist - dist_i
            d[j] = 1 - u[j]
            dist_i += 1
            index_d[j] = (dist_i > dims[j]) & wrap ? dist_i - dims[j] : dist_i
            index_u[j] = index_d[j] + 1
            index_u[j] = (index_u[j] > dims[j]) & wrap ? index_u[j] - dims[j] : index_u[j]
        end #for 
        output[i] = field[index_d[1], index_d[2], index_d[3]] * d[1] * d[2] * d[3] + 
                    field[index_d[1], index_d[2], index_u[3]] * d[1] * d[2] * u[3]+
                    field[index_d[1], index_u[2], index_d[3]] * d[1] * u[2] * d[3]+
                    field[index_d[1], index_u[2], index_u[3]] * d[1] * u[2] * u[3]+
                    field[index_u[1], index_d[2], index_d[3]] * u[1] * d[2] * d[3]+
                    field[index_u[1], index_d[2], index_u[3]] * u[1] * d[2] * u[3]+
                    field[index_u[1], index_u[2], index_d[3]] * u[1] * u[2] * d[3]+
                    field[index_u[1], index_u[2], index_u[3]] * u[1] * u[2] * u[3]

    end #for
    output
end #func

@kernel function read_cic_kernel!(output, field, data_x, data_y, data_z, box_size, box_min, dims, wrap)

    i = @index(Global, Linear)
    dist_x = (data_x[i] - box_min[1]) * dims[1] / box_size[1]
    dist_y = (data_y[i] - box_min[2]) * dims[2] / box_size[2]
    dist_z = (data_z[i] - box_min[3]) * dims[3] / box_size[3]

    dist_i_x = Int(floor(dist_x)) 
    dist_i_y = Int(floor(dist_y)) 
    dist_i_z = Int(floor(dist_z)) 

    ux = dist_x - dist_i_x
    uy = dist_y - dist_i_y
    uz = dist_z - dist_i_z

    dx = 1 - ux
    dy = 1 - uy
    dz = 1 - uz

    dist_i_x += 1
    dist_i_y += 1
    dist_i_z += 1


    index_d_x = (dist_i_x > dims[1]) & wrap ? dist_i_x - dims[1] : dist_i_x
    index_d_y = (dist_i_y > dims[2]) & wrap ? dist_i_y - dims[2] : dist_i_y
    index_d_z = (dist_i_z > dims[3]) & wrap ? dist_i_z - dims[3] : dist_i_z

    index_u_x = index_d_x + 1
    index_u_y = index_d_y + 1
    index_u_z = index_d_z + 1


    index_u_x = (index_u_x > dims[1]) & wrap ? index_u_x - dims[1] : index_u_x
    index_u_y = (index_u_y > dims[2]) & wrap ? index_u_y - dims[2] : index_u_y
    index_u_z = (index_u_z > dims[3]) & wrap ? index_u_z - dims[3] : index_u_z


    output[i] = field[index_d_x, index_d_y, index_d_z] * dx * dy * dz + 
                field[index_d_x, index_d_y, index_u_z] * dx * dy * uz +
                field[index_d_x, index_u_y, index_d_z] * dx * uy * dz +
                field[index_d_x, index_u_y, index_u_z] * dx * uy * uz +
                field[index_u_x, index_d_y, index_d_z] * ux * dy * dz +
                field[index_u_x, index_d_y, index_u_z] * ux * dy * uz +
                field[index_u_x, index_u_y, index_d_z] * ux * uy * dz +
                field[index_u_x, index_u_y, index_u_z] * ux * uy * uz






end #func

function read_cic!(output::CuArray{T}, field::CuArray{T, 3}, data_x::CuArray{T}, data_y::CuArray{T}, data_z::CuArray{T}, box_size::SVector{3, T}, box_min::SVector{3, T}; wrap = true )  where T <: Real

    kernel! = read_cic_kernel!(KernelAbstractions.get_device(output))
    dims = cu([size(field)...])
    ev = kernel!(output, field, data_x, data_y, data_z, box_size, box_min, dims, wrap, ndrange = size(data_x))
    wait(ev)

end #func

function read_cic!(output::AbstractVector{T}, field::PencilArray{T, 3}, data_x::AbstractVector{T}, data_y::AbstractVector{T}, data_z::AbstractVector{T}, box_size::SVector{3, T}, box_min::SVector{3, T}; wrap = true )  where T <: Real

    dims = size_global(field)
    local_range = range_local(field)
    cell_size = map(T, box_size ./ dims)
    u = zeros(T, 3)
    d = zeros(T, 3)
    index_u = zeros(Int, 3)
    index_d = zeros(Int, 3)
    data = (data_x, data_y, data_z)
    for i in eachindex(data_x)
        for j in 1:3
            dist = (data[j][i] - box_min[j]) / cell_size[j]
            dist_i = Int(floor(dist)) 
            u[j] = dist - dist_i
            d[j] = 1 - u[j]
            dist_i += 1
            index_d[j] = (dist_i > dims[j]) & wrap ? dist_i - dims[j] : dist_i
            index_u[j] = index_d[j] + 1
            index_u[j] = (index_u[j] > dims[j]) & wrap ? index_u[j] - dims[j] : index_u[j]
        end #for 
        output[i] = 0
        if is_in_range((index_d[1], index_d[2], index_d[3]), local_range)
            output[i] += field[index_d[1], index_d[2], index_d[3]] * d[1] * d[2] * d[3]
        end #if

        if is_in_range((index_d[1], index_d[2], index_u[3]), local_range)
            output[i] += field[index_d[1], index_d[2], index_u[3]] * d[1] * d[2] * u[3]
        end #if

        if is_in_range((index_d[1], index_u[2], index_d[3]), local_range)
            output[i] += field[index_d[1], index_u[2], index_d[3]] * d[1] * u[2] * d[3]
        end #if

        if is_in_range((index_d[1], index_u[2], index_u[3]), local_range)
            output[i] += field[index_d[1], index_u[2], index_u[3]] * d[1] * u[2] * u[3]
        end #if

        if is_in_range((index_u[1], index_d[2], index_d[3]), local_range)
            output[i] += field[index_u[1], index_d[2], index_d[3]] * u[1] * d[2] * d[3]
        end #if

        if is_in_range((index_u[1], index_d[2], index_u[3]), local_range)
            output[i] += field[index_u[1], index_d[2], index_u[3]] * u[1] * d[2] * u[3]
        end #if

        if is_in_range((index_u[1], index_u[2], index_d[3]), local_range)
            output[i] += field[index_u[1], index_u[2], index_d[3]] * u[1] * u[2] * d[3]
        end #if

        if is_in_range((index_u[1], index_u[2], index_u[3]), local_range)
            output[i] += field[index_u[1], index_u[2], index_u[3]] * u[1] * u[2] * u[3]
        end #if
                                
    end #for
    output
end #func


#=
function read_grad_cic!(out_x::AbstractVector{T}, out_y::AbstractVector{T}, out_z::AbstractVector{T}, field::AbstractArray{T, 3}, data_x::AbstractVector{T}, data_y::AbstractVector{T}, data_z::AbstractVector{T}, box_size::SVector{3, T}, box_min::SVector{3, T}; wrap = true )  where T <: Real

    dims = size(field)
    cell = @.  2 * box_size / dims
    u = zeros(T, 3)
    d = zeros(T, 3)
    index_u = zeros(Int, 3)
    index_d = zeros(Int, 3)
    index_p = zeros(Int, 3)
    index_m = zeros(Int, 3)
    data = (data_x, data_y, data_z)
    for i in eachindex(data_x)
        for j in 1:3
            dist = (data[j][i] - box_min[j]) / cell[j]
            dist_i = Int(floor(dist)) 
            u[j] = dist - dist_i
            d[j] = 1 - u[j]
            dist_i += 1 #1-based indexing
            index_d[j] = (dist_i > dims[j]) & wrap ? dist_i - dims[j] : dist_i #ix0
            index_u[j] = index_d[j] + 1
            index_u[j] = (index_u[j] > dims[j]) & wrap ? index_u[j] - dims[j] : index_u[j] #ixp
            index_p[j] = index_u[j] + 1
            index_p[j] = (index_p[j] > dims[j]) & wrap ? index_p[j] - dims[j] : index_p[j] #ixpp
            index_m[j] = index_d[j] - 1
            index_m[j] = (index_m[j] < 1) & wrap ? index_m[j] + dims[j] : index_m[j] #ixm
        end #for 

        ix0, iy0, iz0 = index_d
        ixp, iyp, izp = index_u
        ixpp, iypp, izpp = index_p
        ixm, iym, izm = index_m

        dx, dy, dz = u
        wx, wy, wz = d

        wt = wx*wy*wz
        px = (field[ixp, iy0, iz0]-field[ixm, iy0, iz0])*wt
        py = (field[ix0, iyp, iz0]-field[ix0, iym, iz0])*wt
        pz = (field[ix0, iy0, izp]-field[ix0, iy0, izm])*wt
        wt = dx*wy*wz;
        px += (field[ixpp, iy0, iz0]-field[ix0, iy0, iz0])*wt
        py += (field[ixp, iyp, iz0]-field[ixp, iym, iz0])*wt
        pz += (field[ixp, iy0, izp]-field[ixp, iy0, izm])*wt
        wt = wx*dy*wz;
        px += (field[ixp, iyp, iz0]-field[ixm, iyp, iz0])*wt
        py += (field[ix0, iypp, iz0]-field[ix0, iy0, iz0])*wt
        pz += (field[ix0, iyp, izp]-field[ix0, iyp, izm])*wt
        wt = wx*wy*dz;
        px += (field[ixp, iy0, izp]-field[ixm, iy0, izp])*wt
        py += (field[ix0, iyp, izp]-field[ix0, iym, izp])*wt
        pz += (field[ix0, iy0, izpp]-field[ix0, iy0, iz0])*wt
        wt = dx*dy*wz;
        px += (field[ixpp, iyp, iz0]-field[ix0, iyp, iz0])*wt
        py += (field[ixp, iypp, iz0]-field[ixp, iy0, iz0])*wt
        pz += (field[ixp, iyp, izp]-field[ixp, iyp, izm])*wt
        wt = dx*wy*dz;
        px += (field[ixpp, iy0, izp]-field[ix0, iy0, izp])*wt
        py += (field[ixp, iyp, izp]-field[ixp, iym, izp])*wt
        pz += (field[ixp, iy0, izpp]-field[ixp, iy0, iz0])*wt
        wt = wx*dy*dz;
        px += (field[ixp, iyp, izp]-field[ixm, iyp, izp])*wt
        py += (field[ix0, iypp, izp]-field[ix0, iy0, izp])*wt
        pz += (field[ix0, iyp, izpp]-field[ix0, iyp, iz0])*wt
        wt = dx*dy*dz;
        px += (field[ixpp, iyp, izp]-field[ix0, iyp, izp])*wt
        py += (field[ixp, iypp, izp]-field[ixp, iy0, izp])*wt
        pz += (field[ixp, iyp, izpp]-field[ixp, iyp, iz0])*wt
        
        out_x[i] = px/cell[1]
        out_y[i] = py/cell[2]
        out_z[i] = pz/cell[3]

    end #for 

    (out_x, out_y, out_z)

end #func
=#



function read_cic_single(field::AbstractArray{T, 3}, dims, data_x::T, data_y::T, data_z::T, box_size::SVector{3, T}, box_min::SVector{3, T}; wrap = true )  where T <: Real

    
    dist_x = (data_x - box_min[1]) * dims[1] / box_size[1]
    dist_y = (data_y - box_min[2]) * dims[2] / box_size[2]
    dist_z = (data_z - box_min[3]) * dims[3] / box_size[3]

    dist_i_x = Int(floor(dist_x)) 
    dist_i_y = Int(floor(dist_y)) 
    dist_i_z = Int(floor(dist_z)) 

    ux = dist_x - dist_i_x
    uy = dist_y - dist_i_y
    uz = dist_z - dist_i_z

    dx = 1 - ux
    dy = 1 - uy
    dz = 1 - uz

    dist_i_x += 1
    dist_i_y += 1
    dist_i_z += 1


    index_d_x = (dist_i_x > dims[1]) & wrap ? dist_i_x - dims[1] : dist_i_x
    index_d_y = (dist_i_y > dims[2]) & wrap ? dist_i_y - dims[2] : dist_i_y
    index_d_z = (dist_i_z > dims[3]) & wrap ? dist_i_z - dims[3] : dist_i_z

    index_u_x = index_d_x + 1
    index_u_y = index_d_y + 1
    index_u_z = index_d_z + 1


    index_u_x = (index_u_x > dims[1]) & wrap ? index_u_x - dims[1] : index_u_x
    index_u_y = (index_u_y > dims[2]) & wrap ? index_u_y - dims[2] : index_u_y
    index_u_z = (index_u_z > dims[3]) & wrap ? index_u_z - dims[3] : index_u_z


    output = field[index_d_x, index_d_y, index_d_z] * dx * dy * dz + 
             field[index_d_x, index_d_y, index_u_z] * dx * dy * uz +
             field[index_d_x, index_u_y, index_d_z] * dx * uy * dz +
             field[index_d_x, index_u_y, index_u_z] * dx * uy * uz +
             field[index_u_x, index_d_y, index_d_z] * ux * dy * dz +
             field[index_u_x, index_d_y, index_u_z] * ux * dy * uz +
             field[index_u_x, index_u_y, index_d_z] * ux * uy * dz +
             field[index_u_x, index_u_y, index_u_z] * ux * uy * uz


end #func



function read_grad_cic!(out_x::AbstractVector{T}, out_y::AbstractVector{T}, out_z::AbstractVector{T}, field::AbstractArray{T, 3}, data_x::AbstractVector{T}, data_y::AbstractVector{T}, data_z::AbstractVector{T}, box_size::SVector{3, T}, box_min::SVector{3, T}; wrap = true )  where T <: Real

    
    function phi_fun(x, y, z)
        return read_cic_single(field, size(field), x, y, z, box_size, box_min)
    end #fun


    Threads.@threads for i in eachindex(data_x)

        ∇field = Zygote.gradient(phi_fun , data_x[i], data_y[i], data_z[i])
        out_x = ∇field[1]
        out_y = ∇field[2]
        out_z = ∇field[3]

    end #for

end #func