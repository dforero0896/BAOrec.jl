function cic!(ρ, 
              data_x, 
              data_y, 
              data_z, 
              data_w, 
              box_size, 
              box_min; 
              wrap::Bool = true)

    
    n_bins = size(ρ)
    @Threads.threads for i in eachindex(data_x)

        if wrap
            data_x[i] = (data_x[i] + box_size[1]) % box_size[1]
            data_y[i] = (data_y[i] + box_size[2]) % box_size[2]
            data_z[i] = (data_z[i] + box_size[3]) % box_size[3]
        end #if

        x::Real = (data_x[i] - box_min[1]) * n_bins[1] / box_size[1] + 1
        y::Real = (data_y[i] - box_min[2]) * n_bins[2] / box_size[2] + 1
        z::Real = (data_z[i] - box_min[3]) * n_bins[3] / box_size[3] + 1

        x0::Int = Int(floor(x))
        y0::Int = Int(floor(y))
        z0::Int = Int(floor(z))

        wx1::Real = x - x0
        wx0::Real = 1 - wx1
        wy1::Real = y - y0
        wy0::Real = 1 - wy1
        wz1::Real = z - z0
        wz0::Real = 1 - wz1

        x0 = (x0 == n_bins[1]+1) ? 1 : x0
        y0 = (y0 == n_bins[2]+1) ? 1 : y0
        z0 = (z0 == n_bins[3]+1) ? 1 : z0

        x1 = (x0 == n_bins[1]) ? 1 : x0 + 1
        y1 = (y0 == n_bins[2]) ? 1 : y0 + 1
        z1 = (z0 == n_bins[3]) ? 1 : z0 + 1

        wx0 *= data_w[i]
        wx1 *= data_w[i]

        
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

function read_cic!(output, 
                   field, 
                   data_x, 
                   data_y, 
                   data_z, 
                   box_size, 
                   box_min) 

    dims = size(field)
    cell_size = map(eltype(box_size), box_size ./ dims)
    u = zeros(Real, 3)
    d = zeros(Real, 3)
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
            index_d[j] = dist_i > dims[j] ? dist_i - dims[j] : dist_i
            index_u[j] = index_d[j] + 1
            index_u[j] = index_u[j] > dims[j] ? index_u[j] - dims[j] : index_u[j]
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