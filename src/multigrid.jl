 # Copied from pyrecon
 
function zeros_halved_array(array::AbstractArray)
    if typeof(array)<:CuArray
        new_arr = CuArray{eltype(array), 3}(undef, map(x->Int(floor(0.5x)), size(array))...)
        fill!(new_arr, 0)
        return new_arr
    else
        return zeros(eltype(array), map(x->Int(floor(0.5x)), size(array))...)
    end #if
end #func


function jacobi!(v::AbstractArray{T,3}, f::AbstractArray{T,3}, x_vec::Tuple{AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}, box_size::SVector{3,T}, box_min::SVector{3,T}, β::T, damping_factor::T, niterations::Int; los=nothing) where T <: AbstractFloat

    jac = similar(v)
    cell = map(T, box_size ./ size(v))
    cell2 = cell .^2
    icell2 = cell2 .^-1   
    losn = los != nothing ? los ./ cell : nothing
    nmesh = size(v)
    Base.Cartesian.@nexprs 3 i -> begin
        nmesh_i = nmesh[i]
        x_vec_i = x_vec[i]
        cell_i = cell[i]
        cell2_i = cell2[i]
        icell2_i = icell2[i]
        losn_i = losn != nothing ? losn[i] : nothing
    end

    for iter in 1:niterations

        #@inbounds Threads.@threads for I in CartesianIndices(v)

        #    ix0, iy0, iz0 = Tuple(I)
        if los == nothing
            @tturbo for iz0 = axes(v,3), iy0 = axes(v,2), ix0 = axes(v,1)

                ixp = ix0 + 1 
                ixp = ixp > nmesh_1 ? ixp - nmesh_1 : ixp
                ixm = ix0 - 1 
                ixm = ixm < 1 ? ixm + nmesh_1 : ixm

                iyp = iy0 + 1 
                iyp = iyp > nmesh_2 ? iyp - nmesh_2 : iyp
                iym = iy0 - 1 
                iym = iym < 1 ? iym + nmesh_2 : iym

                izp = iz0 + 1 
                izp = izp > nmesh_3 ? izp - nmesh_3 : izp
                izm = iz0 - 1 
                izm = izm < 1 ? izm + nmesh_3 : izm
                

                px = los == nothing ? x_vec_1[ix0] / cell_1 : losn_1
                py = los == nothing ? x_vec_2[iy0] / cell_2 : losn_2
                pz = los == nothing ? x_vec_3[iz0] / cell_3 : losn_3

                g = β / (cell2_1 * px^2 + cell2_2 * py^2 + cell2_3 * pz^2)
                gpx2 = icell2_1 + g * px^2
                gpy2 = icell2_2 + g * py^2
                gpz2 = icell2_3 + g * pz^2

                jac[ix0, iy0, iz0] = f[ix0, iy0, iz0]+
                        gpx2*(v[ixp, iy0, iz0]+v[ixm, iy0, iz0])+
                        gpy2*(v[ix0, iyp, iz0]+v[ix0, iym, iz0])+
                        gpz2*(v[ix0, iy0, izp]+v[ix0, iy0, izm])+
                        g/2*(px*py*(v[ixp, iyp, iz0]+v[ixm, iym, iz0]
                                -v[ixm, iyp, iz0]-v[ixp, iym, iz0])+
                        px*pz*(v[ixp, iy0, izp]+v[ixm, iy0, izm]
                            -v[ixm, iy0, izp]-v[ixp, iy0, izm])+
                        py*pz*(v[ix0, iyp, izp]+v[ix0, iym, izm]
                            -v[ix0, iym, izp]-v[ix0, iyp, izm]));


                
                jac[ix0, iy0, iz0] += g*(px*(v[ixp, iy0, iz0]-v[ixm, iy0, iz0])+
                                py*(v[ix0, iyp, iz0]-v[ix0, iym, iz0])+
                                pz*(v[ix0, iy0, izp]-v[ix0, iy0, izm]))
                

                jac[ix0, iy0, iz0] /= 2*(gpx2 + gpy2 + gpz2)
            end #for
        else
                @tturbo for iz0 = axes(v,3), iy0 = axes(v,2), ix0 = axes(v,1)

                    ixp = ix0 + 1 
                    ixp = ixp > nmesh_1 ? ixp - nmesh_1 : ixp
                    ixm = ix0 - 1 
                    ixm = ixm < 1 ? ixm + nmesh_1 : ixm
    
                    iyp = iy0 + 1 
                    iyp = iyp > nmesh_2 ? iyp - nmesh_2 : iyp
                    iym = iy0 - 1 
                    iym = iym < 1 ? iym + nmesh_2 : iym
    
                    izp = iz0 + 1 
                    izp = izp > nmesh_3 ? izp - nmesh_3 : izp
                    izm = iz0 - 1 
                    izm = izm < 1 ? izm + nmesh_3 : izm
                    
    
                    px = los == nothing ? x_vec_1[ix0] / cell_1 : losn_1
                    py = los == nothing ? x_vec_2[iy0] / cell_2 : losn_2
                    pz = los == nothing ? x_vec_3[iz0] / cell_3 : losn_3
    
                    g = β / (cell2_1 * px^2 + cell2_2 * py^2 + cell2_3 * pz^2)
                    gpx2 = icell2_1 + g * px^2
                    gpy2 = icell2_2 + g * py^2
                    gpz2 = icell2_3 + g * pz^2
    
                    jac[ix0, iy0, iz0] = f[ix0, iy0, iz0]+
                            gpx2*(v[ixp, iy0, iz0]+v[ixm, iy0, iz0])+
                            gpy2*(v[ix0, iyp, iz0]+v[ix0, iym, iz0])+
                            gpz2*(v[ix0, iy0, izp]+v[ix0, iy0, izm])+
                            g/2*(px*py*(v[ixp, iyp, iz0]+v[ixm, iym, iz0]
                                    -v[ixm, iyp, iz0]-v[ixp, iym, iz0])+
                            px*pz*(v[ixp, iy0, izp]+v[ixm, iy0, izm]
                                -v[ixm, iy0, izp]-v[ixp, iy0, izm])+
                            py*pz*(v[ix0, iyp, izp]+v[ix0, iym, izm]
                                -v[ix0, iym, izp]-v[ix0, iyp, izm]));
                   
    
                    jac[ix0, iy0, iz0] /= 2*(gpx2 + gpy2 + gpz2)
                end #for
        end #if
     

        
        
        @tturbo @. v = (1-damping_factor)*v + damping_factor*jac;
        

    end #for iter
 end #func

 @kernel function jacobi_kernel!(jac, v, f, x_vec, nmesh, cell, cell2, icell2, β, los, losn)

    I = @index(Global, Cartesian)
    ix0, iy0, iz0 = Tuple(I)

    ixp = ix0 + 1 
    ixp = ixp > nmesh[1] ? ixp - nmesh[1] : ixp
    ixm = ix0 - 1 
    ixm = ixm < 1 ? ixm + nmesh[1] : ixm

    iyp = iy0 + 1 
    iyp = iyp > nmesh[2] ? iyp - nmesh[2] : iyp
    iym = iy0 - 1 
    iym = iym < 1 ? iym + nmesh[2] : iym

    izp = iz0 + 1 
    izp = izp > nmesh[3] ? izp - nmesh[3] : izp
    izm = iz0 - 1 
    izm = izm < 1 ? izm + nmesh[3] : izm
    
    #px = los == nothing ? ix0 + offset[1] : losn[1]
    #py = los == nothing ? iy0 + offset[2] : losn[2]
    #pz = los == nothing ? iz0 + offset[3] : losn[3]

    px = losn == nothing ? x_vec[1][I[1]] / cell[1] : losn[1]
    py = losn == nothing ? x_vec[2][I[2]] / cell[2] : losn[2]
    pz = losn == nothing ? x_vec[3][I[3]] / cell[3] : losn[3]

    g = β / (cell2[1] * px^2 + cell2[2] * py^2 + cell2[3] * pz^2)
    gpx2 = icell2[1] + g * px^2
    gpy2 = icell2[2] + g * py^2
    gpz2 = icell2[3] + g * pz^2

    jac[I] = f[I]+
            gpx2*(v[ixp, iy0, iz0]+v[ixm, iy0, iz0])+
            gpy2*(v[ix0, iyp, iz0]+v[ix0, iym, iz0])+
            gpz2*(v[ix0, iy0, izp]+v[ix0, iy0, izm])+
            g/2*(px*py*(v[ixp, iyp, iz0]+v[ixm, iym, iz0]
                        -v[ixm, iyp, iz0]-v[ixp, iym, iz0])+
            px*pz*(v[ixp, iy0, izp]+v[ixm, iy0, izm]
                    -v[ixm, iy0, izp]-v[ixp, iy0, izm])+
            py*pz*(v[ix0, iyp, izp]+v[ix0, iym, izm]
                    -v[ix0, iym, izp]-v[ix0, iyp, izm]));


    if los == nothing
        jac[I] += g*(px*(v[ixp, iy0, iz0]-v[ixm, iy0, iz0])+
                        py*(v[ix0, iyp, iz0]-v[ix0, iym, iz0])+
                        pz*(v[ix0, iy0, izp]-v[ix0, iy0, izm]))
    end #if

    jac[I] /= 2*(gpx2 + gpy2 + gpz2)


 end #func
 
 function jacobi!(v::CuArray{T,3}, f::CuArray{T,3}, x_vec::Tuple{AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}, box_size::SVector{3,T}, box_min::SVector{3,T}, β::T, damping_factor::T, niterations::Int; los=nothing) where T <: AbstractFloat

    jac = similar(v)
    cell = map((x) -> cu(T(x)), box_size ./ size(v))
    cell2 = cell .^2
    icell2 = cell2 .^-1
    losn = los != nothing ? los ./ cell : nothing
    nmesh = map(cu, size(v))
    x_vec = map(cu, x_vec)
    device  = KernelAbstractions.get_device(v)
    kernel! = jacobi_kernel!(device)
    for iter in 1:niterations
        ev = kernel!(jac, v, f, x_vec, nmesh, cell, cell2, icell2, β, los, losn, ndrange = size(v))
        wait(ev)
        @. v = (1-damping_factor)*v + damping_factor*jac
    end #for iter
    v
 end #func


 function residual!(r::AbstractArray{T,3}, v::AbstractArray{T,3}, f::AbstractArray{T,3}, x_vec::Tuple{AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}, box_size::SVector{3,T}, box_min::SVector{3,T}, β::T, damping_factor::T, niterations::Int; los=nothing) where T <: AbstractFloat


    cell = map(T, box_size ./ size(v))
    cell2 = cell .^2
    icell2 = cell2 .^-1
    losn = los != nothing ? los ./ cell : nothing
    nmesh = size(v)

    Base.Cartesian.@nexprs 3 i -> begin
        nmesh_i = nmesh[i]
        x_vec_i = x_vec[i]
        cell_i = cell[i]
        cell2_i = cell2[i]
        icell2_i = icell2[i]
        losn_i = losn != nothing ? losn[i] : nothing
    end
    
    #@inbounds Threads.@threads for I in CartesianIndices(v)

    #    ix0, iy0, iz0 = Tuple(I)
    if los == nothing
        @tturbo for iz0 = axes(v,3), iy0 = axes(v,2), ix0 = axes(v,1)
            ixp = ix0 + 1 
            ixp = ixp > nmesh_1 ? ixp - nmesh_1 : ixp
            ixm = ix0 - 1 
            ixm = ixm < 1 ? ixm + nmesh_1 : ixm

            iyp = iy0 + 1 
            iyp = iyp > nmesh_2 ? iyp - nmesh_2 : iyp
            iym = iy0 - 1 
            iym = iym < 1 ? iym + nmesh_2 : iym

            izp = iz0 + 1 
            izp = izp > nmesh_3 ? izp - nmesh_3 : izp
            izm = iz0 - 1 
            izm = izm < 1 ? izm + nmesh_3 : izm
            
        

            px = los == nothing ? x_vec_1[ix0] / cell_1 : losn_1
            py = los == nothing ? x_vec_2[iy0] / cell_2 : losn_2
            pz = los == nothing ? x_vec_3[iz0] / cell_3 : losn_3

            g = β / (cell2_1 * px^2 + cell2_2 * py^2 + cell2_3 * pz^2)
            gpx2 = icell2_1 + g * px^2
            gpy2 = icell2_2 + g * py^2
            gpz2 = icell2_3 + g * pz^2

            r[ix0,iy0,iz0] = 2*(gpx2 + gpy2 + gpz2)*v[ix0,iy0,iz0] -
                (gpx2*(v[ixp, iy0, iz0]+v[ixm, iy0, iz0])+
                gpy2*(v[ix0, iyp, iz0]+v[ix0, iym, iz0])+
                gpz2*(v[ix0, iy0, izp]+v[ix0, iy0, izm])+
                g/2*(px*py*(v[ixp, iyp, iz0]+v[ixm, iym, iz0]
                            -v[ixm, iyp, iz0]-v[ixp, iym, iz0])+
                        px*pz*(v[ixp, iy0, izp]+v[ixm, iy0, izm]
                            -v[ixm, iy0, izp]-v[ixp, iy0, izm])+
                        py*pz*(v[ix0, iyp, izp]+v[ix0, iym, izm]
                            -v[ix0, iym, izp]-v[ix0, iyp, izm])));
            
            
                r[ix0,iy0,iz0] -= g*(px*(v[ixp, iy0, iz0]-v[ixm, iy0, iz0])+
                        py*(v[ix0, iyp, iz0]-v[ix0, iym, iz0])+
                        pz*(v[ix0, iy0, izp]-v[ix0, iy0, izm]));
            

        end # for v
    else
        @tturbo for iz0 = axes(v,3), iy0 = axes(v,2), ix0 = axes(v,1)
            ixp = ix0 + 1 
            ixp = ixp > nmesh_1 ? ixp - nmesh_1 : ixp
            ixm = ix0 - 1 
            ixm = ixm < 1 ? ixm + nmesh_1 : ixm

            iyp = iy0 + 1 
            iyp = iyp > nmesh_2 ? iyp - nmesh_2 : iyp
            iym = iy0 - 1 
            iym = iym < 1 ? iym + nmesh_2 : iym

            izp = iz0 + 1 
            izp = izp > nmesh_3 ? izp - nmesh_3 : izp
            izm = iz0 - 1 
            izm = izm < 1 ? izm + nmesh_3 : izm
            
        

            px = los == nothing ? x_vec_1[ix0] / cell_1 : losn_1
            py = los == nothing ? x_vec_2[iy0] / cell_2 : losn_2
            pz = los == nothing ? x_vec_3[iz0] / cell_3 : losn_3

            g = β / (cell2_1 * px^2 + cell2_2 * py^2 + cell2_3 * pz^2)
            gpx2 = icell2_1 + g * px^2
            gpy2 = icell2_2 + g * py^2
            gpz2 = icell2_3 + g * pz^2

            r[ix0,iy0,iz0] = 2*(gpx2 + gpy2 + gpz2)*v[ix0,iy0,iz0] -
                (gpx2*(v[ixp, iy0, iz0]+v[ixm, iy0, iz0])+
                gpy2*(v[ix0, iyp, iz0]+v[ix0, iym, iz0])+
                gpz2*(v[ix0, iy0, izp]+v[ix0, iy0, izm])+
                g/2*(px*py*(v[ixp, iyp, iz0]+v[ixm, iym, iz0]
                            -v[ixm, iyp, iz0]-v[ixp, iym, iz0])+
                        px*pz*(v[ixp, iy0, izp]+v[ixm, iy0, izm]
                            -v[ixm, iy0, izp]-v[ixp, iy0, izm])+
                        py*pz*(v[ix0, iyp, izp]+v[ix0, iym, izm]
                            -v[ix0, iym, izp]-v[ix0, iyp, izm])))

        end # for v
    end #if

    @tturbo for iz0 = axes(v,3), iy0 = axes(v,2), ix0 = axes(v,1)
        r[ix0,iy0,iz0] = f[ix0,iy0,iz0] - r[ix0,iy0,iz0]
    end #for
    r
 end #func

 @kernel function residual_kernel!(r, v, f, x_vec, nmesh, cell, cell2, icell2, β, los, losn)

    I = @index(Global, Cartesian)

    ix0, iy0, iz0 = Tuple(I)

    ixp = ix0 + 1 
    ixp = ixp > nmesh[1] ? ixp - nmesh[1] : ixp
    ixm = ix0 - 1 
    ixm = ixm < 1 ? ixm + nmesh[1] : ixm

    iyp = iy0 + 1 
    iyp = iyp > nmesh[2] ? iyp - nmesh[2] : iyp
    iym = iy0 - 1 
    iym = iym < 1 ? iym + nmesh[2] : iym

    izp = iz0 + 1 
    izp = izp > nmesh[3] ? izp - nmesh[3] : izp
    izm = iz0 - 1 
    izm = izm < 1 ? izm + nmesh[3] : izm


    px = los == nothing ? x_vec[1][I[1]] / cell[1] : losn[1]
    py = los == nothing ? x_vec[2][I[2]] / cell[2] : losn[2]
    pz = los == nothing ? x_vec[3][I[3]] / cell[3] : losn[3]

    g = β / (cell2[1] * px^2 + cell2[2] * py^2 + cell2[3] * pz^2)
    gpx2 = icell2[1] + g * px^2
    gpy2 = icell2[2] + g * py^2
    gpz2 = icell2[3] + g * pz^2

    r[I] = 2*(gpx2 + gpy2 + gpz2)*v[I] -
        (gpx2*(v[ixp, iy0, iz0]+v[ixm, iy0, iz0])+
        gpy2*(v[ix0, iyp, iz0]+v[ix0, iym, iz0])+
        gpz2*(v[ix0, iy0, izp]+v[ix0, iy0, izm])+
        g/2*(px*py*(v[ixp, iyp, iz0]+v[ixm, iym, iz0]
                    -v[ixm, iyp, iz0]-v[ixp, iym, iz0])+
                px*pz*(v[ixp, iy0, izp]+v[ixm, iy0, izm]
                    -v[ixm, iy0, izp]-v[ixp, iy0, izm])+
                py*pz*(v[ix0, iyp, izp]+v[ix0, iym, izm]
                    -v[ix0, iym, izp]-v[ix0, iyp, izm])));
    
    if los == nothing
        r[I] -= g*(px*(v[ixp, iy0, iz0]-v[ixm, iy0, iz0])+
                py*(v[ix0, iyp, iz0]-v[ix0, iym, iz0])+
                pz*(v[ix0, iy0, izp]-v[ix0, iy0, izm]));
    end #if

 end #fun

 function residual!(r::CuArray{T,3}, v::CuArray{T,3}, f::CuArray{T,3}, x_vec::Tuple{AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}, box_size::SVector{3,T}, box_min::SVector{3,T}, β::T, damping_factor::T, niterations::Int; los=nothing) where T <: AbstractFloat


    cell = map(T, box_size ./ size(v))
    cell2 = cell .^2
    icell2 = cell2 .^-1
    losn = los != nothing ? los ./ cell : nothing
    nmesh = map(cu, size(v))
    x_vec = map(cu, x_vec)
    
    device = KernelAbstractions.get_device(r)
    kernel! = residual_kernel!(device)
    
    ev = kernel!(r, v, f, x_vec, nmesh, cell, cell2, icell2, β, los, losn, ndrange = size(r))
    wait(ev)
    @. r = f - r
    r
 end #func


 function prolong!(v1h::AbstractArray{T,3}, v2h::AbstractArray{T,3}) where T <: AbstractFloat
    nmesh = size(v2h)
    nmesh2 = size(v1h)
    Base.Cartesian.@nexprs 3 i -> begin
        nmesh_i = nmesh[i]
        nmesh2_i = nmesh2[i]
    end
    #@inbounds Threads.@threads for I in CartesianIndices(v2h)
    @tturbo for iz0 = axes(v2h,3), iy0 = axes(v2h,2), ix0 = axes(v2h,1)
    #    ix0, iy0, iz0 = Tuple(I)
        

        ixp = ix0 + 1 
        ixp = ixp > nmesh_1 ? ixp - nmesh_1 : ixp
        ixm = ix0 - 1 
        ixm = ixm < 1 ? ixm + nmesh_1 : ixm

        iyp = iy0 + 1 
        iyp = iyp > nmesh_2 ? iyp - nmesh_2 : iyp
        iym = iy0 - 1 
        iym = iym < 1 ? iym + nmesh_2 : iym

        izp = iz0 + 1 
        izp = izp > nmesh_3 ? izp - nmesh_3 : izp
        izm = iz0 - 1 
        izm = izm < 1 ? izm + nmesh_3 : izm

        i2x0 = 2*ix0
        i2xp = i2x0 + 1
        i2xp = i2xp > nmesh2_1 ? i2xp - nmesh2_1 : i2xp

        i2y0 = 2*iy0
        i2yp = i2y0 + 1
        i2yp = i2yp > nmesh2_2 ? i2yp - nmesh2_2 : i2yp

        i2z0 = 2*iz0
        i2zp = i2z0 + 1
        i2zp = i2zp > nmesh2_3 ? i2zp - nmesh2_3 : i2zp

        v1h[i2x0, i2y0, i2z0] = v2h[ix0,iy0,iz0];
        v1h[i2xp, i2y0, i2z0] = (v2h[ix0,iy0,iz0] + v2h[ixp, iy0, iz0])/2;
        v1h[i2x0, i2yp, i2z0] = (v2h[ix0,iy0,iz0] + v2h[ix0, iyp, iz0])/2;
        v1h[i2x0, i2y0, i2zp] = (v2h[ix0,iy0,iz0] + v2h[ix0, iy0, izp])/2;
        v1h[i2xp, i2yp, i2z0] = (v2h[ix0,iy0,iz0] + v2h[ixp, iy0, iz0]
                              + v2h[ix0, iyp, iz0] + v2h[ixp, iyp, iz0])/4;
        v1h[i2x0, i2yp, i2zp] = (v2h[ix0,iy0,iz0] + v2h[ix0, iyp, iz0]
                              + v2h[ix0, iy0, izp] + v2h[ix0, iyp, izp])/4;
        v1h[i2xp, i2y0, i2zp] = (v2h[ix0,iy0,iz0] + v2h[ixp, iy0, iz0]
                              + v2h[ix0, iy0, izp] + v2h[ixp, iy0, izp])/4;
        v1h[i2xp, i2yp, i2zp] = (v2h[ix0,iy0,iz0] + v2h[ixp, iy0, iz0]
                              + v2h[ix0, iyp, iz0] + v2h[ix0, iy0, izp]
                              + v2h[ixp, iyp, iz0] + v2h[ixp, iy0, izp]
                              + v2h[ix0, iyp, izp] + v2h[ixp, iyp, izp])/8;
    end #for v2
    v1h
 end #func


 @kernel function prolong_kernel!(v1h, v2h, nmesh_x, nmesh_y, nmesh_z)
    
    I = @index(Global, Cartesian)
    nmesh2_x = nmesh_x * 2
    nmesh2_y = nmesh_y * 2
    nmesh2_z = nmesh_z * 2

    ix0 = I[1]
    iy0 = I[2]
    iz0 = I[3]

    ixp = ix0 + 1 
    ixp = ixp > nmesh_x ? ixp - nmesh_x : ixp
    ixm = ix0 - 1 
    ixm = ixm < 1 ? ixm + nmesh_x : ixm

    iyp = iy0 + 1 
    iyp = iyp > nmesh_y ? iyp - nmesh_y : iyp
    iym = iy0 - 1 
    iym = iym < 1 ? iym + nmesh_y : iym

    izp = iz0 + 1 
    izp = izp > nmesh_z ? izp - nmesh_z : izp
    izm = iz0 - 1 
    izm = izm < 1 ? izm + nmesh_z : izm

    i2x0 = 2*ix0
    i2xp = i2x0 + 1
    i2xp = i2xp > nmesh2_x ? i2xp - nmesh2_x : i2xp

    i2y0 = 2*iy0
    i2yp = i2y0 + 1
    i2yp = i2yp > nmesh2_y ? i2yp - nmesh2_y : i2yp

    i2z0 = 2*iz0
    i2zp = i2z0 + 1
    i2zp = i2zp > nmesh2_z ? i2zp - nmesh2_z : i2zp

    v1h[i2x0, i2y0, i2z0] = v2h[I];
    v1h[i2xp, i2y0, i2z0] = (v2h[I] + v2h[ixp, iy0, iz0])/2;
    v1h[i2x0, i2yp, i2z0] = (v2h[I] + v2h[ix0, iyp, iz0])/2;
    v1h[i2x0, i2y0, i2zp] = (v2h[I] + v2h[ix0, iy0, izp])/2;
    v1h[i2xp, i2yp, i2z0] = (v2h[I] + v2h[ixp, iy0, iz0]
                            + v2h[ix0, iyp, iz0] + v2h[ixp, iyp, iz0])/4;
    v1h[i2x0, i2yp, i2zp] = (v2h[I] + v2h[ix0, iyp, iz0]
                            + v2h[ix0, iy0, izp] + v2h[ix0, iyp, izp])/4;
    v1h[i2xp, i2y0, i2zp] = (v2h[I] + v2h[ixp, iy0, iz0]
                            + v2h[ix0, iy0, izp] + v2h[ixp, iy0, izp])/4;
    v1h[i2xp, i2yp, i2zp] = (v2h[I] + v2h[ixp, iy0, iz0]
                            + v2h[ix0, iyp, iz0] + v2h[ix0, iy0, izp]
                            + v2h[ixp, iyp, iz0] + v2h[ixp, iy0, izp]
                            + v2h[ix0, iyp, izp] + v2h[ixp, iyp, izp])/8;

 end #func

function prolong!(v1h::CuArray{T,3}, v2h::CuArray{T,3}) where T <: AbstractFloat
    nmesh = size(v2h) 
    device = KernelAbstractions.get_device(v1h)
    kernel! = prolong_kernel!(device)
    ev = kernel!(v1h, v2h, nmesh..., ndrange = size(v2h))
    wait(ev)
    v1h
end #func

function reduce!(v2h::AbstractArray{T,3}, v1h::AbstractArray{T,3}) where T <: AbstractFloat
    
    
    nmesh = size(v1h)

    Base.Cartesian.@nexprs 3 i -> begin
        nmesh_i = nmesh[i]
    end
    
    #@inbounds Threads.@threads for I in CartesianIndices(v2h)
    @tturbo for _iz0 = axes(v2h,3), _iy0 = axes(v2h,2), _ix0 = axes(v2h,1)

    
        #_ix0, _iy0, _iz0 = Tuple(I)

        ix0 = _ix0 * 2
        iy0 = _iy0 * 2
        iz0 = _iz0 * 2
        
        ixp = ix0 + 1 
        ixp = ixp > nmesh_1 ? ixp - nmesh_1 : ixp
        ixm = ix0 - 1 
        ixm = ixm < 1 ? ixm + nmesh_1 : ixm

        iyp = iy0 + 1 
        iyp = iyp > nmesh_2 ? iyp - nmesh_2 : iyp
        iym = iy0 - 1 
        iym = iym < 1 ? iym + nmesh_2 : iym

        izp = iz0 + 1 
        izp = izp > nmesh_3 ? izp - nmesh_3 : izp
        izm = iz0 - 1 
        izm = izm < 1 ? izm + nmesh_3 : izm
        
        v2h[_ix0, _iy0, _iz0] = (8*v1h[ix0, iy0, iz0]+
                                          4*(v1h[ixp, iy0, iz0]+
                                          v1h[ixm, iy0, iz0]+
                                          v1h[ix0, iyp, iz0]+
                                          v1h[ix0, iym, iz0]+
                                          v1h[ix0, iy0, izp]+
                                          v1h[ix0, iy0, izm])+
                                          2*(v1h[ixp, iyp, iz0]+
                                          v1h[ixm, iyp, iz0]+
                                          v1h[ixp, iym, iz0]+
                                          v1h[ixm, iym, iz0]+
                                          v1h[ixp, iy0, izp]+
                                          v1h[ixm, iy0, izp]+
                                          v1h[ixp, iy0, izm]+
                                          v1h[ixm, iy0, izm]+
                                          v1h[ix0, iyp, izp]+
                                          v1h[ix0, iym, izp]+
                                          v1h[ix0, iyp, izm]+
                                          v1h[ix0, iym, izm])+
                                          v1h[ixp, iyp, izp]+
                                          v1h[ixm, iyp, izp]+
                                          v1h[ixp, iym, izp]+
                                          v1h[ixm, iym, izp]+
                                          v1h[ixp, iyp, izm]+
                                          v1h[ixm, iyp, izm]+
                                          v1h[ixp, iym, izm]+
                                          v1h[ixm, iym, izm])/64.0

    end #for v1h
    v2h
end #func

@kernel function reduce_kernel!(v2h, v1h, nmesh, nmesh2)
    
    I = @index(Global, Cartesian)

        

    ix0 = I[1] * 2
    iy0 = I[2] * 2
    iz0 = I[3] * 2

    ixp = ix0 + 1 
    ixp = ixp > nmesh[1] ? ixp - nmesh[1] : ixp
    ixm = ix0 - 1 
    ixm = ixm < 1 ? ixm + nmesh[1] : ixm

    iyp = iy0 + 1 
    iyp = iyp > nmesh[2] ? iyp - nmesh[2] : iyp
    iym = iy0 - 1 
    iym = iym < 1 ? iym + nmesh[2] : iym

    izp = iz0 + 1 
    izp = izp > nmesh[3] ? izp - nmesh[3] : izp
    izm = iz0 - 1 
    izm = izm < 1 ? izm + nmesh[3] : izm
    
    v2h[I] = (8*v1h[ix0, iy0, iz0]+
                4*(v1h[ixp, iy0, iz0]+
                v1h[ixm, iy0, iz0]+
                v1h[ix0, iyp, iz0]+
                v1h[ix0, iym, iz0]+
                v1h[ix0, iy0, izp]+
                v1h[ix0, iy0, izm])+
                2*(v1h[ixp, iyp, iz0]+
                v1h[ixm, iyp, iz0]+
                v1h[ixp, iym, iz0]+
                v1h[ixm, iym, iz0]+
                v1h[ixp, iy0, izp]+
                v1h[ixm, iy0, izp]+
                v1h[ixp, iy0, izm]+
                v1h[ixm, iy0, izm]+
                v1h[ix0, iyp, izp]+
                v1h[ix0, iym, izp]+
                v1h[ix0, iyp, izm]+
                v1h[ix0, iym, izm])+
                v1h[ixp, iyp, izp]+
                v1h[ixm, iyp, izp]+
                v1h[ixp, iym, izp]+
                v1h[ixm, iym, izp]+
                v1h[ixp, iyp, izm]+
                v1h[ixm, iyp, izm]+
                v1h[ixp, iym, izm]+
                v1h[ixm, iym, izm])/64.0


end #func
function reduce!(v2h::CuArray{T,3}, v1h::CuArray{T,3}) where T <: AbstractFloat
    
    nmesh2 = map(cu, size(v2h))
    nmesh = map(cu, size(v1h))
    device = KernelAbstractions.get_device(v1h)
    kernel! = reduce_kernel!(device)
    ev = kernel!(v2h, v1h, nmesh, nmesh2, ndrange = size(v2h))
    wait(ev)
    v2h
    
end #func


function vcycle!(v::AbstractArray{T,3}, f::AbstractArray{T,3}, box_size::SVector{3,T}, box_min::SVector{3,T}, β::T, damping_factor::T, niterations::Int; los = nothing) where T <: AbstractFloat

    x⃗ = x_vec(v, box_size, box_min)
    nmesh = size(v)
    jacobi!(v, f, x⃗, box_size, box_min, β, damping_factor, niterations; los = los)
    recurse = true
    for i in 1:3
        recurse &= (nmesh[i] > 4 && (nmesh[i] % 2 == 0))
    end #for
    if recurse

        r = similar(v)
        residual!(r, v, f, x⃗, box_size, box_min, β, damping_factor, niterations; los = los)
        #f2h = zeros(eltype(v), nmesh2...)
        f2h = zeros_halved_array(r)
        reduce!(f2h, r)
        r = nothing
        v2h = zero(f2h)
        vcycle!(v2h, f2h, box_size, box_min, β, damping_factor, niterations; los = los)
        f2h = nothing
        v1h = zero(v)
        prolong!(v1h, v2h)
        v2h = nothing
        #@inbounds @fastmath Threads.@threads for I in CartesianIndices(v)
        #    v[I] += v1h[I]
        #end #for
        @tturbo @. v += v1h
        v1h = nothing

    end #if recurse

    jacobi!(v, f, x⃗, box_size, box_min, β, damping_factor, niterations; los = los)

end #func

function vcycle!(v::CuArray{T,3}, f::CuArray{T,3}, box_size::SVector{3,T}, box_min::SVector{3,T}, β::T, damping_factor::T, niterations::Int; los = nothing) where T <: AbstractFloat

    x⃗ = x_vec(v, box_size, box_min)
    nmesh = size(v)
    jacobi!(v, f, x⃗, box_size, box_min, β, damping_factor, niterations; los = los)
    recurse = true
    for i in 1:3
        recurse &= (nmesh[i] > 4 && (nmesh[i] % 2 == 0))
    end #for
    if recurse

        r = similar(v)
        residual!(r, v, f, x⃗, box_size, box_min, β, damping_factor, niterations; los = los)
        #f2h = zeros(eltype(v), nmesh2...)
        f2h = zeros_halved_array(r)
        reduce!(f2h, r)
        r = nothing
        v2h = zero(f2h)
        vcycle!(v2h, f2h, box_size, box_min, β, damping_factor, niterations; los = los)
        f2h = nothing
        v1h = zero(v)
        prolong!(v1h, v2h)
        v2h = nothing
        @. v += v1h
        v1h = nothing

    end #if recurse

    jacobi!(v, f, x⃗, box_size, box_min, β, damping_factor, niterations; los = los)

end #func


function fmg(f1h::AbstractArray{T,3}, v1h::Union{AbstractArray{T,3}, Nothing}, box_size::SVector{3,T}, box_min::SVector{3,T}, β::T, jacobi_damping_factor::T, jacobi_niterations::Int, vcycle_niterations::Int; los = nothing) where T <: AbstractFloat

    nmesh = size(f1h)
    recurse = true
    for i in 1:3
        recurse &= (nmesh[i] > 4 && (nmesh[i] % 2 == 0))
    end #for
    if recurse
        f2h = zeros_halved_array(f1h)
        reduce!(f2h, f1h)
        v2h = fmg(f2h, nothing, box_size, box_min, β, jacobi_damping_factor, jacobi_niterations, vcycle_niterations; los = los)
        f2h = nothing
        if v1h == nothing
            v1h = zero(f1h)
        end #if
        prolong!(v1h, v2h)
        v2h = nothing
    else
        if v1h == nothing
            v1h = zero(f1h)
        end #if
    end #if recurse
    for i in 1:vcycle_niterations

        vcycle!(v1h, f1h, box_size, box_min, β, jacobi_damping_factor, jacobi_niterations; los = los)

    end #for
    
    v1h

end #func


function compute_displacements(ϕ::AbstractArray{T, 3}, data_x::AbstractVector{T}, data_y::AbstractVector{T}, data_z::AbstractVector{T}, recon::MultigridRecon) where T <: Real


    Ψ_interp = [similar(data_x) for _ in 1:3]
    #read_grad_cic!(Ψ_interp..., ϕ, data_x, data_y, data_z, recon.box_size, recon.box_min; wrap = true)
    fft_plan = recon.fft_plan
    ϕ_k = fft_plan * ϕ
    k⃗ = k_vec(ϕ, recon.box_size)
    ∂ᵢϕ = similar(ϕ)
    kᵢϕ = similar(ϕ_k)
    for i in 1:3
        @inbounds Threads.@threads for I in CartesianIndices(ϕ_k)
            kᵢϕ[I] = im * k⃗[i][I[i]] * ϕ_k[I]
        end #for
        ldiv!(∂ᵢϕ, fft_plan, kᵢϕ)
        read_cic!(Ψ_interp[i], ∂ᵢϕ, data_x, data_y, data_z, recon.box_size, recon.box_min)
    end #for
    Ψ_interp
end #func


@kernel function mg_gradient_kernel!(out_field, @Const(k⃗), field, i)
    I = @index(Global, Cartesian)
    out_field[I] = im * field[I] * k⃗[i][I[i]]
end #function

function compute_displacements(ϕ::CuArray{T, 3}, data_x::CuVector{T}, data_y::CuVector{T}, data_z::CuVector{T}, recon::MultigridRecon) where T <: Real


    Ψ_interp = [similar(data_x) for _ in 1:3]
    #read_grad_cic!(Ψ_interp..., ϕ, data_x, data_y, data_z, recon.box_size, recon.box_min; wrap = true)
    fft_plan = recon.fft_plan
    ϕ_k = fft_plan * ϕ
    k⃗ = map(cu, k_vec(ϕ, recon.box_size))
    ∂ᵢϕ = similar(ϕ)
    kᵢϕ = similar(ϕ_k)
    kernel! = mg_gradient_kernel!(KernelAbstractions.get_device(ϕ))
    for i in 1:3
        ev = kernel!(kᵢϕ, k⃗, ϕ_k, i, ndrange = size(ϕ_k))
        wait(ev)
        ldiv!(∂ᵢϕ, fft_plan, kᵢϕ)
        read_cic!(Ψ_interp[i], ∂ᵢϕ, data_x, data_y, data_z, recon.box_size, recon.box_min)
    end #for
    Ψ_interp
end #func