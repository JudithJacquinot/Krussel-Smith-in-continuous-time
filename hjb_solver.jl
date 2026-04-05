using SparseArrays
include("utils.jl")

function solve_hjb!(m::Model, vf::ValueFunction, plm::PLM, gen::InfinitesimalGenerator, pol::PolicyFunctions, opt::SolverOptions, prices::Prices)
    #A is the InfinitesimalGenerator of the wealth process, Λ is the transition matrix of the earnings process, L is the InfinitesimalGenerator of the TFP process

    w_grid=prices.w
    r_grid=prices.r

    #preallocation
    c_F = similar(vf.v);  c_B = similar(vf.v);  c_0 = similar(vf.v)  #consumption
    s_F = similar(vf.v);  s_B = similar(vf.v) #savings
    #H_F = similar(vf.v);  H_B = similar(vf.v);  H_0 = similar(vf.v)  #Hamiltonians
    elem_α = similar(vf.v); elem_ξ = similar(vf.v); elem_β = similar(vf.v) #elements of A
    I_F = falses(m.g.Na, m.g.Ny, m.g.Nz, m.g.Nk); I_B = falses(m.g.Na, m.g.Ny, m.g.Nz, m.g.Nk); I_0 = falses(m.g.Na, m.g.Ny, m.g.Nz, m.g.Nk) #flags
    
    iter_hjb = 0
    while iter_hjb <= opt.max_iter_hjb
        iter_hjb = iter_hjb + 1

        # Forward difference
        vf.dv_F[1:(end - 1),:,:,:] .= diff(vf.v; dims = 1) / m.g.da
        #fill the last point
        for iy in 1:m.g.Ny, iz in 1:m.g.Nz, ik in 1:m.g.Nk
            vf.dv_F[end, iy, iz, ik] = u_p(m.p.γ, w_grid[iz, ik]*m.g.y_grid[iy] + r_grid[iz, ik]*m.g.a_grid[end])
        end
        #c_F = u_p_inv(m, max.(vf.dv_F,1e-10))  # Ensure consumption is non-negative
        c_F=max.(vf.dv_F, 1e-10).^(-1/m.p.γ)
        for ia in 1:m.g.Na, iy in 1:m.g.Ny, iz in 1:m.g.Nz, ik in 1:m.g.Nk
            s_F[ia, iy, iz, ik] = w_grid[iz,ik]*m.g.y_grid[iy] + r_grid[iz, ik] *m.g.a_grid[ia] .- c_F[ia, iy, iz, ik] #savings
        end

        # Backward difference
        for iy in 1:m.g.Ny, iz in 1:m.g.Nz, ik in 1:m.g.Nk
            vf.dv_B[1, iy, iz, ik] = u_p(m.p.γ, w_grid[iz, ik]*m.g.y_grid[iy] + r_grid[iz, ik]*m.g.a_grid[1])
        end
        vf.dv_B[2:end,:,:,:] .= vf.dv_F[1:(end - 1), :, :,:]
        #c_B = u_p_inv(m, max.(vf.dv_B,1e-10))  # Ensure consumption is non-negative
        c_B=max.(vf.dv_B, 1e-10).^(-1/m.p.γ)
        for ia in 1:m.g.Na, iy in 1:m.g.Ny, iz in 1:m.g.Nz, ik in 1:m.g.Nk
            s_B[ia, iy, iz, ik] = w_grid[iz,ik]*m.g.y_grid[iy] + r_grid[iz, ik] *m.g.a_grid[ia] .- c_B[ia, iy, iz, ik] #savings
        end

        # Consumption at steady-state
        for ia in 1:m.g.Na, iy in 1:m.g.Ny, iz in 1:m.g.Nz, ik in 1:m.g.Nk
            c_0[ia, iy, iz, ik] = w_grid[iz,ik]*m.g.y_grid[iy] + r_grid[iz, ik] *m.g.a_grid[ia]
        end

        # Upwind scheme chooses forward or backward differences based on the sign of the drift    
        I_F = s_F .> 0          # Positive drift -> Forward difference
        I_B = s_B .< 0          # Negative drift -> Backward difference
        I_0 = 1 .- I_F .- I_B   # Steady state

        #define policy functions
        pol.c = @. c_F*I_F + c_B*I_B + c_0*I_0

        #utility (for RHS of HJB)
        #_u = u.(pol.c)
        _u=@. pol.c.^(1-m.p.γ)./(1-m.p.γ)

        #policy function for asset accumulation

        for ia in 1:m.g.Na, iy in 1:m.g.Ny, iz in 1:m.g.Nz, ik in 1:m.g.Nk
            pol.ȧ .=  w_grid[iz,ik]*m.g.y_grid[iy] + r_grid[iz, ik] *m.g.a_grid[ia] - pol.c[ia, iy, iz, ik]
        end

        # We now build the generator of the wealth process A_c. It is of size Na*Ny*Nz*Nk x Na*Ny*Nz*Nk.
        # In other terms, there is one A matrix for each combination of (y,z,k). We go step by step

        # Step :0

        # All building elements of the A_c matrix : these are of size Na x Ny x Nz x Nk. We will then build
        # the A_c matrix step by step with these elements
        elem_α = @. -min(s_B, 0)/m.g.da     # lower diagonal
        elem_ξ = @. max(s_F, 0)/m.g.da      # upper diagonal
        elem_β = @. - elem_α - elem_ξ       # main diagonal 

        #Initialise the different blocks of the A_c matrix
        A_blocks = Array{SparseMatrixCSC{Float64,Int64}}(undef, m.g.Ny, m.g.Nz, m.g.Nk)
        #Builds the blocks
        for iy in 1:m.g.Ny, iz in 1:m.g.Nz, ik in 1:m.g.Nk
            lowdiag=elem_α[2:end, iy, iz, ik]
            updiag=elem_ξ[1:(end - 1), iy, iz, ik]
            diag=elem_β[:, iy, iz, ik]
            #now we impose reflecting boundaries
            diag[1]+=elem_α[1, iy, iz, ik]    # Reflecting boundary at a_min
            diag[end]+=elem_ξ[end, iy, iz,ik] # Reflecting boundary at a_max
            #create this block
            A_blocks[iy,iz,ik] = spdiagm(-1 => lowdiag, 
                                 0 => diag, 
                                 1 => updiag)
        end

        gen.A_blocks=A_blocks

        gen.A_c=blockdiag(A_blocks...)
        
        # We now turn to building the generator of the perceived process for k


        # drift in K : this is given by the PLM : dk_t=h(k_t, z_t)dt
        μ_k=[plm_compute!(k, z, plm) for k in m.g.K_grid, z in m.g.z_grid]

        elem_α=@. -min(μ_k, 0)/m.g.dK     # lower diagonal
        elem_ξ=@. max(μ_k, 0)/m.g.dK      # upper diagonal
        elem_β=@. - elem_α - elem_ξ       # main diagonal
        #Initialise the different blocks of the A_k matrix
        A_k_blocks = Array{SparseMatrixCSC{Float64,Int64}}(undef, m.g.Nz)
        for iz in 1:m.g.Nz
            lowdiag=elem_α[2:end, iz]
            updiag=elem_ξ[1:(end - 1), iz]
            diag=elem_β[:, iz]
            #now we impose reflecting boundaries
            diag[1]+=elem_α[1, iz]    # Reflecting boundary at K_min
            diag[end]+=elem_ξ[end, iz] # Reflecting boundary at K_max
            #create this block
            A_k_blocks[iz] = spdiagm(-1 => lowdiag, 
                                 0 => diag, 
                                 1 => updiag)
        end
        
        gen.A_k=blockdiag(A_k_blocks...) #Nz.NK x Nz.NK


        #########Last step : combine all generators
        # Create sparse identity matrices of appropriate sizes
        I_Na = spdiagm(0 => ones(m.g.Na))  # Na × Na sparse identity matrix
        I_Nz = spdiagm(0 => ones(m.g.Nz))  # Nz × Nz sparse identity matrix 
        I_Nk = spdiagm(0 => ones(m.g.Nk))  # Nk × Nk sparse identity matrix
        I_Ny = spdiagm(0 => ones(m.g.Ny))  # Ny × Ny sparse identity matrix


        Λ_big = kron(I_Nk, kron(I_Nz, kron(sparse(gen.Λ), I_Na)))
        L_big = kron(I_Nk, kron(sparse(gen.L), kron(I_Ny, I_Na)))
        A_k_big = kron(sparse(gen.A_k), kron(I_Ny, I_Na))

        gen.A .= gen.A_c + Λ_big + L_big + A_k_big

        # Check if A matrix is proper
        maximum(abs.(sum(gen.A, dims = 2))) <= 1e-9 || @warn("Not all rows of A matrix sum to 0")
      

        # Implicit scheme
        B = (m.p.ρ + 1/opt.Δ)*SparseArrays.I - gen.A
        b = vec(_u + vf.v/opt.Δ)

        # Solve system of equations
        v_vec = B\b

        v_new = reshape(v_vec, m.g.Na, m.g.Ny, m.g.Nz, m.g.Nk)
        diff_hjb = maximum(abs.(v_new - vf.v))
        vf.v = copy(v_new)

        println("   HJB iteration #", iter_hjb, "; HJB difference = ", diff_hjb)

        if diff_hjb < opt.tol_hjb
            break
        end
    end
    return(vf, pol, gen)
end