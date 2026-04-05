include("utils.jl")


function simulate_KFE!(m::Model, gen::InfinitesimalGenerator, dist::Distribution, g0, z_path)

    Δt = m.g.dt

    #first step : create the different generators for each combination of and z and K. So we have Nz*Nk generators of size (Na*Ny)x(Na*Ny)


    A_c_yzk = Array{SparseMatrixCSC{Float64,Int}}(undef, m.g.Nz, m.g.Nk)
    for iz in 1:m.g.Nz, ik in 1:m.g.Nk
        A_c_yzk[iz,ik] = blockdiag( (gen.A_blocks[iy,iz,ik] for iy in 1:m.g.Ny)... )
    end
    #recast Λ into a big matrix of the appropriate size
    Q_on_y = kron(sparse(gen.Λ), spdiagm(0 => ones(m.g.Na)))

    #Initialise
    dist.g[1] = copy(g0) #store initial distribution
    T = length(z_path) #T=T1+T0 (includes burn-in period)
    K_path=zeros(T) 


    #g is a vector of size T filled with matrices of size Na x Ny.
    
    #now we build the simulation of the distribution
    for t in 1:T-1
        g=dist.g[t]     #get the distribution at time t
        g_a=sum(dist.g[t], dims=2)  #this is the marginal distribution of asset holdings : we sum over idiosyncratic states.
        K_path[t]=sum(g_a.* m.g.a_grid.*m.g.da)  # dot product between the marginal distribution and the asset grid
        (izd, izu, ωz) = bracket_and_weight(z_path[t], m.g.z_grid) # find the indices and weights for z
        (ikd, iku, ωK) = bracket_and_weight(K_path[t], m.g.K_grid) # find the indices and weights for K

        #Solve KFE on these grid points
        g_p_dd = kfe_step_corner!(vec(g), A_c_yzk[izd,ikd] + Q_on_y, Δt) #we select the right matrix, take the vectorized g and perform a KFE step.
        g_p_du = kfe_step_corner!(vec(g), A_c_yzk[izd,iku] + Q_on_y, Δt)
        g_p_ud = kfe_step_corner!(vec(g), A_c_yzk[izu,ikd] + Q_on_y, Δt)
        g_p_uu = kfe_step_corner!(vec(g), A_c_yzk[izu,iku] + Q_on_y, Δt)

        # Linearly interpolate to get g at (z_path[t], K_path[t])
        g_p_vec = ωz*ωK*g_p_uu .+ (1-ωz)*ωK*g_p_du .+ ωz*(1-ωK)*g_p_ud .+ (1-ωz)*(1-ωK)*g_p_dd
        g_p_vec ./= (sum(g_p_vec) * m.g.da) #normalize

        g_p= reshape(g_p_vec, m.g.Na, m.g.Ny)
        dist.g[t+1] = copy(g_p)  #store the distribution at time t+1
    end

    #fill the last point
    g=dist.g[T]     #get the distribution at time t
    g_a=sum(dist.g[T], dims=2)  #this is the marginal distribution of asset holdings : we sum over idiosyncratic states.
    K_path[T]=sum(g_a.* m.g.a_grid.*m.g.da)  # dot product between the marginal distribution and the asset grid
    return  dist, K_path
end

