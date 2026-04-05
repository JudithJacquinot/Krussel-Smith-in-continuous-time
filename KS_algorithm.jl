function KS_algorithm(m::Model, opt::SolverOptions)

    # Step 0 : Initialize value function, policy functions, PLM, generator, prices
    vf = ValueFunction{4}(v=zeros(m.g.Na, m.g.Ny, m.g.Nz, m.g.Nk),
        dv_F=zeros(m.g.Na, m.g.Ny, m.g.Nz, m.g.Nk),
        dv_B=zeros(m.g.Na, m.g.Ny, m.g.Nz, m.g.Nk))

    
    pol = PolicyFunctions(
        c = zeros(m.g.Na, m.g.Ny, m.g.Nz, m.g.Nk),
        ȧ = zeros(m.g.Na, m.g.Ny, m.g.Nz, m.g.Nk)
    )
    
    plm=PLM(coeffs = [0.1, 0.01, 0.1, 0.01]) #initial coefficients

    gen = InfinitesimalGenerator(m.g)  # uses the constructor that sets Λ appropriately
    gen.A = spzeros(m.g.Na*m.g.Ny*m.g.Nz*m.g.Nk, m.g.Na*m.g.Ny*m.g.Nz*m.g.Nk)
    
    #now we create the price grids
    r_grid=[m.p.α*exp(z)*(k/m.g.y_bar)^(m.p.α - 1) - m.p.δ for z in m.g.z_grid, k in m.g.K_grid]
    w_grid = [(1 - m.p.α)*exp(z)*(k/m.g.y_bar)^m.p.α for z in m.g.z_grid, k in m.g.K_grid]
    prices = Prices(r = r_grid, w = w_grid)

    #good intialisation of the value function : present value of consuming everything at each state

    for ia in 1:m.g.Na, iy in 1:m.g.Ny, iz in 1:m.g.Nz, ik in 1:m.g.Nk
        c_init = w_grid[iz, ik]*m.g.y_grid[iy] + r_grid[iz, ik]*m.g.a_grid[ia]
        vf.v[ia, iy, iz, ik] = u(m.p.γ, c_init) / m.p.ρ
    end

    #store one path of the aggregate z shock

    z_path = OU_simulate!(0.0, m, m.g.T1, m.g.T0)

    #Initialise K_path

    K_path=zeros(m.g.T1)
    K_perceived=zeros(m.g.T1)


    # Now the distribution

    dist=Distribution(g=[zeros(m.g.Na, m.g.Ny) for t in 1:(m.g.T1)]) 

    #We need to create and store the initial distribution :
    dist.g[1] = zeros(m.g.Na, m.g.Ny) #Initialise
    #all mass around the bottom of the asset grid : we ensure that K0 is within the grid
    dist.g[1][2,:] = [0.5, 0.5]
    dist.g[1][3,:] = [1, 1]
    dist.g[1][4,:] = [0.01, 0.01]
    dist.g[1] ./= sum(vec(dist.g[1])) * m.g.da #normalize

    # prepare the loop

    error=1
    accuracy_dh = 1
    iter = 0
    R_squared=0 #this is just to store out outside of the loop

    # convergence loop

    while iter < opt.max_iter_ks && error > opt.tol_ks
        iter += 1
        
        # Step 1: Solve HJB to get policy functions
        (vf, pol, gen) = solve_hjb!(m, vf, plm, gen, pol, opt, prices)
    
        # Step 2: Simulate KFE to get distribution evolution
        dist, K_path = simulate_KFE!(m, gen, dist, dist.g[1], z_path)

        # Step 3: regression to update PLM
        K_series, Φ, R_squared = regression!(m, dist, z_path, K_path)
        #K_series is K_path after the burn-in period

        error = maximum(abs.(plm.coeffs - Φ))


        coeff_new = opt.ν*plm.coeffs + (1 - opt.ν)*Φ
        plm.coeffs = coeff_new

        # Step 4: Store alternative accuracy measure : k_perceived-K_path
        K_perceived=simulate_K_perceived(plm, K_series[1], z_path, m.g.dt)
        accuracy_dh = maximum(abs.(K_perceived[m.g.T0+1:end] - K_series))

        println("KS Iteration: $iter, PLM coeffs: $(plm.coeffs), estimated PLM coeffs : $(Φ), R²: $R_squared, Max coeff change: $error, Max K diff: $accuracy_dh")
    end
    
    return dist, plm, R_squared, K_path, z_path, K_perceived, vf, pol

end

