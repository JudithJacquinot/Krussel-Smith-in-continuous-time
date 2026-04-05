

function u(γ, c)
    u=(c^(1-γ)-1)/(1-γ)
    return u
end

function u_p(γ, c)
    u_p=c^(-γ)
    return u_p
end

function u_p_inv(γ, c)
    u_p_inv.=c^(-1/γ)
    return u_p_inv
end


function plm_compute!(K, z, plm::PLM)
    #K is in levels, z in logs
    θ0=plm.coeffs[1]
    θ1=plm.coeffs[2]
    θ2=plm.coeffs[3]
    θ3=plm.coeffs[4]
    h=K*(θ0 .+ θ1 .* z .+ θ2 .* log(K) .+ θ3 .* log(K) .* z)
    return h
end

function OU_simulate!(z0, m::Model, T, T0)
    z_path = zeros(T)
    z_path[1] = z0
    for t in 2:T
        ε = randn()
        z_path[t] = z_path[t-1] + m.g.η * (m.g.z_bar - z_path[t-1]) * m.g.dt + m.g.σ * sqrt(m.g.dt) * ε
    end
    return z_path
end



function simulate_K_perceived(plm::PLM, K0::Float64, z_path::Vector{Float64}, Δt::Float64)
    T = length(z_path)
    K_perceived = zeros(T)
    K = K0
    for t in 1:T
        # 1. perceived drift
        h = plm_compute!(K, z_path[t], plm)
        # 2. Update K
        K = K + Δt * h     
        K_perceived[t] = K
    end
    return K_perceived
end



function kfe_step_corner!(g::Vector{Float64}, A_corner::SparseMatrixCSC{Float64,Int}, Δt::Float64)
    M = I - Δt * transpose(A_corner)  # Construct the appropriate matrix M
    g_p = M \ g                        # Solve M*g_{t+Δt} = g_t
                                    
    @. g_p = max(g_p, 0.0)           # Ensure non-negativity
end


function bracket_and_weight(x::Float64, grid::AbstractVector{Float64})
    i_up = clamp(searchsortedfirst(grid, x), 2, length(grid))  # si x trop petit, le clamp commence à 2, et si x trop grand, on le bloque à la deernière case valide
    i_low = i_up - 1  #on sait que x entre grid(i-low] et grid[i_up]
    x_low, x_up = grid[i_low], grid[i_up]
    ω = (x - x_low) / (x_up - x_low)
    return i_low, i_up, ω
end