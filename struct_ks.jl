
using Parameters, SparseArrays


const N = 4  # This is the dimensionnality of our problem

@with_kw struct Params
    ρ::Float64 = 0.05           # discount rate
    γ::Float64 = 2.0            # risk aversion
    α::Float64 = 0.35           # capital share
    δ::Float64 = 0.1            # depreciation rate
    A::Float64 = 1.0            # steady state TFP level (\neq mean since log normal)
end

@with_kw struct Grids
    # Asset Grid
    a_min::Float64 = 0   # borrowing constraint
    a_max::Float64 = 100.0
    Na::Int = 71
    a_grid::Vector{Float64} = range(a_min, a_max, Na)
    da = (a_max - a_min) / (Na - 1)
    
    # TFP Grid (common noise)- OU process in logs 
    # dlog(z_t)=\eta(̄z-z_t)dt+σdW_t with ̄z=0
    σ=0.01
    #σ=0.01
    η=0.5
    #η=0.02
    Nz = 21                                          
    z_bar=0                                              #uncond mean of the process
    σz²=σ^2/(2*η)                                    #uncond variance of the process
    # we cover 99 % of the distribution for z
    z_min = z_bar - 2.32*sqrt(σz²/(2*η))                   # min value of z grid
    z_max = z_bar + 2.32*sqrt(σz²/(2*η))                   # max value of z grid
    z_grid::Vector{Float64} = range(z_min, z_max, Nz)    # z grid
    dz = (z_max - z_min) / (Nz - 1)                      # step size in z grid    
    μz = @. η * (z_bar - z_grid)                         # drift term in z
    Z_grid::Vector{Float64} = exp.(z_grid)               # productivity grid in levels

    #Efficiency units grid (cross sectionnal idiosyncratic noise)
    λ1=1/2
    #λ1=1/3
    λ2=1/3
    λ::Vector{Float64}=[λ1, λ2]
    Ny=2
    y1=0.93
    y2=1+(1-y1)*λ2/λ1
    y_grid::Vector{Float64}=[y1, y2]
    y_bar= (y_grid[1]*λ[2] + y_grid[2]*λ[1])/sum(λ) # Average income (i.e. total labor supply)

    # Aggregate capital grid 
    K_min::Float64 = 3.5
    K_max::Float64 = 3.9
    Nk=31
    #Nk=21
    K_grid::Vector{Float64} = range(K_min, K_max, Nk)
    dK = (K_max - K_min) / (Nk - 1)


    #Time grid
    dt=0.1
    T0=1000          #burn-in period
    T1=10000+T0

end

@with_kw struct Model
    p::Params = Params()
    g::Grids = Grids()
end

abstract type HouseholdObject{N} end

@with_kw mutable struct ValueFunction{N} <: HouseholdObject{N}
    v::Array{Float64, N}
    dv_F::Array{Float64, N} #derivative forward wrt A
    dv_B::Array{Float64, N} #derivative backward wrt A
end

@with_kw mutable struct PolicyFunctions{N} <: HouseholdObject{N}
    c::Array{Float64, N}
    ȧ::Array{Float64, N}
end

@with_kw mutable struct PLM 
    coeffs::Vector{Float64}=zeros(4) #vector of coefficients for the plm
end

@with_kw mutable struct Distribution
    g::Vector{Matrix{Float64}} #vector of  size T filled with matrices of size Na x Ny.
end

@with_kw mutable struct InfinitesimalGenerator
    A::SparseMatrixCSC{Float64,Int} = spzeros(0,0)
    Λ::Matrix{Float64} = zeros(2,2)
    L::Matrix{Float64} = spzeros(0,0)
    A_blocks::Array{SparseMatrixCSC{Float64,Int}}=spzeros(0,0)
    A_c::SparseMatrixCSC{Float64,Int64}=spzeros(0,0)
    A_k::SparseMatrixCSC{Float64,Int64}=spzeros(0,0)
end

# Then define the constructor that uses Grids
function InfinitesimalGenerator(g::Grids)
    #Build \Lambda : easy
    Λ = [-g.λ[1] g.λ[1];
         g.λ[2] -g.λ[2]]

    #Build L : infinitesimal generator of the OU process in logs

    elem_α=[-min(-g.η*z, 0)/g.dz+1/2*g.σ^2/g.dz^2 for z in g.z_grid]
    elem_β=[min(-g.η*z, 0)/g.dz-max(-g.η*z, 0)/g.dz-g.σ^2/g.dz^2 for z in g.z_grid]
    elem_ξ=[max(-g.η*z, 0)/g.dz+1/2*g.σ^2/g.dz^2 for z in g.z_grid]

    L = spdiagm(
    -1 => elem_α[2:end], 
     0 => elem_β,         
     1 => elem_ξ[1:end-1] 
    )

    # Reflecting boundaries
    L[1,1] += elem_α[1]   # Reflecting boundary at z_min
    L[end,end] += elem_ξ[end] # Reflecting boundary at z_max
    return InfinitesimalGenerator(A=spzeros(0,0), Λ=Λ, L=L)
end

@with_kw struct SolverOptions
    max_iter_hjb::Int = 1000 #maximum number of iteration for HJB
    tol_hjb::Float64 = 1e-6 
    Δ::Float64 = 1000 #we use the implicit method so we can take it to be large
    max_iter_ks::Int = 1000 #maximum number of step for the aggregation step of the algorithm
    tol_ks::Float64 = 1e-6
    ν=0.3 #dampening paremeter for the plm update
end

@with_kw struct Prices
    r::Matrix{Float64}
    w::Matrix{Float64}
end

@with_kw struct Solution
    m::Model
    vf::ValueFunction{N}
    pf::PolicyFunctions{N}
    plm::PLM
    gen::InfinitesimalGenerator
    dist::Distribution
    prices::Prices
    opt::SolverOptions = SolverOptions()
end

