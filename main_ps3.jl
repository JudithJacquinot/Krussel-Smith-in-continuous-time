using Parameters
using SparseArrays
using LinearAlgebra
using Statistics

include("struct_ks.jl")
include("utils.jl")
include("hjb_solver.jl")
include("kfe.jl")
include("regression.jl")
include("KS_algorithm.jl")


#Define the model and store par
m=Model()
opt=SolverOptions()


dist, plm, R_squared, K_path, z_path, K_perceived, vf, pol = KS_algorithm(m, opt)

using Plots

p_1 = plot(
    title = "",
    xlabel = "Time",
    ylabel = "Capital stock",
    legend = :topright
)

plot!(p_1, K_perceived[m.g.T0+1:end], label="Perceived Path", color=:blue, linestyle=:dash, lw=2)
plot!(p_1, K_path[m.g.T0+1:end], label="Distribution Path", color=:red)

# Save the plot
mkpath("plots")
savefig(p_1, "plots/aggregate_capital_paths.png")



# Sample average capital from K_path over [T0+1, T1]
sample_avg_K = mean(K_path[m.g.T0+1:m.g.T1])
println("Sample average capital from K_path (T0+1..T1): ", sample_avg_K)

# Compute maximum difference between K_perceived and K_path over the sample
max_diff_K = maximum(abs.(K_perceived[m.g.T0+1:m.g.T1] .- K_path[m.g.T0+1:m.g.T1])./K_path[m.g.T0+1:m.g.T1])
println("Maximum percentage difference between K_perceived and K_path (T0+1..T1): ", max_diff_K)

# Plot the consumption policy function at the sample mean of K and for different values of z
#for that we interpolate the sample mean of K on the grid K_grid.

(ikd, iku, ωK) = bracket_and_weight(sample_avg_K, m.g.K_grid)  # find the indices and weights
c_pol_meank=ωK*pol.c[:,:,:,iku] .+ (1-ωK)*pol.c[:,:,:,ikd]

# Compute consumption policy at min and max K over the sample
min_K = minimum(K_path[m.g.T0+1:m.g.T1])
max_K = maximum(K_path[m.g.T0+1:m.g.T1])

(ikd_min, iku_min, ωK_min) = bracket_and_weight(min_K, m.g.K_grid)
c_pol_mink = ωK_min*pol.c[:,:,:,iku_min] .+ (1-ωK_min)*pol.c[:,:,:,ikd_min]

(ikd_max, iku_max, ωK_max) = bracket_and_weight(max_K, m.g.K_grid)
c_pol_maxk = ωK_max*pol.c[:,:,:,iku_max] .+ (1-ωK_max)*pol.c[:,:,:,ikd_max]

# Compute time-averaged cross-sectional distribution from T0 to T1
avg_dist = zeros(m.g.Na, m.g.Ny)
for t in (m.g.T0+1):m.g.T1
    avg_dist .+= dist.g[t]
end
avg_dist ./= (m.g.T1 - m.g.T0)

# Find the time indices where K is at its minimum and maximum
t_min_K = argmin(K_path[m.g.T0+1:m.g.T1]) + m.g.T0
t_max_K = argmax(K_path[m.g.T0+1:m.g.T1]) + m.g.T0

# Get distributions at min and max K
dist_mink = dist.g[t_min_K]
dist_maxk = dist.g[t_max_K]

# Tracé combiné: légende unique pour y1/y2, distribution sur axe droit
iz = floor(Int, m.g.Nz/2)
#iz=m.g.Nz
colors = [:blue, :darkred]

p = plot(
    title = "",
    xlabel = "Assets",
    ylabel = "Consumption",
    legend = :bottomright,
    right_margin = 10Plots.mm,   # espace pour l'axe droit
)

# Consumption (left axis) — keep the legend here
plot!(p, m.g.a_grid, c_pol_mink[:, 1, iz],      color=colors[1], label="y1, K=min", linestyle=:dot, alpha=0.4)
plot!(p, m.g.a_grid, c_pol_meank[:, 1, iz],      color=colors[1], label="y1, K=mean", alpha=0.7)
plot!(p, m.g.a_grid, c_pol_maxk[:, 1, iz],      color=colors[1], label="y1, K=max", linestyle=:dashdot)
plot!(p, m.g.a_grid, c_pol_mink[:, m.g.Ny, iz], color=colors[2], label="y2, K=min", linestyle=:dot, alpha=0.4)
plot!(p, m.g.a_grid, c_pol_meank[:, m.g.Ny, iz], color=colors[2], label="y2, K=mean", alpha=0.7)
plot!(p, m.g.a_grid, c_pol_maxk[:, m.g.Ny, iz], color=colors[2], label="y2, K=max", linestyle=:dashdot)

# Distribution (right axis) — no labels for a single legend
pr = twinx(p)
plot!(pr, m.g.a_grid, dist_mink[:, 1],           color=colors[1], label="", ylabel="Density", linestyle=:dot, alpha=0.4)
plot!(pr, m.g.a_grid, avg_dist[:, 1],            color=colors[1], label="", linestyle=:dash, alpha=0.7)
plot!(pr, m.g.a_grid, dist_maxk[:, 1],           color=colors[1], label="", linestyle=:dashdot)
plot!(pr, m.g.a_grid, dist_mink[:, m.g.Ny],      color=colors[2], label="", linestyle=:dot, alpha=0.4)
plot!(pr, m.g.a_grid, avg_dist[:, m.g.Ny],       color=colors[2], label="", linestyle=:dash, alpha=0.7)
plot!(pr, m.g.a_grid, dist_maxk[:, m.g.Ny],      color=colors[2], label="", linestyle=:dashdot)

display(p)

savefig(p, "plots/consumption_distribution.png")




