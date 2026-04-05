using SparseArrays
using Statistics
include("utils.jl")

function regression!(m::Model, dist::Distribution, z_path, K_path)
#Il faudra revoir la structure distribution, je ne suis pas sûr qu'elle est définie correctement.

K_series=K_path[m.g.T0+1:end]  #Extract the relevant part of the K_path (after burn-in)


# Build regression dataset: keep observations t = 1..(Tlen-Δ) so we can have K_{t+Δ}
T=m.g.T1 - m.g.T0 #this is the length of the data set

logK = log.(K_series[1:T-1])
z    = z_path[m.g.T0+1:end-1]
logKxz = logK .* z
K_plus = K_series[2:T]
cst=ones(length(logK))#the constant

X = hcat(cst, z, logK, logKxz)   # N × 4 matrix: 

y=(log.(K_plus) .- logK)/m.g.dt        # N-vector: Δ log K

Φ=(X'X)\(X'y)  # OLS estimator

#Next we compute the R^2.
y_hat = X * Φ               # Predicted values
SS_tot = sum((y .- mean(y)).^2)      # Total sum of squares
SS_res = sum((y .- y_hat).^2)         # Residual sum of squares
R_squared = 1 - SS_res / SS_tot 

return  K_series, Φ, R_squared

end