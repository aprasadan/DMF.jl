# Arvind Prasadan
# 2019 January
# DMF Package 
# dmf Function (Main)

"""
	dmf(X; tsvd = false, os = false, hil = false, nsv = 2, C_nsv = minimum(size(X)), demean = false, lag = 1)

	Perform the Dynamic Mode Factorization (DMF) Algorithm on data. This is an extension of the Dynamic Mode Decomposition (DMD) Algorithm. Factors X = Q (C + 1 mu^T)^T, Q is mode matrix (unit norm columns), C is matrix of latent time series, mu is mean vector

# Depends on: optshrink.jl

# Arguments
- `X`: Data matrix, columns are samples and rows are variables
- `tsvd`: Boolean flag to use the truncated SVD as a denoising step
- `os`: Boolean flag to use the OptShrink Algorithm as a denoising step; overrides tsvd
- `hil`: Boolean flag to perform the Hilbert transform (after denoising) on the data
- `nsv`: Denoising rank (Integer)
- `C_nsv`: Estimate of the latent rank of X (number of time series) (Integer)
- `demean`: Boolean flag to subtract mean from data
- `lag`: Lag to use in the DMD algorithm (Integer); 1 is classical DMD

# Output
- `w`: DMD eigenvalues (vector) sorted by decreasing magnitude
- `Q`: DMD eigenvectors (modes), paired with w (matrix)
- `C`: Latent time series
- `A`: DMD Matrix (for eigendecomposition)
"""
function dmf(X; tsvd = false, os = false, hil = false, nsv = 2, C_nsv = minimum(size(X)), demean = false, lag = 1)

    # Sanity check inputs: Ranks cannot be more than dimensions, same for lag
    nsv = round(abs(nsv[1])) 
    nsv = (1 <= nsv) ? nsv : 1
    nsv = min(nsv, minimum(size(X)))
    C_nsv = round(abs(C_nsv[1])) 
    C_nsv = (1 <= C_nsv) ? C_nsv : 1
    C_nsv = min(C_nsv, minimum(size(X)))
    lag = round(abs(lag[1])) 
    lag = (1 <= lag) ? lag : 1
    lag = min(lag, size(X, 2) - 1)

    # Denoising
    if true == os
	Xt = optshrink(X, nsv)[1]
    elseif true == tsvd
        U, s, V = svd(X, full = false) 
        Xt = U[:, 1:nsv] * Diagonal(s[1:nsv]) * V[:, 1:nsv]'
    else
        Xt = X
    end
    
    # Mean
    if true == demean
        mu = mean(Xt; dims = 2)
        Xt = Xt .- mu
    else
        mu = zeros(size(Xt, 1))
    end
    
    if true == hil
        Xt = mapslices(hilbert, Xt; dims = 2)
    end
           
    A = Xt[:, (lag + 1):end] * pinv(Xt[:, 1:(end - lag)])
    w, Q = eigen(A)
    sorted_idx = sortperm(abs.(w), rev = true)
    Q = Q[:, sorted_idx]
    w = w[sorted_idx]
    C  = (pinv(Q[:, 1:1:C_nsv]) * (Xt .+ mu))'
    
    if true == hil
        Q = real.(Q)
        C = real.(C)
    end

    return (w, Q, C, A)
end

