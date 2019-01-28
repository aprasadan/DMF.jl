# Arvind Prasadan
# 2019 January
# DMF Package
# SOBI Function (R) Wrapper

"""
	 SOBI_Wrapper(X, lags)

A wrapper function for the Second Order Blind Identification (SOBI) Source Separation algorithm. This is a wrapper that calls the function from R, and depends on the presence of an R install as well as the jointDiag package being installed.

# Arguments
- `X`: Data matrix, p dimensions/sensors x n samples
- `lags`: List of (Integral) lags for the algorithm to operate at

# Outputs
- `Q_hat`: Eigenvector Estimates
- `S_hat: The recovered latent time series
"""
function SOBI_Wrapper(X, lags)
    
    n = size(X, 2)
    p = size(X, 1)
    
    # Sanity check lags
    lags = abs.(Int.(round.(lags)))
    if 1 < length(lags)
        lags = [min(x, n - 1) for x in lags]
        lags = sort(unique(lags))
    end
    
    # Whiten X
    mu = mean(X; dims = 2)
    X = X .- mu
    cov_X = X * X'
    cov_X = (cov_X + cov_X') / 2.0
    cW, cV = eigen(cov_X)
    cov_invsqrt = cV * Diagonal(sqrt.(abs.(pinv(cW)))) * cV'
    cov_sqrt = cV * Diagonal(sqrt.(abs.(cW))) * cV'
    Xw = cov_invsqrt * X

    # Store matrices at different lags
    A = zeros(p, p, length(lags))
    for ll = 1:1:length(lags)
	    A[:, :, ll] = Xw[:, (lags[ll] + 1):end] * pinv(Xw[:, 1:(end - lags[ll])]) + Diagonal(ones(p)) # Regularizer
    end

    # Put Variables in R
    R"library(jointDiag)"
    R"A = $(A)"

    # Run SOBI: Joint Diagonalization
    Q_hat = reval("ajd(A, method = \"jedi\")\$A")
    Q_hat = convert(Array{Float64, 2}, Q_hat)  
    Q_hat = cov_sqrt * Q_hat
    
    S_hat = (pinv(Q_hat) * (X .+ mu))'

    return (Q_hat, S_hat)
end

    
