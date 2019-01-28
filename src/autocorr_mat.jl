# Arvind Prasadan
# 2019 January
# DMF Package
# Creates an Autocorrelation Matrix at a given lag

"""
	autocorr_mat(X, lags = 1)

Creates an Autocorrelation Matrix at a set of given lags.

# Arguments
- `X`: Data matrix: p dimensions/sensors x n samples
- `lags`: list of lags (Integral)

# Outputs
- `R`: Data cube of autocorrelation matrices: # lags x p x p
- `lags`: Lags used
"""
function autocorr_mat(X, lags = 1)
    
    p, n = size(X)
    
    # Subtract Mean
    mu = mean(X; dims = 2)
    X = X .- mu
    
    # Get Standard Deviations
    sigma = std(X; dims = 2)
    
    # Sanity check lags
    lags = abs.(Int.(round.(lags)))
    if 1 == length(lags)
        lags = [lags]
    else
        lags = [min(x, n - 1) for x in lags]
        lags = sort(unique(lags))
    end
    
    R_unshaped = autocor(X', lags)
    R = zeros(length(lags), p, p)
    for l = 1:1:length(lags)
        R[l, :, :] = hcat([circshift(R_unshaped[l, :], t)[:] for t in 0:1:(p - 1)]...)
    end
    
    return R, lags
end

