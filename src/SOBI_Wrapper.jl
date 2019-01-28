# Arvind Prasadan
# 2019 January
# DMF Package
# SOBI Function (R) Wrapper

"""
	 SOBI_Wrapper(X, lags)

A wrapper function for the Second Order Blind Identification (SOBI) Source Separation algorithm. This is a wrapper that calls the function from R, and depends on the presence of an R install as well as the JADE package being installed.

# Arguments
- `X`: Data matrix, p dimensions/sensors x n samples
- `lags`: List of (Integral) lags for the algorithm to operate at

# Outputs
- `Gamma`: The unmixing matrix 
- `Unmixed_Normalized`: The recovered latent time series
"""
function SOBI_Wrapper(X, lags)
    
    n = size(X, 2)
    
    # Sanity check lags
    lags = abs.(Int.(round.(lags)))
    if 1 < length(lags)
        lags = [min(x, n - 1) for x in lags]
        lags = sort(unique(lags))
    end
    
    # Put Variables in R
    R"lags = $(lags)"
    R"X = $(X')"
    R"library(JADE)"
    
    # Run SOBI in R
    R_out = reval("SOBI_OUT = SOBI(X, lags, method = \"frjd\", eps = 1e-10, maxiter = 1000); SOBI_OUT")
    
    Gamma = convert(Array{Float64,2}, R_out[1])
    Unmixed_Normalized = convert(Array{Float64,2}, R_out[4])
    
    return Gamma, Unmixed_Normalized
end

    
