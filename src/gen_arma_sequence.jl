# Arvind Prasadan
# 2019 January
# DMF Package
# Generates a realization of an ARMA process

"""
	gen_arma_sequence(n = 100, ar_comp = [], ma_comp = [], arma_std = 1.0)

Generates a realization of an ARMA process

# Arguments
- `n`: Length of process
- `ar_comp`: List of AR Coefficients; one of this and ma_comp must be non-empty
- `ma_comp`: List of MA Coefficients; one of this and ar_comp must be non-empty
- `arma_std`: Noise Standard Deviation

# Outputs
- `x`: vector of ARMA process realization
"""
function gen_arma_sequence(n = 100, ar_comp = [], ma_comp = [], arma_std = 1.0)
    
    n = round(abs(n[1])) # Samples
    n = (1 <= n) ? n : 1
    
    arma_std = abs(arma_std[1]) # Standard Deviation
    
    if 0 == length(ar_comp) + length(ma_comp)
	x = arma_std * randn(n)
    else
    	# Define ARMA object, simulate from this set of parameters
    	arma = ARMA(ar_comp, ma_comp, arma_std)   
    	x = simulation(arma, ts_length = n)
    end
    
    return x
end

