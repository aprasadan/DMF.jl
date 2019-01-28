# Arvind Prasadan
# 2019 January
# DMF Package
# Evaluates the Autocorrelation Function

"""
	autocorrelation(x; circular = false, bias_correct = false)

Evaluates the Autocorrelation Function of a signal 

# Arguments
- `x`: signal (vector)
- `circular`: Boolean whether to use the circular inner product or not
- `bias_correct`: Boolean whether to use n - k instead of n in normalization for a lag of k (theoretically undesirable); ignored if circular is true

# Outputs
- `gamma`: Autocorrelation function (lags 0 to n - 1)
"""

function autocorrelation(x; circular = false, bias_correct = false)

    x = x[:]
    n = length(x)
    
    x = x .- mean(x)
    
    gamma = zeros(n)
    if false == circular
        for k = 0:1:(n - 1)
            gamma[k + 1] = x[1:1:(n - k)]' * x[(1 + k):1:n]
            
            if true == bias_correct
                gamma[k + 1] = gamma[k + 1] / (n - k)
            end
        end
        
        if false == bias_correct
            gamma = gamma / n
        end
    elseif true == circular
        for k = 0:1:(n - 1)
            gamma[k + 1] = circshift(x, k)' * x
        end
        
        gamma = gamma / n
    end
    
    return gamma 
end

