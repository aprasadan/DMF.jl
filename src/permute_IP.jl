# Arvind Prasadan
# 2019 July
# DMF Package

"""
    permute_IP(Q::Matrix, Q_hat::Matrix)

# Inputs
- Q::Matrix: true vectors
- Q_hat::Matrix: estimate of Q

# Outputs
- Vector of 'best' inner products
"""
function permute_IP(Q, Q_hat)
    k = size(Q, 2)
    
    @assert(k == size(Q_hat, 2), "Q and Q_hat have differing numbers of vectors")
    @assert(size(Q, 1) == size(Q_hat, 1), "Q and Q_hat are not compatible")
    
    # Normalize for unit norm
    Q = mapslices(normalize, Q; dims = 1)
    Q_hat = mapslices(normalize, Q_hat; dims = 1)
    
    # Generate permutations
    pt = permutations(1:k) |> collect
    
    # Find maximum sum of inner products 
    pt_idx = argmax([sum(diag(abs.(Q' * Q_hat[:, x]))) for x in pt])
    pt = pt[pt_idx]
    
    return diag(Q' * Q_hat[:, pt])
end

