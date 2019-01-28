# Arvind Prasadan
# 2019 January
# DMF Package
# Finds the squared error between two sets of eigenvectors, accounting for permutations

"""
	eigenvector_error(Q, Q_hat)

Finds the squared error 
``\\|q_i - p_i \\widehat{q}_i\\|_2^2 = q_i^T q_i + \\widehat{q}_i^T \\widehat{q}_i - 2 p_i q_i^T\\widehat{q}_i = 2 [1 - |q_i^T \\widehat{q}_i|]`` for each q_i

# Arguments
- `Q`: The true vectors (columns)
- `Q_hat`: The estimated vectors (columns)

# Output
- Error
"""
function eigenvector_error(Q, Q_hat)

    if size(Q, 1) != size(Q_hat, 1)
       return Inf
    end

    Q = mapslices(normalize, Q; dims = 1)
    Q_hat = mapslices(normalize, Q_hat; dims = 1)
    k = size(Q, 2)
    
    return abs(2.0 * (k - norm(mapslices(maximum, abs.(Q_hat' * Q); dims = 1), 2)^2.0))
end

