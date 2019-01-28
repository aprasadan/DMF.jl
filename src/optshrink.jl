# Arvind Prasadan
# 2019 January
# DMF Package Code
# Optshrink Function

"""
	optshrink(Y, r)

Perform a rank-r denoising of a data matrix Y via the OptShrink algorithm, described in http://doi.org/10.1109/TIT.2014.2311661 (R R Nadakuditi, 2014)

# Arguments
- `Y`: 2D Array (Data Matrix), where Y = X + Noise, and we want to find X
- `r`: An estimate of the rank of X (Integral)

# Output
- `Xh`: An estimate of X
- `w`: The shrunken singular values; if r = rank(Y), w is all zero
"""
function optshrink(Y, r)

    r = round(abs(r[1])) # Sanity check rank
    r = (1 <= r) ? r : 1

    (m, n) = size(Y)
    r = minimum([r, m, n]) # ensure r <= min(m, n)
    if r == m || r == n
	return Y, zeros(r)
    end

    # U is [m, min(m, n)], s is min(m, n), V is [n, min(n, m)]
    (U, s, V) = svd(Y, full = false)

    sv = s[(r + 1):end] # tail singular values for noise estimation

    w = zeros(r)
    for k = 1:1:r
        (D, Dder) = D_transform_from_vector(s[k], sv, max(m, n) - r, min(m, n) - r)
        w[k] = -2.0 * D / Dder
    end

    w[isnan.(w)] .= 0.0

    Xh = U[:, 1:r] * Diagonal(w) * V[:, 1:r]'

    return Xh, w
end

"""
	 D_transform_from_vector(z, sn, m, n)

Find the D Transform of the singular value spectrum sn at a point z, for an m x n matrix.

# Arguments
- `z`: Point to evaluate transform
- `sn`: Vector of singular values
- `m`: Matrix dimension
- `n`: Matrix dimension

# Outputs
- `D`: D Transform 
- `D_der`: D Transform Derivative
"""
function D_transform_from_vector(z, sn, m, n)
    # sn is of length n <= m

    sm = [sn; zeros(m - n)] # m x 1

    inv_n = 1.0 ./ (z^2.0 .- sn.^2.0) # vector corresponding to diagonal
    inv_m = 1.0 ./ (z^2.0 .- sm.^2.0)

    D1 = (1.0 / n) * sum(z * inv_n)
    D2 = (1.0 / m) * sum(z * inv_m)

    D = D1 * D2 # eq (16a) in paper

    # derivative of D transform
    D1_der = (1.0 / n) * sum(-2.0 * z^2.0 .* inv_n.^2.0 + inv_n)
    D2_der = (1.0 / m) * sum(-2.0 * z^2.0 .* inv_m.^2.0 + inv_m)

    D_der = D1 * D2_der + D2 * D1_der # eq (16b) in paper

    return (D, D_der)
end

