# Arvind Prasadan
# 2019 January
# DMF Package 
# Tests for functions in package

push!(LOAD_PATH, "../src/")
using DMF
using LinearAlgebra
using Statistics
using StatsBase
using Test

# gen_arma_sequence, no arma
tol = 1e-2
n = 10000
ar = []
ma = []
arma_std = 1.0
trials = 200
running_sum = 0.0
for tr = 1:1:trials
    x = gen_arma_sequence(n, ar, ma, arma_std)
    global running_sum += std(x) / trials
end
@test isapprox(arma_std, running_sum; atol = tol)

# gen_arma_sequence, AR(1)
tol = 1e-2
n = 10000
ar = 0.5
ma = []
arma_std = 1.0
trials = 200
running_sum = 0.0
for tr = 1:1:trials
    x = gen_arma_sequence(n, ar, ma, arma_std)
    global running_sum += std(x) / trials
end
@test isapprox(arma_std / sqrt(1.0 - ar^2.0), running_sum; atol = tol)

# gen_cos_sequence, element
tol = 1e-8
n = 1000
f = 2.0
fs = 1.0
phase = 0.1
x, t = gen_cos_sequence(n, f, fs, phase)
@test isapprox(x[2], cos((f / fs) + phase); atol = tol)
@test isapprox(t[2], 1.0 / fs; atol = tol)

# autocorrelation, standard
tol = 1e-8
x = [1; 1; 1; 1]
gamma = autocorrelation(x; circular = false, bias_correct = false)
@test isapprox(gamma[1], 0.0; atol = tol)
@test isapprox(gamma[2], 0.0; atol = tol)

# autocorrelation, bias correction
tol = 1e-8
x = [1; 1; 1; 1]
gamma = autocorrelation(x; circular = false, bias_correct = true)
@test isapprox(gamma[1], 0.0; atol = tol)
@test isapprox(gamma[2], 0.0; atol = tol)

# autocorrelation, circular
tol = 1e-8
x = [1; 1; 1; 1]
gamma = autocorrelation(x; circular = true, bias_correct = true)
@test isapprox(gamma[1], 0.0; atol = tol)
@test isapprox(gamma[2], 0.0; atol = tol)

# autocorr_mat
tol = 1e-2
p = 5
n = 10000
trials = 200
running_sum = zeros(1, p, p)
for tr = 1:1:trials
    global running_sum += autocorr_mat(randn(p, n), 1)[1] / trials
end
@test isapprox(running_sum[1, :, :], zeros(p, p); atol = tol)

# eigenvector_error
tol = 1e-8
p = 10
Q = randn(p, 2)
Q_hat = [Q[:, 2] Q[:, 1] randn(p)]
@test isapprox(eigenvector_error(Q, Q_hat), 0.0; atol = tol)

# For Source Separation Tests

p = 10
k = 2
n = 1000
Q = qr(randn(p, k)).Q
Q = Q[:, 1:k]
S = [gen_cos_sequence(n, 2.0)[1] gen_cos_sequence(n, 0.25)[1]]
S = mapslices(normalize, S; dims = 1)
D = Diagonal(1:k)
X = Q * D * S'

# SOBI_Wrapper
tol = 1e-3
U, s, V = svd(X, full = false)
S_hat = SOBI_Wrapper(U[:, 1:k]' * X, 1)[2]
@test isapprox(eigenvector_error(S, S_hat), 0.0; atol = tol)

# DMF: Standard
tol = 1e-3
w, Q_hat, C_hat, A = dmf(X; tsvd = false, os = false, hil = false, nsv = k, C_nsv = k, demean = false, lag = 1)
@test isapprox(eigenvector_error(Q, Q_hat), 0.0; atol = tol)
@test isapprox(eigenvector_error(S, C_hat), 0.0; atol = tol)
@test size(C_hat, 2) == k

# DMF: Hilbert
tol = 1e-3
w, Q_hat, C_hat, A = dmf(X; tsvd = false, os = false, hil = true, nsv = k, C_nsv = k, demean = false, lag = 1)
@test isapprox(eigenvector_error(Q, Q_hat), 0.0; atol = tol)
@test isapprox(eigenvector_error(S, C_hat), 0.0; atol = tol)
@test size(C_hat, 2) == k

# DMF: tSVD
tol = 1e-3
w, Q_hat, C_hat, A = dmf(X; tsvd = true, os = false, hil = false, nsv = k, C_nsv = k, demean = false, lag = 1)
@test isapprox(eigenvector_error(Q, Q_hat), 0.0; atol = tol)
@test isapprox(eigenvector_error(S, C_hat), 0.0; atol = tol)
@test size(C_hat, 2) == k

# DMF: OptShrink
tol = 1e-3
w, Q_hat, C_hat, A = dmf(X; tsvd = false, os = true, hil = false, nsv = k, C_nsv = k, demean = false, lag = 1)
@test isapprox(eigenvector_error(Q, Q_hat), 0.0; atol = tol)
@test isapprox(eigenvector_error(S, C_hat), 0.0; atol = tol)
@test size(C_hat, 2) == k

# DMF: De-Mean
tol = 1e-3
w, Q_hat, C_hat, A = dmf(X; tsvd = false, os = false, hil = false, nsv = k, C_nsv = k, demean = true, lag = 1)
@test isapprox(eigenvector_error(Q, Q_hat), 0.0; atol = tol)
@test isapprox(eigenvector_error(S, C_hat), 0.0; atol = tol)
@test size(C_hat, 2) == k


# OptShrink, rank = p or n
tol = 1e-8
p = 10
n = 100
X = randn(p, n)
Xh, w = optshrink(X, p)
@test isapprox(X, Xh; atol = tol)
Xh, w = optshrink(X, n)
@test isapprox(X, Xh; atol = tol)
Xh, w = optshrink(X, n + p)
@test isapprox(X, Xh; atol = tol)

# OptShrink
tol = 1e-8
X = ones(p, n)
Xh, w = optshrink(X, 1)
@test isapprox(w[1], norm(ones(p * n), 2.0); atol = tol)
@test isapprox(X, Xh; atol = tol)

