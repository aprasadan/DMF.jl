# Arvind Prasadan
# 2019 January
# DMF Package
# Generates a realization of an cosine sequence

"""
	gen_cos_sequence(n = 100, f = 1.0, fs = 1.0)

Generates a realization of an cosine sequence

# Arguments
- `n`: Length of sequence
- `f`: Frequency
- `fs`: Sampling frequency
- `phase`: Phase shift

# Outputs
- `x`: vector of cosine process realization
- `t`: Time vector
"""
function gen_cos_sequence(n = 100, f = 1.0, fs = 1.0, phase = 0.0)

    n = round(abs(n[1])) # Samples
    n = (1 <= n) ? n : 1
    
    f = abs(f[1]) # Frequency 
    fs = abs(fs[1]) # Sampling Frequency

    phase = phase[1] # Phase shift

    t = collect(0.0:(1.0 / fs):((n - 1) / fs)) # Time vector
    
    x = cos.(f * t .+ phase) # Cosines: time signature
    
    return x, t
end

