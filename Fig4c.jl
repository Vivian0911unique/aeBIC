using StatsBase
using Distributions, LinearAlgebra, StaticArrays,Random
using Plots
# Calculate cross entropy between two distributions
function cross_entropy(p, q)
    eps = 1e-14
    P = max.(eps,p)
    P = P/sum(P)
    Q = max.(eps,q)
    Q = Q/sum(Q)
    return -sum(P .* log.(Q))
end

# Calculate telegraph distributions via finite state projection
function CME_maturemar!(p)
    ρ,σon,σoff,N=p
    d=1
    C = zeros(2*N,2*N)
    C[1:N,1:N] = - diagm(0 => σon*ones(N)) - diagm(0 => d*collect(0:N-1)) + diagm(1 => d*collect(1:N-1))
    C[1:N,N+1:2*N] = diagm(0 => σoff*ones(N)) 
    C[N+1:2*N,1:N] = diagm(0 => σon*ones(N)) 
    C[N+1:2*N,N+1:2*N] = - diagm(0 => σoff*ones(N))  - diagm(0 => ρ*vcat(ones(N-1),0)) + diagm(-1 => ρ*ones(N-1))- diagm(0 => d*collect(0:N-1)) + diagm(1 => d*collect(1:N-1))
    C[1,:].=1 
    pp=C\[1;zeros(2*N-1)]
    pm1=pp[N+1:2*N]+pp[1:N]

    return pm1
end

# ρ=15 in Fig. 4
ρ = 15

# Truncation in FSP
N = 300

# Range of f_on
edge1 = collect(0.01:0.01:1)

# Range of N_σ
edge2 = 10 .^collect(-1:0.02:3.1)


cro_tele = zeros(length(edge1),length(edge2))
cro_nb = zeros(length(edge1),length(edge2))
cro_pois = zeros(length(edge1),length(edge2))
for i in 1:length(edge1) 
    for j in 1:length(edge2)
        fon = edge1[i]
        Nσ = edge2[j]

        # Calculate entropy for telegraph distribution
        σon =fon*Nσ
        σoff = (1-fon)*Nσ
        P = CME_maturemar!((ρ,σon,σoff,N))
        cro_tele[i,j] = cross_entropy(P, P)

        # Calculate cross entropy for negative binomial distribution (method of moments)
        r1 = (1+σoff+σon)*σon/σoff
        b1 = ρ*σoff/((1+σoff+σon)*(σoff+σon))
        p1 = 1/(b1+1)
        R1 = pdf.(NegativeBinomial(r1,p1),0:N-1)
        cro_nb[i,j] = cross_entropy(P, R1) 

        # Calculate cross entropy for Poisson distribution
        m = ρ*σon/(σon+σoff)
        Q = pdf.(Poisson(m),0:N-1)
        cro_pois[i,j] = cross_entropy(P, Q) 
    end
end


using Distributions,StatsPlots

# Number of cells
nc=10000

# aeBIC for each model
bic_pois =  2*nc.*cro_pois.+log(nc) * 1
bic_nb = 2*nc.*cro_nb.+log(nc) * 2
bic_tele = 2*nc.*cro_tele.+log(nc) * 3

# Model selection based on aeBIC
bic1 = zeros(length(edge1),length(edge2))
for i in 1:length(edge1)
    for j in 1:length(edge2)
        bic_values = [bic_pois[i,j],bic_nb[i,j],bic_tele[i,j]]
        bic1[i,j] = findmin(bic_values)[2] 
    end
end
bic1
contourf(log10.(edge2),edge1[1:end],bic1,color=:viridis,levels=2, linewidth=0,xlabel="log10(Nσ)",ylabel="fon",title="aeBIC_select_nc=$(nc)")