using StatsBase
using Distributions, LinearAlgebra, StaticArrays,Random
using Plots

# p_cap ~ Dirac(0.3)
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

# ρ=15*0.3 in Fig. 8a
ρ = 15*0.3

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
contourf(log10.(edge2),edge1[1:end],bic1,color=:viridis,levels=2, linewidth=0,xlabel="log10(Nσ)",ylabel="fon",title="aeBIC_select_+dirac(0.3)_nc=$(nc)")

# Find parameter set whose optimal model is negative binomial
select_nb = count(x->x==2,bic1) 
positions = []
for i in 1:length(edge1) 
    for j in 1:length(edge2)
        if bic1[i,j]==2
            push!(positions,[edge1[i],edge2[j]])
        end
    end
end

positions = Matrix(hcat(positions...)')
tele_bfbs = zeros(select_nb,2) 
effnb_bfbs = zeros(select_nb,2) 
for i in 1 : select_nb
    fon = positions[i,1]
    Nσ = positions[i,2]

    σon = fon*Nσ
    σoff = (1-fon)*Nσ

    # Ground truth of burst size and burst frequency
    tele_bfbs[i,1] = σon
    tele_bfbs[i,2] = ρ/σoff

    # Estimation of burst size and burst frequency
    effnb_bfbs[i,1] = (1+σoff+σon)*σon/σoff
    effnb_bfbs[i,2] = ρ*σoff/((1+σoff+σon)*(σoff+σon))  ####计算每个点所对应的burst frequency和 burst size
end

# Number of scatter dots
n = Int(5000)
bf_xy = zeros(n,2) ####存储每一个burst frequency的(r_true,r_estimate)
bs_xy = zeros(n,2)
for i in 1 : n
    rows = randperm(size(positions, 1))[1:2]  #####随机生成两个不重复的数，作为用来比较的两个点在tele_bfbs和effnb_bfbs的位置

    # Ensure r_true <= 1 (burst frequency)
    if tele_bfbs[rows[1],1]/tele_bfbs[rows[2],1]<=1  
        # Calculate r_true   
        bf_xy[i,1] = (tele_bfbs[rows[1],1]/tele_bfbs[rows[2],1])
        # Calculate r_estimate
        bf_xy[i,2] = (effnb_bfbs[rows[1],1]/effnb_bfbs[rows[2],1]) 
    else 
        # Calculate r_true 
        bf_xy[i,1] = (tele_bfbs[rows[2],1]/tele_bfbs[rows[1],1])
        # Calculate r_estimate
        bf_xy[i,2] = (effnb_bfbs[rows[2],1]/effnb_bfbs[rows[1],1])
    end

    # Ensure r_true <= 1 (burst size)
    if tele_bfbs[rows[1],2]/tele_bfbs[rows[2],2]<=1
        # Calculate r_true 
        bs_xy[i,1] = (tele_bfbs[rows[1],2]/tele_bfbs[rows[2],2])
        # Calculate r_estimate
        bs_xy[i,2] = (effnb_bfbs[rows[1],2]/effnb_bfbs[rows[2],2])
    else 
        # Calculate r_true 
        bs_xy[i,1] = (tele_bfbs[rows[2],2]/tele_bfbs[rows[1],2])
        # Calculate r_estimate
        bs_xy[i,2] = (effnb_bfbs[rows[2],2]/effnb_bfbs[rows[1],2])
    end
end

# Rank flipped
bf_rf = bf_xy[findall(x->x>=1,bf_xy[:,2]),:]
# Rank unchanged (distance decreased)
bf_rudd = bf_xy[setdiff(collect(1:n), union(findall(x->x<=1,bf_xy[:,2]./bf_xy[:,1]),findall(x->x>=1,bf_xy[:,2]))),:] 
# Rank unchanged (distance increased)
bf_rudi = bf_xy[findall(x->x<=1,bf_xy[:,2]./bf_xy[:,1]),:] 

# Rank flipped
bs_rf = bs_xy[findall(x->x>=1,bs_xy[:,2]),:] 
# Rank unchanged (distance decreased)
bs_rudd = bs_xy[setdiff(collect(1:n), union(findall(x->x<=1,bs_xy[:,2]./bs_xy[:,1]),findall(x->x>=1,bs_xy[:,2]))),:] 
# Rank unchanged (distance increased)
bs_rudi = bs_xy[findall(x->x<=1,bs_xy[:,2]./bs_xy[:,1]),:]  

p1 = scatter(bf_rf[:,1],bf_rf[:,2],label="Rank flipped")
scatter!(bf_rudd[:,1],bf_rudd[:,2],label="Rank unchanged (distance decreased)")
scatter!(bf_rudi[:,1],bf_rudi[:,2],xlims=(0,1),ylims=(0,3),label="Rank unchanged (distance increased)",title="burst frequency",xlabel="r_true",ylabel="r_estimate")

p2 = scatter(bs_rf[:,1],bs_rf[:,2],label="Rank flipped")
scatter!(bs_rudd[:,1],bs_rudd[:,2],label="Rank unchanged (distance decreased)")
scatter!(bs_rudi[:,1],bs_rudi[:,2],xlims=(0,1),ylims=(0,3),label="Rank unchanged (distance increased)",title="burst size",xlabel="r_true",ylabel="r_estimate")

plot(p1,p2)