using Distributed
addprocs(10)
@everywhere using LinearAlgebra, Distributions, StaticArrays,StatsBase,Random
@everywhere using DelimitedFiles,Optim
# Calculate cross entropy between two distributions
@everywhere function cross_entropy(p, q)
    eps = 1e-14
    P = max.(eps,p)
    P = P/sum(P)
    Q = max.(eps,q)
    Q = Q/sum(Q)
    return -sum(P .* log.(Q))
end
# Calculate telegraph distributions via finite state projection
@everywhere function CME_maturemar!(p)
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

# Calculate the cross entropy between real distribution and standard Poisson
@everywhere function cross_inf_po(ps,P)
    Q = pdf.(Poisson(ps[1]),0:N-1)
    return cross_entropy(P,Q)
end

@everywhere function cross_bic_po(i,g)
    p = vcat(vec(readdlm("beta_value/beta60140_fon$(i)_Ns$(g).csv",',')),zeros(N-50))
    P = p./sum(p)
    init_ps = [1.0]
    results_po = optimize(ps->cross_inf_po(exp.(ps),P),init_ps,Optim.Options(show_trace=false,g_tol=1e-9,iterations = 500)).minimizer
    ps_po=exp.(results_po)
    Q =  pdf.(Poisson(ps_po[1]),0:N-1)
    return cross_entropy(P, Q)
end

# Calculate the cross entropy between real distribution and standard Negative Binomial
@everywhere function cross_inf_nb(ps,P)
    r1,b1 =  ps
    Q = pdf.(NegativeBinomial(r1,1/(b1+1)),0:N-1)
    return cross_entropy(P,Q)
end

@everywhere function cross_bic_nb(i,g)
    p = vcat(vec(readdlm("beta_value/beta60140_fon$(i)_Ns$(g).csv",',')),zeros(N-50))
    P = p./sum(p)
    init_ps = [1.0;1.0]
    results_nb = optimize(ps->cross_inf_nb(exp.(ps),P),init_ps,Optim.Options(show_trace=false,g_tol=1e-9,iterations = 500)).minimizer
    ps_nb=exp.(results_nb)
    r2 = ps_nb[1]
    b2 = ps_nb[2]
    Q = pdf.(NegativeBinomial(r2,1/(b2+1)),0:N-1)
    return cross_entropy(P, Q)
end

# Calculate the cross entropy between real distribution and standard telegraph
@everywhere function cross_inf_tele(ps,P)
    ρ,σon,σoff =ps
    Q = CME_maturemar!((ρ,σon,σoff,N))
    return cross_entropy(P,Q)
end

@everywhere function cross_bic_tele(i,g)
    p = vcat(vec(readdlm("beta_value/beta60140_fon$(i)_Ns$(g).csv",',')),zeros(N-50))
    P = p./sum(p)
    init_ps = [1.0;1.0;1.0]
    results_tele = optimize(ps->cross_inf_tele(exp.(ps),P),init_ps,Optim.Options(show_trace=false,g_tol=1e-9,iterations = 500)).minimizer
    ps_tele=exp.(results_tele)
    Q =  CME_maturemar!((ps_tele[1],ps_tele[2],ps_tele[3],N))
    return cross_entropy(P, Q)
end

@everywhere N=100

edge1 = collect(0.01:0.01:1)
edge2 = 10 .^collect(-1:0.02:3.1)
cro_tele = zeros(length(edge1),length(edge2))
cro_nb = zeros(length(edge1),length(edge2))
cro_pois = zeros(length(edge1),length(edge2))
for i in 1:length(edge1) 
    cro_pois[i,:] = pmap(g->cross_bic_po(i,g),1:length(edge2))
    cro_nb[i,:] = pmap(g->cross_bic_nb(i,g),1:length(edge2))
    cro_tele[i,:] = pmap(g->cross_bic_tele(i,g),1:length(edge2))
    println(i)
end

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

using StatsPlots
p=contourf(log10.(edge2),edge1[1:end],bic1,color=:viridis,levels=2, linewidth=0,xlabel="log10(Nσ)",ylabel="fon",title="beta(60,140)_nc=$(nc)")
