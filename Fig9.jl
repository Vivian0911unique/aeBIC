using KernelDensity, FastGaussQuadrature, Interpolations,ForwardDiff
using Distributed
#using OptimalTransport,TaylorSeries,Distributions,Distances
using DelimitedFiles,StatsBase
using Distributed,Plots
using Distributions, Random, LinearAlgebra, StaticArrays
#using Plots,Optim
using LinearAlgebra, Distributions, SparseArrays
rd = readdlm("ss3_n682_fibs_umiCounts.csv",',')[2:end,2:end]

cell_props = sum(.!iszero.(rd), dims=1)[:] ./ 30025


rdt = rd[:,findall(cell_props .> 0.3)]

row_props = sum(.!iszero.(rdt), dims=2)[:] ./ size(rdt,2)

# 找出比例 > 0.9 的行号
rows = findall(row_props .> 0.01)

rdt = rdt[rows,:]

nc = 670
ng = size(rdt,1)
#rdt = readdlm("BIC_fon_counts_3200.csv",',') 
tc = sum(rdt,dims=1)
ntc = vec(tc./mean(tc))
β = vec(ntc)

# β Gauss Quadrature
Density = kde(β)
interp = LinearInterpolation(Density.x, Density.density, extrapolation_bc=Interpolations.Flat())
min_β = minimum(β)
max_β = maximum(β)
xl, wl = gausslegendre(17)
Xl = (max_β-min_β)/2*xl.+(max_β+min_β)/2
pf = interp.(Xl)
sum(wl.*pf*(max_β-min_β)/2)
PF = (pf.*wl*(max_β-min_β)/2)./sum(wl.*pf*(max_β-min_β)/2)

addprocs(6)
@everywhere using Turing, LinearAlgebra, Distributions, SparseArrays, Statistics,StatsBase,Random
@everywhere using HypergeometricFunctions,Optim
@everywhere using DelimitedFiles
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

@everywhere function conv_function_delay(pt,Xl,PF,u_counts)
    ρ,σon,σoff =pt
    N=100
    N = Int(maximum(u_counts) < 200 ? 210 : maximum(u_counts) + 50)
    pro = zeros(N)
    
    for i in 1:length(Xl) 
        ll = CME_maturemar!((ρ*Xl[i],σon,σoff,N))*PF[i]
        pro = pro+ll
    end
    
    return pro
    #return prob

end

@everywhere function sum_pro_telegraph_v(pt,u_counts,Xl,PF)
    prob= conv_function_delay(pt,Xl,PF,u_counts)
    sum_prob=0
    for j in 1:length(u_counts)
        a=log(prob[Int(u_counts[j]+1)]+1e-14)
        sum_prob=a+sum_prob
    end
    return -sum_prob
end


@everywhere function optimize_tele_v(u_counts,Xl,PF)
    init_ps = [1.0;1.0;1.0]
    results_FSP = optimize(ps->sum_pro_telegraph_v(exp.(ps),u_counts,Xl,PF),init_ps,Optim.Options(show_trace=false,g_tol=1e-20,iterations = 1000)).minimizer
    ps_FSP=exp.(results_FSP)
    return ps_FSP
end


@everywhere function sum_pro_bursty_v(pt,u_counts,Xl,PF)
    r1,b1 =pt
    #N=300
    N = Int(maximum(u_counts) < 200 ? 210 : maximum(u_counts) + 50)
    U = [pdf.(NegativeBinomial(r1,1/(b1*Xl[i]+1)),0:N-1) for i in eachindex(Xl)]
    prob= mean(U,weights(PF))
    sum_prob=0
    for j in 1:length(u_counts)
        a=log(prob[u_counts[j]+1]+1e-14)
        sum_prob=a+sum_prob
    end
    return -sum_prob
end

@everywhere function optimize_bursty_v(u_counts,Xl,PF)
    init_ps = [1.0;1.0]
    results_FSP = optimize(ps->sum_pro_bursty_v(exp.(ps),u_counts,Xl,PF),init_ps,Optim.Options(show_trace=false,g_tol=1e-20,iterations = 1000)).minimizer
    ps_FSP=exp.(results_FSP)
    return ps_FSP
end

@everywhere function sum_pro_poisson_v(u_counts,Xl,PF)
    #ρ,σon,σoff =pt
    #N=300
    m =mean(u_counts)
    N = Int(maximum(u_counts) < 200 ? 210 : maximum(u_counts) + 50)
    U = [pdf.(Poisson(m*Xl[i]),0:N-1) for i in eachindex(Xl)]
    prob= mean(U,weights(PF))
    sum_prob=0
    for j in 1:length(u_counts)
        a=log(prob[u_counts[j]+1]+1e-14)
        sum_prob=a+sum_prob
    end
    return -sum_prob
end


@everywhere function model_select_aic(u_counts,g,Xl,PF)
    nc=670
    #p = vec(readdlm("value_1_i$(i)_j$(g).csv",','))
    #P = CME_maturemar!((ρ,σon,σoff,N))
    #P = p./sum(p)
    #u_counts = rand(Categorical(P),nc).-1
    # 存储AIC值
    params_nb = zeros(2)
    params_tele = zeros(3)
    aic_values=zeros(3)
        # 计算每个分布的AIC值
    for i in 1:3
        if i == 1
        # 拟合自定义模型
           #params_poisson = mean(u_counts)#optimize_poisson(u_counts)#
            log_likelihood =-sum_pro_poisson_v(u_counts,Xl,PF)        
            k = 1  # 参数数量
        elseif i == 2
            # 使用fit_mle拟合泊松分布
            params_nb = optimize_bursty_v(u_counts,Xl,PF)
            log_likelihood =-sum_pro_bursty_v(params_nb,u_counts,Xl,PF)        
            k = 2  # 参数数量=#
        else
        # 拟合标准概率分布
            params_tele = optimize_tele_v(u_counts,Xl,PF)
            log_likelihood =-sum_pro_telegraph_v(params_tele,u_counts,Xl,PF)        
            k = 3  # 参数数量
        end
        # 计算AIC值
        aic = log(nc) * k - 2 * log_likelihood
        aic_values[i] = aic
        #println(aic_values) 
    end
    best_fit = findmin(aic_values)[2]
    #=if abs((aic_values[1]-aic_values[2])/aic_values[2])<0.01
        best_fit=1
    end=#
    #return aic_values
    writedlm("data/model_qc_g$(g).csv",vcat(best_fit,params_nb,params_tele),',')
    #return vcat(best_fit,params_nb,params_tele)#best_fit
end

t = pmap(g->model_select_aic(Int64.(rdt[g,:]),g,Xl,PF),1:6)



function conv_function_delay_turing(pt,Xl,PF,u_counts)
    ρ,σon,σoff =pt

    N=100
    N = Int(maximum(u_counts) < 200 ? 210 : maximum(u_counts) + 50)
    dist = Beta(σon, σoff)
    x, w = gausslegendre(17)     # xl∈[−1,1], wl 对应权重
    a, b = 0.0, 1.0                # Beta 分布的自然支撑
    X = (b - a)/2 .* x .+ (a + b)/2
    pdf_vals = pdf.(dist, X)
    wts = pdf_vals .* w .* ((b - a)/2)
    PFB = wts ./ sum(wts)

    prob = reduce(
      +,
      (PFB[i] * PF[j]) .* pdf.(Poisson(ρ * X[i] * Xl[j]), 0:N-1)
      for i in eachindex(X), j in eachindex(Xl)
    )
    return prob

end

@model function tele_model(u_counts, Xl, PF)
    # 宽先验，保证后验 ≈ 似然
    ρ   ~ Gamma(2,5)
    σon ~ Gamma(2,5)
    σoff  ~ Gamma(2,5)

    pt = (ρ, σon, σoff)
    prob = conv_function_delay_turing(pt, Xl, PF, u_counts)

    # 手动累加对数似然
    for (i, uc) in enumerate(u_counts)
        Turing.@addlogprob! log(prob[Int(uc + 1)])
    end
end

u_counts = rdt[1,:]

chain = sample(tele_model(u_counts, Xl, PF), NUTS(), 1000)





