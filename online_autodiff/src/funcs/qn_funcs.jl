
# Get arrival/departures to Cr set
function getDataOverCr(paths_class, Cr)
    ta_cr = Vector{Vector{Float64}}(undef, MCsims)
    td_cr = Vector{Vector{Float64}}(undef, MCsims)
    ind_cr = Vector{Any}(undef, MCsims)
    for i = 1:MCsims
        ta_cr[i], td_cr[i], _ = getArrivalDeparture(paths_class[i], Cr)
        _, ind_cr[i] = splitByT(ta_cr[i],[0; simSecEnd])
    end
    return ta_cr, td_cr, ind_cr
end

## Functions for obtaining and manipulating the queue lengths

# Retrieve the queue lengths from arrivals/departures
function getQueueLengths(ta::Vector{Float64}, td::Vector{Float64};
        start_time=0, end_time=0)

    @assert length(ta) > 0
    @assert size(ta) == size(td)
    @assert all(ta .<= td)

    if end_time == 0 
        end_time = max(maximum(ta), maximum(td))
    end

    if end_time < minimum(ta) || start_time > maximum(td)
        if start_time < end_time
            return [start_time 0.0; end_time 0.0]
        end
        return [start_time 0.0; start_time 0.0]
    end

    m = length(td)

    q_diff = vcat(hcat(ta, ones(m)), hcat(td, -1*ones(m)))
    sort_idx = sortperm(q_diff[:, 1])
    q_diff = q_diff[sort_idx, :]

    q = -1*ones(2*m + 1, 2)
    q[1, :] = [0.0 0.0]

    k = 1
    i = 2
    while k <= size(q_diff, 1)

        if q_diff[k, 1] > end_time
            break
        end

        k_s = k + 1
        while k_s <= size(q_diff, 1) && q_diff[k, 1] == q_diff[k_s, 1]
            k_s += 1
        end
        change = sum(q_diff[k:k_s-1, 2])

        if change != 0
            q[i, 1] = q_diff[k, 1]
            q[i, 2] = q[i-1, 2] + change
            i += 1
        end

        k = k_s
    end

    q = q[findall(q[:, 1] .!= -1.0), :]

    if q[end, 1] != end_time
        q = [q; [end_time q[end, 2]]]
    end

    start_idx = findfirst(start_time .< (q[:, 1]))
    q = q[(start_idx-1):end, :]
    q[1, 1] = start_time 

    @assert all(q[:, 1] .>= 0)
    @assert all(q[:, 2] .>= 0)
    @assert all(diff(q[:, 1]) .>= 0)
    @assert length(unique(q[:, 1])) == size(q, 1)

    return q
end

# Retrieve queue lengths if arrivals/departures are DateTime objects
function getQueueLengths(ta::Vector{DateTime}, td::Vector{DateTime}; start_time=0)
    q = getQueueLengths(datetime2unix.(ta), datetime2unix.(td), start_time=start_time)
    return hcat(unix2datetime.(q[:,1]), q[:, 2])
end

# Calculate the mean queue length by weighting each queue length with its duration
function getQueueLengthAvg(q::Matrix{Float64})
    if sum(abs.(diff(q[:, 1]))) == 0
        return 0.0
    end
    return mean(q[1:end-1, 2], weights(diff(q[:, 1])))
end

# Get the average queue length over all MCsims at the given times
function getQueueLengthAvgMC(q::Vector{Matrix{Float64}}, t::Array{Float64, 1})
    q_mc = zeros(length(q), length(t))
    for k = 1:length(q)
        q_mc[k, :] = getQueueLengthAt(q[k], t)
    end
    return mean(q_mc, dims=1)[:]
end

# Get the average queue lengths over classes and queues in each step bin over all 
# MCsims, can also yield the utilization
function getQueueLengthAvgSteps(q::Matrix{Matrix{Float64}}, steps::Vector{Float64}; 
    K=[], D=[])
    (m, n) = size(q)
    q_steps_mc = Vector{Vector{Float64}}(undef, n)
    util_mc = Vector{Vector{Float64}}(undef, n)

    for i = 1:n
        ind_sims = [stepIndicesSorted(q[sim, i][:, 1], steps) for sim = 1:m]
        q_steps = [getQueueLengthAt(q[sim, i], steps) for sim = 1:m]
        q_steps_mc[i] = zeros(length(steps))
        util_mc[i] = zeros(length(steps))
        for k = 2:length(steps)
            qi = [[q_steps[sim][k-1]; q[sim, i][ind_sims[sim][k], 2]] for sim = 1:MCsims]
            ti = [[steps[k-1]; q[sim, i][ind_sims[sim][k], 1]; steps[k]] for sim = 1:MCsims]
            q_steps_mc[i][k] = mean(vcat(qi...), Weights(vcat(diff.(ti)...)))
            if length(K) == length(D) > 0 && D[i] == "PS"
                util_mc[i][k] = mean([getUtil(hcat(ti[sim], 
                    [qi[sim]; qi[sim][end]]), K[i]) for sim = 1:MCsims])
            end
        end
    end
    return q_steps_mc, util_mc
end

# Add queue lengths from a vector into a single queue length
function addQueueLengths(q_v::Vector{Matrix{Float64}})

    q_v_start = [q[1, :] for q in q_v]
    q_v_diff = [[q[2:end, 1] diff(q[:, 2])] for q in q_v]

    q_tot = -1*ones(sum(size.(q_v, 1)) - length(q_v) + 1, 2)
    q_tot[1, 1] = minimum(x -> x[1], q_v_start)
    q_tot[1, 2] = sum(x -> x[2], q_v_start)

    end_time = maximum(x -> x[1], [q[end, :] for q in q_v])

    q_diff = sortslices(vcat(q_v_diff...), dims=1)

    k = 1
    i = 2
    while k <= size(q_diff, 1)
        
        # find all similar timestamps
        k_s = k + 1
        while k_s <= size(q_diff, 1) && q_diff[k, 1] == q_diff[k_s, 1]
            k_s += 1
        end
        change = sum(q_diff[k:k_s-1, 2])

        if change != 0 || q_diff[k, 1] == end_time
            q_tot[i, 1] = q_diff[k, 1]
            q_tot[i, 2] = q_tot[i-1, 2] + change
            i += 1
        end
        k = k_s
    end

    q_tot = q_tot[findall(q_tot[:, 1] .!= -1.0), :]

    @assert all(q_tot[:, 1] .>= 0)
    @assert all(q_tot[:, 2] .>= 0)
    @assert all(diff(q_tot[:, 1]) .>= 0)

    return q_tot
end

# Retrieve the queue length at the given times
function getQueueLengthAt(q::Matrix{Float64}, t::Vector{T}) where T <: Number

    function re_sort(q, idx)
        q_t = zeros(size(q))
        q_t[idx_s] = q
        return q_t
    end

    idx_s = sortperm(t)
    t_sort = t[idx_s]
    q_t_sort = zeros(length(t))

    k = 2
    for i = 1:length(t)
        while t_sort[i] >= q[k, 1]
            k += 1
            if k > size(q, 1)
                q_t_sort[i:end] .= q[end, 2]
                return re_sort(q_t_sort, idx_s)
            end
        end
        q_t_sort[i] = q[k-1, 2]
    end

    q_d = re_sort(q_t_sort, idx_s)
    @assert all(q_d .>= 0)

    return q_d
end

## Functions for handling the fluid models

QNorm1(x::Array{Float64,1}, M::Array{Int64, 1}, Q::Int64) = 
    ([sum(x[M .== i]) for i = 1:Q])[M]

function g(x, c)
    return x > 0 ? min(c, x) / x : 1.0
end

function g_smooth(x, c, p)
    return 1 / norm([1, x/c], p)
end

tr_cdf(t, Ψ, ps, ζ) = 1 .- sum(exp(ps*Ψ'*t)*ζ)
tr_cdf(t, Wb, ps, A, β) = (1 .- (A*β)' * exp(Diagonal(ps) * Wb * t) * 
    ones(size(A, 1), 1))[1]

# Extracts the block diagonal parameter matrices from a vector of phase distributions
function getStackedPHMatrices(ph::Array{EMpht.PhaseType, 1})
    S = [v.p for v in ph]
    Ψ = blockdiag(sparse.((x->x.T).(ph))...)
    A = blockdiag(sparse.(reshape.((x->x.π).(ph), S, 1))...)
    B = blockdiag(sparse.(reshape.((x->x.t).(ph), S, 1))...)
    return Ψ, A, B
end

# Extract the optimal p norm value from a given queue length and utilization
function getOptimPNorm(q_avg::Float64, u::Float64, K::Int64; ub=[0.1, 1.0], pb=[1.0, 5.0])
    p_opt = 0

    if ub[1] < u < ub[2]
        p_opt = fzero((p -> q_avg / norm([K, q_avg], p) - u), 0.1)
    else
       p_opt = (u < ub[1] ? pb[2] : pb[1])
    end

    if !(pb[1] <  p_opt < pb[2])
        p_opt = (p_opt < pb[1] ? pb[1] : pb[2])
    end

    return p_opt
end

function dx_fluid!(dx, x, dx_params, t)
    A, W, M, K, p_smooth, λ, tmpq, tmpps = dx_params

    # Remade the QNorm thingy to avoid some allocations, keeping tmpq and tmpps as temporary storage to avoid allocations
    fill!(tmpq, 0)
    for i in 1:length(M)
        tmpq[M[i]] += x[i]
    end
    tmpq .= g_smooth.(tmpq, K, p_smooth)
    for i in 1:length(M)
        tmpps[i] = x[i] * tmpq[M[i]]
    end

    mul!(dx, W', tmpps)
    #tmp = @view λ[:,si_idx(t)]
    mul!(dx, A, λ, true, true)
end

function solveFluidModel(params; solver=Tsit5())
    Ψ, A, B, P, λ, M, K, p_smooth = params
   
    # Extract the type from the type field in P
    T = eltype(P)

    # Precalc to avoid having to do it inside
    W =  Ψ + B*P*A'

    dx_params = (A, W, M, K, p_smooth, λ, zeros(T, length(K)), zeros(T, length(M)))
    tspan = (0.0, 5.0)
    x0 = zeros(T, length(M))

    prob = ODEProblem(dx_fluid!, x0, tspan, dx_params)
    sol = solve(prob, solver)

    return sol
end

# Can be both calculated with closed form and as an ODE solution
function estimate_p95(params; xf=[], solver=Tsit5(), closed_form=false)
    Ψ, A, B, P, λ, M, K, p_smooth = params
    T = eltype(P)

    if isempty(xf)
        sol_xf = solveFluidModel(params; solver)
        xf = [sum(sol_xf.u[end][M .== i], dims=1)[1] for i in 1:length(K)]
    end

    ps = g_smooth.(xf[M], K[M], p_smooth[M])

    # Since Cr includes all classes, we can obtain the p95 matrices directly
    Wb = Ψ + B * P * A'
    β = normalize(λ, 1)

    if closed_form
        return findQuantile(t -> tr_cdf(t, Wb, ps, A, β), 0.95, x0=0.1)
    else
        function dx_p95!(dx, x, W, t)
            mul!(dx, W', x)
        end

        function p95condition(u,t,integrator) 
            # Returned value should be positive normally, and cross 0 when we want something to happen
            sum(u) - 0.05
        end

        cb = ContinuousCallback(p95condition, terminate!)

        tspan = (0.0, 100.0)
        x0 = T.(A*β)
        
        prob = ODEProblem(dx_p95!, x0, tspan, (diagm(ps)*Wb))
        sol = solve(prob, solver, callback=cb)
        return sol.t[end]
    end
end

## Functions for estimating metrics and parameters in the queuing network

# Obtain an estimate of the utilization given a queue length matrix and 
# server count
function getUtil(q::Matrix{Float64}, k::Int64)

    if k == typemax(Int64)
        return 0.0
    end

    if sum(abs.(diff(q[:, 1]))) == 0
        return 0.0
    end
    
    time_frac = zeros(k+1)
    for i = 2:size(q, 1)
        v = Int64(q[i-1, 2])
        if 0 <= v < k
            time_frac[v+1] += q[i, 1] - q[i-1, 1]
        elseif v >= k
            time_frac[k+1] += q[i, 1] - q[i-1, 1]
        end
    end

    util = 0
    for (i, t) in enumerate(time_frac./sum(time_frac))
        util += t*(i-1)/k  #t/(q[end, 1] - q[1, 1]) * i/k
    end

    return util
end

function fitPhaseDist(s::Array{T, 1}, p::Int64; 
    verbose=true, max_iter=200, nbr_bins=100, timeout=1, ph_structure="Coxian") where T <: AbstractFloat
    
    if isempty(s) 
        # Create a PH dist with mean 0.01
        return EMpht.PhaseType(
            [1.0; zeros(p-1)],
            diagm(-p*100*ones(p)) + diagm(1=>p*100*ones(p-1)),
            [zeros(p-1); p*100.0],
            p
        )
    end

    bins = range(0, stop=maximum(s), length=nbr_bins)
    h = fit(Histogram, s, bins)

    return empht(EMpht.Sample(int=hcat(bins[1:end-1], bins[2:end]), 
        intweight=Float64.(h.weights)), p=p, ph_structure=ph_structure, 
        timeout=timeout, verbose=verbose, max_iter=max_iter) 
end


function findClassesForServiceAndDelays(service::String, 
       classes::Vector{Tuple{T_qn, T_qn, Int64}})

    q_srv = []
    push!(q_srv, service)

    # Apppend downstream and upstream delay queues for the service to be scaled
    append!(q_srv, queues[[(typeof(q) <: Tuple ? q[2] == q_srv[1] : false) for q in queues]])
    append!(q_srv, queues[[(typeof(q) <: Tuple ? q[1] == q_srv[1] : false) for q in queues]])

    # Extract all affected classes
    c_srv = zeros(Bool, length(classes))
    for qs in q_srv
        c_srv .|= (q -> q == qs).(getindex.(classes, 1))
    end

    return c_srv
end

function findScalingClassShifts(scaleFrom::String, scaleInto::String, 
        classes::Vector{Tuple{T_qn, T_qn, Int64}},  queue_type::Dict{T_qn, String})

    @assert queue_type[scaleFrom] == "m"
    @assert queue_type[scaleInto] == "m"
    c_scaleFrom = findClassesForServiceAndDelays(scaleFrom, classes)

    shift_class = Dict{Int64, Int64}()
    for (i, affected) in enumerate(c_scaleFrom)
        if !affected
            continue
        end

        (q, n, u) = classes[i]

        class_idx = []
        # Find class shift for the service queue
        if queue_type[q] == "m"
            append!(class_idx, findall([all((qs, ns, us) .== (scaleInto, n, u))
                for (qs, ns, us) in classes]))
        elseif queue_type[q] == "d"

            # Find class shift for downstream delays
            if q[2] == scaleFrom
                append!(class_idx, findall([all((qs, ns, us) .== ((q[1], scaleInto), n, u)) 
                    for (qs, ns, us) in classes]))
            end
            
            # Find class shift for upstream delay
            if q[1] == scaleFrom
                append!(class_idx, findall([all((qs, ns, us) .== ((scaleInto, q[2]), n, u)) 
                    for (qs, ns, us) in classes]))
            end

        end

        @assert length(class_idx) == 1 # should be exactly 1 match
        shift_class[i] = class_idx[1]
        
    end

    return shift_class
end

function findActiveRequests(ta::Array{Float64,1}, td::Array{Float64,1})
    q = getQueueLengths(ta, td)
    n_v = ones(size(ta))

    ka = 1
    for i = 2:length(n_v)
        n_v[i] = n_v[i-1]

        if td[i] < 0
            continue
        end

        while q[ka, 1] != ta[i]
            ka += 1
        end

        if q[ka, 2] >  n_v[i]
            b = q[ka, 2] - n_v[i]
            kd = ka
            while b > 0
                kd += 1
                if q[kd, 2] - q[kd-1, 2] < 0
                    if q[kd, 1] == td[i]
                        n_v[i] = q[ka, 2] - (q[ka, 2] - n_v[i] -  b)
                        break
                    end
                    b -= 1
                end
            end
        end
    end

    return n_v
end