## ===== Loading common packages and functions =====

include("../src/QueueModelTracking.jl")

using ForwardDiff
using JSON

# Function for retrieving the cost from the queue lengths
function costFromQL(C_q, x)
    return sum(C_q .* x)
end

# Function for retrieving the cost from the p95 estimation
function costFromP95(C_p, p95)
    return C_p * p95
end
 
function softmax(x)
    expx = exp.(x)
    expx = all(isfinite.(expx)) ? expx : map(y -> isfinite(y) ? 0 : 1, expx)

    return expx ./ sum(expx)
end

# The cost function to the differentiated.
function sharp_costfunc(params, p95_data, C) 
    _, _, _, _, _, M, K, _, _ = params

    # Solve the fluid model
    sol = solveFluidModel(params)

    # Calculate the queue length in each of the queues
    xf = [sum(sol.u[end][M .== i], dims=1)[1] for i in 1:length(K)]

    if p95_data > p95_lim
        return costFromP95(C[1], estimate_p95(params; xf))
    else
        return costFromQL(C[2], xf)
    end
end

function smooth_costfunc(params, p95_data, β, C) 
    _, _, _, _, λ, M, K, _, _ = params

    # Solve the fluid model
    sol = solveFluidModel(params)

    # Calculate the queue length in each of the queues
    xf = [sum(sol.u[end][M .== i], dims=1)[1] for i in 1:length(K)]

    qlcost = costFromQL(C[3], xf)

    p95cost_open = costFromP95(C[1], estimate_p95(params, β[1], sol_xf=sol))
    p95cost_open *= exp(p95_penaltyf_slope_open * (p95_data[1] - p95_lim_open))

    p95cost_closed = costFromP95(C[2], estimate_p95(params, β[2], sol_xf=sol))
    p95cost_closed *= exp(p95_penaltyf_slope_closed * (p95_data[2] - p95_lim_closed))

    return qlcost + p95cost_open + p95cost_closed
end

# Corresponding costs for data
function sharp_costfunc_data(pop_queue_mean, p95_data, C) 
    if p95_data > p95_lim
        return costFromP95(C[1], p95_data)
    else
        return costFromQL(C[2], pop_queue_mean)
    end
end

function smooth_costfunc_data(pop_queue_mean, p95_data, C) 
    qlcost = costFromQL(C[3], pop_queue_mean)
    p95cost_open = costFromP95(C[1], p95_data[1])
    p95cost_open *= exp(p95_penaltyf_slope_open * (p95_data[1] - p95_lim_open))
    p95cost_closed = costFromP95(C[2], p95_data[2])
    p95cost_closed *= exp(p95_penaltyf_slope_closed * (p95_data[2] - p95_lim_closed))

    return qlcost + p95cost_open + p95cost_closed
end

# Function updating the parameter struct accordingly
function update_params(params, w::AbstractVector{T}, p_ind_w, slices) where T <: Real
    Ψ, A, B, P, λ, M, K, p_prd, _ = params

    P_new = Matrix{T}(P)

    for s in slices
        P_new[p_ind_w[s]] .= softmax(w[s])
    end
   
    return (Ψ, A, B, P_new, λ, M, K, p_prd, T)
end

function update(params, Pw, p95_data, β, C, p_ind)

    #_, _, _, _, λ, _, _, _, _ = params

    # Create weight vector
    Pw_keys = []
    p_ind_w = Vector{CartesianIndex{2}}()
    w = Vector{Float64}()
    for service in keys(Pw)
        for dest in keys(Pw[service])
            push!(Pw_keys, (service, dest))
            append!(w, Pw[service][dest])
            append!(p_ind_w, p_ind[(service, dest)])
        end
    end

    slices = [(i-1)*3+1:i*3 for i = 1:length(p_ind)]

    # Caclulate the derivative of the cost function with respect to w
    dw = ForwardDiff.gradient(y -> costfunc(update_params(params, y, p_ind_w, slices), 
        p95_data, β, C), w)

    # Bound the step to the maximum norm
    p = [softmax(w[s]) for s in slices]
    p_next = [softmax(w[s] - alpha .* dw[s]) for s in slices]
    if norm(p - p_next, 2) > p_step_size
        c = 0
        try
            c = fzero(c -> norm(p - [softmax(w[s] - alpha .* c .* dw[s]) 
                for s in slices], 2) - p_step_size, 0.0, 1.0)
        catch
            printstyled("fzero failed! Using approximation\n", color=:red);
            c = norm(p, 2)*p_step_size / norm(p - p_next, 2)
        end
        dw_scaled = c * alpha .* dw
    else
        dw_scaled = alpha .* dw
    end

    # Take gradient step
    w_new = w .- dw_scaled

    # Extract Pw_new
    Pw_new = Dict()
    for (i, (service, dest)) in enumerate(Pw_keys)
        if !haskey(Pw_new, service)
            Pw_new[service] = Dict()
        end 
        Pw_new[service][dest] = w_new[slices[i]]
    end

    # Predict p95 for new value of p
    p95_pred_open = estimate_p95(update_params(params, w_new, p_ind_w, slices), β[1])
    p95_pred_closed = estimate_p95(update_params(params, w_new, p_ind_w, slices), β[2])

    return Pw_new, dw, p95_pred_open, p95_pred_closed
end

# Bounding the probabilities from below by limiting the 
function bound_probs(p)
    function reduce(available, needed, red)
        if needed == 0
            return red
        else
            av_idx = available .> 0
            av_min = minimum(available[av_idx])
            if  av_min > needed / sum(av_idx)
                available[av_idx] .-= needed / sum(av_idx)
                red[av_idx] .-= needed / sum(av_idx)
                reduce(available, needed - needed, red)
            else
                available[av_idx] .-= av_min
                red[av_idx] .-= av_min 
                reduce(available, needed - sum(av_idx)*av_min, red)
            end
        end
    end

    # Assert minimum probability values
    if any(p .< p_min)
        to_small = p .< p_min
        needed = sum(p_min .- p[to_small])
        available = max.(p .- p_min, 0)
        red = reduce(available, needed, zeros(length(p)))
        p += red
        p[to_small] .= p_min
        @assert all(p .>= p_min)
        @assert all(p .> 0)
        @assert isapprox(sum(p), 1.0)
        p ./= sum(p)
    end
    return p
end

## ===== Parameters =====

# Choose smooth or sharp costfunc
costfunc(params, p95_data, β, C) = smooth_costfunc(params, p95_data, β, C) 
costfunc_data(pop_queue_mean, p95_data, C) = smooth_costfunc_data(pop_queue_mean, p95_data, C) 

# Test increasing cost per p95 even more!!

# Cost for violating the p95 limit
cost_per_p95_open = 100.0
p95_penaltyf_slope_open = 12.0

cost_per_p95_closed = 100.0
p95_penaltyf_slope_closed = 30.0

# Limit of p95 RT which we will not move past
p95_lim_open = 1.25
p95_lim_closed = 0.4

# Determine the size of the gradient step
alpha = 0.5

# Number of phase states per PH distribution 
phases = Dict("ec"=>1, "m"=>3, "d"=>3)

# Number of processors per service
processors_per_service = 4

# Printout verbosity
verbose = true

# minimum probability
p_min = 0.01

# maximum gradient step in 2-norm of probability
p_step_size = 0.15

function findNextParam()

    # In-parameters
    input_dict = YAML.load_file(joinpath(@__DIR__, "input.yaml"); dicttype=Dict{String, Any})
    logfolder = input_dict["logfolder"]
    tracefolder = input_dict["tracefolder"]
    Pw = input_dict["W"]
    cost_per_ql = input_dict["cost_per_ql"]

    ## ===== Read data =====
    if verbose printstyled("Reading the data\n",bold=true, color=:green); end

    # Read the loadgenerator settings file
    loadgenfiles =  joinpath.(logfolder, filter(x -> occursin("load_generator_", x), readdir(logfolder)))
    loadgensettingsfiles = joinpath.(logfolder, filter(x -> occursin("settings", x), readdir(logfolder)))
    simSettings = Dict{String, Dict{String, Any}}()
    for file in loadgensettingsfiles
        lg = split(split(file, "settings_")[2], ".")[1]
        simSettings[lg] = YAML.load_file(file; dicttype=Dict{String, Any}) 
        simSettings[lg]["experimentTime"] = unix2datetime.(simSettings[lg]["experimentTime"])
    end

    dfs_load = readLoadGenData(loadgenfiles, simSettings)

    # read the trace files
    df_pods, df_err, err_prct, ipname_bimap = readTraceData(tracefolder, simSettings, verbose=false)

    # report errors (if larger than some percentage?)
    if sum(values(err_prct)) > 0.0
        printstyled("Warning, errors detected in reading data\n", color=:red)
        for key in keys(err_prct)
            if err_prct[key] > 0.0
                printstyled("\t$key error: $(round(err_prct[key], digits=2)) %\n", color=:red)
            end
        end
    end

    # extract data into H objects
    data_H = traceData2Hr(df_pods)

    ## ===== Extract the queueing network topology =====
    if verbose printstyled("Extracting network topology\n",bold=true, color=:green); end

    # Extract classes and queue
    classes, ext_arrs,  queue_type = createClasses(data_H, simSettings)
    queues = unique(getindex.(classes, 1))

    # Boolean vectors for class containment in network, application 
    classInApp = ones(Bool, length(classes))
    for (k, class) in enumerate(classes)
        if queue_type[class[1]] == "ec"
            classInApp[k] = 0
        end
    end

    queueInApp = ones(Bool, length(queues))
    queueIsService = zeros(Bool, length(queues))
    for (k, queue) in enumerate(queues)
        if queue_type[queue] == "ec"
            queueInApp[k] = 0
        elseif queue_type[queue] == "m"
            queueIsService[k] = 1
        end
    end

    # Retrieve important queuing network parameters
    Cq = [sum([qc == q for (qc, _, _) in classes]) for q in queues]
    S = zeros(Int64, length(classes))
    queue_disc = Dict{T_qn, String}()
    queue_servers = Dict{T_qn, Int}()
    for (i, (q, _, _)) = enumerate(classes)
        S[i] = phases[queue_type[q]]
    end
    for q in queues
        queue_disc[q] = (queue_type[q] == "m" ? "PS" : "INF")
        queue_servers[q] = (queue_type[q] == "m" ? processors_per_service :  typemax(Int64))
    end
    M, Mc, N = getPhasePos(S, length(queues), Cq)

    # Extract the chains visited by external requests
    #t0, _ = datetime2unix.(getSimExpTimeIntervals(simSettings))
    #H_si = getHInTspan(data_H, simSecInt[i] .+ t0)
    connGraph = getClassRoutes(data_H, classes, queue_type) .> 0

    closed_starts = findall(.!classInApp)
    open_start = findall(getExternalArrivals(data_H, ext_arrs, classes) .> 0)

    chains_closed = findConnComp(connGraph, closed_starts)
    chains_open = findConnComp(connGraph, open_start)

    #visitedClassesNbr = [sort(unique(vcat([chains_open[i]; chains_closed[i]]...))) 
    #    for i = 1:si]
    #visitedClasses = [(c -> c ∈ visitedClassesNbr[i]).(1:length(classes)) for i = 1:si]

    ## ===== Extract queueing network states and variables =====
    if verbose printstyled("Extracting network states and variables\n",bold=true, color=:green); end

    # Calculate arrival/departure times for each class in each sim. 
    ta_class, td_class, ri_class = getArrivalDeparture(data_H, simSettings, classes, queue_type)

    @assert all(vcat(td_class...) .>= vcat(ta_class...))
    tw_class = td_class - ta_class

    # Extract path of requests over classes
    paths_class, paths_err = getAllPaths(ri_class, ta_class, td_class, 
        classes, ext_arrs, queue_type)

    nbr_p_err = length(values(paths_err))
    nbr_p = length(values(paths_class))
    if nbr_p_err > 0
        printstyled("Warning, path extraction errors\n", color=:red)
        printstyled("\tremoving $(round(nbr_p_err / (nbr_p_err + nbr_p) * 100, digits=2)) %\n",
            color=:red)
    end

    paths_class_open = Dict{Int64, DataFrame}()
    paths_class_closed = Dict{Int64, DataFrame}()
    paths_unknown = Dict{Int64, DataFrame}()
    for k in keys(paths_class)
        if paths_class[k].class[1][1] == ipname_bimap["frontend"]
            paths_class_open[k] = paths_class[k]
        elseif paths_class[k].class[1][1] == "st"
            paths_class_closed[k] = paths_class[k]
        else
            paths_unknown[k] = paths_class[k]
        end
    end
    if length(paths_unknown) > 0
        printstyled("Warning, unknown chain types\n", color=:red)
        printstyled("\t removing $(round(length(paths_unknown) / length(paths_class) * 100, 
            digits=2)) %\n", color=:red)
    end

    # Extract populations for queues and classes
    pop_class = getQueueLengths.(ta_class, td_class, start_time=minimum(minimum(ta_class)))
    pop_queue = Vector{Matrix{Float64}}(undef, length(queues))
    for i = 1:length(queues)
        pop_queue[i] = addQueueLengths(pop_class[Mc .== i])
    end

    # Calculate mean queue length in both classes and queues
    pop_class_mean = getQueueLengthAvg.(pop_class)
    pop_queue_mean = getQueueLengthAvg.(pop_queue)

    # Calcualte utilization and optimal smoothing values
    util = zeros(length(queues))
    p_smooth_opt = zeros(length(queues))
    for (i, q) in enumerate(queues)
        util[i] = getUtil(pop_queue[i], queue_servers[q])
        p_smooth_opt[i] = getOptimPNorm(pop_queue_mean[i], util[i], queue_servers[q])
    end

    # Calculate the service times
    ts_class = getServiceTimes(ta_class, td_class, queue_servers, classes) 

    # Fit PH distribution to classes
    if verbose printstyled("Fitting PH dist\n", bold=true, color=:magenta); end
    ph_vec = Vector{EMpht.PhaseType}(undef, length(classes))
    for (i, (q, _, _)) = enumerate(classes)
        if verbose println("\t$i / $(length(classes))"); end
        ph_vec[i] = fitPhaseDist(filtOutliers(ts_class[i], ϵ=0, α=0.99, β=10), 
                phases[queue_type[q]], max_iter=200, verbose=false)
    end

    # Extract arrival rates
    w_a = getExternalArrivals(data_H, ext_arrs, classes)
    λ = map(x -> x > 0 ? x : 0, w_a) ./ 
        (maximum(maximum(td_class)) - minimum(minimum(ta_class)))

    # Extract current routing probability matrix
    P = zeros(length(classes), length(classes))
    class_idx_dict = classes2map(classes)
    w = getClassRoutes(data_H, classes, queue_type)
    for (i, (q, n, u)) in enumerate(classes)
        w_tmp = w_a[i] < 0 ? w_a[class_idx_dict[(q, n, 1)]] : 0
        if (sum(w[i, :]) + w_tmp) > 0
            P[i, :] = w[i, :] ./ (sum(w[i, :]) + w_tmp)
        end
    end

    ## ===== Estimate queue length, p95 and take gradient step =====
    if verbose printstyled("Estimating parameters and autodiff\n",bold=true, color=:green); end

    # Get the indexes of p for all load balancers
    lb_conn =(
        (("frontend", "detect"), ("backend", "detect")),
        (("frontend", "fetch"), ("storage", "fetch")),
        (("backend-v1", "detect"), ("storage", "store")),
        (("backend-v2", "detect"), ("storage", "store")),
        (("backend-v3", "detect"), ("storage", "store"))
    )
    p_ind = Dict()
    for (ds, us) in lb_conn
        depart = (ipname_bimap[ds[1]], "/" * ds[2] * "/", 1)
        us_replicas = sort(collect(filter(x -> occursin(us[1], x), keys(ipname_bimap))))

        p_idx = []
        for rs in us_replicas
            arrives = ((depart[1], ipname_bimap[rs]), (depart[2], "/" * us[2] * "/"), 1)
            push!(p_idx, (findfirst(c -> c == depart, classes),  findfirst(c -> c == arrives, classes)))
        end
        p_ind[(ds[1], us[2])] = CartesianIndex.(p_idx)
    end

    # Check P at these coordinates
    if verbose
        printstyled("Comparing measured and wanted P\n", bold=true)
        pd(x) = round.(x, digits=2)
        for (service, reqtype) in keys(p_ind)
            println("For $service $reqtype")
            println("\tMeasured: $(pd(P[p_ind[(service, reqtype)]]))")
            println("\tSet: $(pd(softmax(Pw[service][reqtype])))")
        end
    end    

    # Extract the fluid model parameters
    Ψ, A, B = getStackedPHMatrices(ph_vec)
    K = [queue_servers[q] for q in queues]

    params = (Ψ, A, B, P, λ, M, K, p_smooth_opt, Float64)

    # The p95 response times according to the data
    ta_cr_open, td_cr_open, _ = getArrivalDeparture(paths_class_open, classes)
    tw_cr_open = td_cr_open - ta_cr_open
    p95_data_open = findQuantileData(tw_cr_open, 0.95)

    paths_class_closed_noClientQueue = Dict{Int64, DataFrame}()
    for key in keys(paths_class_closed)
        paths_class_closed_noClientQueue[key] = paths_class_closed[key][2:end, :]
    end
    ta_cr_closed, td_cr_closed, _ = getArrivalDeparture(paths_class_closed_noClientQueue, classes)
    tw_cr_closed = td_cr_closed - ta_cr_closed
    p95_data_closed = findQuantileData(tw_cr_closed, 0.95)

    violating_open = p95_data_open > p95_lim_open # Use p95_data instead for soft cost
    violating_closed = p95_data_closed > p95_lim_closed # Use p95_data instead for soft cost

    β_open = normalize(λ)
    β_closed = zeros(size(λ))
    β_closed[findfirst(c -> c == (classes[1][2][1], classes[1][2][2], 1), classes)] = 1

    # Extract cost vectors
    C_q = zeros(length(queues))
    for (i, q) in enumerate(queues)
        if haskey(ipname_bimap, q)
            C_q[i] = get(cost_per_ql, ipname_bimap[q], 0)
        end
    end
    C = [cost_per_p95_open, cost_per_p95_closed, C_q]

    # The current cost
    cost_data = costfunc_data(pop_queue_mean, [p95_data_open, p95_data_closed], C)
    cost = costfunc(params, [p95_data_open, p95_data_closed], [β_open, β_closed], C)

    # Estimate population in each queue
    sol = solveFluidModel(params)
    xf = [sum(sol.u[end][M .== i], dims=1)[1] for i in 1:length(queues)]
    xfc = [sum(sol.u[end][N .== i], dims=1)[1] for i in 1:length(classes)]

    # Current cost for queues only
    cost_only_queues_data = costFromQL(C_q, pop_queue_mean)
    cost_only_queues = costFromQL(C_q, xf)

    # Current cost for queus only in the two chains   
    pop_class_mean_open = zeros(length(classes))
    pop_class_mean_closed = zeros(length(classes))
    pop_class_mean_open[chains_open[1]] = pop_class_mean[chains_open[1]]
    pop_class_mean_closed[chains_closed[1]] = pop_class_mean[chains_closed[1]]

    pop_queue_mean_open = [sum(pop_class_mean_open[Mc .== i]) for i in 1:length(queues)]
    pop_queue_mean_closed = [sum(pop_class_mean_closed[Mc .== i]) for i in 1:length(queues)]

    # Estimate p95
    p95_open = estimate_p95(params, β_open, sol_xf=sol)
    p95_closed = estimate_p95(params, β_closed, sol_xf=sol)

    # Calculate gradient step
    Pw_new, dPw, p95_pred_open, p95_pred_closed = update(params, Pw, 
        [p95_data_open, p95_data_closed], [β_open, β_closed], C, p_ind)
    
    Pw_new_prob = Dict()
    for service in keys(Pw_new)
        Pw_new_prob[service] = Dict()
        for dest in keys(Pw_new[service])
            Pw_new_prob[service][dest] = bound_probs(softmax(Pw_new[service][dest]))
        end
    end
    
    if verbose 
        pd(x) = round.(x, digits=2)
        printstyled("Values\n", bold=true, color=:magenta)
        print("\tp95 open:   $(pd(p95_open)), p95_data: $(pd(p95_data_open))\n")
        print("\tp95 closed: $(pd(p95_closed)), p95_data: $(pd(p95_data_closed))\n")
        print("\tql:\t     $(pd(sum(xf))), ql_data: $(pd(sum(pop_queue_mean)))\n")

        printstyled("Relative error\n", bold=true, color=:magenta)
        print("\tp95 open:   $(pd(abs(p95_open-p95_data_open)/p95_data_open * 100)) %\n")
        print("\tp95 closed: $(pd(abs(p95_closed-p95_data_closed)/p95_data_closed * 100)) %\n")
        print("\tql:\t     $(pd(norm(xf - pop_queue_mean, Inf) / sum(pop_queue_mean) * 100)) %\n")

        printstyled("Cost\n", bold=true, color=:magenta)
        print("\tcost data:\t $(pd(cost_data))\n")
        print("\tcost model:\t $(pd(cost))\n")
        print("\tviolating open:   $(violating_open)\n")
        print("\tviolating closed: $(violating_closed)\n")

        printstyled("Autodiff\n", bold=true, color=:magenta)
        print("\t-dC/dw: $(pd(-dPw))\n")
        print("\tp next: $(json(Pw_new_prob, 2))\n")
        print("\tp95 pred open: $(pd(p95_pred_open))\n")
        print("\tp95 pred closed: $(pd(p95_pred_closed))\n")
    end

    # Log data for plotting
    # Create file and print header first time
    tag = "default"
    if !isfile(joinpath(logfolder, "data_$(tag).csv"))
        CSV.write(joinpath(logfolder, "data_$(tag).csv"), []; writeheader=true, header=[
            ["frontend-detect-p$i" for i = 1:3]..., ["frontend-fetch-p$i" for i = 1:3]...,
            ["backend-v1-store-p$i" for i = 1:3]..., ["backend-v2-store-p$i" for i = 1:3]...,
            ["backend-v3-store-p$i" for i = 1:3]...,
            :ql, :ql_data, :p95_open, :p95_data_open, :p95_pred_open, 
            :p95_closed, :p95_data_closed, :p95_pred_closed, :cost, :cost_data,
            :cost_only_queues, :cost_only_queues_data, :violating_open, :violating_closed])
    end
    CSV.write(
        joinpath(logfolder, "data_$(tag).csv"), 
        (   (Symbol("frontend-detect-p$i") => [p] for (i, p) in enumerate(softmax(Pw["frontend"]["detect"])))..., 
            (Symbol("frontend-fetch-p$i") => [p] for (i, p) in enumerate(softmax(Pw["frontend"]["fetch"])))...,
            (Symbol("backend-v1-store-p$i") => [p] for (i, p) in enumerate(softmax(Pw["backend-v1"]["store"])))...,
            (Symbol("backend-v2-store-p$i") => [p] for (i, p) in enumerate(softmax(Pw["backend-v2"]["store"])))...,  
            (Symbol("backend-v3-store-p$i") => [p] for (i, p) in enumerate(softmax(Pw["backend-v3"]["store"])))...,         
            ql=[sum(xf)], ql_data=[sum(pop_queue_mean)], 
            p95_open=[p95_open], p95_data_open=[p95_data_open], p95_pred_open=[p95_pred_open],
            p95_closed=[p95_closed], p95_data_closed=[p95_data_closed], p95_pred_closed=[p95_pred_closed],
            cost=[cost], cost_data=[cost_data], cost_only_queues=[cost_only_queues],
            cost_only_queues_data=[cost_only_queues_data], violating_open=[violating_open],
            violating_closed=[violating_closed], ); 
        append=true,
    )

    output_dict = Dict{String, Any}(
        "W_next" => Pw_new
    )

    # Output to file
    YAML.write_file(joinpath(@__DIR__, "output.yaml"), output_dict)
end


##

# Insert while loop here that checks for lock?
# Also, run the function once before starting the experiment
# to compile

function run()
    lock_file = joinpath(@__DIR__, "lock")
    touch(lock_file)
    last_time = mtime(lock_file)
    while true
        printstyled("Waiting for updated values\n", bold=true, color=:green)
        while last_time == mtime(lock_file)
            sleep(0.5)
        end
        findNextParam()
        touch(lock_file)
        last_time = mtime(lock_file)
        GC.gc()
    end
end

run()