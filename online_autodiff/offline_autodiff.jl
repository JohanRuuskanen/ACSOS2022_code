## ===== Loading common packages and functions =====

include("src/QueueModelTracking.jl")

using ForwardDiff

# Function for retrieving the cost from the queue lengths
function costFromQL(C_q, x)
    return sum(C_q .* x)
end

# Function for retrieving the cost from the p95 estimation
function costFromP95(C_p, p95)
    return C_p * p95
end

# Function for retrieving the cost from step
function costFromStep(C_p, P)
    return C_p * p95
end 
 
function softmax(x)
    expx = exp.(x)
    return expx ./ sum(expx)
end

# The cost function to the differentiated.
function sharp_costfunc(params, p95_data, C) 
    _, _, _, _, _, M, K, _ = params

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
function smooth_costfunc(params, p95_data, C) 
    _, _, _, _, _, M, K, _ = params

    # Solve the fluid model
    sol = solveFluidModel(params)

    # Calculate the queue length in each of the queues
    xf = [sum(sol.u[end][M .== i], dims=1)[1] for i in 1:length(K)]

    qlcost = costFromQL(C[2], xf)
    p95cost = costFromP95(C[1], estimate_p95(params; xf))
    p95cost *= exp(10 * (p95_data - p95_lim))

    return qlcost + p95cost
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
    qlcost = costFromQL(C[2], pop_queue_mean)
    p95cost = costFromP95(C[1], p95_data)
    p95cost *= exp(10 * (p95_data - p95_lim))

    return qlcost + p95cost
end

# Function updating the parameter struct accordingly
function update_params(params, p, p_ind)
    Ψ, A, B, P, λ, M, K, p_prd = params

    P_new = Matrix{typeof(p)}(P)
    P_new[p_ind] .= [p, 1-p]

    return (Ψ, A, B, P_new, λ, M, K, p_prd)
end

function update(params, p, p95_data, C, p_ind)

    # Caclulate the derivative of the cost function with respect to w
    dp = ForwardDiff.derivative(y -> costfunc(update_params(params, y, p_ind), p95_data, C), p)

    # Take gradient step
    p_new = clamp(p - 0.05 * sign(dp), 0, 1)

    # Predict p95 for new value of p
    p95_pred = estimate_p95(update_params(params, p_new, p_ind))

    return p_new, dp, p95_pred
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
costfunc(params, p95_data, C) = smooth_costfunc(params, p95_data, C) 
costfunc_data(pop_queue_mean, p95_data, C) = smooth_costfunc_data(pop_queue_mean, p95_data, C) 

# Cost for violating the p95 limit
const cost_per_p95 = 5.0

# Limit of p95 RT which we will not move past
const p95_lim = 0.60

# Determine the size of the gradient step
const alpha = 0.5

# Number of phase states per PH distribution 
phases = Dict("ec"=>1, "m"=>3, "d"=>3)

# Number of processors per service
processors_per_service = 4

# Printout verbosity
verbose = true

# minimum probability
const p_min = 0.01

# maximum gradient step in 2-norm of probability
const p_step_size = 0.15

function findNextParam()

    # In-parameters
    input_dict = YAML.load_file(joinpath(@__DIR__, "input.yaml"); dicttype=Dict{String, Any})
    logfolder = input_dict["logfolder"]
    tracefolder = input_dict["tracefolder"]
    p0 = input_dict["p0"]
    cost_per_ql = input_dict["cost_per_ql"]
    @show p0

    ## ===== Read data =====
    if verbose printstyled("Reading the data\n",bold=true, color=:green); end

    # Read the loadgenerator settings file
    loadgensettingsfiles = joinpath.(logfolder, filter(x -> occursin("settings", x), readdir(logfolder)))
    simSettings = Dict{String, Dict{String, Any}}()
    for file in loadgensettingsfiles
        lg = split(split(file, "settings_")[2], ".")[1]
        simSettings[lg] = YAML.load_file(file; dicttype=Dict{String, Any}) 
        simSettings[lg]["experimentTime"] = unix2datetime.(simSettings[lg]["experimentTime"])
    end

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
        printstyled("Warning, path extraction error\n", color=:red)
        printstyled("\terr: $(round(nbr_p_err / (nbr_p_err + nbr_p) * 100, digits=2)) %\n",
            color=:red)
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

    # Get the indexes of p for all backends
    depart = (ipname_bimap["frontend"], "/detect/", 1)
    backends = sort(collect(filter(x -> occursin("backend", x), keys(ipname_bimap))))
    p_idx = []
    for backend in backends
        arrives = ((depart[1], ipname_bimap[backend]), (depart[2], "/detect/"), 1)
        push!(p_idx, (findfirst(c -> c == depart, classes),  findfirst(c -> c == arrives, classes)))
    end
    p_ind = CartesianIndex.(p_idx)

    # Extract the fluid model parameters
    Ψ, A, B = getStackedPHMatrices(ph_vec)
    K = [queue_servers[q] for q in queues]
    params = (Ψ, A, B, P, λ, M, K, p_smooth_opt)

    # The p95 response times according to the data
    ta_cr, td_cr, _ = getArrivalDeparture(paths_class, classes)
    tw_cr = td_cr - ta_cr
    p95_data = findQuantileData(tw_cr, 0.95)

    violating = p95_data > p95_lim # Use p95_data instead for soft cost

    # Extract cost vectors
    C_p = cost_per_p95
    C_q = zeros(length(queues))
    for (i, q) in enumerate(queues)
        if haskey(ipname_bimap, q)
            C_q[i] = get(cost_per_ql, ipname_bimap[q], 0)
        end
    end
    C = [C_p, C_q]

    # The current cost
    cost_data = costfunc_data(pop_queue_mean, p95_data, C)
    cost = costfunc(params, p95_data, C)

    # Estimate population in each queue
    sol = solveFluidModel(params)
    xf = [sum(sol.u[end][M .== i], dims=1)[1] for i in 1:length(K)]

    # Estimate p95
    p95 = estimate_p95(params, xf=xf)

    # Calculate gradient step
    p_new, dp, p95_pred = update(params, p0, p95_data, C, p_ind)

    if verbose 
        pd(x) = round.(x, digits=2)
        printstyled("Values\n", bold=true, color=:magenta)
        print("\tp95: $(pd(p95)), p95_data: $(pd(p95_data))\n")
        print("\tql: $(pd(sum(xf))), ql_data: $(pd(sum(pop_queue_mean)))\n")

        printstyled("Relative error\n", bold=true, color=:magenta)
        print("\tp95: $(pd(abs(p95-p95_data)/p95_data * 100)) %\n")
        print("\tql: $(pd(norm(xf - pop_queue_mean, Inf) / sum(pop_queue_mean) * 100)) %\n")

        printstyled("Cost\n", bold=true, color=:magenta)
        print("\tcost data: $(pd(cost_data))\n")
        print("\tcost model: $(pd(cost))\n")
        print("\tviolating: $(violating)\n")

        printstyled("Autodiff\n", bold=true, color=:magenta)
        print("\t-dC/dp: $(pd(-dp))\n")
        print("\tp next: $(pd(p_new))\n")
        print("\tp95 pred: $(pd(p95_pred))\n")
    end

    # Log data for plotting
    # Create file and print header first time
    tag = "default"
    if !isfile(joinpath(logfolder, "data_$(tag).csv"))
        CSV.write(joinpath(logfolder, "data_$(tag).csv"), []; writeheader=true, header=[:p0, :ql, :ql_data, :p95, :p95_data, :cost, :cost_data, :violating, :p95_pred])
    end
    CSV.write(
        joinpath(logfolder, "data_$(tag).csv"), 
        (   
            p0=[p0], 
            ql=[sum(xf)], ql_data=[sum(pop_queue_mean)], p95=[p95], p95_data=[p95_data],
            cost=[cost], cost_data=[cost_data], violating=[violating], p95_pred=[p95_pred]
        ); 
        append=true,
    )

    # Output to file
    output_dict = Dict{String, Any}(
        "p_next" => p_new
    )
    YAML.write_file(joinpath(@__DIR__, "output.yaml"), output_dict)
end


## The main loop of the gradient stepping

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
    end
end

run()