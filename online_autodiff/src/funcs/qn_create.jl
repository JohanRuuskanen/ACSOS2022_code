
# Define classes for all queues. Each class is defined as as 3 tuple, with 
# (queue, method_call ID, upstreams connection ID)
function createClasses(H::T_H, simSetting::T_simSetting)

    classes = Vector{Tuple{T_qn, T_qn, Int64}}(undef, 0)
    ext_arrs = Vector{Tuple{T_qn, T_qn, Int64}}(undef, 0)
    queue_types = Dict{T_qn, String}()

    qrt_keys = sort(collect(keys(H)))
  

    # Add external classes
    for es in keys(simSetting)
        suffix = (simSetting[es]["loadType"] == "closed" ? "c" : "o")
        queue_types[es] = "e" * suffix
        for key in qrt_keys
            if any(occursin.(es, unique(H[key].Dr)))
                if suffix == "c"
                    push!(classes, (es, key, 1))
                elseif suffix == "o"
                    push!(ext_arrs, (es, key, 1))
                end
                break
            end
        end
    end

    # Add service classes
    for key in qrt_keys
        queue_types[key[1]] = "m"
        for k = 1:length(H[key].Ur[1]) + 1 
            push!(classes, (key[1], key[2], k))
        end
    end

    # Add delay classes
    for key_from in qrt_keys
        for key_to in qrt_keys
            if key_to in vcat(unique(H[key_from].Ur)...)
                queue_types[(key_from[1], key_to[1])] = "d"
                push!(classes, ((key_from[1], key_to[1]), (key_from[2], key_to[2]), 1))
                push!(classes, ((key_from[1], key_to[1]), (key_from[2], key_to[2]), 2))
            end
        end
    end

    return classes, ext_arrs, queue_types
end

# Maps each element in classes to its position in the vector, for O(1) lookup time
function classes2map(classes::Vector{Tuple{T_qn, T_qn, Int64}})
    class_idx_dict = Dict{Tuple{T_qn, T_qn, Int64}, Int64}()
    for (i, c) in enumerate(classes)
        class_idx_dict[c] = i
    end
    return class_idx_dict
end

# Get arrivals from external sources to the classes in the given dataframe. 
# Negative values shows where they leave
function getExternalArrivals(H::T_H, ext_arrs::Vector{Tuple{T_qn, T_qn, Int64}}, 
        classes::Vector{Tuple{T_qn, T_qn, Int64}})

    class_idx_dict = classes2map(classes)
    w_a = zeros(length(classes))

    for (i, (eo, (q, n), _)) in enumerate(ext_arrs)
        idx_in = class_idx_dict[(q, n, 1)]
        if size(H[(q, n)], 1) > 0 
            idx_out = class_idx_dict[(q, n, length(H[(q,n)].Ur[1]) + 1)]

            for j = 1:size(H[(q, n)], 1)
                if H[(q, n)].Dr[j] == eo
                    w_a[idx_in] += 1
                    w_a[idx_out] -= 1
                end
            end
        end
    end

    return w_a
end

# Get class to class routes in the given dataframe
function getClassRoutes(H::T_H, classes::Vector{Tuple{T_qn, T_qn, Int64}}, 
        queue_type::Dict{T_qn, String})

    class_idx_dict = classes2map(classes)
    w = zeros(length(classes), length(classes))

    # Internal routes
    for (i, (q, n)) in enumerate(keys(H))
        for (j, Ur) in enumerate(H[(q, n)].Ur)
            for (k, (qu, nu)) in enumerate(Ur)
                if size(H[(qu, nu)], 1) > 0
                    idx_qn_up = class_idx_dict[(q, n, k)]
                    idx_d_up = class_idx_dict[((q, qu), (n, nu), 1)]
                    idx_qnu_in = class_idx_dict[(qu, nu, 1)]
                    idx_qnu_out = class_idx_dict[(qu, nu, length(H[(qu, nu)].Ur[1]) + 1)]
                    idx_d_ret = class_idx_dict[((q, qu), (n, nu), 2)]
                    idx_qn_ret = class_idx_dict[(q, n, k+1)]

                    w[idx_qn_up, idx_d_up] += 1
                    w[idx_d_up, idx_qnu_in] += 1
                    w[idx_qnu_out, idx_d_ret] += 1
                    w[idx_d_ret, idx_qn_ret] += 1
                end
            end
        end
    end

    # Routes from/to ec
    for (i, (ec, (qu, nu), _)) in enumerate(classes)
        if queue_type[ec] == "ec"
            for j = 1:size(H[(qu, nu)], 1)
                if occursin(ec, H[(qu, nu)].Dr[j])
                    idx_qnu_in = class_idx_dict[(qu, nu, 1)]
                    idx_qnu_out = class_idx_dict[(qu, nu, length(H[(qu, nu)].Ur[1]) + 1)]
                    w[i, idx_qnu_in] += 1
                    w[idx_qnu_out, i] += 1
                end
            end
        end
    end

    return w
end

# From the Hr data, extract all arrival/departures for the classes as defined 
# from the createClasses function. Also returns the vector of request IDs
function getArrivalDeparture(H::T_H, simSetting::T_simSetting, 
        classes::Vector{Tuple{T_qn, T_qn, Int64}}, 
        queue_type::Dict{T_qn, String})

    ta_class = Vector{Vector{Float64}}(undef, length(classes))
    td_class = Vector{Vector{Float64}}(undef, length(classes))
    ri_class = Vector{Vector{Int64}}(undef, length(classes))

    t0, _ = datetime2unix.(getSimExpTimeIntervals(simSetting))

    for (i, (q, n, u)) in enumerate(classes)
        if queue_type[q] == "ec"
            ta_class[i] = zeros(0)
            td_class[i] = zeros(0)
            ri_class[i] = zeros(0)

            # Have dict of previous times, start at 0
            isFirst = Dict{String, Bool}()
            for clientID in unique(H[n].Dr)
                isFirst[clientID] = true
            end

            for j = 1:size(H[n], 1)
                if !occursin(q, H[n].Dr[j])
                    continue
                end

                if isFirst[H[n].Dr[j]]
                    append!(ta_class[i], 0.0)
                    append!(td_class[i], datetime2unix(H[n].tr[j]) - t0)
                    append!(ri_class[i], H[n].reqID[j])
                    isFirst[H[n].Dr[j]] = false
                end

                k = j + 1
                while k <= size(H[n], 1)
                    if H[n].Dr[j] == H[n].Dr[k]
                        append!(ta_class[i], datetime2unix(H[n].tr[j] + H[n].dtr[j]) - t0)
                        append!(td_class[i], datetime2unix(H[n].tr[k]) - t0)
                        append!(ri_class[i], H[n].reqID[k])
                        break
                    end
                    k += 1
                end
            end
        elseif queue_type[q] == "m"
            ri_class[i] = H[(q, n)].reqID
            if u == 1
                ta_class[i] = datetime2unix.(H[(q, n)].tr) .- t0
                if length(H[(q, n)].Ur[1]) == 0
                    td_class[i] = datetime2unix.(H[(q, n)].tr .+ H[(q, n)].dtr) .- t0
                else
                    td_class[i] = datetime2unix.(getindex.(H[(q, n)].Tr, 1)) .- t0
                end 
            elseif u == length(H[(q, n)].Ur[1])+1
                ta_class[i] = datetime2unix.(getindex.(H[(q, n)].Tr, u-1) .+
                    getindex.(H[(q, n)].dTr, u-1)) .- t0
                td_class[i] = datetime2unix.(H[(q, n)].tr .+ H[(q, n)].dtr) .- t0
            else
                ta_class[i] = datetime2unix.(getindex.(H[(q, n)].Tr, u-1) .+
                    getindex.(H[(q, n)].dTr, u-1)) .- t0
                td_class[i] = datetime2unix.(getindex.(H[(q, n)].Tr, u)) .- t0
            end
        elseif queue_type[q] == "d"
            ((qd, qu), (nd, nu)) = (q, n)
            ta_class[i] = zeros(0)
            td_class[i] = zeros(0)
            ri_class[i] = zeros(0)
            for (j, Ur) = enumerate(H[(qd, nd)].Ur)
                for (k, (qu_2, nu_2)) = enumerate(Ur)
                    if (qu, nu) == (qu_2, nu_2)
                        req_idx_us = findfirst(H[(qu, nu)].reqID .== H[(qd, nd)].reqID[j])
                        delay = Dates.value(H[(qd, nd)].dTr[j][k] - H[(qu, nu)].dtr[req_idx_us]) / 2 / 1000
                        @assert delay >= 0
                        if u == 1
                            t_tmp = datetime2unix(H[(qd, nd)].Tr[j][k]) - t0
                            append!(ta_class[i], t_tmp)
                            append!(td_class[i], t_tmp + delay)
                        elseif u == 2
                            t_tmp = datetime2unix(H[(qd, nd)].Tr[j][k] + H[(qd, nd)].dTr[j][k]) - t0
                            append!(ta_class[i], t_tmp - delay)
                            append!(td_class[i], t_tmp)
                        end
                        append!(ri_class[i], H[(qd, nd)].reqID[j])
                    end
                end
            end
        else 
            error("No such queue type")
        end

        @assert length(ta_class[i]) == length(td_class[i]) == 
            length(ri_class[i]) "Not the same number of arrivals and departures"

        sortidx = sortperm(ta_class[i])
        ta_class[i] = ta_class[i][sortidx]
        td_class[i] = td_class[i][sortidx]
        ri_class[i] = ri_class[i][sortidx]

        @assert all(ta_class[i] .<= td_class[i]) "Departure before arrival"
        @assert length(unique(ri_class[i])) == length(ri_class[i])
    end
    
    return ta_class, td_class, ri_class
end

# Extract all paths of requests from matched arrays of request IDs, arrival 
# and departure times. Returns a dictionary that maps a request ID to a 
# path dataframe
function getAllPaths(ri::Vector{Vector{Int64}}, ta::Vector{Vector{Float64}},
    td::Vector{Vector{Float64}}, classes::Vector{Tuple{T_qn, T_qn, Int64}},
    ext_arrs::Vector{Tuple{T_qn, T_qn, Int64}}, queue_type::Dict{T_qn, String})

    path = Dict{Int64, DataFrame}()
    path_err = Dict{Int64, DataFrame}()

    df_all = DataFrame(
        reqID = Int64[], 
        ta = Float64[],
        td = Float64[],
        class = Tuple{T_qn, T_qn, Int64}[])
    for (i, c) in enumerate(classes)
        df_all = vcat(df_all,  
            DataFrame(reqID = ri[i], ta = ta[i], td = td[i], class = c))
    end

    #arrival states are ext_arrs: n 1, st: n 1
    #departure states are ext_arrs: n max, st: n max
    arrival_states = []
    departure_states = []
    for (q, n, _) in classes
        if queue_type[q] == "ec"
            push!(arrival_states, (n[1], n[2], 1))
            push!(departure_states, (n[1], n[2], 
                sum([(n[1], n[2]) == (qc, nc) for (qc, nc, _) in classes])))
        end
    end
    for (q, n, _) in ext_arrs
        push!(arrival_states, (n[1], n[2], 1))
        push!(departure_states, (n[1], n[2], 
            sum([(n[1], n[2]) == (qc, nc) for (qc, nc, _) in classes])))
    end


    # The path of each request must contain at least 1 class to where an arrival 
    # can occur to, and at least 1 class from which a departure can occur. 
    # Furthermore, each class can only be visited once. Since no clock 
    # syncrhonization between VMs, we cant simply check the first/last 
    sort!(df_all, [:reqID, :ta])
    requests = unique(df_all.reqID)
    k0 = 1
    for r in requests
        kf = k0
        while kf < size(df_all, 1) && df_all.reqID[kf+1] == r
            kf += 1
        end

        df_req = df_all[k0:kf, [:ta, :td, :class]]

        if sum((c -> c ∈ arrival_states).(df_req.class)) > 0 && 
                sum((c -> c ∈ departure_states).(df_req.class)) > 0 &&
                length(unique(df_req.class)) == length(df_req.class)
            path[r] = df_req
        else
            path_err[r] = df_req
        end

        k0 = kf+1
    end

    return path, path_err
end

# Returns two lists of all arrival/departures occuring to the set of
# classes 
function getArrivalDeparture(path::Dict{Int64, DataFrame},
        Cr::Vector{Tuple{T_qn, T_qn, Int64}})

    ta = Vector{Float64}()
    td = Vector{Float64}()
    ri = Vector{Int64}()

    for (req_id, r_df) in path
        inCr = false
        for i = 1:size(r_df, 1)
            if !inCr && any([r_df.class[i] == c for c in Cr])
                append!(ta, r_df.ta[i])
                append!(ri, req_id)
                inCr = true
            elseif inCr && !any([r_df.class[i] == c for c in Cr])
                append!(td, r_df.td[i-1])
                inCr = false
            elseif (i == size(r_df, 1)) && inCr 
                append!(td, r_df.td[i])
                inCr = false
            end
        end
    end

    idx = sortperm(ta)
    ta = ta[idx]
    td = td[idx]
    ri = ri[idx]

    @assert length(ta) == length(td) == length(ri)
    @assert all(ta .<= ri)

    return ta, td, ri
end

# Join data of arrival/departures in classes into their respective queues. 
# Also sorts the data, but returns the sorting order to enable one to re-obtain which 
# index belongs to what class. 
function joinDataOverQueues(ta::Vector{Vector{Float64}}, td::Vector{Vector{Float64}},
        classes::Vector{Tuple{T_qn, T_qn, Int64}})

    queues = unique(getindex.(classes, 1))

    sort_queue = Array{Array{Int64, 1}, 1}(undef, length(queues))
    ta_queue = Array{Array{Float64, 1}, 1}(undef, length(queues))
    td_queue = Array{Array{Float64, 1}, 1}(undef, length(queues))
    for (i, q) in enumerate(queues)
        idx = findall([q == qc for (qc, _, _) in classes])
        ta_queue[i] = vcat(ta[idx]...)
        td_queue[i] = vcat(td[idx]...)

        idx_sort = sortperm(ta_queue[i])
        sort_queue[i] = idx_sort
        ta_queue[i] = ta_queue[i][idx_sort]
        td_queue[i] = td_queue[i][idx_sort]
    end

    return ta_queue, td_queue, sort_queue
end

# Extract the position mapping between queues, classes and phase states
function getPhasePos(S::Array{Int64,1}, Q::Int64, C::Array{Int64,1})

    # Which state belongs to which queue
    M = vcat([i*ones(Int64, sum((S[sum(C[1:i-1])+1:sum(C[1:i])]))) for i in 1:Q]...)

    # Which class belongs to which queue
    Mc = vcat([i*ones(Int64, C[i]) for i in 1:Q]...)

    # Which state belongs to which class
    N = vcat([i*ones(Int64, S[i]) for i in 1:sum(C)]...)

    return M, Mc, N
end

# Get the service times over a queue given a PS(K) model
function getServiceTimes(ta::Vector{Float64}, td::Vector{Float64}, K::Int64)

    if K == typemax(Int64)
        return td - ta
    end

    s = zeros(size(ta))
    q = getQueueLengths(ta, td)

    ka = 1
    kd = 1

    for i = 1:length(ta)
        if isinf(td[i])
            s[i] = NaN
            continue
        end

        # Move forward from old position to find the position of arrival of request i
        while q[ka, 1] < ta[i]
            ka += 1
        end

        # Necessary for exactness due to numerical issues when truncating queue events
        if ka > 0
            s[i] += (q[ka, 1] - ta[i]) * K / max(q[ka-1, 2], K)
        end

        kd = ka
        while kd < size(q, 1) && q[kd+1, 1] < td[i]
            s[i] += (q[kd+1, 1] - q[kd, 1]) * K / max(q[kd, 2], K)
            kd += 1
        end
        s[i] += (td[i] - q[kd, 1]) * K / max(q[kd, 2], K)
    end

    return s
end

# Get the service times over the classes
function getServiceTimes(ta_c::Vector{Vector{Float64}}, 
        td_c::Vector{Vector{Float64}}, queue_servers::Dict{T_qn, Int64},
        classes::Vector{Tuple{T_qn, T_qn, Int64}})

    ta_join, td_join, sort_join = joinDataOverQueues(ta_c, td_c, classes)
    queues = unique(getindex.(classes, 1))

    ts_c = Vector{Vector{Float64}}(undef, length(ta_c))
    for (i, queue) in enumerate(queues)
        idx = findall([queue == q for q in getindex.(classes, 1)])
        ts_join = zeros(size(ta_join[i]))
        ts_join[sort_join[i]] = getServiceTimes(ta_join[i], td_join[i], 
            queue_servers[queue])
        
        class_split = [0; cumsum(length.(ta_c[idx]))]
        for (j, k) in enumerate(idx)
            ts_c[k] = ts_join[class_split[j]+1:class_split[j+1]]
        end
    end

    @assert all(length.(td_c) .== length.(ts_c))

    return ts_c
end