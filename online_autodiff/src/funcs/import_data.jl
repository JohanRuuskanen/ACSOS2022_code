
# Reads all loadgenerator files, and returns a vector of dictionaries, that
# maps the load generator type to a dataframe for each simulation
function readLoadGenData(loadgenfiles::Vector{Vector{String}}, simSettings::Vector{T_simSetting}, 
        simSecDur::Vector{Float64}; plotting=true, verbose=true, fignbr=1)

    if verbose printstyled("Read the LoadGen data\n",bold=true, color=:magenta); end

    if verbose; printstyled("Reading loadgenfiles\n",bold=true); end

    # Extract the load generator identifiers
    lgs = collect(keys(simSettings[1]))

    # Extract load generator dataframes
    df_load = Vector{Dict{String, DataFrame}}(undef, length(loadgenfiles))
    for (k, f_list) in enumerate(loadgenfiles)
        df_load[k] = Dict{String, DataFrame}()
        for (j, lg) in enumerate(lgs)
            df_load[k][lg] = CSV.read(f_list[j], DataFrame) 
        end
    end

    if verbose; printstyled("Transforming data\n",bold=true); end
    
    failures = Vector{Dict{String, DataFrame}}(undef, length(loadgenfiles))
    missing_vals = Vector{Dict{String, DataFrame}}(undef, length(loadgenfiles))
    for k = 1:length(loadgenfiles)
        failures[k] = Dict{String, DataFrame}()
        missing_vals[k] = Dict{String, DataFrame}()
        for lg in lgs
            df = df_load[k][lg]
            df.ts_arrival = unix2datetime.(df.ts_arrival)
            df.ts_departure = unix2datetime.(df.ts_departure)
            df.ts_response = unix2datetime.(df.ts_response)
    
            failures[k][lg] = df[df.status .!= 200, :]
            missing_vals[k][lg] = df[.!completecases(df), :]

            df = df[completecases(df) .& (df.status .== 200), :]
            df_load[k][lg] = df

            # Test load generator data
            @assert all(df.ts_arrival .>= DateTime(0))
            @assert length(df.req_id) == length(unique(df.req_id))
            @assert all(df.ts_arrival .<= df.ts_departure)
            @assert all(df.ts_departure .<= df.ts_response)
            @assert datetime2unix(minimum(df.ts_arrival)) - 
                datetime2unix(simSettings[k][lg]["experimentTime"][1]) > -1
        end
    end

    # Plot the different queue lengths in the generator, and the system as a whole
    figure(fignbr)
    clf()
    df_all = Dict{String, DataFrame}()
    for (k, lg) in enumerate(lgs)
        df_all[lg] = vcat(get.(df_load, lg, 0)...)
        sort!(df_all[lg], :ts_arrival)

        q_load = getQueueLengths(datetime2unix.(df_all[lg].ts_arrival), 
            datetime2unix.(df_all[lg].ts_departure), start_time=datetime2unix(experiment_start))
        q_load = [q_load; [datetime2unix(experiment_end) 0]]
        q_system = getQueueLengths(datetime2unix.(df_all[lg].ts_departure), 
            datetime2unix.(df_all[lg].ts_response), start_time=datetime2unix(experiment_start))
        q_system = [q_system; [datetime2unix(experiment_end) 0]]

        xl = [0, datetime2unix(experiment_end) - datetime2unix(experiment_start)]
        qmax = maximum(q_system[:, 2])
        
        if plotting
            subplot(3, length(lgs), k)
            step(q_load[:, 1] .- datetime2unix(experiment_start), q_load[:, 2], where="post")
            plotSimIntervals.(getSimSecIntervals(simSecDur, simSettings), maximum(q_load[:, 2]), datetime2unix(experiment_start))
            xlim(xl)
            title("Total requests in LG $(lg)")
    
            subplot(3, length(lgs), k + length(lgs))
            step(q_system[:, 1] .- datetime2unix(experiment_start), q_system[:, 2], where="post")
            plotSimIntervals.(getSimSecIntervals(simSecDur, simSettings), maximum(q_system[:, 2]), datetime2unix(experiment_start))
            xlim(xl)
            title("Total requests in system from $(lg)")

            subplot(3, length(lgs), k + 2*length(lgs))
            f_plot = datetime2unix.(vcat(get.(failures, lg, 0)...).ts_arrival) .- datetime2unix(experiment_start)
            m_plot = datetime2unix.(vcat(get.(missing_vals, lg, 0)...).ts_arrival) .- datetime2unix(experiment_start)
            plot(f_plot, ones(size(f_plot)), "*C0", label="failures")
            plot(m_plot, 2*ones(size(m_plot)), "*C1", label="Missing vals")
            legend()
            ylim([0, 3])
            xlim(xl)
        end
    end
  
    printstyled("Errors\n",bold=true, color=:red)
    for lg in lgs
        f_all = vcat(get.(failures, lg, 0)...)
        m_all = vcat(get.(missing_vals, lg, 0)...)
        
        println("\tfalures $lg: $(size(f_all, 1)/(size(df_all[lg], 1) + size(f_all, 1)) * 100) %")
        println("\tmissing $lg: $(size(m_all, 1)/(size(df_all[lg], 1) + size(m_all, 1)) * 100) %")
        println("\n")
    end

    if verbose; println(""); end

    return df_load
end

# Reads all loadgenerator files assuming a single simulation
function readLoadGenData(loadgenfiles::Vector{String}, simSettings::Dict{String, Dict{String, Any}})

    # Extract load generator dataframes
    df_load = Dict{String, DataFrame}()
    for lg_file in loadgenfiles
        df_load[split(split(lg_file, "_")[end], ".")[1]] = CSV.read(lg_file, DataFrame) 
    end

    # Extract the load generator identifiers
    lgs = collect(keys(df_load))

    failures = Dict{String, DataFrame}()
    missing_vals = Dict{String, DataFrame}()
    for lg in lgs
        df = df_load[lg]
        df.ts_arrival = unix2datetime.(df.ts_arrival)
        df.ts_departure = unix2datetime.(df.ts_departure)
        df.ts_response = unix2datetime.(df.ts_response)

        failures[lg] = df[df.status .!= 200, :]
        missing_vals[lg] = df[.!completecases(df), :]

        df = df[completecases(df) .& (df.status .== 200), :]
        df_load[lg] = df

        # Test load generator data
        @assert all(df.ts_arrival .>= DateTime(0))
        @assert length(df.req_id) == length(unique(df.req_id))
        @assert all(df.ts_arrival .<= df.ts_departure)
        @assert all(df.ts_departure .<= df.ts_response)
        @assert datetime2unix(minimum(df.ts_arrival)) - 
            datetime2unix(simSettings[lg]["experimentTime"][1]) > -1
    end

    return df_load
end

# Return the connection types per row in the given dataframe
function getConnectionTypes(df::DataFrame)
    return (x -> ismissing(x) ? x : split(x, "|")[1]).(df.upstream_cluster)
end

# Get all pod IPs visited by requests in the dataframe
function getPodIPs(df::DataFrame)
    conTypes = getConnectionTypes(df)
    podIPs = Vector{Union{Missing, String}}(undef, length(conTypes))

    for k = 1:size(conTypes, 1)
        if ismissing(conTypes[k])
            podIPs[k] = missing
        elseif conTypes[k] == "inbound"
            podIPs[k] = (x -> ismissing(x) ? x : split(x, ":")[1])(df.upstream_pod_ip[k])
        elseif conTypes[k] == "outbound"
            podIPs[k] = (x -> ismissing(x) ? x : split(x, ":")[1])(df.downstream_pod_ip[k])
        else
            error("connection type must be inbound/outbound")
        end
    end

    return podIPs
end

# For each tracefolder, extract the trace files gathered from Envoy into dataframes,
# and put them into dictionaries with the podIP as key
function readTraceData(tracefolders::Vector{String}, simSettings::Vector{T_simSetting}; 
        verbose=true, plotting=true, fignbr=2)

    if verbose; printstyled("Read the trace data\n",bold=true, color=:magenta); end

    function toMilliseconds(x::AbstractArray{T, 1}) where T <: Any
        y =  Array{Union{Missing, Millisecond}, 1}(undef, length(x))
        for k = 1:length(x)
            if typeof(x[k]) <: Number
                y[k] = Dates.Millisecond(x[k])
            else
                y[k] = missing
            end
        end
        return y
    end
    
    function readTraceFile(tracefile)
        df = CSV.read(tracefile, DataFrame, drop=[:Column1])
    
        # Remove all rows whos reqID end is not a valid Int64
        df.req_id = tryparse.(Int64, df.req_id)
        filter!(row -> !isnothing(row.req_id), df)
    
        # Transform arrival timestamp to DateTime
        if size(df, 1) > 0
            l = length(df.timestamp[1])
            @assert all(length.(df.timestamp) .== l)
            df.timestamp = DateTime.(getindex.(df.timestamp, 
                    [1:l-1 for _ in 1:length(df.timestamp)]))
        end
    
        # Transform duration and URT to Date objects
        df.duration = toMilliseconds(df.duration)
        df.duration_upstream = toMilliseconds(df.duration_upstream)
    
        return df
    end

    function extractSimDataFrame(df_all::DataFrame, simSetting::T_simSetting)

        # Identify rows for which ts_arrival, and ts_arrival+duration is within experimentTime
        expTime_start, expTime_end = getSimExpTimeIntervals(simSetting)
        
        in_expTime_start = expTime_start .<= df_all.timestamp
        in_expTime_ovl = df_all.timestamp .<= expTime_end
        in_expTime_end = df_all.timestamp .+ df_all.duration .<= expTime_end
        
        in_expTime = in_expTime_start .& in_expTime_end
        in_expTime_out = in_expTime_start .& in_expTime_ovl

        df_sim = df_all[in_expTime, :]

        removed = sum(in_expTime .⊻ in_expTime_out)
        if verbose; printstyled("Removing $removed datapoints partially outside sim interval\n",bold=true); end

        # Checking if there are duplicates of data (weird bug)
        df_size_old = size(df_sim, 1)
        df_sim = unique(df_sim)
        if verbose; printstyled("Removing $(df_size_old - size(df_sim, 1)) duplicate datapoints\n",bold=true); end

        podIPs_all = getPodIPs(df_sim)
        podIPs = unique(podIPs_all)
        df_pods = Dict{String, DataFrame}()
        for podIP in podIPs
            df_pods[podIP] = df_sim[podIP .== podIPs_all, :]
        end

        return checkSimDataFrame(df_pods, df_sim, simSetting)
    end

    function checkSimDataFrame(df_pods::Dict{String, DataFrame},
            df_sim::DataFrame, simSetting::T_simSetting)
        
        if verbose; printstyled("Checking data\n",bold=true); end

        # What is checked:
        #   Check that every request ID exists in the load generator
        #   Check that every request ID in a service has a single inbound connection
        #   Check that every upstream of an outbound connection has an inbound 
        #       request of the same ID
        #   Check that every downstream of an inbound connection has an outbound
        #       request of the same ID, if it's a part of the service graph.
        #   Check that the number of outbound connections are correct

        valid_req_ids = Vector{Int64}(undef, 0)
        for lg in keys(simSetting)
            append!(valid_req_ids, parse.(Int64, string.(0:simSetting[lg]["requests"]).*simSetting[lg]["id_nbr"]))
        end
        
        conTypes = getConnectionTypes(df_sim)
        df_inbound = df_sim[conTypes .== "inbound", :]
        df_outbound = df_sim[conTypes .== "outbound", :]

        #df_inbound.req_id = reqID_to_int(df_inbound.req_id)
        #df_outbound.req_id = reqID_to_int(df_outbound.req_id)

        podIPs = unique(getPodIPs(df_sim))

        # Create sets to quickly match upstream/downstream IPs and request IDs
        df_inbound_downstream_req = Array{Tuple{String, Int64}}(undef, sum(conTypes .== "outbound"))
        df_outbound_upstream_req = Array{Tuple{String, Int64}}(undef, sum(conTypes .== "inbound"))
        for k = 1:sum(conTypes .== "inbound")
            df_outbound_upstream_req[k] = (df_inbound.upstream_pod_ip[k], df_inbound.req_id[k])
        end
        for k = 1:sum(conTypes .== "outbound")
            df_inbound_downstream_req[k] = (df_outbound.downstream_pod_ip[k], df_outbound.req_id[k])
        end
        df_inbound_downstream_req = Set(df_inbound_downstream_req)
        df_outbound_upstream_req = Set(df_outbound_upstream_req)

        reqid_err_idx = Dict{Int64, Bool}()

        df_err_invalid_reqid = Dict{String, DataFrame}()
        df_err_wrong_inbounds = Dict{String, DataFrame}()
        df_err_missing_connection = Dict{String, DataFrame}()
        df_err_wrong_nbr_upstreams = Dict{String, DataFrame}()
        for podIP in podIPs

            df = df_pods[podIP]

            if isempty(df)
                continue
            end

            ct_pod = getConnectionTypes(df)

            # Rows with invalid request IDs
            err_idx = [(!(r ∈ valid_req_ids) && !get(reqid_err_idx, r, false)) for r in df.req_id]
            df_err_invalid_reqid[podIP] = df[err_idx, :]
            for r in df.req_id[err_idx]
                    reqid_err_idx[r] = true
            end
 
            # Rows whose request ID does not have 1 inbound
            req_inb = df.req_id[ct_pod .== "inbound"]
            err_idx = (r -> sum(r .== req_inb) != 1 && !get(reqid_err_idx, r, false)).(df.req_id)
            df_err_wrong_inbounds[podIP] = df[err_idx, :]
            for r in df.req_id[err_idx]
                reqid_err_idx[r] = true
            end

            # Check that each upstream/downstream of an outgoing connection has a 
            #    downstream/upstream of the same request ID (if it is in the graph)
            err_idx = zeros(Bool, size(df, 1))
            for k = 1:size(df, 1)
                if ct_pod[k] .== "outbound"
                    if !((df.upstream_pod_ip[k], df.req_id[k]) ∈ df_outbound_upstream_req) &&
                            !get(reqid_err_idx, df.req_id[k], false)
                        err_idx[k] = 1
                    end
                else
                    if df.downstream_pod_ip ∈ podIPs
                        if !((df.downstream_pod_ip[k], df.req_id[k]) ∈ df_inbound_downstream_req) &&
                                !get(reqid_err_idx, df.req_id[k], false)
                            err_idx[k] = 1
                        end
                    end
                end
            end
            df_err_missing_connection[podIP] = df[err_idx, :]
            for r in df.req_id[err_idx]
                reqid_err_idx[r] = true
            end
              
            # Check that each request has the correct number of upstream connections
            requests = df.req_id
            nbr_us = zeros(Int64, size(df, 1))
            call = ["-1" for k = 1:size(df, 1)]
            for (k, req) in enumerate(unique(requests))
                if get(reqid_err_idx, req, false)
                    continue
                end
                idx = df.req_id .== req
                nbr_us[idx] .= sum(idx)
                call[idx] .= df.method_call[idx][ct_pod[idx] .== "inbound"]
            end
            correct_us = zeros(Int64, size(df, 1))
            for c in unique(call)
                correct_us[call .== c] .= median(nbr_us[call .== c])
            end
            err_idx = (nbr_us .!= correct_us)
            df_err_wrong_nbr_upstreams[podIP] = df[err_idx, :]
            for r in df.req_id[err_idx]
                reqid_err_idx[r] = true
            end
        end

        all_errs = vcat(vcat.(values.([
            df_err_invalid_reqid, 
            df_err_wrong_inbounds, 
            df_err_missing_connection, 
            df_err_wrong_nbr_upstreams])...)...)
        reqid_err = collect(keys(reqid_err_idx))

        for podIP in podIPs
            if isempty(df_pods[podIP])
                continue
            end
            keep = [!(r ∈ reqid_err) for r in df_pods[podIP].req_id]
            df_pods[podIP] = df_pods[podIP][keep, :]

        end

        return df_pods, [   df_err_invalid_reqid, 
                            df_err_wrong_inbounds, 
                            df_err_missing_connection, 
                            df_err_wrong_nbr_upstreams]
    end

    df_pods_v = Vector{Dict{String, DataFrame}}(undef, length(tracefolders))
    err_df_v = Matrix{Dict{String, DataFrame}}(undef, length(tracefolders), 5)
    for (i, tracefolder) in enumerate(tracefolders)
        files = readdir(tracefolder)
        files = files[isfile.(joinpath.(tracefolder, files))]
        dfs_array = Vector{DataFrame}(undef, length(files))
        for (j, file) in enumerate(files)
            dfs_array[j] = readTraceFile(joinpath(tracefolder, file))
        end
        df_all = vcat(dfs_array...)
        sort!(df_all, :timestamp)

        # Removing all missing values
        err_df_v[i, 1] = Dict{String, DataFrame}()
        err_df_v[i, 1]["all"] = df_all[.!completecases(df_all), :]
        df_all = df_all[completecases(df_all), :]

        printstyled("Extracting sim $i / $(length(tracefolders))\n",bold=true, color=:green);
        df_pods_v[i], err_df_v[i, 2:end] =  extractSimDataFrame(df_all, simSettings[i])
    end

    @assert length(unique(sort.(collect.(keys.(df_pods_v))))) == 1
    podIPs = vcat(unique(sort.(collect.(keys.(df_pods_v))))...)

    #TODO: Update error checking

    # x1 - missing values
    # x2 - non-valid ID
    # x3 - wrong inbound
    # x4 - missing connection
    # x5 - wrong nbr upstreams
    x1 = vcat(get.(err_df_v[:, 1], "all", 0)...).timestamp
    x2 = [vcat(get.(err_df_v[:, 2], podIP, 0)...).timestamp for podIP in podIPs]
    x3 = [vcat(get.(err_df_v[:, 3], podIP, 0)...).timestamp for podIP in podIPs]
    x4 = [vcat(get.(err_df_v[:, 4], podIP, 0)...).timestamp for podIP in podIPs]
    x5 = [vcat(get.(err_df_v[:, 5], podIP, 0)...).timestamp for podIP in podIPs]

    total_size = sum(df->sum(size.(values(df), 1)), df_pods_v)

    err_prct = zeros(5)
    err_prct[1] = length(x1) / (length(x1) + total_size) * 100
    err_prct[2] = sum(length.(x2)) / (sum(length.(x2)) + total_size) * 100
    err_prct[3] = sum(length.(x3)) / (sum(length.(x3)) + total_size) * 100
    err_prct[4] = sum(length.(x4)) / (sum(length.(x4)) + total_size) * 100
    err_prct[5] = sum(length.(x5)) / (sum(length.(x5)) + total_size) * 100

    printstyled("Mean error\n",bold=true, color=:red)
    println("\tMissing Values: $(err_prct[1]) %")
    println("\tNon-valid IDs: $(err_prct[2]) %")
    println("\tWrong inbounds per req_id: $(err_prct[3]) %")
    println("\tMissing IDs in upstream: $(err_prct[4]) %")
    println("\tWrong nbr of upstreams: $(err_prct[5]) %")

    if plotting
        figure(fignbr)

        ts = [minimum(vcat(get.(values(simSettings[1]), "experimentTime", 0)...)),
            maximum(vcat(get.(values(simSettings[end]), "experimentTime", 0)...))]

        subplot(length(podIPs)+1, 1, 1)
        plot(x1, ones(length(x1)), "C0*")
        title("Missing values")
        xlim(ts)
        ylim([0, 2])

        for (k, podIP) in enumerate(podIPs)

            subplot(length(podIPs)+1, 1, k+1)
            plot(x2[k], ones(size(x2[k])), "C1*", label="non-valid ID")
            plot(x3[k], 2*ones(size(x3[k])), "C2*", label="wrong inbounds")
            plot(x4[k], 3*ones(size(x4[k])), "C3*", label="missing con")
            plot(x5[k], 4*ones(size(x5[k])), "C4*", label="wrong nbr us")

            xlim(ts)
            ylim([0, 5])
            if k == 1
                legend()
            elseif k == length(podIPs)
                xlabel("arrival time at pod")
            end
            ylabel("Violations")
            title("Pod $podIP")
            if k == length(podIPs)
                xlabel("Time")
            end
        end
    end

    return df_pods_v, err_df_v
end

# The single trace folder, extract all traces gathered from Envoy into dataframes,
# and put them into dictionaries with the podIP as key
function readTraceData(tracefolder::String, simSettings::T_simSetting; verbose=true)

    if verbose; printstyled("Read the trace data\n",bold=true, color=:magenta); end

    function toMilliseconds(x::AbstractArray{T, 1}) where T <: Any
        y =  Array{Union{Missing, Millisecond}, 1}(undef, length(x))
        for k = 1:length(x)
            if typeof(x[k]) <: Number
                y[k] = Dates.Millisecond(x[k])
            else
                y[k] = missing
            end
        end
        return y
    end

    function readTraceFile(tracefile)
        df = CSV.read(tracefile, DataFrame, drop=[:Column1])

        # Remove all rows whos reqID end is not a valid Int64
        df.req_id = tryparse.(Int64, df.req_id)
        filter!(row -> !isnothing(row.req_id), df)

        # Transform arrival timestamp to DateTime
        if size(df, 1) > 0
            l = length(df.timestamp[1])
            @assert all(length.(df.timestamp) .== l)
            df.timestamp = DateTime.(getindex.(df.timestamp, 
                    [1:l-1 for _ in 1:length(df.timestamp)]))
        end

        # Transform duration and URT to Date objects
        df.duration = toMilliseconds(df.duration)
        df.duration_upstream = toMilliseconds(df.duration_upstream)

        return df
    end

    function extractSimDataFrame(df_all::DataFrame, simSettings::T_simSetting)

        # Identify rows for which ts_arrival, and ts_arrival+duration is within experimentTime
        expTime_start, expTime_end = getSimExpTimeIntervals(simSettings)
        
        in_expTime_start = expTime_start .<= df_all.timestamp
        in_expTime_ovl = df_all.timestamp .<= expTime_end
        in_expTime_end = df_all.timestamp .+ df_all.duration .<= expTime_end
        
        in_expTime = in_expTime_start .& in_expTime_end
        in_expTime_out = in_expTime_start .& in_expTime_ovl

        df_sim = df_all[in_expTime, :]

        removed = sum(in_expTime .⊻ in_expTime_out)
        if verbose; printstyled("Removing $removed datapoints partially outside sim interval\n",bold=true); end

        # Checking if there are duplicates of data (weird bug)
        df_size_old = size(df_sim, 1)
        df_sim = unique(df_sim)
        if verbose; printstyled("Removing $(df_size_old - size(df_sim, 1)) duplicate datapoints\n",bold=true); end

        podIPs_all = getPodIPs(df_sim)
        podIPs = unique(podIPs_all)
        df_pods = Dict{String, DataFrame}()
        for podIP in podIPs
            df_pods[podIP] = df_sim[podIP .== podIPs_all, :]
        end

        return checkSimDataFrame(df_pods, df_sim)
    end


    function checkSimDataFrame(df_pods::Dict{String, DataFrame}, df_sim::DataFrame)
        
        if verbose; printstyled("Checking data\n",bold=true); end

        # What is checked:
        #   Check that every request ID in a service has a single inbound connection
        #   Check that every upstream of an outbound connection has an inbound 
        #       request of the same ID
        #   Check that every downstream of an inbound connection has an outbound
        #       request of the same ID, if it's a part of the service graph.
        #   Check that the number of outbound connections are correct

        conTypes = getConnectionTypes(df_sim)
        df_inbound = df_sim[conTypes .== "inbound", :]
        df_outbound = df_sim[conTypes .== "outbound", :]

        #df_inbound.req_id = reqID_to_int(df_inbound.req_id)
        #df_outbound.req_id = reqID_to_int(df_outbound.req_id)

        podIPs = unique(getPodIPs(df_sim))

        # Create sets to quickly match upstream/downstream IPs and request IDs
        df_inbound_downstream_req = Array{Tuple{String, Int64}}(undef, sum(conTypes .== "outbound"))
        df_outbound_upstream_req = Array{Tuple{String, Int64}}(undef, sum(conTypes .== "inbound"))
        for k = 1:sum(conTypes .== "inbound")
            df_outbound_upstream_req[k] = (df_inbound.upstream_pod_ip[k], df_inbound.req_id[k])
        end
        for k = 1:sum(conTypes .== "outbound")
            df_inbound_downstream_req[k] = (df_outbound.downstream_pod_ip[k], df_outbound.req_id[k])
        end
        df_inbound_downstream_req = Set(df_inbound_downstream_req)
        df_outbound_upstream_req = Set(df_outbound_upstream_req)

        reqid_err_idx = Dict{Int64, Bool}()

        df_err_wrong_inbounds = Dict{String, DataFrame}()
        df_err_missing_connection = Dict{String, DataFrame}()
        df_err_wrong_nbr_upstreams = Dict{String, DataFrame}()
        for podIP in podIPs

            df = df_pods[podIP]

            if isempty(df)
                continue
            end

            ct_pod = getConnectionTypes(df)

            # Rows whose request ID does not have 1 inbound
            req_inb = df.req_id[ct_pod .== "inbound"]
            err_idx = (r -> sum(r .== req_inb) != 1 && !get(reqid_err_idx, r, false)).(df.req_id)
            df_err_wrong_inbounds[podIP] = df[err_idx, :]
            for r in df.req_id[err_idx]
                reqid_err_idx[r] = true
            end

            # Check that each upstream/downstream of an outgoing connection has a 
            #    downstream/upstream of the same request ID (if it is in the graph)
            err_idx = zeros(Bool, size(df, 1))
            for k = 1:size(df, 1)
                if ct_pod[k] .== "outbound"
                    if !((df.upstream_pod_ip[k], df.req_id[k]) ∈ df_outbound_upstream_req) &&
                            !get(reqid_err_idx, df.req_id[k], false)
                        err_idx[k] = 1
                    end
                else
                    if df.downstream_pod_ip ∈ podIPs
                        if !((df.downstream_pod_ip[k], df.req_id[k]) ∈ df_inbound_downstream_req) &&
                                !get(reqid_err_idx, df.req_id[k], false)
                            err_idx[k] = 1
                        end
                    end
                end
            end
            df_err_missing_connection[podIP] = df[err_idx, :]
            for r in df.req_id[err_idx]
                reqid_err_idx[r] = true
            end
            
            # Check that each request has the correct number of upstream connections
            requests = df.req_id
            nbr_us = zeros(Int64, size(df, 1))
            call = ["-1" for k = 1:size(df, 1)]
            for (k, req) in enumerate(unique(requests))
                if get(reqid_err_idx, req, false)
                    continue
                end
                idx = df.req_id .== req
                nbr_us[idx] .= sum(idx)
                call[idx] .= df.method_call[idx][ct_pod[idx] .== "inbound"]
            end
            correct_us = zeros(Int64, size(df, 1))
            for c in unique(call)
                correct_us[call .== c] .= median(nbr_us[call .== c])
            end
            err_idx = (nbr_us .!= correct_us)
            df_err_wrong_nbr_upstreams[podIP] = df[err_idx, :]
            for r in df.req_id[err_idx]
                reqid_err_idx[r] = true
            end
        end

        reqid_err = collect(keys(reqid_err_idx))

        for podIP in podIPs
            if isempty(df_pods[podIP])
                continue
            end
            keep = [!(r ∈ reqid_err) for r in df_pods[podIP].req_id]
            df_pods[podIP] = df_pods[podIP][keep, :]

        end

        return df_pods, [   df_err_wrong_inbounds, 
                            df_err_missing_connection, 
                            df_err_wrong_nbr_upstreams]
    end

    df_pods = Dict{String, DataFrame}()
    err_df = Vector{Dict{String, DataFrame}}(undef, 4)

    files = readdir(tracefolder)
    files = files[isfile.(joinpath.(tracefolder, files))]
    dfs_array = Vector{DataFrame}(undef, length(files))
    ipname_bimap = Dict{String, String}()
    for (j, file) in enumerate(files)
        dfs_array[j] = readTraceFile(joinpath(tracefolder, file))

        name = split(file, "-cluster")[1]
        ip = filter(x -> !ismissing(x), unique(getPodIPs(dfs_array[j])))
        @assert length(ip) == 1
        ipname_bimap[name] = ip[1]
        ipname_bimap[ip[1]] = String(name) 
    end
    df_all = reduce(vcat, dfs_array) 
    sort!(df_all, :timestamp)

    # Removing all missing values
    err_df[1] = Dict{String, DataFrame}()
    err_df[1]["all"] = df_all[.!completecases(df_all), :]
    df_all = df_all[completecases(df_all), :]

    # Extract error-free dataframes over pods
    df_pods, err_df[2:end] = extractSimDataFrame(df_all, simSettings)

    podIPs = collect(keys(df_pods))

    # x1 - missing values
    # x2 - wrong inbound
    # x3 - missing connection
    # x4 - wrong nbr upstreams
    x1 = get(err_df[1], "all", 0).timestamp
    x2 = [get(err_df[2], podIP, 0).timestamp for podIP in podIPs]
    x3 = [get(err_df[3], podIP, 0).timestamp for podIP in podIPs]
    x4 = [get(err_df[4], podIP, 0).timestamp for podIP in podIPs]

    total_size = sum(size.(values(df_pods), 1))

    err_prct = Dict{String, Float64}()
    err_prct["missing values"] = length(x1) / (length(x1) + total_size) * 100
    err_prct["wrong inbound"] = sum(length.(x2)) / (sum(length.(x2)) + total_size) * 100
    err_prct["missing connections"] = sum(length.(x3)) / (sum(length.(x3)) + total_size) * 100
    err_prct["wrong nbr of upstream"] = sum(length.(x4)) / (sum(length.(x4)) + total_size) * 100

    return df_pods, err_df, err_prct, ipname_bimap
end



# Reads the trace data and generates the corresponding H object containing all necessary
# request data to perform distributed extraction of the queuing network. Assumes that 
# each request to a request type has exactly one inbound, and the same number of 
# outbounds. This is checked in the readTraceData function.
function traceData2Hr(df_pods::Dict{String, DataFrame})
    H = T_H()
    for podIP in keys(df_pods)

        df = df_pods[podIP]
        ct_inb = getConnectionTypes(df) .== "inbound"
        reqTypes = unique(df.method_call[ct_inb])

        # Find the indexes of all req types
        reqTypes_idx = Dict{String, Vector{Bool}}()
        for reqType in reqTypes
            reqTypes_idx[reqType] = zeros(Bool, size(df, 1))
        end
        for (k, req) in enumerate(unique(df.req_id))
            idx = df.req_id .== req
            reqTypes_idx[df.method_call[idx][ct_inb[idx]][1]] .|= idx 
        end

        @assert all(sum(values(reqTypes_idx)) .== 1)

        for reqType in reqTypes
            
            df_rt = sort(df[reqTypes_idx[reqType], :], [:req_id, :timestamp])

            requests = unique(df_rt.req_id)
            m = length(requests)

            Dr = Vector{String}(undef, m)
            Ur = Vector{Vector{Tuple{String, String}}}(undef, m)
            tr = Vector{DateTime}(undef, m)
            dtr = Vector{Millisecond}(undef, m)
            Tr = Vector{Vector{DateTime}}(undef, m)
            dTr = Vector{Vector{Millisecond}}(undef, m)
            dUTr = Vector{Vector{Millisecond}}(undef, m)

            ct = getConnectionTypes(df_rt)

            slice_start = 1
            # Assumes the same number of outgoing calls, and one single inbound
            nbr_us = sum(df_rt.req_id .== requests[1])-1
        
            for (i, req) in enumerate(requests)
                slice_int = slice_start:slice_start+nbr_us
                slice_start += nbr_us+1
                
                ct_req_inb = ct[slice_int] .== "inbound"
                # There should only be one incomming connection per request ID,
                @assert sum(ct_req_inb) == 1
                # Incomming connection should happen first
                @assert ct_req_inb[1] == 1
                # All requests in slice should have the same ID
                @assert all(df_rt.req_id[slice_int] .== req)
                # The request inbound should have the correct request type
                @assert df_rt.method_call[slice_int][ct_req_inb][1] .== reqType

                # The downstream client
                Dr[i] = df_rt.downstream_pod_ip[slice_int][1]

                # The upstream clients in order
                Ur[i] = [(x, y) for (x, y) in zip(
                    df_rt.upstream_pod_ip[slice_int][2:end], 
                    df_rt.method_call[slice_int][2:end])]

                # Timestamp of request connection
                tr[i] = df_rt.timestamp[slice_int][1]

                # Duration of request
                dtr[i] = df_rt.duration[slice_int][1]

                # Timestamps of upstream connections in order
                Tr[i] = df_rt.timestamp[slice_int][2:end]

                # Duration of upstream connections in order
                dTr[i] = df_rt.duration[slice_int][2:end]

                # Upstream duration of upstream connections in order.
                # This also contains network latencies in enovy!
                dUTr[i] = df_rt.duration_upstream[slice_int][2:end]
            end

            H[(podIP, reqType)] = sort(DataFrame(
                "reqID" => requests,
                "Dr" => Dr,
                "Ur" => Ur,
                "tr" => tr,
                "dtr" => dtr,
                "Tr" => Tr,
                "dTr" => dTr,
                "dUTr" => dUTr
            ), [:tr])

        end
    end

    return H
end

# Return the H data object truncated between the two values in the time span
function getHInTspan(H::T_H, tspan::Tuple{Float64, Float64})
    
    H_red = T_H()

    for key in keys(H)
        
        i1 = findfirst(t -> datetime2unix(t) > tspan[1], H[key].tr)
        i2 = findfirst(t -> datetime2unix(t) > tspan[2], H[key].tr)

        i1 = isnothing(i1) ? length(H[key].tr) : i1
        i2 = isnothing(i2) ? length(H[key].tr) : i2 - 1

        H_red[key] = H[key][i1:i2, :]
    end

    return H_red
end

# Returns the simulation start time and end time
function getSimExpTimeIntervals(simSetting::T_simSetting)
    simInts = sort(vcat(get.(values(simSetting), "experimentTime", 0)...))
    return simInts[1], simInts[end]
end


# Returns the duration intervals for each simulation section in each MC simulation
function getSimSecIntervals(simSecDur::Vector{T}, simSettings::Vector{Dict{String, Dict{String, Any}}}) where T <: Number
    ts = cumsum(simSecDur)
    s = Dates.Second.(floor.(Int64, ts)) + 
        Dates.Millisecond.(floor.(Int64, (ts - floor.(Int64, ts))*1000))
    ti = Vector{Vector{Tuple{DateTime, DateTime}}}(undef, MCsims)  
    for (k, simSetting) in enumerate(simSettings)
        simStart, _ = getSimExpTimeIntervals(simSetting)
        tv = vcat(simStart, simStart .+ s)
        ti_v = []
        for k = 2:length(tv)
            push!(ti_v, (tv[k-1], tv[k]))
        end
        ti[k] = ti_v
    end
    return ti
end