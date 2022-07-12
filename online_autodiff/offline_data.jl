using YAML

function run()
    logfolder = joinpath(@__DIR__, "..", "data", "logs")
    lock_file = joinpath(@__DIR__, "lock")
    last_time = mtime(lock_file)
    p0 = [2.2, 0.0]
    while true
        println("Data recorded, sending to julia")
        # Write new data paths
        pv = exp(p0[1]) / sum(exp, p0)
        idx = round(Int, 20 * pv) + 1
        YAML.write_file(joinpath(@__DIR__, "input.yaml"), Dict(
            "p0" => p0,
            "logfolder" => logfolder,
            "tracefolder" => joinpath(logfolder, "traces", "sample$(idx)")
        ))
        # Signal new data ready
        touch(lock_file)
        last_time = mtime(lock_file)
        # Wait for new p
        println("Waiting for updated p")
        while last_time == mtime(lock_file)
            sleep(0.5)
        end
        # Read new p
        input_dict = YAML.load_file(joinpath(@__DIR__, "output.yaml"); dicttype=Dict{String, Any})
        p0 = input_dict["p_next"]
        println("New p = $(exp.(p0) / sum(exp, p0))")
    end
end

run()