using YAML

function run()
    logfolder = "/home/ubuntu/application/facedetect/experiment/logs"
    lock_file = joinpath(@__DIR__, "lock")
    last_time = mtime(lock_file)
    p0 = 0.95
    while true
        println("Data recorded, sending to julia")
        # Write new data paths
        idx = round(Int, 20 * p0) + 1
        YAML.write_file(joinpath(@__DIR__, "input.yaml"), Dict(
            "p0" => p0,
            "logfolder" => logfolder,
            "tracefolder" => joinpath(logfolder, "traces", "sample$(idx)"),
            "cost_per_ql" => Dict("backend-v1" => 3, "backend-v2" => 1)
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
        println("New p = $p0")
    end
end

run()