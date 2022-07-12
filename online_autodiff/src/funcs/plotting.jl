

# Plot the simulation intervals
function plotSimIntervals(simInts::Vector{Tuple{Float64, Float64}}, a_scale::Number)
    for (k, (t0, tf)) in enumerate(simInts)
        fill_between([t0, tf], zeros(2), a_scale*ones(2), alpha=0.2, zorder=1, color="C$(k-1)")
    end
end

# Plot the simulation intervals, if elements of simInts is in DateTime format
function plotSimIntervals(simInts::Vector{Tuple{DateTime, DateTime}}, a_scale::Number, t_shift::Number) 
    plotSimIntervals([datetime2unix.(simInt) .- t_shift for simInt in simInts], a_scale)
end 

# Calculate and plot the CDF given a data vector at the points indicated in x. 
# Returns a matrix where first column is the values of the CDF evaluated at the 
# given values x. The second and third columns are filled with NaN values, 
# where the only non-NaN value marks the mean or α-quantile
function cdf_plot_data(x::Vector{Float64}, data::Vector{Float64}; α::Float64=0.95, 
        color::String="C0", label::String="Data", do_plot::Bool=true)

    cdf_func = cdfData(data)
    mean_val = mean(data)
    quant_val = findQuantileData(data, α)

    y = do_plot ? cdf_func.(x) : zeros(length(x))
    y_m = NaN*ones(size(x))
    y_a = NaN*ones(size(x))
    y_m[findfirst(z -> z > mean_val, x)] = cdf_func(mean_val)
    y_a[findfirst(z -> z > quant_val, x)] = cdf_func(quant_val)

    if do_plot
        plot(x, y, color, label=label)
        plot(mean_val, cdf_func(mean_val), color*"o")
        plot(quant_val, cdf_func(quant_val), color*"^")
    end

    return hcat(y, y_m, y_a)
end

# Calculate and plot the given CDF at the points indicated in x. Returns a matrix where 
# first column is the values of the CDF evaluated at the given values x. The second 
# and third columns are filled with NaN values, where the only non-NaN value marks the 
# mean or α-quantile
function cdf_plot(x::Vector{Float64}, cdf_func::Function;  α::Float64=0.95, 
        color::String="C0", label::String="", do_plot::Bool=true)
    
    y = do_plot ? cdf_func.(x) : zeros(length(x))
    y_m = NaN*ones(size(x))
    y_a = NaN*ones(size(x))

    do_plot ? plot(x, y, color, label=label) : 0
    try
        mean_val = quadgk((t -> 1 - cdf_func(t)), 0, Inf)[1]
        quant_val = fzero((t -> cdf_func(t) - α), mean_val, order=0)

        y_m[findfirst(x -> x > mean_val, x)] = cdf_func(mean_val)
        y_a[findfirst(x -> x > quant_val, x)] = cdf_func(quant_val)

        if do_plot
            plot(mean_val, cdf_func(mean_val), color*"o")
            plot(quant_val, cdf_func(quant_val), color*"^")
        end
    catch e
        println("Warning $label: $e")
    end

    return hcat(y, y_m, y_a)
end