using PyPlot, CSV

runid = "logs"
tag = "default"
data = CSV.File("/home/ubuntu/application/facedetect_3b/experiment/$runid/data_$(tag).csv")

p = [data.p1 data.p2 data.p3]
ql = data.ql
ql_data = data.ql_data
p95 = data.p95
p95_data = data.p95_data
cost = data.cost
cost_data = data.cost_data
violating = data.violating
p95_pred = data.p95_pred

p95_lim = 0.6

t = 0:5:5*length(violating)-1

begin
    figure(32)
    clf()
    subplot(2, 2, 1)
    plot(t, ql_data, "C0", label="data")
    plot(t, ql, "*C1--", label="model")
    ylim([0.9*minimum(ql_data), 1.1*maximum(ql_data)])
    title("Total queue length")
    legend()
    subplot(2, 2, 2)
    plot(t, p95_data, "C0", label="data")
    plot(t, p95, "*C1--", label="model")
    # Add one timestep (5 min) to compare when guess if for with actual value then
    plot(t .+ 5, p95_pred, "oC2:", label="pred") 
    plot([minimum(t), maximum(t)], [p95_lim, p95_lim], "k--", label="limit")
    title("p95 response time")
    ylim([0.9*minimum(p95_data), 1.1*maximum(p95_data)])
    legend()
    subplot(2, 2, 3)
    plot(t, cost_data, "C0", label="data")
    plot(t, cost, "*C1--", label="model")
    ylim([0.9*minimum(cost_data), 1.1*maximum(cost_data)])
    title("Total cost \n (what we minimize over)")
    legend()
    subplot(2, 2, 4)
    plot(t, p[:, 1], "C0", label="p1")
    plot(t, p[:, 2], "C1", label="p2")
    plot(t, p[:, 3], "C2", label="p3")
    title("Placement probabilities")
    legend()
end

gcf()

savefig("/home/ubuntu/application/facedetect_3b/experiment/$runid/plot_$(tag).png")