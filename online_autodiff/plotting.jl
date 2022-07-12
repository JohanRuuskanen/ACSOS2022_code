using PyPlot, CSV

runid = "logs"
tag = "default"
data = CSV.File("/home/ubuntu/application/facedetect/experiment/$runid/data_$(tag).csv")

p = data.p
ql = data.ql
ql_data = data.ql_data
p95 = data.p95
p95_data = data.p95_data
cost = data.cost
cost_data = data.cost_data
violating = data.violating
p95_pred = data.p95_pred

p95_lim = 0.55
cost_per_p95 = 10.0

figure(32)
clf()
subplot(1, 3, 1)
plot(p, ql_data, "C0", label="data")
plot(p, ql, "*C1--", label="model")
xlim([0, 1])
ylim([0.9*minimum(ql_data), 1.1*maximum(ql_data)])
title("Total queue length")
legend()
subplot(1, 3, 2)
plot(p, p95_data, "C0", label="data")
plot(p, p95, "*C1--", label="model")
plot(p, p95_pred, "oC2:", label="pred")
plot([0, 1], [p95_lim, p95_lim], "k--", label="limit")
title("p95 response time")
xlim([0, 1])
ylim([0.9*minimum(p95_data), 1.1*maximum(p95_data)])
legend()
subplot(1, 3, 3)
# Use violating to plot this nicer?
plot(p, cost_data, "C0", label="data")
plot(p, p95_data .* cost_per_p95, "C3", label="p95 data")
plot(p, cost, "*C1--", label="model")
xlim([0, 1])
ylim([0.9*minimum(cost_data), 1.1*maximum(cost_data)])
title("Total cost \n (what we minimize over)")
legend()

savefig("/home/ubuntu/application/facedetect/experiment/$runid/plot_$(tag).png")
gcf()