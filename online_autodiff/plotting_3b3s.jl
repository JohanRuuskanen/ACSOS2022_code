using PyPlot, CSV

runid = "logs"
tag = "default"
data = CSV.File("/home/ubuntu/application/facedetect_3b3s/experiment/$runid/data_$(tag).csv")

frontend_detect_p = hcat([data["frontend-detect-p$i"] for i = 1:3]...)
frontend_fetch_p = hcat([data["frontend-fetch-p$i"] for i = 1:3]...)
backend_v1_store_p = hcat([data["backend-v1-store-p$i"] for i = 1:3]...)
backend_v2_store_p = hcat([data["backend-v2-store-p$i"] for i = 1:3]...)
backend_v3_store_p = hcat([data["backend-v3-store-p$i"] for i = 1:3]...) 

ql = data.ql
ql_data = data.ql_data
p95_open = data.p95_open
p95_data_open = data.p95_data_open
p95_pred_open = data.p95_pred_open
p95_closed = data.p95_closed
p95_data_closed = data.p95_data_closed
p95_pred_closed = data.p95_pred_closed
cost = data.cost
cost_data = data.cost_data
cost_only_queues = data.cost_only_queues
cost_only_queues_data = data.cost_only_queues_data
violating_open = data.violating_open
violating_closed= data.violating_closed

p95_lim_open = 1.25
p95_lim_closed = 0.4

t = 0:5:5*length(violating_open)-1

begin
    figure(32)
    clf()
    subplot(4, 1, 1)
    plot(t, ql_data, "C0", label="data")
    plot(t, ql, "*C1--", label="model")
    ylim([0, 1.5*ql[1]])
    #ylim([0.9*minimum(ql_data), 1.1*maximum(ql_data)])
    title("Total queue length")
    #legend()
    #([0, length(violating_open)-1])
    subplot(4, 1, 2)
    plot(t, p95_data_open, "C0", label="data")
    plot(t, p95_open, "*C1--", label="model")
    # Add one timestep (5 min) to compare when guess if for with actual value then
    plot(t .+ 5, p95_pred_open, "oC2:", label="pred") 
    plot([minimum(t), maximum(t)], [p95_lim_open, p95_lim_open], "k--", label="limit")
    title("p95 response time")
    #ylim([0.9*minimum(p95_data_open), 1.1*maximum(p95_data_open)])
    #xlim([0, length(violating_open)-1])
    ylim([0, 1.5 *p95_lim_open])
    #legend()
    subplot(4, 1, 3)
    plot(t, p95_data_closed, "C0", label="data")
    plot(t, p95_closed, "*C1--", label="model")
    # Add one timestep (5 min) to compare when guess if for with actual value then
    plot(t .+ 5, p95_pred_closed, "oC2:", label="pred") 
    plot([minimum(t), maximum(t)], [p95_lim_closed, p95_lim_closed], "k--", label="limit")
    title("p95 response time")
    #ylim([0.9*minimum(p95_data_closed), 1.1*maximum(p95_data_closed)])
    #xlim([0, length(violating_open)-1])
    ylim([0, 1.5 *p95_lim_closed])
    #legend()
    subplot(4, 1, 4)
    plot(t, cost_data, "C0", label="data")
    plot(t, cost_only_queues_data, "C0--", label="data q")
    plot(t, cost, "*C1", label="model")
    #plot(t, cost_only_queues, "C1--", label="model q")

    #ylim([0.9*minimum(cost_data), 1.1*maximum(cost_data)])
    #xlim([0, length(violating_open)-1])
    ylim([0, 1.5*cost[1]])
    title("Total cost \n (what we minimize over)")
    #legend()
    #subplot(2, 2, 4)
    #plot(t, p[:, 1], "C0", label="p1")
    #plot(t, p[:, 2], "C1", label="p2")
    #plot(t, p[:, 3], "C2", label="p3")
    #title("Placement probabilities")
    #legend()
end
gcf()
savefig("/home/ubuntu/application/facedetect_3b3s/experiment/$runid/plot_$(tag).png")

figure(33)
clf()
subplot(5, 1, 1)
plot(t, frontend_detect_p)
ylim([0, 1])
subplot(5, 1, 2)
plot(t, frontend_fetch_p)
ylim([0, 1])
subplot(5, 1, 3)
plot(t, backend_v1_store_p)
ylim([0, 1])
subplot(5, 1, 4)
plot(t, backend_v2_store_p)
ylim([0, 1])
subplot(5, 1, 5)
plot(t, backend_v3_store_p)
ylim([0, 1])

gcf()
savefig("/home/ubuntu/application/facedetect_3b3s/experiment/$runid/plot_probs_$(tag).png")
