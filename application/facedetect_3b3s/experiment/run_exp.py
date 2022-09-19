import os
import gc
import time
import shutil
import subprocess
import pandas as pd
import json
import threading
import yaml
import copy

from pathlib import Path

from LoadGenerator_online import LoadGenerator

# Load the common settings
exec(open("lg_settings.py").read())

loadgenerators = [LoadGenerator(opt_open), LoadGenerator(opt_closed)]

data_keys = ["req_id",
            "timestamp",
            "method_call",
            "response_code",
            "duration",
            "duration_upstream",
            "bytes_sent", 
            "bytes_received", 
            "downstream_pod_ip",
            "upstream_cluster",
            "upstream_pod_ip"]

gather_time_buffer_before = 5
sample_time = 300

# Settings
FRONTEND_NAME = "frontend"
FRONTEND_CLUSTERS = ["cluster-1"]

BACKEND_V1_NAME = "backend-v1"
BACKEND_V1_CLUSTERS = ["cluster-1"]

BACKEND_V2_NAME = "backend-v2"
BACKEND_V2_CLUSTERS = ["cluster-2"]

BACKEND_V3_NAME = "backend-v3"
BACKEND_V3_CLUSTERS = ["cluster-3"]

STORAGE_V1_NAME = "storage-v1"
STORAGE_V1_CLUSTERS = ["cluster-1"]

STORAGE_V2_NAME = "storage-v2"
STORAGE_V2_CLUSTERS = ["cluster-2"]

STORAGE_V3_NAME = "storage-v3"
STORAGE_V3_CLUSTERS = ["cluster-3"]

ALL_NAMES = [FRONTEND_NAME, 
    BACKEND_V1_NAME, BACKEND_V2_NAME, BACKEND_V3_NAME,
    STORAGE_V1_NAME, STORAGE_V2_NAME, STORAGE_V3_NAME]
ALL_CLUSTERS = [FRONTEND_CLUSTERS, 
    BACKEND_V1_CLUSTERS, BACKEND_V2_CLUSTERS, BACKEND_V3_CLUSTERS,
    STORAGE_V1_CLUSTERS, STORAGE_V2_CLUSTERS, STORAGE_V3_CLUSTERS]

ROOT = "/home/ubuntu/run_on_gateway/clusters"

logpath = "/home/ubuntu/application/facedetect_3b3s/experiment/logs"

AUTODIFF_DIR = "/home/ubuntu/online_autodiff"

def log_data(rawdatapath):
    datafiles = []; dataprocs = []
    Path(rawdatapath).mkdir(parents=True, exist_ok=True)
    def start_data_gathering(name, clusters):
        df = []; dp = []
        for cluster in clusters:
            arg =   "kubectl --context={}".format(cluster).split() + \
                    "-n facedetect-3b3s logs -l app={} -c istio-proxy -f".format(name).split()
            f = os.path.join(rawdatapath, "{}-{}.log".format(name, cluster))
            df.append(open(f, "w"))
            dp.append(subprocess.Popen(arg, stdout=df[-1])) 
        return df, dp
    for (i, name) in enumerate(ALL_NAMES):
        df, dp = start_data_gathering(name, ALL_CLUSTERS[i])
        datafiles += df
        dataprocs += dp
    return dataprocs, datafiles

def post_process(tracepath, rawdatapath):
    for child in Path(rawdatapath).iterdir():
        logfile = str(child)
        if logfile.split('.')[-1] != 'log':
            continue
        lines = []
        with open(logfile, "r") as f:
            lines = f.readlines()
        data = []
        for line in lines:
            try: 
                d_tmp = json.loads(line)
                if set(data_keys) == set(d_tmp.keys()):
                    data.append([d_tmp[key] for key in data_keys])
            except json.decoder.JSONDecodeError:
                pass
        df = pd.DataFrame(data, columns=data_keys).drop_duplicates().sort_values("timestamp").reset_index(drop=True)
        datafile = os.path.join(tracepath, logfile.split('/')[-1].split('.')[0] + '.csv')
        df.to_csv(datafile, sep=",", encoding="utf-8")


if __name__ == "__main__":

    softmax = lambda w: np.exp(w) / np.sum(np.exp(w))

    def stringify_dict_values(P):
        P_s = copy.deepcopy(P)
        for service in W0.keys():
            for dest in W0[service].keys():
                P_s[service][dest] = ",".join(map(str, P[service][dest]))
        return P_s

    def softmax_over_W(W):
        P = copy.deepcopy(W)
        for service in W.keys():
            for dest in W[service].keys():
                P[service][dest] = softmax(W[service][dest])
        return P

    def Pdist(P_new, P):
        d = 0
        for service in P.keys():
            for dest in P[service].keys():
                d += np.sum(np.power(P_new[service][dest] - P[service][dest], 2))
        return np.sqrt(d)

    def headers_func(P):
        P_jsn = json.dumps(P)
        def get_headers(t):
            return {"lb-weights": P_jsn, 
                    "upstream-timeout": "300.0",
                    "storage-extraload": "20"
                }
        return get_headers

    if os.path.isdir(logpath):
        shutil.rmtree(logpath)
    os.mkdir(logpath)

    W0 = {
            'frontend': {
                'detect': [2.0, 0.0, 0.0],
                'fetch': [2.0, 0.0, 0.0]
            },
            'backend-v1': {
                'store': [0.0, 0.0, 0.0]
            },
            'backend-v2': {
                'store': [0.0, 0.0, 0.0]
            },
            'backend-v3': {
                'store': [0.0, 0.0, 0.0]
            },
        }

    ql_cost = {
        'backend-v1' : 6,
        'backend-v2' : 4,
        'backend-v3' : 1,
        'storage-v1' : 6,
        'storage-v2' : 4,
        'storage-v3' : 1,
    }
    
    opt_open.headersAdditional = headers_func(stringify_dict_values(softmax_over_W(W0)))
    opt_closed.headersAdditional = headers_func(stringify_dict_values(softmax_over_W(W0)))

    tracepath = os.path.join(logpath, "traces", "sample1")
    rawdatapath = os.path.join(tracepath, "raw")

    # Start data gathering
    dataprocs, datafiles = log_data(rawdatapath)
    time.sleep(gather_time_buffer_before)

    # Run the load generators
    loadgenerators[0].sim_id = "1"
    loadgenerators[1].sim_id = "1"

    lg_threads = [threading.Thread(target=lg.run, args=(logpath,)) for lg in loadgenerators]
    [th.start() for th in lg_threads]

    # After "sample_time" seconds, restart the data collection, transform the old
    # data into CSV files, and calculate stuff in Julia
    t_start = time.time()
    t0 = t_start
    updated_costs = False
    c = 1
    W = W0
    lock_file = os.path.join(AUTODIFF_DIR, "lock")
    last_mtime = os.path.getmtime(lock_file)
    while any([th.is_alive() for th in lg_threads]):
        time.sleep(0.1)
        if last_mtime != os.path.getmtime(lock_file):
            W_old = W
            with open(os.path.join(AUTODIFF_DIR, "output.yaml"), 'r') as f:
                W = yaml.load(f, Loader=yaml.FullLoader)["W_next"]
            print("NEW P, DISTANCE MOVED: {}".format(Pdist(softmax_over_W(W), softmax_over_W(W_old))))
            opt_open.headersAdditional = headers_func(stringify_dict_values(softmax_over_W(W)))
            opt_closed.headersAdditional = headers_func(stringify_dict_values(softmax_over_W(W)))
            last_mtime = os.path.getmtime(lock_file)

            # start new gathering 
            tracepath = os.path.join(logpath, "traces", "sample" + str(c+1))
            rawdatapath = os.path.join(tracepath, "raw")

            #dataprocs_old = dataprocs
            #datafiles_old = datafiles

            dataprocs, datafiles = log_data(rawdatapath)

            gc.collect()

            #[dp.terminate() for dp in dataprocs_old]
            #[df.close() for df in datafiles_old]

            t0 = time.time()
            c += 1


        if time.time() - t0 > sample_time:
            t0_itr = time.time()

            # Print update itr started
            print("UPDATE ITERATION STARTED itr: {}, at: {:5.2f}".format(c, time.time() - t_start))

            # terminate data gathering
            [dp.terminate() for dp in dataprocs]
            [dp.wait() for dp in dataprocs]
            [df.close() for df in datafiles]

            # Post process data
            post_process(tracepath, rawdatapath)

            # Write data for julia script
            with open(os.path.join(AUTODIFF_DIR, "input.yaml"), 'w') as f:
                datadict = {
                    'W': W,
                    'logfolder': logpath,
                    'tracefolder': tracepath,
                    'cost_per_ql': ql_cost
                }
                print("DATA WRITTEN tracepath={}".format(datadict["tracefolder"]))
                yaml.dump(datadict, f)
                f.flush() # Make sure all is on disk
            with open(lock_file, 'w') as f:
                print("WRITING TO LOCK FILE")
                f.write("iter {}".format(c))
                f.flush() # Make sure all is on disk
            last_mtime = os.path.getmtime(lock_file) # But only signal julia, not self
            print("NEW MTIME {}".format(last_mtime))
            
            # Printout update itr ended
            print("iteration ended, dt_itr: {:5.2f}\n".format(time.time() - t0_itr))

            # Hack to just make sure it does not run again until we have new p
            t0 = 1e100 
    
        # Update the costs to test this disturbance 
        if (not updated_costs) and (time.time() - t_start > 40 * 400):
            print("COSTS UPDATED AT: {:5.2f}\n".format(time.time() - t_start))
            ql_cost["storage-v1"] = 1
            ql_cost["storage-v2"] = 4
            ql_cost["storage-v3"] = 6
            updated_costs = True
    
    [th.join() for th in lg_threads]

    # Terminate gathering
    [dp.terminate() for dp in dataprocs]
    [dp.wait() for dp in dataprocs]
    [df.close() for df in datafiles]