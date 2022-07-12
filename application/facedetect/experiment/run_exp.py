import os
import time
import shutil
import subprocess
import pandas as pd
import json
import threading
import yaml

from pathlib import Path

from LoadGenerator_online import LoadGenerator

# Load the common settings
exec(open("lg_settings_grid.py").read())

loadgenerator = LoadGenerator(opt_open)

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

ALL_NAMES = [FRONTEND_NAME, BACKEND_V1_NAME, BACKEND_V2_NAME]
ALL_CLUSTERS = [FRONTEND_CLUSTERS, BACKEND_V1_CLUSTERS, BACKEND_V2_CLUSTERS]

ROOT = "/home/ubuntu/run_on_gateway/clusters"

logpath = "/home/ubuntu/application/facedetect/experiment/logs"

def log_data(rawdatapath):
    dataprocs = []
    datafiles = []
    Path(rawdatapath).mkdir(parents=True, exist_ok=True)
    for frontend_cluster in FRONTEND_CLUSTERS:
        arg =   "kubectl --context={}".format(frontend_cluster).split() + \
                "-n facedetect-do logs -l app=frontend -c istio-proxy -f".split()
        f = os.path.join(rawdatapath, "frontend-{}.log".format(frontend_cluster))
        datafiles.append(open(f, "w"))
        dataprocs.append(subprocess.Popen(arg, stdout=datafiles[-1])) 
    for backend_cluster in BACKEND_V1_CLUSTERS:
        arg =   "kubectl --context={}".format(backend_cluster).split() + \
                "-n facedetect-do logs -l app=backend-v1 -c istio-proxy -f".split()  
        f = os.path.join(rawdatapath, "backend-v1-{}.log".format(backend_cluster))
        datafiles.append(open(f, "w"))
        dataprocs.append(subprocess.Popen(arg, stdout=datafiles[-1])) 
    for backend_cluster in BACKEND_V2_CLUSTERS:
        arg =   "kubectl --context={}".format(backend_cluster).split() + \
                "-n facedetect-do logs -l app=backend-v2 -c istio-proxy -f".split()  
        f = os.path.join(rawdatapath, "backend-v2-{}.log".format(backend_cluster))
        datafiles.append(open(f, "w"))
        dataprocs.append(subprocess.Popen(arg, stdout=datafiles[-1])) 
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

    if os.path.isdir(logpath):
        shutil.rmtree(logpath)
    os.mkdir(logpath)

    tracepath = os.path.join(logpath, "traces", "sample1")
    rawdatapath = os.path.join(tracepath, "raw")

    # Start data gathering
    dataprocs, datafiles = log_data(rawdatapath)
    time.sleep(gather_time_buffer_before)

    # Run the load generators
    loadgenerator.sim_id = "1"

    lg_thread = threading.Thread(target=loadgenerator.run, args=(logpath,))
    lg_thread.start()

    # After "sample_time" seconds, restart the data collection, transform the old
    # data into CSV files, and calculate stuff in Julia
    t_start = time.time()
    t0 = t_start
    c = 1
    p = 0.9
    while lg_thread.is_alive():
        time.sleep(0.1)
        if time.time() - t0 > sample_time:
            t0_itr = time.time()

            # Print update itr started
            print("UPDATE ITERATION STARTED itr: {}, at: {:5.2f}".format(c, time.time() - t_start))

            # terminate data gathering
            [dp.terminate() for dp in dataprocs]
            [df.close() for df in datafiles]

            # start new gathering 
            tracepath_new = os.path.join(logpath, "traces", "sample" + str(c+1))
            rawdatapath_new = os.path.join(tracepath_new, "raw")
            dataprocs_new, datafiles_new = log_data(rawdatapath_new)

            # Post process data
            post_process(tracepath, rawdatapath)

            # Printout update itr ended
            t0 = time.time()
            print("iteration ended, dt_itr: {:5.2f}\n".format(t0 - t0_itr))

            # Update variables
            tracepath = tracepath_new
            rawdatapath = rawdatapath_new
            dataprocs = dataprocs_new
            datafiles = datafiles_new
            c += 1

    lg_thread.join() 

    # Terminate gathering
    [dp.terminate() for dp in dataprocs]
    [df.close() for df in datafiles]
