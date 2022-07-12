
import os
import random
from glob import glob
import numpy as np
import yaml


from LoadGenerator_online import LoadGeneratorOptions

simSecDur = [300] * 40

# Load generator using an open connection 
opt_open = LoadGeneratorOptions()
opt_open.url = "http://x.x.x.x:32223/detect/" # Insert IP of e.g. master node on cluster 1
opt_open.setID("ir")
opt_open.logFile = "load_generator_ir.csv"
opt_open.settingsFile = "settings_ir.yaml"
opt_open.loadType = "open" 
opt_open.connections = 0
opt_open.simTime = sum(simSecDur)
opt_open.printout = False

# Nested function for generating either the wait-time or interarrival time.
def interarrivalTimes(simSecDur):
    simIntEnd = np.cumsum(simSecDur)
    ratios = [1/15]*14 + [1/22.5]*26
    def timefunc(t):
        i = 0
        for ssd in simIntEnd[:-1]:
            if t < ssd:
                break
            i += 1
        return np.random.exponential(ratios[i])
    return timefunc
opt_open.genTimeFunc = interarrivalTimes(simSecDur)

# Nested function for creating a function closure for the data to send
def img_func(datapath):
    image_paths = []
    for dir,_,_ in os.walk(datapath):
        image_paths.extend(glob(os.path.join(dir, "*.jpg")))
    def rand_img_func(t):
        return  {'imgfile': open(image_paths[random.randint(0, len(image_paths)-1)], "rb")}
    return rand_img_func
opt_open.dataToSend = img_func("../../data/")

# Nested function for processing after the data has sent in the request
def postprocessing():
    def ppfunc(D):
        return D['imgfile'].close()
    return ppfunc
opt_open.postProcessing = postprocessing()

# Nested function for creating a function closure for additional headers
def headers_func(p):
    p_exp = np.exp(p)
    p_prob = p_exp / np.sum(p_exp)
    lbWeights = ",".join(map(str, p_prob))
    print("SETTING lb-weights to {}".format(lbWeights))
    def get_headers(t):
        return {"lb-weights": lbWeights, 
                "upstream-timeout": "300.0",
                "storage-extraload": "20"
            }
    return get_headers
# opt_open.headersAdditional = headers_func(0.5)
opt_open.exportDict = {
    "simSecDur": simSecDur
}