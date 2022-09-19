
import os
import random
from glob import glob
import numpy as np
import yaml


from LoadGenerator_online import LoadGeneratorOptions

simSecDur = [400] * 60

# Load generator using an open connection 
opt_open = LoadGeneratorOptions()
opt_open.url = "http://10.0.1.106:31115/detect/" 
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
    #mid = len(simSecDur) // 2
    #ratios = [1/14]*mid + [1/21]*(len(simSecDur) - mid)
    ratios = [1/15] * 20 + [1/22.5] * 40
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
opt_open.exportDict = {
    "simSecDur": simSecDur
}

# load generator with closed connection example
opt_closed = LoadGeneratorOptions()
opt_closed.url = "http://10.0.1.106:31115/fetch/"
opt_closed.setID("st")
opt_closed.logFile = "load_generator_st.csv"
opt_closed.settingsFile = "settings_st.yaml"
opt_closed.loadType = "closed"
opt_closed.connections = 50
opt_closed.simTime = sum(simSecDur)
opt_closed.printout = False

# Nested function for generating either the wait-time or interarrival time.
def interarrivalTimes_2(simSecDur):
    simIntEnd = np.cumsum(simSecDur)
    ratios = [1.0] * 20 + [1/2.0] * 40
    def timefunc(t):
        i = 0
        for ssd in simIntEnd[:-1]:
            if t < ssd:
                break
            i += 1
        return np.random.exponential(ratios[i])
    return timefunc
opt_closed.genTimeFunc = interarrivalTimes_2(simSecDur)

# Nested function for creating a function closure for the data to send
def img_func_storage(datapath):
    image_paths = []
    for dir,_,_ in os.walk(datapath):
        image_paths.extend(glob(os.path.join(dir, "*.jpg")))
    def rand_img_func(t):
        filename = image_paths[random.randint(0, len(image_paths)-1)]
        return  {'imgfile': filename.split("/")[-1]}
    return rand_img_func
opt_closed.dataToSend = img_func_storage("../../data/") #lambda t: {'filename': "1"}
opt_closed.exportDict = {
    "simSecDur": simSecDur
}
