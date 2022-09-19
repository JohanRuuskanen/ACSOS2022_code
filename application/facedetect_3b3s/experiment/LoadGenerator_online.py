"""
Load balancer using asynchronous posting of https requests.

"""

import os
import time

import queue

import aiohttp
import asyncio

import numpy as np

import yaml

from urllib.parse import urlparse

class LoadGeneratorOptions:
    
    # A load generator options defines the following:
    #   url (string) - where to send the requests
    #   logFile (string) - filname to where to log requests
    #   settingsFile (string) - filename for where to save the settings
    #   loadType (string) - either open/closed
    #   connections (int) - number of generators, only used for closed
    #   simTime (float/int) - simulation time in seconds
    #   genTimeFunc (function t -> float) - time dependent function of the inter-arrival
    #       time or waiting time for the generator(s)
    #   dataToSend (function t -> Dict) - time dependent function of the data dictionary to send
    #   postProcessing (funtction Dict -> 0) - post processing on the data dictionary after 
    #       request is completed
    #   headersAdditional (function t -> Dict) - time dependent function for the additional
    #       headers to send
    #   responseSizeErrorLim (int) - size of response when it should be classified as an error
    #   exportDict (Dict) - Dictionary of extra key/values to export when calling toDict function
    def __init__(self):
        self.url = ""
        self.logFile = "load_generator.csv"
        self.settingsFile = "settings.yaml"
        self.id_str = ""
        self.id_nbr = "0"
        self.loadType = "open"
        self.connections = 0
        self.simTime = 0
        self.genTimeFunc = lambda t: 1
        self.dataToSend = lambda t: "0"
        self.postProcessing = lambda x: 0
        self.headersAdditional = lambda t: {}
        self.asyncioMaxCon = 250
        self.printout = True
        self.exportDict = {}

    def setID(self, string):
        self.id_str = string
        self.id_nbr = "".join([str(ord(s)) for s in string])

    def toDict(self):
        return {
            "url": self.url,
            "logFile": self.logFile,
            "settingsFile": self.settingsFile,
            "loadType": self.loadType,
            "connections": self.connections,
            "simTime": self.simTime,
            "asyncioMaxCon": self.asyncioMaxCon,
            "id_str": self.id_str,
            "id_nbr": self.id_nbr,
            **self.exportDict
        }

class LoadGenerator:

    sim_id = 0
    
    # Opt should be a LoadGeneratorOptions
    def __init__(self, opt):
        assert opt.loadType in ["open", "closed"]
        self.opt = opt
        self.call = urlparse(opt.url).path
        if len(self.call) == 0:
            self.call = '/'

    def run(self, simpath):
        if not os.path.exists(simpath):
                os.makedirs(simpath)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.main(loop, simpath))

    async def postrequest_open(self, session, count, waittime):
        ts_arrival = time.time()

        data = self.opt.dataToSend(ts_arrival - self.t0)

        req_id = str(count) + self.opt.id_nbr
        headers = {
            'x-request-id': req_id,
            'x-downstream-ip': self.opt.id_str,
            **self.opt.headersAdditional(ts_arrival - self.t0)
        }

        async with session.post(self.opt.url, data=data, headers=headers) as response:
            r = await response.read()
            self.opt.postProcessing(data)

            status = response.status
            us_ip = response.headers['x-upstream-ip'] if 'x-upstream-ip' in response.headers else ''

            ts_response = time.time()
            self.f.write("{},{},{},{},{},{},{},{},{}\n".format(
                req_id, count, self.call, ts_arrival-waittime, ts_arrival, ts_response, status, us_ip, len(r)))

            if self.opt.printout:
                print("{}; sim: {}, reqID: {}, clientID: {}, ta: {:10.4f}, td: {:10.4f}, tr: {:10.4f}, st: {}, sz: {}".format( \
                    self.opt.id_str, self.sim_id, req_id, count, ts_arrival - self.t0, 
                    waittime, ts_response - ts_arrival, status, len(r)))
            
            return r

    async def postrequest_closed(self, session, count, task_id):

        ts_arrival = time.time()

        await asyncio.sleep(self.opt.genTimeFunc(ts_arrival - self.t0))

        ts_departure = time.time()

        data = self.opt.dataToSend(ts_departure - self.t0)

        req_id = str(count) + self.opt.id_nbr
        headers = {
            'x-request-id': req_id,
            'x-downstream-ip': self.opt.id_str + "_" + str(task_id),
            **self.opt.headersAdditional(ts_departure - self.t0)
        }

        async with session.post(self.opt.url, data=data, headers=headers) as response:

            r = await response.read()
            self.opt.postProcessing(data)

            status = response.status
            us_ip = response.headers['x-upstream-ip'] if 'x-upstream-ip' in response.headers else ''

            ts_response = time.time()
            self.f.write("{},{},{},{},{},{},{},{},{}\n".format(
                req_id, task_id, self.call, ts_arrival, ts_departure, ts_response, status, us_ip, len(r)))

            if self.opt.printout:
                print("{}; sim: {}, reqID: {}, clientID: {}, ta: {:10.4f}, td: {:10.4f}, tr: {:10.4f}, st: {}, sz: {}".format( \
                    self.opt.id_str, self.sim_id, req_id, task_id, time.time()-self.t0, \
                    ts_departure - ts_arrival, ts_response - ts_departure, status, len(r)))
            
            return r

    async def main(self, loop, simpath):
        conn = aiohttp.TCPConnector(limit=self.opt.asyncioMaxCon)
        async with aiohttp.ClientSession(connector=conn) as session:
            tasks = set()
            self.t0 = time.time()
            await asyncio.sleep(0.001) #Keeps t0 from occationally not being the smallest timestamp
 
            self.f = open(os.path.join(simpath, self.opt.logFile), "w")
            self.f.write("req_id,client_id,remote_call,ts_arrival,ts_departure,ts_response,status,destination,respsize\n")

            count = 0
            t = 0
            pr_count = 1

            if self.opt.loadType == "closed":
                id_queue = queue.Queue()
                for i in range(self.opt.connections):
                    id_queue.put(i)

            # Used for open loop to correctly determine waiting time
            t_prev = time.time() 

            # Export settings
            settingsDict = {**self.opt.toDict(),
                "experimentTime": [self.t0, self.t0 + self.opt.simTime]}

            with open(os.path.join(simpath, self.opt.settingsFile), "w") as f:
                f.write(yaml.dump(settingsDict))

            # waittime 0, will wait 0.01 sec before 
            skip = False
            while t < self.opt.simTime:

                #if not self.opt.printout:
                #    if (t // 5 == pr_count):
                #        print("{}; sim: {},\t count: {},\t tasks: {}, \t t: {:10.4f}".format( \
                #            self.opt.id_str, self.sim_id, count, sum([not task.done() for task in tasks]), \
                #                 t))
                #        pr_count += 1

                #print("{} {} {}".format(self.opt.id_str, t, self.opt.genTimeFunc(time.time() - self.t0)))
                if self.opt.loadType == "closed":
                    if len(tasks) >= self.opt.connections:
                        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            id_queue.put(task.id)
                    
                    if self.opt.genTimeFunc(time.time() - self.t0) == -1:
                       skip = True
                    else:
                        task_id = id_queue.get()
                        t = loop.create_task(self.postrequest_closed(session, count, task_id))
                        t.id = task_id
                        tasks.add(t)

                if self.opt.loadType == "open":

                    # Insert the comp time here aswell to make the wait time correct
                    arr_t = self.opt.genTimeFunc(time.time() - self.t0)
                    t_prev = time.time()

                    if arr_t == -1:
                        skip = True
                    else:
                        dt = time.time() - t_prev
                        waittime = max(arr_t - dt, 0)

                        await asyncio.sleep(waittime)
                        tasks.add(loop.create_task(self.postrequest_open(session, count, waittime)))
            
                if skip:
                    if len(tasks) > 0:
                        await asyncio.wait(tasks)
                    time.sleep(0.01)
                    skip = False
                else:
                    count += 1

                t = time.time() - self.t0

            # Cancel remaining tasks and wait for them
            await asyncio.wait(tasks)
            
            self.f.close()
            
            return tasks