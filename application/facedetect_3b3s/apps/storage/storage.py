import os
import io
import socket
import time
import gunicorn.app.base
import redis
import numpy as np
from flask import Flask, request, redirect, make_response, send_file

app = Flask(__name__)

STORAGE = "store/"
IMG_FILES = list(filter(lambda x: x.split(".")[-1] == "jpg", \
    [p for p in os.listdir(STORAGE)]))

PORT = 5002
REDIS_PORT = 6379

@app.route('/')
def startpage():
    return '''
        <!doctype html>
        <title>Storage</title>
        <h1>The storage microservice</h1>
        <p>use /store/ to store an image or /fetch/ to retrieve it</p>
        </form> '''

@app.route('/store/', methods=['GET', 'POST'])
def storage_set():

    if request.method == 'POST':
        if not len(request.files) == 1:
            print("send exactly 1 file", flush=True)
            return redirect(request.url)
        
        r = redis.Redis(host='localhost', port=REDIS_PORT, db=0)

        filename, file = next(iter(request.files.items()))
        file_data = file.read()

        storage_extraload = int(request.headers.get('storage-extraload', default="5")) 

        r.set(filename, file_data)

        # Perform extra write load
        for k in range(storage_extraload):
            r.set(filename, file_data)

        node_name = os.environ.get('NODE_NAME')
        if not node_name:
            node_name = "localhost"
        
        node_service = os.environ.get('NODE_SERVICE')
        if not node_service:
            node_service = "storage-v0"
            
        response = make_response({
                "storage_service": node_service,
                "storage_node": node_name
            })
        response.headers['x-upstream-ip'] = socket.gethostbyname(socket.gethostname())
    
        return response

    return '''
    <!doctype html>
    <title>Storage</title>
    <h1>Storage, send POST with thing to store</h1>
    </form> '''

@app.route('/fetch/', methods=['GET', 'POST'])
def storage_get():

    if request.method == 'POST':
        data = request.json
        if not data:
            print("No json couldn be retrieved", flush=True)
            return redirect(request.url)

        r = redis.Redis(host='localhost', port=REDIS_PORT, db=0)

        storage_extraload = int(request.headers.get('storage-extraload', default="5"))

        np.random.seed(int(str(time.time()).split(".")[1]))
        if not IMG_FILES:
            response = make_response({"exists": "false"})
        else:
            # Perform extra read load
            x = []
            for k in range(storage_extraload):
                x.append(r.get(np.random.choice(IMG_FILES)))

            file_data = r.get(data["imgfile"])
            if file_data is None:
                file_data = r.get(np.random.choice(IMG_FILES))

        response = make_response(send_file(io.BytesIO(file_data), \
            mimetype='image/jpeg', as_attachment=True, attachment_filename=data["imgfile"]))

        response.headers['x-upstream-ip'] = socket.gethostbyname(socket.gethostname())

        node_name = os.environ.get('NODE_NAME')
        if not node_name:
            node_name = "localhost"
        node_service = os.environ.get('NODE_SERVICE')
        if not node_service:
            node_service = "storage-v0"

        response.headers['storage_node'] = node_name
        response.headers['storage_service'] = node_service

        return response

    return '''
    <!doctype html>
    <title>Storage</title>
    <h1>Storage, send POST filename to retrieve</h1>
    </form> '''

class HttpServer(gunicorn.app.base.BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items()
                  if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

if __name__ == "__main__":
    
    # Loading redis
    r = redis.Redis(host='localhost', port=REDIS_PORT, db=0)
    for imgfile in IMG_FILES:
        with open(STORAGE + imgfile, "rb") as f:
            r.set(imgfile, f.read())
    print("Redis preload complete", flush=True)

    options = {
        "bind": "0.0.0.0:%s" % PORT,
        "worker_tmp_dir": "/dev/shm",
        "log_file": "-",
        "log_level": "debug", 
        "workers": 9,
        "worker_class": "gevent",
        "worker_connections": "1000"
    }

    HttpServer(app, options).run()
