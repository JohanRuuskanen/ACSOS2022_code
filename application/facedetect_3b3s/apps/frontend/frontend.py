import numpy as np
import os
import cv2
import io
import socket
import json
import requests
import gunicorn.app.base

from flask import Flask, request, redirect, send_file, make_response

BACKEND_URL = ["http://backend-v1.facedetect-large:5001",
                "http://backend-v2.facedetect-large:5001", 
                "http://backend-v3.facedetect-large:5001"]
STORAGE_URL = ["http://storage-v1.facedetect-large:5002", 
                "http://storage-v2.facedetect-large:5002",
                "http://storage-v3.facedetect-large:5002"]

#BACKEND_URL = ["http://localhost:5001"]
#STORAGE_URL = ["http://localhost:5002"]

PORT = 5000

ALLOWED_EXTENSIONS = set(['jpg'])

app = Flask(__name__)

def getForwardHeaders(request):
    headers = {}
    incomming_headers = [   
        'x-request-id',
        'x-b3-traceid',
        'x-b3-spanid',
        'x-b3-parentspanid',
        'x-b3-sampled',
        'x-b3-flags',
        'x-ot-span-context'
        ]

    for ihdr in incomming_headers:
        val = request.headers.get(ihdr)
        if val is not None:
            headers[ihdr] = val

    return headers

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def startpage():
    return '''
        <!doctype html>
        <title>Frontend</title>
        <h1>The frontend microservice</h1>
        <p>use /detect/ to detect faces in an image or /fetch/ to fetch an image</p>
        </form> '''

@app.route('/detect/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        if not 'imgfile' in request.files:
            print("imgfile not submitted", flush=True)
            return redirect(request.url)
        file = request.files["imgfile"]

        if file.filename == '':
            print("The uploaded file contained empty filename", flush=True)
            return redirect(request.url)

        if not allowed_file(file.filename):
            print("An uploaded file was of wrong type", flush=True)
            return redirect(request.url)          

        stream = file.read()

        node_name = os.environ.get('NODE_NAME')
        if not node_name:
            node_name = "localhost"
        node_service = os.environ.get('NODE_SERVICE')
        if not node_service:
            node_service = "frontent-v0"

        lb_weights_json = request.headers.get('lb-weights', default='{}')
        lb_weights = json.loads(lb_weights_json)
        
        if node_service in lb_weights and 'detect' in lb_weights[node_service]:
            p = [float(w) for w in lb_weights[node_service]['detect'].split(',')]
        else:
            p = [1.0/len(BACKEND_URL)] * len(BACKEND_URL)

        timeout = float(request.headers.get('upstream-timeout', default="2.0"))
        storage_extraload = request.headers.get('storage-extraload', default="5")

        headers = getForwardHeaders(request)
        headers['x-downstream-ip'] = socket.gethostbyname(socket.gethostname())
        headers['upstream-timeout'] = str(timeout-1)
        headers['storage-extraload'] = storage_extraload
        headers['lb-weights'] = lb_weights_json

        r = requests.post(np.random.choice(BACKEND_URL, p=p) + "/detect/", \
            files={file.filename: stream}, headers=headers, timeout=timeout)
        
        data = r.json()

        img = np.asarray(bytearray(stream), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        for (x,y,w,h) in data['faces']:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,"Frontend: " + node_name,(20,20), font, .5,(255,55,55),1,cv2.LINE_AA)
        cv2.putText(img,"Backend: " + data['backend_node'],(20,40), font, .5,(255,55,55),1,cv2.LINE_AA)
        cv2.putText(img,"Storage: " + data['storage_node'],(20,60), font, .5,(255,55,55),1,cv2.LINE_AA)

        img_encode = cv2.imencode('.jpg', img)[1]

        response = make_response(send_file(io.BytesIO(img_encode), \
            mimetype='image/jpeg', as_attachment=True, attachment_filename=file.filename))
        response.headers['x-upstream-ip'] = socket.gethostbyname(socket.gethostname())

        response.headers['frontend_node'] = node_name
        response.headers['frontend_service'] = node_service
        response.headers['backend_node'] = data['backend_node']
        response.headers['backend_service'] = data['backend_service']
        response.headers['storage_node'] = data['storage_node']
        response.headers['storage_service'] = data['storage_service']

        return response
    return '''
    <!doctype html>
    <title>Upload new file</title>
    <h1>Upload new file new</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=imgfile>
      <input type=submit value=Upload>
    </form> '''

# Get a random saved image
@app.route('/fetch/', methods=['GET', 'POST'])
def get_val():
    if request.method == 'POST':

        filename = request.form['imgfile']
        if filename == '':
            print("The requested filename contained empty filename", flush=True)
            return redirect(request.url)

        if not allowed_file(filename):
            print("The requested file was of wrong type", flush=True)
            return redirect(request.url)         

        timeout = float(request.headers.get('upstream-timeout', default="2.0"))
        storage_extraload = request.headers.get('storage-extraload', default="5") 

        headers = getForwardHeaders(request)
        headers['x-downstream-ip'] = socket.gethostbyname(socket.gethostname())
        headers['upstream-timeout'] = str(timeout-1)
        headers['storage-extraload'] = storage_extraload

        node_name = os.environ.get('NODE_NAME')
        if not node_name:
            node_name = "localhost"
        node_service = os.environ.get('NODE_SERVICE')
        if not node_service:
            node_service = "frontent-v0"

        lb_weights = json.loads(request.headers.get('lb-weights', default='{}'))
        if node_service in lb_weights and 'fetch' in lb_weights[node_service]:
            p = [float(w) for w in lb_weights[node_service]['fetch'].split(',')]
        else:
            p = [1.0/len(STORAGE_URL)] * len(STORAGE_URL)

        r = requests.post(np.random.choice(STORAGE_URL, p=p) + "/fetch/", json=request.form.to_dict(), \
            headers=headers, timeout=timeout)

        if r.headers.get('Content-Type') == 'application/json':
            response = make_response(r.json())
            response.headers['x-upstream-ip'] = socket.gethostbyname(socket.gethostname())
        else:
            img = np.asarray(bytearray(r.content), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "Frontend: " + node_name,(20,20), font, .5,(255,55,55),1,cv2.LINE_AA)
            cv2.putText(img, "Storage: " + r.headers.get('storage_node'),(20,40), font, .5,(255,55,55),1,cv2.LINE_AA)

            img_encode = cv2.imencode('.jpg', img)[1]

            response = make_response(send_file(io.BytesIO(img_encode), \
                mimetype='image/jpeg', as_attachment=True, attachment_filename=filename))
            response.headers['x-upstream-ip'] = socket.gethostbyname(socket.gethostname())

            response.headers['frontend_node'] = node_name
            response.headers['frontend_service'] = node_service
            response.headers['backend_node'] = r.headers.get('backend_node')           
            response.headers['backend_service'] = r.headers.get('backend_service')
            response.headers['storage_node'] = r.headers.get('storage_node')
            response.headers['storage_service'] = r.headers.get('storage_service')

        return response
    return '''
    <!doctype html>
    <title>Get file</title>
    <h1>Get file</h1>
    <form method=post>
      <input type=form name=imgfile>
      <input type=submit value=Retrieve>
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
    options = {
        "bind": "0.0.0.0:%s" % PORT,
        "worker_tmp_dir": "/dev/shm",
        "log_file": "-",
        "log_level": "info", 
        "workers": 9,
        "worker_class": "gevent",
        "worker_connections": "1000"
    }

    HttpServer(app, options).run()
