import os
import numpy as np
import cv2
import pkg_resources
import socket
import gunicorn.app.base

from flask import Flask, request, redirect, make_response

PORT = 5001

haar_xml = pkg_resources.resource_filename(
    'cv2', 'data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haar_xml)

app = Flask(__name__)
#CORS(app)

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

@app.route('/')
def startpage():
    return '''
        <!doctype html>
        <title>Backend</title>
        <h1>The backend microservice</h1>
        <p>use /detect/ to detect faces in an image</p>
        </form> '''

@app.route('/detect/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if not len(request.files) == 1:
            print("Wrong number of files, should be 1", flush=True)
            return redirect(request.url)

        _, file = next(iter(request.files.items()))

        headers = getForwardHeaders(request)
        headers['x-downstream-ip'] = socket.gethostbyname(socket.gethostname())

        img = np.asarray(bytearray(file.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Can be made slower by repeating this line
        faces_v = []
        for k in range(3):
            faces_v.append(face_cascade.detectMultiScale(gray, 1.3, 5))
            
        faces = faces_v[-1]
              
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        node_name = os.environ.get('NODE_NAME')
        if not node_name:
            node_name = "localhost"

        response = make_response({
                "faces":[x.tolist() for x in faces],
                "backend_name": node_name
            })
        response.headers['x-upstream-ip'] = socket.gethostbyname(socket.gethostname())
        return response

    return '''
    <!doctype html>
    <title>Backend</title>
    <h1>Backend, send POST with file</h1>
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