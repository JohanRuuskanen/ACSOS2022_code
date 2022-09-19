FROM registry.gitlab.com/johanruuskanen/docker_images/python3-opencv:latest

WORKDIR /app
COPY . /app

RUN pip install -r /app/requirements.txt

CMD ["python3", "storage.py"]

#CMD ["gunicorn", "--bind", "0.0.0.0:5002" , "storage:app", "--preload", \
#        "--worker-tmp-dir", "/dev/shm", \
#        "--log-file", "-", \
#        "--log-level", "info", \ 
#        "--workers" , "9", \
#        "--worker-class", "gevent", \
#        "--worker-connections", "1000"]