FROM registry.gitlab.com/johanruuskanen/docker_images/python3-opencv:latest

LABEL reason=FacedetectBackend

WORKDIR /app
COPY . /app

RUN pip install -r /app/requirements.txt

CMD ["python3", "backend.py"]


