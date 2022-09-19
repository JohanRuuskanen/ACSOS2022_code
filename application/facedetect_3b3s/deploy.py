import os
import yaml
import time

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

ROOT = "/home/ubuntu/run_on_gateway/clusters/"

def deploy(clusters):
    print("DEPLOY SERVICES")
    for cluster in clusters:
        print("====== {} ======".format(cluster))
        if not ((cluster in FRONTEND_CLUSTERS) or (cluster in BACKEND_V1_CLUSTERS) \
                or (cluster in BACKEND_V2_CLUSTERS) or (cluster in BACKEND_V3_CLUSTERS) \
                or (cluster in STORAGE_V1_CLUSTERS) (cluster in STORAGE_V2_CLUSTERS) \
                or (cluster in STORAGE_V3_CLUSTERS)):
            continue
        os.system("kubectl --context={} apply -f src/namespace.yaml".format(cluster))
        os.system("kubectl --context={} -n facedetect-3b3s create secret generic gitlab-auth".format(cluster) \
            + " --from-file=.dockerconfigjson=/home/ubuntu/.docker/config.json" \
            + " --type=kubernetes.io/dockerconfigjson")
        os.system("kubectl --context={} apply -f src/backend-v1.yaml -l service=backend-v1".format(cluster))
        os.system("kubectl --context={} apply -f src/backend-v2.yaml -l service=backend-v2".format(cluster))
        os.system("kubectl --context={} apply -f src/backend-v3.yaml -l service=backend-v3".format(cluster))
        os.system("kubectl --context={} apply -f src/storage-v1.yaml -l service=storage-v1".format(cluster))
        os.system("kubectl --context={} apply -f src/storage-v2.yaml -l service=storage-v2".format(cluster))
        os.system("kubectl --context={} apply -f src/storage-v3.yaml -l service=storage-v3".format(cluster))

def deploy_frontend():
    print("\nDEPLOY FRONTEND")
    for frontend_cluster in FRONTEND_CLUSTERS:
        print("====== {} ======".format(frontend_cluster))
        os.system("kubectl --context={} apply -f src/frontend.yaml".format(frontend_cluster))

def deploy_backend():
    print("\nDEPLOY BACKEND")
    for backend_cluster in BACKEND_V1_CLUSTERS:
        print("====== {} ======".format(backend_cluster))
        os.system("kubectl --context={} apply -f src/backend-v1.yaml".format(backend_cluster))
    for backend_cluster in BACKEND_V2_CLUSTERS:
        print("====== {} ======".format(backend_cluster))
        os.system("kubectl --context={} apply -f src/backend-v2.yaml".format(backend_cluster))
    for backend_cluster in BACKEND_V3_CLUSTERS:
        print("====== {} ======".format(backend_cluster))
        os.system("kubectl --context={} apply -f src/backend-v3.yaml".format(backend_cluster))

def deploy_storage():
    print("\nDEPLOY STORAGE")
    for storage_cluster in STORAGE_V1_CLUSTERS:
        print("====== {} ======".format(storage_cluster))
        os.system("kubectl --context={} apply -f src/storage-v1.yaml".format(storage_cluster))
    for storage_cluster in STORAGE_V2_CLUSTERS:
        print("====== {} ======".format(storage_cluster))
        os.system("kubectl --context={} apply -f src/storage-v2.yaml".format(storage_cluster))
    for storage_cluster in STORAGE_V3_CLUSTERS:
        print("====== {} ======".format(storage_cluster))
        os.system("kubectl --context={} apply -f src/storage-v3.yaml".format(storage_cluster))

def retrieve_cluster_IPs(clusters):
    ips = {}
    for (i, cluster) in enumerate(clusters):
        CMD = "kubectl --context=%s get nodes -o wide | grep master | awk '{print $6}'" % cluster
        masterip = os.popen(CMD).read().strip()
        ips[cluster] = masterip
    return ips

def generate_nginxconf(clusters):

    ips = retrieve_cluster_IPs(clusters)

    us_imgrec = ""
    for frontend_cluster in FRONTEND_CLUSTERS:
        us_imgrec += "server %s:31115;\n" % ips[frontend_cluster]

    conf = """
    user    nginx;
    worker_processes        1;
    error_log  /var/log/nginx/error.log warn;
    pid        /var/run/nginx.pid;

    events {
            worker_connections      1024;
    }
    http {
            proxy_http_version 1.1;

            client_max_body_size 12M;

            upstream us_imgrec {
                %s
            }

            server {
                    listen 3001;

                    location / {
                            proxy_pass http://us_imgrec/;
                    }

                    location /detect/ {
                            proxy_pass http://us_imgrec/detect/;
                    }

                    location /fetch/ {
                            proxy_pass http://us_imgrec/fetch/;
                    }
            }
    }
    """ % (us_imgrec)

    with open("nginx.conf", "w") as f:
        f.write(conf)


def launch_nginx():
    os.system("docker run -d  --network=host \
        -v %s/nginx.conf:/etc/nginx/nginx.conf:ro \
        --name=nginx_loadbalancer-large nginx:1.17.10" % (os.getcwd()))

if __name__ == '__main__':

    clusters = [dir for dir in os.listdir(ROOT) if "cluster-" in dir]
    clusters.sort()

    deploy(clusters)
    deploy_backend()
    deploy_storage()
    deploy_frontend()
    generate_nginxconf(clusters)
    launch_nginx()
