import os

ROOT = "/home/ubuntu/run_on_gateway/clusters/"

if __name__ == '__main__':

    clusters = [dir for dir in os.listdir(ROOT) if "cluster-" in dir]
    clusters.sort()

    files = os.listdir("src")

    print("Removing the cluster app")
    for cluster in clusters:
        print("====== {} ======".format(cluster))
        os.system("kubectl --context={} -n facedetect-do-b3 delete secret gitlab-auth".format(cluster))
        for file in files:
            if file == "namespace.yaml":
                continue
            os.system("kubectl --context={0} delete -f src/{1}".format(cluster, file))

    print("Removing the frontend load balancer")
    os.system("docker rm -f nginx_loadbalancer-do")
