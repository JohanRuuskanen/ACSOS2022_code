#!/usr/bin/env bash

export GITLAB_REGISTRY=registry.gitlab.com # Insert git container registry, we used gitlab
export REGPATH= # Path to the registry

docker login $GITLAB_REGISTRY

for d in */ ; do
  docker build -t $GITLAB_REGISTRY/$REGPATH/"${d///}":3backends $d.
  docker push $GITLAB_REGISTRY/$REGPATH/"${d///}":3backends
done
