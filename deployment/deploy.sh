#!/bin/bash

cmd=$1

build() {
    docker tag $IMAGE_NAME:$IMAGE_VERSION $IMAGE_NAME:$IMAGE_VERSION-old
    DOCKER_BUILDKIT=1 docker build --tag $IMAGE_NAME:$IMAGE_VERSION -f ./deployment/Dockerfile . || exit 1
    docker rmi $(docker images --format "{{.Repository}}:{{.Tag}}" | grep "$IMAGE_VERSION-old")
}

shift

case $cmd in
build)
    build "$@"
    ;;

up)
    up "$@"
    ;;  
down)
    down "$@"
    ;; 
esac
