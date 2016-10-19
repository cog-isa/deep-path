#!/bin/bash

YOUR_HASHED_PASSWORD=$(cat jupyter.passwd)

BIND_PORT="-p 8889:8888"
if (( $# >= 1 ))
then
    BIND_PORT=""
fi


CMD="docker"
if which nvidia-docker
then
    CMD="nvidia-docker"
fi

sudo $CMD run -ti --rm \
    -e "HASHED_PASSWORD=$YOUR_HASHED_PASSWORD" \
    -e "SSL=" \
    -v /home/rsuvorov/projects/docker-jupyter-keras-tools/certs:/jupyter/certs \
    -v `pwd`:/notebook \
    $BIND_PORT \
    deep-path \
    $@
