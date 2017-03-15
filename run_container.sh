#!/bin/bash

YOUR_HASHED_PASSWORD=$(cat jupyter/jupyter.passwd)

BIND_PORT="-p 8890:8888"
if (( $# >= 1 ))
then
    BIND_PORT=""
fi


CMD="docker"
if which nvidia-docker
then
    CMD="nvidia-docker"
fi

$CMD run -ti --rm \
    -e "HASHED_PASSWORD=$YOUR_HASHED_PASSWORD" \
    -e "SSL=" \
    -v /home/rsuvorov/projects/docker-jupyter-keras-tools/certs:/jupyter/certs \
    -v `pwd`:/notebook \
    $BIND_PORT \
    deep-path \
    $@
