#!/bin/bash

NVIDIA_DRIVER=$(curl -s http://localhost:3476/v1.0/docker/cli | grep -oE 'nvidia_driver_[0-9\.]+')
docker volume create -d nvidia-docker --name $NVIDIA_DRIVER