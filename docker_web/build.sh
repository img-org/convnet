#!/bin/bash
#
# Build docker container for web serving with fastAPI (~2.94GB)
# You can test that the building the docker image worked by running: 
#
# ``bash
# docker images
# 
# REPOSITORY                     TAG       IMAGE ID       CREATED              SIZE
# x86_64/convnet-web-service     0.8.1     2542a416459c   About a minute ago   2.94GB
# ```

VERSION="0.8.1"
ARCH="x86_64"
APP="convnet-web-service"
docker build -f docker_web/dockerfile -t $ARCH/$APP:$VERSION .

