#!/bin/bash
#
# Build docker container for model serving with tensorflow serve
# You can test that the building the docker image worked by running: 
#
# ```bash
# docker images
#
# # REPOSITORY       TAG       IMAGE ID       CREATED              SIZE
# # x86_64/convnet   0.8.1     c225c5af60d4   About a minute ago   298MB
# ```
#
# docker run 

VERSION="0.8.1"
ARCH="x86_64"
APP="convnet-model-service"
docker build -f docker_model/dockerfile -t $ARCH/$APP:$VERSION .

