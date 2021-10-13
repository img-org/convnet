# Convnet

author: steeve LAQUITAINE

## Development & train

## Setup

```bash
# setup tensorflow server dependencies, 
# install conda 4.5.4
# create conda environment, activate and
# install codebase dependencies  
bash setup.sh # install miniconda 4.5.4
```

## train model

```bash
python main.py train
```

## Deployment 

### Prerequisites

* Deployment server:
    * > 4GB RAM
    * docker desktop/and or engine installed

### Setup

* Build model server (300MB) and web server images (~3GB):

```bash
# build services 
bash docker_model/build.sh 
bash docker_web/build.sh
# create an external public network 
docker network create traefik-public
# compose containers
docker-compose up  
```

* Open swagger ui in Chrome or Vivaldi browser http://web.service.localhost/docs. Currently does not work in firefox and safari.


# Tools

* `ngrok`: You can use ngrok to export a port as an external url. Basically, ngrok takes something available/hosted on your localhost and exposes it to the internet with a temporary public URL.

* `Docker Compose`: to configure & start all the containers  

nohup tensorflow_model_server --rest_api_port=8502 --model_name=img_model --model_base_path="${model}" >logs/server.log 2>&1 # path of model to serve

# Challenges
  
* Tensorflow is heavy (500 MB)