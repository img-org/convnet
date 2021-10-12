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

* `Traefik` router:  
  1. Install [traefik](https://github.com/traefik/traefik/releases/tag/v2.4.14)

### Setup

* Build model server (300MB) and web server images (~3GB):

```bash
docker_build_model_server.sh 
docker_build_web_server.sh
```

* Run all containers:  

```bash
docker-compose up
```

# Tools

* `ngrok`: You can use ngrok to export a port as an external url. Basically, ngrok takes something available/hosted on your localhost and exposes it to the internet with a temporary public URL.

* `Docker Compose`: to configure & start all the containers  


nohup tensorflow_model_server --rest_api_port=8502 --model_name=img_model --model_base_path="${model}" >logs/server.log 2>&1 # path of model to serve


