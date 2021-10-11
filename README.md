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

* Build model server (300MB) and web server containers (~3GB):

```bash
docker_build_model_server.sh 
docker_build_web_server.sh
```

# Tools

* `ngrock`: You can use ngrok to export a port as an external url. Basically, ngrok takes something available/hosted on your localhost and exposes it to the internet with a temporary public URL.

