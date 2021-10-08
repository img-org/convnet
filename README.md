# Convnet

author: steeve LAQUITAINE

## Prerequisites

## Setup

```bash
bash setup.sh # setup tensoflow server dependencies
bash setup_conda.sh # install miniconda 4.5.4
conda env create -f src/environment.yml # creates my_env
```

## Run

```bash
python main.py train
```


# Tools

* `ngrock`: You can use ngrok to export a port as an external url. Basically, ngrok takes something available/hosted on your localhost and exposes it to the internet with a temporary public URL.

