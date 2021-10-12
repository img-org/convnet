#!bin/bash
# purpose:
#
#  setup for development and train environment

# install and setup tensorflow server
# echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
# curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
# sudo apt-get update && sudo apt-get install tensorflow-model-server


# Setup and activate a virtual environmment
# -----------------------------------------
install miniconda to work in a virtual environment
MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.4-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX

# configure conda to activate envs.
echo ". /usr/local/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc

# create and activate conda virtual environment  
conda create --name convnet # creates my_env
conda activate convnet
pip install -r src/requirements.txt
