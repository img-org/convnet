#! /bin/bash

# enable job control
# run tensorflow serving in the background
# direct both stdout and stderr to tf_server.log
bg nohup tensorflow_model_server --rest_api_port=8502 --model_name=model --model_base_path=/root/convnet/model > logs/tf_server.out 2>&1