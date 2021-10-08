
from src.convnet.pipes import train
import yaml 
import sys
import os 

with open("conf/parameters.yml") as file:
    PARAMS = yaml.safe_load(file)


if __name__ == "__main__":
    

    if sys.argv[1] == "train":
            
        # train model
        train.run(params=PARAMS)
        
    elif sys.argv[1] == "serve":

        # serve model
        os.environ["MODEL_PATH"] = PARAMS["MODEL_PATH"]
        os.system(f"""nohup tensorflow_model_server \
            --rest_api_port=8502 \ 
            --model_name=img_model \
                --model_base_path={PARAMS['MODEL_PATH']} >server.log 2>&1"""
                )

    elif sys.argv[1] == "predict":
        pass
