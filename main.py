
from src.convnet.pipes import train
import yaml 

with open("conf/parameters.yml") as file:
    PARAMS = yaml.safe_load(file)


if __name__ == "__main__":
    
    train.run(params=PARAMS)