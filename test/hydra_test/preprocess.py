import hydra
from omegaconf import DictConfig, OmegaConf

def preprocess(cfg: DictConfig):
    print("preprocessing...")
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    preprocess()