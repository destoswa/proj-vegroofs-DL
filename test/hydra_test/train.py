import hydra
from omegaconf import DictConfig, OmegaConf

def train(cfg: DictConfig):
    print("training...")
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    train()