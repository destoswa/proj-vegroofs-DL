import hydra
from omegaconf import DictConfig, OmegaConf
from preprocessing import preprocess
from training import train

@hydra.main(version_base=None, config_path="config", config_name="config")
def test_hydra(cfg: DictConfig):
    preprocess(cfg['preprocessing'])
    train(cfg['training'])


if __name__ == '__main__':
    test_hydra()
    