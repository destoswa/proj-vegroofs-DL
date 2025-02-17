import hydra
from omegaconf import DictConfig, OmegaConf
from preprocessing_multiproc import preprocess
from training import train
import logging
from hydra.core.hydra_config import HydraConfig

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def multiprocess(cfg:DictConfig):
    """
    This function is used by the hydra system to run multi-trainings.
    In order to configurate and run a multi-training, please look in file <./multi_training.py>
    
    !!! THIS FILE SHOULD NOT BE MODIFY BY USERS TO CONFIGURATE MULTI-TRAINING !!!
    """
    # create single training's folder's name
    suffix = ""
    for x in cfg['sweeping_var']:
        suffix += x.split('.')[-1] + "=" + str(OmegaConf.select(cfg, str(x))) + "_"
    suffix = suffix[:-1]    # remove last '_'
    OmegaConf.update(cfg, 'training.outputs.folder_name_suffix', suffix)

    # call scripts
    if cfg['do_preprocess']:
        preprocess(cfg['preprocessing'])
    train(cfg)


if __name__ == "__main__":
    multiprocess()
