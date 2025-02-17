import os
import shutil
import pandas as pd
import geopandas as gpd
import dask_geopandas as dg
import tempfile
from time import time
from datetime import date
from omegaconf import DictConfig, OmegaConf
from inference import inference
from training import train
import torch


def big_data_process(cfg: DictConfig):
    # Change current directory
    WORKING_DIRECTORY = cfg.bigdata.working_directory
    current_dir = os.getcwd()
    os.chdir(WORKING_DIRECTORY)

    # Load configs
    MODE = cfg.bigdata.mode
    if MODE == 'inference':
        ROOFS_SRC = cfg.inference.inputs.polygon_src
    elif MODE == 'training':
        ROOFS_SRC = cfg.preprocessing.inputs.polygon_src
    else:
        raise ValueError(
            f"The value given in cfg.mode is not valid:\t{cfg.bigdata.mode}")

    BATCH_SIZE = cfg.bigdata.batch_size
    TEMPFOLDER = tempfile.mkdtemp()
    OUTPUTS = cfg.bigdata.outputs
    RESULT_DIR = OUTPUTS.results_dir
    RESULT_SUFFIXE = OUTPUTS.results_suffixe

    # Prepare architecture for results
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    res_name_original = f"{date.today().strftime("%Y%m%d")}_BIGDATA_{MODE}_{RESULT_SUFFIXE}"
    res_name_final = res_name_original
    num = 0
    while os.path.exists(os.path.join(RESULT_DIR, res_name_final)):
        res_name_final = res_name_original + f"_{num}"
        num += 1
    res_folder_src = os.path.join(RESULT_DIR, res_name_final)
    os.mkdir(res_folder_src)

    # Load roofs
    roofs = dg.read_file(ROOFS_SRC, chunksize=BATCH_SIZE)

    # Start batching
    start_bigdata_time = time()
    for num_batch, partition in enumerate(roofs.to_delayed()):
        start_time = time()
        print(f"Processing batch {num_batch} / {roofs.npartitions - 1}")

        # Create batch of roofs
        roofs_batch = partition.compute()
        roofs_batch_src = os.path.join(TEMPFOLDER, 'roofs_batch.gpkg')
        roofs_batch.to_file(roofs_batch_src, driver='GPKG')

        # Create batch folder
        batch_dir_src = os.path.join(
            RESULT_DIR, res_name_final, f"batch_{num_batch}")
        os.mkdir(batch_dir_src)

        # Modify cfg files
        cfg.preprocessing.inputs.polygon_src = roofs_batch_src
        cfg.inference.inputs.polygon_src = roofs_batch_src
        cfg.inference.outputs.preds_dir = batch_dir_src
        cfg.inference.processes.do_preprocessing = True
        cfg.training.outputs.res_dir = batch_dir_src
        cfg.training.processes.do_preprocessing = True
        cfg.training.parameters.from_pretrained = False if num_batch == 0 else True

        # Run processes
        if MODE == 'training':
            print("="*10, "\n Training...")
            train(cfg)

            # load last model, prepare it for next and update config so that next training load it
            for r, _, f in os.walk(batch_dir_src):
                for file in f:
                    if file == 'model_last.tar':
                        checkpoint = torch.load(
                            os.path.join(r, file), weights_only=False)
                        checkpoint['epoch'] = -1
                        torch.save(checkpoint, os.path.join(r, file))
                        cfg.training.inputs.pretrained_src = os.path.join(
                            r, file)

        elif MODE == 'inference':
            print("="*10, "\n Inference...")
            inference(cfg)

        # Delete tempfile
        os.remove(roofs_batch_src)

        # Print time to process batch
        time_elapsed_batch = time() - start_time
        n_hours = int(time_elapsed_batch / 3600)
        n_min = int((time_elapsed_batch % 3600) / 60)
        n_sec = int(time_elapsed_batch - n_hours * 3600 - n_min * 60)
        print(f'Batch completed in {n_hours}:{n_min}:{n_sec}\n')

    if MODE == 'inference':
        print("="*10 + "\nMERGING RESULTS...")

        # Merge results
        df_results = gpd.GeoDataFrame()
        for r, _, f in os.walk(res_folder_src):
            for file in f:
                if file == 'roofs_batch_preds.gpkg':
                    df_sub_res = gpd.read_file(os.path.join(r, file))
                    df_results = df_sub_res if len(df_results) == 0 else gpd.GeoDataFrame(
                        pd.concat([df_results, df_sub_res], ignore_index=True))

        # Saving results
        df_results.to_file(os.path.join(
            res_folder_src, "results.gpkg"), driver='GPKG')
        


    # Remove Temp folder
    shutil.rmtree(TEMPFOLDER)

    # Print time to process
    time_elapsed = time() - start_bigdata_time
    n_hours = int(time_elapsed / 3600)
    n_min = int((time_elapsed % 3600) / 60)
    n_sec = int(time_elapsed - n_hours * 3600 - n_min * 60)
    print(f'Big Data Process completed in {n_hours}:{n_min}:{n_sec}\n')

    # Go back to current directory
    os.chdir(current_dir)


if __name__ == '__main__':
    cfg_bd = OmegaConf.load("./config/bigdata.yaml")
    cfg_preprocess = OmegaConf.load("./config/preprocessing.yaml")
    cfg_train = OmegaConf.load("./config/training.yaml")
    cfg_inf = OmegaConf.load("./config/inference.yaml")
    cfg = OmegaConf.merge(cfg_bd, cfg_preprocess, cfg_train, cfg_inf)

    big_data_process(cfg)
