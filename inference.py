import os
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from datetime import date
from loguru import logger
import time
import torch
import torchvision.transforms as transforms
from models.ASPP_Classifier import ASPP_Classifier
from src.inference_preprocess import preprocess
from src.dataset import GreenRoofsDataset
from src.dataset_utils import ToTensor, Normalize
from omegaconf import DictConfig, OmegaConf


def inference(cfg:DictConfig):
    # Test cuda compatibility and show torch versions
    if not torch.cuda.is_available():
        logger.info("CUDA NOT AVAILABLE")
    else:
        logger.info("Cuda available")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load arguments
    INFERENCE = cfg['inference']
    WORKING_DIRECTORY = INFERENCE['working_directory']
    INPUTS = INFERENCE['inputs']
    POLYGON_SRC = INPUTS['polygon_src']
    RASTERS_DIR = INPUTS['rasters_dir']
    CLASS_LABELS_DIR = INPUTS['class_labels_dir']
    MODEL_SRC = INPUTS['model_src']

    OUTPUTS = INFERENCE['outputs']
    DATASET_DIR = OUTPUTS['dataset_dir']
    PREDS_DIR = OUTPUTS['preds_dir']
    FOLDER_NAME_SUFFIX = OUTPUTS['folder_name_suffix']

    PROCESSES = INFERENCE['processes']
    DO_PREPROCESSING = PROCESSES['do_preprocessing']

    PARAMETERS = INFERENCE['parameters']
    DATA_FRAC = PARAMETERS['data_frac']
    MODE = PARAMETERS['mode']

    PREPROCESS_OVERRIGHT = INFERENCE['preprocess_overright']
    PREPROC_PROCESSES = PREPROCESS_OVERRIGHT['processes']
    DO_RANGELIMIT = PREPROC_PROCESSES['do_rangelimit']
    DO_MASK = PREPROC_PROCESSES['do_mask']
    DO_SMOOTH_MASK = PREPROC_PROCESSES['do_smooth_mask']
    DO_DROP_OVERLAPPING = PREPROC_PROCESSES['do_drop_overlapping']
    DO_DROP_BASED_ON_NDVI = PREPROC_PROCESSES['do_drop_based_on_ndvi']
    METADATA = PREPROCESS_OVERRIGHT['metadata']
    SAMPLE_SIZE = METADATA['sample_size']
    RANGELIMIT_MODE = METADATA['rangelimit_mode']
    RANGELIMIT_THRESHOLD = METADATA['rangelimit_threshold']
    MULTI_PROCESSING = PREPROCESS_OVERRIGHT['multi_processing']
    MAX_WORKERS = MULTI_PROCESSING['max_workers']

    TRAIN_OVERRIGHT = INFERENCE['train_overright']
    NORM_BOUNDARIES = np.array(TRAIN_OVERRIGHT['norm_boundaries'])
    BATCH_SIZE = TRAIN_OVERRIGHT['batch_size']
    NUM_WORKERS = TRAIN_OVERRIGHT['num_workers']
    BACKBONE = TRAIN_OVERRIGHT['backbone']
    BACKBONE_NUM_LEVELS = TRAIN_OVERRIGHT['backbone_num_levels']
    BACKBONE_NUM_LAYERS = TRAIN_OVERRIGHT['backbone_num_layers']
    ASPP_ATROUS_RATES = TRAIN_OVERRIGHT['aspp_atrous_rates']

    # create result architecture
    #   _create folder name
    folder_name = date.today().strftime("%Y%m%d") + "_inference_" + FOLDER_NAME_SUFFIX
    new_folder_name = folder_name
    i = 1
    while os.path.exists(os.path.join(PREDS_DIR , new_folder_name + '/')):
        new_folder_name = folder_name + "_" + str(i)
        i += 1
    folder_name = new_folder_name + "/"
    preds_root_dir = os.path.join(PREDS_DIR, folder_name)
    os.mkdir(preds_root_dir)

    # load the polygon and create column for prediction
    df_roofs = gpd.read_file(POLYGON_SRC)

    pred_column_name = 'Preds_'+ MODE
    df_roofs[pred_column_name] = np.nan
    df_roofs.EGID = df_roofs.EGID.astype(int)

    # preprocessing
    # _loading and overrighting of the preprocess configuration
    cfg_preprocess = OmegaConf.load('./config/preprocessing.yaml')
    cfg_preprocess.preprocessing.working_directory = WORKING_DIRECTORY
    cfg_preprocess.preprocessing.inputs.polygon_src = POLYGON_SRC
    cfg_preprocess.preprocessing.inputs.rasters_dir = RASTERS_DIR
    cfg_preprocess.preprocessing.inputs.class_labels_dir = CLASS_LABELS_DIR
    cfg_preprocess.preprocessing.outputs.output_dir = DATASET_DIR
    cfg_preprocess.preprocessing.processes.do_rangelimit = DO_RANGELIMIT
    cfg_preprocess.preprocessing.processes.do_mask = DO_MASK
    cfg_preprocess.preprocessing.processes.do_smooth_mask = DO_SMOOTH_MASK
    cfg_preprocess.preprocessing.processes.do_drop_overlapping = DO_DROP_OVERLAPPING
    cfg_preprocess.preprocessing.processes.do_drop_based_on_ndvi = DO_DROP_BASED_ON_NDVI
    cfg_preprocess.preprocessing.processes.do_da_rotation = False
    cfg_preprocess.preprocessing.processes.do_da_flipping = False
    cfg_preprocess.preprocessing.metadata.sample_size = SAMPLE_SIZE
    cfg_preprocess.preprocessing.metadata.rangelimit_mode = RANGELIMIT_MODE
    cfg_preprocess.preprocessing.metadata.rangelimit_threshold = RANGELIMIT_THRESHOLD
    cfg_preprocess.preprocessing.multiprocessing.max_workers = MAX_WORKERS
    cfg_preprocess.preprocessing.security.do_abort = False

    if DO_PREPROCESSING:
        preprocess(cfg_preprocess.preprocessing)

    # create dataset, dataloader and model
    # _set transform
    transform = transforms.Compose([
        Normalize(NORM_BOUNDARIES),
        ToTensor(),
    ])
    dataset_preds = GreenRoofsDataset(DATASET_DIR, 
                                    mode= 'inf', 
                                    data_frac=DATA_FRAC,
                                    transform=transform,
                                    )
    
    # _test if dataset found
    if len(dataset_preds) == 0:
        logger.info("The dataset is empty. It might be due to a mistake in the rasters path or polygons path or no overlapp between the two.")
        quit()
    
    dataloader_preds = torch.utils.data.DataLoader(dataset_preds, 
                                                batch_size=BATCH_SIZE, 
                                                shuffle=False,
                                                num_workers=NUM_WORKERS,
                                                drop_last=False,
                                                )
    input_channels, img_size, _ = dataset_preds[0]['image'].size()
    output_channels = 6 if MODE == 'multi' else 2
    model = ASPP_Classifier(
        input_channels=input_channels,
        output_channels=output_channels,
        img_size=img_size,
        batch_size=BATCH_SIZE,
        backbone=BACKBONE,
        bb_levels=BACKBONE_NUM_LEVELS,
        bb_layers=BACKBONE_NUM_LAYERS,
        aspp_atrous_rates=ASPP_ATROUS_RATES,
        mode=MODE,
    ).double().to(torch.device(DEVICE))

    # _load trained model
    assert(os.path.exists(MODEL_SRC))
    checkpoint = torch.load(MODEL_SRC, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # predictions
    lst_egids = []
    lst_preds = []
    lst_preds_conf = []
    for _, data in tqdm(enumerate(dataloader_preds), total=len(dataloader_preds), desc='Predicting'):
            inputs, egids = data['image'], data['label']
            inputs = inputs.to(torch.device(DEVICE))
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                preds = outputs.data.max(1)[1]
                preds_conf = outputs.data.max(1)[0]
                lst_egids.append(egids.tolist())
                lst_preds.append(preds.tolist())
                lst_preds_conf.append(preds_conf.tolist())

    # flatten lists
    lst_egids = [samp for row in lst_egids for samp in row]
    lst_preds = [samp for row in lst_preds for samp in row]
    lst_preds_conf = [samp for row in lst_preds_conf for samp in row]
    
    # number to char
    if MODE == 'multi':
        dict_num_to_char = {
            0: 'b',
            1: 't',
            2: 's',
            3: 'e',
            4: 'l',
            5: 'i',
        }
    else:
        dict_num_to_char = {
            0: 'b',
            1: 'v',
        }

    lst_preds = [dict_num_to_char[x] for x in lst_preds]

    # add results to roofs
    df_preds = gpd.GeoDataFrame({
         'EGID': lst_egids,
         pred_column_name: lst_preds,
         pred_column_name + '_conf': lst_preds_conf,
    })
    df_preds.EGID = df_preds.EGID.astype(int)
    df_roofs = df_roofs.merge(df_preds, on='EGID', how='outer', suffixes=('_duplicate', ''))
    df_roofs = df_roofs.drop(columns=[pred_column_name + '_duplicate'], axis=1)
    df_roofs = df_roofs.dropna(subset=pred_column_name).reset_index(drop=True)

    # saving results
    new_name = ''.join(POLYGON_SRC.split('/')[-1].split('.')[:-1])
    ext = POLYGON_SRC.split('.')[-1]
    new_name_file = ''.join(new_name) + '_preds.' + ext
    new_name_csv = ''.join(new_name) + '_preds.csv'
    df_roofs.to_file(os.path.join(preds_root_dir, new_name_file))
    df_roofs.to_csv(os.path.join(preds_root_dir, new_name_csv), sep=';', index=None)
    logger.info(f"Resulting fils saved in {preds_root_dir}")

    # figures
    pass


if __name__ == '__main__':
    cfg = OmegaConf.load('./config/inference.yaml')
    inference(cfg)
