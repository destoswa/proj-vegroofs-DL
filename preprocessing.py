import os
import shutil
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from functools import partial
from itertools import islice
from src.preprocess_utils import da_rotation, da_flipping, clip_roofs_to_raster, processes_on_sample
from omegaconf import DictConfig, OmegaConf


def chunks(data, size):
    """Yield successive chunks of data."""
    it = iter(data.index)
    for _ in range(0, len(data), size):
        yield data.loc[islice(it, size)]


def preprocess(cfg: DictConfig):
    """
    This function handles the preprocessing of geospatial raster and polygon data based on the given configuration (cfg) in order to
    extract ready-to-be-used samples for the training of classification models. 
    It performs tasks such as loading input files, applying various transformations like cropping, masking, and normalization, 
    and generating augmented data if required. 
    The function organizes the output into directories. It saves PNG files for visualisation and Pickle files for the trainings
    as well as saving metadata to correspond a sample with its label using the EGID number of the building as unique identifier. 

    Args:
        - cfg (DictConfig): Configuration dictionary containing paths for inputs, outputs, processes to perform (like NDVI, luminosity), 
        and metadata such as sample size and EPSG code.

    Returns:
        - None
    """
    logger.info('Starting preprocessing...')

    # Security
    if cfg['security']['do_abort']:
        quit()

    # Change current directory:
    current_directory = os.getcwd()
    WORKING_DIRECTORY = cfg['working_directory']
    os.chdir(WORKING_DIRECTORY)

    # Load source and target files/dir
    INPUTS = cfg['inputs']
    POLYGON_SRC = INPUTS['polygon_src']
    RASTER_DIR = INPUTS['rasters_dir']
    CLASS_LABELS_DIR = INPUTS['class_labels_dir']

    OUTPUTS = cfg['outputs']
    OUTPUT_DIR = OUTPUTS['output_dir']

    # Load processes flags
    PROCESSES = cfg['processes']
    DO_DA_ROTATION = PROCESSES['do_da_rotation']
    DO_DA_FLIPPING = PROCESSES['do_da_flipping']
    DO_DROP_OVERLAPPING = PROCESSES['do_drop_overlapping']
    DO_DROP_BASED_ON_NDVI = PROCESSES['do_drop_based_on_ndvi']
    DO_PRODUCE_TIF = PROCESSES['do_produce_tif']

    # Load metadata
    METADATA = cfg['metadata']
    PREPROCESS_MODE = METADATA['preprocess_mode']
    BATCH_SIZE = METADATA['batch_size']
    SAMPLE_SIZE = METADATA['sample_size']
    EPSG = METADATA['epsg']

    # Load multiprocessing metadata
    MULTIPROCESSING = cfg['multiprocessing']
    MAX_WORKERS = MULTIPROCESSING['max_workers']
    MAX_WORKERS = None if MAX_WORKERS == 0 else MAX_WORKERS

    # Messages for processes:
    if DO_DA_ROTATION:
        logger.info("_with data augmentation (rotation)")
    if DO_DA_FLIPPING:
        logger.info("_with data augmentation (flipping)")
    if DO_DROP_OVERLAPPING:
        logger.info(
            "_with dropping out of the samples overlapping multiple rasters")

    # Asserts and warnings if incompatibilities in or between parameters
    assert isinstance(EPSG, int)

    if SAMPLE_SIZE < 128:
        warnings.warn(
            "The sample size is set lower than 128. It might results in errors during the training.")

    # Categories of samples
    df_categories = pd.read_csv(CLASS_LABELS_DIR, sep=';')

    # Create architecture if necessary
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    cat_dir = {}
    cat_dir['.'] = OUTPUT_DIR

    if PREPROCESS_MODE == 'training':
        df_categories = pd.read_csv(CLASS_LABELS_DIR, sep=';')
        for let, cat in df_categories[['code_char', 'cat']].values:
            cat_dir[let] = OUTPUT_DIR + "/" + cat
    elif PREPROCESS_MODE == 'inference':
        cat_dir['data'] = OUTPUT_DIR + '/data'
    else:
        raise ValueError('Wrong value for "preprocessing-mode" parameter.')

    for dir in cat_dir.values():
        if not os.path.exists(dir):
            os.mkdir(dir)

    # Get rasters
    raster_src_list = []
    for r, d, f in os.walk(RASTER_DIR):
        for file in f:
            if file.endswith('.tif'):
                file_src = r + '/' + file
                file_src = file_src.replace('\\', '/')
                raster_src_list.append(file_src)

    if len(raster_src_list) == 0:
        raise ValueError('No .tif file found at raster_dir location!')

    # Get polygons of roofs
    roofs = gpd.read_file(POLYGON_SRC).to_crs(EPSG)
    roofs.EGID = roofs.EGID.astype(int)

    # Create dataframe to store metas of samples
    if PREPROCESS_MODE == 'training':
        # Prepare dataset
        df_dataset = pd.DataFrame(roofs[['EGID', 'class']])
        df_dataset.insert(2, 'file_src', '-')
        df_dataset.insert(3, 'label', 0)

        # Add row with class code in dataframe:
        for let, num in df_categories[['code_char', 'code_num']].values:
            df_dataset.loc[df_dataset['class'] == let, 'label'] = num

        # Saving samples metadata:
        roofs[['EGID', 'class', 'area', 'year']].to_csv(os.path.join(
            OUTPUT_DIR, "samples_metadata.csv"), sep=';', index=False, encoding='utf-8-sig')

        # Save categories to csv
        df_categories.to_csv(os.path.join(
            OUTPUT_DIR, "class_names.csv"), sep=';', index=False)
    else:
        df_dataset = pd.DataFrame(roofs[['EGID']])
        df_dataset.insert(1, 'file_src', '-')
        roofs['class'] = '-'

    overhanging_trees = []
    toosmall_samples = []
    overlapping_roofs = []
    nonmatching_roofs = []
    sample_format = ""

    # Test batch size
    if BATCH_SIZE == 0 or BATCH_SIZE > len(roofs):
        BATCH_SIZE = len(roofs)

    # Loop on batches
    if BATCH_SIZE == 0:
        BATCH_SIZE = len(roofs)
    for batch_idx, sub_roofs in enumerate(chunks(roofs, BATCH_SIZE)):
        print(
            f"Processing batch {batch_idx} / {int(np.ceil(len(roofs) / BATCH_SIZE - 1))}")
        # Create samples by clipping the polygons to the rasters
        samples, sub_overlapping_roofs, sub_nonmatching_roofs = clip_roofs_to_raster(
            raster_src_list, sub_roofs, MAX_WORKERS)
        nonmatching_roofs.append(sub_nonmatching_roofs)
        overlapping_roofs.append(sub_overlapping_roofs)

        # Use Manager to create a shared dictionary
        with Manager() as manager:
            shared_dict = manager.dict()

            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                part_func = partial(processes_on_sample,
                                    cfg=cfg,
                                    cat_dir=cat_dir,
                                    shared_dict=shared_dict,
                                    df_categories=df_categories,
                                    overlapping_roofs=overlapping_roofs,
                                    )

                futures = {executor.submit(part_func, egid, sample)
                           for egid, sample in samples.items()}

                for future in tqdm(as_completed(futures), total=len(futures), desc="  Processing"):
                    result = future.result()
                    if result:
                        dict_aborted = result
                        if dict_aborted['overhanging_tree'] != '':
                            overhanging_trees.append(
                                dict_aborted['overhanging_tree'])
                        if dict_aborted['toosmall_sample'] != '':
                            toosmall_samples.append(
                                dict_aborted['toosmall_sample'])

            for egid, src in shared_dict.items():
                df_dataset.loc[df_dataset.EGID == egid, 'file_src'] = src
        
    # add column for keeping track of original egid:
    df_dataset['original_egid'] = df_dataset.loc[:,'EGID'].astype(int).astype('str')

    # saving labelized info to csv:
    df_dataset = df_dataset.loc[df_dataset.file_src != '-',:]

    # saving dataset infos
    df_dataset.to_csv(os.path.join(OUTPUT_DIR, "dataset.csv"), index=False, sep=';')
    
    # Data augmentation
    #   _rotation:
    if DO_DA_ROTATION:
        da_rotation(OUTPUT_DIR, df_categories, MAX_WORKERS, DO_PRODUCE_TIF)
        
    #   _flipping:
    if DO_DA_FLIPPING:
        da_flipping(OUTPUT_DIR, df_categories, MAX_WORKERS, DO_PRODUCE_TIF)

    # flattening lists
    nonmatching_roofs = [val for row in nonmatching_roofs for val in row]
    overlapping_roofs = [val for row in overlapping_roofs for val in row]

    # Show summary
    logger.info("In total:")
    logger.info(f"\t{len(roofs)} roofs processed.")
    logger.info(f"\t{len(nonmatching_roofs)} non-matching roofs dropped.")
    if len(nonmatching_roofs) > 0:
        with open(os.path.join(OUTPUT_DIR, 'nonmatching_roofs.txt'), 'w') as file:
            file.write(str(nonmatching_roofs))

    if DO_DROP_OVERLAPPING and sample_format != 'uint16':
        logger.info(f"\t{len(overlapping_roofs)} overlapping roofs dropped.")
        if len(overlapping_roofs) > 0:
            with open(os.path.join(OUTPUT_DIR, 'overlapping_roofs.txt'), 'w') as file:
                file.write(str(overlapping_roofs))

    if DO_DROP_BASED_ON_NDVI:
        logger.info(
            f"\t{len(overhanging_trees)} bare roofs with overhanging trees dropped.")
        if len(overhanging_trees) > 0:
            with open(os.path.join(OUTPUT_DIR, 'overhanging_trees.txt'), 'w') as file:
                file.write(str(overhanging_trees))

    logger.info("Preprocessing done.")

    # Save copy of config file
    with open(os.path.join(OUTPUT_DIR, "config.yaml"),'w+') as file:
        OmegaConf.save(cfg, file.name)

    # Resetting the current directory
    os.chdir(current_directory)


if __name__ == '__main__':
    # Retrieve parameters
    cfg = OmegaConf.load('./config/preprocessing.yaml')
    preprocess(cfg['preprocessing'])
        