import os
import shutil
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm import tqdm
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from shapely import geometry
from loguru import logger
from tifffile import imread, imwrite
import concurrent.futures
from skimage.transform import resize
from skimage import util
from src.preprocess_utils import range_limiting, ndvi_samp_gen, lum_samp_gen, ndvi_to_mask, mask_nbh_rounding, da_rotation, da_flipping, clip_roofs_to_raster
from omegaconf import DictConfig, OmegaConf


def preprocess(cfg:DictConfig):
    """
    This function handles the preprocessing of geospatial raster and polygon data based on the given configuration (cfg) in order to
    extract ready-to-be-used samples for the training of classification models. 
    It performs tasks such as loading input files, applying various transformations like cropping, masking, and normalization, 
    and generating augmented data if required. 
    The function organizes the output into directories. It saves PNG files for visualisation and Pickle files for the trainings
    as well as saving metadata to correspond a sample with its label using the EGID number of the building as unique identifier. 

    Args:
        - cfg (DictConfig): Configuration dictionary containing paths for inputs, outputs, processes to perform (like NDVI, luminosity), and metadata such as sample size and EPSG code.

    Returns:
        - None
    """
    logger.info('Starting preprocessing...')

    # security
    if cfg['security']['do_abort']:
        quit()

    # load source and target files/dir
    INPUTS = cfg['inputs']
    POLYGON_SRC = INPUTS['polygon_src']
    RASTER_DIR = INPUTS['rasters_dir']

    OUTPUTS = cfg['outputs']
    OUTPUT_DIR = OUTPUTS['output_dir']

    # load processes flags
    PROCESSES = cfg['processes']
    DO_RANGELIMIT = PROCESSES['do_rangelimit']
    DO_MASK = PROCESSES['do_mask']
    DO_SMOOTH_MASK = PROCESSES['do_smooth_mask']
    DO_DROP_BASED_ON_NDVI = PROCESSES['do_drop_based_on_ndvi']

    # load metadata
    METADATA = cfg['metadata']
    SAMPLE_SIZE = METADATA['sample_size']
    RANGELIMIT_MODE = METADATA['rangelimit_mode']
    RANGELIMIT_THRESHOLD = METADATA['rangelimit_threshold']
    EPSG = METADATA['epsg']

    # load multiprocessing metadata
    MULTIPROCESSING = cfg['multiprocessing']
    MAX_WORKERS = MULTIPROCESSING['max_workers']
    MAX_WORKERS = None if MAX_WORKERS == 0 else MAX_WORKERS
    
    # messages for processes:
    if DO_MASK:
        if DO_SMOOTH_MASK:
            logger.info("_with smooth masking")
        else:
            logger.info("_with masking")
    if DO_DROP_BASED_ON_NDVI:
        logger.info("_with dropping out of the 'bare' samples with mean NDVI greater than 0.05")
    
    # asserts and warnings if incompatibilities in or between parameters
    assert isinstance(SAMPLE_SIZE, int)
    assert isinstance(RANGELIMIT_THRESHOLD, int)
    assert isinstance(EPSG, int)
    assert RANGELIMIT_MODE in ["none", "clip", "norm", "log_norm", "self_max_norm"]
    if SAMPLE_SIZE < 128:
        warnings.warn("The sample size is set lower than 128. It might results in errors during the training.")

   # create architecture if necessary
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)
    OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, 'data/')
    os.mkdir(OUTPUT_DATA_DIR)
    
    # get rasters
    raster_src_list = []
    for r, d, f in os.walk(RASTER_DIR):
        for file in f:
            if file.endswith('.tif'):
                file_src = r + '/' + file
                file_src = file_src.replace('\\','/')
                raster_src_list.append(file_src)

    if len(raster_src_list) == 0:
        raise ValueError('No .tif file found at raster_dir location!')

    # get polygons of roofs
    roofs = gpd.read_file(POLYGON_SRC).to_crs(EPSG)
    roofs.EGID = roofs.EGID.astype(int)

    # create dataframe to store metas of samples
    df_dataset = pd.DataFrame(roofs[['EGID']])
    df_dataset.insert(1,'file_src', '-')
    
    # create samples by clipping the polygons to the rasters
    #roofs.insert(len(roofs), 'class','-')
    roofs['class'] = '-'
    samples, overlapping_roofs, nonmatching_roofs = clip_roofs_to_raster(raster_src_list, roofs, MAX_WORKERS)

    overhanging_trees = []
    sample_format = ""
    #logger.info("Processing of samples:")
    for _, (egid, (out_image, out_transform, raster_src, cat)) in tqdm(enumerate(samples.items()), desc="Processing", total = len(samples)):
        
        # dropping roofs that match less than 10% of an image (for 16bits rasters)
        if np.count_nonzero(out_image) / np.size(out_image) < 0.1: # drop if less than 5% non-zero pixels
            nonmatching_roofs.append(egid)
            continue

        num_layers = out_image.shape[0]
        sample_format = out_image.dtype
        egid = str(egid)
        raster = rasterio.open(raster_src)

        # range-limiting
        if DO_RANGELIMIT and sample_format == "uint16":
            out_image = range_limiting(out_image, RANGELIMIT_THRESHOLD, RANGELIMIT_MODE)

        # compute NDVI
        ndvi_canal = ndvi_samp_gen(out_image)
        out_image = np.concatenate([out_image, ndvi_canal])
        num_layers = num_layers + 1

        # compute luminosity
        lum_canal = lum_samp_gen(out_image)
        out_image = np.concatenate([out_image, lum_canal])
        num_layers = num_layers + 1

        # apply mask
        if DO_MASK:                
            # get ndvi      
            ndvi_arr = out_image[4, ...]
            
            # compute masks
            mask_arr = ndvi_to_mask(ndvi_arr)
            if DO_SMOOTH_MASK:
                if not DO_MASK:
                    raise ValueError("The mask needs to be computed in order to apply a smooth mask!")
                mask_arr = mask_nbh_rounding(mask_arr)

            # apply masks on each band
            for i in range(out_image.shape[0]):
                out_image[i,mask_arr] = 0
        
        # normalize sample
        out_image = resize(out_image, [num_layers, SAMPLE_SIZE, SAMPLE_SIZE], anti_aliasing=False)

        # place sample to corresponding folder
        target_tif_src = os.path.join(OUTPUT_DATA_DIR, egid + '.tif')
        target_pkl_src = os.path.join(OUTPUT_DATA_DIR, egid + '.pickle')
        df_dataset.loc[df_dataset.EGID == float(egid), 'file_src'] = f'data/{egid}.pickle'
        
        out_meta = raster.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "width": out_image.shape[2],
            "height": out_image.shape[1],
            "count": num_layers,
            "transform": out_transform,
            "crs": rasterio.CRS.from_epsg(2056),
        })

        # create .tif file for visualization
        with rasterio.open(target_tif_src, "w", **out_meta) as dst:
            dst.write(out_image)

        # create pickle file to keep the numpy array (without rasterio 8-bit transformation)
        with open(target_pkl_src, 'wb') as dst:
            pickle.dump(out_image, dst)


    # saving labelized info to csv:
    df_dataset = df_dataset.loc[df_dataset.file_src != '-',:]

    # saving dataset infos
    df_dataset.to_csv(os.path.join(OUTPUT_DIR, "dataset.csv"), index=False, sep=';')


    logger.info("In total:")
    logger.info(f"\t{len(samples)} roofs processed.")
    logger.info(f"\t{len(nonmatching_roofs)} non-matching roofs dropped.")
    if len(nonmatching_roofs) > 0:
        with open(os.path.join(OUTPUT_DIR, 'nonmatching_roofs.txt'),'w') as file:
                file.write(str(nonmatching_roofs))

    # show summary
    if DO_DROP_BASED_ON_NDVI:
        logger.info(f"\t{len(overhanging_trees)} bare roofs with overhanging trees dropped.")
        if len(overhanging_trees) > 0:
            with open(os.path.join(OUTPUT_DIR, 'overhanging_trees.txt'),'w') as file:
                file.write(str(overhanging_trees))
    logger.info("Preprocessing done.")

    # save copy of config file
    with open(os.path.join(OUTPUT_DIR, "config.yaml"),'w+') as file:
        OmegaConf.save(cfg, file.name)


if __name__ == '__main__':
    # Retrieve parameters
    cfg = OmegaConf.load('./config/preprocessing.yaml')
    preprocess(cfg['preprocessing'])
        