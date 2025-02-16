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
from src.preprocess_utils import range_limiting, ndvi_samp_gen, lum_samp_gen, ndvi_to_mask, mask_nbh_rounding, da_rotation, da_flipping, clip_roofs_to_raster, normalize_sample_size, compute_global_stats
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
    CLASS_LABELS_DIR = INPUTS['class_labels_dir']

    OUTPUTS = cfg['outputs']
    OUTPUT_DIR = OUTPUTS['output_dir']

    # load processes flags
    PROCESSES = cfg['processes']
    IMAGE_NORM_STYLE = PROCESSES['image_norm_style']
    DO_RANGELIMIT = PROCESSES['do_rangelimit']
    DO_MASK = PROCESSES['do_mask']
    DO_SMOOTH_MASK = PROCESSES['do_smooth_mask']
    DO_DA_ROTATION = PROCESSES['do_da_rotation']
    DO_DA_FLIPPING = PROCESSES['do_da_flipping']
    DO_DROP_OVERLAPPING = PROCESSES['do_drop_overlapping']
    DO_DROP_BASED_ON_NDVI = PROCESSES['do_drop_based_on_ndvi']
    DO_DROP_SMALL_SAMPLES = PROCESSES['do_drop_small_samples']
    DO_GLOBAL_STATS = PROCESSES['do_global_stats']

    # load metadata
    METADATA = cfg['metadata']
    SAMPLE_SIZE = METADATA['sample_size']
    RANGELIMIT_MODE = METADATA['rangelimit_mode']
    RANGELIMIT_THRESHOLD = METADATA['rangelimit_threshold']
    EPSG = METADATA['epsg']
    SAMPLE_SIZE_THRESHOLD = METADATA['sample_size_threshold']

    # load multiprocessing metadata
    MULTIPROCESSING = cfg['multiprocessing']
    MAX_WORKERS = MULTIPROCESSING['max_workers']
    MAX_WORKERS = None if MAX_WORKERS == 0 else MAX_WORKERS
    print(MAX_WORKERS)
    
    # messages for processes:
    if DO_MASK:
        if DO_SMOOTH_MASK:
            logger.info("_with smooth masking")
        else:
            logger.info("_with masking")
    if DO_DA_ROTATION:
        logger.info("_with data augmentation (rotation)")
    if DO_DA_FLIPPING:
        logger.info("_with data augmentation (flipping)")
    if DO_DROP_OVERLAPPING:
        logger.info("_with dropping out of the samples overlapping multiple rasters")
    if DO_DROP_BASED_ON_NDVI:
        logger.info("_with dropping out of the 'bare' samples with mean NDVI greater than 0.05")
    
    # asserts and warnings if incompatibilities in or between parameters
    assert isinstance(SAMPLE_SIZE, int)
    assert isinstance(RANGELIMIT_THRESHOLD, int)
    assert isinstance(EPSG, int)
    assert RANGELIMIT_MODE in ["none", "clip", "norm", "log_norm", "self_max_norm"]
    if SAMPLE_SIZE < 128:
        warnings.warn("The sample size is set lower than 128. It might results in errors during the training.")


    # categories of samples
    df_categories = pd.read_csv(CLASS_LABELS_DIR, sep=';')

    # create architecture if necessary
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    cat_dir = {}
    cat_dir['.'] = OUTPUT_DIR

    for let, cat in df_categories[['code_char','cat']].values:
        cat_dir[let] = OUTPUT_DIR + "/" + cat

    for dir in cat_dir.values():
        if not os.path.exists(dir):
            os.mkdir(dir) 

    # save categories to csv
    df_categories.to_csv(os.path.join(OUTPUT_DIR, "class_names.csv"), sep=';', index=False)
    
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

    # saving samples metadata:
    roofs[['EGID','class','area', 'year']].to_csv(os.path.join(OUTPUT_DIR,"samples_metadata.csv"), sep=';', index=False, encoding='utf-8-sig')

    # create dataframe to store metas of samples
    df_dataset = pd.DataFrame(roofs[['EGID', 'class']])
    df_dataset.insert(2,'file_src', '-')
    df_dataset.insert(3,'label', 0)

    # add row with class code in dataframe:
    for let, num in df_categories[['code_char','code_num']].values:
        df_dataset.loc[df_dataset['class'] == let, 'label'] = num
    
    # create samples by clipping the polygons to the rasters
    samples, overlapping_roofs, nonmatching_roofs = clip_roofs_to_raster(raster_src_list, roofs, MAX_WORKERS)

    histogram_range_values, histogram_range_bins = np.histogram(0, bins=100, range=(0, RANGELIMIT_THRESHOLD))
    overhanging_trees = []
    toosmall_samples = []
    sample_format = ""
    for _, (egid, (out_image, out_transform, raster_src, cat)) in tqdm(enumerate(samples.items()), total=len(samples), desc='Processing'):
        # dropping overlapping roofs
        if DO_DROP_OVERLAPPING and egid in overlapping_roofs and sample_format != 'uint16':
            continue
        
        # dropping roofs that match less than 10% of an image (for 16bits rasters)
        if np.count_nonzero(out_image) / np.size(out_image) < 0.1: # drop if less than 5% non-zero pixels
            nonmatching_roofs.append(egid)
            continue
        
        # dropping roofs that are too small
        roof_area = float(out_image.shape[1] * out_image.shape[2])
        if DO_DROP_SMALL_SAMPLES and roof_area < SAMPLE_SIZE_THRESHOLD:
            toosmall_samples.append(egid)
            continue


        num_layers = out_image.shape[0]
        sample_format = out_image.dtype
        egid = str(egid)
        raster = rasterio.open(raster_src)

        # range-limiting
        if DO_RANGELIMIT and sample_format == "uint16":
            out_image = range_limiting(out_image, RANGELIMIT_THRESHOLD, RANGELIMIT_MODE)
        histogram_range_values += np.histogram(out_image, bins=100, range=(0, RANGELIMIT_THRESHOLD))[0]

        # compute NDVI
        ndvi_canal = ndvi_samp_gen(out_image)
        out_image = np.concatenate([out_image, ndvi_canal])
        num_layers = num_layers + 1

        # compute luminosity
        lum_canal = lum_samp_gen(out_image)
        out_image = np.concatenate([out_image, lum_canal])
        num_layers = num_layers + 1
        
        # dropping out bare samples based on ndvi value
        if DO_DROP_BASED_ON_NDVI and cat == 'b' and np.nanmean(ndvi_canal) > 0.05:
            overhanging_trees.append(egid)
            continue

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
        
        # if compute global-stats
        DO_GLOBAL_STATS = True
        if DO_GLOBAL_STATS:
            global_stats = compute_global_stats(out_image)
            

        # normalize sample
        out_image = normalize_sample_size(out_image, SAMPLE_SIZE, IMAGE_NORM_STYLE)
        #out_image = resize(out_image, [num_layers, SAMPLE_SIZE, SAMPLE_SIZE], anti_aliasing=False)

        # place sample to corresponding folder
        if cat in cat_dir.keys():
            target_tif_src = cat_dir[cat] + '/' + egid + '.tif'
            target_pkl_src = cat_dir[cat] + '/' + egid + '.pickle'
            target_gs_pkl_src = cat_dir[cat] + '/' + egid + '_global_stats.pickle'
            df_dataset.loc[df_dataset.EGID == float(egid), 'file_src'] = df_categories.loc[df_categories.code_char == cat, 'cat'].values[0] + '/' + egid + '.pickle'
        else:
            raise ValueError(f"no category with letter '{cat}' !")
        

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

        # create pickle file to keep the global stats
        if DO_GLOBAL_STATS:
            with open(target_gs_pkl_src, 'wb') as dst:
                pickle.dump(global_stats, dst)

    # saving histogram of range of values
    histogram_range_values = (histogram_range_values.astype(float) / len(roofs)).astype(int)
    bands = ['Red', 'Green', 'Blue', 'NIR']
    fig, axs = plt.subplots(4,1)
    for i in range(4):
        axs[i].bar(histogram_range_bins[:-1], histogram_range_values, width=10, log=True)
        axs[i].set_title(bands[i])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,'histogram_range_values.png'))
    plt.close()

    # saving labelized info to csv:
    df_dataset = df_dataset.loc[df_dataset.file_src != '-',:]
    
    # add column for keeping track of original egid:
    df_dataset['original_egid'] = df_dataset.loc[:,'EGID'].astype(int).astype('str')

    # saving dataset infos
    df_dataset.to_csv(os.path.join(OUTPUT_DIR, "dataset.csv"), index=False, sep=';')

    # data augmentation
    # _rotation:
    if DO_DA_ROTATION:
        logger.info("Data augmentation (rotation)...")
        da_rotation(OUTPUT_DIR, df_categories)
        
    # _flipping:
    if DO_DA_FLIPPING:
        logger.info("Data augmentation (flipping)...")
        da_flipping(OUTPUT_DIR, df_categories)

    logger.info("In total:")
    logger.info(f"\t{len(samples)} roofs processed.")
    logger.info(f"\t{len(nonmatching_roofs)} non-matching roofs dropped.")
    if len(nonmatching_roofs) > 0:
        with open(os.path.join(OUTPUT_DIR, 'nonmatching_roofs.txt'),'w') as file:
                file.write(str(nonmatching_roofs))

    # show summary
    if DO_DROP_OVERLAPPING and sample_format != 'uint16':
        logger.info(f"\t{len(overlapping_roofs)} overlapping roofs dropped.")
        if len(overlapping_roofs) > 0:
            with open(os.path.join(OUTPUT_DIR, 'overlapping_roofs.txt'),'w') as file:
                file.write(str(overlapping_roofs))

    if DO_DROP_BASED_ON_NDVI:
        logger.info(f"\t{len(overhanging_trees)} bare roofs with overhanging trees dropped.")
        if len(overhanging_trees) > 0:
            with open(os.path.join(OUTPUT_DIR, 'overhanging_trees.txt'),'w') as file:
                file.write(str(overhanging_trees))

    if DO_DROP_SMALL_SAMPLES:
        logger.info(f"\t{len(toosmall_samples)} roofs not reaching the samples size threshold dropped.")
        if len(toosmall_samples) > 0:
            with open(os.path.join(OUTPUT_DIR, 'too_small_roofs.txt'),'w') as file:
                file.write(str(toosmall_samples))
    logger.info("Preprocessing done.")

    # save copy of config file
    with open(os.path.join(OUTPUT_DIR, "config.yaml"),'w+') as file:
        OmegaConf.save(cfg, file.name)


if __name__ == '__main__':
    # Retrieve parameters
    cfg = OmegaConf.load('./config/preprocessing.yaml')
    preprocess(cfg['preprocessing'])
        