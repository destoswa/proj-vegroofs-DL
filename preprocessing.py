import os
import shutil
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import dask_geopandas as dg
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
from src.preprocess_utils import range_limiting, ndvi_samp_gen, lum_samp_gen, ndvi_to_mask, mask_nbh_rounding, da_rotation, da_flipping
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
    CHM_SRC = INPUTS['chm_src']
    CLASS_LABELS_DIR = INPUTS['class_labels_dir']

    OUTPUTS = cfg['outputs']
    OUTPUT_DIR = OUTPUTS['output_dir']

    # load processes flags
    PROCESSES = cfg['processes']
    DO_RANGELIMIT = PROCESSES['do_rangelimit']
    DO_MASK = PROCESSES['do_mask']
    DO_SMOOTH_MASK = PROCESSES['do_smooth_mask']
    DO_DA_ROTATION = PROCESSES['do_da_rotation']
    DO_DA_FLIPPING = PROCESSES['do_da_flipping']
    DO_DROP_OVERLAPPING = PROCESSES['do_drop_overlapping']
    DO_DROP_BASED_ON_NDVI = PROCESSES['do_drop_based_on_ndvi']

    # load metadata
    METADATA = cfg['metadata']
    SAMPLE_SIZE = METADATA['sample_size']
    RANGELIMIT_MODE = METADATA['rangelimit_mode']
    RANGELIMIT_THRESHOLD = METADATA['rangelimit_threshold']
    EPSG = METADATA['epsg']

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
    raster_list = []

    for r, d, f in os.walk(RASTER_DIR):
        for file in f:
            if file.endswith('.tif'):
                file_src = r + '/' + file
                file_src = file_src.replace('\\','/')
                raster_list.append(rasterio.open(file_src))

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
    
    # create samples by looping on polygons
    logger.info("_Clipping samples...")
    overlapping_roofs = []
    nonmatching_roofs = []
    overhanging_trees = []
    list_egids_not_overlapping_completely = []
    histogram_range_values, histogram_range_bins = np.histogram(0, bins=100, range=(0, RANGELIMIT_THRESHOLD))
    for roof in tqdm(roofs.itertuples(index=True), total=len(roofs), desc="Clipping"):

        # compute bbox of the roof
        geom = roof.geometry

    # Loop on batches
    if BATCH_SIZE == 0:
        BATCH_SIZE = len(roofs)
    for batch_idx, sub_roofs in enumerate(chunks(roofs, BATCH_SIZE)):
        if BATCH_SIZE < len(roofs):
            print(f"Processing batch {batch_idx} / {int(np.ceil(len(roofs) / BATCH_SIZE - 1))}")
            
        # Overlay on CHM
        print("Overlaying with CHM")
        CHM_GPD = dg.read_file(CHM_SRC, chunksize=100000)
        delayed_partitions = CHM_GPD.to_delayed()
        for _, delayed_partition in tqdm(enumerate(delayed_partitions), total=len(delayed_partitions), desc="Overlaying"):
            # Compute the partition (convert to a GeoDataFrame)
            partition_gdf = delayed_partition.compute()
            # Perform operation on the partition
            sub_roofs = gpd.overlay(sub_roofs, partition_gdf, how='difference', keep_geom_type=True)

        # Create samples by clipping the polygons to the rasters
        samples, sub_overlapping_roofs, sub_nonmatching_roofs = clip_roofs_to_raster(
            raster_src_list, sub_roofs, MAX_WORKERS)
        nonmatching_roofs.append(sub_nonmatching_roofs)
        overlapping_roofs.append(sub_overlapping_roofs)

        # loop over the rasters to find the one matching
        matching_rasters = []
        matching_images = []
        sample_formats = ""
        for raster in raster_list:
            # catch the error when polygon doesn't match raster and just continue to next raster
            try:
                out_image, out_transform = mask(raster, [geom], crop=True)
                sample_formats = out_image.dtype
            except ValueError:
                continue
            else:
                if np.count_nonzero(out_image) / np.size(out_image) < 0.1: # drop if less than 5% non-zero pixels
                    nonmatching_roofs.append(egid)
                    continue
                matching_rasters.append(raster)
                matching_images.append((out_image, out_transform))

        # test if polygon match with one or multiple rasters:    
        if len(matching_rasters) == 0:
            nonmatching_roofs.append(egid)
            continue
        if len(matching_rasters) > 1:
            if DO_DROP_OVERLAPPING and sample_formats != 'uint16':
                overlapping_roofs.append(egid)
                continue
            else:
                img_size_max = np.sum(matching_images[0][0].shape)
                for img, transf in matching_images:
                    if np.sum(img.shape) > img_size_max:
                        img_size_max = np.sum(img.shape)
                        out_image = img
                        out_transform = transf
                        list_egids_not_overlapping_completely.append(egid)
        
        num_layers = out_image.shape[0]


        # range-limiting
        if DO_RANGELIMIT and sample_formats == "uint16":
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
        if DO_DROP_BASED_ON_NDVI and cat == 'b':
            if np.nanmean(ndvi_canal) > 0.05:
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
        
        # normalize sample
        out_image = resize(out_image, [num_layers, SAMPLE_SIZE, SAMPLE_SIZE], anti_aliasing=False)

        # place sample to corresponding folder
        if cat in cat_dir.keys():
            target_tif_src = cat_dir[cat] + '/' + egid + '.tif'
            target_pkl_src = cat_dir[cat] + '/' + egid + '.pickle'
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
    
    #print(f"List of egids not completely overlapping : {list_egids_not_overlapping_completely}")

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
        
    # show summary
    logger.info("In total:")
    logger.info(f"\t{len(nonmatching_roofs)} non-matching roofs dropped.")

    if DO_DROP_OVERLAPPING and sample_formats != 'uint16':
        logger.info(f"\t{len(overlapping_roofs)} overlapping roofs dropped.")

    if DO_DROP_BASED_ON_NDVI:
        logger.info(f"\t{len(overhanging_trees)} bare roofs with overhanging trees dropped.")
    logger.info("Preprocessing done.")
    
    print("Non-matching :")
    print(nonmatching_roofs)

    # save copy of config file
    with open(os.path.join(OUTPUT_DIR, "config.yaml"),'w+') as file:
        OmegaConf.save(cfg, file.name)


if __name__ == '__main__':
    # Retrieve parameters
    cfg = OmegaConf.load('./config/preprocessing.yaml')
    preprocess(cfg['preprocessing'])
        