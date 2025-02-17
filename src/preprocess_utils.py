import os
import numpy as np
import pandas as pd
import pickle
import rasterio
from tqdm import tqdm
from loguru import logger
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
from skimage.transform import resize
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from functools import partial


def ndvi_samp_gen(arr_input):
    """
    Generate NDVI (Normalized Difference Vegetation Index) from input array.

    Args:
        - arr_input (numpy.ndarray): Input array with shape (4, height, width) 
                                   containing reflectance data. 
                                   The first channel is Red, and the fourth is NIR.

    Returns:
        - numpy.ndarray: NDVI array reshaped to (1, height, width).
    """
    R = arr_input[1,:,:].astype(float)
    NIR = arr_input[0,:,:].astype(float)
    ndvi = np.divide(NIR - R, NIR + R, out=np.zeros(R.shape, dtype=float), where= NIR + R != 0)
    ndvi = ndvi.reshape((1,ndvi.shape[0], ndvi.shape[1]))
    return ndvi


def lum_samp_gen(arr_input):
    """
    Generate luminance from RGB input array.

    Args:
        - arr_input (numpy.ndarray): Input array with shape (3, height, width) 
                                   containing RGB data.

    Returns:
        - numpy.ndarray: Luminance array reshaped to (1, height, width).
    """
    R = arr_input[0,:,:]
    G = arr_input[1,:,:]
    B = arr_input[2,:,:]
    lum = R + G + B
    lum = lum.reshape((1, lum.shape[0], lum.shape[1]))
    return lum


def ndvi_val_to_8bits(ndvi):
    """
    Convert NDVI values to 8-bit representation.

    Args:
        - ndvi (numpy.ndarray): NDVI array with values in range [-1, 1].

    Returns:
        - numpy.ndarray: 8-bit integer representation of NDVI.
    """
    return ((ndvi + 1)/2 * 256).astype(int)


def rotate_sample(sample_id, list_samples, num_per_cat, df_dataset, df_categories, shared_dict, do_save_tif):
    r, file = list_samples[sample_id]
    egid = file.split('.')[0]
    original_egid = egid.split('_')[0]
    samp_code_num = df_dataset.loc[df_dataset.EGID == float(egid)]['label'].values[0]

    # find corresponding number of rotations
    class_max_rep = np.max(list(num_per_cat.values()))
    num_cat = num_per_cat[samp_code_num]
    num_rot = np.clip(int(class_max_rep/num_cat), 0, 4) - 1

    # if under-represented sample
    if num_rot > 0:
        with open(r + "/" + file, 'rb') as input_file:
            image_arr = pickle.load(input_file)
        
        # create and save rotated samples
        for i in range(num_rot):
            image_arr_rot = np.rot90(image_arr, k=i+1, axes=(1,2))
            file_rot_name = egid + "_" + str(int((i+1)*90))

            # save rotated copy
            if do_save_tif:
                raster = rasterio.open(r + "/" + egid + ".tif")
                with rasterio.open(r + "/" + file_rot_name + ".tif", "w", **raster.meta) as dst:
                    dst.write(image_arr_rot[1:4, ...])
            with open(r + "/" + file_rot_name + ".pickle", 'wb') as output_file:
                pickle.dump(image_arr_rot, output_file)

            # add rotated samples in csv list
            samp_cat = df_categories.loc[df_categories.code_num == samp_code_num, 'cat'].values[0]
            samp_code_char = df_categories.loc[df_categories.code_num == samp_code_num, 'code_char'].values[0]
            # df_dataset.loc[len(df_dataset.index)] = [file_rot_name, samp_code_char, samp_cat + "/" + file_rot_name + ".pickle", samp_code_num, original_egid]
            shared_dict[len(df_dataset) + sample_id * 3 + i] = [file_rot_name, samp_code_char, samp_cat + "/" + file_rot_name + ".pickle", samp_code_num, original_egid]


def da_rotation(dataset_dir, df_categories, max_workers, do_save_tif):
    """
    Perform data augmentation by rotating images and saving the rotated samples.

    Args:
        - dataset_dir (str): Directory containing the dataset and CSV files.
        - categories (dict): Mapping of category names to labels.

    Returns:
        - None
    """
    dataset_list_src = os.path.join(dataset_dir, "dataset.csv")
    assert os.path.exists(dataset_list_src)

    df_dataset = pd.read_csv(dataset_list_src, sep=";")
    df_dataset.dropna(inplace=True)
    if pd.api.types.is_numeric_dtype(df_dataset.EGID):
        df_dataset.EGID = df_dataset.EGID.astype(int)

    # Get all samples
    list_samples = []
    for r, d, f in os.walk(dataset_dir):
        for file in f:
            if file.endswith('.pickle'):
                list_samples.append([r,file])

    num_per_cat = df_dataset[['label','EGID']].groupby('label').count().to_dict()['EGID']

    with Manager() as manager:
        shared_dict = manager.dict()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            part_func = partial(rotate_sample,
                                list_samples=list_samples,
                                num_per_cat=num_per_cat,
                                df_dataset=df_dataset,
                                df_categories=df_categories,
                                shared_dict=shared_dict,
                                do_save_tif=do_save_tif,
                                )
            
            futures = {executor.submit(part_func, id_samp) for id_samp in range(len(list_samples))}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Rotating"):
                future.result()


        # add new samples to dataset DataFrame
        df_dataset = pd.concat([df_dataset, pd.DataFrame.from_dict(shared_dict, orient='index', columns = df_dataset.columns)])

    # Save csv list
    df_dataset.to_csv(dataset_dir + "/" + "dataset.csv", sep=';', index=False)


def flip_sample(sample_id, list_samples, num_per_cat, df_dataset, df_categories, shared_dict, do_save_tif):
    r, file = list_samples[sample_id]
    egid = file.split('.')[0]
    original_egid = egid.split('_')[0]
    samp_code_num = df_dataset.loc[df_dataset.EGID.astype('string') == str(egid)]['label'].values[0]

    # find corresponding number of flipping
    class_max_rep = np.max(list(num_per_cat.values()))
    num_cat = num_per_cat[samp_code_num]
    num_flip = np.clip(int(class_max_rep/num_cat), 0, 2)
    
    # if under-represented sample
    if num_flip > 0:
        with open(r + "/" + file, 'rb') as input_file:
            image_arr = pickle.load(input_file)
        
        # create and save rotated samples
        suffixes = ['hor', 'vert']
        for i in range(num_flip):
            image_arr_flip = np.flip(image_arr, axis=(i+1))
            file_flip_name = egid + "_flip_" + suffixes[i]

            # save rotated copy
            if do_save_tif:
                raster = rasterio.open(r + "/" + egid + ".tif")
                with rasterio.open(r + "/" + file_flip_name + ".tif", "w", **raster.meta) as dst:
                    dst.write(image_arr_flip[1:4, ...])
            with open(r + "/" + file_flip_name + ".pickle", 'wb') as output_file:
                pickle.dump(image_arr_flip, output_file)

            # add rotated samples in csv list
            samp_cat = df_categories.loc[df_categories.code_num == samp_code_num, 'cat'].values[0]
            samp_code_char = df_categories.loc[df_categories.code_num == samp_code_num, 'code_char'].values[0]

            # df_dataset.loc[len(df_dataset.index)] = [file_rot_name, samp_code_char, samp_cat + "/" + file_rot_name + ".pickle", samp_code_num, original_egid]
            shared_dict[len(df_dataset) + sample_id * 2 + i] = [file_flip_name, samp_code_char, samp_cat + "/" + file_flip_name + ".pickle", samp_code_num, original_egid]


def da_flipping(dataset_dir, df_categories, max_workers, do_save_tif):
    """
    Perform data augmentation by flipping images and saving the rotated samples.

    Args:
        - dataset_dir (str): Directory containing the dataset and CSV files.
        - df_categories (dict): Mapping of category names to labels.

    Returns:
        - None
    """
    dataset_list_src = os.path.join(dataset_dir, "dataset.csv")
    assert os.path.exists(dataset_list_src)

    df_dataset = pd.read_csv(dataset_list_src, sep=";")
    df_dataset.dropna(inplace=True)
    if pd.api.types.is_numeric_dtype(df_dataset.EGID):
        df_dataset.EGID = df_dataset.EGID.astype(int)

    # Get all samples
    list_samples = []
    for r, d, f in os.walk(dataset_dir):
        for file in f:
            if file.endswith('.pickle'):
                list_samples.append([r,file])

    num_per_cat = df_dataset[['label','EGID']].groupby('label').count().to_dict()['EGID']

    with Manager() as manager:
        shared_dict = manager.dict()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            part_func = partial(flip_sample,
                                list_samples=list_samples,
                                num_per_cat=num_per_cat,
                                df_dataset=df_dataset,
                                df_categories=df_categories,
                                shared_dict=shared_dict,
                                do_save_tif=do_save_tif,
                                )
            
            futures = {executor.submit(part_func, id_samp) for id_samp in range(len(list_samples))}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Flipping"):
                future.result()

        # add new samples to dataset DataFrame
        df_dataset = pd.concat([df_dataset, pd.DataFrame.from_dict(shared_dict, orient='index', columns=df_dataset.columns)])

    # Save csv list
    df_dataset.to_csv(dataset_dir + "/" + "dataset.csv", sep=';', index=False)


def normalize_sample_size(sample, sample_size, method='stretching'):
    """
    Adjusts the size of a sample to a specified `sample_size` using either "stretching" or "padding" mode.
    Ensures the resulting sample has even dimensions.

    Args:
        - sample (np.ndarray): Input sample with shape `(bands, height, width)`.
        - sample_size (int): Desired size for the output sample.
        - method (str, optional): Method for resizing, either 'stretching' (resizes directly to the target size) 
                                or 'padding' (resizes and centers with black padding). Defaults to 'stretching'.

    Returns:
        - np.ndarray: The resized or padded sample with shape `(bands, sample_size, sample_size)`.
    """
    # Security
    assert method in ['stretching', 'padding']

    # Stretching mode
    if method == 'stretching':
        sample = resize(sample, [sample.shape[0], sample_size, sample_size], anti_aliasing=False)
        return sample
    
    # Padding mode
    #   _get original dimensions
    dimensions = sample.shape[1:3]
    max_side = np.argmax(dimensions)
    min_side = np.argmin(dimensions)
    max_side_size = dimensions[max_side]
    min_side_size = dimensions[min_side]
    ratio = min_side_size / max_side_size

    #   _if max side is bigger than sample_size, resize
    if max_side_size >= sample_size:
        new_min_size = int(sample_size * ratio)
        new_min_size -= new_min_size % 2 # make sure that the size is even
        new_size = [sample.shape[0], 0, 0]
        new_size[max_side+1] = sample_size
        new_size[min_side+1] = new_min_size
        sample = resize(sample, new_size, anti_aliasing=False)
    else:
        new_size = list(sample.shape)
        new_size[1:3] = [x-y for (x,y) in zip(sample.shape[1:3], [x % 2 for x in sample.shape[1:3]])]
        if tuple(new_size) != sample.shape:
            sample = resize(sample, new_size, anti_aliasing=False)

    #   _verification that both sides are even:
    assert sample.shape[1]%2 == 0
    assert sample.shape[2]%2 == 0

    #   _center and add black padding
    new_sample = np.zeros((sample.shape[0], sample_size, sample_size))
    padding_x = int((sample_size - sample.shape[1])/2)
    padding_y = int((sample_size - sample.shape[2])/2)
    new_sample[:, padding_x:padding_x + sample.shape[1], padding_y:padding_y + sample.shape[2]] = sample

    return new_sample


def get_sample(src_path: str, roof: tuple):
    """
    Extracts a raster sample from a source file for a specific roof geometry. 

    Args:
        - src_path (str): Path to the raster source file.
        - roof (tuple): A GeoDataFrame's Row containing informations about a roof.

    Returns:
        - tuple or None: If successful, returns a tuple `(egid, out_image, out_transform, raster_src, cat)`.
                       Returns `None` if the geometry cannot be cropped from the source raster.
    """
    egid = roof[0]
    geom = roof[1]
    raster_src = roof[2]
    cat = roof[3]
    with rasterio.open(src_path) as src:
        try:
            out_image, out_transform = mask(src, [geom], crop=True)
        except ValueError:
            pass
        else:
            return (egid, out_image, out_transform, raster_src, cat)
        
    return ()
         

def clip_roofs_to_raster(lst_rasters_src:list, df_roofs:list, max_workers:int)->list:
    """
    Clips a set of roofs to the matching raster images and selects the largest image for each roof.

    Args:
        - lst_rasters_src (list): List of file paths to raster images.
        - df_roofs (DataFrame): DataFrame containing roof geometries with at least two columns: 'EGID' (unique ID) and 'geometry'.

    Returns:
        - samples (list): A dictionary with egid as keys and a tuple containing the selected raster image (as an array), and the raster's transform as values.
    """
    
    # Identify roofs that overlap with each raster and add matching roofs for each raster
    lst_matching_roofs = []
    lst_matching_roofs = []
    for raster_src in lst_rasters_src:
        raster = rasterio.open(raster_src)
        lst_sub_matching = []
        for roof in df_roofs.itertuples():
            bound = roof.geometry.bounds
            bound_bbox = rasterio.coords.BoundingBox(bound[0], bound[1], bound[2], bound[3])
            if not rasterio.coords.disjoint_bounds(bound_bbox, raster.bounds):
                lst_sub_matching.append(roof.EGID)
        matching_roofs = df_roofs.loc[df_roofs.EGID.isin(lst_sub_matching)][['EGID', 'class', 'geometry']]
        lst_matching_roofs.append([(samp.EGID, samp.geometry, raster_src, samp[2]) for samp in matching_roofs.itertuples()])

    # Initialize a dictionary to hold images and transforms for each roof EGID
    dict_matching_rasters = {int(egid): [] for egid in df_roofs.EGID.values}

    # Retrieve image samples asynchronously for each raster and add to dictionary
    for id_r, raster_src in tqdm(enumerate(lst_rasters_src), total=len(lst_rasters_src), desc="  Clipping: "):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(get_sample, raster_src, roof) for roof in lst_matching_roofs[id_r]}
            for future in as_completed(futures):
                try:
                    res = future.result()
                    dict_matching_rasters[res[0]].append(res[1::])
                except Exception as exc:
                    print("An error happened during the results extraction in concurrent.futures process")

    # Select the largest image for each roof
    samples = {}
    overlapping_roofs = []
    nonmatching_roofs = []
    for (egid, lst_res) in dict_matching_rasters.items():
        img_size_max = 0
        out_image = None
        out_transform = None
        out_raster_src = None
        out_cat = ''

        # test if polygon match with one or multiple rasters:    
        if len(lst_res) == 0:
            nonmatching_roofs.append(egid)
            continue
        if len(lst_res) > 1:
                overlapping_roofs.append(egid)

        for (img, transf, raster_src, cat) in lst_res:
            if np.sum(img.shape) > img_size_max:
                img_size_max = np.sum(img.shape)
                out_image = img
                out_transform = transf
                out_raster_src = raster_src
                out_cat = cat
        samples[egid] = (out_image, out_transform, out_raster_src, out_cat)
    return samples, overlapping_roofs, nonmatching_roofs


def processes_on_sample(egid, sample, cfg, cat_dir, shared_dict, df_categories, overlapping_roofs):
    out_image = sample[0]
    out_transform = sample[1]
    raster_src = sample[2]
    cat = sample[3]

    # Load processes flags
    PROCESSES = cfg['processes']
    IMAGE_NORM_STYLE = PROCESSES['image_norm_style']
    DO_DROP_OVERLAPPING = PROCESSES['do_drop_overlapping']
    DO_DROP_BASED_ON_NDVI = PROCESSES['do_drop_based_on_ndvi']
    DO_PRODUCE_TIF = PROCESSES['do_produce_tif']

    # Load metadata
    METADATA = cfg['metadata']
    PREPROCESS_MODE = METADATA['preprocess_mode']
    SAMPLE_SIZE = METADATA['sample_size']

    # create dictionary when value is aborted
    dict_aborted = {
        "toosmall_sample": "",
        "overhanging_tree": "",
    }
    # dropping overlapping roofs
    if DO_DROP_OVERLAPPING and egid in overlapping_roofs and sample_format != 'uint16':
        return None

    num_layers = out_image.shape[0]
    sample_format = out_image.dtype
    egid = str(egid)
    raster = rasterio.open(raster_src)

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
        dict_aborted['overhanging_tree'] = str(egid)
        return dict_aborted
    
    # normalize sample
    out_image = normalize_sample_size(out_image, SAMPLE_SIZE, IMAGE_NORM_STYLE)

    # place sample to corresponding folder
    if PREPROCESS_MODE == 'training':
        if cat in cat_dir.keys():
            target_tif_src = cat_dir[cat] + '/' + egid + '.tif'
            target_pkl_src = cat_dir[cat] + '/' + egid + '.pickle'
            shared_dict[float(egid)] = df_categories.loc[df_categories.code_char == cat, 'cat'].values[0] + '/' + egid + '.pickle'
        else:
            raise ValueError(f"no category with letter '{cat}' !")
    else:
        target_tif_src = os.path.join(cat_dir['data'], egid + '.tif')
        target_pkl_src = os.path.join(cat_dir['data'], egid + '.pickle')
        shared_dict[float(egid)] = f'data/{egid}.pickle'
    
    # create tif file for visualization
    if DO_PRODUCE_TIF:
        out_meta = raster.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "width": out_image.shape[2],
            "height": out_image.shape[1],
            "count": 3,
            "transform": out_transform,
            "crs": rasterio.CRS.from_epsg(2056),
        })
        with rasterio.open(target_tif_src, "w", **out_meta) as dst:
            dst.write(out_image[1:4, ...])

    # create pickle file to keep the numpy array (without rasterio 8-bit transformation)
    with open(target_pkl_src, 'wb') as dst:
        pickle.dump(out_image, dst)

    return None
