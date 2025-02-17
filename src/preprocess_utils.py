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
import concurrent.futures


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


def ndvi_to_mask(ndvi, threshold=0.0):
    """
    Create a binary mask from NDVI values based on a threshold.

    Args:
        - ndvi (numpy.ndarray): NDVI array with shape (1, height, width).
        - threshold (float, optional): Threshold value to create the mask. Default is 0.0.

    Returns:
        - numpy.ndarray: Boolean mask indicating where NDVI is below the threshold.
    """
    return (ndvi < threshold)


def mask_nbh_rounding(mask, nbh=1, add=0):
    """
    Round neighborhood mask based on the count of neighboring positive values.

    Args:
        - mask (numpy.ndarray): Boolean mask with shape (height, width).
        - nbh (int, optional): Neighborhood size. Must be >= 1. Default is 1.
        - add (int, optional): Additional count to adjust the threshold. Default is 0.

    Returns:
        - numpy.ndarray: Boolean mask after neighborhood rounding.
    """
    assert(nbh>=1)
    assert(len(mask.shape) == 2)
    mask = mask.astype(int)
    new_mask = np.copy(mask)
    width = mask.shape[1]
    height = mask.shape[0]
    for i in range(height):
        for j in range(width):
            if mask[i,j] == 1:
                count = np.sum(mask[np.clip(i-nbh,a_min=0, a_max=None):np.clip(i+nbh + 1,a_max=height, a_min=None),
                                    np.clip(j-nbh,a_min=0, a_max=None):np.clip(j+nbh + 1, a_max=width, a_min=None),
                                    ])
                if count <= 2**(nbh+1) + add:
                    new_mask[i,j] = 0
    return new_mask.astype(bool)


def da_rotation(dataset_dir, df_categories):
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
    assert os.path.exists(dataset_dir)

    dataset_list = pd.read_csv(dataset_list_src, sep=";")
    dataset_list.dropna(inplace=True)
    if pd.api.types.is_numeric_dtype(dataset_list.EGID):
        dataset_list.EGID = dataset_list.EGID.astype(int)

    # get all samples
    list_samples = []
    for r, d, f in os.walk(dataset_dir):
        for file in f:
            if file.endswith('.pickle') and not file.endswith('global_stats.pickle'):
                list_samples.append([r,file])

    num_per_cat = dataset_list[['label','EGID']].groupby('label').count().to_dict()['EGID']

    # go through each sample
    for _, (r,file) in tqdm(enumerate(list_samples), total=len(list_samples), desc="Rotating"):
        egid = file.split('.')[0]
        original_egid = egid.split('_')[0]
        samp_code_num = dataset_list.loc[dataset_list.EGID == int(egid)]['label'].values[0]
        # get raster
        raster = rasterio.open(r + "/" + egid + ".tif")

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
                with rasterio.open(r + "/" + file_rot_name + ".tif", "w", **raster.meta) as dst:
                    dst.write(image_arr_rot)
                with open(r + "/" + file_rot_name + ".pickle", 'wb') as output_file:
                    pickle.dump(image_arr_rot, output_file)

                # add rotated samples in csv list
                samp_cat = df_categories.loc[df_categories.code_num == samp_code_num, 'cat'].values[0]
                samp_code_char = df_categories.loc[df_categories.code_num == samp_code_num, 'code_char'].values[0]
                dataset_list.loc[len(dataset_list.index)] = [file_rot_name, samp_code_char, samp_cat + "/" + file_rot_name + ".pickle", samp_code_num, original_egid]

    # save csv list
    dataset_list.to_csv(dataset_dir + "/" + "dataset.csv", sep=';', index=False)


def da_flipping(dataset_dir, df_categories):
    dataset_list_src = os.path.join(dataset_dir, "dataset.csv")
    assert os.path.exists(dataset_list_src)
    assert os.path.exists(dataset_dir)

    dataset_list = pd.read_csv(dataset_list_src, sep=";")
    dataset_list.dropna(inplace=True)
    if pd.api.types.is_numeric_dtype(dataset_list.EGID):
        dataset_list.EGID = dataset_list.EGID.astype(int)

    # get all samples
    list_samples = []
    for r, d, f in os.walk(dataset_dir):
        for file in f:
            if file.endswith('.pickle') and not file.endswith('global_stats.pickle'):
                list_samples.append([r,file])

    num_per_cat = dataset_list[['label','EGID']].groupby('label').count().to_dict()['EGID']

    # go through each sample
    for _, (r,file) in tqdm(enumerate(list_samples), total=len(list_samples), desc="Flipping"):
        egid = file.split('.')[0]
        original_egid = egid.split('_')[0]
        samp_code_num = dataset_list.loc[dataset_list.EGID.astype('string') == str(egid)]['label'].values[0]

        # get raster
        raster = rasterio.open(r + "/" + egid + ".tif")

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
                image_arr_flip = np.flip(image_arr, axis= (i+1))
                file_flip_name = egid + "_flip_" + suffixes[i]

                # save rotated copy
                with rasterio.open(r + "/" + file_flip_name + ".tif", "w", **raster.meta) as dst:
                    dst.write(image_arr_flip)
                with open(r + "/" + file_flip_name + ".pickle", 'wb') as output_file:
                    pickle.dump(image_arr_flip, output_file)

                # add rotated samples in csv list
                samp_cat = df_categories.loc[df_categories.code_num == samp_code_num, 'cat'].values[0]
                samp_code_char = df_categories.loc[df_categories.code_num == samp_code_num, 'code_char'].values[0]
                dataset_list.loc[len(dataset_list.index)] = [file_flip_name, samp_code_char, samp_cat + "/" + file_flip_name + ".pickle", samp_code_num, original_egid]

    # save csv list
    dataset_list.to_csv(dataset_dir + "/" + "dataset.csv", sep=';', index=False)


def range_limiting(sample, threshold, method='norm'):
    """
    Limit the range of a sample based on the specified method.

    Args:
        - sample (numpy.ndarray): Input array of any size.
        - threshold (int): Threshold value for limiting the range.
        - method (str, optional): Method for limiting range. 
                                Can be one of {'clip', 'norm', 'log_norm', 'self_max_norm'}. 
                                Default is 'norm'.

    Returns:
        - numpy.ndarray: Output array of the same size as the input sample, with limited range.
    """
    assert(method in ['none', 'clip','norm', 'log_norm', 'self_max_norm'])
    assert(isinstance(threshold, int))
    assert(threshold > 0)
    assert(isinstance(sample, np.ndarray))

    # methods
    def clip(sample, threshold):
        return np.clip(sample,a_min=0,a_max=threshold)
    
    def norm(sample, threshold):
        return sample / (2**16 - 1) * threshold
    
    def log_norm(sample, threshold):
        val_log_max = np.log(2**16 - 1)
        sample[sample != 0] = np.log(sample[sample != 0]) / val_log_max * threshold
        return sample

    def self_max_norm(sample, threshold):
        sample[sample != 0] = sample[sample != 0] / np.max(sample) * threshold
        return sample
    
    if method == 'none':
        return sample
    elif method == 'clip':
        return clip(sample, threshold)
    elif method == 'norm':
        return norm(sample, threshold)
    elif method == 'log_norm':
        return log_norm(sample, threshold)
    elif method == 'self_max_norm':
        return self_max_norm(sample, threshold)
    
 # add comment for next functions...
 #    


def normalize_sample_size(sample, sample_size, method='stretching'):
    # security
    assert method in ['stretching', 'padding']

    # if stretch no matter the sample's size
    if method == 'stretching':
        sample = resize(sample, [sample.shape[0], sample_size, sample_size], anti_aliasing=False)
        return sample
    
    # get original dimensions
    dimensions = sample.shape[1:3]
    max_side = np.argmax(dimensions)
    min_side = np.argmin(dimensions)
    max_side_size = dimensions[max_side]
    min_side_size = dimensions[min_side]
    ratio = min_side_size / max_side_size

    # if max side is bigger than sample_size, resize
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

    # verification that both sides are even:
    assert sample.shape[1]%2 == 0
    assert sample.shape[2]%2 == 0

    # center and add black padding
    new_sample = np.zeros((sample.shape[0], sample_size, sample_size))
    padding_x = int((sample_size - sample.shape[1])/2)
    padding_y = int((sample_size - sample.shape[2])/2)
    new_sample[:, padding_x:padding_x + sample.shape[1], padding_y:padding_y + sample.shape[2]] = sample

    return new_sample


def compute_global_stats(sample):
    features = []
    for band in range(sample.shape[0]):
        features.append(np.mean(sample[band,...]))
        features.append(np.std(sample[band,...]))
        features.append(np.median(sample[band,...]))
        features.append(np.min(sample[band,...]))
        features.append(np.max(sample[band,...]))
    return features



def get_sample(src_path: str, roof: tuple):
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
    for id_r, raster_src in tqdm(enumerate(lst_rasters_src), total=len(lst_rasters_src), desc="Clipping: "):
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(get_sample, raster_src, roof) for roof in lst_matching_roofs[id_r]}
            for future in concurrent.futures.as_completed(futures):
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


# small util for testing reset
def test_reset(dataset_dir):
    for r, d, f in os.walk(dataset_dir):
        for file in f:
            file_name_split = file.split('_')

            if len(file_name_split) > 1 and not file.endswith('.csv'):
                os.remove(r + "/" + file)
                

if __name__ == '__main__':
    # test

    src_raster = "./data/sources/scratch_dataset"
    src_roofs = "./data/sources/gt_tot.gpkg"

    roofs = gpd.read_file(src_roofs).to_crs(2056)

    lst_rasters_src = []
    raster_list = []
    for r, d, f in os.walk(src_raster):
        for file in f:
            if file.endswith('.tif'):
                file_src = r + '/' + file
                file_src = file_src.replace('\\','/')
                lst_rasters_src.append(file_src)
                raster_list.append(rasterio.open(file_src))

    #samples = clip_roofs_to_raster(lst_rasters_src, roofs.sample(frac=0.05))
    egid_very_small = 295077114
    egid_small = 1029750
    egid_big = 1012243
    egid_very_big = 11524802

    list_arr_imgs = []
    norm_boundaries = [[0,255],[0,255],[0,255],[0,255],[-1,1],[0,765]]
    for egid, title in zip([egid_very_small, egid_small, egid_big, egid_very_big], ['Very small', 'Small', 'Big', 'Very big']):

        roof = roofs.loc[roofs.EGID.astype(int) == egid]
        geom = roof.geometry.values[0]

        out_image = np.empty((0,0))
        for raster in raster_list:
            # catch the error when polygon doesn't match raster and just continue to next raster
            try:
                out_image, out_transform = mask(raster, [geom], crop=True)
                sample_formats = out_image.dtype
            except ValueError:
                continue
        print(title)
        ndvi_canal = ndvi_samp_gen(out_image)
        out_image = np.concatenate([out_image, ndvi_canal])
        print(f"\tshape : {out_image.shape}")
        print(f"\tarea : {roof.area.values[0]}")
        print(f"\tmin value: {np.min(out_image[4,...])}")
        print(f"\tmax value: {np.max(out_image[4,...])}")
        out_image_padd = normalize_sample_size(out_image, 512, method='padding')
        out_image_stretch = normalize_sample_size(out_image, 512, method='stretching')
        list_arr_imgs.append((out_image_stretch, out_image_padd))
        print(f"\tnew shape: {out_image_padd.shape}")
        print(f"\tnew min value: {np.min(out_image_padd[4,...])}")
        print(f"\tnew max value: {np.max(out_image_padd[4,...])}")
        print("---")

    fig, axs = plt.subplots(4,2, figsize=(5,15), sharex=True, sharey=True)
    for i in range(4):
        axs[i, 0].imshow(np.moveaxis(list_arr_imgs[i][0][1:4, ...].astype('uint8'), 0,2))
        axs[i, 1].imshow(np.moveaxis(list_arr_imgs[i][1][1:4, ...].astype('uint8'), 0,2))
    axs[0, 0].set_title('stretched')
    axs[0, 1].set_title('padded')
    plt.show()
    plt.close()
    
    quit()


    dataset_dir = './data/dataset'
    df_categories = pd.read_csv('./data/sources/class_labels_multi.csv', sep=';')
    da_flipping(dataset_dir, df_categories)
    quit()

    df_categories = pd.read_csv("./data/dataset_test/class_names.csv", sep=';') 
    print(df_categories)
    da_flipping("./data/dataset_test", df_categories)
    quit()
    img_arr = np.array([
        [1,2,3],
        [4,5,6],
    ])
    img_arr = np.arange(1,10).reshape((3,3))
    print(img_arr)
    print(np.flip(img_arr, axis=0))
    print(np.flip(img_arr, axis=1))

    quit()

    