import numpy as np
import rasterio
import rasterio.coords
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
import concurrent.futures
import threading
import os
from tqdm import tqdm
from functools import partial
import time
from dask.distributed import Client, LocalCluster, as_completed
import dask_geopandas as dgpd
from rasterstats import zonal_stats
import scipy.stats
from dask.dataframe import from_delayed


class clip_raster(object):
    def __init__(self, list_rasters, list_geom):
        self.matching_rasters = []
        self.list_rasters = list_rasters
        self.list_geom = list_geom

    def setup(self):
        print(len(self.list_geom))
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            #_ = list(tqdm(executor.map(self.clip_raster, list_geom), total=len(list_geom)))
            results = executor.map(self.clip_raster, range(len(self.list_geom)))
        """for res in results:
            self.matching_rasters.append(res)"""
        
        return self.matching_rasters

    def clip_raster(self, geom_id):
        #return 1
        #self.matching_rasters.append(1)

        for raster in self.list_rasters:
            try:
                out_image, out_transform = mask(raster, [self.list_geom[geom_id]], crop=True)
            except ValueError:
                return
            else:
                self.matching_rasters.append((out_image, out_transform))
     
def kurtosis(arr):
    try:
        k = scipy.stats.kurtosis(arr.compressed())
    except AttributeError:  # Not a masked array - 'numpy.ndarray' object has no attribute 'compressed'
        k = scipy.stats.kurtosis(arr)
    return k


def zonal_statistics(row, *args, **kwargs):
    from random import randint
    if randint(0, 3) == 1:
        raise RuntimeError('Fake random error to test')

    gdf = row.compute()
    return gdf.join(pd.DataFrame(zonal_stats(gdf, *args, **kwargs), index=gdf.index))

def get_stats(src_path: str, geom: dict):
    with rasterio.open(src_path) as src:
        try:
            out_image, out_transform = mask(src, [geom], crop=True)
            sample_formats = out_image.dtype
        except ValueError:
            pass
        else:
            return (out_image, out_transform)
    return ()

def main():
    roofs = gpd.read_file("./data/sources/gt_tot.gpkg").to_crs(2056)
    #roofs = roofs.iloc[0:100,:]
    dataset_src = "./data/sources/tlm_dataset"
    raster_list = []
    raster_list_src = []
    matching_rasters = []
    for r, d, f in os.walk(dataset_src):
        for file in f:
            if file.endswith('.tif'):
                file_src = r + '/' + file
                file_src = file_src.replace('\\','/')
                raster = rasterio.open(file_src)
                #print(raster.bounds)
                raster_list.append(rasterio.open(file_src))
                raster_list_src.append(file_src)

    """geoms = roofs.geometry
    bounds = geoms.bounds
    #print(bounds)
    lst_matching_roofs = []
    for _, raster in tqdm(enumerate(raster_list), total=len(raster_list)):
        lst_sub_matching = []
        for roof in roofs.itertuples():
            bound = roof.geometry.bounds
            bound_bbox = rasterio.coords.BoundingBox(bound[0], bound[1], bound[2], bound[3])
            if not rasterio.coords.disjoint_bounds(bound_bbox, raster.bounds):
                lst_sub_matching.append(roof.EGID)
        lst_matching_roofs.append(roofs.loc[roofs.EGID.isin(lst_sub_matching)].geometry)

    for id_r, raster in enumerate(raster_list):
        print(f"{raster_list_src[id_r]} - {len(lst_matching_roofs[id_r])} matching roofs")


    output = []
    for id_r, raster_src in tqdm(enumerate(raster_list_src), total=len(raster_list_src)):
        #roofs_geom = [roof.geometry for roof in lst_matching_roofs[id_r]]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(get_stats, raster_src, geom) for geom in lst_matching_roofs[id_r]}
            for future in concurrent.futures.as_completed(futures):
                try:
                    output.append(future.result())
                except Exception as exc:
                    print("%r generated an exception: %s" % exc)
    print(len(output))


    roof_geoms = roofs.geometry.values
    for roof in tqdm(roofs.itertuples(), total=len(roofs)):
        for raster_src in raster_list_src:
            cluster = LocalCluster()
            client = Client(cluster)
            stats = ["mean", "std"]
            add_stats = {"kurotsis": kurtosis}

            gdf = gpd.GeoDataFrame(roof)
            ddf = dgpd.from_geopandas(gdf, npartitions=16)

            meta = ddf._meta.join(pd.DataFrame(columns=stats+list(add_stats.keys())))

            futures = [client.submit(zonal_statistics, p, raster=raster_src, stats=stats, add_stats=add_stats, all_touched=True) for p in ddf.partitions]
            #res = ddf.map_partitions(zonal_statistics, meta=meta, raster=raster_src, stats=stats, add_stats=add_stats, all_touched=True)

            completed = []
            for future in as_completed(futures):
                try:
                    data = future.result()
                    completed.append(future)
                except Exception as exc:
                    pass

            results = from_delayed(completed, meta=meta, verify_meta=False).compute()

            print(gdf.head())
            print(f"{len(gdf)} rows")
            print(results.head())
            print(f"{len(results)} rows")


    clip = clip_raster(raster_list, roof_geoms)
    results = clip.setup()
    print(results)"""
    """for roof in tqdm(roofs.itertuples(), total=len(roofs)):
        geom = roof.geometry
        clip = clip_raster(roof.geometry)
        results = clip.setup(raster_list)
        print(results)


        partial_clip_raster = partial(clip_raster, geom=roof.geometry)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            matching_rasters.append()


    for roof in tqdm(roofs.itertuples(), total=len(roofs)):
        egid = roof.EGID
        geom = roof.geometry
        for raster_src in raster_list_src:
            with rasterio.open(raster_src) as src:
                windows = [window for ij, window in src.block_windows()]
                profile = src.profile
                profile.update(blockxsize=128, blockysize=128, tiled=True)

                def process(window):
                    try 
"""
    time_full = time.time()
    time_mask = 0
    #roofs = roofs.iloc[0:100,:]
    for roof in tqdm(roofs.itertuples(), total=len(roofs)):
        geom = roof.geometry
        for raster in raster_list:
            # catch the error when polygon doesn't match raster and just continue to next raster
            try:
                time_mask_start = time.time()
                out_image, out_transform = mask(raster, [geom], crop=True)
                time_mask += time.time() - time_mask_start
                sample_formats = out_image.dtype
            except ValueError:
                continue
            else:
                matching_rasters.append((out_image, out_transform))

    time_full = time.time() - time_full
    print(f"time full : {time_full}")
    print(f"time mask : {time_mask}")
if __name__ == '__main__':
    main()
