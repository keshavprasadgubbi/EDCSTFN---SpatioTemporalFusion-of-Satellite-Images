#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:02:10 2020

@author: gunjanthakuria
"""


import copy
import glob
import os
import shutil
import sys
import traceback
import warnings
try:
    import osr
    import gdal
except ImportError:
    from gdal import osr, gdal
import pandas as pd
import numpy as np
import rasterio
from datetime import timedelta
from distutils.dir_util import copy_tree
from rasterio.mask import mask
from netCDF4 import Dataset
from xml.dom.minidom import parseString
from zipfile import ZipFile

if not sys.warnoptions:
    warnings.simplefilter("ignore")

creodias_codes='creodias-finder'
sys.path.append(creodias_codes)

from creodias_finder import query
from creodias_finder import download

creodias_user = os.environ.get('CREODIAS_USER')
creodias_password = os.environ.get('CREODIAS_PASSWORD')

import cut
import logger 
import shape as shp
import utils
from config import Identifier
from data import save_cloud_data, get_cloud_data

logger = logger.get_logger(__file__)

class DownloadSentinelS3():
    satellite = 'S3'

    def __init__(self, config):
        self.config = config
        self.start_date = self.config.start_date
        self.end_date = self.config.end_date
        self.shape_dict = shp.get_shape_dict(self.config.shape_dir)
        self.weekly_retained_files = 3

    @staticmethod
    def S3download_creodias(config, shape_path, record_date_str):
        try:
            # specify the search polygon
            footprint = shp.read_shape_file(shape_path).geometry.loc[0]
            result = query.query('Sentinel3',
                                start_date=record_date_str,
                                end_date=record_date_str,
                                geometry=footprint,
                                productType='RBT',
                                orbitDirection='descending')
            if bool(result)==False:
                logger.info("S3 data not available for plant %s on %s", shape_path, record_date_str)
                return None, None
            ids = [key['id'] for key in result.values()]
            out = os.path.join(config.get_image_dir(satellite='S3'), record_date_str + 'S3.zip')
            download.download(ids[0], creodias_user, creodias_password, outfile=out, show_progress=True)
            logger.info("Download Completed for %s %s", config.plant, record_date_str)
            # Title of product
            sen3title = result[list(result.keys())[0]]['properties']['title']
            return sen3title, out
        except Exception as e:
            logger.error("Exception in S3download_creodias for %s exception: %s trace: %s", record_date_str, str(e), traceback.format_exc())
            return None,None

    @staticmethod
    def unzipfile(zip_file_name):
        try:
            with ZipFile(zip_file_name, "r") as zip:
                logger.debug('Extracting Sentinel3 files %s', zip_file_name)
                zip.extractall(path=os.path.dirname(zip_file_name))
                return True
        except:
            logger.error("Oops! Exception occurred while unzipping %s. trace: %s", zip_file_name, traceback.format_exc())
            return False
    
    def get_region_epsg(self):
        plant_shape = shp.read_shape_file(self.shape_dict[Identifier.FullPlant]).geometry
        lat, lon = plant_shape.centroid.loc[0].y, plant_shape.centroid.loc[0].x
        return utils.calculate_epsg(lat, lon)

    def convertS3_cloud(self, path, out_image_path):
        with  Dataset(os.path.join(path, "flags_an.nc")) as flags_nc:
            cloud = flags_nc.variables["cloud_an"][:,:]
        with Dataset(os.path.join(path, "geodetic_an.nc")) as geod_nc:
            lat = geod_nc.variables["latitude_an"][:]
            lon = geod_nc.variables["longitude_an"][:]  
            nx = geod_nc.dimensions['columns'].size
            ny = geod_nc.dimensions['rows'].size

        # Latitude and longitude bounds
        xmin, ymin, xmax, ymax = [lon.min(), lat.min(), lon.max(), lat.max()]

        xres = (xmax - xmin) / float(nx)
        yres = (ymax - ymin) / float(ny)
        geotransform = (xmin, xres, 0, ymax, 0, -yres)
        
        # Create cloud raster image
        cld_ds = gdal.GetDriverByName('GTiff').Create('cld_img.tif', nx, ny, 1, gdal.GDT_UInt16)
        cld_ds.SetGeoTransform(geotransform)    
        srs = osr.SpatialReference()            
        srs.ImportFromEPSG(4326)                
        cld_ds.SetProjection(srs.ExportToWkt()) 
        ndv_cloud = int(cloud.fill_value)               # no data value for cloud
        cld_ds.GetRasterBand(1).WriteArray(cloud)
        cld_ds.GetRasterBand(1).SetNoDataValue(ndv_cloud)
        # TODO: Calculating region EPSG
        target_epsg = 'EPSG:' + str(self.get_region_epsg())
        output = gdal.Warp(out_image_path, cld_ds, format='GTiff', srcSRS='EPSG:4326', dstSRS=target_epsg, resampleAlg='near')
        cld_ds = None
        output = None
        return True

    def cloud_to_binary(self, cloud_file, out_cloud_file):
        cloud_file=rasterio.open(cloud_file)
        cloud_file_array=cloud_file.read()
        # convert clud pixels to Binary; 0 value means no cloud which has been assigned the number 100, so 100 means no cloud     
        cloud_binary=np.where(cloud_file_array==0,100,200)
        cloud_binary=cloud_binary.astype('uint16')
        out_meta = cloud_file.meta.copy()  
        out_meta.update({"driver": "GTiff",                        
                        "height": cloud_file_array.shape[1],
                        "width": cloud_file_array.shape[2],
                        "transform": cloud_file.transform})
        
        logger.info("writing binary cloud file: %s", out_cloud_file)
        with rasterio.open(out_cloud_file, "w", **out_meta) as dest:
            dest.write(cloud_binary)
        return True
   

    def resample(self, in_file, out_file, xres,yres):
        try:
            output = gdal.Warp(out_file, in_file, format='GTiff', xRes=xres, yRes=yres, resampleAlg='near')
            output = None
        except Exception as e:
            logger.error("Exception in resample ex: %s trace: %s", str(e), traceback.format_exc())

    @staticmethod
    def get_s2_dates(config, cloud_threshold):
        cloud_file_path = os.path.join(config.out_dir, 'cloud_data_S2_creodias.csv')
        satellite = 'S2'
        s2_dates_df = None
        if "db" in config.sinks:
            s2_dates_df = get_cloud_data(config, satellite=satellite, approved_only=True)
        elif os.path.exists(cloud_file_path):
            df = pd.read_csv(cloud_file_path)
            s2_dates_df = df[(df['shape'] == Identifier.FullPlant)&(df['cloudcoverage'] <= cloud_threshold)].loc[:,'date']
        else:
            logger.warning("S2 Cloud file missing won't be able to preserve s2 dates")
        if s2_dates_df is not None and not s2_dates_df.empty:
            file_list = glob.glob(os.path.join(config.get_image_dir(satellite=satellite), 's2-*', 'S2_coreg_shape_20m_B12_8A_*.tif'))
            if len(config.allowed_tiles) > 0:
                file_list = [f for f in file_list for tile in config.allowed_tiles if tile in f]
            s2_dates = list(map(lambda x: os.path.basename(x)[-14:-4], file_list))
            s2_present_dates_df = pd.DataFrame(s2_dates, columns=['date'])
            return pd.merge(s2_dates_df, s2_present_dates_df, on='date')['date'].to_list()
        return []

    def process_weekly_data(self, week_data, s2_dates):
        if len(week_data) > 0:
            WeekData = pd.DataFrame.from_records(week_data)
            tile_df = WeekData[['date','tile_cloud']].drop_duplicates()
            cloud_df = WeekData.pivot(columns='shape', values='cloudcoverage', index='date').reset_index()
            pixel_df = WeekData.pivot(columns='shape', values='totalpixels', index='date').reset_index()
            df_ = pd.merge(pixel_df, cloud_df, how='left', on=['date'], suffixes=('_totalpixels','_cloud'))
            pivot_df = pd.merge(df_, tile_df, how='left', on=['date'])
            pivot_df.loc[:, 'is_S2'] = pivot_df['date'].apply(lambda x: x in s2_dates)
            pivot_df.sort_values(by=['is_S2', Identifier.FullPlant + '_cloud'], ascending=(False, True), inplace=True)
            pivot_df.reset_index(drop=True, inplace=True)
            pivot_df.loc[:, 'is_deleted'] = 0
            logger.info("sorted entries added to Weekdata: %s", pivot_df)
            for index in pivot_df[pivot_df[Identifier.FullPlant + '_cloud'] > self.config.const_cropped_plant_cloud_percentage].index.to_list():
                row = pivot_df.iloc[index]
                # Deleting the folder
                logger.debug(f"Deleting the folder {self.config.plant} for date: {row.date}")
                shutil.rmtree(WeekData[WeekData['date'] == row.date].iloc[0].Folder)
                pivot_df.at[index, 'is_deleted'] = 1
            return pivot_df

    @classmethod
    def download_S3(cls, config, shape_path, record_date_str):
        S3_title, zip_file_name = cls.S3download_creodias(config, shape_path, record_date_str)
        # Skip for empty product
        if S3_title == None:
            logger.info("S3 data not available on {}".format(record_date_str))

        is_success = False if zip_file_name is None else cls.unzipfile(zip_file_name)
        if not is_success:
            if zip_file_name is not None:
                logger.error("unzipping not done, corrupt zip file %s", zip_file_name)
                os.remove(zip_file_name)
        else:
            logger.info("product unzipped for %s", record_date_str)
            S3product_folder = os.path.join(config.get_image_dir(satellite=cls.satellite), "s3-" + record_date_str)
            extracted_path = os.path.join(config.get_image_dir(satellite=cls.satellite), S3_title)
            if not os.path.exists(S3product_folder):
                os.rename(extracted_path, S3product_folder)
            else:
                copy_tree(extracted_path, S3product_folder)
            os.remove(zip_file_name)

    @staticmethod
    def get_tile_cloud(meta_file_path):
        if os.path.exists(meta_file_path):
            try:
                with open(meta_file_path, 'r') as f:
                    e = parseString(f.read())
                    parent_attr = 'grid'
                    parent_attr_value = '1 km'
                    search_tag = 'sentinel3:cloudyPixels'
                    search_attr = 'percentage'
                    result = list(filter(lambda x: x[1]== parent_attr_value, map(lambda x: (x.getAttribute(search_attr), x.parentNode.getAttribute(parent_attr)), e.getElementsByTagName(search_tag))))
                    if len(result) > 0:
                        return float(result[0][0])
            except Exception as e:
                logger.warning("Exception while extracting tile_cloud from %s Exception: %s", meta_file_path, str(e))
        return 0


    def main(self):
        #Initialize dates dataframe
        cloud_df = None
        week_data = []
        date_count = 0

        fusion_shape_path = self.shape_dict[Identifier.FusionShape]
        # Input correct EPSG values throughout
        # Initialize the date
        record_date = self.start_date - timedelta(1)
        s2_dates = self.get_s2_dates(self.config, self.config.const_cropped_scene_cloud_percentage)
        logger.info("Starting DownloadSentinelS3 for %s", self.config.plant)
        plant_shape = self.shape_dict[Identifier.FullPlant]
        while record_date < self.end_date:
            record_date += timedelta(days=1)
            date_count += 1
            record_date_str = record_date.strftime(self.config.FILE_DATE_FORMAT)
            logger.info("record date is : %s", record_date_str)
            S3product_folder = os.path.join(self.config.get_image_dir(satellite='S3'), "s3-" + record_date_str)
            if not os.path.exists(S3product_folder):
                record_date_str = record_date.strftime('%Y-%m-%d')
                self.download_S3(self.config, plant_shape, record_date_str)

            if os.path.exists(S3product_folder):
                try:
                    #Create cloud image
                    logger.info("Generating Sentinel3 cloud for %s img of %s", self.config.plant, record_date_str)
                    cloud_image = os.path.join(S3product_folder, 'S3cloud_' + record_date_str + '.tif')
                    self.convertS3_cloud(S3product_folder, cloud_image)

                    logger.info("Converting Sentinel3 cloud file to binary for %s date %s", self.config.plant, record_date_str)

                    cloud_binary_file = os.path.join(S3product_folder, 'S3cloud_binary_' + record_date_str + '.tif')
                    self.cloud_to_binary(cloud_image, cloud_binary_file)
                    tile_cloud = self.get_tile_cloud(os.path.join(S3product_folder, 'xfdumanifest.xml'))
                    # Clip to fusion shape and Resample cloud file
                    cloud_fusion_shape_500m = os.path.join(S3product_folder, 'S3_fusion_shape_' + record_date_str + '.tif')
                    cut.crop_image(cloud_binary_file, cloud_fusion_shape_500m, fusion_shape_path)
                    logger.info("Resampling the file %s", cloud_fusion_shape_500m)
                    cloud_resmapled_20m = os.path.join(S3product_folder, 'S3_resampled_' + record_date_str + '.tif')
                    self.resample(cloud_fusion_shape_500m, cloud_resmapled_20m, 20, 20)

                    for shape in self.shape_dict.keys():
                        shape_path = self.shape_dict.get(shape)
                        is_mask = (shape_path is not None) and os.path.exists(shape_path)
                        with rasterio.open(cloud_resmapled_20m) as src:
                            if is_mask:
                                geoms = shp.read_shape_file(shape_path, crs=src.crs).geometry
                            cropped_img, _ =  mask(src, geoms, crop=True, filled=False) if is_mask else (src.read(), src.transform)

                        cropped_good_pixels = np.count_nonzero(cropped_img==100)
                        cropped_cloud_pixels = np.count_nonzero(cropped_img==200)
                        total_pixels = cropped_good_pixels + cropped_cloud_pixels
                        cropped_good_pixels_per = cropped_good_pixels*100/max(1, total_pixels)
                        cropped_cloud_pixels_per = 100 if total_pixels == 0 else cropped_cloud_pixels*100/max(1, total_pixels)
                        if shape == Identifier.FusionShape:
                            logger.info("fusion area good pixel percentage for plant %s on %s is: %.3f", self.config.plant, record_date_str, cropped_good_pixels_per)
                        elif shape == Identifier.FullPlant:
                            logger.info("plant area good pixel percentage for plant %s on %s is %.3f totalpixels: %d", self.config.plant, record_date_str, cropped_good_pixels_per, total_pixels)
                        week_data.append({'date': record_date_str, 'shape': shape, 'Folder': S3product_folder, 'cloudcoverage': cropped_cloud_pixels_per, 'totalpixels': total_pixels, 'tile_cloud': tile_cloud})
                    os.remove(cloud_resmapled_20m)
                except Exception as e:
                    logger.error("Exception in s3_download_crop for plant %s date %s, Exception: %s trace: %s", self.config.plant, record_date_str, str(e), traceback.format_exc())
                self.remove_files(S3product_folder)
            logger.debug("count of days is : %d", date_count)

            if date_count==8:
                df = self.process_weekly_data(week_data, s2_dates)
                cloud_df = df if cloud_df is None else pd.concat([cloud_df, df])
                logger.info("Reset WeekData and date_count to 0")
                week_data = []
                date_count=0

        if len(week_data) > 0:
            df = self.process_weekly_data(week_data, s2_dates)
            cloud_df = df if cloud_df is None else pd.concat([cloud_df, df])
        if cloud_df is not None:
            cloud_df.sort_values(by=['date'], inplace=True)
            logger.debug("Final cloud data for %s with %d rows", self.config.plant, cloud_df.shape[0])
            if "db" in self.config.sinks:
                save_cloud_data(self.config, cloud_df, satellite='S3')
            else:
                cloud_file_path = os.path.join(self.config.out_dir, 'cloud_data_S3_creodias.csv')
                logger.info("Saving clouddata file at %s", cloud_file_path)
                cloud_df.to_csv(cloud_file_path, mode='a', index=False, header=not os.path.exists(cloud_file_path))
        logger.info("Completed DownloadSentinelS3 for %s", self.config.plant)
        
    @staticmethod
    def remove_files(folder_path):
        preserved_files = set(list(map(lambda _file: os.path.join(folder_path, _file), ['flags_an.nc', 'geodetic_an.nc', 'S3_radiance_an.nc', 'S6_radiance_an.nc', 'xfdumanifest.xml'])))
        files_to_remove = set(list(map(lambda _file: os.path.join(folder_path, _file), os.listdir(folder_path)))) - preserved_files

        for file_ in files_to_remove:
            os.remove(file_)


if __name__ == '__main__':
    args = utils.parse_config_params()
    config = utils.create_config(args, satellite='S3')
    instance = DownloadSentinelS3(config)
    instance.main()