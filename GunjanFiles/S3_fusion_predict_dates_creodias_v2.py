#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:02:10 2020

@author: gunjanthakuria
"""


from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from sentinelhub import WebFeatureService, BBox, CRS, DataSource, SHConfig
from sentinelhub import SentinelHubRequest, SentinelHubDownloadClient, DataSource, MimeType, DownloadRequest,CRS, BBox
from sentinelhub import AwsProductRequest, AwsTile, AwsTileRequest
from osgeo import osr
from netCDF4 import Dataset
from zipfile import ZipFile
import numpy as np
import rasterio
from rasterio.mask import mask
import os
from datetime import date, timedelta, datetime
import pandas as pd
import shutil
from osgeo import gdal
import sys
import warnings
import glob
import traceback

if not sys.warnoptions:
    warnings.simplefilter("ignore")

creodias_codes='creodias-finder'
sys.path.append(creodias_codes)

from creodias_finder import query
from creodias_finder import download
creodias_user='keshav.quantumboy@gmail.com'
creodias_password= 'Sangeetha2021'

import cut
import logger 
import shape as shp
import utils
from config import Identifier

logger = logger.get_logger(__file__)

class DownloadSentinelS3():
    """
    Cloud info gathered after resampling
    """
    def __init__(self, config):
        self.config = config
        self.start_date = self.config.start_date
        self.end_date = self.config.end_date
        self.shape_dict = shp.get_shape_dict(self.config.shape_dir)

    def S3download_creodias(self, date):
        try:
            # specify the search polygon
            footprint = shp.read_shape_file(self.shape_dict[Identifier.FullPlant]).geometry.loc[0]

            result = query.query('Sentinel3',
                                start_date=date.strftime('%Y-%m-%d'),
                                end_date=date.strftime('%Y-%m-%d'),
                                geometry=footprint,
                                productType='RBT',
                                orbitDirection='descending')
            if bool(result)==False:
                print ("data not available")
                return None, None
            ids = [key['id'] for key in result.values()]
            out = os.path.join(self.config.image_dir, date.strftime('%Y-%m-%d-') + 'S3.zip')
            download.download(ids[0], creodias_user, creodias_password, outfile=out, show_progress=True)
            logger.info("Download Completed for %s", date.strftime('%Y-%m-%d'))
            # Title of product
            sen3title = result[list(result.keys())[0]]['properties']['title']
            return sen3title, out
        except:
            print("Oops!", traceback.format_exc(), "occurred.")
            return None,None

    def unzipfile(self, zip_file_name):
        try:
            with ZipFile(zip_file_name, "r") as zip:
                logger.debug('Extracting Sentinel3 files %s', zip_file_name)
                zip.extractall(path=os.path.dirname(zip_file_name))
                return True
        except:
            logger.error("Oops! Exception occurred.", traceback.format_exc())
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
        # Calculating region EPSG
        target_epsg = 'EPSG:' + str(self.get_region_epsg())
        output = gdal.Warp(out_image_path, cld_ds, format='GTiff', srcSRS='EPSG:4326', dstSRS=target_epsg, resampleAlg='near')
        cld_ds = None
        output = None
        return True

    def resample(self, in_file, out_file, xres,yres):
        try:
            output = gdal.Warp(out_file, in_file, format='GTiff', xRes=xres, yRes=yres, resampleAlg='near')
            output = None
        except Exception as e:
            logger.error("Exception in resample: %s traceback: %s", str(e), traceback.format_exc())

    def get_s2_dates(self):
        cloud_file_path = os.path.join(self.config.out_dir, 'cloud_data_S2_creodias.csv')
        if os.path.exists(cloud_file_path):
            s2_dates = pd.read_csv(cloud_file_path)['logdate'].to_list()
            return s2_dates
        else:
            logger.warning("S2 Cloud file missing won't be able to preserve s2 dates")

    def main(self):

        #Initialize dates dataframe
        CloudData = pd.DataFrame(columns=['Date','Folder','InputArea-Cloud','PlantArea-Cloud'])
        WeekData = CloudData.copy()
        week_data = []
        date_count = 0
        image_count = 0

        fusion_shape_path = self.shape_dict[Identifier.FullPlant]
        plant_shape_path = self.shape_dict[Identifier.FullPlant]


        # Input correct EPSG values throughout
        # Initialize the date
        record_date = self.start_date
        month = record_date.month
        day = record_date.day
        s2_dates = self.get_s2_dates()
        logger.info("Starting DownloadSentinelS3")
        while record_date <= self.end_date:
            date = record_date.strftime("%Y%m%d")
            record_date_str = record_date.strftime("%Y-%m-%d")
            logger.info("record date is : %s", record_date_str)
            S3_title, zip_file_name = self.S3download_creodias(record_date)
            
            # Skip for empty product
            if S3_title == None:
                logger.info("S3 data not availbale on {}".format(record_date))
                record_date += timedelta(days=1)
                # date_count += 1 # NOT SURE
                continue

            is_success = self.unzipfile(zip_file_name)
            if not is_success:
                logger.error("unzipping not done, corrupt zip file %s", zip_file_name)
                record_date += timedelta(days=1)
                os.remove(zip_file_name)
                date_count += 1
                continue
            else:
                logger.info("product unzipped")
                extracted_path = os.path.join(self.config.image_dir, S3_title)
                S3product_folder = os.path.join(self.config.image_dir, "s3-" + record_date_str)
                os.rename(extracted_path, S3product_folder)
                os.remove(zip_file_name)
            
                #Create cloud image
                logger.info("Generating Sentinel3 cloud tif img of {}".format(record_date))
                cloud_image = os.path.join(S3product_folder, 'S3cloud_' + record_date_str + '.tif')
                self.convertS3_cloud(S3product_folder, cloud_image)
            
                logger.info("Converting Sentinel 3 cloud file to binary for date {}".format(record_date_str))
                
                try:
                    # Clip to fusion shape and Resample cloud file 
                    cloud_cropped_shape_500m = os.path.join(S3product_folder, 'S3_fusion_shape_' + record_date_str + '.tif')
                    cut.crop_image(cloud_image, cloud_cropped_shape_500m, fusion_shape_path)

                    logger.info("Resampling the file %s", cloud_cropped_shape_500m)
                    cloud_resmapled_20m = os.path.join(S3product_folder, 'S3_resampled_' + record_date_str + '.tif')
                    self.resample(cloud_cropped_shape_500m, cloud_resmapled_20m, 20, 20)
                    logger.debug("Resampled file created:", os.path.exists(cloud_resmapled_20m))
                
                    # Calculate cloud % of fusion shape
                    logger.debug("calculate cloud % of fushion shape")

                    with rasterio.open(cloud_resmapled_20m) as src:
                        geoms = shp.read_shape_file(fusion_shape_path, crs=src.crs).geometry
                        fusion_img, _ = mask(src, geoms, crop=True, filled=False)

                    logger.info(np.unique(fusion_img, return_counts=True))
                    fusion_good_pixels = np.count_nonzero(fusion_img==0)
                    fusion_cloud_pixels = np.count_nonzero(fusion_img>0)
                    total_pixels = fusion_good_pixels + fusion_cloud_pixels
                    fusionarea_goodpixel_perc = fusion_good_pixels*100/total_pixels
                    fusionarea_cloudpixel_perc = fusion_cloud_pixels*100/total_pixels
                    logger.info("fusion area good pixel percentage on %s is: %.3f", record_date_str, fusionarea_goodpixel_perc)
            
                    #Calculate cloud % of plant area
            
                    with rasterio.open(cloud_resmapled_20m) as src:
                        geoms = shp.read_shape_file(plant_shape_path, crs=src.crs).geometry
                        plant_img, _ = mask(src, geoms, crop=True, filled=False)
                    logger.info(np.unique(plant_img, return_counts=True))
                    plant_good_pixels = np.count_nonzero(plant_img==0)
                    plant_cloud_pixels = np.count_nonzero(plant_img>0)
                    total_pixels = plant_good_pixels + plant_cloud_pixels
                    plantarea_goodpixel_perc = plant_good_pixels*100/total_pixels
                    plantarea_cloudpixel_perc = plant_cloud_pixels*100/total_pixels
            
                    logger.info("plant area good pixel percentage on %s is %.3f totalpixels: %d", record_date_str, plantarea_goodpixel_perc, total_pixels)
                except Exception as e:
                    logger.error(str(e))
                    fusionarea_goodpixel_perc = 0
                    plantrea_goodpixel_perc = 0
                # if fusionarea_goodpixel_perc > 65 and plantarea_goodpixel_perc > 90:
                week_data.append({'logdate':record_date_str, 'Folder': S3product_folder, 'InputArea-Cloud': fusionarea_cloudpixel_perc, 'plant_cloud_pixels': plantarea_cloudpixel_perc, 'is_S2': int(record_date_str in s2_dates)})

                self.remove_files(S3product_folder)
                record_date += timedelta(days=1)
                month = record_date.month
                year = record_date.year
                day = record_date.day
                date_count += 1
                logger.debug("count of days is : %d", date_count)
            
                if date_count==8:
                    if len(week_data) > 0:
                        WeekData = pd.DataFrame.from_records(week_data)
                        WeekData.sort_values(by=['is_S2','plant_cloud_pixels'], ascending=(False, True), inplace=True)
                        WeekData.reset_index(drop=True, inplace=True)
                        WeekData.loc[:, 'is_deleted'] = 0
                        if len(WeekData) > 3:
                            for row in WeekData[WeekData.index[3:]].itertuples():
                                shutil.rmdir(row.Folder)
                            WeekData.at[row.Index,'is_deleted'] = 1
                        CloudData = CloudData.append(WeekData)
                        logger.info("Clouddata appended:", CloudData)
                        WeekData.drop(WeekData.index, inplace=True)
                    logger.info("Reset WeekData and date_count to 0")
                    week_data = []
                    date_count=0
        if not WeekData.empty:
            CloudData = CloudData.append(WeekData)
        CloudData.drop(columns=['Folder'], inplace=True)    
        CloudData.sort_values(by=['Date'], inplace=True)
        logger.debug("Final cloud data with %d rows", CloudData.shape[0])
        cloud_file_path = os.path.join(self.config.out_dir, 'cloud_data_S3_creodias.csv')
        logger.info("Saving clouddata file at %s", cloud_file_path)
        CloudData.to_csv(cloud_file_path, mode='a', index=False, header=not os.path.exists(cloud_file_path))
        logger.info("Completed DownloadSentinelS3")
    
    @staticmethod
    def remove_files(folder_path):
        preserved_files = set(list(map(lambda _file: os.path.join(folder_path, _file), ['mapflags_an.nc', 'geodetic_an.nc', 'S3_radiance_an.nc', 'S6_radiance_an.nc'])))
        files_to_remove = set(os.listdir(folder_path)) - preserved_files 
        for _file in files_to_remove:
            os.remove(_file)


if __name__ == '__main__':
    args = utils.parse_config_params()
    config = utils.create_config(args, satellite='S3')
    instance = DownloadSentinelS3(config)
    instance.main()