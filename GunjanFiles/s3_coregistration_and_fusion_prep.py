#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:18:08 2020

@author: gunjanthakuria
"""
import copy
import glob
import json
import os
import sys
import time
import traceback
from datetime import datetime

import shape as shp

import pandas as pd
import numpy as np
try:
    import osr
    import gdal
except ImportError:
    from osgeo import osr, gdal

from arosics import COREG_LOCAL, DESHIFTER
from netCDF4 import Dataset
from download_s3 import DownloadSentinelS3
import cut
import data
import logger
import utils
from config import Identifier


creodias_codes='creodias-finder'
sys.path.append(creodias_codes)

creodias_user = os.environ.get('CREODIAS_USER')
creodias_password = os.environ.get('CREODIAS_PASSWORD')

logger = logger.get_logger(__file__)

class DataPreprationS3():

    def __init__(self, config):
        self.config = config
        self.start_date = self.config.start_date
        self.shape_dict = shp.get_shape_dict(self.config.shape_dir)
        self.create_dirs(self.config)

    @staticmethod
    def convertS3(path, out_file_path, epsg=None):
        logger.debug("Processing image for path %s", path)
        with  Dataset("{}/S6_radiance_an.nc".format(path)) as S6_rad_nc:
            rad_s6 = S6_rad_nc.variables["S6_radiance_an"][:,:]
        with  Dataset("{}/S3_radiance_an.nc".format(path)) as S3_rad_nc:
            rad_s3 = S3_rad_nc.variables["S3_radiance_an"][:,:]
        with  Dataset("{}/flags_an.nc".format(path)) as flags_nc:
            cloud = flags_nc.variables["cloud_an"][:,:]
        with Dataset("{}/geodetic_an.nc".format(path)) as geod_nc:
            lat = geod_nc.variables["latitude_an"][:]
            lon = geod_nc.variables["longitude_an"][:]
            nx = geod_nc.dimensions['columns'].size
            ny = geod_nc.dimensions['rows'].size

        # Latitude and longitude bounds
        xmin, ymin, xmax, ymax = [lon.min(), lat.min(), lon.max(), lat.max()]

        xres = (xmax - xmin) / float(nx)
        yres = (ymax - ymin) / float(ny)
        geotransform = (xmin, xres, 0, ymax, 0, -yres)
        
        # Masking cloud pixels
        cld_mask = np.where(cloud==0,1,0) 
        rad_s6_masked = cld_mask*rad_s6
        rad_s3_masked = cld_mask*rad_s3
        logger.debug("rad_s6_masked min: %d", rad_s6_masked.min())
        logger.debug("rad_s3_masked min: %d", rad_s3_masked.min())

        #Convert mask to 0
        rad_s6_unmask = np.where(rad_s6_masked.mask==1,0,rad_s6_masked.data)
        rad_s3_unmask = np.where(rad_s3_masked.mask==1,0,rad_s3_masked.data)
        logger.debug(" rad_s6_unmask min: %d", rad_s6_unmask.min())
        logger.debug(" rad_s3_unmask min: %d", rad_s3_unmask.min())

        #removing negative 
        
        rad_s6_nonnegative=np.where(rad_s6_unmask<0, 0, rad_s6_unmask)
        rad_s3_nonnegative=np.where(rad_s3_unmask<0, 0, rad_s3_unmask)

        logger.debug("rad_s6_masked_nonnegative min: %d", rad_s6_nonnegative.min())
        logger.debug("rad_s3_masked_nonnegative min: %d", rad_s3_nonnegative.min())

        
        #Compute band ratio
        band_ratio = np.where(rad_s3_nonnegative==0, 0, rad_s6_nonnegative/rad_s3_nonnegative)
        logger.info("shape of original band: %s" , str(band_ratio.shape))   

        # Create tif image 
        dst_ds = gdal.GetDriverByName('GTiff').Create('output_img.tif', nx, ny, 1, gdal.GDT_Float64)
        dst_ds.SetGeoTransform(geotransform)    # specify coordinates
        srs = osr.SpatialReference()            # establish encoding
        srs.ImportFromEPSG(4326)                # WGS84 lat/long
        dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file

        # write band value to the raster 
        dst_ds.GetRasterBand(1) .WriteArray(band_ratio)
        dst_ds.GetRasterBand(1) .SetNoDataValue(0)
        
        # Clip the image according to the outputBounds, resample it to 20m resolution and reproject to UTM
        # Calculating region EPSG
        target_epsg = 'EPSG:' + str(epsg)
        output = gdal.Warp(out_file_path, dst_ds, xRes=500, yRes=-500, format='GTiff', srcSRS='EPSG:4326', dstSRS=target_epsg)
        dst_ds = None
        output = None

    @classmethod
    def get_common_date(cls, config):
        s3_cloud_file_path = os.path.join(config.out_dir, 'cloud_data_S3_creodias.csv')
        s2_cloud_file_path = os.path.join(config.out_dir, 'cloud_data_S2_creodias.csv')

        if "db" in config.sinks:
            s2_cloud_df = data.get_cloud_data(config, satellite='S2', approved_only=True)
            s3_cloud_df = data.get_cloud_data(config, satellite='S3', approved_only=True)
        elif os.path.exists(s3_cloud_file_path) and os.path.exists(s2_cloud_file_path):
            # Use data from filesystem
            s3_cloud_df = pd.read_csv(s3_cloud_file_path)
            s3_cloud_df.rename(columns={Identifier.CoregistrationShape + '_cloud': 'aoi_cloud', Identifier.FullPlant + '_cloud': 'plant_cloud'}, inplace=True)
            s2_cloud_df = pd.read_csv(s2_cloud_file_path)
            s2_cloud_df.rename(columns={Identifier.CoregistrationShape + '_cloud': 'aoi_cloud', Identifier.FullPlant + '_cloud': 'plant_cloud'}, inplace=True)

        # Check if dataframes exist and not empty
        if not (s2_cloud_df.empty or s3_cloud_df.empty):
            # Update logic to get common dates
            s2_cloud_df.drop_duplicates(['date'], keep='last', inplace=True)
            s2_cloud_df.sort_values(by=['plant_cloud', 'aoi_cloud'], ascending=(True, True), inplace=True)
            s3_cloud_df.drop_duplicates(['date'], keep='last', inplace=True)
            s3_cloud_df.sort_values(by=['plant_cloud', 'aoi_cloud'], ascending=(True, True), inplace=True)

            dates_df = pd.merge(s2_cloud_df['date'], s3_cloud_df['date'], on='date')
            if dates_df.empty:
                return None
            else:
                return dates_df['date'].iloc[0]
        return None

    @staticmethod
    def create_dirs(config):
        s3_coreg_path = os.path.join(config.out_dir, "s3-train")
        s2_coreg_path = os.path.join(config.out_dir, "s2-train")
        s2_pred_path = os.path.join(config.out_dir, "s2-pred")
        for path in [s2_coreg_path, s3_coreg_path, s2_pred_path]:
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def resampled_coregistered(filename, out_file_path):
        output = gdal.Warp(out_file_path, filename, format='GTiff', xRes=20, yRes=-20, resampleAlg='cubic') # outbounds to match the S3 image
        output = None

    @classmethod
    def coregister(cls, reffile, targetfile, coreg=None, align=False):
        start_time = time.time()
        if coreg is None:
            logger.info('Starting local coregistration for ref: %s and target: %s', reffile, targetfile)
            im_reference =  reffile
            im_target    =  targetfile
            kwargs = {
                # Standardize the path_out
                'path_out'     : targetfile[:-4] + '__shifted_to__' + os.path.basename(reffile),
                'match_gsd'    : align,
                'grid_res'     : 20,
                'window_size'  : (20,20),
                'q'            : False,
                'fmt_out'      : 'GTIFF',
                'projectDir'   : os.path.dirname(targetfile)
            }
            logger.info('Kwargs: %s', json.dumps(kwargs))
            CRL = COREG_LOCAL(im_reference, im_target,**kwargs)
            CRL.correct_shifts()
            end_time = time.time()
            logger.info('Finished local coregistration in %.2f seconds for target: %s and ref: %s', end_time-start_time, targetfile, reffile)
            return CRL
        else:
            logger.info('Applying local coregistration shift for target: %s and ref: %s', targetfile, reffile)
            # Standardize the path_out
            DESHIFTER(targetfile, coreg.coreg_info, path_out=targetfile[:-4] + '__shifted_to__' + os.path.basename(reffile), fmt_out='GTIFF', align_grids=align, match_gsd=align).correct_shifts()
            end_time = time.time()
            logger.info('Finished Applying local coregistration shift in %.2f seconds for target: %s and ref: %s', end_time-start_time, targetfile, reffile)
            return coreg

    @classmethod
    def process_s3(cls, config, shape_path, record_date_str, epsg=None):
        logger.info("Processing record for date_start is : %s ", record_date_str)
        S3product_folder = os.path.join(config.get_image_dir(satellite='S3'), 's3-' + record_date_str)
        if os.path.exists(S3product_folder):
            logger.info("Generating Sentinel3 band-ratio tif img of {}".format(record_date_str))
            converted_file_path = os.path.join(S3product_folder, 'S3_{}.tif'.format(record_date_str))
            cls.convertS3(S3product_folder, converted_file_path, epsg=epsg)
            logger.info("Created Sentinel3 band ratio tif image of %s", record_date_str)

            logger.info("Clipping Converted Sentinel3 img to coreg shape  of %s", record_date_str)
            cropped_image_path = os.path.join(S3product_folder, 'S3_coreg_shape_' + record_date_str + '.tif')
            cut.crop_image(converted_file_path, cropped_image_path, shape_path)
            logger.info("Coreg cropped image at path: %s", cropped_image_path)
            logger.debug("deleting original band ratio file")
            os.remove(converted_file_path)

            logger.debug("Resampling coreg shape file to 20m resolution")
            resampled_file_path = os.path.join(S3product_folder , 'S3_coreg_shape_20m_{}.tif'.format(record_date_str))
            cls.resampled_coregistered(cropped_image_path, resampled_file_path)
            logger.debug("Resampled image at path : %s", resampled_file_path)
            logger.debug("Deleting previous file after resampling")
            os.remove(cropped_image_path)
        else:
            logger.info("folder not present for %s date %s", config.plant, record_date_str)

    @classmethod
    def coregister_image(cls, config, record_date_str, ref_date_str, CRL=None):
        config.CRL = CRL or config.CRL
        file_list = glob.glob(os.path.join(config.get_image_dir(satellite='S2'), 's2-*' + ref_date_str, 'S2_coreg_shape_20m_B12_8A_{}.tif'.format(ref_date_str)))
        if len(config.allowed_tiles) > 0:
            file_list = [f for f in file_list for tile in config.allowed_tiles if tile in f]
        ref_file = file_list[0]
        target_file = os.path.join(config.get_image_dir(satellite='S3'), 's3-' + ref_date_str, 'S3_coreg_shape_20m_{}.tif'.format(ref_date_str))
        if config.CRL is None:
            logger.info("Starting co-registration process for %s", record_date_str)
            config.CRL = cls.coregister(ref_file, target_file)
        coregister_file = list(filter(lambda x: '__shifted_to__' not in x, glob.glob(os.path.join(config.get_image_dir(satellite='S3'), 's3-' + record_date_str, 'S3_coreg_shape_20m_*.tif'))))[0]
        if isinstance(config.CRL, COREG_LOCAL):
            config.CRL = cls.coregister(ref_file, coregister_file, coreg=config.CRL, align=True)
        logger.info("Co-registration completed for %s", coregister_file)
        return config.CRL

    @classmethod
    def prepare_s3(cls, config, shape_path, record_date_str):
        s3_coreg_path = os.path.join(config.out_dir, "s3-train")
        coregistered_image_path = glob.glob(os.path.join(config.get_image_dir(satellite='S3'), 's3-' + record_date_str, 'S3_coreg_shape_20m_*__shifted_to__*.tif'))
        # Cropping image for fusion
        if len(coregistered_image_path) > 0:
            coregistered_image_path = coregistered_image_path[0]
            logger.info("clipping file date %s to fusion shape", record_date_str)
            coregistered_cropped_path = os.path.join(s3_coreg_path, 'S3_coreg_fusion_shape_{}.tif'.format(record_date_str))
            cut.crop_image(coregistered_image_path, coregistered_cropped_path, shape_path)
            logger.debug("deleting renamed co-registered file")
            os.remove(coregistered_image_path)

    @classmethod
    def prepare_s2(cls, config, shape_path, record_date_str):
        s2_coreg_path = os.path.join(config.out_dir, "s2-train")
        file_list = glob.glob(os.path.join(config.get_image_dir(satellite='S2'), 's2-*' + record_date_str, 'S2_coreg_shape_20m_B12_8A_{}.tif'.format(record_date_str)))
        if len(config.allowed_tiles) > 0:
            file_list = [f for f in file_list for tile in config.allowed_tiles if tile in f]
        s2_image_path = file_list[0]
        tile_name = s2_image_path.split('/')[-2].split('-')[1]
        coregistered_cropped_path_s2 = os.path.join(s2_coreg_path, f'S2_coreg_fusion_shape_{tile_name}_{record_date_str}.tif')
        cut.crop_image(s2_image_path, coregistered_cropped_path_s2, shape_path)

    def data_prepration_s2(self):
        logger.info("Moving S2 files with plant_cloud < %f after cropping for %s", self.config.const_cropped_scene_cloud_percentage, self.config.plant)
        fusion_shape = self.shape_dict[Identifier.FusionShape]
        s2_dates = DownloadSentinelS3.get_s2_dates(self.config, self.config.const_cropped_scene_cloud_percentage)
        for record_date_str in s2_dates:
            self.prepare_s2(self.config, fusion_shape, record_date_str)
    
    
    def check_common_files(self, date):
        s2_common_date_files = glob.glob(os.path.join(self.config.get_image_dir(satellite='S2'), "s2-*" + date, f'S2_coreg_shape_20m_B12_8A_{date}.tif'))
        if len(self.config.allowed_tiles) > 0:
            s2_common_date_files = [f for f in s2_common_date_files for tile in self.config.allowed_tiles if tile in f]
        s3_common_date_files = glob.glob(os.path.join(self.config.get_image_dir(satellite='S3'), 's3-*' + date, 'S3_coreg_shape_20m_{}.tif'.format(date)))
        logger.info("Check common files for %s date %s is S2: %s S3: %s", self.config.plant, date, str(s2_common_date_files), str(s3_common_date_files))
        return bool(s2_common_date_files) and bool(s3_common_date_files)


    def main(self):
        logger.info("Starting S3 data processing")
        coreg_shape = self.shape_dict[Identifier.CoregistrationShape]
        fusion_shape = self.shape_dict[Identifier.FusionShape]
        logger.info("Starting S3 coregistration")
        common_date = self.get_common_date(self.config)
        logger.info("Common date selected for coregistration for plant %s is %s", self.config.plant, common_date)
        if common_date is not None:
            file_list = glob.glob(os.path.join(self.config.get_image_dir(satellite='S2'), 's2-*' + common_date, f'S2_coreg_shape_20m_B12_8A_{common_date}.tif'))
            if len(self.config.allowed_tiles) > 0:
                file_list = [f for f in file_list for tile in self.config.allowed_tiles if tile in f]
            ref_file = file_list[0]
            epsg = utils.get_epsg_from_file(ref_file)
            logger.info("Selected EPSG: %d value from file %s", epsg, ref_file)
            for s3product_folder in glob.glob(os.path.join(self.config.get_image_dir(satellite='S3'), 's3-*')):
                record_date_str = os.path.basename(s3product_folder)[-10:]
                record_date = datetime.strptime(record_date_str, self.config.FILE_DATE_FORMAT).date()
                if record_date > self.config.end_date:
                    continue
                try:
                    self.process_s3(self.config, coreg_shape, record_date_str, epsg=epsg)
                except Exception as e:
                    logger.error("Error in s3 band ratio image processing for %s exception: %s trace: %s", record_date_str, str(e), traceback.format_exc())
            CRL = None
            for resampled_file_path in list(filter(lambda x: '__shifted_to__' not in x, glob.glob(os.path.join(self.config.get_image_dir(satellite='S3'), 's3-*','S3_coreg_shape_20m_*.tif')))):
                record_date_str = os.path.dirname(resampled_file_path)[-10:]
                try:
                    CRL = self.coregister_image(self.config, record_date_str, common_date, CRL=CRL)
                    self.prepare_s3(self.config, fusion_shape, record_date_str)
                except Exception as e:
                    logger.error("Error in coregistration for %s exception: %s trace: %s", record_date_str, str(e), traceback.format_exc())
            logger.info("Completed coregistration")
        else:
            logger.critical("Common date is not available so coregistration can't be proceeded")
        self.data_prepration_s2()
        logger.info("Completed all data prepration steps")

if __name__ == '__main__':
    args = utils.parse_config_params()
    config = utils.create_config(args, satellite='S3')
    instance = DataPreprationS3(config)
    instance.main()