#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:24:01 2020

@author: gunjanthakuria
"""

import glob
import re
import os
import traceback
from datetime import datetime

import pandas as pd
import numpy as np
import numpy.ma as ma
import rasterio

from lxml import etree
from rasterio.mask import mask

import logger
import utils
import shape as shp
from config import Identifier
from data import save_cloud_data

logger = logger.get_logger(__file__)

# NODATA, CLOUD_MEDIUM, CLOUD_HIGH and THIN_CIRRUS levels added
obj_dict_cloud_pixels = dict(map(lambda p: (p, 1), [0, 8, 9, 10]))

class SentinelStats():

    def __init__(self, config):
        self.config = config
        self.shape_dict = shp.get_shape_dict(self.config.shape_dir)
        self.shape_file_path =  self.shape_dict[Identifier.CoregistrationShape]
        self.pivoted_stats_path = os.path.join(self.config.common_dir, self.config.plant + '_pivoted_stats_with_cloud_info.csv')
    
    @classmethod
    def create_cloud_stats(cls, cloud_image_path, shape_path=None):
        try:
            is_mask = (shape_path is not None) and os.path.exists(shape_path)
            with rasterio.open(cloud_image_path) as src:
                if is_mask:
                    geoms = shp.read_shape_file(shape_path, crs=src.crs).geometry
                cloud_image, _ = mask(src, geoms, crop=True, filled=False) if is_mask else (src.read(), src.transform)
            uni, cnt = np.unique(cloud_image, return_counts=True)
            obj_dict_qa_pixels = dict(zip(uni[uni.mask==False], cnt[uni.mask==False]))

            total_pixels, cloud_pixels = 0, 0
            for key, value in obj_dict_qa_pixels.items():
                if obj_dict_cloud_pixels.get(key, -1) != -1:
                    cloud_pixels = cloud_pixels + value
                total_pixels = total_pixels + value
        except Exception as e:
            logger.warn("Exception %s for path %s shape %s setting cloudcoverage to 100", str(e), os.path.basename(cloud_image_path), os.path.basename(shape_path) if is_mask else '')
            cloud_pixels, total_pixels = 0, 0
        return cloud_pixels, total_pixels

    @classmethod
    def create_band(cls, config, shape_file_path, record_date_str, tile_name):
        B12_8A_path = None
        if(len(config.allowed_tiles) == 0 or tile_name in config.allowed_tiles):
            base_folder = os.path.join(config.get_image_dir(satellite='S2'), f's2-{tile_name}-'+ record_date_str)
            file_ = glob.glob(os.path.join(base_folder, 'IMG_DATA', '*B12*.jp2'))
            try:
                with rasterio.open(file_[0]) as src:
                    geoms = shp.read_shape_file(shape_file_path, crs=src.crs).geometry
                    band12_cut, _ = mask(src, geoms, crop=True)

                band12_arr = band12_cut.astype('float64')
                B12_nodata = np.ma.masked_array(band12_arr, mask=(band12_arr == 0))

                B8A = glob.glob(os.path.join(os.path.dirname(file_[0]), '*B8A*.jp2'))
                with rasterio.open(B8A[0]) as src:
                    geoms = shp.read_shape_file(shape_file_path, crs=src.crs).geometry
                    band8A_cut, B8A_transform = mask(src, geoms, crop=True)
                band8A_arr = band8A_cut.astype('float64')
                B8A_nodata = np.ma.masked_array(band8A_arr, mask=(band8A_arr == 0))

                out_meta = src.meta.copy()

                B12_8A = (B12_nodata/B8A_nodata)
                B12_8A_path = os.path.join(base_folder, 'S2_coreg_shape_20m_B12_8A_' + record_date_str + '.tif')

                out_meta.update({"driver": "GTiff",
                                "height": B12_8A.shape[1],
                                "width": B12_8A.shape[2],
                                "dtype":"float64",
                                "transform": B8A_transform})
                with rasterio.open(B12_8A_path, "w", **out_meta) as dest:
                    dest.write(B12_8A)
                logger.info("Data written at %s for %s", B12_8A_path, record_date_str)
            except Exception as e:
                logger.error("Error in create_band for %s Exception: %s",record_date_str, traceback.format_exc())
        return B12_8A_path

    def calculate_day_stat(self, config, shape_file_path, record_date_str, tile_name):
        statistics = []
        base_folder = os.path.join(config.get_image_dir(satellite='S2'), f's2-{tile_name}-'+ record_date_str)
        try:
            B12_8A_path = self.create_band(config, shape_file_path, record_date_str, tile_name)

            meta_data = glob.glob(os.path.join(base_folder, '*.xml'))
            if len(meta_data) > 0:
                logger.debug("meta data is : %s", meta_data)
                tree = etree.parse(meta_data[0])
                nss = tree.getroot().nsmap
            tile_cloud = float(tree.xpath("/n1:Level-1C_Tile_ID/n1:Quality_Indicators_Info/Image_Content_QI/CLOUDY_PIXEL_PERCENTAGE/text()", namespaces=nss)[0])
            slc = glob.glob(os.path.join(base_folder, 'IMG_DATA', '*_SCL*.jp2'))
            for shape in self.shape_dict.keys():
                try:
                    shape_path = self.shape_dict.get(shape)
                    is_mask = (shape_path is not None) and os.path.exists(shape_path)
                    # Getting cloud info
                    if len(slc) > 0:
                        cloud_pixels, total_pixels = self.create_cloud_stats(slc[0], shape_path)
                    else:
                        cloud_pixels, total_pixels = tile_cloud, 100
                    # Getting band ratio info
                    with rasterio.open(B12_8A_path) as src:
                        if is_mask:
                            geoms = shp.read_shape_file(shape_path, crs=src.crs).geometry
                        B12_8A, _ = mask(src, geoms, crop=True, filled=False) if is_mask else (src.read(), src.transform)
                    B12_8A_mean, B12_8A_max, B12_8A_min, B12_8A_std = B12_8A.mean(), B12_8A.max(), B12_8A.min(), B12_8A.std()
                    cloud_coverage = float(cloud_pixels*100)/total_pixels if total_pixels > 0 else 100.0
                    row_content = {'plant': config.plant, 'date': record_date_str, 'B12_8A_mean': B12_8A_mean,
                                    'B12_8A_max': B12_8A_max, 'B12_8A_min': B12_8A_min, 'B12_8A_std': B12_8A_std,
                                    'shape': shape, 'cloudcoverage': cloud_coverage, 'totalpixels': total_pixels, 'tile_cloud': tile_cloud, 'tilename': tile_name}
                    statistics.append(row_content)
                except:
                    logger.error("Error for shape %s and date record %s", shape, record_date_str)
        except Exception as e:
            logger.error("Exception occurred in calculate_day_stat for date %s Exception: %s", record_date_str, traceback.format_exc())
        return statistics

    def calculate_stats(self):
        statistics=[]
        file_list = glob.glob(os.path.join(self.config.get_image_dir(satellite='S2'), '*', 'IMG_DATA', '*B12*.jp2'))
        if len(self.config.allowed_tiles) > 0:
            file_list = [f for f in file_list for tile in self.config.allowed_tiles if tile in f]
        for file_ in sorted(file_list):
            date_str = re.search('.*(\d{8}T\d{6}).*', file_)
            if date_str:
                tile_name = os.path.basename(file_).split('_')[0][1:]
                if(tile_name in self.config.allowed_tiles or len(self.config.allowed_tiles) == 0):
                    logdate = datetime.strptime(date_str.group(1), '%Y%m%dT%H%M%S').date()
                    if(logdate >= self.config.start_date and logdate <= self.config.end_date):
                        record_date_str = logdate.strftime('%Y-%m-%d')
                        stats = self.calculate_day_stat(self.config, self.shape_file_path, record_date_str, tile_name)
                        if len(stats) > 0:
                            statistics += stats
        return statistics

    def save_dataframe(self, df):
        if df is not None and df.empty is False:
            out_file = os.path.join(self.config.out_dir, 'cloud_data_S2_creodias.csv')
            df.drop_duplicates(['plant','date', 'tilename','shape'], keep='last', inplace=True)
            df.sort_values(by='date', inplace=True)
            if "file" in self.config.sinks:
                columns = ['plant', 'date','shape','cloudcoverage', 'totalpixels']
                df.to_csv(out_file, mode='a', index=False, header=not os.path.exists(out_file), columns=columns)
                logger.info("Data written at %s", out_file)
                if os.path.exists(out_file):
                    df_temp = pd.read_csv(out_file)
                    df = df[df_temp.plant==self.config.plant]
                    df.drop_duplicates(['plant','date','shape'], keep='last', inplace=True)
            tile_df = df[['date','tilename','tile_cloud']].drop_duplicates()
            tile_df.set_index(['date', 'tilename'], inplace=True)
            index_df = df.pivot_table(columns='shape', values='B12_8A_mean', index=['date', 'tilename'])
            cloud_df = df.pivot_table(columns='shape', values='cloudcoverage', index=['date', 'tilename'])
            df_ = pd.merge(index_df, cloud_df, how='left', left_index=True, right_index=True, suffixes=('_B12_8A_mean','_cloud'))
            pixel_df = df.pivot_table(columns='shape', values='totalpixels', index=['date', 'tilename'])
            pixel_df = pixel_df.add_suffix('_totalpixels')
            df_ = pd.merge(df_, pixel_df, how='left', left_index=True, right_index=True)
            pivot_df = pd.merge(df_, tile_df, how='left', left_index=True, right_index=True).reset_index()
            pivot_df.fillna(0, inplace=True)
            pivot_df.sort_values(by=['date', Identifier.FullPlant+'_totalpixels'], inplace=True)
            pivot_df.drop_duplicates(['date'], keep='last', inplace=True)
            if "file" in self.config.sinks:
                pivot_df.to_csv(self.pivoted_stats_path, index=False)
            if "db" in self.config.sinks:
                save_cloud_data(self.config, pivot_df, satellite='S2')
            logger.info("Saved calculated stats at %s and %s", out_file, self.pivoted_stats_path)

    def main(self):
        statistics = self.calculate_stats()
        df = pd.DataFrame.from_records(statistics)
        self.save_dataframe(df)
        logger.info("Sentinal stats completed")


if __name__ == '__main__':
    args = utils.parse_config_params()
    config = utils.create_config(args, satellite='S2')
    instance = SentinelStats(config)
    instance.main()