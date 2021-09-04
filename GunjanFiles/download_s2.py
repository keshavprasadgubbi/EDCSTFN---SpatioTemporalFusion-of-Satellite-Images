#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:11:11 2020

@author: gunjanthakuria
"""
import datetime
import glob
import os
import traceback
import shutil
from sentinelhub import WebFeatureService, BBox, CRS, DataSource
from sentinelhub.config import SHConfig
from sentinelhub import AwsTileRequest

from distutils import dir_util
import logger 
import shape as shp
import utils
from config import Identifier

logger = logger.get_logger(__file__)

class DownloadSentinelS2():
    satellite = 'S2'
    def __init__(self, config):
        self.config = config
        self.shape_dict = shp.get_shape_dict(self.config.shape_dir)

    @classmethod
    def download_s2(cls, config, shape_path, start_date=None, end_date=None):
        logger.info('Reading shape from path %s', shape_path)
        shape = shp.read_shape_file(shape_path)
        aoi_bbox = dict(shape.bounds.loc[0])
        search_bbox = BBox(bbox=[aoi_bbox['minx'], aoi_bbox['miny'], aoi_bbox['maxx'], aoi_bbox['maxy']], crs=CRS.WGS84)
        if start_date is not None and end_date is not None:
            search_time_interval = (start_date.strftime('%Y-%m-%dT%H:%M:%S'), end_date.strftime('%Y-%m-%d') +'T23:59:59')
        else:
            search_time_interval = (config.start_date.strftime('%Y-%m-%dT%H:%M:%S'), config.end_date.strftime('%Y-%m-%d') +'T23:59:59')
        logger.info('Data download for interval between %s to %s' % search_time_interval)
        sh_config = SHConfig()
        wfs_iterator = WebFeatureService(
            search_bbox,
            search_time_interval,
            data_source=DataSource.SENTINEL2_L1C,
            maxcc=1.0,
            config=sh_config
        )
        data_dir = config.get_image_dir(satellite=cls.satellite)
        for (tile_name, date, aws) in wfs_iterator.get_tiles():
            if tile_name in config.allowed_tiles or len(config.allowed_tiles) == 0:
                tile_date = datetime.datetime(*list(map(int, date.split('-'))))
                formatted_date = tile_date.strftime('%Y-%m-%d')
                logger.info('Downloading tile %s for %s', tile_name, date)
                try:
                    tile_requests = []
                    try:
                        tile_request = AwsTileRequest(tile=tile_name, time=date, aws_index=aws,
                                                    bands=['R20m/SCL'],
                                                    data_folder=data_dir, data_source=DataSource.SENTINEL2_L2A,
                                                    safe_format=True)
                        tile_requests.append(tile_request)
                    except:
                        logger.warning("Download of SCL layer failed for %s", formatted_date)
                    logger.info("Trying download of L1C product for tile %s date %s", tile_name, formatted_date)
                    tile_request = AwsTileRequest(tile=tile_name, time=date, aws_index=aws,
                                                bands=['B12','B8A'],
                                                metafiles=['metadata'],
                                                data_folder=data_dir, data_source=DataSource.SENTINEL2_L1C,
                                                safe_format=True)
                    tile_requests.append(tile_request)
                except Exception as e:
                    logger.warning("Error while download for %s %s", formatted_date, str(e))

                out_directories = map(lambda req: os.path.join(data_dir, (req.get_filename_list()[0]).split(os.path.sep)[0]), tile_requests)
                renamed_directory = os.path.join(data_dir, f's2-{tile_name}-' + tile_date.strftime('%Y-%m-%d'))
                try:
                    for index, out_directory in enumerate(out_directories):
                        if not os.path.exists(out_directory):
                            tile_requests[index].save_data()
                        # TODO check for the intersection of plant shape with the tiled data
                        if not os.path.exists(renamed_directory) :
                            os.rename(out_directory, renamed_directory)
                            logger.info("Saving data at %s", renamed_directory)
                        else:
                            dir_util._path_created = {} # https://stackoverflow.com/a/28055993/4465743
                            dir_util.copy_tree(out_directory, renamed_directory)
                            shutil.rmtree(out_directory)
                    if os.path.exists(renamed_directory):
                        for res_folder in glob.glob(os.path.join(renamed_directory, 'IMG_DATA','R*')):
                            dir_util._path_created = {}
                            dir_util.copy_tree(res_folder, os.path.dirname(res_folder))
                            shutil.rmtree(res_folder)
                except Exception as e:
                    logger.error('Error in downloading tile %s for %s exception: %s, trace: %s', tile_name, date, str(e), traceback.format_exc())
                finally:
                    if os.path.exists(out_directory):
                        shutil.rmtree(out_directory)
        logger.info('Downloading completed. Files downloaded at path %s', data_dir)

    def main(self):
        shape_path = self.shape_dict.get(Identifier.FullPlant, None)
        self.download_s2(self.config, shape_path)

if __name__ == '__main__':
    args = utils.parse_config_params()
    config = utils.create_config(args, satellite='S2')
    instance = DownloadSentinelS2(config)
    instance.main()