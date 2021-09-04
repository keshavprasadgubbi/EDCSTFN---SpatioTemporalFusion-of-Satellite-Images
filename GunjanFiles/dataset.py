import copy
import os
from datetime import datetime
import shape as shp
import utils, glob
import logging

from config import Identifier
from download_s2 import DownloadSentinelS2
from s2_stats import SentinelStats
from download_s3 import DownloadSentinelS3
from s3_coregistration_and_fusion_prep import DataPreprationS3

def prepare_data_for_date(config, record_dates_str):
    logging.info(f"Preparing data for date: {record_dates_str}")
    config.satellite = 'S3'
    shape_dict = shp.get_shape_dict(config.shape_dir)
    plant_shape = shape_dict[Identifier.FullPlant]
    coreg_shape = shape_dict[Identifier.CoregistrationShape]
    fusion_shape = shape_dict[Identifier.FusionShape]
    if record_dates_str is None:
        return None
    elif isinstance(record_dates_str, str):
        record_dates_str = [record_dates_str]
    for record_date_str in record_dates_str:
        record_date = datetime.strptime(record_date_str, '%Y-%m-%d')
        DownloadSentinelS2.download_s2(config, plant_shape, start_date=record_date, end_date=record_date)
        folder_list = glob.glob(os.path.join(config.get_image_dir(satellite='S2'), "s2-*" + record_date_str))
        if config.allowed_tiles:
            folder_list = [f for f in folder_list for tile in config.allowed_tiles if tile in f]
        if len(folder_list) > 0:
            s2_folder = folder_list[0]
            tile_name = os.path.basename(s2_folder).split('-')[1]
            SentinelStats.create_band(config, coreg_shape, record_date_str, tile_name)
        DownloadSentinelS3.download_S3(config, plant_shape, record_date_str)
        S3product_folder = os.path.join(config.get_image_dir(satellite='S3'), "s3-" + record_date_str)
        if os.path.exists(S3product_folder):
            DownloadSentinelS3.remove_files(S3product_folder)
    common_date = DataPreprationS3.get_common_date(config)
    file_list = glob.glob(os.path.join(config.get_image_dir(satellite='S2'), 's2-*' + common_date, f'S2_coreg_shape_20m_B12_8A_{common_date}.tif'))
    if len(config.allowed_tiles) > 0:
        file_list = [f for f in file_list for tile in config.allowed_tiles if tile in f]
    epsg = utils.get_epsg_from_file(file_list[0])

    for record_date_str in record_dates_str:
        DataPreprationS3.process_s3(config, coreg_shape, record_date_str, epsg=epsg)
        config.CRL = DataPreprationS3.coregister_image(config, record_date_str, common_date, CRL=config.CRL)
        DataPreprationS3.prepare_s3(config, fusion_shape, record_date_str)
        folder_list = glob.glob(os.path.join(config.get_image_dir(satellite='S2'), "s2-*" + record_date_str))
        if config.allowed_tiles:
            folder_list = [f for f in folder_list for tile in config.allowed_tiles if tile in f]
        if len(folder_list) > 0:
            DataPreprationS3.prepare_s2(config, fusion_shape, record_date_str)

if __name__ =='__main__':
    args = utils.parse_config_params()
    config = utils.create_config(args, satellite='S3')
    prepare_data_for_date(config, config.start_date.strftime('%Y-%m-%d'))