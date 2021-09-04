import copy
import glob
import os
import shutil
import logger
from datetime import datetime
from pathlib import Path
from s3_coregistration_and_fusion_prep import DataPreprationS3
from fusion_automate import FusionAutomate


logger = logger.get_logger(__file__)

def common_approved_dates(config):
    # Gets the common approved dates for S2 and S3
    fusion_automate_obj = FusionAutomate(config)
    s2_dates_df = fusion_automate_obj.get_cloud_free_dates(config=config, satellite='S2')
    s3_dates_df = fusion_automate_obj.get_cloud_free_dates(config=config, satellite='S3')
    common_dates = list(set(s3_dates_df).intersection(set(s2_dates_df)))
    common_dates.sort()
    common_dates = list(map(lambda x: x.strftime(config.FILE_DATE_FORMAT), common_dates))
    return common_dates

def clear_files(config, retain_files_count=3):
    s3_coreg_path = os.path.join(config.out_dir, "s3-train")
    s2_coreg_path = os.path.join(config.out_dir, "s2-train")
    s2_pred_path = os.path.join(config.out_dir, "s2-pred")
    # Get list of latest S2 and S3 common approved dates
    # make a find pairs like function with sorted dates (cloudfree dates)
    common_date_pairs = common_approved_dates(config)
    
    preserved_dates = common_date_pairs[-1*retain_files_count:]
    # Get S3 date pairs
    # This is done to pick the file for correg
    common_date = DataPreprationS3.get_common_date(config)
    if common_date is not None:
        logger.info("Common date for plant %s is %s", config.plant, common_date)
        preserved_dates.append(common_date)
    logger.info("Preserved dates for plant %s is %s", config.plant, str(preserved_dates))

    # Remove the files for extra dates
    for file_ in glob.glob(os.path.join(s2_coreg_path, 'S2_coreg_fusion_shape_*.tif')):
        record_date = Path(file_).stem[-10:]
        if record_date not in preserved_dates:
            os.remove(file_)
    for file_ in glob.glob(os.path.join(s3_coreg_path, 'S3_coreg_fusion_shape_*.tif')):
        record_date = Path(file_).stem[-10:]
        if record_date not in preserved_dates:
            os.remove(file_)
    for dir_ in glob.glob(os.path.join(config.get_image_dir(satellite='S3'), '*')) + glob.glob(os.path.join(config.get_image_dir(satellite='S2'), '*')):
        record_date = Path(dir_).stem[-10:]
        if record_date not in preserved_dates:
            if os.path.isdir(dir_):
                shutil.rmtree(dir_)
            else:
                os.remove(dir_)

    if os.path.exists(s2_pred_path):
        shutil.rmtree(s2_pred_path)
    logger.info("File clearing completed for plant %s", config.plant)