# 1. AWS KEY SETUP
# 2. /anaconda3/envs/sentinel/lib/python<>/site-packages/sentinelhub/config.json
# "instance_id": "f600d465-c324-4f84-9da0-8889353edfbe"
# 3. https://portal.creodias.eu/register.php Create an account
# 4. Setup the LINE 41 and LINE 42 creodias_user creodias_password or either set appropriate values in the environment download_s3.py
# python main_modify.py handan --start-date 2021-08-29 --end-date 2021-09-02
import os
from datetime import timedelta
from download_s2 import DownloadSentinelS2
from download_s3 import DownloadSentinelS3
import utils

args = utils.parse_config_params()
config = utils.create_config(args, satellite='S3')

if __name__ == '__main__':
    instance = DownloadSentinelS2(config=config)
    plant_shape_path = "" # KML path of plant
    instance.download_s2(config, plant_shape_path)

    instance = DownloadSentinelS3(config=config)
    record_date = config.start_date - timedelta(1)
    while record_date < config.end_date:
        record_date += timedelta(days=1)
        record_date_str = record_date.strftime(config.FILE_DATE_FORMAT)
        print("record date is : %s", record_date_str)
        S3product_folder = os.path.join(config.get_image_dir(satellite='S3'), "s3-" + record_date_str)
        if not os.path.exists(S3product_folder):
            record_date_str = record_date.strftime('%Y-%m-%d')
            instance.download_S3(config, plant_shape_path, record_date_str)