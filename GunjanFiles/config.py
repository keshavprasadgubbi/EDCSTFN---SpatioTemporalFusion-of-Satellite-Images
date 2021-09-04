import glob
import os
import sys
import boto3

from datetime import datetime, date, timedelta, timezone


__all__ = ['BASE_IMAGE_DIR', 'BASE_HEATMAP_DIR', 'BASE_SHAPE_DIR', 'BASE_OUT_DIR', 'BASE_DATA_DIR', 'Config', 'Identifier']



_BASE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
# Out directory prioritized on EFS storage path if present
_PREFERRED_ROOT_PATH = "/data"
BASE_IMAGE_DIR = os.path.join(_BASE_DIR, "Volumes") if not os.path.exists(_PREFERRED_ROOT_PATH) else os.path.join(_PREFERRED_ROOT_PATH, "Volumes")
BASE_HEATMAP_DIR = os.path.join(_BASE_DIR, "Heatmap")
BASE_SHAPE_DIR = os.path.join(_BASE_DIR, "Shape") if not os.path.exists(_PREFERRED_ROOT_PATH) else os.path.join(_PREFERRED_ROOT_PATH, "Shape")
BASE_OUT_DIR = os.path.join(_BASE_DIR, "Output") if not os.path.exists(_PREFERRED_ROOT_PATH) else os.path.join(_PREFERRED_ROOT_PATH, "Output")
TEMP_DIR = os.path.join(_BASE_DIR, "tmp")


class Config():
    FILE_DATE_FORMAT = '%Y-%m-%d'

    def __init__(self, plant='', satellite='SENTINEL', user_identifier=''):
        """
        Parameters:
        plant: plant name to be used as identifier across project
        user_identifier: For user specific output and shape directory
        """
        self._user_identifier = user_identifier
        self._landsat_api_username = os.environ.get('SENTINEL_API_USERNAME', '')
        self._landsat_api_password = os.environ.get('SENTINEL_API_PASSWORD', '')
        self.plant = plant
        self.satellite = satellite
        self.start_date = date.today() - timedelta(1) # Set appropriate date for batch task date(year, month, day)
        self.end_date = datetime.today().date()
        self.common_dir = BASE_OUT_DIR  # For comman out data file which need to be access across projects
        self.heatmap_dir = os.path.join(BASE_HEATMAP_DIR, self.plant) # For storing heatmap image
        self.shape_dir = os.path.join(os.path.dirname(BASE_SHAPE_DIR), self._user_identifier, 'Shape', self.plant) # For storing shape files for a plant
        self.out_dir = os.path.join(os.path.dirname(BASE_OUT_DIR), self._user_identifier, 'Output', self.plant) # For storing output data specific to plant
        self.shape_list = [Identifier.FullPlant, Identifier.BlastFurnace, Identifier.NearPoint, Identifier.OuterPoint, Identifier.CoregistrationShape, Identifier.FusionShape]
        self.sinks = []
        self.allowed_tiles = []
        self.conf = None
        self.CRL = None
        self.const_cropped_scene_cloud_percentage = 35 # Duplicated at s3_download file
        self.const_cropped_plant_cloud_percentage = 10.0
        if plant is not None:
            self._verify_and_create_dir()

    def _verify_and_create_dir(self):
        for path in [self.image_dir, self.heatmap_dir, self.shape_dir, self.out_dir]:
            os.makedirs(path, exist_ok=True)

    def set_api_credentials(self, username, password):
        self._landsat_api_username = username
        self._landsat_api_password = password

    def _verify_shape_directory(self):
        client = boto3.client('s3')
        s3 = boto3.resource('s3')
        bucket_name = self.conf.get('shape_bucket','')
        data = client.list_objects(Bucket=bucket_name, Prefix=self.plant)
        objects = data['Contents']
        present_files = set(glob.glob(os.path.join(self.shape_dir,"*")))
        s3_files = []
        for obj in objects:
            local_obj_path = os.path.join(self.shape_dir, *obj['Key'].split('/')[1:])
            s3_files.append(local_obj_path)
            if not os.path.exists(local_obj_path):
                s3.meta.client.download_file(bucket_name, obj['Key'], local_obj_path)
                print("Downloaded missing shape file {}".format(local_obj_path), file=sys.stdout)
            else:
                local_file_modified_time = datetime.fromtimestamp(os.path.getmtime(local_obj_path), timezone.utc)
                s3_file_modified_time = obj['LastModified']
                if s3_file_modified_time > local_file_modified_time:
                    s3.meta.client.download_file(bucket_name, obj['Key'], local_obj_path)
                    print("Downloaded modified shape file {}".format(local_obj_path), file=sys.stdout)
        files_to_be_removed = present_files - set(s3_files)
        for file_path in files_to_be_removed:
            os.remove(file_path)

    def set_conf(self, config):
        self.conf = config
        self._verify_shape_directory()

    @property
    def landsat_api_username(self):
        return self._landsat_api_username

    @property
    def landsat_api_password(self):
        return self._landsat_api_password
    
    @property
    def image_dir(self):
        return os.path.join(BASE_IMAGE_DIR, self.satellite, self.plant) # For satellite images

    def get_image_dir(self, satellite=None):
        satellite = satellite or self.satellite
        return os.path.join(BASE_IMAGE_DIR, satellite, self.plant) # For satellite images


class Identifier():
    NearPoint = 'NP'
    BlastFurnace = 'BF'
    OuterPoint = 'FP'
    FullPlant = 'OP'
    CoregistrationShape = 'CS'
    FusionShape = 'FU'


if __name__ == '__main__':
    for dir_path in [BASE_IMAGE_DIR, BASE_HEATMAP_DIR, BASE_SHAPE_DIR, BASE_OUT_DIR, TEMP_DIR]:
        os.makedirs(dir_path, exist_ok=True)
        print('Creating Path', dir_path)