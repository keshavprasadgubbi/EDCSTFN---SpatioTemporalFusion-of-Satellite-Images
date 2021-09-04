import argparse
import glob
import json
import os
import requests
import boto3
import numpy as np
import urllib.request
from urllib.parse import quote, urlparse
import rasterio


from datetime import datetime, timedelta, date
from config import Config


def get_quarter(_date):
    """
    Parameter:
        date: dateetime value
    Returns:
        Financial quarter of the date
    """
    return ((_date.month-1)//3 + 3)%4 + 1


def get_quarter_from_month(month):
    """
    Parameter:
        month: Integer value of month
    Returns:
        Financial quarter of the month
    """
    return ((month-1)//3 + 3)%4 + 1


def get_date(year, month,  isfirst=True, format='%d-%m-%Y'):
    date_ = date(year, month, 1) if isfirst is True else date(year, month+1, 1) - timedelta(1)
    return date_.strftime(format)


def get_matching_filenames(identifier, path, suffix=".TIF",):
    """
    Returns: List of files in `path` containing identifier and suffix
    """
    filenames = glob.glob(os.path.join(path, "*" + suffix))
    return list(filter(lambda filename: identifier.lower() in filename.lower(), filenames))


def download(file_url, out_path=None):
    """
    Downloads the file at file_url and saves at out_path
    if no out_path given file is saved at current directory with filename as quoted url
    """
    out_path = out_path if out_path is not None else os.path.join(os.getcwd(), quote(file_url, ''))
    req = requests.get(file_url, stream = True, allow_redirects=True)
    if req.status_code == 200:
        with open(out_path,"wb") as bound:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    bound.write(chunk)
        return True
    raise Exception("File couldn't be downloaded. URL %s returned status code %d".format(file_url, req.status_code))


def calculate_epsg(lat, lon):
    return 32700 - round((45 + lat)/90.0)*100 + round((183 + lon)/6.0)

def get_epsg_from_file(file_path):
    if os.path.isfile(file_path):
        with rasterio.open(file_path) as src:
            return int(str(src.crs)[5:])
    return None



def parse_config_params():
    parser = argparse.ArgumentParser(description='Pass parameter for processing')
    parser.add_argument('plant', type=str,help='plant name')
    parser.add_argument('--start-date', '--start_date', type=lambda i: datetime.strptime(i, '%Y-%m-%d').date(), help='provide start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', '--end_date', type=lambda i: datetime.strptime(i, '%Y-%m-%d').date(), help='provide end date in YYYY-MM-DD format')
    parser.add_argument('--user', type=str, default='', help='provide user for user directory creation')
    parser.add_argument('--api_pwd', '--api-pwd', '--api-password', type=str, help='provide api password')
    parser.add_argument('--api_user', '--api-user', '--api-username', type=str, help='provide api username')
    parser.add_argument('--sinks', type=str, nargs='+', default=['S3'], help='provide the sinks e.g S3 DB')
    parser.add_argument('--allowed-tiles','--allowed_tiles', dest='allowed_tiles', type=str, nargs='+', default=[], help='allowed tile paths list for data resource')
    parser.add_argument('--lst', action='count', default=0, help='mention to perform lst')
    parser.add_argument('--sentinel2', action='count', default=0, help='mention to only use sentinel 2')
    parser.add_argument('--fusion', action='count', default=0, help='mention to perform fusion')
    parser.add_argument('--all', action='count', default=0, help='mention to perform all functions (lst sentinel with fusion)')
    parser.add_argument('--enable-shutdown', '--enable_shutdown', action='count', default=0, help='mention to enable shutdown')
    parser.add_argument('--enable-delete', '--enable_delete', action='count', default=0, help='mention to enable file deletion')
    parser.add_argument('--request_id', type=str, help='provide request id')
    parser.add_argument('--config-url', '--config_url', type=str, help='provide config url')
    parser.add_argument('--release', action='store_true', default=0, help='mention to enable release mode 0 works in debug' )
    return parser.parse_args()

def create_config(args, satellite='LANDSAT'):
    config = Config(plant=args.plant, satellite=satellite, user_identifier=args.user)
    if args.start_date is not None:
        config.start_date = args.start_date
    if args.end_date is not None:
        config.end_date = args.end_date
    if args.api_user is not None and args.api_pwd is not None:
        config.set_api_credentials(args.api_user, args.api_pwd)
    if args.sinks is not None:
        config.sinks = list(map(lambda x: x.lower(), args.sinks))
    if args.allowed_tiles is not None:
        config.allowed_tiles = args.allowed_tiles
    if args.config_url is not None:
        # Based on following ULR construct https://tathyaconfig.s3.amazonaws.com/staging_config.json
        url_components = urlparse(args.config_url)
        bucket_name = url_components.netloc.split('.s3.amazonaws')[0]
        key_name = url_components.path[1:]
        s3 = boto3.resource('s3')
        obj = s3.Object(bucket_name, key_name)
        config.set_conf(json.loads(obj.get()['Body'].read()))
    return config


def get_current_instance_info():
    try:
        instance_id = urllib.request.urlopen("http://169.254.169.254/latest/meta-data/instance-id/").read().decode('utf-8')
        region = urllib.request.urlopen("http://169.254.169.254/latest/meta-data/placement/availability-zone/").read().decode('utf-8')
        return get_instance_info(instance_id, region[:-1])
    except:
        return {'instance_id': '', 'instance_ip': '', 'instance_name': ''}


def get_instance_info(instance_id, region):
    try:
        ec2 = boto3.resource('ec2', region_name=region)
        instance = ec2.Instance(instance_id)
        name_tag = list(filter(lambda x: x.get('Key', None)=='Name', instance.tags))
        instance_name = '' if len(name_tag)==0 else name_tag[0]['Value']
        return {'instance_id': instance_id, 'instance_ip': instance.public_ip_address, 'instance_name': instance_name}
    except:
        return {'instance_id': '', 'instance_ip': '', 'instance_name': ''}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NpEncoder, self).default(obj)