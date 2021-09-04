import csv
import os

from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

def get_shape_dict(shape_dir, separator='_'):
    shapes = []
    for filepath in os.listdir(shape_dir):
        file_name = Path(filepath).stem
        shapes.append((file_name.split(separator)[-1], os.path.join(shape_dir, filepath)))
    return dict(shapes)


def read_shape_file(shape_file_path, crs=None):
    """
    Returns: 
    GeoDataframe crossponding to shape_file_path

    Parameters:
    shape_file_path: path of the shape file either geojson, kml or csv 
    crs: output GeoDataFrame projection default is epsg:4326 for csv
        and respective input shapefile projection
    """
    file_suffix = Path(shape_file_path).suffix
    df = None
    if file_suffix == '.csv':
        with open(shape_file_path, 'rt', encoding='utf-8') as f:
            data= filter(None, map(lambda x: None if x[1].isalpha() else (float(x[1]), float(x[0])), csv.reader(f)))
            polygon_geom = Polygon(data)
            crs_ = 'epsg:4326'
            df = gpd.GeoDataFrame(index=[0], crs=crs_, geometry=[polygon_geom]) 
    if file_suffix in ['.geojson', '.shp']:
        df = gpd.read_file(shape_file_path)
    elif file_suffix == '.kml':
        df = gpd.read_file(shape_file_path, driver='KML')
    if crs is not None and df is not None:
        df.to_crs(crs=crs, inplace=True)
    return df