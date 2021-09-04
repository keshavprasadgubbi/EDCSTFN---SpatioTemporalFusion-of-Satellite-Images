import os

import shape as shp

import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.mask import mask

def crop_image(image_path, out_image_path, shape_path=None):
    if os.path.exists(image_path):
        with rasterio.open(image_path) as src:
            geoms = shp.read_shape_file(shape_path, crs=src.crs).geometry if shape_path else None
            src_data, src_transform = mask(src, geoms, nodata=0, crop=True) if geoms is not None else (src.read(), src.transform)
            out_meta_data = src.meta.copy()
            out_meta_data.update({"driver": "GTiff",
                "height": src_data.shape[1],
                "width": src_data.shape[2],
                "transform": src_transform,
                "dtype": str(src_data.dtype)
            })
            with rasterio.open(out_image_path,'w', **out_meta_data) as dest:
                dest.write(src_data)
    else:
        raise Exception('No file exists at image_path ' + image_path)

def crop_shape(image_path, out_shape_path=None, filter=None):
    """
    Returns: Geodataframe for filtered area in image
    
    Parameters:
    image_path: path of TIF image to be processed
    out_shape_path: path of shape file for filtered area; Supported format(geojson, shp, kml) [optional]
    filter: function for applying filter on the image [optional]
            If no filter provided only area with nodata values is extracted.
    """
    if os.path.exists(image_path):
        with rasterio.open(image_path) as src:
            src_data = src.read(1).astype('float32') # float64 not allowed in feature.shapes
            if not callable(filter):
                filter = lambda x: x==0
            mask = filter(src_data)
            src_data.fill(1)
            shapes = features.shapes(src_data, mask=mask, transform=src.transform)
            geoms = list({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(shapes))
            gpd_polygonized_raster  = gpd.GeoDataFrame.from_features(geoms)
            gpd_polygonized_raster.crs = src.crs
            gpd_polygonized_raster.to_crs('epsg:4326')
            if out_shape_path is not None:
                if os.path.isfile(out_shape_path):
                    os.remove(out_shape_path)
                if out_shape_path.endswith('.kml'):
                    gpd_polygonized_raster.to_file(out_shape_path, driver='KML')
                elif out_shape_path.endswith('.geojson'):
                    gpd_polygonized_raster.to_file(out_shape_path, driver='GeoJSON')
                else:
                    gpd_polygonized_raster.to_file(out_shape_path)
            return gpd_polygonized_raster
    raise Exception('No file exists at image_path %s'.format(image_path))