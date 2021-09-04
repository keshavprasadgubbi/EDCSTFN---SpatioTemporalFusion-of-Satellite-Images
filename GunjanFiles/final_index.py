import glob
import logging
import os

import shape as shp

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

from config import Identifier
import data
import logger
import utils


logger = logger.get_logger(__file__, logging.INFO)


class FinalIndex():


    def __init__(self, config):
        self.config = config
        self.source = 'SENTINEL'
        self.shape_file_dict = shp.get_shape_dict(self.config.shape_dir)
        self.final_index_path = os.path.join(self.config.out_dir, 'sentinel_final_index.csv')
        self.final_pivoted_index_path = os.path.join(self.config.out_dir, self.config.plant + '_sentinel_final_pivoted_index.csv')
        self.fine_train_path = os.path.join(self.config.out_dir, "s2-train")
        self.fine_pred_path = os.path.join(self.config.out_dir, "s2-pred")

    def calculate_index(self, file_path):
        day_lst_data = []
        tile_name = ''
        file_name = os.path.basename(file_path).split('_')
        dirname = os.path.dirname(file_path)
        folder_name = os.path.basename(dirname)
        if folder_name == 's2-train':
            tile_name = file_name[4]
        with rasterio.open(file_path, 'r') as src:
            for shape in set(self.shape_file_dict.keys()) - set([Identifier.CoregistrationShape, Identifier.FusionShape]):
                try:
                    shape_path = self.shape_file_dict.get(shape)
                    if shape_path is not None:
                        geoms = shp.read_shape_file(self.shape_file_dict[shape], crs=src.crs).geometry
                        cropped_image, _ = mask(src, geoms, crop=True, filled=False)
                        uni, cnt = np.unique(cropped_image, return_counts=True)
                        obj_dict_pixels = dict(zip(uni[uni.mask==False], cnt[uni.mask==False]))
                        cloud_coverage = obj_dict_pixels.get(0, 0)*100/sum(obj_dict_pixels.values())
                        day_lst_data.append({'plant': self.config.plant,
                            'date': os.path.basename(file_path)[:10] if self.fine_pred_path in file_path else os.path.basename(file_path)[-14:-4],
                            'predicted': int(self.fine_pred_path in file_path),
                            'shape': shape,
                            'indexmean': cropped_image.mean(),
                            'indexmax': cropped_image.max(),
                            'indexmin': cropped_image.min(),
                            'indexstd': cropped_image.std(),
                            'cloudcoverage': cloud_coverage,
                            'tilename': tile_name
                        })
                    else:
                        logger.warning("No file present for shape %s", shape)
                except Exception as e:
                    logger.error("Error in calculate_index for shape %s", shape)
        return day_lst_data

    def save_dataframe(self, df):
        if df is not None and df.empty is False:
            df.drop_duplicates(['plant','date','shape','predicted','tilename'], keep='last', inplace=True)
            df.sort_values('date', inplace=True)
            df.to_csv(self.final_index_path, mode='a', index=False, header=not os.path.exists(self.final_index_path))
            if 'file' in self.config.sinks:
                df = pd.read_csv(self.final_index_path)
                df = df[df.plant==self.config.plant]
                df.drop_duplicates(['plant','date','shape','predicted'], keep='last', inplace=True)
            index_df = df.pivot_table(columns='shape', values='indexmean', index=['date', 'predicted', 'tilename'])
            cloud_df = df.pivot_table(columns='shape', values='cloudcoverage', index=['date', 'predicted', 'tilename'])
            pivot_df = pd.merge(index_df, cloud_df, how='left', left_index=True, right_index=True, suffixes=('_mean','_cloud')).reset_index().rename_axis(None, axis=1)
            pivot_df = pivot_df[~pivot_df[Identifier.FullPlant + '_cloud'].isna()].sort_values(['date', 'predicted', Identifier.FullPlant + '_cloud'])
            pivot_df.fillna(0, inplace=True)
            pivot_df.drop_duplicates(['date', 'predicted'], keep='first', inplace=True)
            pivot_df.to_csv(self.final_pivoted_index_path, index=False)
            if 'db' in self.config.sinks:
                pivot_df.loc[:,'source'] = pivot_df['predicted'].apply(lambda x: 'S3' if x else 'S2')
                data.save_raw_data(self.config, pivot_df)
        else:
            logger.info("Dataframe is empty in save_dataframe.")

    def main(self):
        lst_data = []
        logger.info("Started calculating index for path %s and %s", self.fine_train_path, self.fine_pred_path)
        for lst_file in glob.glob(self.fine_train_path + '/*.tif') + glob.glob(self.fine_pred_path + '/*.tif'):
            logger.info("Calculating index for file %s", os.path.basename(lst_file))
            day_lst_data = self.calculate_index(lst_file)
            lst_data += day_lst_data
        df = pd.DataFrame.from_records(lst_data)
        self.save_dataframe(df)
        logger.info("Saved calculated lst at %s and %s", self.final_index_path, self.final_pivoted_index_path)



if __name__ == '__main__':
    args = utils.parse_config_params()
    config = utils.create_config(args)
    instance = FinalIndex(config)
    instance.main()

