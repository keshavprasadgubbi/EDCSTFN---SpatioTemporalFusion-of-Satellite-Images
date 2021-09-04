import json
from datetime import datetime
from dateutil import tz

from database import DbConnection
import utils
import pandas as pd
from config import Identifier

def get_cloud_data(config, satellite=None, conn=None, approved_only=False):
    satellite = satellite or config.satellite
    if conn is None:
        conn = DbConnection(config.conf.get('dbconnection', {}))
    try:
        df = pd.DataFrame(columns=['date', 'tile_cloud', 'aoi_cloud', 'plant_cloud'])
        approved_clause = '' if approved_only is False else f"and {config.conf.get('cloud_table')}.approved='true'"
        df_ = pd.read_sql("select date, tile_cloud, aoi_cloud, plant_cloud from {cloud_table} join {plant_table} on {cloud_table}.plant_id = {plant_table}.id where {plant_table}.identifier='{plant_identifier}' and {cloud_table}.source='{source}' {approved_clause} order by date desc".format(plant_table=config.conf.get('plant_table'), cloud_table=config.conf.get('cloud_table'), plant_identifier=config.plant, source=satellite, approved_clause=approved_clause), conn.connection, chunksize=4)
        for i in df_:
            if df is None:
                df = i
            else:
                df = pd.concat([df, i])
        df.loc[:, 'date'] = df.date.apply(str)
        df.reset_index(drop=True, inplace=True)
    finally:
        if conn :
            conn.close()
    return df

def save_cloud_data(config, df, satellite=None, conn=None):
    satellite = satellite or config.satellite
    if conn is None:
        conn = DbConnection(config.conf.get('dbconnection', {}))
    try:
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            df.loc[ :,'source'] = satellite
            conn.execute("SELECT id, name, identifier from %s where identifier='%s';" % (config.conf.get('plant_table'), config.plant))
            plant = conn.cursor.fetchone()
            if plant is not None:
                plant_id = plant[0]
                df.loc[:, 'date_'] = df.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
                df = df[(df.date_ >= config.start_date)&(df.date_ <= config.end_date)]
                df.drop(columns=['date_'], axis=1, inplace=True)
                df.reset_index(drop=True, inplace=True)
                current_time = datetime.now(tz = tz.tzlocal())
                data = list(map(lambda i : (df.loc[i].source, df.loc[i].date, json.dumps(df.loc[i].to_dict(), cls=utils.NpEncoder), plant_id, df.loc[i].get('tile_cloud', 0), df.loc[i].get(Identifier.CoregistrationShape + '_cloud', 0), df.loc[i].get(Identifier.FullPlant + '_cloud', 0), str((df.loc[i].get('tile_cloud', 0) <= config.const_cropped_scene_cloud_percentage if df.loc[i].source=='S2' else df.loc[i].get(Identifier.CoregistrationShape + '_cloud', 0) <= config.const_cropped_scene_cloud_percentage) and df.loc[i].get(Identifier.FullPlant + '_cloud', 0) <= config.const_cropped_plant_cloud_percentage), current_time, current_time), df.index))
                conn.executemany("INSERT INTO {} (source, date, info, plant_id, tile_cloud, aoi_cloud, plant_cloud, approved, added_on, update_on) VALUES %s  ON CONFLICT (source, date, plant_id) DO UPDATE SET (tile_cloud, aoi_cloud, plant_cloud, info, update_on) = (EXCLUDED.tile_cloud, EXCLUDED.aoi_cloud, EXCLUDED.plant_cloud, EXCLUDED.info, EXCLUDED.update_on);".format(config.conf.get('cloud_table')), data)
    finally:
        if conn:
            conn.close()

def get_fusion_data(config, satellite=None, conn=None):
    satellite = satellite or config.satellite
    if conn is None:
        conn = DbConnection(config.conf.get('dbconnection', {}))
    try:
        df = pd.DataFrame(columns=['date', 'train_dates', 'processed'])
        df_ = pd.read_sql("select date, train_dates, processed from {fusion_table} join {plant_table} on {fusion_table}.plant_id = {plant_table}.id where {plant_table}.identifier='{plant_identifier}' and {fusion_table}.source='{source}' order by date desc".format(plant_table=config.conf.get('plant_table'), fusion_table=config.conf.get('fusion_table'), plant_identifier=config.plant, source=satellite), conn.connection, chunksize=4)
        for i in df_:
            if df is None:
                df = i
            else:
                df = pd.concat([df, i])
        df.loc[:, 'date'] = df.date.apply(str)
        df.loc[:, 'train_dates'] = df.train_dates.apply(lambda x: tuple(x))
        df.rename(columns={'date': 'test_date'}, inplace=True)
        df.reset_index(drop=True, inplace=True)
    finally:
        if conn:
            conn.close()
    return df

def save_fusion_data(config, df, satellite=None, conn=None):
    satellite = satellite or config.satellite
    if conn is None:
        conn = DbConnection(config.conf.get('dbconnection', {}))
    try:
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            conn.execute("SELECT id, name, identifier from %s where identifier='%s';" % (config.conf.get('plant_table'), config.plant))
            plant = conn.cursor.fetchone()
            if plant is not None:
                df.loc[ :,'source'] = satellite
                plant_id = plant[0]
                df.loc[:, 'date'] = df.test_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
                df = df[(df.date >= config.start_date)&(df.date <= config.end_date)]
                df.reset_index(drop=True, inplace=True)
                current_time = datetime.now(tz = tz.tzlocal())
                data = list(map(lambda i : (df.loc[i].source, df.loc[i].date, plant_id, list(df.loc[i]['train_dates']), df.loc[i]['processed'], current_time, current_time), df.index))
                conn.executemany("INSERT INTO {} (source, date, plant_id, train_dates, processed, added_on, update_on) VALUES %s ON CONFLICT (source, date, plant_id) DO UPDATE SET (train_dates, processed, update_on) = (EXCLUDED.train_dates, EXCLUDED.processed, EXCLUDED.update_on);".format(config.conf.get('fusion_table')), data)
    finally:
        if conn:
            conn.close()

def save_raw_data(config, df, conn=None):
    if conn is None:
        conn = DbConnection(config.conf.get('dbconnection', {}))
    try:
        if df is not None and isinstance(df, pd.DataFrame):
            conn.execute("SELECT id, name, identifier from %s where identifier='%s';" % (config.conf.get('plant_table'), config.plant))
            plant = conn.cursor.fetchone()
            if plant is not None:
                plant_id = plant[0]
                is_approved = True
                df.loc[:, 'date_'] = df.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
                df = df[(df.date_ >= config.start_date)&(df.date_ <= config.end_date)]
                df.drop(columns=['date_'], axis=1, inplace=True)
                current_time = datetime.now(tz = tz.tzlocal())
                data = list(map(lambda i : (df.loc[i].source, df.loc[i].date, json.dumps(df.loc[i].to_dict(), cls=utils.NpEncoder), plant_id, current_time, current_time, is_approved), df.index))
                conn.executemany("INSERT INTO {} (source, date, info, plant_id, added_on, update_on, approved) VALUES %s ON CONFLICT (source, date, plant_id) DO UPDATE SET (info, update_on, approved) = (excluded.info, excluded.update_on, excluded.approved);".format(config.conf.get('data_table')), data)
    finally:
        if conn:
            conn.close()        