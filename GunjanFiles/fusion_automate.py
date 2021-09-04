import ast
import os, copy
import glob
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta

import pandas as pd

import alert
import data
import logger
import utils
from dataset import prepare_data_for_date
from s3_coregistration_and_fusion_prep import DataPreprationS3

logger = logger.get_logger(__file__)

class FusionAutomate():
    estarfm_parameter_file_path = "/home/ec2-user/cuESTARFM/Codes/parameters_naresh.txt"
    cuESTARFM_path = "/home/ec2-user/cuESTARFM/Codes/cuESTARFM"
    starfm_parameter_file_path = "/home/ec2-user/cuSTARFM/Codes/parameters_example.txt"
    cuSTARFM_path = "/home/ec2-user/cuSTARFM/Codes/cuSTARFM"
    closest_train_pair_count = 2
    max_gap_between_test_and_train_in_days = 120
    predict_fine_train_days = False # Flag for Predicting fine training days too
    is_startfm_enabled = os.path.isfile(cuSTARFM_path) # Override this flag to disable starfm
    
    def __init__(self, config):
        self.config = config
        
        self.coarse_train_path = os.path.join(self.config.out_dir, "s3-train")
        self.fine_train_path = os.path.join(self.config.out_dir, "s2-train")
        self.fine_pred_path = os.path.join(self.config.out_dir, "s2-pred")
        self.fusion_test_csv_path = os.path.join(self.config.out_dir, 'sentinel_fusion_test.csv')
        self.fusion_train_csv_path = os.path.join(self.config.out_dir, 'sentinel_fusion_train.csv')
        self.fusion_dict = {}
        self.ensure_directories()

    def ensure_directories(self):
        DataPreprationS3.create_dirs(self.config)

    def get_cloud_free_dates(self, config=None, satellite=None):
        # Checks only for a satellite either S2 or S3 using config, change this to use the arg provided
        cloud_free_dates = []
        # Override if provided
        config = config or self.config
        satellite = satellite or config.satellite
        cloud_file = 'cloud_data_S2_creodias.csv' if satellite == 'S2' else 'cloud_data_S3_creodias.csv'
        try:
            # Use csv file for both S2 and S3
            cloud_file_path = os.path.join(self.config.out_dir, cloud_file)
            if "db" in self.config.sinks:
                cloud_df = data.get_cloud_data(config, satellite=satellite, approved_only=True)
            elif os.path.exists(cloud_file_path):
                cloud_df = pd.read_csv(cloud_file_path)
                cloud_df = cloud_df[(cloud_df['tile_cloud'] <= self.config.const_cropped_scene_cloud_percentage)]

            cloud_df.drop_duplicates(['date'], keep='last', inplace=True)
            
            cloud_free_dates = list(map(lambda x: datetime.strptime(x, config.FILE_DATE_FORMAT), cloud_df.date.unique().tolist()))
        except Exception as e:
            logger.error("Error in get_cloud_free_dates for {} Exception: {}".format(config.plant ,str(e)), exc_info=sys.exc_info())

        return cloud_free_dates

    def create_fusion_dict(self):
        fine_train_dates = self.get_cloud_free_dates(satellite='S2')
        coarse_train_dates = self.get_cloud_free_dates(satellite='S3')
        # Getting list of common dates among the cloud free dates
        common_dates = list(set(fine_train_dates).intersection(set(coarse_train_dates)))
        common_dates.sort()
        coarse_dates = list(map(lambda x: datetime.strptime(os.path.basename(x)[-14:-4], self.config.FILE_DATE_FORMAT), glob.glob(os.path.join(self.coarse_train_path, "*.tif"))))

        coarse_test_dates =  coarse_dates if self.predict_fine_train_days else list(set(coarse_dates).difference(set(fine_train_dates)))
        for _date in coarse_test_dates:
            chosen_train_dates = []
            for candidate_train_date in common_dates:
                if _date > candidate_train_date:
                    # Saving training date only if either it is empty or gap between test and training is less than 60 days
                    if len(chosen_train_dates) >= self.closest_train_pair_count or (self.is_startfm_enabled and (len(chosen_train_dates) > 0 and (_date - chosen_train_dates[0] > timedelta(self.max_gap_between_test_and_train_in_days)))):
                        chosen_train_dates.pop(0)
                    if _date - candidate_train_date  <= timedelta(self.max_gap_between_test_and_train_in_days):
                        chosen_train_dates.append(candidate_train_date)
                else:
                    if candidate_train_date == _date:
                        # Same date not to be picked
                        continue 
                    # TODO: To be checked since only useful for historical data but not for future data
                    # Also useful for data past which historical data not considered
                    if (len(chosen_train_dates) < self.closest_train_pair_count or candidate_train_date - _date  < _date - chosen_train_dates[0]) and (not self.is_startfm_enabled or (candidate_train_date - _date <= timedelta(self.max_gap_between_test_and_train_in_days))):
                        if len(chosen_train_dates) >= self.closest_train_pair_count:
                            chosen_train_dates.pop(0)
                        if candidate_train_date - _date <= timedelta(self.max_gap_between_test_and_train_in_days):
                            chosen_train_dates.append(candidate_train_date)
            num_constraint = 0 if self.is_startfm_enabled else 1
            if len(chosen_train_dates) > num_constraint:
                train_pair = tuple(_date.strftime(self.config.FILE_DATE_FORMAT) for _date in chosen_train_dates)
                self.fusion_dict[_date.strftime(self.config.FILE_DATE_FORMAT)] = train_pair

    def edit_parameter_file(self, path, train_set, pred_set):
        data = self.read_file(path)
        for idx, line in enumerate(data):
            if line.__contains__("IN_PAIR_MODIS_FNAME"):
                data[idx] = self.modify_parameter(self.coarse_train_path, line, train_set)
            elif line.__contains__("IN_PAIR_LANDSAT_FNAME"):
                data[idx] = self.modify_parameter(self.fine_train_path, line, train_set)
            elif line.__contains__("IN_PDAY_MODIS_FNAME"):
                data[idx] = self.modify_parameter(self.coarse_train_path, line, pred_set)
            elif line.__contains__("OUT_PDAY_LANDSAT_FNAME"):
                tmp = line.split(" ")[0:4]
                for test_date in pred_set:
                    pred_file = os.path.join(self.fine_pred_path, test_date + "-s2-pred.tif")
                    tmp.append(pred_file)
                tmp.append("\n")
                data[idx] = " ".join(tmp)
        self.write_file(path, data)

    @staticmethod
    def is_dataset_present(config, record_date_str, type_='s3'):
        file_list = glob.glob(os.path.join(config.out_dir, "s2-train", 'S2_coreg_fusion_shape_*'+'_{}.tif'.format(record_date_str)))
        if len(config.allowed_tiles) > 0:
            file_list = [f for f in file_list for tile in config.allowed_tiles if tile in f]
        s2_train_files = file_list
        s3_train_file = os.path.join(config.out_dir, "s3-train", 'S3_coreg_fusion_shape_{}.tif'.format(record_date_str))
        if type_ == 'both':
            return len(s2_train_files) > 0 and os.path.isfile(s3_train_file)
        else:
            return os.path.isfile(s3_train_file)

    def update_progress_test(self, test_df, test_dates, is_processed=True):
        test_df.loc[test_df.test_date.isin(test_dates), 'processed'] = is_processed
        if "db" in self.config.sinks:
            data.save_fusion_data(self.config, test_df[test_df.test_date.isin(test_dates)], satellite='S2')
        else:
            test_df.to_csv(self.fusion_test_csv_path, index=False)

    def main(self):
        self.create_fusion_dict()
        if self.fusion_dict == {}:
            logger.info("No data present in local fusion dictionary for %s.", self.config.plant)
            # sys.exit(1)
        
        fusion_list_by_test_date = list(self.fusion_dict.items())
        fusion_list_by_test_date.sort()
        if len(fusion_list_by_test_date) > 0:
            test_df = pd.DataFrame.from_dict(fusion_list_by_test_date)
            test_df.columns = ['test_date','train_dates']
            test_df.loc[:, 'processed'] = False
        else:
            test_df = pd.DataFrame(columns=['test_date','train_dates','processed'])

        if "db" in self.config.sinks:
            old_fusion_list_by_test_date = data.get_fusion_data(self.config, satellite='S2')
            test_df = pd.concat([old_fusion_list_by_test_date, test_df])
            test_df.drop_duplicates(['test_date','train_dates'], keep='first', inplace=True)
            test_df.drop_duplicates(['test_date'], keep='first', inplace=True)
            test_df.sort_values('test_date')
        elif not os.path.exists(self.fusion_test_csv_path):
            test_df.sort_values('test_date')
            test_df.to_csv(self.fusion_test_csv_path, header=True, index=False)
        else:
            old_fusion_list_by_test_date = pd.read_csv(self.fusion_test_csv_path, converters={'train_dates': ast.literal_eval})
            test_df = pd.concat([old_fusion_list_by_test_date, test_df])
            test_df.drop_duplicates(['test_date','train_dates'], inplace=True)
            test_df.drop_duplicates(['test_date'], keep='last', inplace=True)
            test_df.sort_values('test_date')
            test_df.to_csv(self.fusion_test_csv_path, header=True, index=False)

        filtered_df = test_df[(test_df.processed==False)&(test_df.test_date <= self.config.end_date.strftime('%Y-%m-%d'))&(test_df.test_date >= self.config.start_date.strftime('%Y-%m-%d'))]
        logger.info("Total test dates for %s fusion %d", self.config.plant, filtered_df.shape[0])
        # Filter non processed and within dates constraint
        train_df = filtered_df.groupby(by='train_dates').agg({'test_date':list}).reset_index()
        for train_tuple in train_df.itertuples():
            train_set = train_tuple.train_dates
            pred_set = train_tuple.test_date
            absent_dates = []
            for train_date in train_set:
                if not self.is_dataset_present(self.config, train_date, type_='both'):
                    absent_dates.append(train_date)
            for train_date in pred_set:
                if not self.is_dataset_present(self.config, train_date):
                    absent_dates.append(train_date)
            if len(absent_dates) > 0:
                logger.info("Preparing data for %s dates %s", self.config.plant, absent_dates)
                prepare_data_for_date(self.config, absent_dates)

            try:
                start = time.time()
                is_completed = False
                full_batch_processed = True
                batch_num = 1
                batch_size = 7 # Constraint of https://github.com/HPSCIL/cuESTARFM/blob/d582c548cfbce8b54fa39c1bf2e414d45c22689e/Codes/trans.cpp#L13
                pred_set_count = len(pred_set)
                while not is_completed and pred_set_count > 0:
                    current_pred_batch = pred_set[batch_size*(batch_num-1):batch_size*batch_num]
                    logger.info("Prediction running for %s %s train set %s", self.config.plant, ", ".join(current_pred_batch), str(train_set))
                    self.update_progress_test(test_df, current_pred_batch, is_processed=False)
                    if len(train_set) > 1:
                        self.edit_parameter_file(self.estarfm_parameter_file_path, train_set, current_pred_batch)
                        process = subprocess.run([self.cuESTARFM_path, self.estarfm_parameter_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    else:
                        self.edit_parameter_file(self.starfm_parameter_file_path, train_set, current_pred_batch)
                        process = subprocess.run([self.cuSTARFM_path, self.starfm_parameter_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if batch_size*batch_num >= pred_set_count:
                        is_completed = True
                    if process.returncode == 0:
                        self.update_progress_test(test_df, current_pred_batch)
                        logger.debug(process.stdout)
                        logger.info("Prediction complete for %s %s", self.config.plant, ", ".join(current_pred_batch))
                    else:
                        full_batch_processed = False
                        logger.debug("Process return code is %d", process.returncode)
                        logger.error(process.stderr)
                        logger.info("Prediction failed for %s %s", self.config.plant, ", ".join(current_pred_batch))
                    batch_num += 1
                end = time.time()
                logger.info("Prediction completed for %s in %.2f secs for %d files. Train dates %s Predicted Dates: %s", self.config.plant, end-start, len(pred_set), ", ".join(train_set), ", ".join(pred_set))
            except Exception as e:
                logger.error("Exception occured while Fusion for %s prediction dates %s Exception: %s trace: %s", self.config.plant, ", ".join(pred_set), str(e), traceback.format_exc())
        logger.info("Prediction completed for %s all test images", self.config.plant)
        alert.send_alert("Fusion completed for *{}* all test images".format(self.config.plant))

    @classmethod
    def read_file(cls, path):
        data = []
        with open(path, 'r') as file:
                # read a list of lines into data
                data = file.readlines()
        return data

    @classmethod
    def write_file(cls, path, data):
        with open(path, 'w') as file:
            file.writelines(data)

    def modify_parameter(self, path, parameter_line, image_dates):
        tmp = parameter_line.split(" ")[0:4]
        for img_date in image_dates :
            file_list = glob.glob(os.path.join(path, "*" + img_date + ".tif"))
            if path == self.fine_train_path:
                if len(self.config.allowed_tiles) > 0:
                    file_list = [f for f in file_list for tile in self.config.allowed_tiles if tile in f]
            tmp.append(file_list[0])
        tmp.append("\n")
        return " ".join(tmp)


if __name__ == '__main__':
    args = utils.parse_config_params()
    config = utils.create_config(args, satellite='S2')
    instance = FusionAutomate(config)
    instance.main()
