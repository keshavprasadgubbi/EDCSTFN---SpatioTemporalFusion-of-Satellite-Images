import os
import subprocess
import sys
import traceback

import alert
import utils
from send import Queue

def main():
    """
    Ensure that before running script you have shape files with identifier
    NP, OP, BF, [FP, FU, FUS](for fusion)
    Also ensure setting appropriate settings in config.py and ALERT_URL in environment variables
    """
    # Changing directory to current file path to fix import statements for credios-finder
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    instance_info = utils.get_current_instance_info()
    instance_name = instance_info.get('instance_name')
    instance_ip = instance_info.get('instance_ip')
    args = utils.parse_config_params()

    from download_s2 import DownloadSentinelS2
    from s2_stats import SentinelStats
    from download_s3 import DownloadSentinelS3
    from s3_coregistration_and_fusion_prep import DataPreprationS3
    from fusion_automate import FusionAutomate
    from final_index import FinalIndex
    from dataset import prepare_data_for_date

    # Flag to enable operations
    SENTINEL2, FUSION = (True, True) if args.all > 0 else (args.sentinel2 > 0, args.fusion > 0)

    is_shutdown_enabled = args.enable_shutdown > 0
    is_delete_enabled = args.enable_delete > 0
    is_release = args.release
    success = False
    try:
        config = utils.create_config(args, satellite='S3')
        queue_config = config.conf.get('queue_config')
        if SENTINEL2:
            instance = DownloadSentinelS2(config)
            instance.main()
            instance = SentinelStats(config)
            instance.main()

        if FUSION:
            instance = DownloadSentinelS3(config)
            instance.main()
            # Check common dates / if dates not exist download
            instance = DataPreprationS3(config)
            common_date = instance.get_common_date(config)
            if common_date:
                if not instance.check_common_files(common_date):
                    prepare_data_for_date(config, common_date)

                
            instance.main()
            alert.send_alert("Finished fusion data prepration for *{}*".format(config.plant))
            instance = FusionAutomate(config)
            instance.main()
            alert.send_alert("Finished Fusion calculation for *{}*".format(config.plant))
        else:
            instance = DataPreprationS3(config)
            instance.data_prepration_s2()

        instance = FinalIndex(config)
        instance.main()
        if "s3" in config.sinks:
            subprocess.run(['aws', 's3', 'cp', os.path.join(config.out_dir, config.plant + '_sentinel_final_pivoted_index.csv'), 's3://{}/{}/'.format(config.bucket_name, config.plant)])
        # File deletion flag check
        if is_delete_enabled:
            from remove_files import clear_files
            clear_files(config)
        success = True
    except Exception as e:
        print("Exception occurred in main method {}".format(str(e)), file=sys.stderr)
        print("Trace {}".format(traceback.format_exc()), file=sys.stderr)
    
    if is_release:
        try:
            q = Queue(queue_config.get('response_queue'), host=queue_config.get('host'), port=queue_config.get('port'), username=queue_config.get('username'), password=queue_config.get('password'))
            q.publish({
                "request_id": args.request_id,
                "plant_identifier": args.plant,
                "start_date": args.start_date.isoformat(),
                "end_date": args.end_date.isoformat(),
                "status": "COMPLETED" if success else "FAILED"
            })
            q.close()
        except:
            alert.send_alert("Error occurred in publishing message for request id: {} plant: *{}*".format(args.request_id, args.plant))
    if is_shutdown_enabled:
        alert.send_alert("Shutting down machine: {} IP: {} for *{}*".format(instance_name, instance_ip, config.plant))
        subprocess.run(['sudo', 'poweroff'])

if __name__ == '__main__':
    main()