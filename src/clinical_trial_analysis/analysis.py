from common.load_xls import load_xls
from include.sole_header import folder_path_name, force_sensor_sync
from include.sole_header import load_GRF, load_SENSOR_vol


MAX_TRIAL = 100


def get_full_file_path(prefix, suffix, index):
    path = ""
    for pre in prefix:
        path += pre

    # To make 2 digit number
    path += str(index).zfill(2)

    for suf in suffix:
        path += suf

    return path


"""
이민희 잘해라
"""


def lmh(trial_num, walk_num):

    path = '../../data/RH-%s/' % (str(trial_num))
    force_sync_path = '../../data/analyzed/sole/df_sync_force.csv'
    sensor_sync_path = '../../data/analyzed/sole/df_sync.csv'

    # load sync time
    (force_start_time, L_sensor_start_time,
     R_sensor_start_time) = force_sensor_sync(
         force_sync_path, sensor_sync_path, trial_num, walk_num)

    # modify walk num (string)
    if (len(str(walk_num)) == 1):
        walk_num = "0" + str(walk_num)
    else:
        walk_num = str(walk_num)

    # GRF dataframe, end time
    (GRF_file, GRF_name) = folder_path_name(path, "end", "WALK%s.XLS" % (
        walk_num), 1)
    (df_didim_GRF, GRF_end_time) = load_GRF(GRF_file[0])

    # sensor path, end time
    sensor_path = path + "RasPi/sole/"
    L_sensor_end_time = L_sensor_start_time + float(GRF_end_time)
    R_sensor_end_time = R_sensor_start_time + float(GRF_end_time)
    # foot pressure sensor data
    vol = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']

    # sensor L, R path
    (L_walk_data_list,
     L_walk_name_list) = folder_path_name(sensor_path, "start", "L", 1)
    (R_walk_data_list,
     R_walk_name_list) = folder_path_name(sensor_path, "start", "R", 1)

    # L dataframe
    df_vol_L = load_SENSOR_vol(L_walk_data_list[0], trial_num)
    df_vol_L = df_vol_L.loc[(df_vol_L.time >= L_sensor_start_time) & (
        df_vol_L.time <= L_sensor_end_time)][vol]
    # initialize L time
    df_vol_L.time -= L_sensor_start_time

    # R dataframe
    df_vol_R = load_SENSOR_vol(R_walk_data_list[0], trial_num)
    df_vol_R = df_vol_R.loc[(df_vol_R.time >= R_sensor_start_time) & (
        df_vol_R.time <= R_sensor_end_time)][vol]
    # initialize R time
    df_vol_R.time -= R_sensor_start_time

    return df_didim_GRF, df_vol_L, df_vol_R


"""
여기까지
"""

def kjj(trial_num, walk_num):
    # Define File Paths
    path_test = '../../data/RH-09/'
    index_list = []

    # Use key value for purposes.
    # 'device' for our own device data
    # 'motion_capture' for didim data
    data_type_prefix = {'device': 'data/', 'motion_capture': ''}
    file_name_prefix = {'device': 'temp', 'motion_capture': 'TRIMMED_WALK'}
    file_name_suffix = {'device': 'temp', 'motion_capture': '.XLS'}

    # Load data and make dataframe
    df_didim = []
    key = 'motion_capture'

    for i in range(1, MAX_TRIAL):
        full_path_didim = get_full_file_path(
            prefix=[path_test, data_type_prefix[key], file_name_prefix[key]],
            suffix=[file_name_suffix[key]],
            index=i
        )
        #
        try:
            df_didim.append(load_xls(full_path_didim))
        except FileNotFoundError:
            print("END of reading files")
            break
    print(len(df_didim))
    """
    Kinematic Data Plot

      1. Roll, Pitch, Yaw Angle of 7 limbs.

    """
    #

    return df_didim_kinematics


def main():

    lmh("07", 12)
    
    return 0


if __name__ == "__main__":
    main()
