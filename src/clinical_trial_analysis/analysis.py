from include.load_imu_data import load_xls, load_imu
from include.sole_sensor_preprocessing import folder_path_name, force_sensor_sync
from include.sole_sensor_preprocessing import load_GRF, load_SENSOR_vol
from include.config import PlotFlag

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


def get_dataframe_sole_sensor(trial_num, walk_num):

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
    (GRF_file, GRF_name) = folder_path_name(
        path, "end",
        "WALK%s.XLS" % walk_num,
        T_F=1)
    (df_didim_GRF, GRF_end_time) = load_GRF(GRF_file[0])

    # sensor path, end time
    sensor_path = path + "RasPi/sole/"
    L_sensor_end_time = L_sensor_start_time + float(GRF_end_time)
    R_sensor_end_time = R_sensor_start_time + float(GRF_end_time)
    # foot pressure sensor data
    vol = ['time', 'v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']

    # sensor L, R path
    (L_walk_data_list, L_walk_name_list) = \
        folder_path_name(sensor_path, "start", "L", 1)
    (R_walk_data_list, R_walk_name_list) = \
        folder_path_name(sensor_path, "start", "R", 1)

    # L dataframe
    df_vol_L = load_SENSOR_vol(L_walk_data_list[0], trial_num)
    df_vol_L = df_vol_L.loc[
        (df_vol_L["time"] >= L_sensor_start_time) &
        (df_vol_L["time"] <= L_sensor_end_time)
        ][vol]
    # initialize L time
    df_vol_L["time"] = df_vol_L["time"] - L_sensor_start_time

    # R dataframe
    df_vol_R = load_SENSOR_vol(R_walk_data_list[0], trial_num)
    df_vol_R = df_vol_R.loc[(df_vol_R["time"] >= R_sensor_start_time) & (
            df_vol_R["time"] <= R_sensor_end_time)][vol]
    # initialize R time
    df_vol_R["time"] = df_vol_R["time"] - R_sensor_start_time

    return df_didim_GRF, df_vol_L, df_vol_R


def get_dataframe_imu(trial_num, walk_num):
    # Define File Paths
    # Fill zero to make directory name a form of RH-##
    path_trial = '../../data/RH-{}/'.format(str(trial_num).zfill(2))
    path_trimmed_imu = path_trial + "imu/"

    df_didim_kinematics = load_xls(
        path_trial + "TRIMMED_WALK{}.xls".format(str(walk_num).zfill(2))
    )

    df_imu = load_imu(path_trimmed_imu + "trimmed_walk{}_with_imu.xlsx".format(str(walk_num).zfill(2)))

    return df_didim_kinematics, df_imu


def main():
    trial_num, walk_num = 8, 15

    # ----------------- DATA LOADING ------------------- #

    df_didim_kinematics, df_imu = \
        get_dataframe_imu(trial_num, walk_num)
    df_didim_GRF, df_vol_L, df_vol_R = \
        get_dataframe_sole_sensor(trial_num, walk_num)

    # -------------------  PLOT  ----------------------- #
    

    return 0


if __name__ == "__main__":
    main()
