import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


path_configfile1 = 'D:/OneDrive - SNU/AFO_analysis/afo_clinical_trial/src/clinical_trial_analysis/include'
sys.path.append(os.path.dirname(os.path.expanduser(path_configfile1)))


from include.load_imu_data import load_xls, load_imu
from include.sole_sensor_preprocessing import folder_path_name
from include.sole_sensor_preprocessing import force_sensor_sync
from include.sole_sensor_preprocessing import load_GRF, load_SENSOR_vol
from include.sole_sensor_preprocessing import N_data_preprocessing
from include.sole_sensor_preprocessing import GPR_df_save
from include.sole_sensor_preprocessing import load_GPR
from include.config import PlotFlag


class DataSet:
    def __init__(self, time, data, title):
        self.time = time
        self.data = data
        self.title = title


def get_dataframe_sole_sensor(trial_num, walk_num):
    # trial number (int -> string)
    trial_num = str(trial_num).zfill(2)

    path = '../../data/RH-%s/' % (str(trial_num))
    force_sync_path = '../../data/analyzed/sole/df_sync_force.csv'
    sensor_sync_path = '../../data/analyzed/sole/df_sync.csv'

    # load sync time
    (force_start_time, L_sensor_start_time,
     R_sensor_start_time) = force_sensor_sync(
        force_sync_path, sensor_sync_path, trial_num, walk_num)

    # modify walk num (string)
    walk_num = str(walk_num).zfill(2)

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
    volt_header = ['time', 'v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']

    # sensor L, R path
    (L_walk_data_list, L_walk_name_list) = \
        folder_path_name(sensor_path, "start", "L", 1)
    (R_walk_data_list, R_walk_name_list) = \
        folder_path_name(sensor_path, "start", "R", 1)

    # L dataframe
    df_vol_L = load_SENSOR_vol(L_walk_data_list[0], trial_num)
    df_vol_L = df_vol_L.loc[(df_vol_L["time"] >= L_sensor_start_time) & (
        df_vol_L["time"] <= L_sensor_end_time)][volt_header]
    # initialize L time
    df_vol_L["time"] = df_vol_L["time"] - L_sensor_start_time

    # R dataframe
    df_vol_R = load_SENSOR_vol(R_walk_data_list[0], trial_num)
    df_vol_R = df_vol_R.loc[(df_vol_R["time"] >= R_sensor_start_time) & (
        df_vol_R["time"] <= R_sensor_end_time)][volt_header]
    # initialize R time
    df_vol_R["time"] = df_vol_R["time"] - R_sensor_start_time

    # ##########################################################
    # # N-data preprocessing for GPR prediction
    # ##########################################################
    GPR_save_path = '../../data/analyzed/sole/RH-%s/converted_data' % (
        trial_num)

    # Model loading and GPR force df save (ONLY USE FOR SAVING !!!!!)
    # GPR_df_save(trial_num, df_vol_L, df_vol_R, volt_header, GPR_save_path)
    # Load GPR force df
    (df_force_L, df_force_R) = load_GPR(GPR_save_path)

    return df_didim_GRF, df_vol_L, df_vol_R, df_force_L, df_force_R, GRF_end_time


def get_dataframe_imu(trial_num, walk_num):
    # Define File Paths
    # Fill zero to make directory name a form of RH-##
    path_trial = '../../data/RH-{}/'.format(str(trial_num).zfill(2))
    path_trimmed_imu = path_trial + "imu/"

    df_didim_kinematics = load_xls(
        path_trial + "TRIMMED_WALK{}.xls".format(str(walk_num).zfill(2))
    )

    df_imu = load_imu(path_trimmed_imu + "v2_trimmed_walk{}_with_imu.xlsx".format(str(walk_num)))

    return df_didim_kinematics, df_imu


def save_GPR_predicted_data():
    walk_num = {"02": 9, "03": 10, "04": 12, "05": 14, "06": 16, "07": 16,
                "08": 10, "09": 10, "10": 12}
    walk_end = {"02": 15, "03": 18, "04": 18, "05": 19, "06": 20, "07": 24,
                "08": 16, "09": 20, "10": 21}
    
    for trial_num in range(2,11,1):
        for walk_num in range(walk_num[str(trial_num).zfill(2)],
                              walk_end[str(trial_num).zfill(2)], 1):
            df_didim_GRF, df_vol_L, df_vol_R, df_force_L, df_force_R, end_time = \
                get_dataframe_sole_sensor(trial_num, walk_num)

    return 0

def plot_walk(trial_num, walk_num):
    output_name = '/tn{}_wn{}.png'.format(trial_num, walk_num)
    output_prefix = '../../image/plot/'
    # ----------------- DATA LOADING ------------------- #

    df_didim_kinematics, df_imu = \
        get_dataframe_imu(trial_num, walk_num)
    df_didim_GRF, df_vol_L, df_vol_R, df_force_L, df_force_R, end_time = \
        get_dataframe_sole_sensor(trial_num, walk_num)

    # ------------------ FLAG HANDLING ----------------- #
    data = []
    if PlotFlag.USE_DIDIM_GRF:
        output_prefix += '_GRF'
        grf_data = list()
        grf_data.append(DataSet(
            df_didim_GRF['time'],
            df_didim_GRF['L_GRF_VRT'],
            'Left foot GRF'))
        grf_data.append(DataSet(
            df_didim_GRF['time'],
            df_didim_GRF['R_GRF_VRT'],
            'Right foot GRF'
        ))
        data.append(grf_data)

    if PlotFlag.USE_VOLT:
        output_prefix += '_VOLT'
        if PlotFlag.VOLT_SEP:
            if PlotFlag.USE_VOLT_LorR:
                for key in df_vol_L:
                    if key == 'time':
                        continue
                    data.append([DataSet(
                        df_vol_L['time'], df_vol_L[key], key
                    )])
            else:
                for key in df_vol_R:
                    if key == 'time':
                        continue
                    data.append([DataSet(
                        df_vol_R['time'], df_vol_R[key], key
                    )]  )
        else:
            # Left
            sole_data_left = list()
            for key in df_vol_L:
                if key == 'time':
                    continue
                sole_data_left.append(DataSet(
                    df_vol_L['time'], df_vol_L[key], key
                ))
            # Right
            sole_data_right = list()
            for key in df_vol_R:
                if key == 'time':
                    continue
                sole_data_right.append(DataSet(
                    df_vol_R['time'], df_vol_R[key], key
                ))
            data.append(sole_data_left)
            data.append(sole_data_right)

    if PlotFlag.USE_DIDIM_KINEMATICS:
        if PlotFlag.USE_DIDIM_KINEMATICS_ALL:
            output_prefix += '_KIN'
        else:
            output_prefix += '_FLEX'
        joint_data_left = list()
        joint_data_right = list()
        for key in df_didim_kinematics:
            if key == 'time':
                continue
            elif not PlotFlag.USE_DIDIM_KINEMATICS_ALL:
                if ('Abd' in key) or ('Rot' in key):
                    continue
            if key[0] == 'L':
                joint_data_left.append(DataSet(
                    df_didim_kinematics['time'], df_didim_kinematics[key], key
                ))
            elif key[0] == 'R':
                joint_data_right.append(DataSet(
                    df_didim_kinematics['time'], df_didim_kinematics[key], key
                ))
        data.append(joint_data_left)
        data.append(joint_data_right)

    data_length = len(data)    # TEMP

    # -------------------  PLOT  ----------------------- #
    matplotlib.rcParams['figure.figsize'] = 20, 3 * data_length
    fig = plt.figure()
    ax = [fig.add_subplot(data_length, 1, i) for i in range(1, data_length + 1)]

    for i in range(data_length):
        current_ax = ax[i]
        for dataset in data[i]:
            current_ax.plot(dataset.time, dataset.data, label=dataset.title)

        current_ax.set_xticks(np.arange(0, end_time, 0.25))
        current_ax.set_xlim([0, end_time+0.7])
        current_ax.legend(loc='right')
        current_ax.grid(axis='x',
                        linestyle='--')

    plt.tight_layout()
    os.makedirs(output_prefix, exist_ok=True)
    plt.savefig(output_prefix + output_name)
    return 0


def main():
    wn_list = {2:[9, 15], 3:[10, 18], 4:[12, 18], 5:[14, 19], 6:[16, 20],
               7:[16, 24], 8:[10, 16], 9:[10, 20], 10:[12, 21]}
    for tn in range(2, 11):
        for wn in range(wn_list[tn][0], wn_list[tn][1]+1):
            plot_walk(tn, wn)


if __name__ == "__main__":
    main()
