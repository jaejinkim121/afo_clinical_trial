import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

path_configfile1 = 'D:/OneDrive - SNU/AFO_analysis/' +\
    'afo_clinical_trial/src/clinical_trial_analysis/include'
sys.path.append(os.path.dirname(os.path.expanduser(path_configfile1)))


from include.load_imu_data import load_xls, load_imu
from include.sole_sensor_preprocessing import load_GRF, load_SENSOR_vol
from include.sole_sensor_preprocessing import convert_sole_header
from include.sole_sensor_preprocessing import GPR_df_save, load_GPR
from include.config import PlotFlag


class DataSet:
    def __init__(self, time, data, title,
                 line_type=None,
                 line_color=''):
        self.time = time
        self.data = data
        self.title = title
        self.line_type = line_type
        self.line_color = line_color


# input argument: trial_num, walk_num, force_df=True only for RH-02 ~ RH-06
def get_dataframe_sole_sensor(
        trial_num, walk_num,
        save_sole_header=False,
        save_GPR_priction=False,
        force_df=False
        ):

    path = '../../data/RH-%s/' % (str(trial_num).zfill(2))
    force_sync_path = '../../data/analyzed/sole/df_sync_force.csv'
    sensor_sync_path = '../../data/analyzed/sole/df_sync.csv'
    sole_header_path = '../../data/sole_header_info.json'
    GPR_save_path = \
        '../../data/analyzed/sole/RH-%s/converted_data'\
        % str(trial_num).zfill(2)

    # GRF df, end time
    (df_didim_GRF, GRF_end_time) = load_GRF(path, walk_num)

    # Read sole_header dict
    sole_header_dict = convert_sole_header(
        sole_header_path, save_flag=save_sole_header
        )

    # load L, R sensor df
    (df_vol_L, df_vol_R) = \
        load_SENSOR_vol(path, force_sync_path, sensor_sync_path,
                        str(trial_num).zfill(2), walk_num,
                        sole_header_dict,
                        GRF_end_time)

    # save GPR priction
    if force_df:
        if save_GPR_priction:
            GPR_df_save(
                str(trial_num).zfill(2), str(walk_num).zfill(2),
                df_vol_L, df_vol_R,
                sole_header_dict,
                GPR_save_path)
        # force df
        (df_force_L, df_force_R) = load_GPR(GPR_save_path,
                                            str(walk_num).zfill(2))

        return df_didim_GRF, df_vol_L, df_vol_R,\
            df_force_L, df_force_R, GRF_end_time
    else:
        return df_didim_GRF, df_vol_L, df_vol_R, \
               None, None, GRF_end_time


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


def save_GPR_pricted_data():
    wn_list = {2: [9, 15], 3: [10, 18], 4: [12, 18], 5: [14, 19],
               6: [16, 20], 7: [16, 24], 8: [10, 16], 9: [10, 20],
               10: [12, 21]}
    flag = 0
    for tn in range(7, 11):
        for wn in range(wn_list[tn][0], wn_list[tn][1]+1):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("RH: %s" % str(tn).zfill(2))
            print("walk: %s" % str(wn).zfill(2))
            if flag > 0:
                df_didim_GRF, df_vol_L, df_vol_R, df_force_L, df_force_R, _ = \
                    get_dataframe_sole_sensor(tn, wn,
                                              save_sole_header=False,
                                              save_GPR_priction=True,
                                              force_df=True)
            else:
                df_didim_GRF, df_vol_L, df_vol_R, df_force_L, df_force_R, _ = \
                    get_dataframe_sole_sensor(tn, wn,
                                              save_sole_header=True,
                                              save_GPR_priction=True,
                                              force_df=True)
            flag += 1

    return 0

def plot_walk(trial_num, walk_num):
    output_name = '/tn{}_wn{}.png'.format(trial_num, walk_num)
    output_prefix = '../../image/plot/'
    # ----------------- DATA LOADING ------------------- #

    df_didim_kinematics, df_imu = \
        get_dataframe_imu(trial_num, walk_num)
    df_didim_GRF, df_vol_L, df_vol_R, df_force_L, df_force_R, end_time = \
        get_dataframe_sole_sensor(
            trial_num, walk_num, force_df=PlotFlag.USE_FORCE)

    # ------------------ FLAG HANDLING ----------------- #
    data = []
    if PlotFlag.USE_DIDIM_GRF:
        output_prefix += '_GRF'
        grf_data = list()
        grf_data.append(DataSet(
            df_didim_GRF['time'],
            df_didim_GRF['L_GRF_VRT'],
            'Left GRF VRT'))
        grf_data.append(DataSet(
            df_didim_GRF['time'],
            df_didim_GRF['R_GRF_VRT'],
            'Right GRF VRT'
        ))
        data.append(grf_data)

    if PlotFlag.USE_DIDIM_GRF_LAT:
        output_prefix += '_LAT'
        grf_data = list()
        grf_data.append(DataSet(
            df_didim_GRF['time'],
            df_didim_GRF['L_GRF_LAT'],
            'Left GRF LAT'))
        grf_data.append(DataSet(
            df_didim_GRF['time'],
            df_didim_GRF['R_GRF_LAT'],
            'Right GRF LAT'
        ))
        data.append(grf_data)

    if PlotFlag.USE_DIDIM_GRF_FWD:
        output_prefix += '_FWD'
        grf_data = list()
        grf_data.append(DataSet(
            df_didim_GRF['time'],
            df_didim_GRF['L_GRF_FWD'],
            'Left GRF FWD'))
        grf_data.append(DataSet(
            df_didim_GRF['time'],
            df_didim_GRF['R_GRF_FWD'],
            'Right GRF FWD'
        ))
        data.append(grf_data)

    if PlotFlag.USE_VOLT:
        output_prefix += '_VOLT'
        if PlotFlag.VOLT_SEP:
            if PlotFlag.USE_VOLT_LorR:
                for key in df_vol_L:
                    if key == 'time':
                        continue
                    if key.split()[-1] == 'head':
                        line_color = 'r'
                        if key.split()[1] == 'first':
                            line_type = 'solid'
                        elif key.split()[1] == 'third':
                            line_type = 'dash'
                        else:
                            line_type = 'dot'
                    elif key.split()[-1] == 'toe':
                        line_color = 'b'
                        if key.split()[1] == 'first':
                            line_type = 'solid'
                        else:
                            line_type = 'dot'
                    elif key.split()[-1] == 'tuberosity':
                        line_color = 'g'
                        if key.split()[1] == 'medial':
                            line_type = 'solid'
                        elif key.split()[1] == 'lateral':
                            line_type = 'dash'
                        else:
                            line_type = 'dot'
                    data.append([DataSet(
                        df_vol_L['time'], df_vol_L[key], key,
                        line_type=line_type, line_color=line_color
                    )])
            else:
                for key in df_vol_R:
                    if key == 'time':
                        continue
                    if key.split()[-1] == 'head':
                        plot_option = 'dot'
                    elif key.split()[-1] == 'toe':
                        plot_option = 'solid'
                    elif key.split()[-1] == 'tuberosity':
                        plot_option = 'dash'
                    data.append([DataSet(
                        df_vol_R['time'], df_vol_R[key], key
                    )])
        else:
            # Left
            sole_data_left = list()
            for key in df_vol_L:
                if key == 'time':
                    continue
                if key.split()[-1] == 'head':
                    line_color = 'r'
                    if key.split()[1] == 'first':
                        line_type = 'solid'
                    elif key.split()[1] == 'third':
                        line_type = 'dash'
                    else:
                        line_type = 'dot'
                elif key.split()[-1] == 'toe':
                    line_color = 'b'
                    if key.split()[1] == 'first':
                        line_type = 'solid'
                    else:
                        line_type = 'dot'
                elif key.split()[-1] == 'tuberosity':
                    line_color = 'g'
                    if key.split()[1] == 'medial':
                        line_type = 'solid'
                    elif key.split()[1] == 'lateral':
                        line_type = 'dash'
                    else:
                        line_type = 'dot'
                sole_data_left.append(DataSet(
                    df_vol_L['time'], df_vol_L[key], key,
                    line_type=line_type, line_color=line_color
                ))
            # Right
            sole_data_right = list()
            for key in df_vol_R:
                if key == 'time':
                    continue
                if key.split()[-1] == 'head':
                    line_color = 'r'
                    if key.split()[1] == 'first':
                        line_type = 'solid'
                    elif key.split()[1] == 'third':
                        line_type = 'dash'
                    else:
                        line_type = 'dot'
                elif key.split()[-1] == 'toe':
                    line_color = 'b'
                    if key.split()[1] == 'first':
                        line_type = 'solid'
                    else:
                        line_type = 'dash'
                elif key.split()[-1] == 'tuberosity':
                    line_color = 'g'
                    if key.split()[1] == 'medial':
                        line_type = 'solid'
                    elif key.split()[1] == 'lateral':
                        line_type = 'dash'
                    else:
                        line_type = 'dot'
                sole_data_right.append(DataSet(
                    df_vol_R['time'], df_vol_R[key], key,
                    line_type=line_type, line_color=line_color
                ))
            data.append(sole_data_left)
            data.append(sole_data_right)

    if PlotFlag.USE_FORCE:
        output_prefix += '_FORCE'
        if PlotFlag.VOLT_SEP:
            if PlotFlag.USE_VOLT_LorR:
                for key in df_vol_L:
                    if key == 'time':
                        continue
                    if key.split()[-1] == 'head':
                        line_color = 'r'
                        if key.split()[1] == 'first':
                            line_type = 'solid'
                        elif key.split()[1] == 'third':
                            line_type = 'dash'
                        else:
                            line_type = 'dot'
                    elif key.split()[-1] == 'toe':
                        line_color = 'b'
                        if key.split()[1] == 'first':
                            line_type = 'solid'
                        else:
                            line_type = 'dot'
                    elif key.split()[-1] == 'tuberosity':
                        line_color = 'g'
                        if key.split()[1] == 'medial':
                            line_type = 'solid'
                        elif key.split()[1] == 'lateral':
                            line_type = 'dash'
                        else:
                            line_type = 'dot'
                    data.append([DataSet(
                        df_force_L['time'], df_force_L[key], key,
                        line_type=line_type, line_color=line_color
                    )])
            else:
                for key in df_vol_R:
                    if key == 'time':
                        continue
                    if key.split()[-1] == 'head':
                        plot_option = 'dot'
                    elif key.split()[-1] == 'toe':
                        plot_option = 'solid'
                    elif key.split()[-1] == 'tuberosity':
                        plot_option = 'dash'
                    data.append([DataSet(
                        df_force_R['time'], df_force_R[key], key
                    )])
        else:
            # Left
            sole_data_left = list()
            for key in df_vol_L:
                if key == 'time':
                    continue
                if key.split()[-1] == 'head':
                    line_color = 'r'
                    if key.split()[1] == 'first':
                        line_type = 'solid'
                    elif key.split()[1] == 'third':
                        line_type = 'dash'
                    else:
                        line_type = 'dot'
                elif key.split()[-1] == 'toe':
                    line_color = 'b'
                    if key.split()[1] == 'first':
                        line_type = 'solid'
                    else:
                        line_type = 'dot'
                elif key.split()[-1] == 'tuberosity':
                    line_color = 'g'
                    if key.split()[1] == 'medial':
                        line_type = 'solid'
                    elif key.split()[1] == 'lateral':
                        line_type = 'dash'
                    else:
                        line_type = 'dot'
                sole_data_left.append(DataSet(
                    df_force_L['time'], df_force_L[key], key,
                    line_type=line_type, line_color=line_color
                ))
            # Right
            sole_data_right = list()
            for key in df_vol_R:
                if key == 'time':
                    continue
                if key.split()[-1] == 'head':
                    line_color = 'r'
                    if key.split()[1] == 'first':
                        line_type = 'solid'
                    elif key.split()[1] == 'third':
                        line_type = 'dash'
                    else:
                        line_type = 'dot'
                elif key.split()[-1] == 'toe':
                    line_color = 'b'
                    if key.split()[1] == 'first':
                        line_type = 'solid'
                    else:
                        line_type = 'dash'
                elif key.split()[-1] == 'tuberosity':
                    line_color = 'g'
                    if key.split()[1] == 'medial':
                        line_type = 'solid'
                    elif key.split()[1] == 'lateral':
                        line_type = 'dash'
                    else:
                        line_type = 'dot'
                sole_data_right.append(DataSet(
                    df_force_R['time'], df_force_R[key], key,
                    line_type=line_type, line_color=line_color
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

    if PlotFlag.USE_IMU_ACCEL:
        output_prefix += '_ACC'
        data_acc = list()
        for imu in df_imu:
            key = imu.keys()[-2]
            if key.split()[0] == 'Left':
                line_color = 'r'
            elif key.split()[0] == 'Right':
                line_color = 'b'
            else:
                line_color = 'g'
            if key.split()[-1].split('_')[0] == 'Thigh':
                line_type = 'dash'
            elif key.split()[-1].split('_')[0] == 'Shank':
                line_type = 'dot'
            else:
                line_type = 'solid'

            data_acc.append(DataSet(
                imu['time'], imu[key], key,
                line_type=line_type, line_color=line_color
            ))
        data.append(data_acc)

    if PlotFlag.USE_IMU_GYRO:
        output_prefix += '_GYRO'
        data_gyro = list()
        for imu in df_imu:
            key = imu.keys()[-1]
            if key.split()[0] == 'Left':
                line_color = 'r'
            elif key.split()[0] == 'Right':
                line_color = 'b'
            else:
                line_color = 'g'
            if key.split()[-1].split('_')[0] == 'Thigh':
                line_type = 'dash'
            elif key.split()[-1].split('_')[0] == 'Shank':
                line_type = 'dot'
            else:
                line_type = 'solid'
            data_gyro.append(DataSet(
                imu['time'], imu[key], key,
                line_type=line_type, line_color=line_color
            ))
        data.append(data_gyro)

    data_length = len(data)    # TEMP

    # -------------------  PLOT  ----------------------- #
    matplotlib.rcParams['figure.figsize'] = 20, 3 * data_length
    line_type = {
        None: '-',
        'solid': '-',
        'dash': '--',
        'dash_dot': '-.',
        'dot': ':'
    }
    fig = plt.figure()
    ax = [fig.add_subplot(data_length, 1, i) for i in range(1, data_length + 1)]

    for i in range(data_length):
        current_ax = ax[i]
        for dataset in data[i]:
            current_ax.plot(dataset.time,
                            dataset.data,
                            dataset.line_color + line_type[dataset.line_type],
                            label=dataset.title,
                            )

        current_ax.set_xticks(np.arange(0, end_time, 0.25))
        current_ax.set_xlim([0, end_time+1])
        current_ax.legend(loc='right')
        current_ax.grid(axis='x',
                        linestyle='--')
        current_ax.hlines(0, xmin=0, xmax=end_time, colors='k', linestyles='--')
    plt.jet()
    plt.tight_layout()
    os.makedirs(output_prefix, exist_ok=True)
    plt.savefig(output_prefix + output_name)
    return 0


def main():
    wn_list = {2: [9, 15], 3: [10, 18], 4: [12, 18], 5: [14, 19],
               6: [16, 20], 7: [16, 24], 8: [10, 16], 9: [10, 20],
               10: [12, 21]}
    for tn in range(2, 11):
        for wn in range(wn_list[tn][0], wn_list[tn][1]+1):
            print("Current Trial: {}, Current Walk: {}".format(tn, wn))
            plot_walk(tn, wn)




if __name__ == "__main__":
    main()

