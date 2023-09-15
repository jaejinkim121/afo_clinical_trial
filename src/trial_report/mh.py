from bagpy import bagreader
import os
import glob
import pandas as pd
import numpy as np
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import DataProcess


#############################################################################
#############################################################################
# BY JJ
#############################################################################
def get_initial_contact_time(gait_phase:pd.DataFrame, data_name="value"):
    if 'time' not in gait_phase.columns:
        print("Time data missing")
        return -1
    if data_name not in gait_phase.columns:
        print("Value data missing")
        return -1

    time_initial_contact = gait_phase[gait_phase[data_name] == 1]["time"]

    return time_initial_contact.tolist()


def get_foot_off_time(gait_phase:pd.DataFrame, data_name="value"):
    if 'time' not in gait_phase.columns:
        print("Time data missing")
        return -1
    if data_name not in gait_phase.columns:
        print("Value data missing")
        return -1

    time_foot_off = gait_phase[gait_phase[data_name] == 2]["time"]

    return time_foot_off.tolist()


def get_gait_event_time(gait_phase:pd.DataFrame, data_name="value"):
    time_initial_contact = get_initial_contact_time(gait_phase, data_name)
    time_foot_off = get_foot_off_time(gait_phase, data_name)

    return time_initial_contact, time_foot_off


def gait_phase_pre_processing(
        gait_phase_paretic:pd.DataFrame,
        gait_phase_nonparetic:pd.DataFrame, data_name="value"):
    paretic_ic, paretic_fo = \
        get_gait_event_time(gait_phase_paretic, data_name)
    nonparetic_ic, nonparetic_fo = \
        get_gait_event_time(gait_phase_nonparetic, data_name)
    
    mean_cycle = (paretic_ic[-1] - paretic_ic[0]) / (len(paretic_ic) - 1)

    idx_np = 0
    time_diff = []
    for tic in paretic_ic:
        idx_np += DataProcess.search_index(nonparetic_ic[idx_np:], tic)
        time_diff.append(nonparetic_ic[idx_np] - tic)

    time_diff = np.array(time_diff)
    mean_diff = np.mean(time_diff) / mean_cycle
    std_diff = np.std(time_diff) / mean_cycle

    if paretic_ic[0] > paretic_fo[0]:
        paretic_fo = paretic_fo[1:]
    if paretic_ic[-1] > paretic_fo[-1]:
        paretic_ic = paretic_ic[:-1]

    time_diff = []
    for tic, tfo in zip(paretic_ic, paretic_fo):
        time_diff.append(tfo-tic)
    time_diff = np.array(time_diff)
    mean_cycle_time_paretic = np.mean(time_diff) / mean_cycle
    std_cycle_time_paretic = np.std(time_diff) / mean_cycle

    if nonparetic_ic[0] > nonparetic_fo[0]:
        nonparetic_fo = nonparetic_fo[1:]
    if nonparetic_ic[-1] > nonparetic_fo[-1]:
        nonparetic_ic = nonparetic_ic[:-1]

    time_diff = []
    for tic, tfo in zip(nonparetic_ic, nonparetic_fo):
        time_diff.append(tfo - tic)
    time_diff = np.array(time_diff)
    mean_cycle_time_nonparetic = np.mean(time_diff) / mean_cycle
    std_cycle_time_nonparetic = np.std(time_diff) / mean_cycle

    return [mean_diff, std_diff,
            mean_cycle_time_paretic, std_cycle_time_paretic,
            mean_cycle_time_nonparetic, std_cycle_time_nonparetic]


def divider_data_by_gait_phase_df(data_df, gait_phase_df, data_name="value"):
    divided_array = []
    time_initial_contact = get_initial_contact_time(gait_phase_df, data_name)
    time_initial_contact.append(data_df["time"].iloc[-1])

    for i in range(len(time_initial_contact) - 1):
        divided_df_current = \
            data_df[
                (data_df["time"] >= time_initial_contact[i]) &
                (data_df["time"] < time_initial_contact[i+1])
            ]
        divided_array.append(divided_df_current.to_numpy())

    return divided_array


def graph_averaged_data(collection_data, title_graph, data_label, x_num=101):
    mean, std = DataProcess.average_cropped_time_series(
        collection_data, x_num
    )
    x = np.linspace(0, 100, x_num)
    plt.plot(x, mean, 'k-')
    plt.fill_between(x, mean - std, mean + std)
    plt.xlabel("Cycle Percentage")
    plt.xlim(0, 100)
    plt.ylabel(data_label)
    plt.title(title_graph)
    plt.show()


def graph_both_cycle_data(collection_data_paretic, collection_data_nonparetic,
                          data_gait_paretic, data_gait_nonparetic,
                          data_gait_label="value",
                          title_graph=None, data_label=None, x_num=101):
    [mean_diff_both, std_diff_both,
     mean_diff_paretic, std_diff_paretic,
     mean_diff_nonparetic, std_diff_nonparetic] = gait_phase_pre_processing(
        data_gait_paretic, data_gait_nonparetic, data_gait_label)

    mean_paretic, std_paretic = DataProcess.average_cropped_time_series(
        collection_data_paretic, x_num
    )
    mean_nonparetic, std_nonparetic = DataProcess.average_cropped_time_series(
        collection_data_nonparetic, x_num
    )

    x_paretic = np.linspace(0, 100, x_num)
    x_nonparetic = np.linspace(mean_diff_both*100, (1+mean_diff_both)*100,
                               x_num)

    plt.plot(x_paretic, mean_paretic, 'k-')
    plt.fill_between(x_paretic,
                     mean_paretic - std_paretic,
                     mean_paretic + std_paretic)
    plt.plot(x_nonparetic, mean_nonparetic, 'r-')
    plt.fill_between(x_nonparetic,
                     mean_nonparetic - std_nonparetic,
                     mean_nonparetic + std_nonparetic)

    plt.xlabel("Cycle Percentage")
    plt.ylabel(data_label)
    plt.title(title_graph)
    plt.xlim(0, (1+mean_diff_both)*100)
    plt.show()

#############################################################################
#############################################################################

# Create Directory
def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


# Function for folder path reading
def folder_path_name(path, option=None, char=None, T_F=None):

    folder_name_path = str(path)
    folder_path = folder_name_path + "*"

    file_list = glob.glob(folder_path)
    file_name_list = os.listdir(folder_name_path)

    if option == "start":
        exp_list = [file for (file, name) in zip(file_list, file_name_list)
                    if name.startswith(str(char)) == int(T_F)]
        exp_name_list = [name for (file, name) in
                         zip(file_list, file_name_list)
                         if name.startswith(str(char)) == int(T_F)]

    elif option == "end":
        exp_list = [file for (file, name) in zip(file_list, file_name_list)
                    if name.endswith(str(char)) == int(T_F)]
        exp_name_list = [name for (file, name)
                         in zip(file_list, file_name_list)
                         if name.endswith(str(char)) == int(T_F)]

    elif option == "include":
        exp_list = [file for (file, name) in zip(file_list, file_name_list)
                    if str(char) in name]
        exp_name_list = [name for (file, name)
                         in zip(file_list, file_name_list)
                         if str(char) in name]

    else:
        exp_list = [file for (file, name) in zip(file_list, file_name_list)]
        exp_name_list = [name for (file, name)
                         in zip(file_list, file_name_list)]

    exp_list = [file.replace('\\', '/') for file in exp_list]

    return exp_list, exp_name_list


# LSTM model structure for calibration
class LSTM_Calib(nn.Module):
    def __init__(self, input_size=1, hidden_size_1=15, hidden_size_2=60,
                 hidden_size_3=20, num_layers=2, drop_p=0.3):
        super(LSTM_Calib, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.num_layers = num_layers
        self.dropout = drop_p

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True,
            num_layers=num_layers,
            dropout=drop_p
            )
        self.linear1 = nn.Linear(in_features=self.hidden_size_1,
                                 out_features=self.hidden_size_2)
        self.linear2 = nn.Linear(in_features=self.hidden_size_2,
                                 out_features=self.hidden_size_3)
        self.linear3 = nn.Linear(in_features=self.hidden_size_3,
                                 out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size_1).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size_1).requires_grad_()

        outn, (hn, cn) = self.lstm(x, (h0, c0))
        out1 = self.linear1(outn[:, :, -1])
        out2 = self.linear2(out1)
        out = self.linear3(out2)

        return out


# LSTM model structure for GRF estimation
class LSTM_GRF(nn.Module):
    def __init__(self, input_size=6, hidden_size_1=15, hidden_size_2=60,
                 hidden_size_3=20, num_layers=2, drop_p=0.3):
        super(LSTM_GRF, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.num_layers = num_layers
        self.dropout = drop_p

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True,
            num_layers=num_layers,
            dropout=drop_p
            )
        self.linear1 = nn.Linear(in_features=self.hidden_size_1,
                                 out_features=self.hidden_size_2)
        self.linear2 = nn.Linear(in_features=self.hidden_size_2,
                                 out_features=self.hidden_size_3)
        self.linear3 = nn.Linear(in_features=self.hidden_size_3,
                                 out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size_1, device=x.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size_1, device=x.device).requires_grad_()

        outn, (hn, cn) = self.lstm(x, (h0, c0))
        out1 = self.linear1(outn[:, :, -1])
        out2 = self.linear2(out1)
        out = self.linear3(out2)

        return out


# 후에 force output에 filtering하고 GRF 추정하도록 수정하기
# GRF output이 음수가 나오지 않도록 재확인해보자
class GRF_predictor:
    def __init__(self, start_time: float,
                 leftPath: str,
                 rightPath: str,
                 pareticPath: str,
                 nonpareticPath: str,
                 modelPathCalib: str,
                 modelPathGRF: str,
                 size: str,
                 paretic_side: str,
                 BW: float):
        '''
        Parameters
        ----------
        start_time : float
        leftPath : str
            The example is
            '../../bag/log_2023-08-04-16-26-19/afo_sensor-soleSensor_left.csv'.
        rightPath : str
            The example is
            '../../bag/log_2023-08-04-16-26-19/afo_sensor-soleSensor_right.csv'.
        pareticPath : str
            The example is
            '../../bag/log_2023-08-04-16-26-19/afo_detector-gait_paretic.csv'.
        nonpareticPath : str
            The example is
            '../../bag/log_2023-08-04-16-26-19/afo_detector-gait_nonparetic.csv'.
        modelPathCalib : str
            The example is
            '../../model/CHAR_230815_280_LP/'.
        modelPathGRF : str
            The example is
            '../../model/GRF_230815/LSTM_GRF.pt'.
        size : str
            The example is '280'
        paretic_side : str
            The example is 'L'.
        BW : float

        Returns
        -------
        None.

        '''

        self.start_time = float(start_time)
        self.pareticPath = pareticPath
        self.nonpareticPath = nonpareticPath
        self.paretic_side = str(paretic_side)
        self.BW = BW
        self.leftPath = leftPath
        self.rightPath = rightPath
        self.modelPathCalib = modelPathCalib
        self.modelPathGRF = modelPathGRF
        self.device = torch.device('cpu')
        self.size = str(size)
        self.sensor_num = int(6)
        self.calib_input_length = int(15)
        self.GRF_input_length = int(25)
        # Read gait detection data
        self.Gait_detection_data_read()

    def get_voltage_raw_data(self):
        # Read vout data
        self.vout_data_read()
        # Calib model loading
        self.force_model_load()
        # Calibration model inference
        self.Calib_total_prediction()
        self.GRF_preprocessing()

        left_stance_time = pd.DataFrame(
            self.leftTime.isin(self.SyncedLeftTime) * 1.0
            )
        right_stance_time = pd.DataFrame(
            self.rightTime.isin(self.SyncedRightTime) * 1.0
            )

        left_stance_time.columns = ["stance time"]
        right_stance_time.columns = ["stance time"]

        left_voltage = pd.concat([self.leftData, left_stance_time], axis=1)
        right_voltage = pd.concat([self.rightData, right_stance_time], axis=1)

        return left_voltage, right_voltage

    def get_force_raw_data(self):
        # Read vout data
        self.vout_data_read()
        # Calib model loading
        self.force_model_load()
        # Calibration model inference
        self.Calib_total_prediction()
        self.GRF_preprocessing()

        left_stance_time = pd.DataFrame(
            self.leftTime.isin(self.SyncedLeftTime) * 1.0
            )
        right_stance_time = pd.DataFrame(
            self.rightTime.isin(self.SyncedRightTime) * 1.0
            )

        left_stance_time.columns = ["stance time"]
        right_stance_time.columns = ["stance time"]

        left_force = pd.concat(
            [self.leftTime, pd.DataFrame(self.leftForce), left_stance_time],
            axis=1
            )
        right_force = pd.concat(
            [self.rightTime, pd.DataFrame(self.rightForce), right_stance_time],
            axis=1
            )

        return left_force, right_force

    # Set GRF values of swing phase as zero
    def get_GRF_raw_data(self):
        self.GRF_initialization()

        left_GRF_stance = pd.concat(
            [self.SyncedLeftTime, pd.DataFrame(self.GRFleftData)], axis=1
            )
        right_GRF_stance = pd.concat(
            [self.SyncedRightTime, pd.DataFrame(self.GRFrightData)], axis=1
            )

        left_GRF = self.leftTime.to_frame().merge(
            left_GRF_stance, how='left', on="time"
            )
        left_GRF = left_GRF.fillna(0)
        right_GRF = self.rightTime.to_frame().merge(
            right_GRF_stance, how='left', on="time"
            )
        right_GRF = right_GRF.fillna(0)

        return left_GRF, right_GRF

    def GRF_initialization(self):
        # Read vout data
        self.vout_data_read()
        # Calib model loading
        self.force_model_load()
        # Calibration model inference
        self.Calib_total_prediction()
        self.GRF_preprocessing()
        # GRF model loading
        self.GRF_model_load()
        # GRF model inference
        self.GRF_total_prediction()

    def vout_data_read(self):
        leftData = pd.read_csv(self.leftPath, header=0)
        rightData = pd.read_csv(self.rightPath, header=0)
        # Column selection - time, vout only
        left_column_list = ["time",
                            "v1_L", "v2_L", "v3_L", "v4_L", "v5_L", "v6_L"]
        right_column_list = ["time",
                             "v1_R", "v2_R", "v3_R", "v4_R", "v5_R", "v6_R"]
        column_num_list = [0]
        column_num_list.extend(list(np.arange(3, 3+self.sensor_num)))
        leftData = leftData.iloc[:, column_num_list]
        rightData = rightData.iloc[:, column_num_list]
        leftData.columns = left_column_list
        rightData.columns = right_column_list
        self.leftDataColumns = left_column_list
        self.rightDataColumns = right_column_list
        # Time initialization with respect to start time
        leftData = leftData.astype(float)
        rightData = rightData.astype(float)
        leftData.time -= self.start_time
        rightData.time -= self.start_time
        # Vout data update
        self.leftData = leftData
        self.leftTime = leftData.time
        self.rightData = rightData
        self.rightTime = rightData.time

    def force_model_load(self):
        self.leftModel = np.array([])
        self.rightModel = np.array([])
        for num in np.arange(self.sensor_num):
            leftmodel = LSTM_Calib(hidden_size_1=self.calib_input_length,
                                   hidden_size_2=45,
                                   hidden_size_3=35,
                                   num_layers=6,
                                   drop_p=0.1)
            rightmodel = LSTM_Calib(hidden_size_1=self.calib_input_length,
                                    hidden_size_2=45,
                                    hidden_size_3=35,
                                    num_layers=6,
                                    drop_p=0.1)
            leftmodel.load_state_dict(torch.load(
                self.modelPathCalib + self.size + "Left_" +
                str(num + 1) + ".pt",
                map_location=self.device))
            rightmodel.load_state_dict(torch.load(
                self.modelPathCalib + self.size + "Right_" +
                str(num + 1) + ".pt",
                map_location=self.device))
            self.leftModel = np.append(self.leftModel, leftmodel)
            self.rightModel = np.append(self.rightModel, rightmodel)

    def LSTM_Calib_transform(self, num, sensor_dir="L"):
        if sensor_dir == "L":
            # leftData
            if len(self.leftVout) < self.calib_input_length:
                extra_row = np.repeat(
                    [self.leftVout[0]],
                    repeats=self.calib_input_length - len(self.leftVout),
                    axis=0
                    )
                input_data = np.concatenate((extra_row, self.leftVout), axis=0)
                x = input_data[
                    (len(input_data) - self.calib_input_length):
                        len(input_data), (num - 1)
                        ]
            else:
                input_data = self.leftVout
                x = input_data[
                    (len(self.leftVout) - self.calib_input_length):
                        len(self.leftVout), (num - 1)]
        else:
            # rightData
            if len(self.rightVout) < self.calib_input_length:
                extra_row = np.repeat(
                    [self.rightVout[0]],
                    repeats=self.calib_input_length - len(self.rightVout),
                    axis=0
                    )
                input_data = np.concatenate(
                    (extra_row, self.rightVout),
                    axis=0
                    )
                x = input_data[(len(input_data) - self.calib_input_length):
                               len(input_data),
                               (num - 1)]
            else:
                input_data = self.rightVout
                x = input_data[
                    (len(self.rightVout) - self.calib_input_length):
                        len(self.rightVout), (num - 1)]
        x = torch.from_numpy(x)
        x = x.unsqueeze(1)
        x = torch.stack([x], dim=0).float()

        return x

    def Calib_one_prediction(self, idx, sensor_dir="L"):
        if sensor_dir == "L":
            # leftData
            # leftData indexing
            if idx <= len(self.leftData)-2:
                self.leftVout = self.leftData[0:idx+1][:]
                self.leftVout = self.leftVout.iloc[:, 1:]
                self.leftVout = np.array(self.leftVout)
            else:
                self.leftVout = np.zeros((1, 6))
            # leftData prediction
            _, left_name_list = folder_path_name(
                self.modelPathCalib, "include", "Left")
            left_name_list = [
                name for name in left_name_list
                if int(name[-4]) <= self.sensor_num
                ]
            left_name_list = sorted(left_name_list, key=lambda x: int(x[-4]),
                                    reverse=False)
            leftOutput = np.array([])

            for name in left_name_list:
                model = self.leftModel[int(name[-4]) - 1]
                model.eval()
                with torch.no_grad():
                    x = self.LSTM_Calib_transform(int(name[-4]),
                                                  str(sensor_dir))
                    leftOutput = np.append(leftOutput, model(x))

            return np.expand_dims(leftOutput, axis=0)
        else:
            # rightData
            # rightData indexing
            if idx <= len(self.rightData)-2:
                self.rightVout = self.rightData[0:idx+1][:]
                self.rightVout = self.rightVout.iloc[:, 1:]
                self.rightVout = np.array(self.rightVout)
            else:
                self.rightVout = np.zeros((1, 6))
            # rightData prediction
            _, right_name_list = folder_path_name(
                self.modelPathCalib, "include", "Right")
            right_name_list = [
                name for name in right_name_list
                if int(name[-4]) <= self.sensor_num
                ]
            right_name_list = sorted(right_name_list, key=lambda x: int(x[-4]),
                                     reverse=False)
            rightOutput = np.array([])

            for name in right_name_list:
                model = self.rightModel[int(name[-4]) - 1]
                model.eval()
                with torch.no_grad():
                    x = self.LSTM_Calib_transform(int(name[-4]),
                                                  str(sensor_dir))
                    rightOutput = np.append(rightOutput, model(x))

            return np.expand_dims(rightOutput, axis=0)

    def Calib_total_prediction(self):
        # leftData
        self.leftForce = self.Calib_one_prediction(idx=0, sensor_dir="L")
        left_idx = 1
        for _ in np.arange(1, len(self.leftData)):
            self.leftForce = np.append(self.leftForce,
                                       self.Calib_one_prediction(
                                           idx=left_idx, sensor_dir="L"),
                                       axis=0)
            left_idx += 1
        # rightData
        self.rightForce = self.Calib_one_prediction(idx=0, sensor_dir="R")
        right_idx = 1
        for _ in np.arange(1, len(self.rightData)):
            self.rightForce = np.append(
                self.rightForce,
                self.Calib_one_prediction(
                    idx=right_idx, sensor_dir="R"
                    ), axis=0)
            right_idx += 1

    def Gait_detection_data_read(self):
        # stance = 1, swing = 2
        # stance time: 1 -> 2
        pareticData = pd.read_csv(self.pareticPath, header=0)
        nonpareticData = pd.read_csv(self.nonpareticPath, header=0)
        # Time initialization with respect to start time
        pareticData = pareticData.astype(float)
        nonpareticData = nonpareticData.astype(float)
        pareticData.Time -= self.start_time
        nonpareticData.Time -= self.start_time
        # stance start timing, end timing, duration
        # paretic side
        pareticEvent_df = pd.DataFrame(
            columns=["start time", "end time", "duration"])
        par_flag = 0
        for par_idx in np.arange(len(pareticData)):
            # start time
            if pareticData.iloc[par_idx, 1] == 1:
                paretic_start = pareticData.iloc[par_idx, 0]
                par_flag = 1
            elif (pareticData.iloc[par_idx, 1] == 2) & (par_flag == 1):
                paretic_end = pareticData.iloc[par_idx, 0]
                par_flag = 2
            else:
                pass
            if par_flag == 2:
                paretic_duration = paretic_end - paretic_start
                paretic_add = pd.DataFrame({'start time': [paretic_start],
                                            'end time': [paretic_end],
                                            'duration': [paretic_duration]})
                pareticEvent_df = pd.concat([pareticEvent_df, paretic_add],
                                            axis=0, ignore_index=True)
        # nonparetic side
        nonpareticEvent_df = pd.DataFrame(
            columns=["start time", "end time", "duration"])
        nonpar_flag = 0
        for nonpar_idx in np.arange(len(nonpareticData)):
            # start time
            if nonpareticData.iloc[nonpar_idx, 1] == 1:
                nonparetic_start = nonpareticData.iloc[nonpar_idx, 0]
                nonpar_flag = 1
            elif (nonpareticData.iloc[nonpar_idx, 1] == 2) &\
                 (nonpar_flag == 1):
                nonparetic_end = nonpareticData.iloc[nonpar_idx, 0]
                nonpar_flag = 2
            else:
                pass
            if nonpar_flag == 2:
                nonparetic_duration = nonparetic_end - nonparetic_start
                nonparetic_add = pd.DataFrame(
                    {'start time': [nonparetic_start],
                     'end time': [nonparetic_end],
                     'duration': [nonparetic_duration]}
                    )
                nonpareticEvent_df = pd.concat(
                    [nonpareticEvent_df, nonparetic_add],
                    axis=0,
                    ignore_index=True
                    )
        # df update - L, R designation
        if self.paretic_side == 'L':
            self.leftTiming_df = pareticEvent_df
            self.rightTiming_df = nonpareticEvent_df
        else:
            self.leftTiming_df = nonpareticEvent_df
            self.rightTiming_df = pareticEvent_df

    def GRF_preprocessing(self):
        # stance phase만 골라서 GRF inference에 활용
        leftForce_df = pd.DataFrame()
        leftForce_df["time"] = self.leftTime
        left_df_add = pd.DataFrame(self.leftForce)
        leftForce_df = pd.concat([leftForce_df, left_df_add], axis=1,
                                 ignore_index=True)
        leftForce_df.columns = ["time",
                                "f1", "f2", "f3",
                                "f4", "f5", "f6"]
        rightForce_df = pd.DataFrame()
        rightForce_df["time"] = self.rightTime
        right_df_add = pd.DataFrame(self.rightForce)
        rightForce_df = pd.concat(
            [rightForce_df, right_df_add],
            axis=1,
            ignore_index=True
            )
        rightForce_df.columns = ["time",
                                 "f1", "f2", "f3",
                                 "f4", "f5", "f6"]
        # Synced df with respect to stance timing
        # left
        for left_idx in np.arange(len(self.leftTiming_df)):
            if left_idx == 0:
                SyncedLeftForce = leftForce_df[
                    (
                        leftForce_df.time
                        >= self.leftTiming_df.loc[left_idx, "start time"]
                    ) &
                    (
                        leftForce_df.time
                        <= self.leftTiming_df.loc[left_idx, "end time"]
                    )
                    ]
            else:
                SyncedLeftForce = pd.concat([SyncedLeftForce, leftForce_df[
                    (
                        leftForce_df.time
                        >= self.leftTiming_df.loc[left_idx, "start time"]
                    ) &
                    (
                        leftForce_df.time
                        <= self.leftTiming_df.loc[left_idx, "end time"]
                    )
                    ]], axis=0, ignore_index=True)
        self.SyncedLeftTime = SyncedLeftForce.iloc[:, 0]
        self.SyncedLeftTime.reset_index(drop=True, inplace=True)
        self.SyncedLeftForce = np.array(SyncedLeftForce.iloc[:, 1:])
        # right
        for right_idx in np.arange(len(self.rightTiming_df)):
            if right_idx == 0:
                SyncedRightForce = rightForce_df[
                    (
                        rightForce_df.time
                        >= self.rightTiming_df.loc[right_idx, "start time"]
                    ) &
                    (
                        rightForce_df.time
                        <= self.rightTiming_df.loc[right_idx, "end time"]
                    )
                    ]
            else:
                SyncedRightForce = pd.concat([SyncedRightForce, rightForce_df[
                    (
                        rightForce_df.time
                        >= self.rightTiming_df.loc[right_idx, "start time"]
                    ) &
                    (
                        rightForce_df.time
                        <= self.rightTiming_df.loc[right_idx, "end time"]
                    )
                    ]], axis=0, ignore_index=True)
        self.SyncedRightTime = SyncedRightForce.iloc[:, 0]
        self.SyncedRightTime.reset_index(drop=True, inplace=True)
        self.SyncedRightForce = np.array(SyncedRightForce.iloc[:, 1:])

    def GRF_model_load(self):
        GRFmodel = LSTM_GRF(input_size=6, hidden_size_1=self.GRF_input_length,
                            hidden_size_2=20, hidden_size_3=15,
                            num_layers=2, drop_p=0.1)
        GRFmodel.load_state_dict(torch.load(self.modelPathGRF,
                                            map_location=self.device))
        self.GRFmodel = GRFmodel

    def LSTM_GRF_transform(self, idx, sensor_dir):
        if sensor_dir == 'L':
            if (int(idx) + 1) < self.GRF_input_length:
                for x_num in np.arange(self.sensor_num):
                    if x_num == 0:
                        x = np.append(
                            self.SyncedLeftForce[0, x_num]
                            * np.ones(
                                (1, self.GRF_input_length - int(idx) - 1)
                                ),
                            self.SyncedLeftForce[0:int(idx) + 1, x_num])
                        x = x.reshape(self.GRF_input_length, 1)
                    else:
                        x_temp = np.append(
                            self.SyncedLeftForce[0, x_num]
                            * np.ones(
                                (1, self.GRF_input_length - int(idx) - 1)
                                ),
                            self.SyncedLeftForce[0:int(idx) + 1, x_num])
                        x_temp = x_temp.reshape(self.GRF_input_length, 1)
                        x = np.append(x, x_temp, axis=1)
            else:
                x = self.SyncedLeftForce[
                    int(idx) + 1 - self.GRF_input_length:int(idx) + 1, :]
        else:
            if (int(idx) + 1) < self.GRF_input_length:
                for x_num in np.arange(self.sensor_num):
                    if x_num == 0:
                        x = np.append(
                            self.SyncedRightForce[0, x_num]
                            * np.ones(
                                (1, self.GRF_input_length - int(idx) - 1)
                                ),
                            self.SyncedRightForce[0:int(idx) + 1, x_num])
                        x = x.reshape(self.GRF_input_length, 1)
                    else:
                        x_temp = np.append(
                            self.SyncedRightForce[0, x_num]
                            * np.ones(
                                (1, self.GRF_input_length - int(idx) - 1)
                                ),
                            self.SyncedRightForce[0:int(idx) + 1, x_num])
                        x_temp = x_temp.reshape(self.GRF_input_length, 1)
                        x = np.append(x, x_temp, axis=1)
            else:
                x = self.SyncedRightForce[
                    int(idx) + 1 - self.GRF_input_length:int(idx) + 1, :]
        x = torch.from_numpy(x)
        x = torch.stack([x], dim=0).float()

        return x

    def GRF_one_prediction(self, idx, sensor_dir):
        model = self.GRFmodel
        model.eval()
        with torch.no_grad():
            x = self.LSTM_GRF_transform(idx, sensor_dir)
            output = model(x).detach().cpu().numpy()
            output = output.squeeze(0)

        return output

    def GRF_total_prediction(self):
        self.GRFleftData = np.array([])
        self.GRFrightData = np.array([])
        # LEFT
        for g_ind in np.arange(len(self.SyncedLeftForce)):
            # prediction
            leftOutput = self.GRF_one_prediction(idx=g_ind, sensor_dir='L')
            # post-processing
            leftOutput /= np.sum(self.SyncedLeftForce[g_ind])
            leftOutput *= (self.BW * 9.807)
            self.GRFleftData = np.append(self.GRFleftData,
                                         leftOutput,
                                         axis=0)
        # RIGHT
        for g_ind in np.arange(len(self.SyncedRightForce)):
            # prediction
            rightOutput = self.GRF_one_prediction(idx=g_ind, sensor_dir='R')
            # post-processing
            rightOutput /= np.sum(self.SyncedRightForce[g_ind])
            rightOutput *= (self.BW * 9.807)
            self.GRFrightData = np.append(
                self.GRFrightData, rightOutput, axis=0)

    def GRF_trimmed_by_gait(self):
        # trimmed with respect to stance timing
        self.leftGRFtrimmed = pd.DataFrame(columns=['impulse', 'maximum'])
        self.rightGRFtrimmed = pd.DataFrame(columns=['impulse', 'maximum'])
        # left
        for left_idx in np.arange(len(self.leftTiming_df)):
            # index determination
            index_list = list(
                self.SyncedLeftTime[
                    (
                        self.SyncedLeftTime
                        >= self.leftTiming_df.loc[left_idx, "start time"]
                    ) &
                    (
                        self.SyncedLeftTime
                        <= self.leftTiming_df.loc[left_idx, "end time"]
                    )
                    ].index)
            # impulse calculation
            GRF_leftImpulse = 0
            for L_idx in index_list[:-1]:
                delta_time = self.SyncedLeftTime.loc[L_idx+1] -\
                    self.SyncedLeftTime.loc[L_idx]
                GRF_value = float(self.GRFleftData[L_idx])
                GRF_leftImpulse += delta_time * GRF_value
            # maximum calculation
            GRF_leftMaximum = max(self.GRFleftData[index_list])
            # df update
            left_update_df = pd.DataFrame({'impulse': [GRF_leftImpulse],
                                           'maximum': [GRF_leftMaximum]})
            self.leftGRFtrimmed = pd.concat([self.leftGRFtrimmed,
                                             left_update_df], axis=0,
                                            ignore_index=True)
        # right
        for right_idx in np.arange(len(self.rightTiming_df)):
            # index determination
            index_list = list(
                self.SyncedRightTime[
                    (
                        self.SyncedRightTime
                        >= self.rightTiming_df.loc[right_idx, "start time"]
                    ) &
                    (
                        self.SyncedRightTime
                        <= self.rightTiming_df.loc[right_idx, "end time"]
                    )
                    ].index)
            # impulse calculation
            GRF_rightImpulse = 0
            for R_idx in index_list[:-1]:
                delta_time = self.SyncedRightTime.loc[R_idx+1] -\
                    self.SyncedRightTime.loc[R_idx]
                GRF_value = float(self.GRFrightData[R_idx])
                GRF_rightImpulse += delta_time * GRF_value
            # maximum calculation
            GRF_rightMaximum = max(self.GRFrightData[index_list])
            # df update
            right_update_df = pd.DataFrame({'impulse': [GRF_rightImpulse],
                                           'maximum': [GRF_rightMaximum]})
            self.rightGRFtrimmed = pd.concat([self.rightGRFtrimmed,
                                             right_update_df], axis=0,
                                             ignore_index=True)


class ClinicalIndexMH:
    @staticmethod
    def get_symmetry_index_stanceTime(start_time,
                                      pareticPath, nonpareticPath,
                                      paretic_side='L'):
        ST = GRF_predictor(start_time=float(start_time),
                           pareticPath=pareticPath,
                           nonpareticPath=nonpareticPath,
                           paretic_side=paretic_side)
        if paretic_side == 'L':
            pareticTiming = ST.leftTiming_df
            nonpareticTiming = ST.rightTiming_df
        else:
            nonpareticTiming = ST.leftTiming_df
            pareticTiming = ST.rightTiming_df
        # Calculation
        mean_paretic = np.mean(pareticTiming.duration)
        std_paretic = np.std(pareticTiming.duration)
        mean_non_paretic = np.mean(nonpareticTiming.duration)
        std_non_paretic = np.std(nonpareticTiming.duration)
        # mean based calculation
        symmetry = mean_paretic / (mean_paretic + mean_non_paretic) * 100

        return [mean_paretic, std_paretic,
                mean_non_paretic, std_non_paretic, symmetry]

    @staticmethod
    def get_data_stance_time(
            start_time, leftPath, rightPath,
            pareticPath, nonpareticPath,
            modelPathCalib, modelPathGRF,
            size='280', paretic_side='L', BW=float(100)):
        ST = GRF_predictor(start_time=float(start_time),
                           leftPath=leftPath,
                           rightPath=rightPath,
                           pareticPath=pareticPath,
                           nonpareticPath=nonpareticPath,
                           modelPathCalib=modelPathCalib,
                           modelPathGRF=modelPathGRF,
                           size=size,
                           paretic_side=paretic_side,
                           BW=BW)
        if paretic_side == 'L':
            pareticTiming = ST.leftTiming_df
            nonpareticTiming = ST.rightTiming_df
        else:
            nonpareticTiming = ST.leftTiming_df
            pareticTiming = ST.rightTiming_df

        return np.array(pareticTiming.duration),\
               np.array(nonpareticTiming.duration)

    @staticmethod
    def get_symmetry_index_GRFmax(start_time, leftPath, rightPath,
                                  pareticPath, nonpareticPath,
                                  modelPathCalib, modelPathGRF,
                                  size='280', paretic_side='L',
                                  BW=float(100)):
        grf_max = GRF_predictor(start_time=float(start_time),
                                leftPath=leftPath,
                                rightPath=rightPath,
                                pareticPath=pareticPath,
                                nonpareticPath=nonpareticPath,
                                modelPathCalib=modelPathCalib,
                                modelPathGRF=modelPathGRF,
                                size=size,
                                paretic_side=paretic_side,
                                BW=BW)
        grf_max.GRF_initialization()
        grf_max.GRF_trimmed_by_gait()

        if paretic_side == 'L':
            pareticMaximum = grf_max.leftGRFtrimmed
            nonpareticMaximum = grf_max.rightGRFtrimmed
        else:
            nonpareticMaximum = grf_max.leftGRFtrimmed
            pareticMaximum = grf_max.rightGRFtrimmed
        # Calculation
        mean_paretic = np.mean(pareticMaximum.maximum)
        std_paretic = np.std(pareticMaximum.maximum)
        mean_non_paretic = np.mean(nonpareticMaximum.maximum)
        std_non_paretic = np.std(nonpareticMaximum.maximum)
        # mean based calculation
        symmetry = mean_paretic / (mean_paretic + mean_non_paretic) * 100

        return [mean_paretic, std_paretic,
                mean_non_paretic, std_non_paretic, symmetry]

    @staticmethod
    def get_data_GRFmax(
            start_time, leftPath, rightPath,
            pareticPath, nonpareticPath,
            modelPathCalib, modelPathGRF,
            size='280', paretic_side='L', BW=float(100)):
        grf_max = GRF_predictor(start_time=float(start_time),
                                leftPath=leftPath,
                                rightPath=rightPath,
                                pareticPath=pareticPath,
                                nonpareticPath=nonpareticPath,
                                modelPathCalib=modelPathCalib,
                                modelPathGRF=modelPathGRF,
                                size=size,
                                paretic_side=paretic_side,
                                BW=BW)
        grf_max.GRF_initialization()
        grf_max.GRF_trimmed_by_gait()

        if paretic_side == 'L':
            pareticMaximum = grf_max.leftGRFtrimmed
            nonpareticMaximum = grf_max.rightGRFtrimmed
        else:
            nonpareticMaximum = grf_max.leftGRFtrimmed
            pareticMaximum = grf_max.rightGRFtrimmed

        return np.array(pareticMaximum.maximum),\
               np.array(nonpareticMaximum.maximum)

    @staticmethod
    def get_symmetry_index_GRFimpulse(start_time, leftPath, rightPath,
                                      pareticPath, nonpareticPath,
                                      modelPathCalib, modelPathGRF,
                                      size='280', paretic_side='L',
                                      BW=float(100)):
        grf_impulse = GRF_predictor(start_time=float(start_time),
                                    leftPath=leftPath,
                                    rightPath=rightPath,
                                    pareticPath=pareticPath,
                                    nonpareticPath=nonpareticPath,
                                    modelPathCalib=modelPathCalib,
                                    modelPathGRF=modelPathGRF,
                                    size=size,
                                    paretic_side=paretic_side,
                                    BW=BW)
        grf_impulse.GRF_initialization()
        grf_impulse.GRF_trimmed_by_gait()

        if paretic_side == 'L':
            pareticImpulse = grf_impulse.leftGRFtrimmed
            nonpareticImpulse = grf_impulse.rightGRFtrimmed
        else:
            nonpareticImpulse = grf_impulse.leftGRFtrimmed
            pareticImpulse = grf_impulse.rightGRFtrimmed
        # Calculation
        mean_paretic = np.mean(pareticImpulse.impulse)
        std_paretic = np.std(pareticImpulse.impulse)
        mean_non_paretic = np.mean(nonpareticImpulse.impulse)
        std_non_paretic = np.std(nonpareticImpulse.impulse)
        # mean based calculation
        symmetry = mean_paretic / (mean_paretic + mean_non_paretic) * 100

        return [mean_paretic, std_paretic,
                mean_non_paretic, std_non_paretic, symmetry]

    @staticmethod
    def get_data_GRFimpulse(
            start_time, leftPath, rightPath,
            pareticPath, nonpareticPath,
            modelPathCalib, modelPathGRF,
            size='280', paretic_side='L', BW=float(100)):
        grf_impulse = GRF_predictor(start_time=float(start_time),
                                    leftPath=leftPath,
                                    rightPath=rightPath,
                                    pareticPath=pareticPath,
                                    nonpareticPath=nonpareticPath,
                                    modelPathCalib=modelPathCalib,
                                    modelPathGRF=modelPathGRF,
                                    size=size,
                                    paretic_side=paretic_side,
                                    BW=BW)
        grf_impulse.GRF_initialization()
        grf_impulse.GRF_trimmed_by_gait()

        if paretic_side == 'L':
            pareticImpulse = grf_impulse.leftGRFtrimmed
            nonpareticImpulse = grf_impulse.rightGRFtrimmed
        else:
            nonpareticImpulse = grf_impulse.leftGRFtrimmed
            pareticImpulse = grf_impulse.rightGRFtrimmed

        return np.array(pareticImpulse.impulse),\
               np.array(nonpareticImpulse.impulse)

    @staticmethod
    def save_raw_data(
            start_time: float, left_path: str, right_path: str,
            paretic_path: str, nonparetic_path: str,
            model_path_calib: str, model_path_GRF: str,
            shoe_size: str, paretic_side: str, body_weight: float,
            save_folder: str, session_name: str
            ):

        grf_class = GRF_predictor(
            start_time=start_time,
            leftPath=left_path,
            rightPath=right_path,
            pareticPath=paretic_path,
            nonpareticPath=nonparetic_path,
            modelPathCalib=model_path_calib,
            modelPathGRF=model_path_GRF,
            size=shoe_size,
            paretic_side=paretic_side,
            BW=body_weight)

        # raw data - voltage, force, GRF
        raw_voltage_left, raw_voltage_right =\
            grf_class.get_voltage_raw_data()
        raw_force_left, raw_force_right =\
            grf_class.get_force_raw_data()
        raw_GRF_left, raw_GRF_right =\
            grf_class.get_GRF_raw_data()

        # raw data saving
        raw_voltage_left.to_csv(
            save_folder + 'RAW_VOLTAGE_LEFT_' + session_name + '.csv',
            sep=",", header=True, index=False
            )
        raw_voltage_right.to_csv(
            save_folder + 'RAW_VOLTAGE_RIGHT_' + session_name + '.csv',
            sep=",", header=True, index=False
            )
        raw_force_left.to_csv(
            save_folder + 'RAW_FORCE_LEFT_' + session_name + '.csv',
            sep=",", header=True, index=False
            )
        raw_force_right.to_csv(
            save_folder + 'RAW_FORCE_RIGHT_' + session_name + '.csv',
            sep=",", header=True, index=False
            )
        raw_GRF_left.to_csv(
            save_folder + 'RAW_GRF_LEFT_' + session_name + '.csv',
            sep=",", header=True, index=False
            )
        raw_GRF_right.to_csv(
            save_folder + 'RAW_GRF_RIGHT_' + session_name + '.csv',
            sep=",", header=True, index=False
            )

    @staticmethod
    def make_plot(
            session_name: str, base_path: str,
            cycle_data_path: str, raw_data_path: str,
            paretic_side: str,
            paretic_color='red',
            nonparetic_color='blue',
            std_alpha=0.35
            ):
        # GRF raw data plot
        raw_report_path = base_path + raw_data_path + session_name + "/"
        raw_left_path_list, raw_left_name_list =\
            folder_path_name(
                raw_report_path,
                option='start',
                char='RAW_GRF_LEFT',
                T_F=1
                )
        raw_right_path_list, raw_right_name_list =\
            folder_path_name(
                raw_report_path,
                option='start',
                char='RAW_GRF_RIGHT',
                T_F=1
                )

        if paretic_side == 'L':
            raw_paretic_path = raw_left_path_list[0]
            raw_nonparetic_path = raw_right_path_list[0]
        elif paretic_side == 'R':
            raw_paretic_path = raw_right_path_list[0]
            raw_nonparetic_path = raw_left_path_list[0]

        df_raw_paretic = pd.read_csv(
            raw_paretic_path,
            header=0,
            delimiter=",")
        df_raw_paretic.columns = ["time", "grf"]
        df_raw_nonparetic = pd.read_csv(
            raw_nonparetic_path,
            header=0,
            delimiter=",")
        df_raw_nonparetic.columns = ["time", "grf"]

        paretic_start_time = gait_start_time(
            df_raw_paretic, "time", "grf"
            )
        paretic_end_time = gait_end_time(
            df_raw_paretic, "time", "grf"
            )
        nonparetic_start_time = gait_start_time(
            df_raw_nonparetic, "time", "grf"
            )
        nonparetic_end_time = gait_end_time(
            df_raw_nonparetic, "time", "grf"
            )

        paretic_mean_time_series, paretic_std_time_series =\
            DataProcess.average_time_series(
                x=np.array(df_raw_paretic.time),
                y=np.array(df_raw_paretic.grf),
                x_start=paretic_start_time,
                x_end=paretic_end_time)
        nonparetic_mean_time_series, nonparetic_std_time_series =\
            DataProcess.average_time_series(
                x=np.array(df_raw_nonparetic.time),
                y=np.array(df_raw_nonparetic.grf),
                x_start=nonparetic_start_time,
                x_end=nonparetic_end_time)

        # Only paretic
        fig1 = plt.figure(figsize=(18, 8))
        ax = fig1.gca()
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        plt.plot(
            np.linspace(0, 100, 101),
            paretic_mean_time_series,
            color=paretic_color,
            label='paretic side')
        plt.fill_between(
            np.linspace(0, 100, 101),
            paretic_mean_time_series - paretic_std_time_series,
            paretic_mean_time_series + paretic_std_time_series,
            color=paretic_color,
            alpha=std_alpha)
        plt.xlabel("Gait Cycle [%]", fontsize=30)
        plt.ylabel("Vertical Ground Reaction Force [N]", fontsize=30)
        plt.xticks([0, 20, 40, 60, 80, 100], fontsize=25)
        plt.yticks(
            [-200, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600], fontsize=25)
        plt.xlim([0, 100])
        plt.ylim([-200, 1600])
        plt.legend(loc='upper left', fontsize=25)
        plt.show()
        fig1.savefig(raw_report_path + 'paretic_side_vGRF.png')

        # Only nonparetic
        fig2 = plt.figure(figsize=(18, 8))
        bx = fig2.gca()
        for axis in ['top', 'bottom', 'left', 'right']:
            bx.spines[axis].set_linewidth(2)
        plt.plot(
            np.linspace(0, 100, 101),
            nonparetic_mean_time_series,
            color=nonparetic_color,
            label='non-paretic side')
        plt.fill_between(
            np.linspace(0, 100, 101),
            nonparetic_mean_time_series - nonparetic_std_time_series,
            nonparetic_mean_time_series + nonparetic_std_time_series,
            color=nonparetic_color,
            alpha=std_alpha)
        plt.xlabel("Gait Cycle [%]", fontsize=30)
        plt.ylabel("Vertical Ground Reaction Force [N]", fontsize=30)
        plt.xticks([0, 20, 40, 60, 80, 100], fontsize=25)
        plt.yticks(
            [-200, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600], fontsize=25)
        plt.xlim([0, 100])
        plt.ylim([-200, 1600])
        plt.legend(loc='upper left', fontsize=25)
        plt.show()
        fig2.savefig(raw_report_path + 'non-paretic_side_vGRF.png')

        # Comparison
        fig3 = plt.figure(figsize=(18, 8))
        cx = fig3.gca()
        for axis in ['top', 'bottom', 'left', 'right']:
            cx.spines[axis].set_linewidth(2)
        plt.plot(
            np.linspace(0, 100, 101),
            paretic_mean_time_series,
            color=paretic_color,
            label='paretic side')
        plt.fill_between(
            np.linspace(0, 100, 101),
            paretic_mean_time_series - paretic_std_time_series,
            paretic_mean_time_series + paretic_std_time_series,
            color=paretic_color,
            alpha=std_alpha)
        plt.plot(
            np.linspace(0, 100, 101),
            nonparetic_mean_time_series,
            color=nonparetic_color,
            label='non-paretic side')
        plt.fill_between(
            np.linspace(0, 100, 101),
            nonparetic_mean_time_series - nonparetic_std_time_series,
            nonparetic_mean_time_series + nonparetic_std_time_series,
            color=nonparetic_color,
            alpha=std_alpha)
        plt.xlabel("Gait Cycle [%]", fontsize=30)
        plt.ylabel("Vertical Ground Reaction Force [N]", fontsize=30)
        plt.xticks([0, 20, 40, 60, 80, 100], fontsize=25)
        plt.yticks(
            [-200, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600], fontsize=25)
        plt.xlim([0, 100])
        plt.ylim([-200, 1600])
        plt.legend(loc='upper left', fontsize=25)
        plt.show()
        fig3.savefig(raw_report_path + 'comparison_vGRF.png')

        ######################################################################
        ######################################################################
        # gait cycle plot
        cycle_report_path = base_path + cycle_data_path + session_name + "/"
        # stance time
        df_cycle_paretic_stance_time = pd.read_csv(
            cycle_report_path + 'paretic_stance_time.csv',
            header=None,
            delimiter=",")
        df_cycle_nonparetic_stance_time = pd.read_csv(
            cycle_report_path + 'non-paretic_stance_time.csv',
            header=None,
            delimiter=",")
        # GRF maximum
        df_cycle_paretic_GRF_maximum = pd.read_csv(
            cycle_report_path + 'paretic_grf_max.csv',
            header=None,
            delimiter=",")
        df_cycle_nonparetic_GRF_maximum = pd.read_csv(
            cycle_report_path + 'non-paretic_grf_max.csv',
            header=None,
            delimiter=",")
        # GRF impulse
        df_cycle_paretic_GRF_impulse = pd.read_csv(
            cycle_report_path + 'paretic_grf_impulse.csv',
            header=None,
            delimiter=",")
        df_cycle_nonparetic_GRF_impulse = pd.read_csv(
            cycle_report_path + 'non-paretic_grf_impulse.csv',
            header=None,
            delimiter=",")

        # stance time plot
        fig4 = plt.figure(figsize=(18, 8))
        dx = fig4.gca()
        for axis in ['top', 'bottom', 'left', 'right']:
            dx.spines[axis].set_linewidth(2)
        plt.plot(
            df_cycle_paretic_stance_time.iloc[:, 0],
            color=paretic_color,
            label='paretic side')
        plt.plot(
            df_cycle_nonparetic_stance_time.iloc[:, 0],
            color=nonparetic_color,
            label='non-paretic side')
        plt.xlabel("Gait Cycle Number", fontsize=30)
        plt.ylabel("Stance time [sec]", fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.ylim([0, 2.5])
        plt.legend(loc='upper left', fontsize=25)
        plt.show()
        fig4.savefig(cycle_report_path + 'stance_time.png')

        # GRF maximum
        fig5 = plt.figure(figsize=(18, 8))
        ex = fig5.gca()
        for axis in ['top', 'bottom', 'left', 'right']:
            ex.spines[axis].set_linewidth(2)
        plt.plot(
            df_cycle_paretic_GRF_maximum.iloc[:, 0],
            color=paretic_color,
            label='paretic side')
        plt.plot(
            df_cycle_nonparetic_GRF_maximum.iloc[:, 0],
            color=nonparetic_color,
            label='non-paretic side')
        plt.xlabel("Gait Cycle Number", fontsize=30)
        plt.ylabel("vGRF Maximum [N]", fontsize=30)
        plt.title("Vertical Ground Reaction Force", fontsize=45)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.ylim([0, 1500])
        plt.legend(loc='upper left', fontsize=25)
        plt.show()
        fig5.savefig(cycle_report_path + 'GRF_maximum.png')

        # GRF impulse
        fig6 = plt.figure(figsize=(18, 8))
        fx = fig6.gca()
        for axis in ['top', 'bottom', 'left', 'right']:
            fx.spines[axis].set_linewidth(2)
        plt.plot(
            df_cycle_paretic_GRF_impulse.iloc[:, 0],
            color=paretic_color,
            label='paretic side')
        plt.plot(
            df_cycle_nonparetic_GRF_impulse.iloc[:, 0],
            color=nonparetic_color,
            label='non-paretic side')
        plt.xlabel("Gait Cycle Number", fontsize=30)
        plt.ylabel("vGRF Impulse [N sec]", fontsize=30)
        plt.title("Vertical Ground Reaction Force", fontsize=45)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.ylim([0, 2000])
        plt.legend(loc='upper left', fontsize=25)
        plt.show()
        fig6.savefig(cycle_report_path + 'GRF_impulse.png')


def df_for_raw_GRF_make_plot(
        session_name: str, base_path: str,
        raw_data_path: str, paretic_side: str
        ):
    # GRF raw data
    raw_report_path = base_path + raw_data_path + session_name + "/"
    raw_left_path_list, raw_left_name_list =\
        folder_path_name(
            raw_report_path,
            option='start',
            char='RAW_GRF_LEFT',
            T_F=1
            )
    raw_right_path_list, raw_right_name_list =\
        folder_path_name(
            raw_report_path,
            option='start',
            char='RAW_GRF_RIGHT',
            T_F=1
            )

    if paretic_side == 'L':
        paretic_raw_data_path = raw_left_path_list[0]
        nonparetic_raw_data_path = raw_right_path_list[0]
    elif paretic_side == 'R':
        paretic_raw_data_path = raw_right_path_list[0]
        nonparetic_raw_data_path = raw_left_path_list[0]

    df_raw_paretic = pd.read_csv(
        paretic_raw_data_path,
        header=0,
        delimiter=",")
    df_raw_paretic.columns = ["time", "grf"]
    df_raw_nonparetic = pd.read_csv(
        nonparetic_raw_data_path,
        header=0,
        delimiter=",")
    df_raw_nonparetic.columns = ["time", "grf"]

    return df_raw_paretic, df_raw_nonparetic


def gait_start_time(df, time_column_name: str, column_name: str):
    start_time_array = np.array([])
    # swing: 0, stance: 1
    phase_flag = 0
    for idx in np.arange(len(df)):
        if (df.loc[idx, column_name] != 0) & (phase_flag == 0):
            start_time = df.loc[idx, time_column_name]
            start_time_array = np.append(start_time_array, start_time)
            phase_flag = 1
        elif (df.loc[idx, column_name] == 0) & (phase_flag == 1):
            phase_flag = 0
        else:
            pass

    return start_time_array


def gait_end_time(df, time_column_name: str, column_name: str):
    end_time_array = np.array([])
    # swing: 0, stance: 1
    phase_flag = 0
    for idx in np.arange(len(df)):
        if (df.loc[idx, column_name] != 0) & (phase_flag == 0):
            if idx == 0:
                continue
            end_time = df.loc[idx - 1, time_column_name]
            end_time_array = np.append(end_time_array, end_time)
            phase_flag = 1
        elif (df.loc[idx, column_name] == 0) & (phase_flag == 1):
            phase_flag = 0
        else:
            pass

    # Dismiss the first end time
    end_time_array = end_time_array[1:]

    # Add the last end time
    end_time_array = np.append(
        end_time_array,
        df.loc[len(df) - 1, time_column_name])

    return end_time_array


##############################################################################
# TEST CODE for MAIN
##############################################################################
def basic_test():
    grf = GRF_predictor()
    # Sensor data reading, model loading
    grf.GRF_initialization()
    grf.GRF_trimmed_by_gait()


def stanceTime_test():
    CI = ClinicalIndexMH()
    mean_paretic, std_paretic, mean_non_paretic, std_non_paretic, symmetry =\
        CI.get_symmetry_index_stanceTime(
            start_time=float(1691133979.160155),
            pareticPath='../../bag/log_2023-08-04-16-26-19'
            + '/afo_detector-gait_paretic.csv',
            nonpareticPath='../../bag/log_2023-08-04-16-26-19'
            + '/afo_detector-gait_nonparetic.csv',
            paretic_side='L')
    print(mean_paretic)
    print(std_paretic)
    print(mean_non_paretic)
    print(std_non_paretic)
    print(symmetry)


def GRFimpulse_test():
    CI = ClinicalIndexMH()
    mean_paretic, std_paretic, mean_non_paretic, std_non_paretic, symmetry =\
        CI.get_symmetry_index_GRFimpulse(
            start_time=float(1691133979.160155),
            leftPath='../../bag/log_2023-08-04-16-26-19'
            + '/afo_sensor-soleSensor_left.csv',
            rightPath='../../bag/log_2023-08-04-16-26-19'
            + '/afo_sensor-soleSensor_right.csv',
            pareticPath='../../bag/log_2023-08-04-16-26-19'
            + '/afo_detector-gait_paretic.csv',
            nonpareticPath='../../bag/log_2023-08-04-16-26-19'
            + '/afo_detector-gait_nonparetic.csv',
            modelPathCalib='../../model/CHAR_230815_280_LP/',
            modelPathGRF='../../model/GRF_230815/LSTM_GRF.pt',
            size='280',
            paretic_side='L',
            BW=float(100)
            )
    print(mean_paretic)
    print(std_paretic)
    print(mean_non_paretic)
    print(std_non_paretic)
    print(symmetry)


def GRFmax_test():
    CI = ClinicalIndexMH()
    mean_paretic, std_paretic, mean_non_paretic, std_non_paretic, symmetry =\
        CI.get_symmetry_index_GRFmax(
            start_time=float(1691133979.160155),
            leftPath='../../bag/log_2023-08-04-16-26-19'
            + '/afo_sensor-soleSensor_left.csv',
            rightPath='../../bag/log_2023-08-04-16-26-19'
            + '/afo_sensor-soleSensor_right.csv',
            pareticPath='../../bag/log_2023-08-04-16-26-19'
            + '/afo_detector-gait_paretic.csv',
            nonpareticPath='../../bag/log_2023-08-04-16-26-19'
            + '/afo_detector-gait_nonparetic.csv',
            modelPathCalib='../../model/CHAR_230815_280_LP/',
            modelPathGRF='../../model/GRF_230815/LSTM_GRF.pt',
            size='280',
            paretic_side='L',
            BW=float(100)
            )
    print(mean_paretic)
    print(std_paretic)
    print(mean_non_paretic)
    print(std_non_paretic)
    print(symmetry)

###############################################################################
###############################################################################


def Rawdata_saving(
        bag_name: str, session_name: str,
        calib_folder_name: str, GRF_model_name: str,
        shoe_size: str, paretic_side: str, body_weight: float
        ):
    # Read bag file
    # example_path = ../../bag/log_2023-08-16-15-40-08.bag
    path = '../../bag/' + bag_name + '.bag'
    # bag_name[4:-9] = 2023-08-16
    save_folder = '../../data/report/' + bag_name[4:-9] + '/'
    create_folder(save_folder)
    # example_path = ../../data/report/2023-08-16/10MWT_OFF_CueOFF.csv
    # save_path = save_folder + session_name + '.csv'

    bag = bagreader(path)
    start_time = bag.start_time

    # To filter specific topics with interests
    TOPIC_MH = (
        "/afo_sensor/soleSensor_left",
        "/afo_sensor/soleSensor_right",
        "/afo_detector/gait_paretic",
        "/afo_detector/gait_nonparetic"
    )

    # Definition of Clinical Indices
    left_sole_path = ""
    right_sole_path = ""
    paretic_gait_path = ""
    nonparetic_gait_path = ""

    # calib_model_path example = ../../model/CHAR_230815_280_LP/
    # GRF_model_path example = ../../model/GRF_230815/LSTM_GRF.pt
    calib_model_path = '../../model/' + calib_folder_name + '/'
    GRF_model_path = '../../model/' + GRF_model_name
    # 85.1
    BW = body_weight
    # 'L'
    paretic_side = paretic_side
    # '280'
    size = shoe_size

    # Read Topics and calculate clinical index
    for topic in bag.topics:
        msg_topic = bag.message_by_topic(topic)

        # Use own module and methods
        if topic == TOPIC_MH[0]:
            left_sole_path = msg_topic
        elif topic == TOPIC_MH[1]:
            right_sole_path = msg_topic
        elif topic == TOPIC_MH[2]:
            paretic_gait_path = msg_topic
        elif topic == TOPIC_MH[3]:
            nonparetic_gait_path = msg_topic

    # Raw data saving
    CI = ClinicalIndexMH()
    CI.save_raw_data(
        start_time=start_time,
        left_path=left_sole_path,
        right_path=right_sole_path,
        paretic_path=paretic_gait_path,
        nonparetic_path=nonparetic_gait_path,
        model_path_calib=calib_model_path,
        model_path_GRF=GRF_model_path,
        shoe_size=size,
        paretic_side=paretic_side,
        body_weight=BW,
        save_folder=save_folder,
        session_name=session_name
        )

def Gaitcycledata_saving(
        base_path: str, cycle_data_path: str,
        bag_name: str, session_name: str,
        calib_folder_name: str, GRF_model_name: str,
        shoe_size: str, paretic_side: str, body_weight: float
        ):
    # Read bag file
    path = base_path + 'bag/' + bag_name + '.bag'
    # bag_name[4:-9] = 2023-08-16
    save_folder = base_path + cycle_data_path + session_name + '/'
    create_folder(save_folder)

    bag = bagreader(path)
    start_time = bag.start_time

    # To filter specific topics with interests
    TOPIC_MH = (
        "/afo_sensor/soleSensor_left",
        "/afo_sensor/soleSensor_right",
        "/afo_detector/gait_paretic",
        "/afo_detector/gait_nonparetic"
    )

    # Definition of Clinical Indices
    left_sole_path = ""
    right_sole_path = ""
    paretic_gait_path = ""
    nonparetic_gait_path = ""

    # calib_model_path example = ../../model/CHAR_230815_280_LP/
    # GRF_model_path example = ../../model/GRF_230815/LSTM_GRF.pt
    calib_model_path = '../../model/' + calib_folder_name + '/'
    GRF_model_path = '../../model/' + GRF_model_name
    # 85.1
    BW = body_weight
    # 'L'
    paretic_side = paretic_side
    # '280'
    size = shoe_size

    # Read Topics and calculate clinical index
    for topic in bag.topics:
        msg_topic = bag.message_by_topic(topic)

        # Use own module and methods
        if topic == TOPIC_MH[0]:
            left_sole_path = msg_topic
        elif topic == TOPIC_MH[1]:
            right_sole_path = msg_topic
        elif topic == TOPIC_MH[2]:
            paretic_gait_path = msg_topic
        elif topic == TOPIC_MH[3]:
            nonparetic_gait_path = msg_topic

    # gait cycle data saving
    stance_time_paretic_array, stance_time_nonparetic_array =\
        ClinicalIndexMH.get_data_stance_time(
            start_time=start_time,
            leftPath=left_sole_path,
            rightPath=right_sole_path,
            pareticPath=paretic_gait_path,
            nonpareticPath=nonparetic_gait_path,
            modelPathCalib=calib_model_path,
            modelPathGRF=GRF_model_path,
            size=size,
            paretic_side=paretic_side,
            BW=BW)

    pd.DataFrame(stance_time_paretic_array).to_csv(
        save_folder + "paretic_stance_time.csv",
        header=False, index=False)
    pd.DataFrame(stance_time_nonparetic_array).to_csv(
        save_folder + "non-paretic_stance_time.csv",
        header=False, index=False)

    grf_max_paretic_array, grf_max_nonparetic_array =\
        ClinicalIndexMH.get_data_GRFmax(
            start_time=start_time,
            leftPath=left_sole_path,
            rightPath=right_sole_path,
            pareticPath=paretic_gait_path,
            nonpareticPath=nonparetic_gait_path,
            modelPathCalib=calib_model_path,
            modelPathGRF=GRF_model_path,
            size=size,
            paretic_side=paretic_side,
            BW=BW)

    pd.DataFrame(grf_max_paretic_array).to_csv(
        save_folder + "paretic_grf_max.csv",
        header=False, index=False)
    pd.DataFrame(grf_max_nonparetic_array).to_csv(
        save_folder + "non-paretic_grf_max.csv",
        header=False, index=False)

    grf_impulse_paretic_array, grf_impulse_nonparetic_array =\
        ClinicalIndexMH.get_data_GRFimpulse(
            start_time=start_time,
            leftPath=left_sole_path,
            rightPath=right_sole_path,
            pareticPath=paretic_gait_path,
            nonpareticPath=nonparetic_gait_path,
            modelPathCalib=calib_model_path,
            modelPathGRF=GRF_model_path,
            size=size,
            paretic_side=paretic_side,
            BW=BW)

    pd.DataFrame(grf_impulse_paretic_array).to_csv(
        save_folder + "paretic_grf_impulse.csv",
        header=False, index=False)
    pd.DataFrame(grf_impulse_nonparetic_array).to_csv(
        save_folder + "non-paretic_grf_impulse.csv",
        header=False, index=False)


def raw_data_save_test():
    bag_list = [
        "log_2023-08-16-15-40-08",
        "log_2023-08-16-15-41-19",
        "log_2023-08-16-15-43-10",
        "log_2023-08-16-15-44-06",
        "log_2023-08-16-15-46-14",
        "log_2023-08-16-15-51-39",
        "log_2023-08-16-15-56-59",
        "log_2023-08-16-16-02-17",
        "log_2023-08-16-16-23-57",
        "log_2023-08-16-16-27-28",
        "log_2023-08-16-16-29-04"
        ]
    session_list = [
        "10MWT_OFF_CueOFF",
        "10MWT_ON_CueOFF",
        "10MWT_ON_CueON_1",
        "10MWT_ON_CueON_2",
        "10MWT_ON_CueON_3",
        "2MWT_OFF_CueOFF_89.4m",
        "2MWT_ON_CueOFF_88.2m",
        "2MWT_ON_CueON_64.2m",
        "2MWT_BARE_CueOFF_90m",
        "10MWT_BARE_CueOFF_1",
        "10MWT_BARE_CueOFF_2"
        ]

    base_path =\
        'D:/OneDrive - SNU/범부처-DESKTOP-2JL44HH/C_임상_2023/2023-08-16/'
    cycle_data_path = 'report_CYCLE/'
    raw_data_path = 'report_RAW/'
    calib_folder_name = 'CHAR_230815_280_LP'
    GRF_model_name = 'GRF_230815/LSTM_GRF.pt'
    # 장비 무게 포함
    body_weight = 85.1
    paretic_side = 'L'
    shoe_size = '280'

    for bag_ind in np.arange(len(bag_list)):

        bag_name = bag_list[bag_ind]
        session_name = session_list[bag_ind]

        Rawdata_saving(
            bag_name=bag_name,
            session_name=session_name,
            calib_folder_name=calib_folder_name,
            GRF_model_name=GRF_model_name,
            shoe_size=shoe_size,
            paretic_side=paretic_side,
            body_weight=body_weight
            )

        Gaitcycledata_saving(
            base_path=base_path,
            cycle_data_path=cycle_data_path,
            bag_name=bag_name,
            session_name=session_name,
            calib_folder_name=calib_folder_name,
            GRF_model_name=GRF_model_name,
            shoe_size=shoe_size,
            paretic_side=paretic_side,
            body_weight=body_weight
            )


def make_plot_test():
    # bag_list = [
    #     "log_2023-08-16-15-40-08",
    #     "log_2023-08-16-15-41-19",
    #     "log_2023-08-16-15-43-10",
    #     "log_2023-08-16-15-44-06",
    #     "log_2023-08-16-15-46-14",
    #     "log_2023-08-16-15-51-39",
    #     "log_2023-08-16-15-56-59",
    #     "log_2023-08-16-16-02-17",
    #     "log_2023-08-16-16-23-57",
    #     "log_2023-08-16-16-27-28",
    #     "log_2023-08-16-16-29-04"
    #     ]
    # session_list = [
    #     "10MWT_OFF_CueOFF",
    #     "10MWT_ON_CueOFF",
    #     "10MWT_ON_CueON_1",
    #     "10MWT_ON_CueON_2",
    #     "10MWT_ON_CueON_3",
    #     "2MWT_OFF_CueOFF_89.4m",
    #     "2MWT_ON_CueOFF_88.2m",
    #     "2MWT_ON_CueON_64.2m",
    #     "2MWT_BARE_CueOFF_90m",
    #     "10MWT_BARE_CueOFF_1",
    #     "10MWT_BARE_CueOFF_2"
    #     ]
    bag_list = [
        "log_2023-08-16-15-56-59",
        "log_2023-08-16-16-02-17"
        ]
    session_list = [
        "2MWT_ON_CueOFF_88.2m",
        "2MWT_ON_CueON_64.2m"
        ]

    base_path =\
        'D:/OneDrive - SNU/범부처-DESKTOP-2JL44HH/C_임상_2023/2023-08-16/'
    cycle_data_path = 'report_CYCLE/'
    raw_data_path = 'report_RAW/'
    calib_folder_name = 'CHAR_230815_280_LP'
    GRF_model_name = 'GRF_230815/LSTM_GRF.pt'
    # 장비 무게 포함
    body_weight = 85.1
    paretic_side = 'L'
    shoe_size = '280'

    for bag_ind in np.arange(len(bag_list)):

        bag_name = bag_list[bag_ind]
        session_name = session_list[bag_ind]

        ClinicalIndexMH.make_plot(
            session_name=session_name,
            base_path=base_path,
            cycle_data_path=cycle_data_path,
            raw_data_path=raw_data_path,
            paretic_side='L',
            paretic_color='red',
            nonparetic_color='blue',
            std_alpha=0.35
            )


def main():
    bag_list = [
        "log_2023-08-16-15-56-59",
        "log_2023-08-16-16-02-17"
        ]
    session_list = [
        "2MWT_ON_CueOFF_88.2m",
        "2MWT_ON_CueON_64.2m"
        ]

    base_path =\
        'D:/OneDrive - SNU/범부처-DESKTOP-2JL44HH/C_임상_2023/2023-08-16/'
    cycle_data_path = 'report_CYCLE/'
    raw_data_path = 'report_RAW/'
    calib_folder_name = 'CHAR_230815_280_LP'
    GRF_model_name = 'GRF_230815/LSTM_GRF.pt'
    # 장비 무게 포함
    body_weight = 85.1
    paretic_side = 'L'
    shoe_size = '280'

    for bag_ind in np.arange(len(bag_list)):

        bag_name = bag_list[bag_ind]
        session_name = session_list[bag_ind]

        # Read bag file
        path = base_path + 'bag/' + bag_name + '.bag'
        # bag_name[4:-9] = 2023-08-16
        save_folder = base_path + raw_data_path + session_name + '/'
        create_folder(save_folder)

        bag = bagreader(path)
        start_time = bag.start_time

        # To filter specific topics with interests
        TOPIC_MH = (
            "/afo_sensor/soleSensor_left",
            "/afo_sensor/soleSensor_right",
            "/afo_detector/gait_paretic",
            "/afo_detector/gait_nonparetic"
        )

        # Definition of Clinical Indices
        left_sole_path = ""
        right_sole_path = ""
        paretic_gait_path = ""
        nonparetic_gait_path = ""

        # calib_model_path example = ../../model/CHAR_230815_280_LP/
        # GRF_model_path example = ../../model/GRF_230815/LSTM_GRF.pt
        calib_model_path = '../../model/' + calib_folder_name + '/'
        GRF_model_path = '../../model/' + GRF_model_name
        # 85.1
        BW = body_weight
        # 'L'
        paretic_side = paretic_side
        # '280'
        size = shoe_size

        # Read Topics and calculate clinical index
        for topic in bag.topics:
            msg_topic = bag.message_by_topic(topic)

            # Use own module and methods
            if topic == TOPIC_MH[0]:
                left_sole_path = msg_topic
            elif topic == TOPIC_MH[1]:
                right_sole_path = msg_topic
            elif topic == TOPIC_MH[2]:
                paretic_gait_path = msg_topic
            elif topic == TOPIC_MH[3]:
                nonparetic_gait_path = msg_topic
        
        # gait phase df read
        gait_phase_paretic = pd.read_csv(paretic_gait_path, header=0)
        gait_phase_nonparetic = pd.read_csv(nonparetic_gait_path, header=0)
        # Time initialization with respect to start time
        gait_phase_paretic = gait_phase_paretic.astype(float)
        gait_phase_nonparetic = gait_phase_nonparetic.astype(float)
        gait_phase_paretic.Time -= start_time
        gait_phase_nonparetic.Time -= start_time
        # column change
        gait_phase_paretic.columns = ["time", "grf"]
        gait_phase_nonparetic.columns = ["time", "grf"]
        
        # For raw GRF plot
        df_paretic_raw_data, df_nonparetic_raw_data =\
            df_for_raw_GRF_make_plot(
                session_name=session_name, base_path=base_path,
                raw_data_path=raw_data_path, paretic_side=paretic_side
                )
        collection_paretic_raw_grf = \
            divider_data_by_gait_phase_df(df_paretic_raw_data,
                                            gait_phase_paretic, "grf")
        collection_nonparetic_raw_grf = \
            divider_data_by_gait_phase_df(df_nonparetic_raw_data,
                                            gait_phase_nonparetic, "grf")

        graph_both_cycle_data(collection_paretic_raw_grf,
                              collection_nonparetic_raw_grf,
                              gait_phase_paretic, gait_phase_nonparetic,
                              data_gait_label="grf",
                              x_num=101)


if __name__ == "__main__":
    main()
