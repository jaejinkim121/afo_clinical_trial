import glob
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

from utils import *


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
                 save_path: str,
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
        # sample save path:
        # D://OneDrive - SNU/범부처-DESKTOP-2JL44HH/C_임상_2023/
        # 2023-08-16/report_RAW/2MWT_BARE_CueOFF_90m/
        self.save_path = save_path
        self.device = torch.device('cpu')
        self.size = str(size)
        self.sensor_num = int(6)
        self.calib_input_length = int(16)
        self.GRF_input_length = int(25)
        self.save_path_read()
        # Read gait detection data
        # 이 기능은 후에 class 밖으로 빼자
        # self.Gait_detection_data_read()

    def get_time_data(self):
        return np.array(self.leftTime), np.array(self.rightTime)

    def get_voltage_raw_data_modify(self):
        self.left_voltage = self.leftData
        self.right_voltage = self.rightData

    # def get_voltage_raw_data(self):
    #     # Read vout data
    #     self.vout_data_read()
    #     # Calib model loading
    #     self.force_model_load()
    #     # Calibration model inference
    #     self.Calib_total_prediction()
    #     self.GRF_preprocessing()

    #     left_stance_time = pd.DataFrame(
    #         self.leftTime.isin(self.SyncedLeftTime) * 1.0
    #         )
    #     right_stance_time = pd.DataFrame(
    #         self.rightTime.isin(self.SyncedRightTime) * 1.0
    #         )

    #     left_stance_time.columns = ["stance time"]
    #     right_stance_time.columns = ["stance time"]

    #     left_voltage = pd.concat([self.leftData, left_stance_time], axis=1)
    #     right_voltage = pd.concat([self.rightData, right_stance_time], axis=1)

    #     return left_voltage, right_voltage

    # def get_force_raw_data(self):
    #     # Read vout data
    #     self.vout_data_read()
    #     # Calib model loading
    #     self.force_model_load()
    #     # Calibration model inference
    #     self.Calib_total_prediction()
    #     self.GRF_preprocessing()

    #     left_stance_time = pd.DataFrame(
    #         self.leftTime.isin(self.SyncedLeftTime) * 1.0
    #         )
    #     right_stance_time = pd.DataFrame(
    #         self.rightTime.isin(self.SyncedRightTime) * 1.0
    #         )

    #     left_stance_time.columns = ["stance time"]
    #     right_stance_time.columns = ["stance time"]

    #     left_force = pd.concat(
    #         [self.leftTime, pd.DataFrame(self.leftForce), left_stance_time],
    #         axis=1
    #         )
    #     right_force = pd.concat(
    #         [self.rightTime, pd.DataFrame(self.rightForce), right_stance_time],
    #         axis=1
    #         )

    #     return left_force, right_force

    def get_force_raw_data_modify(self):
        left_force_data = pd.DataFrame()
        left_force_data["time"] = self.leftTime
        left_force_data_add = pd.DataFrame(self.leftForce)
        left_force_data = pd.concat(
            [left_force_data, left_force_data_add],
            axis=1, ignore_index=True
            )
        left_force_data.columns = [
            "time", "f1", "f2", "f3", "f4", "f5", "f6"
            ]
        right_force_data = pd.DataFrame()
        right_force_data["time"] = self.rightTime
        right_force_data_add = pd.DataFrame(self.rightForce)
        right_force_data = pd.concat(
            [right_force_data, right_force_data_add],
            axis=1, ignore_index=True
            )
        right_force_data.columns = [
            "time", "f1", "f2", "f3", "f4", "f5", "f6"
            ]
        self.left_force = left_force_data
        self.right_force = right_force_data

    # # Set GRF values of swing phase as zero
    # def get_GRF_raw_data(self):
    #     self.GRF_initialization()

    #     left_GRF_stance = pd.concat(
    #         [self.SyncedLeftTime, pd.DataFrame(self.GRFleftData)], axis=1
    #         )
    #     right_GRF_stance = pd.concat(
    #         [self.SyncedRightTime, pd.DataFrame(self.GRFrightData)], axis=1
    #         )

    #     left_GRF = self.leftTime.to_frame().merge(
    #         left_GRF_stance, how='left', on="time"
    #         )
    #     left_GRF = left_GRF.fillna(0)
    #     right_GRF = self.rightTime.to_frame().merge(
    #         right_GRF_stance, how='left', on="time"
    #         )
    #     right_GRF = right_GRF.fillna(0)

    #     return left_GRF, right_GRF

    def get_grf_raw_data_modify(self):
        left_grf_data = pd.DataFrame()
        left_grf_data["time"] = self.leftTime
        left_grf_data["grf"] = self.GRFleftData

        right_grf_data = pd.DataFrame()
        right_grf_data["time"] = self.rightTime
        right_grf_data["grf"] = self.GRFrightData

        self.left_grf = left_grf_data
        self.right_grf = right_grf_data

    def save_path_read(self):
        # Save path reading
        path_list, name_list = folder_path_name(
            self.save_path, option="start", char="RAW", T_F=1
            )
        if len(path_list) != 6:
            self.grf_initialization()
        else:
            self.saved_data_load(path_list, name_list)

    def saved_data_load(self, path_list, name_list):
        for num, name in enumerate(name_list):
            data_type = name[4:7]  # voltage, force, grf
            data_direction = name[-7:-4]
            if data_direction == 'EFT':  # left
                if data_type == 'VOL':
                    self.left_voltage = pd.read_csv(path_list[num], header=0)
                if data_type == 'FOR':
                    self.left_force = pd.read_csv(path_list[num], header=0)
                if data_type == 'GRF':
                    self.left_grf = pd.read_csv(path_list[num], header=0)
            else:
                if data_type == 'VOL':
                    self.right_voltage = pd.read_csv(path_list[num], header=0)
                if data_type == 'FOR':
                    self.right_force = pd.read_csv(path_list[num], header=0)
                if data_type == 'GRF':
                    self.right_grf = pd.read_csv(path_list[num], header=0)

    def grf_initialization(self):
        # Data process
        self.vout_data_read()
        self.force_model_load()
        self.Calib_total_prediction()
        self.GRF_model_load()
        self.GRF_total_prediction()

        # Data load for saving
        self.get_voltage_raw_data_modify()
        self.get_force_raw_data_modify()
        self.get_grf_raw_data_modify()

        # Save dataframe
        self.left_voltage.to_csv(
            self.save_path + 'RAW_VOLTAGE_LEFT.csv',
            sep=",", header=True, index=False
            )
        self.right_voltage.to_csv(
            self.save_path + 'RAW_VOLTAGE_RIGHT.csv',
            sep=",", header=True, index=False
            )
        self.left_force.to_csv(
            self.save_path + 'RAW_FORCE_%s_LEFT.csv' %
            self.modelPathCalib[-19:-1],  # CHAR_230815_280_LP
            sep=",", header=True, index=False
            )
        self.right_force.to_csv(
            self.save_path + 'RAW_FORCE_%s_RIGHT.csv' %
            self.modelPathCalib[-19:-1],  # CHAR_230815_280_LP
            sep=",", header=True, index=False
            )
        self.left_grf.to_csv(
            self.save_path + 'RAW_GRF_%s_LEFT.csv' %
            (self.modelPathGRF[-18:-12] + "_" +  # 230815_
             self.modelPathGRF[-11:-7]),  # LSTM or SVR_
            sep=",", header=True, index=False
            )
        self.right_grf.to_csv(
            self.save_path + 'RAW_GRF_%s_RIGHT.csv' %
            (self.modelPathGRF[-18:-12] + "_" +  # 230815_
             self.modelPathGRF[-11:-7]),  # LSTM or SVR_
            sep=",", header=True, index=False
            )

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
                                   hidden_size_2=30,  # previous: 45
                                   hidden_size_3=15,  # previous: 35
                                   num_layers=4,  # previous: 6
                                   drop_p=0.1)
            rightmodel = LSTM_Calib(hidden_size_1=self.calib_input_length,
                                    hidden_size_2=30,  # previous: 45
                                    hidden_size_3=15,  # previous: 35
                                    num_layers=4,  # previous: 6
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

    # def Gait_detection_data_read(self):
    #     # stance = 1, swing = 2
    #     # stance time: 1 -> 2
    #     pareticData = pd.read_csv(self.pareticPath, header=0)
    #     nonpareticData = pd.read_csv(self.nonpareticPath, header=0)
    #     # Time initialization with respect to start time
    #     pareticData = pareticData.astype(float)
    #     nonpareticData = nonpareticData.astype(float)
    #     pareticData.Time -= self.start_time
    #     nonpareticData.Time -= self.start_time
    #     # stance start timing, end timing, duration
    #     # paretic side
    #     pareticEvent_df = pd.DataFrame(
    #         columns=["start time", "end time", "duration"])
    #     par_flag = 0
    #     for par_idx in np.arange(len(pareticData)):
    #         # start time
    #         if pareticData.iloc[par_idx, 1] == 1:
    #             paretic_start = pareticData.iloc[par_idx, 0]
    #             par_flag = 1
    #         elif (pareticData.iloc[par_idx, 1] == 2) & (par_flag == 1):
    #             paretic_end = pareticData.iloc[par_idx, 0]
    #             par_flag = 2
    #         else:
    #             pass
    #         if par_flag == 2:
    #             paretic_duration = paretic_end - paretic_start
    #             paretic_add = pd.DataFrame({'start time': [paretic_start],
    #                                         'end time': [paretic_end],
    #                                         'duration': [paretic_duration]})
    #             pareticEvent_df = pd.concat([pareticEvent_df, paretic_add],
    #                                         axis=0, ignore_index=True)
    #     # nonparetic side
    #     nonpareticEvent_df = pd.DataFrame(
    #         columns=["start time", "end time", "duration"])
    #     nonpar_flag = 0
    #     for nonpar_idx in np.arange(len(nonpareticData)):
    #         # start time
    #         if nonpareticData.iloc[nonpar_idx, 1] == 1:
    #             nonparetic_start = nonpareticData.iloc[nonpar_idx, 0]
    #             nonpar_flag = 1
    #         elif (nonpareticData.iloc[nonpar_idx, 1] == 2) &\
    #              (nonpar_flag == 1):
    #             nonparetic_end = nonpareticData.iloc[nonpar_idx, 0]
    #             nonpar_flag = 2
    #         else:
    #             pass
    #         if nonpar_flag == 2:
    #             nonparetic_duration = nonparetic_end - nonparetic_start
    #             nonparetic_add = pd.DataFrame(
    #                 {'start time': [nonparetic_start],
    #                  'end time': [nonparetic_end],
    #                  'duration': [nonparetic_duration]}
    #                 )
    #             nonpareticEvent_df = pd.concat(
    #                 [nonpareticEvent_df, nonparetic_add],
    #                 axis=0,
    #                 ignore_index=True
    #                 )
    #     # df update - L, R designation
    #     if self.paretic_side == 'L':
    #         self.leftTiming_df = pareticEvent_df
    #         self.rightTiming_df = nonpareticEvent_df
    #     else:
    #         self.leftTiming_df = nonpareticEvent_df
    #         self.rightTiming_df = pareticEvent_df

    # def GRF_preprocessing(self):
    #     # stance phase만 골라서 GRF inference에 활용
    #     leftForce_df = pd.DataFrame()
    #     leftForce_df["time"] = self.leftTime
    #     left_df_add = pd.DataFrame(self.leftForce)
    #     leftForce_df = pd.concat([leftForce_df, left_df_add], axis=1,
    #                              ignore_index=True)
    #     leftForce_df.columns = ["time",
    #                             "f1", "f2", "f3",
    #                             "f4", "f5", "f6"]
    #     rightForce_df = pd.DataFrame()
    #     rightForce_df["time"] = self.rightTime
    #     right_df_add = pd.DataFrame(self.rightForce)
    #     rightForce_df = pd.concat(
    #         [rightForce_df, right_df_add],
    #         axis=1,
    #         ignore_index=True
    #         )
    #     rightForce_df.columns = ["time",
    #                              "f1", "f2", "f3",
    #                              "f4", "f5", "f6"]
    #     # Synced df with respect to stance timing
    #     # left
    #     for left_idx in np.arange(len(self.leftTiming_df)):
    #         if left_idx == 0:
    #             SyncedLeftForce = leftForce_df[
    #                 (
    #                     leftForce_df.time
    #                     >= self.leftTiming_df.loc[left_idx, "start time"]
    #                 ) &
    #                 (
    #                     leftForce_df.time
    #                     <= self.leftTiming_df.loc[left_idx, "end time"]
    #                 )
    #                 ]
    #         else:
    #             SyncedLeftForce = pd.concat([SyncedLeftForce, leftForce_df[
    #                 (
    #                     leftForce_df.time
    #                     >= self.leftTiming_df.loc[left_idx, "start time"]
    #                 ) &
    #                 (
    #                     leftForce_df.time
    #                     <= self.leftTiming_df.loc[left_idx, "end time"]
    #                 )
    #                 ]], axis=0, ignore_index=True)
    #     self.SyncedLeftTime = SyncedLeftForce.iloc[:, 0]
    #     self.SyncedLeftTime.reset_index(drop=True, inplace=True)
    #     self.SyncedLeftForce = np.array(SyncedLeftForce.iloc[:, 1:])
    #     # right
    #     for right_idx in np.arange(len(self.rightTiming_df)):
    #         if right_idx == 0:
    #             SyncedRightForce = rightForce_df[
    #                 (
    #                     rightForce_df.time
    #                     >= self.rightTiming_df.loc[right_idx, "start time"]
    #                 ) &
    #                 (
    #                     rightForce_df.time
    #                     <= self.rightTiming_df.loc[right_idx, "end time"]
    #                 )
    #                 ]
    #         else:
    #             SyncedRightForce = pd.concat([SyncedRightForce, rightForce_df[
    #                 (
    #                     rightForce_df.time
    #                     >= self.rightTiming_df.loc[right_idx, "start time"]
    #                 ) &
    #                 (
    #                     rightForce_df.time
    #                     <= self.rightTiming_df.loc[right_idx, "end time"]
    #                 )
    #                 ]], axis=0, ignore_index=True)
    #     self.SyncedRightTime = SyncedRightForce.iloc[:, 0]
    #     self.SyncedRightTime.reset_index(drop=True, inplace=True)
    #     self.SyncedRightForce = np.array(SyncedRightForce.iloc[:, 1:])

    def GRF_model_load(self):
        if self.modelPathGRF[-11:-7] == 'LSTM':
            GRFmodel = LSTM_GRF(input_size=6,
                                hidden_size_1=self.GRF_input_length,
                                hidden_size_2=20, hidden_size_3=15,
                                num_layers=2, drop_p=0.1)
            GRFmodel.load_state_dict(torch.load(self.modelPathGRF,
                                                map_location=self.device))
            self.GRFmodel = GRFmodel
        else:  # SVR_
            self.GRFmodel = joblib.load(self.modelPathGRF)

    def SVR_GRF_transform(self):
        if self.size == '280':
            xCOP_list = np.expand_dims(
                np.array([-0.15, 0.35, 0.15, 0.55, 0.45, 0.0]), axis=1)
            yCOP_list = np.expand_dims(
                np.array([1.43, 2.2, 1.54, 2.09, 1.54, 0.0]), axis=1)
        else:  # 260
            xCOP_list = np.expand_dims(
                np.array([-0.15, 0.25, 0.15, 0.45, 0.45, 0.0]), axis=1)
            yCOP_list = np.expand_dims(
                np.array([1.215, 1.985, 1.435, 1.875, 1.435, 0.0]), axis=1)

        left_sum = np.expand_dims(
            np.sum(np.array(self.leftForce), axis=1), axis=1)  # N*1
        left_xCOP = np.dot(np.array(self.leftForce), xCOP_list)  # N*1
        left_yCOP = np.dot(np.array(self.leftForce), yCOP_list)  # N*1
        left_xCOP /= left_sum  # N*1
        left_yCOP /= left_sum  # N*1
        left_prop = np.array(self.leftForce) / left_sum  # N*6
        self.GRF_leftinput = pd.DataFrame(np.concatenate(
            (left_prop, left_sum, left_xCOP, left_yCOP), axis=1))

        right_sum = np.expand_dims(
            np.sum(np.array(self.rightForce), axis=1), axis=1)  # N*1
        right_xCOP = np.dot(np.array(self.rightForce), xCOP_list)  # N*1
        right_yCOP = np.dot(np.array(self.rightForce), yCOP_list)  # N*1
        right_xCOP /= right_sum  # N*1
        right_yCOP /= right_sum  # N*1
        right_prop = np.array(self.rightForce) / right_sum  # N*6
        self.GRF_rightinput = pd.DataFrame(np.concatenate(
            (right_prop, right_sum, right_xCOP, right_yCOP), axis=1))

    def LSTM_GRF_transform(self, idx, sensor_dir):
        if sensor_dir == 'L':
            if (int(idx) + 1) < self.GRF_input_length:
                for x_num in np.arange(self.sensor_num):
                    if x_num == 0:
                        x = np.append(
                            np.array(self.leftForce)[0, x_num]
                            * np.ones(
                                (1, self.GRF_input_length - int(idx) - 1)
                                ),
                            np.array(self.leftForce)[0:int(idx) + 1, x_num])
                        x = x.reshape(self.GRF_input_length, 1)
                    else:
                        x_temp = np.append(
                            np.array(self.leftForce)[0, x_num]
                            * np.ones(
                                (1, self.GRF_input_length - int(idx) - 1)
                                ),
                            np.array(self.leftForce)[0:int(idx) + 1, x_num])
                        x_temp = x_temp.reshape(self.GRF_input_length, 1)
                        x = np.append(x, x_temp, axis=1)
            else:
                x = np.array(self.leftForce)[
                    int(idx) + 1 - self.GRF_input_length:int(idx) + 1, :]
        else:
            if (int(idx) + 1) < self.GRF_input_length:
                for x_num in np.arange(self.sensor_num):
                    if x_num == 0:
                        x = np.append(
                            np.array(self.rightForce)[0, x_num]
                            * np.ones(
                                (1, self.GRF_input_length - int(idx) - 1)
                                ),
                            np.array(self.rightForce)[0:int(idx) + 1, x_num])
                        x = x.reshape(self.GRF_input_length, 1)
                    else:
                        x_temp = np.append(
                            np.array(self.rightForce)[0, x_num]
                            * np.ones(
                                (1, self.GRF_input_length - int(idx) - 1)
                                ),
                            np.array(self.rightForce)[0:int(idx) + 1, x_num])
                        x_temp = x_temp.reshape(self.GRF_input_length, 1)
                        x = np.append(x, x_temp, axis=1)
            else:
                x = np.array(self.rightForce)[
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
        if self.modelPathGRF[-11:-7] == 'LSTM':
            self.GRFleftData = np.array([])
            self.GRFrightData = np.array([])
            # LEFT
            for g_ind in np.arange(len(np.array(self.leftForce))):
                # prediction
                leftOutput = self.GRF_one_prediction(
                    idx=g_ind, sensor_dir='L'
                    )
                # post-processing
                leftOutput /= np.sum(np.array(self.leftForce)[g_ind])
                leftOutput *= (self.BW * 9.807)
                self.GRFleftData = np.append(self.GRFleftData,
                                             leftOutput,
                                             axis=0)
            # RIGHT
            for g_ind in np.arange(len(np.array(self.rightForce))):
                # prediction
                rightOutput = self.GRF_one_prediction(
                    idx=g_ind, sensor_dir='R'
                    )
                # post-processing
                rightOutput /= np.sum(np.array(self.rightForce)[g_ind])
                rightOutput *= (self.BW * 9.807)
                self.GRFrightData = np.append(
                    self.GRFrightData, rightOutput, axis=0)
        else:  # SVR_
            # preprocessing
            self.SVR_GRF_transform()

            self.GRFleftData = np.array(
                self.GRFmodel.predict(self.GRF_leftinput)) *\
                (self.BW * 9.807)
            self.GRFrightData = np.array(
                self.GRFmodel.predict(self.GRF_rightinput)) *\
                (self.BW * 9.807)

    # def GRF_trimmed_by_gait(self):
    #     # trimmed with respect to stance timing
    #     self.leftGRFtrimmed = pd.DataFrame(columns=['impulse', 'maximum'])
    #     self.rightGRFtrimmed = pd.DataFrame(columns=['impulse', 'maximum'])
    #     # left
    #     for left_idx in np.arange(len(self.leftTiming_df)):
    #         # index determination
    #         index_list = list(
    #             self.SyncedLeftTime[
    #                 (
    #                     self.SyncedLeftTime
    #                     >= self.leftTiming_df.loc[left_idx, "start time"]
    #                 ) &
    #                 (
    #                     self.SyncedLeftTime
    #                     <= self.leftTiming_df.loc[left_idx, "end time"]
    #                 )
    #                 ].index)
    #         # impulse calculation
    #         GRF_leftImpulse = 0
    #         for L_idx in index_list[:-1]:
    #             delta_time = self.SyncedLeftTime.loc[L_idx+1] -\
    #                 self.SyncedLeftTime.loc[L_idx]
    #             GRF_value = float(self.GRFleftData[L_idx])
    #             GRF_leftImpulse += delta_time * GRF_value
    #         # maximum calculation
    #         GRF_leftMaximum = max(self.GRFleftData[index_list])
    #         # df update
    #         left_update_df = pd.DataFrame({'impulse': [GRF_leftImpulse],
    #                                        'maximum': [GRF_leftMaximum]})
    #         self.leftGRFtrimmed = pd.concat([self.leftGRFtrimmed,
    #                                          left_update_df], axis=0,
    #                                         ignore_index=True)
    #     # right
    #     for right_idx in np.arange(len(self.rightTiming_df)):
    #         # index determination
    #         index_list = list(
    #             self.SyncedRightTime[
    #                 (
    #                     self.SyncedRightTime
    #                     >= self.rightTiming_df.loc[right_idx, "start time"]
    #                 ) &
    #                 (
    #                     self.SyncedRightTime
    #                     <= self.rightTiming_df.loc[right_idx, "end time"]
    #                 )
    #                 ].index)
    #         # impulse calculation
    #         GRF_rightImpulse = 0
    #         for R_idx in index_list[:-1]:
    #             delta_time = self.SyncedRightTime.loc[R_idx+1] -\
    #                 self.SyncedRightTime.loc[R_idx]
    #             GRF_value = float(self.GRFrightData[R_idx])
    #             GRF_rightImpulse += delta_time * GRF_value
    #         # maximum calculation
    #         GRF_rightMaximum = max(self.GRFrightData[index_list])
    #         # df update
    #         right_update_df = pd.DataFrame({'impulse': [GRF_rightImpulse],
    #                                        'maximum': [GRF_rightMaximum]})
    #         self.rightGRFtrimmed = pd.concat([self.rightGRFtrimmed,
    #                                          right_update_df], axis=0,
    #                                          ignore_index=True)


##
# GRF_predictor class에 save_path 추가해서 쭉 돌려놓기!
class ClinicalIndexMH:
    @staticmethod
    def get_symmetry_index_grf(
            start_time, left_path, right_path,
            paretic_path, non_paretic_path,
            model_path_calib, model_path_grf,
            raw_data_save_path, cycle_timeseries_data_save_path,
            size='280', paretic_side='L',
            body_weight=float(100), ignore_cycle=(None, None)
    ):
        grf_class = GRF_predictor(
            start_time=float(start_time),
            leftPath=left_path,
            rightPath=right_path,
            pareticPath=paretic_path,
            nonpareticPath=non_paretic_path,
            modelPathCalib=model_path_calib,
            modelPathGRF=model_path_grf,
            save_path=raw_data_save_path,
            size=size,
            paretic_side=paretic_side,
            BW=body_weight
            )
        if paretic_side == 'L':
            paretic_data = grf_class.left_grf
            non_paretic_data = grf_class.right_grf
        else:
            paretic_data = grf_class.right_grf
            non_paretic_data = grf_class.left_grf

        max_array, impulse_array = DataProcess.data_process(
            paretic_data,
            non_paretic_data,
            paretic_path,
            non_paretic_path,
            save_path=cycle_timeseries_data_save_path,
            data_label="grf [N]",
            title_label="GRF",
            ignore_cycle=ignore_cycle,
            start_time=start_time,
            max_flag=True,
            impulse_flag=True
            )

        return max_array, impulse_array

    @staticmethod
    def get_symmetry_index_stanceTime(start_time,
                                      paretic_path, non_paretic_path,
                                      ignore_cycle=(None, None)):

        df_paretic_gait = DataProcess.read_data_file_by_path(
            paretic_path)
        df_non_paretic_gait = DataProcess.read_data_file_by_path(
            non_paretic_path)

        df_paretic_gait.iloc[:, 0] -= start_time
        df_non_paretic_gait.iloc[:, 0] -= start_time

        df_paretic_gait = \
            get_ignored_cycle(df_paretic_gait, ignore_cycle)
        df_non_paretic_gait = \
            get_ignored_cycle(df_non_paretic_gait, ignore_cycle)

        [_, _, _, _, _, _,
         paretic_mean, paretic_stdev,
         non_paretic_mean, non_paretic_stdev] =\
            DataProcess.gait_phase_pre_processing(
                df_paretic_gait, df_non_paretic_gait)

        symmetry = paretic_mean / (paretic_mean + non_paretic_mean) * 100

        return [paretic_mean, paretic_stdev,
                non_paretic_mean, non_paretic_stdev, symmetry]


if __name__ == "__main__":
    start_run_time = time.time()
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
    calib_folder_name = 'CHAR_230815_280_LP'
    GRF_model_name = 'GRF_230815/LSTM_GRF.pt'
    # 장비 무게 포함
    body_weight = 85.1
    paretic_side = 'L'
    size = '280'

    for bag_ind in np.arange(len(bag_list)):

        bag_name = bag_list[bag_ind]
        session_name = session_list[bag_ind]

        # Read bag file
        path = base_path + 'bag/' + bag_name + '.bag'
        # bag_name[4:-9] = 2023-08-16
        save_folder = base_path + 'report/' + session_name + '/'
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
        non_paretic_gait_path = ""

        # calib_model_path example = ../../model/CHAR_230815_280_LP/
        # GRF_model_path example = ../../model/GRF_230815/LSTM_GRF.pt
        calib_model_path = '../../model/' + calib_folder_name + '/'
        GRF_model_path = '../../model/' + GRF_model_name

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
                non_paretic_gait_path = msg_topic

        raw_data_path = '../../data/report/2023-08-16/' + session_name + '/'
        cycle_timeseries_data_save_path =\
            '../../graph/2023-08-16/' + session_name + '/'
        # create directory
        create_folder(raw_data_path)
        create_folder(cycle_timeseries_data_save_path)

        start_inference_time = time.time()

        ClinicalIndexMH.get_symmetry_index_grf(
            start_time=float(start_time),
            left_path=left_sole_path,
            right_path=right_sole_path,
            paretic_path=paretic_gait_path,
            non_paretic_path=non_paretic_gait_path,
            model_path_calib=calib_model_path,
            model_path_grf=GRF_model_path,
            raw_data_save_path=raw_data_path,
            cycle_timeseries_data_save_path=cycle_timeseries_data_save_path,
            size=size,
            paretic_side=paretic_side,
            body_weight=body_weight,
            ignore_cycle=(None, None)
            )
        # grf_class = GRF_predictor(
        #     start_time=float(start_time),
        #     leftPath=left_sole_path,
        #     rightPath=right_sole_path,
        #     pareticPath=paretic_gait_path,
        #     nonpareticPath=non_paretic_gait_path,
        #     modelPathCalib=calib_model_path,
        #     modelPathGRF=GRF_model_path,
        #     save_path=save_folder,
        #     size=size,
        #     paretic_side=paretic_side,
        #     BW=body_weight
        #     )
        end_inference_time = time.time()
        print("Inference time: ", end_inference_time - start_inference_time)
    end_run_time = time.time()
    print("Running time: ", end_run_time - start_run_time)

    # raw data plotting

