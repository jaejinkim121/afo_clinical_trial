# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:14:22 2022

@author: mleem
"""

import json
import numpy as np
import pandas as pd
import glob
import os
import joblib
import time


def basic_info_save_json():
    file_path = "../../../data/basic_info.json"
    basic_info = {
        "RH-02": {
            "size": 260,
            "affected side": "left",
            "stage": "chronic",
            "left cane": False,
            "right cane": False
            },
        "RH-03": {
            "size": 270,
            "affected side": "right",
            "stage": "chronic",
            "left cane": False,
            "right cane": False
            },
        "RH-04": {
            "size": 270,
            "affected side": "left",
            "stage": "chronic",
            "left cane": False,
            "right cane": False
            },
        "RH-05": {
            "size": 260,
            "affected side": "right",
            "stage": "acute",
            "left cane": False,
            "right cane": False
            },
        "RH-06": {
            "size": 250,
            "affected side": "right",
            "stage": "chronic",
            "left cane": False,
            "right cane": True
            },
        "RH-07": {
            "size": 270,
            "affected side": "right",
            "stage": "chronic",
            "left cane": False,
            "right cane": False
            },
        "RH-08": {
            "size": 260,
            "affected side": "right",
            "stage": "chronic",
            "left cane": True,
            "right cane": False
            },
        "RH-09": {
            "size": 290,
            "affected side": "left",
            "stage": "chronic",
            "left cane": False,
            "right cane": False
            },
        "RH-10": {
            "size": 280,
            "affected side": "right",
            "stage": "chronic",
            "left cane": False,
            "right cane": False
            }
        }

    with open(file_path, "w", encoding='utf-8') as file:
        json.dump(basic_info, file)

    return 0


def sole_header_info_save_json():
    file_path = "../../../data/sole_header_info.json"
    sole_header = {
        "RH-02": {
            "left": [
                "L lateral calcaneal tuberosity",
                "L fifth metatarsal head",
                "L second toe",
                "L third metatarsal head",
                "L first toe",
                "L first metatarsal head",
                "L medial calcaneal tuberosity",
                "L calcaneal tuberosity"
                ],
            "right": [
                "R medial calcaneal tuberosity",
                "R fifth metatarsal head",
                "R second toe",
                "R third metatarsal head",
                "R first toe",
                "R first metatarsal head",
                "R lateral calcaneal tuberosity",
                "R calcaneal tuberosity"
                ]
            },
        "RH-03": {
            "left": [
                "L lateral calcaneal tuberosity",
                "L fifth metatarsal head",
                "L third metatarsal head",
                "L second toe",
                "L first toe",
                "L first metatarsal head",
                "L medial calcaneal tuberosity",
                "L calcaneal tuberosity"
                ],
            "right": [
                "R lateral calcaneal tuberosity",
                "R fifth metatarsal head",
                "R third metatarsal head",
                "R second toe",
                "R first toe",
                "R first metatarsal head",
                "R medial calcaneal tuberosity",
                "R calcaneal tuberosity"
                ]
            },
        "RH-04": {
            "left": [
                "L lateral calcaneal tuberosity",
                "L fifth metatarsal head",
                "L third metatarsal head",
                "L second toe",
                "L first toe",
                "L first metatarsal head",
                "L medial calcaneal tuberosity",
                "L calcaneal tuberosity"
                ],
            "right": [
                "R lateral calcaneal tuberosity",
                "R fifth metatarsal head",
                "R third metatarsal head",
                "R second toe",
                "R first toe",
                "R first metatarsal head",
                "R medial calcaneal tuberosity",
                "R calcaneal tuberosity"
                ]
            },
        "RH-05": {
            "left": [
                "L lateral calcaneal tuberosity",
                "L fifth metatarsal head",
                "L second toe",
                "L third metatarsal head",
                "L first toe",
                "L first metatarsal head",
                "L medial calcaneal tuberosity",
                "L calcaneal tuberosity"
                ],
            "right": [
                "R medial calcaneal tuberosity",
                "R fifth metatarsal head",
                "R second toe",
                "R third metatarsal head",
                "R first toe",
                "R first metatarsal head",
                "R lateral calcaneal tuberosity",
                "R calcaneal tuberosity"
                ]
            },
        "RH-06": {
            "left": [
                "L lateral calcaneal tuberosity",
                "L fifth metatarsal head",
                "L second toe",
                "L third metatarsal head",
                "L first toe",
                "L first metatarsal head",
                "L medial calcaneal tuberosity",
                "L calcaneal tuberosity"
                ],
            "right": [
                "R lateral calcaneal tuberosity",
                "R fifth metatarsal head",
                "R second toe",
                "R third metatarsal head",
                "R first toe",
                "R first metatarsal head",
                "R medial calcaneal tuberosity",
                "R calcaneal tuberosity"
                ]
            },
        "RH-07": {
            "left": [
                "L second toe",
                "L third metatarsal head",
                "L fifth metatarsal head",
                "L lateral calcaneal tuberosity",
                "L calcaneal tuberosity",
                "L medial calcaneal tuberosity",
                "L first metatarsal head",
                "L first toe"
                ],
            "right": [
                "R second toe",
                "R third metatarsal head",
                "R fifth metatarsal head",
                "R lateral calcaneal tuberosity",
                "R calcaneal tuberosity",
                "R medial calcaneal tuberosity",
                "R first metatarsal head",
                "R first toe"
                ]
            },
        "RH-08": {
            "left": [
                "L third metatarsal head",
                "L second toe",
                "L fifth metatarsal head",
                "L lateral calcaneal tuberosity",
                "L calcaneal tuberosity",
                "L medial calcaneal tuberosity",
                "L first metatarsal head",
                "L first toe"
                ],
            "right": [
                "R third metatarsal head",
                "R second toe",
                "R fifth metatarsal head",
                "R lateral calcaneal tuberosity",
                "R calcaneal tuberosity",
                "R medial calcaneal tuberosity",
                "R first metatarsal head",
                "R first toe"
                ]
            },
        "RH-09": {
            "left": [
                "L second toe",
                "L third metatarsal head",
                "L fifth metatarsal head",
                "L lateral calcaneal tuberosity",
                "L calcaneal tuberosity",
                "L medial calcaneal tuberosity",
                "L first metatarsal head",
                "L first toe"
                ],
            "right": [
                "R third metatarsal head",
                "R second toe",
                "R fifth metatarsal head",
                "R lateral calcaneal tuberosity",
                "R calcaneal tuberosity",
                "R medial calcaneal tuberosity",
                "R first metatarsal head",
                "R first toe"
                ]
            },
        "RH-10": {
            "left": [
                "L second toe",
                "L third metatarsal head",
                "L fifth metatarsal head",
                "L lateral calcaneal tuberosity",
                "L calcaneal tuberosity",
                "L medial calcaneal tuberosity",
                "L first metatarsal head",
                "L first toe"
                ],
            "right": [
                "R third metatarsal head",
                "R second toe",
                "R fifth metatarsal head",
                "R lateral calcaneal tuberosity",
                "R calcaneal tuberosity",
                "R medial calcaneal tuberosity",
                "R first metatarsal head",
                "R first toe"
                ]
            }
        }

    with open(file_path, "w", encoding='utf-8') as file:
        json.dump(sole_header, file)

    return 0


def folder_path_name(path, start_or_end=None, char=None, T_F=None):

    folder_name_path = str(path)
    folder_path = folder_name_path + "*"

    file_list = glob.glob(folder_path)
    file_name_list = os.listdir(folder_name_path)

    if start_or_end == "start":
        exp_list = [file for (file, name) in zip(file_list, file_name_list)
                    if name.startswith(str(char)) == int(T_F)]
        exp_name_list = [name for (file, name) in
                         zip(file_list, file_name_list)
                         if name.startswith(str(char)) == int(T_F)]

    elif start_or_end == "end":
        exp_list = [file for (file, name) in zip(file_list, file_name_list)
                    if name.endswith(str(char)) == int(T_F)]
        exp_name_list = [name for (file, name)
                         in zip(file_list, file_name_list)
                         if name.endswith(str(char)) == int(T_F)]

    else:
        exp_list = [file for (file, name) in zip(file_list, file_name_list)]
        exp_name_list = [name for (file, name)
                         in zip(file_list, file_name_list)]

    exp_list = [file.replace('\\', '/') for file in exp_list]

    return exp_list, exp_name_list


def force_sensor_sync(force_sync_path, sensor_sync_path, RH_num, walk_num):

    # read sync csv file
    df_force_sync = pd.read_csv(force_sync_path, header=0)
    df_sensor_sync = pd.read_csv(sensor_sync_path, header=0)

    # start time: GRF time 0
    force_start_time = df_force_sync.loc[
        (df_force_sync["RH_num"] == "RH-" + str(RH_num)) &
        (df_force_sync["number"] == int(walk_num))]["time start"].values[0]

    # sensor start time (L, R)
    # Left
    L_sensor_start_time = df_sensor_sync.loc[
        (df_sensor_sync["RH_num"] == "RH-" + str(RH_num)) &
        (df_sensor_sync["R or L"] == "L") &
        (df_sensor_sync["number"] == int(walk_num))]["time start"].values[0]
    L_sensor_start_time -= force_start_time
    # Right
    R_sensor_start_time = df_sensor_sync.loc[
        (df_sensor_sync["RH_num"] == "RH-" + str(RH_num)) &
        (df_sensor_sync["R or L"] == "R") &
        (df_sensor_sync["number"] == int(walk_num))]["time start"].values[0]
    R_sensor_start_time -= force_start_time

    return force_start_time, L_sensor_start_time, R_sensor_start_time


def load_GRF(path, walk_num):
    # Find GRF path of given walk_num
    (GRF_path, _) = folder_path_name(
        path, "end", "WALK%s.XLS" % str(walk_num).zfill(2), T_F=1
        )
    # GRF data reading
    GRF_data = pd.read_csv(GRF_path[0], delimiter="\t", header=23)
    GRF_data = GRF_data.astype(float)
    # GRF column labelling: FWD, LAT, VRT
    index_data = list(GRF_data.index)
    df_index = pd.DataFrame(index_data, columns=["time"])
    df_index = (df_index * 5) / 600
    end_time = df_index.iloc[-1, 0]
    R_GRF_data = GRF_data[["R_GRF_FWD", "R_GRF_LAT", "R_GRF_VRT"]]
    L_GRF_data = GRF_data[["L_GRF_FWD", "L_GRF_LAT", "L_GRF_VRT"]]
    # Total GRF dataframe
    walk_GRF = pd.concat([df_index, L_GRF_data], axis=1)
    walk_GRF = pd.concat([walk_GRF, R_GRF_data], axis=1)

    return walk_GRF, end_time


def read_SENSOR_vol(path, RH_num):
    # data reading ##################################################
    data = pd.read_csv(path, sep=" |,", header=None)

    data = data.astype(float)
    data = data.dropna(axis=1)
    data = data.dropna(axis=0)
    data.columns = range(data.columns.size)
    data.reset_index(drop=True, inplace=True)

    data = data[np.arange(data.columns[-10], data.columns[-1]+1, 1)]
    data.columns = range(data.columns.size)

    if str(RH_num) == "00":
        data.columns = ['time', 0, 1, 2, 3, 4, 5, 6, 7]
        data.reset_index(drop=True, inplace=True)

    else:
        data.columns = ['sync', 'time', 0, 1, 2, 3, 4, 5, 6, 7]
        data.reset_index(drop=True, inplace=True)

    return data


def convert_sole_header(sole_header_path, save_flag=False):
    # save json (basic info, sole_header)
    if save_flag:
        basic_info_save_json()
        sole_header_info_save_json()
    # sole header json reading
    with open(sole_header_path, 'r') as file:
        sole_header = json.load(file)

    return sole_header


def load_SENSOR_vol(
        path, force_sync_path, sensor_sync_path,
        RH_num, walk_num, sole_header, GRF_end_time
        ):
    # load sync time
    (force_start_time,
     L_sensor_start_time, R_sensor_start_time) = \
        force_sensor_sync(force_sync_path, sensor_sync_path, RH_num, walk_num)
    # sensor end time
    L_sensor_end_time = L_sensor_start_time + GRF_end_time
    R_sensor_end_time = R_sensor_start_time + GRF_end_time

    # sensor L, R path
    (L_data_path, _) = \
        folder_path_name(path + "RasPi/sole/", "start", "L", 1)
    (R_data_path, _) = \
        folder_path_name(path + "RasPi/sole/", "start", "R", 1)

    # data reading ##################################################
    volt_header = ['time', 0, 1, 2, 3, 4, 5, 6, 7]

    #################################################################
    # L dataframe
    df_vol_L = read_SENSOR_vol(L_data_path[0], RH_num)
    df_vol_L = df_vol_L.loc[
        (df_vol_L['time'] >= L_sensor_start_time) &
        (df_vol_L['time'] <= L_sensor_end_time)
        ][volt_header]
    # initialize L time
    df_vol_L["time"] = df_vol_L["time"] - L_sensor_start_time
    # sole header matching for each trial
    L_sole_header = ['time']
    for n in range(0, 8):
        L_sole_header.append(sole_header["RH-" + RH_num]["left"][int(n)])
    df_vol_L.columns = L_sole_header

    #################################################################
    # R dataframe
    df_vol_R = read_SENSOR_vol(R_data_path[0], RH_num)
    df_vol_R = df_vol_R.loc[
        (df_vol_R['time'] >= R_sensor_start_time) &
        (df_vol_R['time'] <= R_sensor_end_time)
        ][volt_header]
    # initialize R time
    df_vol_R["time"] = df_vol_R["time"] - R_sensor_start_time
    # sole header matching for each trial
    R_sole_header = ['time']
    for n in range(0, 8):
        R_sole_header.append(sole_header["RH-" + RH_num]["right"][int(n)])
    df_vol_R.columns = R_sole_header

    return df_vol_L, df_vol_R


def N_data_preprocessing(data, NUM_PRE=30, WINDOWS=30, tol=0.01):

    data.sort_values(by=['time'], axis=0, ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)

    time_tmp = data.time.values.tolist()
    time_tmp = np.insert(time_tmp, 0, np.nan)
    time_tmp = np.delete(time_tmp, -1)

    data["pre_time"] = time_tmp

    for h in np.arange(1, NUM_PRE+1, 1):
        tmp = data.vout.values.tolist()
        for p in np.arange(0, h, 1):
            tmp = np.insert(tmp, int(p), np.nan)
            tmp = np.delete(tmp, -1)

        data["pre_%s" % (str(h))] = tmp

    data = data.astype(float)

    data["del_V"] = (data["vout"] - data["pre_1"])
    data["del_V"] = data["del_V"].rolling(window=WINDOWS,
                                          min_periods=1, center=True).mean()

    data["del_time"] = (data["time"] - data["pre_time"])

    data["loading_type"] = 0.0

    for q in np.arange(1, NUM_PRE+1, 1):
        data["loading_type"] += data["pre_%s" % (str(q))]

    data["loading_type"] = (data["loading_type"])/float(len(
        np.arange(1, NUM_PRE+1, 1)))
    data["loading_type"] -= data["vout"]
    data["loading_type"] = data["loading_type"].rolling(
        window=WINDOWS, min_periods=1, center=True).mean()

    loading_index1 = []
    loading_index1 = data[abs(data["loading_type"]) > tol].index
    # data["loading_type"] = np.sign(-data["loading_type"])

    data["loading_type1"] = 0.0
    data.loc[loading_index1, "loading_type1"] = np.sign(-data.loc[
        loading_index1, "loading_type"])

    delV_index1 = []
    delV_index1 = data[abs(data["del_V"]) > tol].index

    data["loading_type2"] = 0.0
    data.loc[delV_index1, "loading_type2"] = np.sign(-data.loc[
        delV_index1, "del_V"])

    delT_index1_1 = []
    delT_index1_1 = data[data["loading_type1"] != 0.0].index

    data["del_time1"] = 0.0
    data.loc[delT_index1_1, "del_time1"] = data.loc[delT_index1_1, "del_time"]

    data["elapsed_time1"] = np.cumsum(data["del_time1"])

    delT_index1_2 = []
    delT_index1_2 = data[data["loading_type2"] != 0.0].index

    data["del_time2"] = 0.0
    data.loc[delT_index1_2, "del_time2"] = data.loc[delT_index1_2, "del_time"]

    data["elapsed_time2"] = np.cumsum(data["del_time2"])

    data = data.dropna(axis=0)
    data.reset_index(drop=True, inplace=True)

    return data


def GPR_prediction(df, model_path, sensor_dir, sensor_num):

    X = df[["vout"]]
    ###############################################################
    # N-dimensional GPR
    ##############################################################
    # load GPR model
    gaussian_process = joblib.load(model_path +
                                   "GPR_CASE20_%s_%s.sav"
                                   % (str(sensor_dir),
                                      str(int(sensor_num) + 1)))
    # prediction
    mean_pred, std_pred = gaussian_process.predict(X, return_std=True)

    return mean_pred


def GPR_df_save(RH_num, df_vol_L, df_vol_R, sole_header, save_path):

    # create csv path
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    except OSError:
        pass

    # model path
    model_path = '../../data/analyzed/sole/RH-%s/model/' % RH_num

    # Define L, R force df
    ###############################################################
    # Left df
    df_force_L = pd.DataFrame(columns=df_vol_L.columns)
    df_force_L["time"] = df_vol_L["time"]
    # GPR prediction for each model
    for sensor in list(df_vol_L.columns):
        if sensor == "time":
            continue
        sensor_num = sole_header["RH-%s" % RH_num]["left"].index(sensor)
        # individual sensor vout for GPR prediction
        df_left_sensor = pd.DataFrame(df_vol_L[["time", sensor]])
        df_left_sensor.columns = ["time", "vout"]
        # N data preprocessing
        # df_left_sensor = N_data_preprocessing(df_left_sensor)
        # GPR prediction
        df_force_L[sensor] = GPR_prediction(
            df_left_sensor, model_path, "Left", sensor_num)
    # save GPR df
    df_force_L.to_csv(str(save_path) + "/df_force_L.csv",
                      header=True, index=False, sep=',')

    ###############################################################
    # Right df
    df_force_R = pd.DataFrame(columns=df_vol_R.columns)
    df_force_R["time"] = df_vol_R["time"]
    # GPR prediction for each model
    for sensor in list(df_vol_R.columns):
        if sensor == "time":
            continue
        sensor_num = sole_header["RH-%s" % RH_num]["right"].index(sensor)
        # individual sensor vout for GPR prediction
        df_right_sensor = pd.DataFrame(df_vol_R[["time", sensor]])
        df_right_sensor.columns = ["time", "vout"]
        # N data preprocessing
        # df_right_sensor = N_data_preprocessing(df_right_sensor)
        # GPR prediction
        df_force_R[sensor] = GPR_prediction(
            df_right_sensor, model_path, "Right", sensor_num)
    # save GPR df
    df_force_R.to_csv(str(save_path) + "/df_force_R.csv",
                      header=True, index=False, sep=',')

    return 0


def load_GPR(path):

    df_force_L = pd.read_csv(path + "/df_force_L.csv", sep=",", header=0)
    df_force_R = pd.read_csv(path + "/df_force_R.csv", sep=",", header=0)

    return df_force_L, df_force_R
