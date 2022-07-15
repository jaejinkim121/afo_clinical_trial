# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:14:22 2022

@author: mleem
"""

import numpy as np
import pandas as pd
import glob
import os
import joblib
import time


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


def load_GRF(GRF_path):

    GRF_data = pd.read_csv(GRF_path, delimiter="\t", header=23)
    GRF_data = GRF_data.astype(float)

    index_data = list(GRF_data.index)
    df_index = pd.DataFrame(index_data, columns=["time"])
    df_index = (df_index * 5) / 600
    end_time = df_index.iloc[-1, 0]
    R_GRF_data = GRF_data[["R_GRF_VRT"]]
    L_GRF_data = GRF_data[["L_GRF_VRT"]]

    walk_GRF = pd.concat([df_index, L_GRF_data], axis=1)
    walk_GRF = pd.concat([walk_GRF, R_GRF_data], axis=1)

    return walk_GRF, end_time


def load_SENSOR_vol(path, RH_num):

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
        data.columns = ['time', 'v0', 'v1', 'v2', 'v3',
                        'v4', 'v5', 'v6', 'v7']
        data.reset_index(drop=True, inplace=True)

    else:
        data.columns = ['sync', 'time', 'v0', 'v1', 'v2',
                        'v3', 'v4', 'v5', 'v6', 'v7']
        data.reset_index(drop=True, inplace=True)

    return data


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

    X = df["vout"]
    ###############################################################
    # N-dimensional GPR
    ##############################################################
    pred_start_time = time()

    # load GPR model
    gaussian_process = joblib.load(model_path + "GPR_220515_CASE20_280%s_%s"
                                   % (str(sensor_dir), str(sensor_num)))
    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
    
    pred_end_time = time()
    # save the model
    df_test = df_test.append({"case_num": str(case),"RH": str(RH_num), "stance_or_walk": "walk", "stance_or_walk_num": str(walk_num), "sensor_dir": str(sensor_dir[0].upper()),"sensor_num": str(sensor_num),"sample_num": NUM_MIN_SAMPLE, "pred_time":float(pred_end_time - pred_start_time)}, ignore_index=True)
    
    pred_directory = str(walk_final_name_path)+"prediction"
    
    try:
        if not os.path.exists(pred_directory):
            os.makedirs(pred_directory)
    except OSError:
        pass
    
    final_test_data = pd.DataFrame(data = reg_test_data, columns = reg_test_data.columns)
    final_test_data["mean_prediction"] = mean_prediction
    final_test_data["std_prediction"] = std_prediction
    final_test_data.to_csv(pred_directory+"/pred_SENSOR_%s.csv" %(str(sensor_num)),sep=',', index = False, header = True)
                    