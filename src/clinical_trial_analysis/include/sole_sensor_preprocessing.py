# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:14:22 2022

@author: mleem
"""

import numpy as np
import pandas as pd
import glob
import os


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
    df_index = pd.DataFrame(index_data, columns=["GRF_time"])
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
