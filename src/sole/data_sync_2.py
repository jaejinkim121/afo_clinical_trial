# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:17:22 2022

@author: mleem
"""


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
from scipy.signal import find_peaks
from include.calib_functions import folder_path_name, instron_interp_preprocessing
from include.calib_functions import raspi_interp_preprocessing, N_data_preprocessing
from include.calib_functions import calib_interp, calib_result_plot

NUM_PRE = 30
WINDOWS = 30
tol = 0.01

df_sync = pd.DataFrame(columns=["RH_num", "folder_name", "pre_post", "time"])

# read the folder name
(exp_list, exp_name_list) = folder_path_name(
    "D:/OneDrive - SNU/AFO_analysis/afo_clinical_trial/data/",
    "start", "RH", 1)

# analysis path
anal_path = "D:/OneDrive - SNU/AFO_analysis/afo_clinical_trial/" +\
    "data/analyzed/sole/"

for (RH_list, RH_name) in zip(exp_list, exp_name_list):
    # RH-##
    RH_num = str(RH_name[-2:])
    # calibration case: once or twice
    if not os.path.exists(str(anal_path)+"RH-%s/Calibration/" % (RH_num)):
        calib_case = ["/Calibration_pre/", "/Calibration_post/"]
    else:
        calib_case = ["/Calibration/"]
    # for loop - calibration case
    for calib_name in calib_case:

        RH_name_path = str(anal_path) + "RH-%s" %(RH_num) +str(calib_name)
        # read folder name - size direction number
        (calib_test_list, calib_name_list) = folder_path_name(
            RH_name_path, "end", "txt", 0)
        # for loop - individual sensor folders
        for (test_list, name_list) in zip(calib_test_list, calib_name_list):
            # sensor number for pcb sequence
            sensor_num = str(name_list[-1])
            sensor_dir = str(name_list[3:-2])
            # internal path - individual sensor
            int_name_path = str(test_list)+"/"
            (int_list, int_name_list) = folder_path_name(int_name_path)
            # interpolation preprocessing
            for (int_path, int_name) in zip(int_list, int_name_list):
                # instron data
                if int_name.startswith("interlink") == 1:
                    print("calib_name")
                    print(calib_name)
                    print(name_list)
                    print(int_name)
                    # instron data sync, preprocessing
                    sync_time = instron_interp_preprocessing(
                        int_path, int_name_path + "instron_cycle300um_mod.csv")
                    df_sync = df_sync.append({"RH_num": RH_num,
                                              "folder_name": name_list,
                                              "pre_post":
                                                  str(calib_name)[1:-1],
                                              "time": sync_time},
                                             ignore_index=True)
                # sensor data
                elif int_name.startswith("pointSensor") == 1:
                    print("calib_name")
                    print(calib_name)
                    print(name_list)
                    print(int_name)
                    # sensor numbering from 1
                    num = int(sensor_num)
                    # sensor data preprocessing
                    raspi_interp_preprocessing(int_path, num,
                                               int_name_path +
                                               "sensor_cycle300um_mod.csv")

            # interpolation
            calib_interp(int_name_path + "instron_cycle300um_mod.csv",
                         sync_time,
                         int_name_path + "sensor_cycle300um_mod.csv",
                         int_name_path + "force_conversion_test.csv")

            # N data save directory
            inner_save_directory = str(int_name_path)+"N_data_test.csv"

            # individual sensor for loop - interpolation check & N_data
            for (int_path, int_name) in zip(int_list, int_name_list):
                # interpolation result
                if int_name.endswith("conversion_test.csv"):
                    calib_result_plot(str(int_path),
                                      RH_num, name_list+calib_name[1:-1])

                    final_data = pd.read_csv(str(int_path), header=0)
                    final_data = final_data.astype(float)

                    data = pd.DataFrame()
                    data[["time", "vout", "force"]] = final_data[["time",
                                                                  "vout",
                                                                  "force"]]
                    data = N_data_preprocessing(data, NUM_PRE, WINDOWS, tol)
                    data.to_csv(inner_save_directory, sep=',', index=False)
