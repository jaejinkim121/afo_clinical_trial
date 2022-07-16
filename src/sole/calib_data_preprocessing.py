# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 05:42:35 2021

@author: mleem
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
from scipy.signal import find_peaks
import statistics
from include.calib_functions import folder_path_name, clinical_force_data_reading
from include.calib_functions import sync_for_clinical_data, synced_data_plot
from include.calib_functions import COP_data_read_plot, GRF_data_read_plot
from include.calib_functions import sensor_stance_data_sync_plot_save
from include.calib_functions import sensor_walk_data_sync_plot_save
from include.calib_functions import force_sync_indexing, sensor_re_sync, force_interp
from include.calib_functions import N_data_preprocessing


# stance_num = {"03": 2, "04": 2, "05": 2, "06": 2, "07": 2,
#               "08": 2, "09": 2, "10": 2}
walk_num = {"02": 9, "03": 10, "04": 12, "05": 14, "06": 16, "07": 16,
            "08": 10, "09": 10, "10": 12}
walk_end = {"02": 15, "03": 18, "04": 18, "05": 19, "06": 20, "07": 24,
            "08": 16, "09": 20, "10": 21}

df_sync = pd.DataFrame(columns=["RH_num", "R or L", "stance or walk",
                                "number", "index start", "index end",
                                "time start", "time end"])
df_sync_force = pd.DataFrame(columns=["RH_num", "stance or walk", "number",
                                      "index start", "index end",
                                      "total end of index", "time start",
                                      "time end"])

# arguments for N preprocessing
NUM_PRE = 30
WINDOWS = 30
tol = 0.01

# analysis path
anal_path = "D:/OneDrive - SNU/AFO_analysis/" + \
    "afo_clinical_trial/data/analyzed/sole/"

# experiment folder
(exp_list, exp_name_list) = folder_path_name(
    "D:/OneDrive - SNU/AFO_analysis/afo_clinical_trial/data/",
    "start", "RH", 1)

##############################################
# # force plate data sync, plot, csv save
##############################################
# for (RH_list, RH_name) in zip(exp_list, exp_name_list):

#     # RH-##
#     RH_num = RH_name[-2:]

#     # walk data
#     walk_path = str(RH_list) + "/"

#     (walk_data_list, walk_data_name_list) = folder_path_name(
#         walk_path, "end", "anc", 1)

#     for (walk_file, walk_name) in zip(walk_data_list,
#                                       walk_data_name_list):
#         walk_number = str(walk_name[-6:-4])

#         if (int(walk_number) < walk_num[RH_num]) | (
#                 int(walk_number) > walk_end[RH_num]):
#             continue

#         # data read, raw data plot
#         force1_data, force2_data, time_data, \
#             sync_data = clinical_force_data_reading(
#                 walk_file, walk_name[:-4], str(anal_path) +
#                 "RH-%s/Graph/force_plate_raw" % (str(RH_num)), str(RH_num))

#         ##########################################################
#         # walk force synced data plotting
#         if (str(RH_num) == "04") & any(walk_name[-6:-4] == s
#                                        for s in ["12", "17"]):
#             synced_index = sync_for_clinical_data(
#                 sync_data, sync_condition="small")
#         else:
#             synced = sync_data[sync_data[["Sync"]] > 500]
#             synced.dropna(inplace=True)
#             synced_index = list(synced.index)
#         ##############################################################
#         # synced index : 싱크 들어간 index
#         ##############################################################
#         synced_data_plot(
#             force1_data, force2_data, time_data, synced,
#             synced_index, RH_num, walk_name[:-4],
#             str(anal_path) + "RH-%s/Graph/force_plate_synced" % (str(RH_num)),
#             str(anal_path) + "RH-%s/synced_data" % (str(RH_num)))

#         df_sync_force = df_sync_force.append({
#             "RH_num": "RH-" + str(RH_num),
#             "stance or walk": "walk",
#             "number": str(walk_number),
#             "index start": synced_index[0],
#             "index end": synced_index[-1],
#             "total end of index": time_data.index[-1],
#             "time start": time_data.loc[synced_index[0], "Name"],
#             "time end": time_data.loc[synced_index[-1], "Name"]},
#             ignore_index=True)

#         ##########################################################
#         # walk COP data
#         walk_COP_file = str(walk_file[:-4])+".forces"

#         COP_data_read_plot(
#             walk_COP_file, synced, synced_index, time_data,
#             walk_name[:-4], RH_num, str(anal_path) +
#             "RH-%s/Graph/force_plate_synced" % (str(RH_num)), str(anal_path) +
#             "RH-%s/synced_data" % (str(RH_num)))

#         ##############################################################
#         ##############################################################
#         # GRF data
#         (walk_GRF_data_list,
#          walk_GRF_data_name_list) = folder_path_name(walk_path,
#                                                      "end", "XLS", 1)
#         for (GRF_file, GRF_name) in zip(walk_GRF_data_list,
#                                         walk_GRF_data_name_list):

#             GRF_data_read_plot(GRF_file, GRF_name[:-4], RH_num,
#                                str(anal_path) + "RH-%s/Graph/GRF" % (
#                                     str(RH_num)), str(anal_path) +
#                                "RH-%s/GRF_data" % (str(RH_num)))

# df_sync_force.to_csv(str(anal_path) + "df_sync_force.csv", sep=",",
#                      index=False, header=True)

# ##############################################
# # # sensor data sync, plot, csv save
# ##############################################
# for (RH_list, RH_name) in zip(exp_list, exp_name_list):

#     # RH-##
#     RH_num = RH_name[-2:]
#     ##############################################################
#     #               FOOT PRESSURE SENSOR DATA             ########
#     ##############################################################
#     # foot pressure sensor data
#     vol = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']
#     ##########################################################
#     ##########################################################
#     # walk data
#     walk_path = str(RH_list) + "/RasPi/sole/"

#     (L_walk_data_list,
#       L_walk_name_list) = folder_path_name(
#           walk_path, "start", "L", 1)
#     (R_walk_data_list, R_walk_name_list) = folder_path_name(
#         walk_path, "start", "R", 1)

#     df_sync = sensor_walk_data_sync_plot_save(
#         L_walk_data_list, L_walk_name_list, R_walk_data_list,
#         R_walk_name_list, df_sync, RH_num, walk_num,
#         str(anal_path) + "RH-%s/Graph/sensor_data" % (str(RH_num)),
#         str(anal_path) + "RH-%s/synced_data" % (str(RH_num)))

# df_sync.to_csv(str(anal_path) + "/df_sync.csv", sep=",",
#                 index=False, header=True)

# ##############################################
# # # re sync for sensor data
# ##############################################
# for (RH_list, RH_name) in zip(exp_list, exp_name_list):

#     # RH-##
#     RH_num = RH_name[-2:]
#     ##############################################################
#     #               FOOT PRESSURE SENSOR DATA             ########
#     ##############################################################
#     # foot pressure sensor data
#     if RH_num == "02":
#         pass
#     elif RH_num == "04":
#         pass
#     else:
#         ##########################################################
#         ##########################################################
#         # walk data
#         (walk_data_list, walk_data_name_list) = folder_path_name(
#             str(folder_list) + "/walk_data/", "start", "walk", 1)

#         for (walk_file, walk_name) in zip(walk_data_list,
#                                           walk_data_name_list):

#             walk_num = walk_name[5:]
#             if (len(str(walk_num)) > 1) & (str(walk_num)[0] == "0"):
#                 walk_num = str(walk_num)[1]
#             else:
#                 pass

#             # reference index, time from force plate data
#             ref_start_index_W, ref_end_index_W, \
#                 true_start_index_W, ref_start_time_W, \
#                 ref_end_time_W, ref_del_time_W = force_sync_indexing(
#                         df_sync_force, RH_num, "walk", walk_num)
#             ######################################################
#             # L data
#             (Lwalk_data_list,
#               Lwalk_data_name_list) = folder_path_name(str(
#                   walk_file) + "/L/", "start", "RH", 1)

#             for (Lwalk_final_file, Lwalk_final_name) in zip(
#                     Lwalk_data_list, Lwalk_data_name_list):

#                 sensor_num = str(Lwalk_final_name[-5:-4])
#                 Lsensor_walk_file = Lwalk_final_file

#                 # re sync
#                 sensor_re_sync(Lsensor_walk_file, sensor_num, str(
#                   walk_file) + "/L/", ref_start_index_W,
#                     ref_end_index_W, true_start_index_W,
#                     ref_start_time_W, ref_end_time_W,
#                     ref_del_time_W)
#             ######################################################
#             # R data
#             (Rwalk_data_list,
#               Rwalk_data_name_list) = folder_path_name(str(
#                   walk_file) + "/R/", "start", "RH", 1)

#             for (Rwalk_final_file, Rwalk_final_name) in zip(
#                     Rwalk_data_list, Rwalk_data_name_list):

#                 sensor_num = str(Rwalk_final_name[-5:-4])
#                 Rsensor_walk_file = Rwalk_final_file

#                 # re sync
#                 sensor_re_sync(Rsensor_walk_file, sensor_num, str(
#                   walk_file) + "/R/", ref_start_index_W,
#                     ref_end_index_W, true_start_index_W,
#                     ref_start_time_W, ref_end_time_W,
#                     ref_del_time_W)

# ##############################################
# # interpolation
# ##############################################
# for (RH_list, RH_name) in zip(exp_list, exp_name_list):

#     (RH_int_list, RH_int_name_list) = folder_path_name(RH_list + "/")
#     # RH-##
#     RH_num = [file for file in RH_int_name_list if file.startswith("RH") == 1]
#     RH_num = RH_num[0][3:]
#     ##############################################################
#     #               FOOT PRESSURE SENSOR DATA             ########
#     ##############################################################
#     # RasPi stance path
#     (stance_list, stance_name_list) = folder_path_name(
#         RH_list + "/RasPi/stance_data/", "start", "stance", 1)
#     # experiment folder internal for loop
#     for (folder_list, folder_name) in zip(stance_list, stance_name_list):
#         # stance number
#         stance_num = folder_name[-2:]

#         # force COP data path
#         (force_path, force_name) = folder_path_name(
#             RH_list + "/RH-%s/stance_data/stance_%s/" % (
#                 str(RH_num), str(stance_num)), "end", "(processed).csv", 1)
#         force_data = pd.read_csv(force_path[0], delimiter=",", header=0)

#         # Left sensor
#         (L_path, L_name) = folder_path_name(folder_list + "/L/",
#                                             "start", "SENSOR", 1)
#         for (L_data, L_data_name) in zip(L_path, L_name):
#             L_sensor_data = pd.read_csv(L_data, sep=",", header=0)
#             force_interp(L_sensor_data, force_data, L_data_name[6:7],
#                          folder_list + "/L/")

#         # Right sensor
#         (R_path, R_name) = folder_path_name(folder_list + "/R/",
#                                             "start", "SENSOR", 1)
#         for (R_data, R_data_name) in zip(R_path, R_name):
#             R_sensor_data = pd.read_csv(R_data, sep=",", header=0)
#             force_interp(R_sensor_data, force_data, R_data_name[6:7],
#                          folder_list + "/R/")

#     # RasPi walk path
#     (walk_list, walk_name_list) = folder_path_name(
#         RH_list + "/RasPi/walk_data/", "start", "walk", 1)
#     # experiment folder internal for loop
#     for (folder_list, folder_name) in zip(walk_list, walk_name_list):
#         # walk number
#         walk_num = folder_name[-2:]

#         # force COP data path
#         (force_path, force_name) = folder_path_name(
#             RH_list + "/RH-%s/walk_data/walk_%s/" % (
#                 str(RH_num), str(walk_num)), "end", "(processed).csv", 1)
#         force_data = pd.read_csv(force_path[0], delimiter=",", header=0)

#         # Left sensor
#         (L_path, L_name) = folder_path_name(folder_list + "/L/",
#                                             "start", "SENSOR", 1)
#         for (L_data, L_data_name) in zip(L_path, L_name):
#             L_sensor_data = pd.read_csv(L_data, sep=",", header=0)
#             force_interp(L_sensor_data, force_data, L_data_name[6:7],
#                          folder_list + "/L/")

#         # Right sensor
#         (R_path, R_name) = folder_path_name(folder_list + "/R/",
#                                             "start", "SENSOR", 1)
#         for (R_data, R_data_name) in zip(R_path, R_name):
#             R_sensor_data = pd.read_csv(R_data, sep=",", header=0)
#             force_interp(R_sensor_data, force_data, R_data_name[6:7],
#                          folder_list + "/R/")

# ##############################################
# # N_data preprocessing
# ##############################################
for (RH_list, RH_name) in zip(exp_list, exp_name_list):

    # RH-##
    RH_num = RH_name[-2:]
    ##############################################################
    #               FOOT PRESSURE SENSOR DATA             ########
    ##############################################################
    # RasPi walk path
    (walk_list, walk_name_list) = folder_path_name(
        RH_list + "/RasPi/walk_data/", "start", "walk", 1)
    # experiment folder internal for loop
    for (folder_list, folder_name) in zip(walk_list, walk_name_list):
        # walk number
        walk_num = folder_name[-2:]

        # Left sensor
        L_csv_path = folder_list + "/L/"
        (L_path, L_name) = folder_path_name(L_csv_path, "start", "interp", 1)
        for (L_data, L_data_name) in zip(L_path, L_name):

            L_interp_data = pd.read_csv(L_data, sep=",", header=0)
            L_data_N = N_data_preprocessing(L_interp_data, NUM_PRE,
                                            WINDOWS, tol)
            L_data_N.to_csv(L_csv_path + "N_data_SENSOR_%s.csv" %
                            (L_data_name[-5:-4]), header=True,
                            index=False, sep=',')

        # Right sensor
        R_csv_path = folder_list + "/R/"
        (R_path, R_name) = folder_path_name(R_csv_path, "start", "interp", 1)
        for (R_data, R_data_name) in zip(R_path, R_name):

            R_interp_data = pd.read_csv(R_data, sep=",", header=0)
            R_data_N = N_data_preprocessing(R_interp_data, NUM_PRE,
                                            WINDOWS, tol)
            R_data_N.to_csv(R_csv_path + "N_data_SENSOR_%s.csv" %
                            (R_data_name[-5:-4]), header=True,
                            index=False, sep=',')
