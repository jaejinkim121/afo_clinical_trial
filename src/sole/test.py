# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 23:26:53 2022

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
from include.calib_functions import force_sync_indexing, sensor_re_sync, GRF_indexing

prop_title = dict(
    family='Times New Roman',
    style='normal',
    weight='bold',
    size=30
)

prop_label = dict(
    family='Times New Roman',
    style='normal',
    weight='bold',
    size=25
)

prop_tick = dict(
    family='Times New Roman',
    style='normal',
    weight='bold',
    size=20
)

prop_legend = dict(
    family='Times New Roman',
    style='normal',
    weight='bold',
    size=15
)


NUM_PRE = 30
NUM_AVG = 5
gait_cycle_tol = 50
WINDOWS = 30
tol = 0.01

# walk_force_labeling = {"02": {9: ["R", "L"], 10: ["L", "R"], 11: ["R", "L"],
#                               12: ["L", "R"], 13: ["R", "L"], 14: ["L", "R"],
#                               15: ["R", "L"]},
#                        "03": {10: [], 11: ["R", "L"], 12: ["L", "R"],
#                               13: ["R", "L"], 14: [], 15: ["R", "L"],
#                               16: ["L", "R"], 17: [], 18: ["L", "R"]},
#                        "04": {12: ["R", "L"], 13: [], 14: ["R", "L"],
#                               15: ["L", "R"], 16: [], 17: [], 18: []},
#                        "05": {14: ["R", "L"], 15: ["L", "R"], 16: ["R", "L"],
#                               17: ["L", "R"], 18: ["R", "L"], 19: ["L", "R"]},
#                        "06": {16: [], 17: ["R", "L"], 18: ["R", "L"],
#                               19: ["L", "R"], 20: ["R", "L"]},
#                        "07": {16: ["R", "L"], 17: ["L", "R"], 18: ["R", "L"],
#                               19: ["L", "R"], 20: ["R", "L"], 21: ["L", "R"],
#                               22: ["R", "L"], 23: ["L", "R"], 24: ["R", "L"]},
#                        "08": {10: ["L", "R"], 11: ["R", "L"], 12: ["L", "R"],
#                               13: [], 14: ["L", "R"], 15: ["R", "L"],
#                               16: ["L", "R"]},
#                        "09": {10: ["R", "L"], 11: ["L", "R"], 12: ["R", "L"],
#                               13: ["L", "R"], 14: ["R", "L"], 15: ["L", "R"],
#                               16: ["R", "L"], 17: ["L", "R"], 18: ["R", "L"],
#                               19: ["L", "R"], 20: ["R", "L"]},
#                        "10": {12: ["R", "L"], 13: ["L", "R"], 14: ["R", "L"],
#                               15: ["R", "L"], 16: ["R", "L"], 17: ["R", "L"],
#                               18: ["R", "L"], 19: ["R", "L"], 20: ["R", "L"],
#                               21: ["R", "L"]}}
# left -> right 순서
walk_ref_num = {"09": {10: {"L": 1, "R": 1}, 11: {"L": 1, "R": 2},
                       12: {"L": 1, "R": 2}, 13: {"L": 2, "R": 2},
                       14: {"L": 2, "R": 1}, 15: {"L": 1, "R": 1},
                       16: {"L": 1, "R": 2}, 17: {"L": 2, "R": 1},
                       18: {"L": 1, "R": 2}, 19: {"L": 2, "R": 2},
                       20: {"L": 2, "R": 2}}}

# affected direction
RH_affected = {"09": "L"}

(exp_list, exp_name_list) = folder_path_name(
    "D:/OneDrive - SNU/AFO_exp_data/", "start", "exp", 1)

total_index = pd.DataFrame()
total_index_counter = pd.DataFrame()

for (RH_list, RH_name) in zip(exp_list, exp_name_list):

    RH_num = str(RH_name[-2:])
    (walk_test_list, walk_name_list) = folder_path_name(str(
        RH_list)+"/RH-%s/walk_data/" % (RH_num), "start", "walk", 1)

    # gait start, end index
    L_start_end_index = pd.DataFrame(columns=["walk_num", "start_index",
                                              "end_index"])
    R_start_end_index = pd.DataFrame(columns=["walk_num", "start_index",
                                              "end_index"])

    # final gait index
    L_final_index = pd.DataFrame(columns=["walk_num", "start_index",
                                          "end_index"])
    R_final_index = pd.DataFrame(columns=["walk_num", "start_index",
                                          "end_index"])

    # maximum freq (within walk)
    L_max_walk_num = "0"
    L_max_num = 0
    R_max_walk_num = "0"
    R_max_num = 0

    # 환자별 vGRF saving
    vGRF_L_ref = pd.DataFrame(columns=["index data", "L_GRF_VRT"])
    vGRF_R_ref = pd.DataFrame(columns=["index data", "R_GRF_VRT"])
    vGRF_L_total = pd.DataFrame(columns=["cycle"])
    vGRF_R_total = pd.DataFrame(columns=["cycle"])

    ######################################################################
    # GRF indexing for loop
    for (folder_list, folder_name) in zip(walk_test_list, walk_name_list):

        # force data path reading
        (force_path, force_name) = folder_path_name(str(
            folder_list)+"/", "end", "GRF.csv", 1)

        # data reading
        walk_num = str(folder_name[-2:])
        force_data = pd.read_csv(force_path[0], header=0)
        # time_col_name = "del_time"

        # # Left, right force number labelling
        # if walk_force_labeling[RH_num][int(walk_num)][0] == "L":
        #     L_col_name = "FZ1"
        #     R_col_name = "FZ2"
        # else:
        #     L_col_name = "FZ2"
        #     R_col_name = "FZ1"

        # left, right force dataframe
        vGRF_L = force_data[["index data", "L_GRF_VRT"]]
        vGRF_R = force_data[["index data", "R_GRF_VRT"]]

        ############################################################
        # #                LEFT ONLY                               #
        ############################################################
        # Find GRF start, end index
        (L_all, L_final) = GRF_indexing(vGRF_L, walk_num, "L_GRF_VRT",
                                        gait_cycle_tol, walk_ref_num[RH_num])
        L_start_end_index = pd.concat([L_start_end_index, L_all],
                                      ignore_index=True)
        L_final_index = pd.concat([L_final_index, L_final], ignore_index=True)

        # GRF data indexing
        vGRF_L_synced = vGRF_L.loc[L_final.loc[0, "start_index"]:L_final.loc[
            0, "end_index"], :]
        L_total_length = L_final.loc[0, "end_index"] - L_final.loc[
            0, "start_index"]
        vGRF_L_synced.reset_index(drop=True, inplace=True)
        vGRF_L_synced.loc[:, "index data"] = (
            vGRF_L_synced.loc[:, "index data"] - L_final.loc[
                0, "start_index"]) * 100 / L_total_length

        # Find max walk_num
        if L_max_num <= L_total_length:
            L_max_num = L_total_length
            L_max_walk_num = walk_num
            vGRF_L_ref = vGRF_L_synced

        #############################################################
        # #                RIGHT ONLY                               #
        #############################################################
        # Find GRF start, end index
        (R_all, R_final) = GRF_indexing(vGRF_R, walk_num, "R_GRF_VRT",
                                        gait_cycle_tol, walk_ref_num[RH_num])
        R_start_end_index = pd.concat([R_start_end_index, R_all],
                                      ignore_index=True)
        R_final_index = pd.concat([R_final_index, R_final], ignore_index=True)

        # GRF data indexing
        vGRF_R_synced = vGRF_R.loc[R_final.loc[0, "start_index"]:R_final.loc[
            0, "end_index"], :]
        R_total_length = R_final.loc[0, "end_index"] - R_final.loc[
            0, "start_index"]
        vGRF_R_synced.reset_index(drop=True, inplace=True)
        vGRF_R_synced.loc[:, "index data"] = (
            vGRF_R_synced.loc[:, "index data"] - R_final.loc[
                0, "start_index"]) * 100 / R_total_length

        # Find max walk_num
        if R_max_num <= R_total_length:
            R_max_num = R_total_length
            R_max_walk_num = walk_num
            vGRF_R_ref = vGRF_R_synced

    ########################################################################
    # GRF interpolation for loop
    for (folder_list, folder_name) in zip(walk_test_list, walk_name_list):

        # force data path reading
        (force_path, force_name) = folder_path_name(str(
            folder_list)+"/", "end", "GRF.csv", 1)

        # data reading
        walk_num = str(folder_name[-2:])
        force_data = pd.read_csv(force_path[0], header=0)

        # left, right force dataframe
        vGRF_L = force_data[["index data", "L_GRF_VRT"]]
        vGRF_R = force_data[["index data", "R_GRF_VRT"]]

        ############################################################
        # #                LEFT ONLY                               #
        ############################################################
        # reload start, end index
        GRF_L_start = L_final_index[L_final_index["walk_num"] == walk_num][
            "start_index"].iloc[0]
        GRF_L_end = L_final_index[L_final_index["walk_num"] == walk_num][
                    "end_index"].iloc[0]

        # GRF data indexing
        vGRF_L_synced = vGRF_L.loc[GRF_L_start:GRF_L_end, :]
        L_total_length = GRF_L_end - GRF_L_start
        vGRF_L_synced.reset_index(drop=True, inplace=True)
        vGRF_L_synced.loc[:, "index data"] = (
            vGRF_L_synced.loc[:, "index data"] - GRF_L_start
            ) * 100 / L_total_length

        # GRF data interpolation
        vGRF_L_interp = pd.DataFrame(np.interp(vGRF_L_ref["index data"],
                                               vGRF_L_synced["index data"],
                                               vGRF_L_synced["L_GRF_VRT"]),
                                     columns=["L_GRF_VRT"])
        vGRF_L_total["cycle"] = vGRF_L_ref["index data"]
        vGRF_L_total["walk_%s" % (walk_num)] = vGRF_L_interp["L_GRF_VRT"]

        #############################################################
        # #                RIGHT ONLY                               #
        #############################################################
        # reload start, end index
        GRF_R_start = R_final_index[
            R_final_index["walk_num"] == walk_num]["start_index"].iloc[0]
        GRF_R_end = R_final_index[R_final_index["walk_num"] == walk_num][
                    "end_index"].iloc[0]

        # GRF data indexing
        vGRF_R_synced = vGRF_R.loc[GRF_R_start:GRF_R_end, :]
        R_total_length = GRF_R_end - GRF_R_start
        vGRF_R_synced.reset_index(drop=True, inplace=True)
        vGRF_R_synced.loc[:, "index data"] = (
            vGRF_R_synced.loc[:, "index data"] - GRF_R_start
            ) * 100 / R_total_length

        # GRF data interpolation
        vGRF_R_interp = pd.DataFrame(np.interp(vGRF_R_ref["index data"],
                                               vGRF_R_synced["index data"],
                                               vGRF_R_synced["R_GRF_VRT"]),
                                     columns=["R_GRF_VRT"])
        vGRF_R_total["cycle"] = vGRF_R_ref["index data"]
        vGRF_R_total["walk_%s" % (walk_num)] = vGRF_R_interp["R_GRF_VRT"]

    ########################################################################
    # GRF avg, std, saving
    vGRF_L_total["avg"] = np.mean(vGRF_L_total.loc[:, walk_name_list], axis=1)
    vGRF_L_total["std"] = np.std(vGRF_L_total.loc[:, walk_name_list], axis=1)

    vGRF_R_total["avg"] = np.mean(vGRF_R_total.loc[:, walk_name_list], axis=1)
    vGRF_R_total["std"] = np.std(vGRF_R_total.loc[:, walk_name_list], axis=1)

    ########################################################################
    # GRF plotting
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("RH-%s_GRF(affected side: %s)" % (str(RH_num),
                                                   RH_affected[str(RH_num)]),
                 fontsize=15, **prop_title)

    bx1 = plt.subplot(2, 1, 1)
    for axis in ['top', 'bottom', 'left', 'right']:
        bx1.spines[axis].set_linewidth(2)
    plt.errorbar(vGRF_L_total["cycle"], vGRF_L_total["avg"],
                 vGRF_L_total["std"], mfc="blue", mec="blue", fmt='-o',
                 ecolor="lightskyblue")
    # plt.xlabel("Gait cycle [%]", **prop_tick)
    plt.ylabel("Left foot", **prop_tick)
    plt.xticks(np.arange(0, 110, 10), fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.grid(True)

    bx2 = plt.subplot(2, 1, 2, sharex=bx1)
    for axis in ['top', 'bottom', 'left', 'right']:
        bx2.spines[axis].set_linewidth(2)
    plt.errorbar(vGRF_R_total["cycle"], vGRF_R_total["avg"],
                 vGRF_R_total["std"], mfc="darkorange", mec="darkorange",
                 fmt='-o', ecolor="orange")
    plt.xlabel("Gait cycle [%]", **prop_tick)
    plt.ylabel("Right foot", **prop_tick)
    plt.xticks(np.arange(0, 110, 10), fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.grid(True)

    save_path = str(RH_list)+"/Graph/gait_analysis/"
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    except OSError:
        pass

    vGRF_L_total.to_csv(save_path + "RH-%s_GRF_L.csv" % (str(RH_num)),
                        sep=",", index=False)
    vGRF_R_total.to_csv(save_path + "RH-%s_GRF_R.csv" % (str(RH_num)),
                        sep=",", index=False)

    plt.savefig(save_path + "RH-%s_GRF(affected side %s).png" % (
        str(RH_num), RH_affected[str(RH_num)]))
