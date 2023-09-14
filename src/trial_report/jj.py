import pandas as pd
import numpy as np
import csv
from utils import DataProcess


def get_initial_contact_time(gait_phase:pd.DataFrame):
    if 'time' not in gait_phase.columns:
        print("Time data missing")
        return -1
    if 'value' not in gait_phase.columns:
        print("Value data missing")
        return -1

    time_initial_contact = gait_phase[gait_phase["value"] == 1]["time"]

    return time_initial_contact.tolist()


def divider_data_by_gait_phase(data_path, gait_phase_path):
    data = pd.DataFrame()
    gait_phase = pd.DataFrame()

    value_data = []
    time_data = []
    value_gait_phase = []
    time_gait_phase = []

    with open(data_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        for csv_row in csv_reader:
            if csv_row[0] == "time":
                continue
            value_data.append(float(csv_row[1]))
            time_data.append(float(csv_row[0]))

    with open(gait_phase_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        for csv_row in csv_reader:
            if csv_row[0] == "time":
                continue
            value_gait_phase.append(float(csv_row[1]))
            time_gait_phase.append(float(csv_row[0]))

    data["time"] = time_data
    data["value"] = value_data
    gait_phase["time"] = time_gait_phase
    gait_phase["value"] = value_gait_phase

    divided_array = []
    time_initial_contact = get_initial_contact_time(gait_phase)
    time_initial_contact.append(time_data[-1])

    for i in range(len(time_initial_contact) - 1):
        divided_df_current = \
            data[
                (data["time"] >= time_initial_contact[i]) &
                (data["time"] < time_initial_contact[i+1])
            ]
        divided_array.append(divided_df_current.to_numpy())

    return divided_array


def graph_averaged_data(collection_data, x_num=101):
    mean, std = DataProcess.average_cropped_time_series(
        collection_data, x_num
    )
    x = np.linspace(0, 100, x_num)



    ...


class ClinicalIndexJJ:
    @staticmethod
    def data_process_toe_clearance(paretic_data_path,
                                   paretic_gait_path,
                                   non_paretic_data_path,
                                   non_paretic_gait_path):

        collection_paretic_toe_clearance = \
            divider_data_by_gait_phase(paretic_data_path,
                                       paretic_gait_path)
        collection_non_paretic_toe_clearance = \
            divider_data_by_gait_phase(non_paretic_data_path,
                                       non_paretic_gait_path)

        # Graph Processing


        # Statistics Processing
        max_toe_clearance_paretic = []
        max_toe_clearance_non_paretic = []

        for da in collection_paretic_toe_clearance:
            max_toe_clearance_paretic.append(
                np.mean(da[:, 1])
            )
        for da in collection_non_paretic_toe_clearance:
            max_toe_clearance_non_paretic.append(
                np.mean(da[:, 1])
            )

        np_ptc = np.array(max_toe_clearance_paretic)
        np_nptc = np.array(max_toe_clearance_non_paretic)

        paretic_mean = np.mean(np_ptc)
        paretic_stdev = np.std(np_ptc)
        non_paretic_mean = np.mean(np_nptc)
        non_paretic_stdev = np.std(np_nptc)
        symmetry = paretic_mean / (paretic_mean + non_paretic_mean)

        return [paretic_mean, paretic_stdev,
                non_paretic_mean, non_paretic_stdev,
                symmetry
                ]

    @staticmethod
    def data_process_gait_speed_imu(stride_path):
        stride = []
        with open(stride_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for csv_row in csv_reader:
                stride.append(float(csv_row[1]))
        stride = np.array(stride)

        return np.sum(stride) / 120.0

    @staticmethod
    def data_process_gait_speed_analog(distance=None, time_sec=None):
        if distance is not None:
            gait_speed = distance / 120.0
        elif time_sec is not None:
            gait_speed = 10.0 / time_sec
        else:
            ...

        return gait_speed


path_paretic = "../../dummy.csv"
path_nonparetic = "../../dummy.csv"
path_gait_paretic = "../../dummy_gait_phase.csv"

divider_data_by_gait_phase(path_paretic, path_gait_paretic)
