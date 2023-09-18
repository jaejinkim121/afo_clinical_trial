import pandas as pd
import numpy as np
import csv
from utils import DataProcess
import matplotlib.pyplot as plt


class ClinicalIndexJJ:
    @staticmethod
    def data_process_toe_clearance(paretic_data_path,
                                   paretic_gait_path,
                                   non_paretic_data_path,
                                   non_paretic_gait_path):
        collection_paretic_toe_clearance, gait_phase_paretic = \
            DataProcess.divider_data_by_gait_phase_path(
                paretic_data_path,
                paretic_gait_path)
        collection_non_paretic_toe_clearance, gait_phase_nonparetic = \
            DataProcess.divider_data_by_gait_phase_path(
                non_paretic_data_path,
                non_paretic_gait_path)

        # Graph Processing
        # graph_averaged_data(collection_paretic_toe_clearance,
        #                     "Toe clearance - paretic",
        #                     "height [m]")
        # graph_averaged_data(collection_non_paretic_toe_clearance,
        #                     "Toe clearance - nonparetic",
        #                     "height [m]")
        DataProcess.graph_both_cycle_data(
            collection_paretic_toe_clearance,
            collection_non_paretic_toe_clearance,
            gait_phase_paretic, gait_phase_nonparetic,
            x_num=101)

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


def main():
    path_paretic = "../../dummy.csv"
    path_nonparetic = "../../dummy.csv"
    path_gait_paretic = "../../dummy_gait_phase_p.csv"
    path_gait_nonparetic = "../../dummy_gait_phase_np.csv"

    DataProcess.divider_data_by_gait_phase_path(
        path_paretic, path_gait_paretic)
    ClinicalIndexJJ.data_process_toe_clearance(path_paretic,
                                               path_gait_paretic,
                                               path_nonparetic,
                                               path_gait_nonparetic)


if __name__ == "__main__":
    main()
