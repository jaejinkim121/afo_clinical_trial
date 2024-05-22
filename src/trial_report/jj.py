import pandas as pd
import numpy as np
import csv
from utils import DataProcess
import matplotlib.pyplot as plt
from define import Bag, Side


class ClinicalIndexJJ:
    @staticmethod
    def data_process_kinematics(y_path,
                                z_path,
                                paretic_gait_path,
                                non_paretic_gait_path,
                                metadata:Bag,
                                default_path,
                                start_time):
        report_save_path = default_path + "/report/data/" + \
                           metadata.test_label + "/" + metadata.session_type

        y_data = pd.read_csv(y_path, header=0)
        z_data = pd.read_csv(z_path, header=0)

        y_time = y_data["Time"]
        y_time -= start_time
        left_y = y_data["data_0"] * 1000.0
        right_y = y_data["data_7"] * 1000.0

        left_y = pd.DataFrame({"time":y_time, "value":left_y})
        right_y = pd.DataFrame({"time":y_time, "value":right_y})

        paretic_side = metadata.paretic_side

        if paretic_side == Side.LEFT:
            paretic_data = left_y
            non_paretic_data = right_y
        else:
            paretic_data = right_y
            non_paretic_data = left_y

        ignore_cycle = (metadata.ignore_cycle[0], metadata.ignore_cycle[1])
        max_, impulse_, stance_ = DataProcess.data_process(
            paretic_data,
            non_paretic_data,
            paretic_gait_path,
            non_paretic_gait_path,
            save_path=report_save_path,
            data_label="mm",
            title_label="Toe Clearance",
            ignore_cycle=ignore_cycle,
            start_time=start_time,
            max_flag=True)

        return max_

        #
        # collection_paretic_toe_clearance, gait_phase_paretic = \
        #     DataProcess.divider_data_by_gait_phase_path(
        #         paretic_data_path,
        #         paretic_gait_path)
        # collection_non_paretic_toe_clearance, gait_phase_nonparetic = \
        #     DataProcess.divider_data_by_gait_phase_path(
        #         non_paretic_data_path,
        #         non_paretic_gait_path)

        # Graph Processing
        # graph_averaged_data(collection_paretic_toe_clearance,
        #                     "Toe clearance - paretic",
        #                     "height [m]")
        # graph_averaged_data(collection_non_paretic_toe_clearance,
        #                     "Toe clearance - nonparetic",
        #                     "height [m]")
        # DataProcess.graph_both_cycle_data(
        #     collection_paretic_toe_clearance,
        #     collection_non_paretic_toe_clearance,
        #     gait_phase_paretic, gait_phase_nonparetic,
        #     x_num=101)

        # Statistics Processing

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
    ClinicalIndexJJ.data_process_kinematics(path_paretic,
                                            path_gait_paretic,
                                            path_nonparetic,
                                            path_gait_nonparetic)


if __name__ == "__main__":
    main()
