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
                                metadata: Bag,
                                default_path,
                                start_time,
                                report_start_time,
                                report_duration,
                                current_gait_event_filter=None):
        report_save_path = default_path + "/report/data/" + \
                           metadata.test_label + "/" + metadata.session_type

        clearance_file_path = report_save_path + \
                              "/process_data/inference_data/clearance.csv"

        df_clearance_ = pd.read_csv(clearance_file_path,
                                    names=["t",
                                           "Left toe",
                                           "Left heel",
                                           "Right toe",
                                           "Right heel"],
                                    header=None)
        torso_height = 800
        df_toe_left = pd.DataFrame(
            {"time": df_clearance_["t"],
             "value": df_clearance_["Left toe"] + torso_height})
        df_heel_left = pd.DataFrame(
            {"time": df_clearance_["t"],
             "value": df_clearance_["Left heel"] + torso_height})
        df_toe_right = pd.DataFrame(
            {"time": df_clearance_["t"],
             "value": df_clearance_["Right toe"] + torso_height})
        df_heel_right = pd.DataFrame(
            {"time": df_clearance_["t"],
             "value": df_clearance_["Right heel"] + torso_height})

        paretic_side = metadata.paretic_side

        if paretic_side == Side.LEFT:
            df_toe_paretic = df_toe_left
            df_heel_paretic = df_heel_left
            df_toe_nonparetic = df_toe_right
            df_heel_nonparetic = df_heel_right
        else:
            df_toe_paretic = df_toe_right
            df_heel_paretic = df_heel_right
            df_toe_nonparetic = df_toe_left
            df_heel_nonparetic = df_heel_left

        ignore_cycle = (metadata.ignore_cycle[0], metadata.ignore_cycle[1])
        max_toe, impulse_toe, stance_ = DataProcess.data_process(
            df_toe_paretic,
            df_toe_nonparetic,
            paretic_gait_path,
            non_paretic_gait_path,
            save_path=report_save_path,
            data_label="Clearance [mm]",
            title_label="Toe Clearance",
            ignore_cycle=ignore_cycle,
            start_time=start_time,
            max_flag=True,
            clearance_flag=True,
            report_start_time=report_start_time,
            report_duration=report_duration,
            idx_gait_event_filter=current_gait_event_filter
        )

        max_heel, impulse_heel, stance_ = DataProcess.data_process(
            df_heel_paretic,
            df_heel_nonparetic,
            paretic_gait_path,
            non_paretic_gait_path,
            save_path=report_save_path,
            data_label="Clearance [mm]",
            title_label="Heel Clearance",
            ignore_cycle=ignore_cycle,
            start_time=start_time,
            max_flag=True,
            clearance_flag=True,
            report_start_time=report_start_time,
            report_duration=report_duration,
            idx_gait_event_filter=current_gait_event_filter
        )

        return max_toe

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
