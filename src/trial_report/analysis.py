# from src.trial_report.utils import *
# from src.trial_report.define import *
# from src.trial_report.mh import GRF_predictor

from utils import *
from define import *
from mh import GRF_predictor


class ClinicalAnalysis:
    @staticmethod
    def data_analysis_grf(
            meta_data,
            default_path,
            start_time,
            report_start_time,
            report_duration,
            gait_event_filter,
            left_path,
            right_path,
            paretic_path,
            non_paretic_path,
            save_each_cycle_flag=False,
            save_label=None
            ):
        # parameter assign
        model_path_cell = default_path + "/model/" +\
            meta_data.model_cell + "/"
        model_path_grf = default_path + "/model/" +\
            meta_data.model_grf
        sole_size = str(meta_data.sole_size)
        paretic_side = meta_data.paretic_side.value[0]  # L or R
        body_weight = meta_data.body_weight
        ignore_cycle = (meta_data.ignore_cycle[0], meta_data.ignore_cycle[1])
        report_save_path = default_path + "/report/data/" +\
            meta_data.test_label + "/" + meta_data.session_type
        inference_data_save_path = report_save_path +\
            "/process_data/inference_data/"

        create_folder(inference_data_save_path)
        #####################################################################
        # GRF class assign
        #  +
        # "piecewise_linear_info_30N_ver2.json"
        grf_class = GRF_predictor(
            start_time=start_time,
            report_start=report_start_time,
            report_duration=report_duration,
            left_path=left_path,
            right_path=right_path,
            paretic_path=paretic_path,
            nonparetic_path=non_paretic_path,
            calib_model_path=model_path_cell,
            grf_model_path=model_path_grf,
            save_path=inference_data_save_path,
            size=sole_size,
            paretic_side=paretic_side,
            body_weight=body_weight
        )
        if meta_data.paretic_side == Side.LEFT:
            paretic_data = grf_class.left_grf
            non_paretic_data = grf_class.right_grf

            paretic_extra_data = grf_class.left_force
            non_paretic_extra_data = grf_class.right_force
        else:
            paretic_data = grf_class.right_grf
            non_paretic_data = grf_class.left_grf

            paretic_extra_data = grf_class.right_force
            non_paretic_extra_data = grf_class.left_force

        if (save_each_cycle_flag):
            DataProcess.graph_each_cycle_data(
                paretic_data,
                non_paretic_data,
                paretic_extra_data,
                non_paretic_extra_data,
                paretic_path,
                non_paretic_path,
                save_path=report_save_path,
                data_label=("grf", "force"),
                unit_label=("[N]", "[N]"),
                title_label=("vGRF", "Force"),
                ignore_cycle=ignore_cycle,
                start_time=start_time,
                report_start_time=report_start_time,
                report_duration=report_duration,
                idx_gait_event_filter=gait_event_filter,
                save_label=save_label
            )

        max_array, impulse_, stance_ = DataProcess.data_process(
            paretic_data,
            non_paretic_data,
            paretic_path,
            non_paretic_path,
            save_path=report_save_path,
            data_label="vGRF [N]",
            title_label=save_label+"vGRF",
            ignore_cycle=ignore_cycle,
            start_time=start_time,
            max_flag=True,
            impulse_flag=False,
            stance_flag=False,
            report_start_time=report_start_time,
            report_duration=report_duration,
            idx_gait_event_filter=gait_event_filter
            # grf_flag=True
        )
        
        max_, impulse_array, stance_ = DataProcess.data_process(
            paretic_data,
            non_paretic_data,
            paretic_path,
            non_paretic_path,
            save_path=report_save_path,
            data_label="vGRF Impulse [N sec]",
            title_label=save_label+"vGRF",
            ignore_cycle=ignore_cycle,
            start_time=start_time,
            max_flag=False,
            impulse_flag=True,
            stance_flag=False,
            report_start_time=report_start_time,
            report_duration=report_duration,
            idx_gait_event_filter=gait_event_filter
            # grf_flag=True
        )
        
        max_, impulse_, stance_array = DataProcess.data_process(
            paretic_data,
            non_paretic_data,
            paretic_path,
            non_paretic_path,
            save_path=report_save_path,
            data_label="vGRF [N]",
            title_label=save_label,
            ignore_cycle=ignore_cycle,
            start_time=start_time,
            max_flag=False,
            impulse_flag=False,
            stance_flag=True,
            report_start_time=report_start_time,
            report_duration=report_duration,
            idx_gait_event_filter=gait_event_filter
            # grf_flag=True
        )

        return max_array, impulse_array, stance_array

    @staticmethod
    def data_analysis_stance_time(
            start_time,
            paretic_path, non_paretic_path,
            ignore_cycle=(None, None)
    ):

        df_paretic_gait = DataProcess.read_data_file_by_path(
            paretic_path)
        df_non_paretic_gait = DataProcess.read_data_file_by_path(
            non_paretic_path)

        df_paretic_gait.iloc[:, 0] -= start_time
        df_non_paretic_gait.iloc[:, 0] -= start_time

        df_paretic_gait = \
            get_ignored_cycle(df_paretic_gait, ignore_cycle)
        df_non_paretic_gait = \
            get_ignored_cycle(df_non_paretic_gait, ignore_cycle)

        [_, _, _, _, _, _,
         paretic_mean, paretic_stdev,
         non_paretic_mean, non_paretic_stdev] =\
            DataProcess.gait_phase_pre_processing(
                df_paretic_gait, df_non_paretic_gait)

        symmetry = paretic_mean / (paretic_mean + non_paretic_mean) * 100

        return [paretic_mean, paretic_stdev,
                non_paretic_mean, non_paretic_stdev, symmetry]

    @staticmethod
    def data_analysis_toe_clearance():
        ...

    @staticmethod
    def data_analysis_analog_data():
        ...

