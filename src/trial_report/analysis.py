from bagpy import bagreader
import os
import glob
import time
import pandas as pd
import numpy as np
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import *
from mh import GRF_predictor


class ClinicalAnalysis:
    @staticmethod
    def data_analysis_grf(
        start_time, left_path, right_path,
        paretic_path, non_paretic_path,
        model_path_calib, model_path_grf,
        raw_data_save_path, cycle_timeseries_data_save_path,
        size='280', paretic_side='L',
        body_weight=float(100), ignore_cycle=(None, None)
    ):
        grf_class = GRF_predictor(
            start_time=float(start_time),
            leftPath=left_path,
            rightPath=right_path,
            pareticPath=paretic_path,
            nonpareticPath=non_paretic_path,
            modelPathCalib=model_path_calib,
            modelPathGRF=model_path_grf,
            save_path=raw_data_save_path,
            size=size,
            paretic_side=paretic_side,
            BW=body_weight
        )
        if paretic_side == 'L':
            paretic_data = grf_class.left_grf
            non_paretic_data = grf_class.right_grf
        else:
            paretic_data = grf_class.right_grf
            non_paretic_data = grf_class.left_grf

        max_array, impulse_array = DataProcess.data_process(
            paretic_data,
            non_paretic_data,
            paretic_path,
            non_paretic_path,
            save_path=cycle_timeseries_data_save_path,
            data_label="grf [N]",
            title_label="GRF",
            ignore_cycle=ignore_cycle,
            start_time=start_time,
            max_flag=True,
            impulse_flag=True
        )

        return max_array, impulse_array

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

    def data_analysis_toe_clearance:
        ...

    def data_analysis_analog_data:
        ...

