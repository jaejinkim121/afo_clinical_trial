from bagpy import bagreader
import pandas as pd
import numpy as np
import csv


class ClinicalIndexJJ:
    @staticmethod
    def get_clinical_index_max_toe_clearance(paretic_path, non_paretic_path):
        paretic_toe_clearance = []
        non_paretic_toe_clearance = []
        # paretic
        with open(paretic_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for csv_row in csv_reader:
                paretic_toe_clearance.append(float(csv_row[1]))

        with open(non_paretic_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for csv_row in csv_reader:
                non_paretic_toe_clearance.append(float(csv_row[1]))

        np_ptc = np.array(paretic_toe_clearance)
        np_nptc = np.array(non_paretic_toe_clearance)

        paretic_mean = np.mean(np_ptc)
        paretic_stdev = np.std(np_ptc)
        non_paretic_mean = np.mean(np_nptc)
        non_paretic_stdev = np.std(np_nptc)
        symmetry = paretic_mean / (paretic_mean + non_paretic_mean)

        return (paretic_mean, paretic_stdev,
                non_paretic_mean, non_paretic_stdev,
                symmetry
                )

    @staticmethod
    def get_clinical_index_gait_speed_imu(stride_path):
        stride = []
        with open(stride_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for csv_row in csv_reader:
                stride.append(float(csv_row[1]))
        stride = np.array(stride)

        return np.sum(stride) / 120.0

    @staticmethod
    def get_clinical_index_gait_speed_distance(file_path):
        clinical_index = 0.0

        ...

        return clinical_index
