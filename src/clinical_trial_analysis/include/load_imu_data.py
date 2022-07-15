import pandas as pd
import numpy as np


# Load .XLS file from didim data and add time scale
def load_xls(file_path):
    # (raw data freq) = 600
    # (xls data freq) = (raw data freq) / 5
    time_scale = 5 / 600

    use_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8,
                12, 13, 14, 15, 16, 17, 18, 19, 20]

    df_xls = pd.read_csv(
        file_path, skiprows=23, sep='\t', usecols=use_cols
    )
    df_xls = df_xls.reset_index().rename(columns={'index':'time'}).iloc[:, :-1]
    df_xls['time'] *= time_scale

    return df_xls


def load_imu(file_path):
    imu_sheet_name = [
        "LEFT_IMU_0", "LEFT_IMU_1", "LEFT_IMU_4", "LEFT_IMU_5",
        "RIGHT_IMU_1", "RIGHT_IMU_4", "RIGHT_IMU_5"
    ]

    df_imu = [
        pd.read_excel(file_path, sheet_name=sheet_name, usecols="C, E:N")
        for sheet_name in imu_sheet_name]

    return df_imu


# To test load_xls function
def test():
    test_file_path = '../../../data/RH-07/TRIMMED_WALK01.XLS'
    df = load_xls(test_file_path)
    print(df)


if __name__ == '__main__':
    test()