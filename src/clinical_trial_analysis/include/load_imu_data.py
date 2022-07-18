import pandas as pd
import numpy as np
import math


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
    df_xls = df_xls.reset_index().rename(columns={'index':'time'}).iloc[:, :]
    df_xls['time'] *= time_scale

    return df_xls


def load_imu(file_path):
    imu_sheet_name = [
        "LEFT_IMU_0", "LEFT_IMU_1", "LEFT_IMU_4", "LEFT_IMU_5",
        "RIGHT_IMU_1", "RIGHT_IMU_4", "RIGHT_IMU_5"
    ]

    df_imu = [
        pd.read_excel(file_path, sheet_name=sheet_name, usecols="B, D:N")
        for sheet_name in imu_sheet_name]

    for i in range(len(df_imu)):
        imu = df_imu[i]
        ser_accel = pd.Series(name='abs_accel')
        ser_gyro = pd.Series(name='abs_gyro')

        for index, row in imu.iterrows():
            abs_accel = math.sqrt(row.iloc[-1] ** 2 + row.iloc[-2] ** 2 + row.iloc[-3] ** 2)
            abs_gyro = math.sqrt(row.iloc[-4] ** 2 + row.iloc[-5] ** 2 + row.iloc[-6] ** 2)
            ser_accel = pd.concat([ser_accel, pd.Series(abs_accel)])
            ser_gyro = pd.concat([ser_gyro, pd.Series(abs_gyro)])

        ser_accel = ser_accel.reset_index()
        ser_gyro = ser_gyro.reset_index()
        df_imu[i] = pd.concat([imu, ser_accel], axis=1)

    return df_imu


# To test load_xls function
def test():
    test_file_path = '../../../data/RH-07/TRIMMED_WALK01.XLS'
    df = load_xls(test_file_path)
    print(df)


if __name__ == '__main__':
    test()