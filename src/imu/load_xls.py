import pandas as pd
import numpy as np


# Load .XLS file from didim data and add time scale
def load_xls(file_path):
    # (raw data freq) = 600
    # (xls data freq) = (raw data freq) / 5
    time_scale = 5 / 600

    df_xls = pd.read_csv(
        file_path, skiprows=23, sep='\t'
    )
    df_xls = df_xls.reset_index().rename(columns={'index':'time'}).iloc[:, :-1]
    df_xls['time'] *= time_scale
    return df_xls


# To test load_xls function
def test():
    test_file_path = '../../data/220119/didim/TRIMMED_WALK01.XLS'
    df = load_xls(test_file_path)
    print(df)


if __name__ == '__main__':
    test()