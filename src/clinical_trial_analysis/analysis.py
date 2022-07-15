from common.load_xls import load_xls


MAX_TRIAL = 100


def get_full_file_path(prefix, suffix, index):
    path = ""
    for pre in prefix:
        path += pre

    # To make 2 digit number
    path += str(index).zfill(2)

    for suf in suffix:
        path += suf

    return path


"""
이민희 잘해라
"""
def lmh(trial_num, walk_num):
    return df_didim_GRF, df_fsr

"""
여기까지
"""

def kjj(trial_num, walk_num):
    # Define File Paths
    path_test = '../../data/RH-09/'
    index_list = []

    # Use key value for purposes.
    # 'device' for our own device data
    # 'motion_capture' for didim data
    data_type_prefix = {'device': 'data/', 'motion_capture': ''}
    file_name_prefix = {'device': 'temp', 'motion_capture': 'TRIMMED_WALK'}
    file_name_suffix = {'device': 'temp', 'motion_capture': '.XLS'}

    # Load data and make dataframe
    df_didim = []
    key = 'motion_capture'

    for i in range(1, MAX_TRIAL):
        full_path_didim = get_full_file_path(
            prefix=[path_test, data_type_prefix[key], file_name_prefix[key]],
            suffix=[file_name_suffix[key]],
            index=i
        )
        #
        try:
            df_didim.append(load_xls(full_path_didim))
        except FileNotFoundError:
            print("END of reading files")
            break
    print(len(df_didim))
    """
    Kinematic Data Plot

     1. Roll, Pitch, Yaw Angle of 7 limbs.

    """
    #


return df_didim_kinematics


def main():

    return 0


if __name__ == "__main__":
    main()
