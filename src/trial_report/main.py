from bagpy import bagreader
import pandas as pd
from jj import ClinicalIndexJJ
from mh import ClinicalIndexMH


# return some clinical index -> side-dependent case
# 1. Give each value of paretic and non-paretic side
# 2. Give symmetry index by a formula
# paretic / (paretic + non-paretic) * 100
def get_symmetry_index_sth(filePath):
    # Temporary variable for template
    paretic = 0
    non_paretic = 0
    symmetry = 0.55

    ...

    return paretic, non_paretic, symmetry


# return some clinical index -> side-independent case
# Just give single value for whole trial
def get_clinical_index_sth(file_path):
    clinical_index = 0.0

    ...

    return clinical_index


# 0~100% (interval: 1%)
def get_cycle_mean_data_sth(filepath):
    df_mean_data = pd.DataFrame()

    ...

    return df_mean_data   # Form of each data = (cycle_percentage, data)


def main():
    # Read bag file
    path = "../../bag/log_2023-08-04-16-26-19.bag"
    save_path = '../../data/report/report_df_230816_.csv'
    bag = bagreader(path)
    start_time = bag.start_time

    # To filter specific topics with interests
    TOPIC_MH = (
        "/afo_sensor/soleSensor_left",
        "/afo_sensor/soleSensor_right",
        "/afo_detector/gait_paretic",
        "/afo_detector/gait_nonparetic"
    )

    TOPIC_JJ = ("/afo_gui/left_toe_clearance",
                "/afo_gui/right_toe_clearance",
                "/afo_gui/stride")

    # Definition of Clinical Indices
    # Sample
    cadence_trial_mean = 0.0

    left_toe_path = ""
    right_toe_path = ""
    stride_path = ""
    
    left_sole_path = ""
    right_sole_path = ""
    paretic_gait_path = ""
    nonparetic_gait_path = ""
    
    calib_model_path = '../../model/CHAR_230815_280_LP/'
    GRF_model_path = '../../model/GRF_230815/LSTM_GRF.pt'
    
    BW = float(85.1) # 장비무게포함
    paretic_side = 'L'
    size = '280'

    # Read Topics and calculate clinical index
    for topic in bag.topics:
        msg_topic = bag.message_by_topic(topic)

        # Use own module and methods
        if topic == TOPIC_MH[0]:
            left_sole_path = msg_topic
        elif topic == TOPIC_MH[1]:
            right_sole_path = msg_topic
        elif topic == TOPIC_MH[2]:
            paretic_gait_path = msg_topic
        elif topic == TOPIC_MH[3]:
            nonparetic_gait_path = msg_topic

        if topic == TOPIC_JJ[0]:
            left_toe_path = msg_topic
        elif topic == TOPIC_JJ[1]:
            right_toe_path = msg_topic
        elif topic == TOPIC_JJ[2]:
            stride_path = msg_topic
        
    report_df = pd.DataFrame(columns = ['mean_paretic', 'std_paretic',
                                        'mean_nonparetic', 'std_nonparetic',
                                        'symmetry'],
                             index = ['toeClearance', 'stride', 'GRFmax'
                                      'GRFimpulse', 'stanceTime'])
    # toe_clearance_data = \
    #     ClinicalIndexJJ.get_clinical_index_max_toe_clearance(
    #         left_toe_path, right_toe_path)

    # stride_data = ClinicalIndexJJ.get_clinical_index_gait_speed_imu(
    #     stride_path)
    
    GRF_maximum_data = \
        ClinicalIndexMH.get_symmetry_index_GRFmax(start_time = start_time,
                                                  leftPath = left_sole_path,
                                                  rightPath = right_sole_path,
                                                  pareticPath = paretic_gait_path,
                                                  nonpareticPath = nonparetic_gait_path,
                                                  modelPathCalib = calib_model_path,
                                                  modelPathGRF = GRF_model_path,
                                                  size=size,
                                                  paretic_side=paretic_side,
                                                  BW=BW)

    GRF_impulse_data = \
        ClinicalIndexMH.get_symmetry_index_GRFimpulse(start_time = start_time,
                                                      leftPath = left_sole_path,
                                                      rightPath = right_sole_path,
                                                      pareticPath = paretic_gait_path,
                                                      nonpareticPath = nonparetic_gait_path,
                                                      modelPathCalib = calib_model_path,
                                                      modelPathGRF = GRF_model_path,
                                                      size=size,
                                                      paretic_side=paretic_side,
                                                      BW=BW)

    stance_time_data = ClinicalIndexMH.get_symmetry_index_stanceTime(
        start_time = start_time,
        pareticPath = paretic_gait_path,
        nonpareticPath = nonparetic_gait_path,
        paretic_side='L')
    
    
    # add report df
    # report_df.loc['toeClearance', :] = toe_clearance_data
    # report_df.loc['stride', :] = stride_data
    report_df.loc['GRFmax', :] = GRF_maximum_data
    report_df.loc['GRFimpulse', :] = GRF_impulse_data
    report_df.loc['stanceTime', :] = stance_time_data

    print(report_df)
    report_df.to_csv(save_path, sep=',',
                     columns = ['mean_paretic', 'std_paretic',
                                'mean_nonparetic', 'std_nonparetic',
                                'symmetry'],
                     index = ['toeClearance', 'stride', 'GRFmax',
                              'GRFimpulse', 'stanceTime'])
    # Document Formatting
    ...


if __name__ == "__main__":
    main()
