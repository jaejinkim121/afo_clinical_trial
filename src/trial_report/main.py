from bagpy import bagreader
import pandas as pd

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
    bag = bagreader(path)
    start_time = bag.start_time
    print(start_time)

    # To filter specific topics with interests
    TOPIC_MH = (
        "/afo_sensor/soleSensor_left",
        "/afo_sensor/soleSensor_right",
        "/afo_detector/gait_paretic",
        "/afo_detector/gait_nonparetic"
    )

    TOPIC_JJ = ()

    # Definition of Clinical Indices
    # Sample
    cadence_trial_mean = 0.0

    # Read Topics and calculate clinical index
    for topic in bag.topics:
        msg_topic = bag.message_by_topic(topic)
        print(msg_topic)

        # Use own module and methods
        if topic in TOPIC_MH:
            msg_topic = bag.message_by_topic(topic)
            print(msg_topic)

        if topic in TOPIC_JJ:
            ...

    # Document Formatting
    ...


if __name__ == "__main__":
    main()
