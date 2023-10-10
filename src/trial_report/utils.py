import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from bagpy import bagreader


SAVE_EACH_CYCLE_DATA = True


# Create Directory
def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def get_ignored_cycle(array_df, cycle_num):
    if cycle_num[1] is None:
        array_df = array_df[cycle_num[0]:]
    else:
        array_df = array_df[cycle_num[0]:-cycle_num[1]]
    return array_df


def get_torque_info(path, start_time):
    df_command = pd.DataFrame()
    df_actual = pd.DataFrame()

    command_data = []
    actual_data = []
    time_data = []

    with open(path) as csvfile:
        csv_reader = csv.reader(csvfile)
        for csv_row in csv_reader:
            try:
                time_data.append(float(csv_row[0]) - start_time)
                command_data.append(float(csv_row[2]))
                actual_data.append(float(csv_row[5]))
            except ValueError:
                continue

    df_command["time"] = time_data
    df_command["command"] = command_data
    df_actual["time"] = time_data
    df_actual["actual"] = actual_data

    return df_command, df_actual


def save_each_cycle_timeseries_data(
        collection_data, data_label, title_label, color, save_path):
    for num, data in enumerate(collection_data):
        fig = plt.figure(figsize=(16, 8))
        plt.plot(data[:, 0], data[:, 1], color=color)
        plt.xlabel("Time [s]", fontsize=30)
        plt.ylabel(data_label, fontsize=30)
        plt.title(title_label + '_' + str(num), fontsize=45)
        # plt.show()
        fig.savefig(
            save_path + title_label + '_' + str(num) + '.png'
            )


def save_each_cycle_bar_plot(data_paretic, data_non_paretic,
                             data_label, title_label, save_path):
    # 2 array input
    # path assign
    graph_save_path = save_path + "/graph/"
    data_save_path = save_path + "/process_data/"
    # Plotting & Saving
    np_paretic = np.array(data_paretic)
    np_non_paretic = np.array(data_non_paretic)

    fig = plt.figure(figsize=(16, 8))
    plt.plot(
        np.arange(1, len(np_paretic) + 1), np_paretic,
        color='red', label='paretic side'
        )
    plt.plot(
        np.arange(1, len(np_non_paretic) + 1), np_non_paretic,
        color='blue', label='non-paretic side'
        )
    plt.xlabel("Gait cycle number", fontsize=30)
    plt.ylabel(data_label, fontsize=30)
    plt.title(title_label, fontsize=45)
    plt.legend(loc='best', fontsize=25)
    create_folder(graph_save_path)
    fig.savefig(graph_save_path + title_label + '_along_cycle.png')
    # df saving
    df_data = pd.DataFrame(columns=['paretic side', 'non-paretic side'])
    df_data['paretic side'] = np_paretic
    df_data['non-paretic side'] = np_non_paretic
    df_data.to_csv(data_save_path + title_label + '_along_cycle.csv',
                   sep=",", header=True)


def match_both_side_cycle(collection_paretic, collection_non_paretic,
                          df_gait_paretic, df_gait_non_paretic):
    time_ic_paretic = \
        DataProcess.get_initial_contact_time(df_gait_paretic)
    time_ic_non_paretic = \
        DataProcess.get_initial_contact_time(df_gait_non_paretic)
    collection_paretic_matched = []
    collection_non_paretic_matched = []
    gait_paretic_matched = []
    gait_non_paretic_matched = []

    idx_p, idx_np = (0, 0)

    while True:
        if idx_np >= len(time_ic_non_paretic):
            break
        if idx_p >= len(time_ic_paretic):
            break

        cur_p = time_ic_paretic[idx_p]
        cur_np = time_ic_non_paretic[idx_np]

        if idx_p == len(time_ic_paretic) - 1:
            next_p = cur_np + 1
        else:
            next_p = time_ic_paretic[idx_p + 1]

        if cur_p > cur_np:
            idx_np += 1
            continue
        if next_p < cur_np:
            idx_p += 1
            continue

        collection_paretic_matched.append(collection_paretic[idx_p])
        collection_non_paretic_matched.append(collection_non_paretic[idx_np])
        gait_paretic_matched.append(df_gait_paretic[idx_p])
        gait_non_paretic_matched.append(df_gait_non_paretic[idx_np])
        idx_p += 1

    return (collection_paretic_matched, collection_non_paretic_matched,
            gait_paretic_matched, gait_non_paretic_matched)


class DataProcess:
    @staticmethod
    def normalize_time_series(
            x: np.array, y: np.array, x_target=None, x_num=100):
        if x_target is None:
            x_target = np.linspace(0, 1, x_num)

        x_target = x_target / x_target[-1]
        x -= x[0]
        x = x / x[-1]
        y_n = np.interp(x_target, x, y)

        return y_n

    @staticmethod
    def average_time_series(x, y, x_start, x_end=None, x_num=101):
        if x_end is None:
            x_end = copy.deepcopy(x_start[1:])
            x_end = np.append(x_end, x[-1])

        if len(x_start) != len(x_end):
            print("start and end indices must have same lengths")
            return None

        ind_start = 0
        ind_end = 0
        crop_time_series = []

        for i in range(len(x_start)):
            target = x_start[i]
            ind_start = DataProcess.search_index(x[ind_end:], target) + ind_end
            target = x_end[i]
            ind_end = DataProcess.search_index(x[ind_start:], target) + ind_start

            y_cropped = DataProcess.normalize_time_series(
                x[ind_start:ind_end+1],
                y[ind_start:ind_end+1],
                x_num=x_num)
            crop_time_series.append(y_cropped)

        crop_time_series = np.array(crop_time_series)
        mean_time_series = np.mean(crop_time_series, axis=0)
        std_time_series = np.std(crop_time_series, axis=0)

        return mean_time_series, std_time_series

    @staticmethod
    def average_cropped_time_series(collection_data, x_num=101):
        crop_time_series = []
        for data in collection_data:
            y_cropped = DataProcess.normalize_time_series(data[:, 0],
                                                          data[:, 1],
                                                          x_num=x_num)
            crop_time_series.append(y_cropped)
        crop_time_series = np.array(crop_time_series)
        mean_time_series = np.mean(crop_time_series, axis=0)
        std_time_series = np.std(crop_time_series, axis=0)

        return mean_time_series, std_time_series

    @staticmethod
    def search_index(sorted_array, search_target):
        ind = 0
        for x in sorted_array:
            if x < search_target:
                ind += 1
                continue
            else:
                return ind

        return -1

    @staticmethod
    def get_initial_contact_time(gait_phase: pd.DataFrame):
        time_initial_contact = \
            gait_phase[gait_phase.iloc[:, 1] == 1].iloc[:, 0]
        return time_initial_contact.tolist()

    @staticmethod
    def get_foot_off_time(gait_phase: pd.DataFrame):
        time_foot_off = \
            gait_phase[gait_phase.iloc[:, 1] == 2].iloc[:, 0]

        return time_foot_off.tolist()

    @staticmethod
    def get_gait_event_time(gait_phase: pd.DataFrame):
        time_initial_contact = \
            DataProcess.get_initial_contact_time(gait_phase)
        time_foot_off = \
            DataProcess.get_foot_off_time(gait_phase)

        return time_initial_contact, time_foot_off

    @staticmethod
    def gait_phase_pre_processing(
            gait_phase_paretic: pd.DataFrame,
            gait_phase_nonparetic: pd.DataFrame):
        paretic_ic, paretic_fo = \
            DataProcess.get_gait_event_time(gait_phase_paretic)
        nonparetic_ic, nonparetic_fo = \
            DataProcess.get_gait_event_time(gait_phase_nonparetic)

        mean_cycle = (paretic_ic[-1] - paretic_ic[0]) / (len(paretic_ic) - 1)
        mean_cycle = mean_cycle / 100.0
        idx_np = 0
        time_diff = []
        for tic in paretic_ic:
            idx_np += DataProcess.search_index(nonparetic_ic[idx_np:], tic)
            time_diff.append(nonparetic_ic[idx_np] - tic)

        time_diff = np.array(time_diff)
        mean_ic_diff = np.mean(time_diff) / mean_cycle
        std_ic_diff = np.std(time_diff) / mean_cycle

        if paretic_ic[0] > paretic_fo[0]:
            paretic_fo = paretic_fo[1:]
        if paretic_ic[-1] > paretic_fo[-1]:
            paretic_ic = paretic_ic[:-1]

        time_diff = []
        for tic, tfo in zip(paretic_ic, paretic_fo):
            time_diff.append(tfo - tic)
        time_diff = np.array(time_diff)
        mean_stance_time_paretic = np.mean(time_diff)
        std_stance_time_paretic = np.std(time_diff)
        mean_stance_percent_paretic = mean_stance_time_paretic / mean_cycle
        std_stance_percent_paretic = std_stance_time_paretic / mean_cycle

        if nonparetic_ic[0] > nonparetic_fo[0]:
            nonparetic_fo = nonparetic_fo[1:]
        if nonparetic_ic[-1] > nonparetic_fo[-1]:
            nonparetic_ic = nonparetic_ic[:-1]

        time_diff = []
        for tic, tfo in zip(nonparetic_ic, nonparetic_fo):
            time_diff.append(tfo - tic)
        time_diff = np.array(time_diff)
        mean_stance_time_nonparetic = np.mean(time_diff)
        std_stance_time_nonparetic = np.std(time_diff)
        mean_stance_percent_nonparetic = mean_stance_time_nonparetic / mean_cycle
        std_stance_percent_nonparetic = std_stance_time_nonparetic / mean_cycle

        return [mean_ic_diff, std_ic_diff,
                mean_stance_percent_paretic, std_stance_percent_paretic,
                mean_stance_percent_nonparetic, std_stance_percent_nonparetic,
                mean_stance_time_paretic, std_stance_time_paretic,
                mean_stance_time_nonparetic, std_stance_time_nonparetic]

    @staticmethod
    def read_data_file_by_path(data_path):
        data = pd.DataFrame()

        value_data = []
        time_data = []

        with open(data_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for csv_row in csv_reader:
                try:
                    time_data.append(float(csv_row[0]))
                    value_data.append(float(csv_row[1]))
                except ValueError:
                    continue

        data["time"] = time_data
        data["value"] = value_data
        return data

    @staticmethod
    def divider_data_by_gait_phase_df(data_df, gait_phase_df, ignore_cycle):
        divided_array = []
        time_initial_contact = \
            DataProcess.get_initial_contact_time(gait_phase_df)
        time_initial_contact.append(data_df["time"].iloc[-1])

        for i in range(len(time_initial_contact) - 1):
            divided_df_current = \
                data_df[
                    (data_df["time"] >= time_initial_contact[i]) &
                    (data_df["time"] < time_initial_contact[i + 1])
                    ]
            divided_array.append(divided_df_current.to_numpy())

        divided_array = get_ignored_cycle(divided_array, ignore_cycle)

        return divided_array

    @staticmethod
    def graph_averaged_data(collection_data, title_graph, data_label,
                            x_num=101):
        mean, std = DataProcess.average_cropped_time_series(
            collection_data, x_num
        )
        x = np.linspace(0, 100, x_num)
        plt.plot(x, mean, 'k-')
        plt.fill_between(x, mean - std, mean + std)
        plt.xlabel("Cycle Percentage")
        plt.xlim(0, 100)
        plt.ylabel(data_label)
        plt.title(title_graph)
        plt.show()

    @staticmethod
    def graph_both_cycle_data(collection_data_paretic,
                              collection_data_nonparetic,
                              data_gait_paretic, data_gait_nonparetic,
                              data_label, title_graph, save_path,
                              x_num=101):
        [mean_diff_both, std_diff_both,
         mean_diff_paretic, std_diff_paretic,
         mean_diff_nonparetic,
         std_diff_nonparetic,
         _, _, _, _] = DataProcess.gait_phase_pre_processing(
            data_gait_paretic, data_gait_nonparetic)

        mean_paretic, std_paretic = DataProcess.average_cropped_time_series(
            collection_data_paretic, x_num
        )
        mean_nonparetic, std_nonparetic = \
            DataProcess.average_cropped_time_series(
                collection_data_nonparetic, x_num
            )

        x_paretic = np.linspace(0, 100, x_num)
        x_nonparetic = np.linspace(mean_diff_both, 100 + mean_diff_both,
                                   x_num)

        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]})
        fig.subplots_adjust(hspace=0.3)

        axs[0].plot(x_paretic, mean_paretic, 'r-')
        axs[0].fill_between(x_paretic,
                            mean_paretic - std_paretic,
                            mean_paretic + std_paretic,
                            color=(0.9, 0.1, 0.1, 0.2),
                            linewidth=0)
        axs[0].plot(x_nonparetic, mean_nonparetic, 'b-')
        axs[0].fill_between(x_nonparetic,
                            mean_nonparetic - std_nonparetic,
                            mean_nonparetic + std_nonparetic,
                            color=(0.1, 0.1, 0.9, 0.2),
                            linewidth=0)

        axs[0].set_ylabel(data_label)
        axs[0].set_title(title_graph)

        axs[1].axis('off')
        axs[1].barh(
            np.arange(2),
            [mean_diff_both + mean_diff_nonparetic + std_diff_nonparetic,
             mean_diff_paretic + std_diff_paretic],
            color=[[0.1, 0.1, 0.9, 0.2], [0.9, 0.1, 0.1, 0.2]]
        )
        axs[1].barh(
            np.arange(2),
            [mean_diff_both + mean_diff_nonparetic - std_diff_nonparetic,
             mean_diff_paretic - std_diff_paretic],
            color=['w', 'w'],
        )
        axs[1].barh(np.arange(2), [100, 100],
                    left=[mean_diff_both, 0],
                    color=[[0, 0, 0, 0], [0, 0, 0, 0]],
                    edgecolor=['k', 'k'])
        axs[1].barh(np.arange(1),
                    2 * std_diff_both,
                    left=mean_diff_both - std_diff_both,
                    color=[0.2, 0.2, 0.2, 0.2]
                    )

        axs[1].plot([mean_diff_paretic, mean_diff_paretic], [0.6, 1.38], 'r-')
        axs[1].plot([mean_diff_nonparetic + mean_diff_both,
                     mean_diff_nonparetic + mean_diff_both],
                    [-0.4, 0.37], 'b-')

        xlim_upper = max(mean_diff_nonparetic + std_diff_nonparetic + 1,
                         101 + mean_diff_both)

        axs[0].set_xlim(0, xlim_upper)
        axs[1].set_xlim(0, xlim_upper)
        create_folder(save_path)
        fig.savefig(save_path + '%s_mean_cycle.png' % title_graph)

    @staticmethod
    def data_process(
            paretic_data,
            non_paretic_data,
            paretic_gait_path,
            non_paretic_gait_path,
            save_path,  # {test_label}/{session_type}
            data_label="data",
            title_label="data",
            ignore_cycle=(None, None),
            start_time=0.0,
            max_flag=True,
            impulse_flag=False,
            stance_flag=False
    ):
        if type(paretic_data) != pd.DataFrame:
            df_paretic = \
                DataProcess.read_data_file_by_path(paretic_data)
        else:
            df_paretic = copy.deepcopy(paretic_data)

        if type(non_paretic_data) != pd.DataFrame:
            df_non_paretic = \
                DataProcess.read_data_file_by_path(non_paretic_data)
        else:
            df_non_paretic = copy.deepcopy(non_paretic_data)
        df_paretic_gait = DataProcess.read_data_file_by_path(
            paretic_gait_path)
        df_non_paretic_gait = DataProcess.read_data_file_by_path(
            non_paretic_gait_path)
        df_paretic_gait.iloc[:, 0] -= start_time
        df_non_paretic_gait.iloc[:, 0] -= start_time

        collection_paretic = \
            DataProcess.divider_data_by_gait_phase_df(
                df_paretic, df_paretic_gait,
                ignore_cycle
            )

        collection_non_paretic = \
            DataProcess.divider_data_by_gait_phase_df(
                df_non_paretic,
                df_non_paretic_gait,
                ignore_cycle
            )

        if SAVE_EACH_CYCLE_DATA:
            save_each_cycle_timeseries_data(
                collection_paretic,
                data_label=data_label + "[N]",
                title_label=title_label,
                color='red',
                save_path=save_path + '/graph/each_cycle/paretic/')

            save_each_cycle_timeseries_data(
                collection_non_paretic,
                data_label=data_label + "[N]",
                title_label=title_label,
                color='blue',
                save_path=save_path + '/graph/each_cycle/non_paretic/')

        df_paretic_gait = \
            get_ignored_cycle(df_paretic_gait, ignore_cycle)
        df_non_paretic_gait = \
            get_ignored_cycle(df_non_paretic_gait, ignore_cycle)

        DataProcess.graph_both_cycle_data(
            collection_paretic,
            collection_non_paretic,
            df_paretic_gait, df_non_paretic_gait,
            data_label + '[N]', title_label,
            save_path=save_path + '/graph/',
            x_num=101
        )
        ###
        # Statistics Processing
        max_paretic_mean = 0
        max_paretic_stdev = 0
        max_non_paretic_mean = 0
        max_non_paretic_stdev = 0
        max_symmetry = 0
        if max_flag:
            max_paretic = []
            max_non_paretic = []

            for da in collection_paretic:
                max_paretic.append(
                    np.max(da[:, 1])
                )
            for da in collection_non_paretic:
                max_non_paretic.append(
                    np.max(da[:, 1])
                )

            np_p_max = np.array(max_paretic)
            np_np_max = np.array(max_non_paretic)
            save_each_cycle_bar_plot(
                np_p_max, np_np_max,
                data_label + '[N]', title_label + "_max",
                save_path
            )
            max_paretic_mean = np.mean(np_p_max)
            max_paretic_stdev = np.std(np_p_max)
            max_non_paretic_mean = np.mean(np_np_max)
            max_non_paretic_stdev = np.std(np_np_max)
            max_symmetry = 1 -\
                abs(max_paretic_mean - max_non_paretic_mean)\
                / (max_paretic_mean + max_non_paretic_mean)
        impulse_paretic_mean = 0
        impulse_paretic_stdev = 0
        impulse_non_paretic_mean = 0
        impulse_non_paretic_stdev = 0
        impulse_symmetry = 0

        if impulse_flag:
            impulse_paretic = []
            impulse_non_paretic = []

            for da in collection_paretic:
                impulse_paretic.append(
                    np.trapz(da[:, 1], x=da[:, 0])
                )
            for da in collection_non_paretic:
                impulse_non_paretic.append(
                    np.trapz(da[:, 1], x=da[:, 0])
                )

            np_p_impulse = np.array(impulse_paretic)
            np_np_impulse = np.array(impulse_non_paretic)
            save_each_cycle_bar_plot(
                np_p_impulse, np_np_impulse,
                data_label + '[N sec]', title_label + "_impulse",
                save_path
            )
            impulse_paretic_mean = np.mean(np_p_impulse)
            impulse_paretic_stdev = np.std(np_p_impulse)
            impulse_non_paretic_mean = np.mean(np_np_impulse)
            impulse_non_paretic_stdev = np.std(np_np_impulse)
            impulse_symmetry = 1 -\
                abs(impulse_paretic_mean - impulse_non_paretic_mean)\
                / (impulse_paretic_mean + impulse_non_paretic_mean)

        np_stance_paretic = []
        np_stance_non_paretic = []

        stance_paretic_mean = 0
        stance_paretic_stdev = 0
        stance_non_paretic_mean = 0
        stance_non_paretic_stdev = 0
        stance_symmetry = 0

        if stance_flag:
            save_each_cycle_bar_plot(
                np_stance_paretic, np_stance_non_paretic,
                'stance time [s]',  "Stance_Time",
                save_path
            )
            stance_paretic_mean = np.mean(np_stance_paretic)
            stance_paretic_stdev = np.std(np_stance_paretic)
            stance_non_paretic_mean = np.mean(np_stance_non_paretic)
            stance_non_paretic_stdev = np.std(np_stance_non_paretic)
            stance_symmetry = 1 -\
                abs(stance_paretic_mean - stance_non_paretic_mean)\
                / (stance_paretic_mean + stance_non_paretic_mean)


        return [max_paretic_mean, max_paretic_stdev,
                max_non_paretic_mean, max_non_paretic_stdev,
                max_symmetry
                ], [impulse_paretic_mean, impulse_paretic_stdev,
                    impulse_non_paretic_mean, impulse_non_paretic_stdev,
                    impulse_symmetry
                    ], [stance_paretic_mean, stance_paretic_stdev,
                        stance_non_paretic_mean, stance_non_paretic_stdev,
                        stance_symmetry
                        ]


def main():
    bag = bagreader("../../bag.bag")
    path_paretic = "../../grf_l.csv"
    path_nonparetic = "../../grf_r.csv"
    path_gait_paretic = "../../gait_p.csv"
    path_gait_nonparetic = "../../gait_np.csv"

    DataProcess.data_process(
        path_paretic,
        path_nonparetic,
        path_gait_paretic,
        path_gait_nonparetic,
        "../../",
        ignore_cycle=(None, 3),
        start_time=bag.start_time
    )


if __name__ == "__main__":
    main()