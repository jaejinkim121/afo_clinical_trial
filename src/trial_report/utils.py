import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
import pandas as pd
import csv
import os
from bagpy import bagreader


SAVE_EACH_CYCLE_DATA = False


# Create Directory
def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def get_ignored_cycle(array_df, cycle_num, is_gait_phase=False):
    multiplier = 1 + is_gait_phase
    if cycle_num[1] is None:
        if cycle_num[0] is not None:
            array_df = array_df[cycle_num[0]*multiplier:]
    elif cycle_num[0] is not None:
        array_df = array_df[cycle_num[0]*multiplier:-cycle_num[1]*multiplier]
    else:
        array_df = array_df[:-cycle_num[1]*multiplier]
    return array_df


def get_index_outlier(df_gait_paretic, df_gait_non_paretic):
    paretic_ic, paretic_fo = \
        DataProcess.get_gait_event_time(df_gait_paretic)
    nonparetic_ic, nonparetic_fo = \
        DataProcess.get_gait_event_time(df_gait_non_paretic)

    ic_last_idx_paretic = None
    ic_last_idx_non_paretic = None
    if paretic_ic[0] > paretic_fo[0]:
        paretic_fo = paretic_fo[1:]

    if paretic_ic[-1] > paretic_fo[-1]:
        ic_last_idx_paretic = len(paretic_ic) - 1
        paretic_ic = paretic_ic[:-1]
    if nonparetic_ic[0] > nonparetic_fo[0]:
        nonparetic_fo = nonparetic_fo[1:]
    if nonparetic_ic[-1] > nonparetic_fo[-1]:
        ic_last_idx_non_paretic = len(nonparetic_ic) - 1
        nonparetic_ic = nonparetic_ic[:-1]

    time_diff_paretic = []
    time_diff_non_paretic = []
    for tic, tfo in zip(paretic_ic, paretic_fo):
        time_diff_paretic.append(tfo - tic)
    for tic, tfo in zip(nonparetic_ic, nonparetic_fo):
        time_diff_non_paretic.append(tfo - tic)
    #
    # picker_paretic = Picker(time_diff_paretic)
    # picker_non_paretic = Picker(time_diff_non_paretic)
    #
    # if ic_last_idx_paretic:
    #     picker_paretic.selected_idx.append(ic_last_idx_paretic)
    # if ic_last_idx_non_paretic:
    #     picker_non_paretic.selected_idx.append(ic_last_idx_non_paretic)
    return [],[], time_diff_paretic, time_diff_non_paretic
    # return picker_paretic.selected_idx, picker_non_paretic.selected_idx,\
    #     time_diff_paretic, time_diff_non_paretic


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
        collection_data, df_gait,
        data_label, unit_label, title_label, color, save_path
        ):
    create_folder(save_path)

    main_collection_data = collection_data[0]
    extra_collection_data = collection_data[1:]

    for cycle_num, main_data in enumerate(main_collection_data):
        start_time = main_data[0, 0]
        end_time = main_data[len(main_data) - 1, 0]
        # indexing gait dataframe
        df_gait_cycle = df_gait[
            (df_gait.iloc[:, 0] >= start_time) &
            (df_gait.iloc[:, 0] <= end_time)
            ]
        foot_off_timing = DataProcess.get_foot_off_time(df_gait_cycle)[0]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        ax1.plot(main_data[:, 0], main_data[:, 1],
                 color=color, label=data_label[0]
                 )
        ax1.vlines(foot_off_timing,
                   0, np.max(main_data[:, 1]), color='black',
                   linewidth=2
                   )
        ax1.set_ylabel(data_label[0] + '' + unit_label[0], fontsize=20)
        ax1.set_title(title_label[0] + '_' + str(cycle_num), fontsize=20)
        ax1.tick_params(axis='both', labelsize=15)
        ax1.legend(loc="best")

        extra_data_max = 0
        for extra_num, extra_dataset in enumerate(extra_collection_data):
            ax2.plot(
                extra_dataset[cycle_num][:, 0],
                extra_dataset[cycle_num][:, 1],
                label=data_label[1] + str(extra_num)
                )
            if extra_data_max <= np.max(extra_dataset[cycle_num][:, 1]):
                extra_data_max = np.max(extra_dataset[cycle_num][:, 1])
        ax1.vlines(foot_off_timing, 0, extra_data_max,
                   color='black', linewidth=2)
        ax2.set_xlabel("Time [s]", fontsize=20)
        ax2.set_ylabel(data_label[1] + '' + unit_label[1], fontsize=20)
        ax2.set_title(title_label[1] + '_' + str(cycle_num), fontsize=20)
        ax2.tick_params(axis='both', labelsize=15)
        ax2.legend(loc="best")
        fig.savefig(
            save_path + title_label[0] + '_' + title_label[1] + '_' +
            str(cycle_num) + '.png'
            )
        plt.close(fig)


def save_each_cycle_bar_plot(data_paretic, data_non_paretic,
                             mean_paretic, mean_non_paretic,
                             data_label, title_label, save_path):
    # 2 array input
    # path assign
    graph_save_path = save_path + "/graph/"
    data_save_path = save_path + "/process_data/"
    create_folder(graph_save_path)
    create_folder(data_save_path)
    # Plotting & Saving
    np_paretic = np.array(data_paretic)
    np_non_paretic = np.array(data_non_paretic)

    fig = plt.figure(figsize=(16, 12))
    plt.plot(
        np.arange(1, len(np_paretic) + 1), np_paretic,
        color='red', label='paretic side'
        )
    plt.plot(
        np.arange(1, len(np_non_paretic) + 1), np_non_paretic,
        color='blue', label='non-paretic side'
        )
    plt.axhline(y=mean_paretic, color=(1, 0, 0, 0.3), linewidth=2.5)
    plt.axhline(y=mean_non_paretic, color=(0, 0, 1, 0.3), linewidth=2.5)
    plt.xlabel("Gait cycle number", fontsize=30)
    plt.ylabel(data_label, fontsize=30)
    plt.title(title_label, fontsize=45)
    plt.legend(loc='best', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    create_folder(graph_save_path)
    fig.tight_layout()
    fig.savefig(graph_save_path + title_label + '_along_cycle.png')
    # df saving
    df_data = pd.DataFrame()
    df_data = pd.concat(
        [df_data, pd.Series(np_paretic, name='paretic side')], axis=0
        )
    df_data = pd.concat(
        [df_data, pd.Series(np_non_paretic, name='non-paretic side')], axis=0
        )
    df_data.to_csv(data_save_path + title_label + '_along_cycle.csv',
                   sep=",", header=True)
    plt.close(fig)


def match_both_side_cycle(collection_paretic, collection_non_paretic,
                          df_gait_paretic, df_gait_non_paretic):
    time_ic_paretic = \
        DataProcess.get_initial_contact_time(df_gait_paretic)
    time_ic_non_paretic = \
        DataProcess.get_initial_contact_time(df_gait_non_paretic)
    idx_paretic_matched = []
    idx_non_paretic_matched = []

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

        idx_paretic_matched.append(idx_p)
        idx_non_paretic_matched.append(idx_np)

        idx_p += 1

    return idx_paretic_matched, idx_non_paretic_matched


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
            ind_start = \
                DataProcess.search_index(x[ind_end:], target) + ind_end
            target = x_end[i]
            ind_end = \
                DataProcess.search_index(x[ind_start:], target) + ind_start

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
            try:
                y_cropped = DataProcess.normalize_time_series(data[:, 0],
                                                              data[:, 1],
                                                              x_num=x_num)
            except IndexError:
                continue
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
        time_diff_crop = time_diff[
            np.where(np.logical_and(0.1 < time_diff, time_diff < 1.5))
        ]
        mean_ic_diff = np.mean(time_diff_crop) / mean_cycle
        std_ic_diff = np.std(time_diff_crop) / mean_cycle

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
        mean_stance_percent_nonparetic = \
            mean_stance_time_nonparetic / mean_cycle
        std_stance_percent_nonparetic = \
            std_stance_time_nonparetic / mean_cycle

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
    def get_gait_phase_collection_data(
            paretic_data,  # Dataframe(time, data)
            non_paretic_data,  # Dataframe(time, data)
            paretic_gait_path,
            non_paretic_gait_path,
            ignore_cycle=(None, None),
            start_time=0.0,
            report_start_time=None,
            report_duration=None,
            idx_gait_event_filter=None
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

        if idx_gait_event_filter is not None:
            df_paretic_gait.drop(idx_gait_event_filter[0], inplace=True)
            # df_paretic_gait.reset_index(drop=True, inplace=True)
            df_non_paretic_gait.drop(idx_gait_event_filter[1], inplace=True)
            # df_non_paretic_gait.reset_index(drop=True, inplace=True)

        if report_start_time is not None:
            df_paretic_gait = df_paretic_gait[
                (df_paretic_gait.time >= report_start_time) &
                (df_paretic_gait.time <= report_start_time + report_duration)]
            df_non_paretic_gait = df_non_paretic_gait[
                (df_non_paretic_gait.time >= report_start_time) &
                (df_non_paretic_gait.time <=
                 report_start_time + report_duration)]

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
        return collection_paretic, collection_non_paretic,\
            df_paretic_gait, df_non_paretic_gait

    @staticmethod
    def graph_each_cycle_data(
            paretic_data,  # columns: time, data
            non_paretic_data,  # columns: time, data
            paretic_extra_data,  # columns: time, data 1 ~ data 6
            non_paretic_extra_data,  # columns: time, data 1 ~ data 6
            paretic_gait_path,
            non_paretic_gait_path,
            save_path,  # {test_label}/{session_type}
            data_label=("data", None),
            unit_label=("[N]", None),
            title_label=("data", None),
            ignore_cycle=(None, None),
            start_time=0.0
    ):
        total_collection_paretic = []
        total_collection_non_paretic = []

        # main collection data
        main_collection_paretic, main_collection_non_paretic,\
            df_paretic_gait, df_non_paretic_gait =\
            DataProcess.get_gait_phase_collection_data(
                paretic_data, non_paretic_data,
                paretic_gait_path, non_paretic_gait_path,
                ignore_cycle=ignore_cycle,
                start_time=start_time
                )
        total_collection_paretic.append(main_collection_paretic)
        total_collection_non_paretic.append(main_collection_non_paretic)
        # extra collection data
        for data_ind in np.arange(1, len(paretic_extra_data.columns)):
            paretic_extra_df = pd.DataFrame()
            paretic_extra_df["time"] = paretic_extra_data.iloc[:, 0]
            paretic_extra_df["data"] =\
                paretic_extra_data.iloc[:, data_ind]

            non_paretic_extra_df = pd.DataFrame()
            non_paretic_extra_df["time"] = non_paretic_extra_data.iloc[:, 0]
            non_paretic_extra_df["data"] =\
                non_paretic_extra_data.iloc[:, data_ind]

            extra_collection_paretic, extra_collection_non_paretic, _, _ =\
                DataProcess.get_gait_phase_collection_data(
                    paretic_extra_df, non_paretic_extra_df,
                    paretic_gait_path, non_paretic_gait_path,
                    ignore_cycle=ignore_cycle,
                    start_time=start_time
                    )
            total_collection_paretic.append(extra_collection_paretic)
            total_collection_non_paretic.append(extra_collection_non_paretic)

        save_each_cycle_timeseries_data(
            total_collection_paretic,
            df_paretic_gait,
            data_label=data_label,
            unit_label=unit_label,
            title_label=title_label,
            color='red',
            save_path=save_path + '/graph/each_cycle/paretic/')

        save_each_cycle_timeseries_data(
            total_collection_non_paretic,
            df_non_paretic_gait,
            data_label=data_label,
            unit_label=unit_label,
            title_label=title_label,
            color='blue',
            save_path=save_path + '/graph/each_cycle/non_paretic/')

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
                              idx_paretic_ignore, idx_non_paretic_ignore,
                              data_label, title_graph, save_path,
                              x_num=101):
        collection_data_paretic_sel = \
            copy.deepcopy(collection_data_paretic)
        collection_data_nonparetic_sel = \
            copy.deepcopy(collection_data_nonparetic)

        for idx in sorted(idx_paretic_ignore, reverse=True):
            collection_data_paretic_sel.pop(idx)
        for idx in sorted(idx_non_paretic_ignore, reverse=True):
            collection_data_nonparetic_sel.pop(idx)

        [mean_diff_both, std_diff_both,
         mean_diff_paretic, std_diff_paretic,
         mean_diff_nonparetic,
         std_diff_nonparetic,
         _, _, _, _] = DataProcess.gait_phase_pre_processing(
            data_gait_paretic, data_gait_nonparetic)
        mean_paretic, std_paretic = DataProcess.average_cropped_time_series(
            collection_data_paretic_sel, x_num
        )

        mean_nonparetic, std_nonparetic = \
            DataProcess.average_cropped_time_series(
                collection_data_nonparetic_sel, x_num
            )

        x_paretic = np.linspace(0, 100, x_num)
        x_nonparetic = np.linspace(mean_diff_both, 100 + mean_diff_both,
                                   x_num)

        fig, axs = plt.subplots(2, 1,
                                gridspec_kw={'height_ratios': [5, 1]},
                                figsize=(16, 12))
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

        axs[0].set_ylabel(data_label, fontsize=30)
        axs[0].set_title(title_graph, fontsize=45)
        axs[0].xaxis.set_tick_params(labelsize=25)
        axs[0].yaxis.set_tick_params(labelsize=25)
        axs[0].grid(True)
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
        fig.tight_layout()
        fig.savefig(save_path + '%s_mean_cycle.png' % title_graph)
        plt.close(fig)

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
            stance_flag=False,
            clearance_flag=False,
            report_start_time=None,
            report_duration=None,
            idx_gait_event_filter=None,
            df_sub_paretic=None,
            df_sub_non_paretic=None
    ):

        collection_paretic, collection_non_paretic, \
        df_paretic_gait, df_non_paretic_gait = \
            DataProcess.get_gait_phase_collection_data(
                paretic_data, non_paretic_data,
                paretic_gait_path, non_paretic_gait_path,
                ignore_cycle=ignore_cycle,
                start_time=start_time,
                report_start_time=report_start_time,
                report_duration=report_duration,
                idx_gait_event_filter=idx_gait_event_filter
            )

        df_paretic_gait = \
            get_ignored_cycle(df_paretic_gait, ignore_cycle, True)
        df_non_paretic_gait = \
            get_ignored_cycle(df_non_paretic_gait, ignore_cycle, True)
        idx_paretic_matched, idx_non_paretic_matched = \
            match_both_side_cycle(
                collection_paretic, collection_non_paretic,
                df_paretic_gait, df_non_paretic_gait
            )

        ####################################################
        ## Clearance histogram drawing
        ## Flag should be True only for clearance
        if clearance_flag:
            paretic_swing = []
            paretic_swing_sub = []
            non_paretic_swing = []
            non_paretic_swing_sub = []
            # 1. Get data only for swing
            for i in range(len(df_paretic_gait) - 1):
                if df_paretic_gait.iloc[i, 1] == 1:
                    continue
                else:
                    tmp = paretic_data["value"][
                        (paretic_data["time"] >= df_paretic_gait.iloc[i, 0]) &
                        (paretic_data["time"] < df_paretic_gait.iloc[i + 1, 0])
                        ]
                    tmp = tmp.to_list()
                    paretic_swing += tmp
                    tmp = df_sub_paretic["value"][
                        (df_sub_paretic["time"] >= df_paretic_gait.iloc[i, 0]) &
                        (df_sub_paretic["time"] < df_paretic_gait.iloc[i + 1, 0])
                        ]
                    tmp = tmp.to_list()
                    paretic_swing_sub += tmp

            for i in range(len(df_non_paretic_gait) - 1):
                if df_non_paretic_gait.iloc[i, 1] == 1:
                    continue
                else:
                    tmp = non_paretic_data["value"][
                        (non_paretic_data["time"] >= df_non_paretic_gait.iloc[i, 0]) &
                        (non_paretic_data["time"] < df_non_paretic_gait.iloc[i + 1, 0])
                        ]
                    tmp = tmp.to_list()
                    non_paretic_swing += tmp
                    tmp = df_sub_non_paretic["value"][
                        (df_sub_non_paretic["time"] >= df_non_paretic_gait.iloc[i, 0]) &
                        (df_sub_non_paretic["time"] < df_non_paretic_gait.iloc[i + 1, 0])
                        ]
                    tmp = tmp.to_list()
                    non_paretic_swing_sub += tmp


            # 2. Draw Histogram
            DataProcess.draw_histogram(
                paretic_swing,
                non_paretic_swing,
                paretic_swing_sub,
                non_paretic_swing_sub,
                save_path,
                title_label
            )
        ####################################################


        ####################################################
        ## Disabled Picker
        idx_paretic_ignore, idx_non_paretic_ignore, \
            stance_time_paretic, stance_time_non_paretic = \
            get_index_outlier(
                df_paretic_gait, df_non_paretic_gait
            )
        idx_paretic_ignore = []
        idx_non_paretic_ignore = []
        #####################################################

        DataProcess.graph_both_cycle_data(
            collection_paretic, collection_non_paretic,
            df_paretic_gait, df_non_paretic_gait,
            idx_paretic_ignore, idx_non_paretic_ignore,
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

            for idx_paretic, idx_non_paretic in \
                    zip(idx_paretic_matched, idx_non_paretic_matched):
                if idx_paretic in idx_paretic_ignore:
                    continue
                if idx_non_paretic in idx_non_paretic_ignore:
                    continue
                da = collection_paretic[idx_paretic]

                try:
                    max_paretic.append(
                        np.max(da[:, 1])
                    )
                except ValueError:
                    print("max paretic - Empty collection - index "
                          + str(idx_paretic))
                da = collection_non_paretic[idx_non_paretic]
                try:
                    max_non_paretic.append(
                        np.max(da[:, 1])
                    )
                except ValueError:
                    print("max non paretic - Empty collection - index "
                          + str(idx_non_paretic))

            np_p_max = np.array(max_paretic)
            np_np_max = np.array(max_non_paretic)
            max_paretic_mean = np.mean(np_p_max)
            max_paretic_stdev = np.std(np_p_max)
            max_non_paretic_mean = np.mean(np_np_max)
            max_non_paretic_stdev = np.std(np_np_max)
            max_symmetry = 1 - \
                           abs(max_paretic_mean - max_non_paretic_mean) \
                           / (max_paretic_mean + max_non_paretic_mean)
            save_each_cycle_bar_plot(
                np_p_max, np_np_max, max_paretic_mean, max_non_paretic_mean,
                data_label + '[N]', title_label + "_max",
                save_path
            )
        impulse_paretic_mean = 0
        impulse_paretic_stdev = 0
        impulse_non_paretic_mean = 0
        impulse_non_paretic_stdev = 0
        impulse_symmetry = 0

        if impulse_flag:
            impulse_paretic = []
            impulse_non_paretic = []

            for idx_paretic, idx_non_paretic in \
                    zip(idx_paretic_matched, idx_non_paretic_matched):
                if idx_paretic in idx_paretic_ignore:
                    continue
                if idx_non_paretic in idx_non_paretic_ignore:
                    continue
                da = collection_paretic[idx_paretic]
                impulse_paretic.append(
                    np.trapz(da[:, 1], x=da[:, 0])
                )
                da = collection_non_paretic[idx_non_paretic]
                impulse_non_paretic.append(
                    np.trapz(da[:, 1], x=da[:, 0])
                )

            np_p_impulse = np.array(impulse_paretic)
            np_np_impulse = np.array(impulse_non_paretic)
            impulse_paretic_mean = np.mean(np_p_impulse)
            impulse_paretic_stdev = np.std(np_p_impulse)
            impulse_non_paretic_mean = np.mean(np_np_impulse)
            impulse_non_paretic_stdev = np.std(np_np_impulse)
            impulse_symmetry = 1 - \
                               abs(impulse_paretic_mean - impulse_non_paretic_mean) \
                               / (impulse_paretic_mean + impulse_non_paretic_mean)
            save_each_cycle_bar_plot(
                np_p_impulse, np_np_impulse,
                impulse_paretic_mean, impulse_non_paretic_mean,
                data_label + '[N sec]', title_label + "_impulse",
                save_path
            )

        stance_paretic_mean = 0
        stance_paretic_stdev = 0
        stance_non_paretic_mean = 0
        stance_non_paretic_stdev = 0
        stance_symmetry = 0

        if stance_flag:
            stance_time_paretic_ignored = []
            stance_time_non_paretic_ignored = []
            for idx_paretic, idx_non_paretic in \
                    zip(idx_paretic_matched, idx_non_paretic_matched):
                if idx_paretic in idx_paretic_ignore:
                    continue
                if idx_non_paretic in idx_non_paretic_ignore:
                    continue
                # try:
                #     stance_time_paretic_ignored.append(
                #         stance_time_paretic[idx_paretic]
                #     )
                #     stance_time_non_paretic_ignored.append(
                #         stance_time_non_paretic[idx_non_paretic])
                # except IndexError:
                #     print("Warning : At stance cropping step, list index out of range")
            stance_paretic_mean = np.mean(stance_time_paretic)
            stance_paretic_stdev = np.std(stance_time_paretic)
            stance_non_paretic_mean = np.mean(stance_time_non_paretic)
            stance_non_paretic_stdev = np.std(stance_time_non_paretic)
            save_each_cycle_bar_plot(
                stance_time_paretic, stance_time_non_paretic,
                stance_paretic_mean, stance_non_paretic_mean,
                'stance time [s]', "Stance_Time",
                save_path
            )
            # save_each_cycle_bar_plot(
            #     stance_time_paretic_ignored, stance_time_non_paretic_ignored,
            #     stance_paretic_mean, stance_non_paretic_mean,
            #     'stance time [s]', "Stance_Time_ignored"
            #                        "",
            #     save_path
            # )
            stance_symmetry = 1 - \
                              abs(stance_paretic_mean - stance_non_paretic_mean) \
                              / (stance_paretic_mean + stance_non_paretic_mean)

        return [max_paretic_mean, max_paretic_stdev,
                max_non_paretic_mean, max_non_paretic_stdev,
                max_symmetry
                ], \
               [impulse_paretic_mean, impulse_paretic_stdev,
                impulse_non_paretic_mean, impulse_non_paretic_stdev,
                impulse_symmetry
                ], \
               [stance_paretic_mean, stance_paretic_stdev,
                stance_non_paretic_mean, stance_non_paretic_stdev,
                stance_symmetry
                ]

    @staticmethod
    def draw_histogram(paretic_swing,
                       non_paretic_swing,
                       paretic_swing_sub,
                       non_paretic_swing_sub,
                       report_save_path,
                       plot_title):
        array_paretic = np.array(paretic_swing)
        array_non_paretic = np.array(non_paretic_swing)
        array_paretic_sub = np.array(paretic_swing_sub)
        array_non_paretic_sub = np.array(non_paretic_swing_sub)

        weights_paretic = \
            np.ones_like(array_paretic) / len(array_paretic)
        weights_non_paretic = \
            np.ones_like(array_non_paretic) / len(array_non_paretic)
        weights_paretic_sub = np.ones_like(array_paretic_sub) / len(array_paretic_sub)
        weights_non_paretic_sub = np.ones_like(array_non_paretic_sub) / len(array_non_paretic_sub)

        fig, axs = plt.subplots(1, 2, sharey='all', tight_layout=True)
        axs[0].hist(array_paretic, weights=weights_paretic, bins=15, color='red', alpha=0.2)
        axs[0].hist(array_non_paretic, weights=weights_non_paretic, bins=15, color='blue', alpha=0.2)
        axs[1].hist(array_paretic_sub, weights=weights_paretic_sub, bins=15, color='red', alpha=0.2)
        axs[1].hist(array_non_paretic_sub, weights=weights_non_paretic_sub, bins=15, color='blue', alpha=0.2)

        axs[0].set_xlim(left=-100, right=400)
        axs[0].set_title("Clearance - Toe")
        axs[0].set_xlabel("Clearance [mm]")
        axs[0].set_ylabel("Frequency")
        axs[1].set_xlim(left=-100, right=400)
        axs[1].set_title("Clearance - Heel")
        axs[1].set_xlabel("Clearance [mm]")
        create_folder(report_save_path+"/graph")
        fig.savefig(report_save_path + "/graph/" + "clearance_toe_heel.png")

        fig, axs = plt.subplots(1, 2, sharey='all', tight_layout=True)
        axs[0].hist(array_paretic, weights=weights_paretic, bins=15,
                    color='red', alpha=0.2)
        axs[0].hist(array_paretic_sub, weights=weights_paretic_sub, bins=15,
                    color='red', histtype='step')
        axs[1].hist(array_non_paretic, weights=weights_non_paretic, bins=15,
                    color='blue', alpha=0.2)
        axs[1].hist(array_non_paretic_sub, weights=weights_non_paretic_sub,
                    bins=15, color='blue', histtype='step')

        axs[0].set_xlim(left=-100, right=400)
        axs[0].set_title("Clearance - Paretic")
        axs[0].set_xlabel("Clearance [mm]")
        axs[0].set_ylabel("Frequency")
        axs[1].set_xlim(left=-100, right=400)
        axs[1].set_title("Clearance - Non-paretic")
        axs[1].set_xlabel("Clearance [mm]")
        fig.savefig(report_save_path + "/graph/" + "clearance_pnp.png")


class Picker:
    def __init__(self, data):
        self.del_index = []
        self.selected_idx = []
        self.data = data
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.2)
        self.ax.plot(data, 'bo', picker=True, pickradius=5)
        self.ax.set_title("c", picker=True)
        self.ax.set_ylabel("y", picker=True)
        self.fig.canvas.mpl_connect('pick_event', self.pick)
        self.ax_update = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        self.ax_redraw = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        self.ax_reset = self.fig.add_axes([0.59, 0.05, 0.1, 0.075])
        self.b_update = Button(self.ax_update, 'Update')
        self.b_redraw = Button(self.ax_redraw, 'Redraw')
        self.b_reset = Button(self.ax_reset, 'Reset')
        self.b_update.on_clicked(self.update)
        self.b_redraw.on_clicked(self.redraw)
        self.b_reset.on_clicked(self.reset)
        plt.show()

    def pick(self, event):
        if isinstance(event.artist, Line2D):
            ind = event.ind[0]
            if ind not in self.del_index:
                self.del_index.append(ind)
                self.ax.plot(ind, self.data[ind], 'r*')
                plt.draw()

    def update(self, event):
        self.selected_idx = sorted(self.del_index)
        plt.close()

    def redraw(self, event):
        data_del_ = copy.deepcopy(self.data)
        for idx in self.del_index:
            data_del_[idx] = None
        self.ax.clear()
        self.ax.plot(data_del_, 'bo', picker=True, pickradius=5)
        plt.draw()

    def reset(self, event):
        self.del_index = []
        self.ax.clear()
        self.ax.plot(self.data, 'bo', picker=True, pickradius=5)


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
