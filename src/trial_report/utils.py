import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd


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
    def get_initial_contact_time(gait_phase: pd.DataFrame, data_name="value"):
        if 'time' not in gait_phase.columns:
            print("Time data missing")
            return -1
        if data_name not in gait_phase.columns:
            print("Value data missing")
            return -1

        time_initial_contact = gait_phase[gait_phase[data_name] == 1]["time"]

        return time_initial_contact.tolist()

    @staticmethod
    def get_foot_off_time(gait_phase: pd.DataFrame, data_name="value"):
        if 'time' not in gait_phase.columns:
            print("Time data missing")
            return -1
        if data_name not in gait_phase.columns:
            print("Value data missing")
            return -1

        time_foot_off = gait_phase[gait_phase[data_name] == 2]["time"]

        return time_foot_off.tolist()

    @staticmethod
    def get_gait_event_time(gait_phase: pd.DataFrame, data_name="value"):
        time_initial_contact = \
            DataProcess.get_initial_contact_time(gait_phase, data_name)
        time_foot_off = \
            DataProcess.get_foot_off_time(gait_phase, data_name)

        return time_initial_contact, time_foot_off

    @staticmethod
    def gait_phase_pre_processing(
            gait_phase_paretic: pd.DataFrame,
            gait_phase_nonparetic: pd.DataFrame, data_name="value"):
        paretic_ic, paretic_fo = \
            DataProcess.get_gait_event_time(gait_phase_paretic, data_name)
        nonparetic_ic, nonparetic_fo = \
            DataProcess.get_gait_event_time(gait_phase_nonparetic, data_name)

        mean_cycle = (paretic_ic[-1] - paretic_ic[0]) / (len(paretic_ic) - 1)
        mean_cycle = mean_cycle / 100.0
        idx_np = 0
        time_diff = []
        for tic in paretic_ic:
            idx_np += DataProcess.search_index(nonparetic_ic[idx_np:], tic)
            time_diff.append(nonparetic_ic[idx_np] - tic)

        time_diff = np.array(time_diff)
        mean_diff = np.mean(time_diff) / mean_cycle
        std_diff = np.std(time_diff) / mean_cycle

        if paretic_ic[0] > paretic_fo[0]:
            paretic_fo = paretic_fo[1:]
        if paretic_ic[-1] > paretic_fo[-1]:
            paretic_ic = paretic_ic[:-1]

        time_diff = []
        for tic, tfo in zip(paretic_ic, paretic_fo):
            time_diff.append(tfo - tic)
        time_diff = np.array(time_diff)
        mean_cycle_time_paretic = np.mean(time_diff) / mean_cycle
        std_cycle_time_paretic = np.std(time_diff) / mean_cycle

        if nonparetic_ic[0] > nonparetic_fo[0]:
            nonparetic_fo = nonparetic_fo[1:]
        if nonparetic_ic[-1] > nonparetic_fo[-1]:
            nonparetic_ic = nonparetic_ic[:-1]

        time_diff = []
        for tic, tfo in zip(nonparetic_ic, nonparetic_fo):
            time_diff.append(tfo - tic)
        time_diff = np.array(time_diff)
        mean_cycle_time_nonparetic = np.mean(time_diff) / mean_cycle
        std_cycle_time_nonparetic = np.std(time_diff) / mean_cycle

        return [mean_diff, std_diff,
                mean_cycle_time_paretic, std_cycle_time_paretic,
                mean_cycle_time_nonparetic, std_cycle_time_nonparetic]

    @staticmethod
    def divider_data_by_gait_phase_path(
            data_path, gait_phase_path, data_name="value"):
        data = pd.DataFrame()
        gait_phase = pd.DataFrame()

        value_data = []
        time_data = []
        value_gait_phase = []
        time_gait_phase = []

        with open(data_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for csv_row in csv_reader:
                if csv_row[0] == "time":
                    continue
                value_data.append(float(csv_row[1]))
                time_data.append(float(csv_row[0]))

        with open(gait_phase_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for csv_row in csv_reader:
                if csv_row[0] == "time":
                    continue
                value_gait_phase.append(float(csv_row[1]))
                time_gait_phase.append(float(csv_row[0]))

        data["time"] = time_data
        data[data_name] = value_data
        gait_phase["time"] = time_gait_phase
        gait_phase[data_name] = value_gait_phase

        divided_array = []
        time_initial_contact = \
            DataProcess.get_initial_contact_time(gait_phase, data_name)
        time_initial_contact.append(time_data[-1])

        for i in range(len(time_initial_contact) - 1):
            divided_df_current = \
                data[
                    (data["time"] >= time_initial_contact[i]) &
                    (data["time"] < time_initial_contact[i + 1])
                    ]
            divided_array.append(divided_df_current.to_numpy())

        return divided_array, gait_phase

    @staticmethod
    def divider_data_by_gait_phase_df(data_df, gait_phase_df,
                                      data_name="value"):
        divided_array = []
        time_initial_contact = \
            DataProcess.get_initial_contact_time(
                gait_phase_df, data_name)
        time_initial_contact.append(data_df["time"].iloc[-1])

        for i in range(len(time_initial_contact) - 1):
            divided_df_current = \
                data_df[
                    (data_df["time"] >= time_initial_contact[i]) &
                    (data_df["time"] < time_initial_contact[i + 1])
                    ]
            divided_array.append(divided_df_current.to_numpy())

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
                              data_gait_label="value",
                              title_graph=None, data_label=None, x_num=101):
        [mean_diff_both, std_diff_both,
         mean_diff_paretic, std_diff_paretic,
         mean_diff_nonparetic,
         std_diff_nonparetic] = DataProcess.gait_phase_pre_processing(
            data_gait_paretic, data_gait_nonparetic, data_gait_label)

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

        plt.show()
