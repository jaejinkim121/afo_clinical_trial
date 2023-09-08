import numpy as np
import copy


class DataProcess:
    @staticmethod
    def normalize_time_series(
            x: np.array, y: np.array, x_target=None, x_num=100):
        if x_target is None:
            x_target = np.linspace(0, 1, x_num)

        x_target = x_target / x_target[-1]
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
            ind_start = DataProcess.search_index(x[ind_end:], target)
            target = x_end[i]
            ind_end = DataProcess.search_index(x[ind_start:], target)

            y_cropped = DataProcess.normalize_time_series(
                x[ind_start:ind_end+1],
                y[ind_start:ind_end+1],
                x_num=x_num)
            crop_time_series.append(y_cropped)

        crop_time_series = np.array(crop_time_series)
        mean_time_series = np.mean(crop_time_series, axis=0)
        std_time_series = np.mean(crop_time_series, axis=0)

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


