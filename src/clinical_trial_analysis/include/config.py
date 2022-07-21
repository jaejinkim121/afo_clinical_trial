import enum


class PlotFlag(enum.Flag):
    # For LorR Flag -> True when Left-only, False when Right-only
    USE_VOLT = False
    VOLT_SEP = False
    USE_VOLT_LorR = True

    USE_FORCE = False
    FORCE_SEP = False
    USE_FORCE_LorR = True

    USE_DIDIM_GRF = True
    USE_DIDIM_GRF_LAT = True
    USE_DIDIM_GRF_FWD = True

    USE_DIDIM_KINEMATICS = False
    USE_DIDIM_KINEMATICS_ALL = False

    USE_IMU_ACCEL = False
    USE_IMU_GYRO = False


class DataSet:
    def __init__(self, time, data, title,
                 line_type=None,
                 line_color=''):
        self.time = time
        self.data = data
        self.title = title
        self.line_type = line_type
        self.line_color = line_color
