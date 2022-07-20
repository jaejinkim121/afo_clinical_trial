import enum


class PlotFlag(enum.Flag):
    # For LorR Flag -> True when Left-only, False when Right-only
    USE_VOLT = False
    VOLT_SEP = False
    USE_VOLT_LorR = True

    USE_FORCE = True
    FORCE_SEP = False
    USE_FORCE_LorR = True

    USE_DIDIM_GRF = True
    USE_DIDIM_KINEMATICS = True
    USE_DIDIM_KINEMATICS_ALL = False

    USE_IMU_ACCEL = True
    USE_IMU_GYRO = True