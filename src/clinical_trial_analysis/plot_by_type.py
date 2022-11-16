from include.config import PlotFlag, DataSet


def add_data_GRF(data, df_didim_GRF):
    grf_data = list()
    grf_data.append(DataSet(
        df_didim_GRF['time'],
        df_didim_GRF['L_GRF_VRT'],
        'Left GRF VRT'))
    grf_data.append(DataSet(
        df_didim_GRF['time'],
        df_didim_GRF['R_GRF_VRT'],
        'Right GRF VRT'
    ))
    data.append(grf_data)

    if PlotFlag.USE_DIDIM_GRF_LAT:
        grf_data = list()
        grf_data.append(DataSet(
            df_didim_GRF['time'],
            df_didim_GRF['L_GRF_LAT'],
            'Left GRF LAT'))
        grf_data.append(DataSet(
            df_didim_GRF['time'],
            df_didim_GRF['R_GRF_LAT'],
            'Right GRF LAT'
        ))
        data.append(grf_data)

    if PlotFlag.USE_DIDIM_GRF_FWD:
        grf_data = list()
        grf_data.append(DataSet(
            df_didim_GRF['time'],
            df_didim_GRF['L_GRF_FWD'],
            'Left GRF FWD'))
        grf_data.append(DataSet(
            df_didim_GRF['time'],
            df_didim_GRF['R_GRF_FWD'],
            'Right GRF FWD'
        ))
        data.append(grf_data)


def add_data_sole(data, df_L, df_R, selected_sensor=None):
    sole_data_left = list()
    sole_data_right = list()
    for key in df_L:
        if selected_sensor is not None:
            if key.split()[-1] != selected_sensor:
                continue
        if key == 'L lateral heel' or key == 'L medial heel':
            continue
        if key == 'time':
            continue
        if key.split()[-1] == 'head':
            line_color = 'r'
            if key.split()[1] == 'first':
                line_type = 'solid'
            elif key.split()[1] == 'third':
                line_type = 'dash'
            else:
                line_type = 'dot'
        elif key.split()[-1] == 'toe':
            line_color = 'b'
            if key.split()[1] == 'first':
                line_type = 'solid'
            else:
                line_type = 'dot'
        elif key.split()[-1] == 'heel':
            line_color = 'g'
            if key.split()[1] == 'medial':
                line_type = 'solid'
            elif key.split()[1] == 'lateral':
                line_type = 'dash'
            else:
                line_type = 'dot'
        sole_data_left.append(DataSet(
            df_L['time'], df_L[key], key,
            line_type=line_type, line_color=line_color
        ))

    for key in df_R:
        if selected_sensor is not None:
            if key.split()[-1] != selected_sensor:
                continue
        if key == 'R lateral heel' or key == 'R medial heel':
            continue
        if key == 'time':
            continue
        if key.split()[-1] == 'head':
            line_color = 'r'
            if key.split()[1] == 'first':
                line_type = 'solid'
            elif key.split()[1] == 'third':
                line_type = 'dash'
            else:
                line_type = 'dot'
        elif key.split()[-1] == 'toe':
            line_color = 'b'
            if key.split()[1] == 'first':
                line_type = 'solid'
            else:
                line_type = 'dot'
        elif key.split()[-1] == 'heel':
            line_color = 'g'
            if key.split()[1] == 'medial':
                line_type = 'solid'
            elif key.split()[1] == 'lateral':
                line_type = 'dash'
            else:
                line_type = 'dot'
        sole_data_right.append(DataSet(
            df_R['time'], df_R[key], key,
            line_type=line_type, line_color=line_color
        ))

    data.append(sole_data_left)
    data.append(sole_data_right)


def add_data_kinematics(data, df_didim_kinematics):
    joint_data_left = list()
    joint_data_right = list()
    for key in df_didim_kinematics:
        if key == 'time':
            continue
        elif not PlotFlag.USE_DIDIM_KINEMATICS_ALL:
            if ('Abd' in key) or ('Rot' in key):
                continue
        if key[0] == 'L':
            joint_data_left.append(DataSet(
                df_didim_kinematics['time'], df_didim_kinematics[key], key
            ))
        elif key[0] == 'R':
            joint_data_right.append(DataSet(
                df_didim_kinematics['time'], df_didim_kinematics[key], key
            ))
    data.append(joint_data_left)
    data.append(joint_data_right)


def add_data_accel(data, df_imu):
    data_acc = list()
    for imu in df_imu:
        key = imu.keys()[-2]
        if key.split()[0] == 'Left':
            line_color = 'r'
        elif key.split()[0] == 'Right':
            line_color = 'b'
        else:
            line_color = 'g'
        if key.split()[-1].split('_')[0] == 'Thigh':
            line_type = 'dash'
        elif key.split()[-1].split('_')[0] == 'Shank':
            line_type = 'dot'
        else:
            line_type = 'solid'

        data_acc.append(DataSet(
            imu['time'], imu[key], key,
            line_type=line_type, line_color=line_color
        ))
    data.append(data_acc)


def add_data_gyro(data, df_imu):
    data_gyro = list()
    for imu in df_imu:
        key = imu.keys()[-1]
        if key.split()[0] == 'Left':
            line_color = 'r'
        elif key.split()[0] == 'Right':
            line_color = 'b'
        else:
            line_color = 'g'
        if key.split()[-1].split('_')[0] == 'Thigh':
            line_type = 'dash'
        elif key.split()[-1].split('_')[0] == 'Shank':
            line_type = 'dot'
        else:
            line_type = 'solid'
        data_gyro.append(DataSet(
            imu['time'], imu[key], key,
            line_type=line_type, line_color=line_color
        ))
    data.append(data_gyro)
